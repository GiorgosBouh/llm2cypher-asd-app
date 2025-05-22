import numpy as np
import networkx as nx
from node2vec import Node2Vec
from neo4j import GraphDatabase
import sys
import os
import time
import tempfile
import logging
from typing import List, Optional
import shutil
import json

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class EmbeddingGenerator:
    def __init__(self):
        self.EMBEDDING_DIM = 128
        self.MIN_CONNECTIONS = 3
        self.MAX_RETRIES = 3
        self.MIN_SIMILARITY = 5  # Minimum shared answers for similarity
        self.EMBEDDING_NORMALIZATION = True
        
        self.NODE2VEC_WALK_LENGTH = 30
        self.NODE2VEC_NUM_WALKS = 100
        self.NODE2VEC_P = 1.0
        self.NODE2VEC_Q = 0.5
        self.NODE2VEC_WORKERS = 4
        self.NODE2VEC_WINDOW = 5
        self.NODE2VEC_BATCH_WORDS = 1000

    def get_driver(self):
        return GraphDatabase.driver(
            os.getenv("NEO4J_URI"),
            auth=(os.getenv("NEO4J_USER"), os.getenv("NEO4J_PASSWORD")),
            connection_timeout=30,
            max_connection_lifetime=7200,
            max_connection_pool_size=50
        )

    def validate_embedding(self, embedding: List[float]) -> bool:
        if not embedding or len(embedding) != self.EMBEDDING_DIM:
            return False
            
        if any(np.isnan(x) for x in embedding):
            logger.warning("Embedding contains NaN values")
            return False
            
        if np.all(np.array(embedding) == 0):
            logger.warning("Embedding is all zeros")
            return False
            
        if self.EMBEDDING_NORMALIZATION:
            norm = np.linalg.norm(embedding)
            if norm == 0:
                return False
            embedding = [x/norm for x in embedding]
            
        return True

    def insert_temporary_case(self, driver, upload_id: str, case_data: dict) -> bool:
        """Insert temporary Case node and related nodes/relationships"""
        logger.info(f"Inserting temporary case with upload_id {upload_id}")
        with driver.session() as session:
            # Δημιουργία κόμβου Case με upload_id
            # και δημιουργία σχέσεων με BehaviorQuestion, DemographicAttribute, SubmitterType
            # Τα δεδομένα του case_data περιλαμβάνουν απαντήσεις A1-A10, Demographics κλπ

            try:
                # Δημιουργία του κόμβου Case
                session.run("""
                    CREATE (c:Case {upload_id: $upload_id, id: $case_no})
                """, upload_id=upload_id, case_no=int(case_data["Case_No"]))

                # Συνδέουμε με BehaviorQuestion nodes και τις απαντήσεις (HAS_ANSWER)
                for i in range(1, 11):
                    question_label = f"A{i}"
                    answer_value = int(case_data.get(question_label, 0))
                    session.run("""
                        MATCH (c:Case {upload_id: $upload_id}), (q:BehaviorQuestion {name: $question})
                        MERGE (c)-[r:HAS_ANSWER]->(q)
                        SET r.value = $value
                    """, upload_id=upload_id, question=question_label, value=answer_value)

                # Συνδέουμε με DemographicAttribute nodes (HAS_DEMOGRAPHIC)
                demographic_fields = ["Sex", "Ethnicity", "Jaundice", "Family_mem_with_ASD"]
                for field in demographic_fields:
                    val = str(case_data.get(field, "")).strip()
                    if val:
                        session.run("""
                            MATCH (c:Case {upload_id: $upload_id}), (d:DemographicAttribute {type: $field, value: $val})
                            MERGE (c)-[:HAS_DEMOGRAPHIC]->(d)
                        """, upload_id=upload_id, field=field, val=val)

                # Συνδέουμε με SubmitterType node (SUBMITTED_BY)
                submitter = str(case_data.get("Who_completed_the_test", "")).strip()
                if submitter:
                    session.run("""
                        MATCH (c:Case {upload_id: $upload_id}), (s:SubmitterType {type: $submitter})
                        MERGE (c)-[:SUBMITTED_BY]->(s)
                    """, upload_id=upload_id, submitter=submitter)

                logger.info("Temporary case inserted successfully")
                return True

            except Exception as e:
                logger.error(f"Failed to insert temporary case: {str(e)}")
                return False

    def delete_temporary_case(self, driver, upload_id: str) -> None:
        """Delete the temporary case node and all its relationships"""
        logger.info(f"Deleting temporary case with upload_id {upload_id}")
        with driver.session() as session:
            try:
                session.run("""
                    MATCH (c:Case {upload_id: $upload_id})
                    DETACH DELETE c
                """, upload_id=upload_id)
                logger.info("Temporary case deleted successfully")
            except Exception as e:
                logger.error(f"Failed to delete temporary case: {str(e)}")

    def build_base_graph(self, driver, upload_id: str) -> Optional[tuple]:
        """Construct the initial graph structure"""
        with driver.session() as session:
            result = session.run("""
                MATCH (c:Case {upload_id: $upload_id})
                OPTIONAL MATCH (c)-[r:HAS_ANSWER|HAS_DEMOGRAPHIC|SUBMITTED_BY]->(neighbor)
                WHERE neighbor.id IS NOT NULL OR neighbor:BehaviorQuestion OR neighbor:DemographicAttribute
                RETURN c.id AS case_id, 
                       collect(DISTINCT CASE 
                           WHEN neighbor:Case THEN neighbor.id 
                           ELSE id(neighbor) 
                       END) AS neighbors
            """, upload_id=upload_id).single()

            if not result or not result["case_id"]:
                logger.error("Case not found or missing ID")
                return None

            case_id = f"Case_{result['case_id']}"
            G = nx.Graph()
            G.add_node(case_id)
            
            for neighbor in result["neighbors"]:
                if neighbor:
                    neighbor_id = f"Case_{neighbor}" if isinstance(neighbor, int) else str(neighbor)
                    G.add_edge(case_id, neighbor_id)
            
            return G, case_id

    def augment_with_similarity(self, driver, G: nx.Graph, case_id: str) -> None:
        original_case_id = int(case_id.split('_')[1])
        
        with driver.session() as session:
            similar = session.run("""
                MATCH (c:Case {id: $case_id})-[:HAS_ANSWER]->(q)<-[:HAS_ANSWER]-(similar:Case)
                WHERE c <> similar AND similar.embedding IS NOT NULL
                WITH similar, count(q) AS shared_answers
                WHERE shared_answers >= $min_similarity
                RETURN similar.id, shared_answers
                ORDER BY shared_answers DESC 
                LIMIT 20
            """, case_id=original_case_id, min_similarity=self.MIN_SIMILARITY)

            for record in similar:
                similar_id = f"Case_{record['similar.id']}"
                weight = record['shared_answers'] / 10.0
                G.add_edge(case_id, similar_id, weight=weight)

    def generate_embedding(self, G: nx.Graph, case_id: str) -> Optional[List[float]]:
        temp_dir = None
        try:
            temp_dir = tempfile.mkdtemp()
            
            node2vec = Node2Vec(
                G,
                dimensions=self.EMBEDDING_DIM,
                walk_length=self.NODE2VEC_WALK_LENGTH,
                num_walks=self.NODE2VEC_NUM_WALKS,
                workers=self.NODE2VEC_WORKERS,
                p=self.NODE2VEC_P,
                q=self.NODE2VEC_Q,
                quiet=True,
                temp_folder=temp_dir,
                weighted=True,  # Χρήση βαρών από τις σχέσεις
                weight_key='weight'
            )
            
            model = node2vec.fit(
                window=self.NODE2VEC_WINDOW,
                min_count=1,
                batch_words=self.NODE2VEC_BATCH_WORDS
            )
            
            embedding = model.wv[case_id].tolist()
            
            if np.std(embedding) < 0.01:  # Πολύ μικρή διακύμανση
                logger.warning(f"Low variance embedding for {case_id}")
                return None
            
            if not self.validate_embedding(embedding):
                logger.error("Generated invalid embedding")
                return None
                
            return embedding
            
        except Exception as e:
            logger.error(f"Embedding generation failed: {str(e)}")
            return None
        finally:
            if temp_dir and os.path.exists(temp_dir):
                try:
                    shutil.rmtree(temp_dir)
                except Exception as e:
                    logger.warning(f"Could not remove temp directory: {str(e)}")

    def generate_embedding_for_case(self, upload_id: str, case_data: dict) -> Optional[List[float]]:
        """Main embedding generation workflow for temporary case"""
        driver = None
        try:
            driver = self.get_driver()

            # Step 0: Insert temporary case node with relationships
            inserted = self.insert_temporary_case(driver, upload_id, case_data)
            if not inserted:
                logger.error("Failed to insert temporary case. Aborting embedding generation.")
                return None
            
            # Step 1: Build base graph
            graph_result = self.build_base_graph(driver, upload_id)
            if not graph_result:
                logger.error("Failed to build base graph")
                self.delete_temporary_case(driver, upload_id)
                return None
                
            G, case_id = graph_result
            
            # Step 2: Augment with similarity if needed
            if len(G.edges(case_id)) < self.MIN_CONNECTIONS:
                logger.info(f"Initial connections low ({len(G.edges(case_id))}), augmenting...")
                self.augment_with_similarity(driver, G, case_id)
                
                if len(G.edges(case_id)) < self.MIN_CONNECTIONS:
                    logger.warning(f"Insufficient connections ({len(G.edges(case_id))}) after augmentation")
                    self.delete_temporary_case(driver, upload_id)
                    return None

            # Step 3: Generate embedding
            logger.info(f"Generating embedding for {case_id} with {len(G.nodes)} nodes and {len(G.edges)} edges")
            embedding = self.generate_embedding(G, case_id)
            if not embedding:
                logger.error("Embedding generation returned None")
                self.delete_temporary_case(driver, upload_id)
                return None

            # Step 4: Delete temporary node
            self.delete_temporary_case(driver, upload_id)

            logger.info(f"Successfully generated embedding for case {case_id}")
            return embedding

        except Exception as e:
            logger.error(f"Embedding generation failed: {str(e)}")
            if driver:
                self.delete_temporary_case(driver, upload_id)
            return None

        finally:
            if driver:
                driver.close()

if __name__ == "__main__":
    try:
        if len(sys.argv) < 3:
            raise ValueError("Missing parameters. Usage: generate_case_embedding.py <upload_id> <case_data_json>")
        
        upload_id = sys.argv[1]
        case_data_json = sys.argv[2]
        case_data = json.loads(case_data_json)

        logger.info(f"Starting embedding generation for upload_id: {upload_id}")
        generator = EmbeddingGenerator()
        embedding = generator.generate_embedding_for_case(upload_id, case_data)

        if embedding is not None:
            logger.info("Embedding generated successfully")
            # Επιστρέφουμε το embedding ως JSON στο stdout για να το πάρει το main_app.py
            print(json.dumps(embedding))
            sys.exit(0)
        else:
            logger.error("Embedding generation failed")
            sys.exit(1)
        
    except Exception as e:
        logger.critical(f"Fatal error: {str(e)}")
        sys.exit(1)