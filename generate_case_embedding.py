import numpy as np
import networkx as nx
from node2vec import Node2Vec
from neo4j import GraphDatabase
import sys
import os
import time
import tempfile
import logging
from typing import List, Optional, Tuple # Ensure Tuple is imported
import shutil
import json
from dotenv import load_dotenv # ADDED: For consistent env var loading
import pandas as pd # ADDED: Needed for pd.isna, pd.to_numeric

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ADDED: Load environment variables here for standalone script execution
load_dotenv()

class EmbeddingGenerator:
    def __init__(self):
        self.EMBEDDING_DIM = 128
        self.MIN_CONNECTIONS = 5  # Increased minimum connections
        self.MIN_SIMILARITY = 3   # Lowered similarity threshold
        
        # Updated Node2Vec parameters
        self.NODE2VEC_WALK_LENGTH = 30
        self.NODE2VEC_NUM_WALKS = 200  # Increased walks
        self.NODE2VEC_P = 0.5  # More BFS-like
        self.NODE2VEC_Q = 2.0  # More DFS-like
        self.NODE2VEC_WORKERS = 4
        self.NODE2VEC_WINDOW = 10
    def get_driver(self):
        """Initializes Neo4j driver using environment variables."""
        return GraphDatabase.driver(
            os.getenv("NEO4J_URI"),
            auth=(os.getenv("NEO4J_USER"), os.getenv("NEO4J_PASSWORD")),
            connection_timeout=600,
            max_connection_lifetime=7200,
            max_connection_pool_size=50
        )

    def validate_embedding(self, embedding: List[float]) -> bool:
        """Validates a generated embedding list."""
        if not embedding or len(embedding) != self.EMBEDDING_DIM:
            logger.warning(f"Embedding invalid: Length {len(embedding)} != {self.EMBEDDING_DIM} or empty.")
            return False
            
        if any(np.isnan(x) for x in embedding):
            logger.warning("Embedding contains NaN values.")
            return False
            
        if np.all(np.array(embedding) == 0):
            logger.warning("Embedding is all zeros.")
            return False
            
        return True

    def insert_temporary_case(self, driver, upload_id: str, case_data: dict) -> bool:
        """Insert temporary Case node and related nodes/relationships."""
        logger.info(f"Inserting temporary case with upload_id {upload_id}")
        with driver.session() as session:
            try:
                # Create the Case node. Use Case_No from case_data for 'id' property.
                session.run("""
                    CREATE (c:Case {upload_id: $upload_id, id: $case_no})
                """, upload_id=upload_id, case_no=int(case_data.get("Case_No", -1))) # Default -1 if Case_No missing
                
                # Connect to BehaviorQuestion nodes (HAS_ANSWER)
                for i in range(1, 11):
                    question_label = f"A{i}"
                    # Ensure answer_value is an integer, default to 0 if missing or invalid
                    answer_value = pd.to_numeric(case_data.get(question_label, np.nan), errors='coerce')
                    answer_value = int(answer_value) if not pd.isna(answer_value) else 0 

                    session.run("""
                        MATCH (c:Case {upload_id: $upload_id}), (q:BehaviorQuestion {name: $question})
                        MERGE (c)-[r:HAS_ANSWER]->(q)
                        SET r.value = $value
                    """, upload_id=upload_id, question=question_label, value=answer_value)

                # Connect to DemographicAttribute nodes (HAS_DEMOGRAPHIC)
                demographic_fields = ["Sex", "Ethnicity", "Jaundice", "Family_mem_with_ASD"]
                for field in demographic_fields:
                    val = str(case_data.get(field, "")).strip()
                    if val:
                        session.run("""
                            MATCH (c:Case {upload_id: $upload_id}), (d:DemographicAttribute {type: $field, value: $val})
                            MERGE (c)-[:HAS_DEMOGRAPHIC]->(d)
                        """, upload_id=upload_id, field=field, val=val)

                # Connect to SubmitterType node (SUBMITTED_BY)
                submitter = str(case_data.get("Who_completed_the_test", "")).strip()
                if submitter:
                    session.run("""
                        MATCH (c:Case {upload_id: $upload_id}), (s:SubmitterType {type: $submitter})
                        MERGE (c)-[:SUBMITTED_BY]->(s)
                    """, upload_id=upload_id, submitter=submitter)

                logger.info(f"Temporary case {upload_id} inserted successfully with relationships.")
                return True

            except Exception as e:
                logger.error(f"Failed to insert temporary case {upload_id}: {str(e)}", exc_info=True)
                return False

    def delete_temporary_case(self, driver, upload_id: str) -> None:
        """Delete the temporary case node and all its relationships."""
        logger.info(f"Deleting temporary case with upload_id {upload_id}")
        with driver.session() as session:
            try:
                session.run("""
                    MATCH (c:Case {upload_id: $upload_id})
                    DETACH DELETE c
                """, upload_id=upload_id)
                logger.info(f"Temporary case {upload_id} deleted successfully.")
            except Exception as e:
                logger.error(f"Failed to delete temporary case {upload_id}: {str(e)}", exc_info=True)

    def build_base_graph(self, driver, upload_id: str) -> Optional[Tuple[nx.Graph, str]]:
        """Enhanced graph construction with better handling of edge cases"""
        with driver.session() as session:
            # Get both direct connections AND reverse connections (who answered similarly)
            result = session.run(f"""
                MATCH (c:Case {{upload_id: '{upload_id}'}})
                OPTIONAL MATCH (c)-[r]->(n)
                OPTIONAL MATCH (c)<-[r2]-(n2)
                OPTIONAL MATCH (c)-[s:SIMILAR_TO]-(other_case:Case)
                WHERE other_case.embedding IS NOT NULL 
                AND other_case.id IS NOT NULL 
                AND other_case <> c
                RETURN c.id AS case_id_num,
                    collect(DISTINCT {{ node: n, rel_type: type(r), rel_value: r.value }}) 
                        AS direct_neighbors,
                    collect(DISTINCT {{ node: n2, rel_type: type(r2), rel_value: r2.value }}) 
                        AS incoming_neighbors,
                    collect(DISTINCT {{ node: other_case, rel_type: type(s), rel_value: s.value }}) 
                        AS similar_cases_neighbors
            """).single()

            if not result or result["case_id_num"] is None:
                logger.error(f"Case {upload_id} not found in graph")
                return None

            case_id_num = result['case_id_num']
            case_node_name = f"Case_{case_id_num}"
            G = nx.Graph()
            G.add_node(case_node_name, type="Case", id=case_id_num)

            # Helper function to add nodes and edges
            def add_neighbor_data(neighbor_data, is_incoming=False):
                for data in neighbor_data:
                    neighbor = data['node']
                    if not neighbor:
                        continue
                        
                    # Create neighbor node
                    if 'BehaviorQuestion' in neighbor.labels:
                        neighbor_node_name = f"Q_{neighbor['name']}"
                        G.add_node(neighbor_node_name, type="BehaviorQuestion", name=neighbor['name'])
                    elif 'DemographicAttribute' in neighbor.labels:
                        neighbor_node_name = f"D_{neighbor['type']}_{str(neighbor['value']).replace(' ', '_')}"
                        G.add_node(neighbor_node_name, type="DemographicAttribute", 
                                attribute_type=neighbor['type'], value=neighbor['value'])
                    elif 'SubmitterType' in neighbor.labels:
                        neighbor_node_name = f"S_{str(neighbor['type']).replace(' ', '_')}"
                        G.add_node(neighbor_node_name, type="SubmitterType", 
                                submitter_type=neighbor['type'])
                    else:  # Case node
                        neighbor_node_name = f"Case_{neighbor['id']}"
                        G.add_node(neighbor_node_name, type="Case", id=neighbor['id'])

                    # Add edge with weight based on answer value (for HAS_ANSWER)
                    edge_attrs = {"type": data['rel_type']}
                    if data['rel_value'] is not None:
                        edge_attrs["value"] = data['rel_value']
                        if data['rel_type'] == "HAS_ANSWER":
                            edge_attrs["weight"] = float(data['rel_value'])  # Higher weight for stronger answers

                    if is_incoming:
                        G.add_edge(neighbor_node_name, case_node_name, **edge_attrs)
                    else:
                        G.add_edge(case_node_name, neighbor_node_name, **edge_attrs)

            # Add all connection types
            add_neighbor_data(result['direct_neighbors'])
            add_neighbor_data(result['incoming_neighbors'], is_incoming=True)
            add_neighbor_data(result['similar_cases_neighbors'])

            # Special handling for cases with all zero answers
            if len(G.edges(case_node_name)) == 0:
                logger.warning("Case has no answers - connecting to default nodes")
                for q in [f"Q_A{i}" for i in range(1, 11)]:
                    if q not in G:
                        G.add_node(q, type="BehaviorQuestion", name=q.split('_')[1])
                    G.add_edge(case_node_name, q, type="HAS_ANSWER", value=0, weight=0.1)

            return G, case_node_name

    def augment_with_similarity(self, driver, G: nx.Graph, case_node_name: str) -> None:
        """Enhanced similarity augmentation with inverse relationships"""
        original_case_id = int(case_node_name.split('_')[1])
        
        with driver.session() as session:
            # Find cases with similar answer patterns (both same and inverse)
            similar_results = session.run("""
                MATCH (c:Case {id: $case_id_num})-[:HAS_ANSWER]->(q:BehaviorQuestion)
                OPTIONAL MATCH (similar_case:Case)-[r:HAS_ANSWER]->(q)
                WHERE c <> similar_case AND similar_case.embedding IS NOT NULL
                WITH 
                    similar_case,
                    sum(CASE WHEN r.value = 0 THEN 1 ELSE 0 END) AS zero_matches,
                    sum(CASE WHEN r.value > 0 THEN 1 ELSE 0 END) AS non_zero_matches,
                    count(q) AS total_shared
                WHERE total_shared >= $min_similarity
                RETURN 
                    similar_case.id AS similar_id_num,
                    zero_matches,
                    non_zero_matches,
                    total_shared,
                    CASE 
                        WHEN zero_matches >= non_zero_matches THEN 1.0 - (zero_matches/10.0)
                        ELSE non_zero_matches/10.0
                    END AS similarity_score
                ORDER BY similarity_score DESC
                LIMIT 20
            """, case_id_num=original_case_id, min_similarity=self.MIN_SIMILARITY)

            for record in similar_results:
                similar_id_num = record['similar_id_num']
                similar_node_name = f"Case_{similar_id_num}"
                weight = record['similarity_score']

                if similar_node_name not in G:
                    G.add_node(similar_node_name, type="Case", id=similar_id_num)
                
                if case_node_name != similar_node_name and not G.has_edge(case_node_name, similar_node_name):
                    G.add_edge(case_node_name, similar_node_name, 
                            type="SIMILAR_TO", 
                            weight=weight,
                            zero_matches=record['zero_matches'],
                            non_zero_matches=record['non_zero_matches'])

    def generate_embedding(self, G: nx.Graph, case_node_name: str) -> Optional[List[float]]:
        """Enhanced embedding generation with weighted walks"""
        """Modified embedding generation with leakage checks"""
        # 1. Verify no label information in graph
        for node in G.nodes:
            if 'label' in G.nodes[node] and G.nodes[node]['label'] in ['Yes', 'No']:
                raise ValueError("Label information found in graph nodes")
        
        for u, v, data in G.edges(data=True):
            if data.get('type') == 'SCREENED_FOR':
                raise ValueError("Label relationship found in graph edges")
        temp_dir = None
        try:
            temp_dir = tempfile.mkdtemp()
            
            # Ensure minimum connectivity
            if len(G.edges(case_node_name)) < self.MIN_CONNECTIONS:
                logger.warning(f"Insufficient connections ({len(G.edges(case_node_name))}) for meaningful embedding")
                return [0.0] * self.EMBEDDING_DIM  # Return zero vector as fallback

            # Use weighted Node2Vec if available
            try:
                node2vec = Node2Vec(
                    G,
                    dimensions=self.EMBEDDING_DIM,
                    walk_length=self.NODE2VEC_WALK_LENGTH,
                    num_walks=self.NODE2VEC_NUM_WALKS,
                    workers=self.NODE2VEC_WORKERS,
                    p=self.NODE2VEC_P,
                    q=self.NODE2VEC_Q,
                    weight_key='weight',  # Use edge weights if available
                    temp_folder=temp_dir
                )
            except TypeError:  # Fallback for Node2Vec versions without weight_key
                node2vec = Node2Vec(
                    G,
                    dimensions=self.EMBEDDING_DIM,
                    walk_length=self.NODE2VEC_WALK_LENGTH,
                    num_walks=self.NODE2VEC_NUM_WALKS,
                    workers=self.NODE2VEC_WORKERS,
                    p=self.NODE2VEC_P,
                    q=self.NODE2VEC_Q,
                    temp_folder=temp_dir
                )
            
            model = node2vec.fit(window=self.NODE2VEC_WINDOW, min_count=1)
            
            if case_node_name not in model.wv:
                logger.error(f"No embedding generated for {case_node_name}")
                return None

            embedding = model.wv[case_node_name].tolist()
            # Post-processing normalization
            embedding_norm = np.linalg.norm(embedding)
            if embedding_norm > 0:
                embedding = (embedding / embedding_norm).tolist()
            
            return embedding
            
        except Exception as e:
            logger.error(f"Embedding generation failed: {str(e)}")
            return None
        finally:
            if temp_dir and os.path.exists(temp_dir):
                shutil.rmtree(temp_dir, ignore_errors=True)

    def generate_embedding_for_case(self, upload_id: str, case_data: dict) -> Optional[List[float]]:
        """Main embedding generation workflow for a temporary new case."""
        driver = None
        try:
            driver = self.get_driver()

            # Step 0: Insert temporary case node with relationships
            inserted = self.insert_temporary_case(driver, upload_id, case_data)
            if not inserted:
                logger.error("Failed to insert temporary case. Aborting embedding generation.")
                return None
            
            # Step 1: Build base graph (including the new temporary case and its direct neighbors)
            graph_result = self.build_base_graph(driver, upload_id)
            if not graph_result:
                logger.error("Failed to build base graph for temporary case.")
                self.delete_temporary_case(driver, upload_id)
                return None
                
            G, case_node_name = graph_result
            
            # Step 2: Augment with similarity if initial connections are low
            # The count of edges from the current case node to others
            if len(G.edges(case_node_name)) < self.MIN_CONNECTIONS:
                logger.info(f"Initial connections for {case_node_name} low ({len(G.edges(case_node_name))}), augmenting with similar cases...")
                self.augment_with_similarity(driver, G, case_node_name)
                
                if len(G.edges(case_node_name)) < self.MIN_CONNECTIONS:
                    logger.warning(f"Insufficient connections ({len(G.edges(case_node_name))}) for {case_node_name} even after augmentation. Embedding might be poor.")
            
            # Step 3: Generate embedding
            logger.info(f"Generating embedding for {case_node_name} (NetworkX graph: {len(G.nodes)} nodes, {len(G.edges)} edges)")
            embedding = self.generate_embedding(G, case_node_name)
            if not embedding:
                logger.error(f"Embedding generation returned None for {case_node_name}. Could be due to no edges, or other issues.")
                self.delete_temporary_case(driver, upload_id)
                return None

            # Step 4: Delete temporary node
            self.delete_temporary_case(driver, upload_id)

            logger.info(f"Successfully generated embedding for case {case_node_name}")
            return embedding

        except Exception as e:
            logger.critical(f"Fatal error in generate_embedding_for_case workflow: {str(e)}", exc_info=True)
            # Ensure temporary node is deleted even on critical errors
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
            # Return the embedding as JSON to stdout for main_app.py to capture
            print(json.dumps(embedding))
            sys.exit(0)
        else:
            logger.error("Embedding generation failed")
            sys.exit(1)
        
    except Exception as e:
        logger.critical(f"Fatal error in __main__ of generate_case_embedding.py: {str(e)}", exc_info=True)
        sys.exit(1)