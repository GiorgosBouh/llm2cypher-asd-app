import numpy as np
import networkx as nx
from node2vec import Node2Vec
from neo4j import GraphDatabase
import sys
import os
import time
import tempfile
import logging
from typing import List, Optional, Tuple, Dict, Any
import shutil
import json
from dotenv import load_dotenv
import pandas as pd

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

class EmbeddingGenerator:
    def __init__(self):
        self.EMBEDDING_DIM = 128
        self.MIN_CONNECTIONS = 3  # Reduced for better coverage
        self.MIN_SIMILARITY = 2   # Reduced similarity threshold
        
        # Updated Node2Vec parameters
        self.NODE2VEC_WALK_LENGTH = 10
        self.NODE2VEC_NUM_WALKS = 50
        self.NODE2VEC_P = 0.5
        self.NODE2VEC_Q = 2.0
        self.NODE2VEC_WORKERS = 4
        self.NODE2VEC_WINDOW = 10

    def get_driver(self):
        """Initialize Neo4j driver using environment variables"""
        uri = os.getenv("NEO4J_URI")
        user = os.getenv("NEO4J_USER")
        password = os.getenv("NEO4J_PASSWORD")
        
        if not uri or not user or not password:
            raise ValueError("Missing Neo4j environment variables")
            
        return GraphDatabase.driver(
            uri,
            auth=(user, password),
            connection_timeout=900,
            max_connection_lifetime=7200,
            max_connection_pool_size=50
        )

    def validate_embedding(self, embedding: List[float]) -> bool:
        """Validate a generated embedding list"""
        if not embedding or len(embedding) != self.EMBEDDING_DIM:
            logger.warning(f"Invalid embedding length: {len(embedding) if embedding else 0} != {self.EMBEDDING_DIM}")
            return False
            
        if any(np.isnan(x) or not np.isfinite(x) for x in embedding):
            logger.warning("Embedding contains NaN or infinite values")
            return False
            
        if np.all(np.array(embedding) == 0):
            logger.warning("Embedding is all zeros")
            return False
            
        return True

    def insert_temporary_case(self, driver, upload_id: str, case_data: Dict[str, Any]) -> bool:
        """Insert temporary Case node and related nodes/relationships"""
        logger.info(f"Inserting temporary case with upload_id {upload_id}")
        
        with driver.session() as session:
            try:
                # Create the Case node
                case_no = int(case_data.get("Case_No", -1))
                session.run("""
                    CREATE (c:Case {upload_id: $upload_id, id: $case_no})
                """, upload_id=upload_id, case_no=case_no)
                
                # Connect to BehaviorQuestion nodes (HAS_ANSWER)
                for i in range(1, 11):
                    question_label = f"A{i}"
                    raw_val = case_data.get(question_label, "")
                    
                    try:
                        # Handle string values that might be comma-separated decimals
                        val_str = str(raw_val).strip().replace(",", ".")
                        val = float(val_str) if val_str else 0
                        if np.isfinite(val):
                            answer_value = int(val)
                        else:
                            logger.warning(f"Non-finite value for {question_label}: {raw_val}")
                            answer_value = 0
                    except (ValueError, TypeError) as e:
                        logger.warning(f"Failed to convert {question_label}: {raw_val} -> {e}")
                        answer_value = 0

                    session.run("""
                        MATCH (c:Case {upload_id: $upload_id}), (q:BehaviorQuestion {name: $question})
                        MERGE (c)-[r:HAS_ANSWER]->(q)
                        SET r.value = $value
                    """, upload_id=upload_id, question=question_label, value=answer_value)

                # Connect to DemographicAttribute nodes (HAS_DEMOGRAPHIC)
                demographic_fields = ["Sex", "Ethnicity", "Jaundice", "Family_mem_with_ASD"]
                for field in demographic_fields:
                    val = str(case_data.get(field, "")).strip()
                    if val and val.lower() not in ['nan', 'none', '']:
                        session.run("""
                            MATCH (c:Case {upload_id: $upload_id}), 
                                  (d:DemographicAttribute {type: $field, value: $val})
                            MERGE (c)-[:HAS_DEMOGRAPHIC]->(d)
                        """, upload_id=upload_id, field=field, val=val)

                # Connect to SubmitterType node (SUBMITTED_BY)
                submitter = str(case_data.get("Who_completed_the_test", "")).strip()
                if submitter and submitter.lower() not in ['nan', 'none', '']:
                    session.run("""
                        MATCH (c:Case {upload_id: $upload_id}), (s:SubmitterType {type: $submitter})
                        MERGE (c)-[:SUBMITTED_BY]->(s)
                    """, upload_id=upload_id, submitter=submitter)

                logger.info(f"Temporary case {upload_id} inserted successfully")
                return True

            except Exception as e:
                logger.error(f"Failed to insert temporary case {upload_id}: {str(e)}", exc_info=True)
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
                logger.info(f"Temporary case {upload_id} deleted successfully")
            except Exception as e:
                logger.error(f"Failed to delete temporary case {upload_id}: {str(e)}")

    def build_base_graph(self, driver, upload_id: str) -> Optional[Tuple[nx.Graph, str]]:
        """Build base graph for the case with enhanced connectivity"""
        with driver.session() as session:
            try:
                # Get comprehensive case information
                result = session.run("""
                    MATCH (c:Case {upload_id: $upload_id})
                    OPTIONAL MATCH (c)-[r]->(n)
                    WHERE n IS NOT NULL
                    OPTIONAL MATCH (c)<-[r2]-(n2)
                    WHERE n2 IS NOT NULL
                    OPTIONAL MATCH (c)-[s:SIMILAR_TO]-(other_case:Case)
                    WHERE other_case.embedding IS NOT NULL 
                      AND other_case.id IS NOT NULL 
                      AND other_case <> c
                    RETURN c.id AS case_id_num,
                        collect(DISTINCT {
                            node: n, 
                            rel_type: type(r), 
                            rel_value: r.value,
                            labels: labels(n)
                        }) AS direct_neighbors,
                        collect(DISTINCT {
                            node: n2, 
                            rel_type: type(r2), 
                            rel_value: r2.value,
                            labels: labels(n2)
                        }) AS incoming_neighbors,
                        collect(DISTINCT {
                            node: other_case, 
                            rel_type: type(s), 
                            rel_value: s.weight,
                            labels: labels(other_case)
                        }) AS similar_cases
                """, upload_id=upload_id).single()

                if not result or result["case_id_num"] is None:
                    logger.error(f"Case {upload_id} not found in graph")
                    return None

                case_id_num = result['case_id_num']
                case_node_name = f"Case_{case_id_num}"
                G = nx.Graph()
                G.add_node(case_node_name, type="Case", id=case_id_num)

                # Helper function to add neighbors
                def add_neighbors(neighbor_data, is_incoming=False):
                    for data in neighbor_data:
                        neighbor = data['node']
                        if not neighbor:
                            continue
                            
                        labels = data.get('labels', [])
                        
                        # Create appropriate neighbor node name
                        if 'BehaviorQuestion' in labels:
                            neighbor_node_name = f"Q_{neighbor.get('name', 'unknown')}"
                            G.add_node(neighbor_node_name, type="BehaviorQuestion")
                        elif 'DemographicAttribute' in labels:
                            attr_type = neighbor.get('type', 'unknown')
                            attr_value = str(neighbor.get('value', 'unknown')).replace(' ', '_')
                            neighbor_node_name = f"D_{attr_type}_{attr_value}"
                            G.add_node(neighbor_node_name, type="DemographicAttribute")
                        elif 'SubmitterType' in labels:
                            submitter_type = str(neighbor.get('type', 'unknown')).replace(' ', '_')
                            neighbor_node_name = f"S_{submitter_type}"
                            G.add_node(neighbor_node_name, type="SubmitterType")
                        elif 'Case' in labels:
                            neighbor_node_name = f"Case_{neighbor.get('id', 'unknown')}"
                            G.add_node(neighbor_node_name, type="Case")
                        else:
                            continue

                        # Add edge with appropriate weight
                        edge_attrs = {"type": data['rel_type']}
                        rel_value = data.get('rel_value')
                        if rel_value is not None:
                            edge_attrs["value"] = rel_value
                            # Set weight based on relationship type
                            if data['rel_type'] == "HAS_ANSWER":
                                edge_attrs["weight"] = max(0.1, float(rel_value))
                            elif data['rel_type'] == "SIMILAR_TO":
                                edge_attrs["weight"] = max(0.1, float(rel_value))
                            else:
                                edge_attrs["weight"] = 1.0
                        else:
                            edge_attrs["weight"] = 1.0

                        if is_incoming:
                            G.add_edge(neighbor_node_name, case_node_name, **edge_attrs)
                        else:
                            G.add_edge(case_node_name, neighbor_node_name, **edge_attrs)

                # Add all types of neighbors
                add_neighbors(result['direct_neighbors'])
                add_neighbors(result['incoming_neighbors'], is_incoming=True)
                add_neighbors(result['similar_cases'])

                # Ensure minimum connectivity
                if len(G.edges(case_node_name)) == 0:
                    logger.warning("Case has no connections - adding default connections")
                    for i in range(1, 11):
                        q_node = f"Q_A{i}"
                        if q_node not in G:
                            G.add_node(q_node, type="BehaviorQuestion")
                        G.add_edge(case_node_name, q_node, type="HAS_ANSWER", value=0, weight=0.1)

                logger.info(f"Built graph with {len(G.nodes)} nodes and {len(G.edges)} edges")
                return G, case_node_name

            except Exception as e:
                logger.error(f"Error building base graph: {str(e)}", exc_info=True)
                return None

    def augment_with_similarity(self, driver, G: nx.Graph, case_node_name: str) -> None:
        """Augment graph with similarity relationships"""
        try:
            original_case_id = int(case_node_name.split('_')[1])
            
            with driver.session() as session:
                # Find cases with similar demographics and answer patterns
                similar_results = session.run("""
                    MATCH (c:Case {id: $case_id})
                    OPTIONAL MATCH (c)-[:HAS_ANSWER]->(q:BehaviorQuestion)
                    WITH c, collect({question: q.name, value: c-[:HAS_ANSWER]->(q)}) as case_answers
                    
                    MATCH (other:Case)-[:HAS_ANSWER]->(q2:BehaviorQuestion)
                    WHERE c <> other AND other.embedding IS NOT NULL
                    
                    WITH c, case_answers, other, 
                         collect({question: q2.name, value: other-[:HAS_ANSWER]->(q2)}) as other_answers
                    
                    // Calculate similarity based on shared answer patterns
                    WITH c, other,
                         size([x IN case_answers WHERE x.value = 1]) as case_pos,
                         size([y IN other_answers WHERE y.value = 1]) as other_pos,
                         size([x IN case_answers WHERE x IN other_answers AND x.value = 1]) as shared_pos
                    
                    WHERE shared_pos >= $min_similarity OR 
                          (case_pos = 0 AND other_pos = 0) // Both have no positive answers
                    
                    RETURN other.id AS similar_id,
                           CASE 
                               WHEN case_pos + other_pos = 0 THEN 0.8
                               ELSE toFloat(shared_pos) / (case_pos + other_pos - shared_pos)
                           END AS similarity_score
                    ORDER BY similarity_score DESC
                    LIMIT 10
                """, case_id=original_case_id, min_similarity=self.MIN_SIMILARITY)

                for record in similar_results:
                    similar_id = record['similar_id']
                    similar_node_name = f"Case_{similar_id}"
                    weight = max(0.1, record['similarity_score'])

                    if similar_node_name not in G:
                        G.add_node(similar_node_name, type="Case", id=similar_id)
                    
                    if not G.has_edge(case_node_name, similar_node_name):
                        G.add_edge(case_node_name, similar_node_name, 
                                 type="SIMILAR_TO", weight=weight)

                logger.info(f"Added {len(list(similar_results))} similarity connections")

        except Exception as e:
            logger.error(f"Error in similarity augmentation: {str(e)}")

    def generate_embedding(self, G: nx.Graph, case_node_name: str) -> Optional[List[float]]:
        """Generate embedding vector for the case using Node2Vec"""
        temp_dir = None
        try:
            # Leakage check: ensure no SCREENED_FOR relationships
            for u, v, data in G.edges(data=True):
                if data.get('type') == 'SCREENED_FOR':
                    raise ValueError("❌ Label relationship (SCREENED_FOR) found in graph")

            temp_dir = tempfile.mkdtemp()

            # Validate weights
            for u, v, data in G.edges(data=True):
                weight = data.get("weight", 1.0)
                if not isinstance(weight, (float, int)) or not np.isfinite(weight) or weight <= 0:
                    data["weight"] = 1.0

            # Check connectivity
            if case_node_name not in G:
                logger.error(f"Case node '{case_node_name}' not found in graph")
                return None

            degree = G.degree(case_node_name)
            if degree < self.MIN_CONNECTIONS:
                logger.warning(f"Node '{case_node_name}' has only {degree} connections")
                # Return a simple default embedding based on available connections
                if degree == 0:
                    return [0.1] * self.EMBEDDING_DIM
                
            # Run Node2Vec with error handling
            try:
                node2vec = Node2Vec(
                    G,
                    dimensions=self.EMBEDDING_DIM,
                    walk_length=self.NODE2VEC_WALK_LENGTH,
                    num_walks=self.NODE2VEC_NUM_WALKS,
                    workers=self.NODE2VEC_WORKERS,
                    p=self.NODE2VEC_P,
                    q=self.NODE2VEC_Q,
                    temp_folder=temp_dir,
                    quiet=True
                )
            except Exception as e:
                logger.warning(f"Node2Vec initialization failed: {e}. Using fallback parameters.")
                node2vec = Node2Vec(
                    G,
                    dimensions=self.EMBEDDING_DIM,
                    walk_length=5,
                    num_walks=10,
                    workers=1,
                    temp_folder=temp_dir,
                    quiet=True
                )

            model = node2vec.fit(
                window=self.NODE2VEC_WINDOW, 
                min_count=1,
                sg=1,  # Skip-gram
                epochs=10
            )

            if case_node_name not in model.wv:
                logger.error(f"No embedding generated for '{case_node_name}'")
                return None

            embedding = model.wv[case_node_name].tolist()

            # Validate and normalize
            if not all(np.isfinite(embedding)):
                logger.error(f"Embedding contains non-finite values")
                return None

            norm = np.linalg.norm(embedding)
            if norm == 0:
                logger.warning("Zero-norm embedding - returning raw vector")
                return embedding if self.validate_embedding(embedding) else None

            normalized_embedding = (np.array(embedding) / norm).tolist()
            return normalized_embedding if self.validate_embedding(normalized_embedding) else None

        except Exception as e:
            logger.error(f"Embedding generation failed: {str(e)}", exc_info=True)
            return None

        finally:
            if temp_dir and os.path.exists(temp_dir):
                shutil.rmtree(temp_dir, ignore_errors=True)

    def generate_embedding_for_case(self, upload_id: str, case_data: Dict[str, Any]) -> Optional[List[float]]:
        """Main workflow for generating embedding for a new case"""
        driver = None
        try:
            driver = self.get_driver()

            # Step 1: Insert temporary case
            if not self.insert_temporary_case(driver, upload_id, case_data):
                return None

            # Step 2: Build base graph
            graph_result = self.build_base_graph(driver, upload_id)
            if not graph_result:
                return None

            G, case_node_name = graph_result

            # Step 3: Augment with similarity if needed
            if len(G.edges(case_node_name)) < self.MIN_CONNECTIONS:
                self.augment_with_similarity(driver, G, case_node_name)

            # Step 4: Generate embedding
            embedding = self.generate_embedding(G, case_node_name)

            # Step 5: Store embedding in the temporary node
            if embedding and self.validate_embedding(embedding):
                with driver.session() as session:
                    session.run("""
                        MATCH (c:Case {upload_id: $upload_id})
                        SET c.embedding = $embedding,
                            c.embedding_version = 2.1,
                            c.last_embedding_update = timestamp()
                    """, upload_id=upload_id, embedding=embedding)
                
                logger.info(f"Successfully generated and stored embedding for {upload_id}")
                return embedding
            else:
                logger.error("Failed to generate valid embedding")
                return None

        except Exception as e:
            logger.critical(f"Fatal error in embedding generation workflow: {str(e)}", exc_info=True)
            return None
        finally:
            if driver:
                driver.close()


if __name__ == "__main__":
    try:
        if len(sys.argv) < 3:
            raise ValueError("Usage: generate_case_embedding.py <upload_id> <case_data_json>")
        
        upload_id = sys.argv[1]
        case_data_json = sys.argv[2]
        
        try:
            case_data = json.loads(case_data_json)
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON data: {e}")
            sys.exit(1)

        logger.info(f"Starting embedding generation for upload_id: {upload_id}")
        generator = EmbeddingGenerator()
        embedding = generator.generate_embedding_for_case(upload_id, case_data)

        if embedding is not None and generator.validate_embedding(embedding):
            logger.info("Embedding generated successfully")
            print(json.dumps(embedding))
            sys.exit(0)
        else:
            logger.error("Embedding generation failed")
            sys.exit(1)
        
    except Exception as e:
        logger.critical(f"Fatal error in main: {str(e)}", exc_info=True)
        sys.exit(1)