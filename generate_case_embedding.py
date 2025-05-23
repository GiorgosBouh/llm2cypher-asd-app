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
from dotenv import load_dotenv # ADDED: For consistent env var loading

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
        self.MIN_CONNECTIONS = 3
        self.MAX_RETRIES = 3 # Not explicitly used, but good to keep if intended
        self.MIN_SIMILARITY = 5  # Minimum shared answers for similarity
        
        # Node2Vec parameters should ideally match kg_builder_2.py for consistent embeddings
        self.NODE2VEC_WALK_LENGTH = 30
        self.NODE2VEC_NUM_WALKS = 100
        self.NODE2VEC_P = 1.0
        self.NODE2VEC_Q = 0.5
        self.NODE2VEC_WORKERS = 4
        self.NODE2VEC_WINDOW = 10 # Adjusted to match kg_builder_2.py's default
        # self.NODE2VEC_BATCH_WORDS = 1000 # This parameter is usually managed by Node2Vec.fit()

    def get_driver(self):
        """Initializes Neo4j driver using environment variables."""
        return GraphDatabase.driver(
            os.getenv("NEO4J_URI"),
            auth=(os.getenv("NEO4J_USER"), os.getenv("NEO4J_PASSWORD")),
            connection_timeout=30,
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
            
        # Removed self.EMBEDDING_NORMALIZATION logic here. 
        # Normalization should be handled by the ML model or explicitly after embedding generation if needed.
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
        """
        Construct the initial NetworkX graph containing the temporary case and its direct neighbors.
        Ensures consistent node naming as "Type_ID" for Node2Vec compatibility.
        """
        with driver.session() as session:
            # Fetch the temporary case and its direct neighbors (questions, demographics, submitter)
            # and potentially other Case nodes it is similar to (from SIMILAR_TO relationships).
            result = session.run(f"""
                MATCH (c:Case {{upload_id: '{upload_id}'}})
                OPTIONAL MATCH (c)-[r]->(n)
                OPTIONAL MATCH (c)-[s:SIMILAR_TO]-(other_case:Case)
                WHERE other_case.embedding IS NOT NULL AND other_case <> c
                RETURN c.id AS case_id_num,
                       collect(DISTINCT {{ node: n, rel_type: type(r), rel_value: r.value, rel_weight: r.weight }}) AS direct_neighbors,
                       collect(DISTINCT {{ node: other_case, rel_type: type(s), rel_value: s.value, rel_weight: s.weight }}) AS similar_cases_neighbors
            """).single()

            if not result or result["case_id_num"] is None:
                logger.error(f"Temporary case with upload_id {upload_id} not found in graph during base graph build.")
                return None

            case_id_num = result['case_id_num']
            case_node_name = f"Case_{case_id_num}"
            G = nx.Graph()
            G.add_node(case_node_name, type="Case", id=case_id_num) # Add the new case node

            # Add direct relationships from the temporary case
            for neighbor_data in result['direct_neighbors']:
                neighbor = neighbor_data['node']
                rel_type = neighbor_data['rel_type']
                rel_value = neighbor_data['rel_value']
                rel_weight = neighbor_data['rel_weight'] if neighbor_data['rel_weight'] is not None else 1.0

                if neighbor:
                    # Consistent naming scheme for non-Case nodes
                    if 'name' in neighbor: # BehaviorQuestion
                        neighbor_node_name = f"Q_{neighbor['name']}"
                        G.add_node(neighbor_node_name, type="BehaviorQuestion", name=neighbor['name'])
                    elif 'type' in neighbor and 'value' in neighbor: # DemographicAttribute or SubmitterType
                        if 'DemographicAttribute' in neighbor.labels:
                             neighbor_node_name = f"D_{neighbor['type']}_{str(neighbor['value']).replace(' ', '_')}"
                             G.add_node(neighbor_node_name, type="DemographicAttribute", attribute_type=neighbor['type'], value=neighbor['value'])
                        elif 'SubmitterType' in neighbor.labels:
                             neighbor_node_name = f"S_{str(neighbor['type']).replace(' ', '_')}"
                             G.add_node(neighbor_node_name, type="SubmitterType", submitter_type=neighbor['type'])
                        else:
                            neighbor_node_name = f"Node_{neighbor.id}" # Fallback
                            G.add_node(neighbor_node_name, type="Generic")
                    else: # Fallback for unexpected nodes or Case nodes from initial match (should be handled by similar_cases_neighbors if also matched)
                        neighbor_node_name = f"Node_{neighbor.id}"
                        G.add_node(neighbor_node_name, type="Generic")

                    edge_attrs = {"type": rel_type, "weight": rel_weight}
                    if rel_value is not None:
                        edge_attrs["value"] = rel_value
                    G.add_edge(case_node_name, neighbor_node_name, **edge_attrs)

            # Add SIMILAR_TO relationships (already handled by main_app.py if needed, or by kg_builder_2.py)
            # This is more for ensuring existing similar nodes are in the graph for walks
            for similar_data in result['similar_cases_neighbors']:
                similar_node = similar_data['node']
                rel_type = similar_data['rel_type']
                rel_weight = similar_data['rel_weight'] if similar_data['rel_weight'] is not None else 1.0
                
                if similar_node and 'id' in similar_node:
                    similar_case_node_name = f"Case_{similar_node['id']}"
                    G.add_node(similar_case_node_name, type="Case", id=similar_node['id'])
                    G.add_edge(case_node_name, similar_case_node_name, type=rel_type, weight=rel_weight)


            logger.info(f"Base graph for {case_node_name} built with {len(G.nodes)} nodes and {len(G.edges)} edges.")
            if not G.nodes:
                raise ValueError("Graph built for temporary case is empty. No nodes found.")
            if not G.edges:
                logger.warning(f"Graph built for temporary case has no edges. Node2Vec might not generate meaningful embeddings.")
            
            return G, case_node_name

    def augment_with_similarity(self, driver, G: nx.Graph, case_node_name: str) -> None:
        """
        Augment the graph for the temporary case with SIMILAR_TO relationships
        to existing cases in the main graph based on shared answers.
        """
        original_case_id = int(case_node_name.split('_')[1])
        
        with driver.session() as session:
            # Query for existing cases that have similar answers and already have embeddings
            similar_results = session.run("""
                MATCH (c:Case {id: $case_id_num})-[:HAS_ANSWER]->(q)<-[:HAS_ANSWER]-(similar_case:Case)
                WHERE c <> similar_case AND similar_case.embedding IS NOT NULL
                WITH similar_case, c, count(q) AS shared_answers
                WHERE shared_answers >= $min_similarity
                RETURN similar_case.id AS similar_id_num, shared_answers
                ORDER BY shared_answers DESC 
                LIMIT 20
            """, case_id_num=original_case_id, min_similarity=self.MIN_SIMILARITY)

            for record in similar_results:
                similar_id_num = record['similar_id_num']
                similar_node_name = f"Case_{similar_id_num}"
                weight = record['shared_answers'] / 10.0 # Normalize similarity score

                # Add similar case node to graph if not already present
                if similar_node_name not in G:
                    G.add_node(similar_node_name, type="Case", id=similar_id_num)
                
                # Add edge, ensuring it's not a self-loop and doesn't already exist
                if case_node_name != similar_node_name and not G.has_edge(case_node_name, similar_node_name):
                    G.add_edge(case_node_name, similar_node_name, type="SIMILAR_TO", weight=weight)
                    logger.info(f"Added similarity edge: {case_node_name} -- {similar_node_name} (weight: {weight:.2f})")
                
        logger.info(f"Augmented graph for {case_node_name} with {len(G.edges(case_node_name))} total connections.")


    def generate_embedding(self, G: nx.Graph, case_node_name: str) -> Optional[List[float]]:
        """Generate Node2Vec embedding for the given case node in the graph."""
        temp_dir = None
        try:
            temp_dir = tempfile.mkdtemp()
            
            # Ensure the graph has at least one edge for meaningful Node2Vec results
            if len(G.edges(case_node_name)) == 0:
                logger.warning(f"Case node {case_node_name} has no edges in the graph. Node2Vec will not generate meaningful embeddings.")
                return None # Cannot generate useful embedding without connections

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
                weighted=True,  # ADDED: Ensure weights are used
                weight_key='weight' # ADDED: Specify weight key
            )
            
            model = node2vec.fit(
                window=self.NODE2VEC_WINDOW,
                min_count=1,
                batch_words=self.NODE2VEC_BATCH_WORDS
            )
            
            if case_node_name not in model.wv:
                logger.error(f"Node2Vec did not generate an embedding for {case_node_name}.")
                return None

            embedding = model.wv[case_node_name].tolist()
            
            if not self.validate_embedding(embedding):
                logger.error(f"Generated invalid embedding for {case_node_name} after Node2Vec fit.")
                return None
                
            return embedding
            
        except Exception as e:
            logger.error(f"Embedding generation failed for {case_node_name}: {str(e)}", exc_info=True)
            return None
        finally:
            if temp_dir and os.path.exists(temp_dir):
                try:
                    shutil.rmtree(temp_dir)
                except Exception as e:
                    logger.warning(f"Could not remove temp directory {temp_dir}: {str(e)}")

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
                    # Decide if you want to return None here or generate a potentially poor embedding
                    # For now, allow generation but log warning.
                    # self.delete_temporary_case(driver, upload_id)
                    # return None

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