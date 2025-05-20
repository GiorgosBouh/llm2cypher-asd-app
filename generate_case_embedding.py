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
import shutil  # Make sure this is imported

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class EmbeddingGenerator:
    def __init__(self):
        # Make these parameters consistent with main app
        self.EMBEDDING_DIM = 128
        self.MIN_CONNECTIONS = 3
        self.MAX_RETRIES = 3
        self.MIN_SIMILARITY = 5  # Minimum shared answers for similarity
        self.EMBEDDING_NORMALIZATION = True
        
        # Add Node2Vec parameters to match kg_builder
        self.NODE2VEC_WALK_LENGTH = 30
        self.NODE2VEC_NUM_WALKS = 100
        self.NODE2VEC_P = 1.0
        self.NODE2VEC_Q = 0.5
        self.NODE2VEC_WORKERS = 4
        self.NODE2VEC_WINDOW = 5
        self.NODE2VEC_BATCH_WORDS = 1000

    def get_driver(self):
        """Create a Neo4j driver with robust settings"""
        return GraphDatabase.driver(
            os.getenv("NEO4J_URI"),
            auth=(os.getenv("NEO4J_USER"), os.getenv("NEO4J_PASSWORD")),
            connection_timeout=30,
            max_connection_lifetime=7200,
            max_connection_pool_size=50
        )

    def validate_embedding(self, embedding: List[float]) -> bool:
        """Ensure embedding is valid and normalized"""
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

    def build_base_graph(self, driver, upload_id: str) -> Optional[tuple]:
        """Construct the initial graph structure"""
        with driver.session() as session:
            # Enhanced query to get more relevant connections
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

            case_id = f"Case_{result['case_id']}"  # Consistent with kg_builder format
            G = nx.Graph()
            G.add_node(case_id)
            
            # Add initial connections with validation
            for neighbor in result["neighbors"]:
                if neighbor:
                    neighbor_id = f"Case_{neighbor}" if isinstance(neighbor, int) else str(neighbor)
                    G.add_edge(case_id, neighbor_id)
            
            return G, case_id

    def augment_with_similarity(self, driver, G: nx.Graph, case_id: str) -> None:
        """Enhance graph with similarity-based connections"""
        original_case_id = int(case_id.split('_')[1])  # Extract numeric ID from "Case_123"
        
        with driver.session() as session:
            # More comprehensive similarity query
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
                weight = record['shared_answers'] / 10.0  # Normalize weight to 0-2 range
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
                temp_folder=temp_dir
            )
            
            model = node2vec.fit(
                window=self.NODE2VEC_WINDOW,
                min_count=1,
                batch_words=self.NODE2VEC_BATCH_WORDS
            )
            
            embedding = model.wv[case_id].tolist()
            
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

    def store_embedding(self, driver, upload_id: str, embedding: List[float]) -> bool:
        """Safely store embedding in Neo4j"""
        try:
            with driver.session() as session:
                result = session.run("""
                    MATCH (c:Case {upload_id: $upload_id})
                    SET c.embedding = $embedding,
                        c.embedding_version = 2.1,
                        c.last_embedding_update = timestamp()
                    RETURN count(c) AS updated
                """, upload_id=upload_id, embedding=embedding).single()
                
                return result["updated"] > 0
        except Exception as e:
            logger.error(f"Failed to store embedding: {str(e)}")
            return False

    def generate_embedding_for_case(self, upload_id: str) -> bool:
        """Main embedding generation workflow"""
        driver = None
        try:
            driver = self.get_driver()
            
            for attempt in range(self.MAX_RETRIES):
                try:
                    # Step 1: Build base graph
                    graph_result = self.build_base_graph(driver, upload_id)
                    if not graph_result:
                        logger.error("Failed to build base graph")
                        return False
                        
                    G, case_id = graph_result
                    
                    # Step 2: Augment with similarity if needed
                    if len(G.edges(case_id)) < self.MIN_CONNECTIONS:
                        logger.info(f"Initial connections low ({len(G.edges(case_id))}), augmenting...")
                        self.augment_with_similarity(driver, G, case_id)
                        
                        if len(G.edges(case_id)) < self.MIN_CONNECTIONS:
                            logger.warning(f"Insufficient connections ({len(G.edges(case_id))}) after augmentation")
                            return False

                    # Step 3: Generate embedding
                    logger.info(f"Generating embedding for {case_id} with {len(G.nodes)} nodes and {len(G.edges)} edges")
                    embedding = self.generate_embedding(G, case_id)
                    if not embedding:
                        logger.error("Embedding generation returned None")
                        return False
                        
                    # Step 4: Store embedding
                    if not self.store_embedding(driver, upload_id, embedding):
                        logger.error("Failed to store embedding in Neo4j")
                        return False
                        
                    logger.info(f"Successfully generated embedding for case {case_id}")
                    return True

                except Exception as e:
                    if attempt == self.MAX_RETRIES - 1:
                        raise
                    wait_time = 2 ** attempt
                    logger.warning(f"Attempt {attempt + 1} failed, retrying in {wait_time}s...")
                    time.sleep(wait_time)

        except Exception as e:
            logger.error(f"Embedding generation failed: {str(e)}")
            return False

        finally:
            if driver:
                driver.close()

if __name__ == "__main__":
    try:
        if len(sys.argv) < 2:
            raise ValueError("Missing upload_id parameter")
        
        upload_id = sys.argv[1]
        logger.info(f"Starting embedding generation for upload_id: {upload_id}")
        generator = EmbeddingGenerator()
        success = generator.generate_embedding_for_case(upload_id)
        logger.info(f"Embedding generation {'succeeded' if success else 'failed'}")
        sys.exit(0 if success else 1)
        
    except Exception as e:
        logger.critical(f"Fatal error: {str(e)}")
        sys.exit(1)