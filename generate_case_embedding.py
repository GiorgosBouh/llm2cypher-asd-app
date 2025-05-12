import numpy as np
import networkx as nx
from node2vec import Node2Vec
from neo4j import GraphDatabase
import sys
import os
import time
import tempfile  # <-- Missing import added here
import logging
from typing import List, Optional

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
            result = session.run("""
                MATCH (c:Case {upload_id: $upload_id})
                OPTIONAL MATCH (c)-[r]->(neighbor)
                WHERE neighbor.id IS NOT NULL
                RETURN c.id AS case_id, 
                       collect(DISTINCT neighbor.id) AS neighbors
            """, upload_id=upload_id).single()

            if not result or not result["case_id"]:
                logger.error("Case not found or missing ID")
                return None

            case_id = str(result["case_id"])
            G = nx.Graph()
            G.add_node(case_id)
            
            # Add initial connections with validation
            for neighbor in result["neighbors"]:
                if neighbor:
                    G.add_edge(case_id, str(neighbor))
            
            return G, case_id

    def augment_with_similarity(self, driver, G: nx.Graph, case_id: str) -> None:
        """Enhance graph with similarity-based connections"""
        with driver.session() as session:
            similar = session.run("""
                MATCH (c:Case {id: $case_id})-[:HAS_ANSWER]->(q)<-[:HAS_ANSWER]-(similar:Case)
                WHERE c <> similar
                WITH similar, count(q) AS shared_answers
                WHERE shared_answers >= $min_similarity
                RETURN similar.id 
                ORDER BY shared_answers DESC 
                LIMIT 20
            """, case_id=int(case_id), min_similarity=self.MIN_SIMILARITY)

            for record in similar:
                G.add_edge(case_id, str(record["similar.id"]))

    def generate_embedding(self, G: nx.Graph, case_id: str) -> Optional[List[float]]:
        """Generate node2vec embedding with validation"""
        try:
            # Create temp directory safely
            temp_dir = tempfile.mkdtemp()
            
            node2vec = Node2Vec(
                G,
                dimensions=self.EMBEDDING_DIM,
                walk_length=20,
                num_walks=100,
                workers=2,
                quiet=True,
                temp_folder=temp_dir
            )
            
            model = node2vec.fit(
                window=5,
                min_count=1,
                batch_words=1000
            )
            
            embedding = model.wv[case_id].tolist()
            
            # Clean up temp directory
            try:
                shutil.rmtree(temp_dir)
            except Exception as e:
                logger.warning(f"Could not remove temp directory: {str(e)}")
            
            if not self.validate_embedding(embedding):
                logger.error("Generated invalid embedding")
                return None
                
            return embedding
            
        except Exception as e:
            logger.error(f"Embedding generation failed: {str(e)}")
            return None

    def store_embedding(self, driver, upload_id: str, embedding: List[float]) -> bool:
        """Safely store embedding in Neo4j"""
        try:
            with driver.session() as session:
                result = session.run("""
                    MATCH (c:Case {upload_id: $upload_id})
                    SET c.embedding = $embedding,
                        c.embedding_version = 2.0,
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
                        return False
                        
                    G, case_id = graph_result
                    
                    # Step 2: Augment with similarity if needed
                    if len(G.edges(case_id)) < self.MIN_CONNECTIONS:
                        self.augment_with_similarity(driver, G, case_id)
                        
                        if len(G.edges(case_id)) < self.MIN_CONNECTIONS:
                            logger.warning(f"Insufficient connections ({len(G.edges(case_id))})")
                            return False

                    # Step 3: Generate embedding
                    embedding = self.generate_embedding(G, case_id)
                    if not embedding:
                        return False
                        
                    # Step 4: Store embedding
                    if not self.store_embedding(driver, upload_id, embedding):
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
        generator = EmbeddingGenerator()
        success = generator.generate_embedding_for_case(upload_id)
        sys.exit(0 if success else 1)
        
    except Exception as e:
        logger.critical(f"Fatal error: {str(e)}")
        sys.exit(1)