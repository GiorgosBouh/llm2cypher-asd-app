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
from sklearn.preprocessing import normalize

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class EmbeddingGenerator:
    def __init__(self):
        # Αυτά πρέπει να ταιριάζουν με το main script
        self.EMBEDDING_DIM = 256  # Τώρα ταιριάζει με το main script
        self.MIN_CONNECTIONS = 5  # Αυξημένο από 3
        self.MAX_RETRIES = 3
        self.MIN_SIMILARITY = 5
        self.EMBEDDING_NORMALIZATION = True
        self.WALK_LENGTH = 30  # Αυξημένο από 20
        self.NUM_WALKS = 200  # Αυξημένο από 100

    def get_driver(self):
        """Enhanced driver configuration"""
        return GraphDatabase.driver(
            os.getenv("NEO4J_URI"),
            auth=(os.getenv("NEO4J_USER"), os.getenv("NEO4J_PASSWORD")),
            connection_timeout=45,
            max_connection_lifetime=7200,
            max_connection_pool_size=100,
            max_retry_time=15
        )

    def validate_embedding(self, embedding: List[float]) -> bool:
        """Enhanced validation with statistical checks"""
        if not embedding or len(embedding) != self.EMBEDDING_DIM:
            return False
            
        if any(np.isnan(x) or np.isinf(x) for x in embedding):
            logger.warning("Embedding contains invalid values")
            return False
            
        if np.std(embedding) < 0.01:  # Check for variance
            logger.warning("Embedding lacks variance")
            return False
            
        return True

    def build_enriched_graph(self, driver, upload_id: str) -> Optional[tuple]:
        """Enhanced graph construction with more relationships"""
        with driver.session() as session:
            # Get case and extended relationships
            result = session.run("""
                MATCH (c:Case {upload_id: $upload_id})
                OPTIONAL MATCH (c)-[:HAS_ANSWER]->(q:BehaviorQuestion)
                OPTIONAL MATCH (c)-[:HAS_DEMOGRAPHIC]->(d:DemographicAttribute)
                OPTIONAL MATCH (c)-[:SUBMITTED_BY]->(s:SubmitterType)
                OPTIONAL MATCH (c)-[:SIMILAR_TO]-(similar:Case)
                RETURN c.id AS case_id, 
                       collect(DISTINCT q.name) AS questions,
                       collect(DISTINCT d.value) AS demographics,
                       collect(DISTINCT s.type) AS submitters,
                       collect(DISTINCT similar.id) AS similar_cases
            """, upload_id=upload_id).single()

            if not result or not result["case_id"]:
                logger.error("Case not found or missing ID")
                return None

            case_id = str(result["case_id"])
            G = nx.Graph()
            G.add_node(case_id, 
                      questions=result["questions"],
                      demographics=result["demographics"],
                      submitters=result["submitters"])
            
            # Add relationships with weights
            for neighbor in result["similar_cases"]:
                if neighbor:
                    G.add_edge(case_id, str(neighbor), weight=1.0)
            
            return G, case_id

    def augment_graph(self, driver, G: nx.Graph, case_id: str) -> None:
        """Enhanced graph augmentation"""
        with driver.session() as session:
            # Add similar cases based on multiple criteria
            similar = session.run("""
                MATCH (c:Case {id: $case_id})
                OPTIONAL MATCH (c)-[:HAS_ANSWER]->(q)<-[:HAS_ANSWER]-(similar:Case)
                WITH similar, count(q) AS answer_similarity
                WHERE answer_similarity >= $min_similarity
                
                OPTIONAL MATCH (c)-[:HAS_DEMOGRAPHIC]->(d)<-[:HAS_DEMOGRAPHIC]-(similar)
                WITH similar, answer_similarity, count(d) AS demo_similarity
                
                RETURN similar.id, 
                       (answer_similarity * 0.6 + demo_similarity * 0.4) AS combined_score
                ORDER BY combined_score DESC 
                LIMIT 25
            """, case_id=int(case_id), min_similarity=self.MIN_SIMILARITY)

            for record in similar:
                G.add_edge(case_id, str(record["similar.id"]), 
                          weight=float(record["combined_score"]))

    def generate_embedding(self, G: nx.Graph, case_id: str) -> Optional[List[float]]:
        """Enhanced embedding generation"""
        try:
            temp_dir = tempfile.mkdtemp()
            
            # Enhanced Node2Vec parameters
            node2vec = Node2Vec(
                G,
                dimensions=self.EMBEDDING_DIM,
                walk_length=self.WALK_LENGTH,
                num_walks=self.NUM_WALKS,
                workers=4,  # Increased parallelism
                weight_key='weight',  # Use edge weights
                p=0.5,  # Adjusted parameters
                q=2.0,
                quiet=False,
                temp_folder=temp_dir
            )
            
            model = node2vec.fit(
                window=10,  # Larger context window
                min_count=1,
                epochs=50,  # More training iterations
                batch_words=2000
            )
            
            embedding = model.wv[case_id].tolist()
            
            # Enhanced normalization
            if self.EMBEDDING_NORMALIZATION:
                embedding = normalize([embedding], norm='l2')[0].tolist()
            
            # Clean up
            try:
                shutil.rmtree(temp_dir)
            except Exception as e:
                logger.warning(f"Temp cleanup failed: {str(e)}")
            
            if not self.validate_embedding(embedding):
                logger.error("Invalid embedding generated")
                return None
                
            return embedding
            
        except Exception as e:
            logger.error(f"Embedding generation failed: {str(e)}", exc_info=True)
            return None

    def store_embedding(self, driver, upload_id: str, embedding: List[float]) -> bool:
        """Enhanced embedding storage"""
        try:
            with driver.session() as session:
                result = session.run("""
                    MATCH (c:Case {upload_id: $upload_id})
                    SET c.embedding = $embedding,
                        c.embedding_version = '3.0',
                        c.last_embedding_update = datetime(),
                        c.embedding_stats = {
                            min: apoc.coll.min($embedding),
                            max: apoc.coll.max($embedding),
                            mean: apoc.coll.avg($embedding),
                            std: apoc.coll.stdev($embedding)
                        }
                    RETURN c.id
                """, upload_id=upload_id, embedding=embedding).single()
                
                return bool(result)
        except Exception as e:
            logger.error(f"Storage failed: {str(e)}")
            return False

    def generate_embedding_for_case(self, upload_id: str) -> bool:
        """Robust generation workflow"""
        driver = None
        try:
            driver = self.get_driver()
            
            for attempt in range(self.MAX_RETRIES):
                try:
                    # Enhanced graph construction
                    graph_result = self.build_enriched_graph(driver, upload_id)
                    if not graph_result:
                        return False
                        
                    G, case_id = graph_result
                    
                    # Smart augmentation
                    if len(G.edges(case_id)) < self.MIN_CONNECTIONS:
                        self.augment_graph(driver, G, case_id)
                        
                        if len(G.edges(case_id)) < self.MIN_CONNECTIONS:
                            logger.warning(f"Insufficient connections after augmentation")
                            return False

                    # Generate and validate embedding
                    embedding = self.generate_embedding(G, case_id)
                    if not embedding:
                        return False
                        
                    # Store with enhanced metadata
                    if not self.store_embedding(driver, upload_id, embedding):
                        return False
                        
                    logger.info(f"Successfully generated enhanced embedding for case {case_id}")
                    return True

                except Exception as e:
                    if attempt == self.MAX_RETRIES - 1:
                        raise
                    wait_time = 2 ** attempt
                    logger.warning(f"Attempt {attempt + 1} failed, retrying in {wait_time}s...")
                    time.sleep(wait_time)

        except Exception as e:
            logger.error(f"Embedding generation failed: {str(e)}", exc_info=True)
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
        logger.critical(f"Fatal error: {str(e)}", exc_info=True)
        sys.exit(1)