import numpy as np
import networkx as nx
from node2vec import Node2Vec
from neo4j import GraphDatabase
import sys
import os
import time
from tqdm import tqdm
import logging

# Ρύθμιση logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def generate_embedding_for_case(upload_id: str) -> bool:
    # Ρύθμιση παραμέτρων
    EMBEDDING_DIM = 128
    MIN_CONNECTIONS = 3
    MAX_RETRIES = 3
    
    try:
        # Αρχικοποίηση Neo4j driver με ανθεκτικότητα
        driver = GraphDatabase.driver(
            os.getenv("NEO4J_URI"),
            auth=(os.getenv("NEO4J_USER"), os.getenv("NEO4J_PASSWORD")),
            connection_timeout=30,
            max_connection_lifetime=7200
        )
        
        for attempt in range(MAX_RETRIES):
            try:
                G = nx.Graph()
                case_id = None

                # Βήμα 1: Φόρτωση βασικού γραφήματος
                with driver.session() as session:
                    # Ανακτήστε το case_id και βασικές συνδέσεις
                    result = session.run("""
                        MATCH (c:Case {upload_id: $upload_id})
                        OPTIONAL MATCH (c)-[r]->(neighbor)
                        RETURN c.id AS case_id, 
                               collect(DISTINCT neighbor.id) AS neighbors
                    """, upload_id=upload_id).single()

                    if not result or not result["case_id"]:
                        logger.error("Case not found or missing ID")
                        return False

                    case_id = str(result["case_id"])
                    G.add_node(case_id)
                    
                    # Προσθήκη αρχικών συνδέσεων
                    for neighbor in result["neighbors"]:
                        if neighbor:
                            G.add_edge(case_id, str(neighbor))

                # Βήμα 2: Επαύξηση με similarity relationships
                if len(G.edges(case_id)) < MIN_CONNECTIONS:
                    with driver.session() as session:
                        similar = session.run("""
                            MATCH (c:Case {id: $case_id})
                            MATCH (similar:Case)
                            WHERE c <> similar AND
                            size([(c)-[:HAS_ANSWER]->(q)<-[:HAS_ANSWER]-(similar) | q]) >= 5
                            RETURN similar.id LIMIT 20
                        """, case_id=int(case_id))

                        for record in similar:
                            G.add_edge(case_id, str(record["similar.id"]))

                # Έλεγχος επάρκειας συνδέσεων
                if len(G.edges(case_id)) < MIN_CONNECTIONS:
                    logger.warning(f"Insufficient connections ({len(G.edges(case_id))})")
                    return False

                # Βήμα 3: Δημιουργία embeddings
                node2vec = Node2Vec(
                    G,
                    dimensions=EMBEDDING_DIM,
                    walk_length=20,
                    num_walks=100,
                    workers=2,
                    quiet=True
                )
                
                model = node2vec.fit(window=5, min_count=1)
                embedding = model.wv[case_id].tolist()

                # Βήμα 4: Αποθήκευση
                with driver.session() as session:
                    session.run("""
                        MATCH (c:Case {upload_id: $upload_id})
                        SET c.embedding = $embedding,
                            c.last_embedding_update = timestamp()
                    """, upload_id=upload_id, embedding=embedding)

                logger.info(f"Successfully generated embedding for case {case_id}")
                return True

            except Exception as e:
                if attempt == MAX_RETRIES - 1:
                    raise
                wait_time = 2 ** attempt
                logger.warning(f"Attempt {attempt + 1} failed, retrying in {wait_time}s...")
                time.sleep(wait_time)

    except Exception as e:
        logger.error(f"Embedding generation failed: {str(e)}")
        return False

    finally:
        if 'driver' in locals():
            driver.close()

if __name__ == "__main__":
    try:
        if len(sys.argv) < 2:
            raise ValueError("Missing upload_id parameter")
        
        upload_id = sys.argv[1]
        success = generate_embedding_for_case(upload_id)
        sys.exit(0 if success else 1)
        
    except Exception as e:
        logger.critical(f"Fatal error: {str(e)}")
        sys.exit(1)