import numpy as np
import networkx as nx
from node2vec import Node2Vec
from neo4j import GraphDatabase
import sys
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor

@lru_cache(maxsize=1)
def get_graph(driver):
    """Cache του γράφου με LRU policy"""
    G = nx.Graph()
    with driver.session() as session:
        # Χρήση batch loading για nodes
        nodes = session.run("MATCH (n) RETURN id(n) AS node_id", batch_size=1000)
        G.add_nodes_from(str(node["node_id"]) for node in nodes)
        
        # Χρήση batch loading για edges
        edges = session.run("""
            MATCH (n1)-[r]->(n2)
            RETURN id(n1) AS source, id(n2) AS target,
                   COALESCE(toFloat(r.value), 1.0) AS weight
            """, batch_size=1000)
        G.add_weighted_edges_from(
            (str(edge["source"]), str(edge["target"]), edge["weight"])
            for edge in edges
        )
    return G

def parallel_walks(node2vec):
    """Παράλληλη εκτέλεση random walks"""
    with ThreadPoolExecutor(max_workers=4) as executor:
        return list(executor.map(node2vec.walk, node2vec.nodes))

def generate_embedding_for_case(driver, upload_id):
    try:
        # 1. Έλεγχος για προ-υπολογισμένα embeddings
        with driver.session() as session:
            result = session.run("""
                MATCH (c:Case {upload_id: $upload_id})
                OPTIONAL MATCH (c)-[r:SIMILAR_TO]->(other)
                WHERE other.embedding IS NOT NULL AND r.score > 0.85
                RETURN id(c) AS case_id, other.embedding AS embedding
                ORDER BY r.score DESC LIMIT 1
            """, upload_id=upload_id)
            
            record = result.single()
            if record and record["embedding"]:
                session.run("""
                    MATCH (c:Case {upload_id: $upload_id})
                    SET c.embedding = $embedding
                """, upload_id=upload_id, embedding=record["embedding"])
                return True

        # 2. Φόρτωση γράφου (με cache)
        G = get_graph(driver)
        case_id = str(record["case_id"])

        # 3. Βελτιστοποιημένο Node2Vec
        node2vec = Node2Vec(
            G,
            dimensions=32,
            walk_length=15,
            num_walks=20,
            workers=4,
            p=0.5,
            q=1.0,
            quiet=True
        )
        
        # 4. Παράλληλη εκτέλεση walks
        walks = parallel_walks(node2vec)
        model = node2vec.fit(
            walks=walks,  # Χρήση pre-computed walks
            window=3,
            min_count=1,
            batch_words=4
        )

        # 5. Αποθήκευση embedding
        embedding = model.wv[case_id].tolist()
        
        with driver.session() as session:
            session.run("""
                MATCH (c:Case {upload_id: $upload_id})
                SET c.embedding = $embedding
            """, upload_id=upload_id, embedding=embedding)
            
            # 6. Αποθήκευση σχέσεων ομοιότητας
            session.run("""
                MATCH (c1:Case {upload_id: $upload_id})
                MATCH (c2:Case)
                WHERE c2.embedding IS NOT NULL AND c1 <> c2
                WITH c1, c2, 
                     gds.similarity.cosine(c1.embedding, c2.embedding) AS similarity
                WHERE similarity > 0.8
                MERGE (c1)-[:SIMILAR_TO {score: similarity}]->(c2)
            """, upload_id=upload_id)
        
        return True

    except Exception as e:
        print(f"❌ Error: {str(e)}", file=sys.stderr)
        return False

if __name__ == "__main__":
    from neo4j import GraphDatabase
    import os
    
    if len(sys.argv) < 2:
        print("❌ Usage: python generate_case_embedding.py <upload_id>")
        sys.exit(1)
        
    upload_id = sys.argv[1]
    
    driver = GraphDatabase.driver(
        os.getenv("NEO4J_URI"),
        auth=(os.getenv("NEO4J_USER"), os.getenv("NEO4J_PASSWORD"))
    )
    
    try:
        success = generate_embedding_for_case(driver, upload_id)
        sys.exit(0 if success else 1)
    finally:
        driver.close()