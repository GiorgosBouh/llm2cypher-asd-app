import numpy as np
import networkx as nx
from node2vec import Node2Vec
from neo4j import GraphDatabase
import sys

def generate_embedding_for_case(driver, upload_id):
    try:
        # 1. Φόρτωση ολόκληρου του γράφου
        G = nx.Graph()
        
        with driver.session() as session:
            # Προσθήκη όλων των nodes
            nodes = session.run("MATCH (n) RETURN id(n) AS node_id")
            for node in nodes:
                G.add_node(str(node["node_id"]))
            
            # Προσθήκη όλων των edges
            edges = session.run("""
                MATCH (n1)-[r]->(n2)
                RETURN id(n1) AS source, id(n2) AS target, 
                       CASE WHEN r.value IS NOT NULL THEN toFloat(r.value) ELSE 1.0 END AS weight
            """)
            for edge in edges:
                G.add_edge(str(edge["source"]), str(edge["target"]), weight=edge["weight"])
            
            # 2. Βρείτε το node_id του νέου case
            case_node = session.run(
                "MATCH (c:Case {upload_id: $upload_id}) RETURN id(c) AS case_id",
                upload_id=upload_id
            ).single()
            
            if not case_node:
                print("❌ Case not found in graph")
                return False
                
            case_node_id = str(case_node["case_id"])
            
        # 3. Δημιουργία embeddings για ολόκληρο το γράφο
        print(f"⏳ Training Node2Vec on graph with {len(G.nodes())} nodes and {len(G.edges())} edges")
        node2vec = Node2Vec(
            G,
            dimensions=64,
            walk_length=10,
            num_walks=50,
            workers=1,
            seed=42
        )
        
        model = node2vec.fit(window=5, min_count=1)
        
        # 4. Αποθήκευση embedding για το συγκεκριμένο case
        vector = model.wv[case_node_id].astype(float).tolist()
        
        with driver.session() as session:
            session.run(
                "MATCH (c:Case {upload_id: $upload_id}) SET c.embedding = $embedding",
                upload_id=upload_id,
                embedding=vector
            )
        
        print("✅ Embedding generated and saved")
        return True
        
    except Exception as e:
        print(f"❌ Error generating embedding: {str(e)}")
        return False

if __name__ == "__main__":
    from neo4j import GraphDatabase
    import os
    
    # Παράμετροι σύνδεσης
    uri = os.getenv("NEO4J_URI")
    user = os.getenv("NEO4J_USER")
    password = os.getenv("NEO4J_PASSWORD")
    upload_id = sys.argv[1] if len(sys.argv) > 1 else None
    
    if not upload_id:
        print("❌ Missing upload_id parameter")
        sys.exit(1)
        
    driver = GraphDatabase.driver(uri, auth=(user, password))
    success = generate_embedding_for_case(driver, upload_id)
    driver.close()
    
    sys.exit(0 if success else 1)