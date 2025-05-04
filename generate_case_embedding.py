import os
from dotenv import load_dotenv
from neo4j import GraphDatabase
import networkx as nx
from node2vec import Node2Vec
import numpy as np

# === Load environment variables ===
load_dotenv()
NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USER = os.getenv("NEO4J_USER")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

# === Connect to Neo4j ===
driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

def generate_embedding_for_case(upload_id):
    G = nx.Graph()

    with driver.session() as session:
        # Step 1: Get the case node
        result = session.run("""
            MATCH (c:Case {upload_id: $upload_id})
            RETURN c.id AS id
        """, upload_id=upload_id)

        record = result.single()
        if not record:
            print(f"❌ Case with upload_id {upload_id} not found.")
            return

        node_id = str(record["id"])
        G.add_node(node_id)

        # Step 2: Get its neighbors
        neighbors = session.run("""
            MATCH (c:Case {upload_id: $upload_id})-[r]->(n)
            RETURN c.id AS source, n.id AS target
        """, upload_id=upload_id)

        for r in neighbors:
            if r["source"] and r["target"]:
                G.add_edge(str(r["source"]), str(r["target"]))

        if len(G.nodes) < 2:
            print("⚠️ Not enough structure to build a subgraph for Node2Vec.")
            return

        print(f"⏳ Training Node2Vec for case {upload_id} ({len(G.nodes)} nodes)...")

        node2vec = Node2Vec(G, dimensions=64, walk_length=10, num_walks=50, workers=1, seed=42)
        model = node2vec.fit(window=5, min_count=1)

        vec = model.wv[node_id].tolist()

        # Step 3: Store the new embedding
        session.run("""
            MATCH (c:Case {upload_id: $upload_id})
            SET c.embedding = $embedding
        """, upload_id=upload_id, embedding=vec)

        print(f"✅ Embedding stored for case {upload_id}.")

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python generate_case_embedding.py <upload_id>")
        exit(1)

    upload_id = sys.argv[1]
    generate_embedding_for_case(upload_id)
    driver.close()
