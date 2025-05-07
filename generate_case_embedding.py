import numpy as np
import networkx as nx
from node2vec import Node2Vec
from neo4j import GraphDatabase
import sys
import os

def generate_embedding_for_case(driver, upload_id):
    try:
        G = nx.Graph()

        with driver.session() as session:
            # 🔎 Ανάκτηση του upload_id-based Case
            result = session.run(
                "MATCH (c:Case {upload_id: $upload_id}) RETURN c.id AS case_id",
                upload_id=upload_id
            ).single()

            if not result or result["case_id"] is None:
                print("❌ Case not found in graph.")
                return False

            case_id = result["case_id"]

            # 🔄 Φόρτωση κόμβων
            nodes = session.run("MATCH (n) RETURN n.id AS id")
            for node in nodes:
                if node["id"] is not None:
                    G.add_node(str(node["id"]))

            # 🔗 Φόρτωση σχέσεων με έλεγχο βάρους
            edges = session.run("""
                MATCH (n1)-[r]->(n2)
                WHERE n1.id IS NOT NULL AND n2.id IS NOT NULL
                RETURN n1.id AS source, n2.id AS target,
                       CASE WHEN r.value IS NOT NULL THEN toFloat(r.value) ELSE 1.0 END AS weight
            """)

            for edge in edges:
                weight = edge["weight"]
                if weight is None or not np.isfinite(weight):
                    continue
                G.add_edge(str(edge["source"]), str(edge["target"]), weight=weight)

        # ⚠️ Έλεγχος ελάχιστων κόμβων
        if len(G.nodes) < 2:
            print("⚠️ Not enough nodes to build graph.")
            return False

        print(f"✅ Graph built: {len(G.nodes)} nodes, {len(G.edges)} edges")

        # 🧠 Εκπαίδευση Node2Vec
        node2vec = Node2Vec(
            G,
            dimensions=128,
            walk_length=20,
            num_walks=100,
            workers=2,
            seed=42
        )
        model = node2vec.fit(window=5, min_count=1)

        # ✅ Ανάκτηση embedding
        if str(case_id) not in model.wv:
            print(f"❌ Node {case_id} not found in embedding space.")
            return False

        vector = model.wv[str(case_id)]
        if not np.all(np.isfinite(vector)):
            print("❌ Embedding contains non-finite values.")
            return False

        embedding = vector.tolist()

        # 💾 Αποθήκευση στο Neo4j
        with driver.session() as session:
            session.run(
                "MATCH (c:Case {upload_id: $upload_id}) SET c.embedding = $embedding",
                upload_id=upload_id,
                embedding=embedding
            )

        print("✅ Embedding generated and stored successfully.")
        return True

    except Exception as e:
        print(f"❌ Error during embedding generation: {e}")
        return False

if __name__ == "__main__":
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
