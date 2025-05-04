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
            # 🔍 Ανάκτηση internal ID για το συγκεκριμένο case
            result = session.run(
                "MATCH (c:Case {upload_id: $upload_id}) RETURN id(c) AS case_id",
                upload_id=upload_id
            ).single()

            if not result or "case_id" not in result:
                print("❌ Case not found.")
                return False

            case_id = result["case_id"]

            # 🔄 Φόρτωση edges και γειτονικών κόμβων του συγκεκριμένου case
            edge_result = session.run(
                """
                MATCH (c:Case {upload_id: $upload_id})-[r]-(n)
                RETURN id(c) AS source, id(n) AS target,
                       CASE WHEN r.value IS NOT NULL THEN toFloat(r.value) ELSE 1.0 END AS weight
                """,
                upload_id=upload_id
            )

            for record in edge_result:
                src = str(record["source"])
                tgt = str(record["target"])
                weight = record["weight"]

                if not np.isfinite(weight):
                    continue

                G.add_edge(src, tgt, weight=weight)

        # ⚠️ Έλεγχος ελάχιστων κόμβων
        if len(G.nodes) < 2:
            print("⚠️ Not enough connected nodes to build embedding.")
            return False

        print(f"✅ Subgraph: {len(G.nodes)} nodes, {len(G.edges)} edges")

        # 🧠 Εκπαίδευση Node2Vec
        node2vec = Node2Vec(
            G,
            dimensions=64,
            walk_length=10,
            num_walks=50,
            workers=1,
            seed=42
        )
        model = node2vec.fit(window=5, min_count=1)

        # ✅ Ανάκτηση embedding
        case_id_str = str(case_id)
        if case_id_str not in model.wv:
            print(f"❌ Case node {case_id} not found in embedding space.")
            return False

        vector = model.wv[case_id_str]
        if not np.all(np.isfinite(vector)):
            print("❌ Non-finite values in embedding.")
            return False

        embedding = vector.tolist()

        # 💾 Αποθήκευση στο Neo4j
        with driver.session() as session:
            session.run(
                "MATCH (c:Case {upload_id: $upload_id}) SET c.embedding = $embedding",
                upload_id=upload_id,
                embedding=embedding
            )

        print("✅ Embedding saved for case:", upload_id)
        return True

    except Exception as e:
        print(f"❌ Error: {str(e)}")
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