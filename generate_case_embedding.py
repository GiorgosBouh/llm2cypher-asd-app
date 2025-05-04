import numpy as np
import networkx as nx
from node2vec import Node2Vec

def generate_embedding_for_case(driver, upload_id):
    try:
        with driver.session() as session:
            result = session.run(
                """
                MATCH (c:Case {upload_id: $upload_id})-[r]-(n)
                RETURN id(c) AS case_id, id(n) AS neighbor_id, r.value AS value
                """,
                upload_id=upload_id
            )

            edges = []
            nodes = set()
            case_node_id = None

            for record in result:
                case_node_id = record["case_id"]
                neighbor_id = record["neighbor_id"]
                raw_val = record["value"]

                try:
                    weight = float(raw_val) if raw_val is not None else 1.0
                    if not np.isfinite(weight):
                        continue  # skip invalid weights
                except:
                    continue  # skip unconvertible values

                nodes.update([case_node_id, neighbor_id])
                edges.append((case_node_id, neighbor_id, {"weight": weight}))

            if not edges or len(nodes) < 2:
                print("⚠️ Not enough structure to build a subgraph for Node2Vec.")
                return False

            subgraph = nx.Graph()
            subgraph.add_nodes_from(nodes)
            subgraph.add_edges_from(edges)

            print(f"✅ Subgraph for upload_id {upload_id}: {len(subgraph.nodes())} nodes, {len(subgraph.edges())} edges")

            node2vec = Node2Vec(
                subgraph,
                dimensions=64,
                walk_length=10,
                num_walks=50,
                workers=1,
                seed=42,
                quiet=True
            )
            model = node2vec.fit(window=5, min_count=1, batch_words=4)

            vector = model.wv[str(case_node_id)]
            vector = vector.astype(float).tolist()

            session.run(
                """
                MATCH (c:Case {upload_id: $upload_id})
                SET c.embedding = $embedding
                """,
                upload_id=upload_id,
                embedding=vector
            )

            if not np.all(np.isfinite(vector)):
                print("❌ Embedding contains non-finite values.")
                return False

            print("✅ Embedding generated and saved.")
            return True

    except Exception as e:
        print(f"❌ Error generating embedding: {e}")
        return False