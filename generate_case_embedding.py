import numpy as np
import networkx as nx
from node2vec import Node2Vec

def generate_embedding_for_case(driver, upload_id):
    try:
        with driver.session() as session:
            # Step 1: Fetch all relationships for the given case
            result = session.run(
                """
                MATCH (c:Case {upload_id: $upload_id})-[r]-(n)
                RETURN id(c) AS case_id, id(n) AS neighbor_id, type(r) AS rel_type, r.value AS value
                """,
                upload_id=upload_id
            )

            edges = []
            nodes = set()
            case_node_id = None

            for record in result:
                case_node_id = record["case_id"]
                neighbor_id = record["neighbor_id"]
                rel_type = record["rel_type"]

                try:
                    value = float(record["value"])
                    if np.isnan(value) or not np.isfinite(value):
                        value = 1.0
                except:
                    value = 1.0

                nodes.update([case_node_id, neighbor_id])
                edges.append((case_node_id, neighbor_id, {"weight": value}))

            if not edges or len(nodes) < 2:
                print("⚠️ Not enough structure to build a subgraph for Node2Vec.")
                return False

            print(f"🔗 Number of connected nodes: {len(nodes)}")

            # Step 2: Build the subgraph
            subgraph = nx.Graph()
            subgraph.add_nodes_from(nodes)
            subgraph.add_edges_from(edges)

            print(f"[DEBUG] Subgraph nodes: {list(subgraph.nodes())}")
            print(f"[DEBUG] Subgraph edges: {list(subgraph.edges(data=True))}")

            # Step 3: Run Node2Vec on the subgraph
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

            # Step 4: Get the embedding for the case node
            vector = model.wv[str(case_node_id)]
            vector = vector.astype(float).tolist()

            print(f"[DEBUG] Embedding for upload_id {upload_id}: {vector[:5]}... (len={len(vector)})")

            # Step 5: Save to Neo4j
            session.run(
                """
                MATCH (c:Case {upload_id: $upload_id})
                SET c.embedding = $embedding
                """,
                upload_id=upload_id,
                embedding=vector
            )

            if np.isnan(vector).any():
                print("❌ Embedding contains NaN values")
                return False

            print("✅ Embedding generated and saved successfully.")
            return True

    except Exception as e:
        print(f"❌ Error generating embedding: {e}")
        return False