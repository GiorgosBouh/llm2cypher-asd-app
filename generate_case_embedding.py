from node2vec import Node2Vec
import networkx as nx
import numpy as np

def generate_embedding_for_case(driver, upload_id: str):
    print(f"[INFO] Generating embedding for case: {upload_id}")

    G = nx.Graph()

    with driver.session() as session:
        # 1. Ανάκτησε τον κόμβο και τους γείτονες
        result = session.run("""
            MATCH (c:Case {upload_id: $upload_id})-[r]-(n)
            RETURN c.id AS case_id, id(n) AS neighbor_id
        """, upload_id=upload_id)

        edges = []
        case_id = None

        for record in result:
            case_id = str(record["case_id"])
            neighbor_id = str(record["neighbor_id"])
            edges.append((case_id, neighbor_id))

        if not edges:
            print("\u26a0\ufe0f Not enough structure to build a subgraph for Node2Vec.")
            return False

        # 2. Δημιούργησε γράφο
        for edge in edges:
            G.add_node(edge[0])
            G.add_node(edge[1])
            G.add_edge(edge[0], edge[1])

    print(f"[DEBUG] Subgraph Nodes: {list(G.nodes)}")
    print(f"[DEBUG] Subgraph Edges: {list(G.edges)}")

    if len(G.nodes) < 2:
        print("\u26a0\ufe0f Subgraph too small for Node2Vec walk.")
        return False

    # 3. Εκπαίδευσε Node2Vec στο subgraph
    node2vec = Node2Vec(
        G, dimensions=64, walk_length=10, num_walks=20, workers=1, seed=42
    )
    model = node2vec.fit(window=5, min_count=1)

    # 4. Απόθηκευσε το embedding στο γράφο
    vec = model.wv[case_id].tolist()
    with driver.session() as session:
        session.run("""
            MATCH (c:Case {upload_id: $upload_id})
            SET c.embedding = $embedding
        """, upload_id=upload_id, embedding=vec)

    print("\u2705 Local embedding saved successfully!")
    return True
