from dotenv import load_dotenv
load_dotenv()
import numpy as np
import networkx as nx
from node2vec import Node2Vec
from neo4j import GraphDatabase
import sys
import os
from tqdm import tqdm  # Για progress bars

def generate_embedding_for_case(driver, upload_id):
    try:
        G = nx.Graph()
        case_id = None

        # Βελτιστοποίηση: Single query για φόρτωση όλων των απαραίτητων δεδομένων
        with driver.session() as session:
            result = session.run("""
                MATCH (c:Case {upload_id: $upload_id}) 
                OPTIONAL MATCH (c)-[r]->(neighbor)
                WHERE c.id IS NOT NULL AND neighbor.id IS NOT NULL
                RETURN c.id AS case_id, 
                       collect(DISTINCT neighbor.id) AS neighbors,
                       count(r) AS degree
            """, upload_id=upload_id).single()

            if not result or not result["case_id"]:
                print("❌ Case not found or has no ID")
                return False

            case_id = str(result["case_id"])
            degree = result["degree"]
            
            if degree < 3:  # Ελάχιστο όριο συνδέσεων
                print(f"⚠️ Case has too few connections ({degree}) for meaningful embedding")
                return False

            # Φόρτωση ολόκληρου του υπογραφήματος
            graph_result = session.run("""
                MATCH (c:Case {id: $case_id})-[:GRAPH_SIMILARITY*..2]-(other)
                WHERE other.id IS NOT NULL
                RETURN DISTINCT other.id AS node_id
                LIMIT 500  # Όριο για απόδοση
            """, case_id=int(case_id))

            G.add_node(case_id)
            for record in graph_result:
                G.add_edge(case_id, str(record["node_id"]))

        # Επαύξηση με κοινά γειτονικά nodes
        if len(G.edges) < 10:
            print("⚠️ Augmenting with behavioral similarities")
            with driver.session() as session:
                similar_cases = session.run("""
                    MATCH (c1:Case {id: $case_id})
                    MATCH (c2:Case)
                    WHERE c1 <> c2 AND
                    size([q IN ['A1','A2','A3','A4','A5','A6','A7','A8','A9','A10'] 
                         WHERE (c1)-[:HAS_ANSWER {value: (c2)-[:HAS_ANSWER {name: q}]->value}]->q]) >= 7
                    RETURN c2.id LIMIT 20
                """, case_id=int(case_id))

                for record in similar_cases:
                    G.add_edge(case_id, str(record["c2.id"]))

        # Δημιουργία embeddings με τις ίδιες παραμέτρους όπως στο κύριο γράφημα
        node2vec = Node2Vec(
            G,
            dimensions=128,
            walk_length=30,
            num_walks=200,
            workers=2,
            p=1.0,
            q=0.5,
            quiet=True  # Απενεργοποίηση verbose output
        )

        model = node2vec.fit(
            window=10,
            min_count=1
        )

        embedding = model.wv[case_id].tolist()
        
        # Ενημέρωση Neo4j
        with driver.session() as session:
            session.run("""
                MATCH (c:Case {upload_id: $upload_id})
                SET c.embedding = $embedding,
                    c.embedding_generated_at = datetime()
            """, upload_id=upload_id, embedding=embedding)

        print(f"✅ Generated embedding for case {case_id} (Dimension: {len(embedding)})")
        return True

    except Exception as e:
        print(f"❌ Error during embedding generation: {str(e)}")
        return False

if __name__ == "__main__":
    # Βελτιστοποιημένο error handling
    try:
        uri = os.getenv("NEO4J_URI")
        user = os.getenv("NEO4J_USER")
        password = os.getenv("NEO4J_PASSWORD")
        upload_id = sys.argv[1] if len(sys.argv) > 1 else None

        if not upload_id:
            raise ValueError("Missing upload_id parameter")

        driver = GraphDatabase.driver(uri, auth=(user, password))
        success = generate_embedding_for_case(driver, upload_id)
        sys.exit(0 if success else 1)
        
    except Exception as e:
        print(f"❌ Fatal error: {str(e)}")
        sys.exit(1)
