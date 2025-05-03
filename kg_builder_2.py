import pandas as pd
import numpy as np
from neo4j import GraphDatabase
import networkx as nx
from node2vec import Node2Vec
from random import shuffle

# --- Neo4j Aura σύνδεση ---
from dotenv import load_dotenv
load_dotenv()

def connect_to_neo4j():
    import os
    uri = os.getenv("NEO4J_URI")
    user = os.getenv("NEO4J_USER")
    password = os.getenv("NEO4J_PASSWORD")

    if not all([uri, user, password]):
        raise ValueError("❌ Missing Neo4j credentials in environment")

    print(f"🌐 Connecting to Neo4j Aura: {uri}", flush=True)
    return GraphDatabase.driver(uri, auth=(user, password))

# --- Καθαρισμός CSV ---
def parse_csv(file_path):
    df = pd.read_csv(file_path, sep=";", encoding="utf-8-sig")
    df.columns = [col.strip() for col in df.columns]
    df = df.apply(lambda col: col.str.strip() if col.dtypes == 'object' else col)

    numeric_cols = ['Case_No', 'A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'A8', 'A9', 'A10', 'Age_Mons']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col].astype(str).str.replace(",", "."), errors='coerce')

    print("\u2705 Καθαρίστηκαν οι στήλες:", df.columns.tolist())
    return df.dropna()

# --- Δημιουργία κόμβων ---
def create_nodes(tx, df):
    for q in [f"A{i}" for i in range(1, 11)]:
        tx.run("MERGE (:BehaviorQuestion {name: $q})", q=q)

    for column in ["Sex", "Ethnicity", "Jaundice", "Family_mem_with_ASD"]:
        for val in df[column].dropna().unique():
            tx.run("MERGE (:DemographicAttribute {type: $type, value: $val})", type=column, val=val)

    for val in df["Who_completed_the_test"].dropna().unique():
        tx.run("MERGE (:SubmitterType {type: $val})", val=val)

# --- Δημιουργία σχέσεων ---
def create_relationships(tx, df):
    case_data = [{"id": int(row["Case_No"])} for _, row in df.iterrows()]
    answer_data, demo_data, submitter_data = [], [], []

    for _, row in df.iterrows():
        case_id = int(row["Case_No"])
        for q in [f"A{i}" for i in range(1, 11)]:
            answer_data.append({"case_id": case_id, "q": q, "val": int(row[q])})

        for col in ["Sex", "Ethnicity", "Jaundice", "Family_mem_with_ASD"]:
            demo_data.append({"case_id": case_id, "type": col, "val": row[col]})

        submitter_data.append({"case_id": case_id, "val": row["Who_completed_the_test"]})

    tx.run("""
    UNWIND $data as row
    MERGE (c:Case {id: row.id})
    SET c.embedding = null
    """, data=case_data)

    tx.run("""
    UNWIND $data as row
    MATCH (c:Case {id: row.case_id}), (b:BehaviorQuestion {name: row.q})
    MERGE (c)-[:HAS_ANSWER {value: row.val}]->(b)
    """, data=answer_data)

    tx.run("""
    UNWIND $data as row
    MATCH (c:Case {id: row.case_id}), (d:DemographicAttribute {type: row.type, value: row.val})
    MERGE (c)-[:HAS_DEMOGRAPHIC]->(d)
    """, data=demo_data)

    tx.run("""
    UNWIND $data as row
    MATCH (c:Case {id: row.case_id}), (s:SubmitterType {type: row.val})
    MERGE (c)-[:SUBMITTED_BY]->(s)
    """, data=submitter_data)

# --- Ελαφριές σχέσεις ομοιότητας ---
def create_similarity_relationships(tx, df, max_pairs=3000):
    pairs = []
    for col in ["Ethnicity", "Who_completed_the_test"]:
        grouped = df.groupby(col)["Case_No"].apply(list)
        for ids in grouped:
            for i in range(len(ids)):
                for j in range(i+1, len(ids)):
                    pairs.append((int(ids[i]), int(ids[j])))

    for i, row1 in df.iterrows():
        for j, row2 in df.iloc[i + 1:].iterrows():
            if abs(row1["Qchat-10-Score"] - row2["Qchat-10-Score"]) <= 1:
                pairs.append((int(row1["Case_No"]), int(row2["Case_No"])))

    shuffle(pairs)
    pairs = pairs[:max_pairs]

    tx.run("""
    UNWIND $batch AS pair
    MATCH (c1:Case {id: pair.id1}), (c2:Case {id: pair.id2})
    MERGE (c1)-[:GRAPH_SIMILARITY]->(c2)
    """, batch=[{"id1": i, "id2": j} for i, j in pairs])

# --- Embeddings ---
def generate_embeddings(driver):
    G = nx.Graph()
    with driver.session() as session:
        for r in session.run("MATCH (c:Case) RETURN c.id AS id"):
            G.add_node(str(r["id"]))

        rels = session.run("""
        MATCH (c1:Case)-[r:HAS_ANSWER|HAS_DEMOGRAPHIC|SCREENED_FOR|SUBMITTED_BY|GRAPH_SIMILARITY]->(c2)
        RETURN c1.id AS source, labels(c2)[0] AS target_label, coalesce(c2.id, elementId(c2)) AS target_id
        """)
        for r in rels:
            source = str(r["source"])
            target = f"{r['target_label']}::{r['target_id']}"
            G.add_edge(source, target)

    if len(G.nodes) == 0:
        print("⚠️ Δεν βρέθηκαν κόμβοι!")
        return

    print("⏳ Εκπαίδευση Node2Vec...")
    node2vec = Node2Vec(G, dimensions=64, walk_length=10, num_walks=50, workers=1, seed=42)
    model = node2vec.fit(window=5, min_count=1)

    with driver.session() as session:
        for node_id in G.nodes():
            vec = model.wv[str(node_id)].tolist()
            session.run("MATCH (c:Case {id: toInteger($id)}) SET c.embedding = $embedding",
                        id=node_id, embedding=vec)
    print("✅ Embeddings αποθηκεύτηκαν!")

# --- Run all ---
def build_graph():
    driver = connect_to_neo4j()
    file_path = "https://raw.githubusercontent.com/GiorgosBouh/llm2cypher-asd-app/main/Toddler_Autism_dataset_July_2018_2.csv"

    try:
        df = parse_csv(file_path)
        print("🧠 First row:", df.iloc[0].to_dict())

        with driver.session() as session:
            print("⏳ Δημιουργία κόμβων...", flush=True)
            session.execute_write(create_nodes, df)
            print("⏳ Δημιουργία σχέσεων...")
            session.execute_write(create_relationships, df)
            print("⏳ Δημιουργία σχέσεων ομοιότητας...")
            session.execute_write(create_similarity_relationships, df)

        print("⏳ Δημιουργία embeddings...")
        generate_embeddings(driver)

        print("✅ Ολοκληρώθηκε επιτυχώς!")

    except Exception as e:
        print(f"❌ Σφάλμα: {str(e)}", flush=True)
        import sys
        sys.exit(1)

if __name__ == "__main__":
    try:
        build_graph()
        import sys
        sys.exit(0)
    except Exception as e:
        print(f"❌ Σφάλμα: {str(e)}", flush=True)
        import sys
        sys.exit(1)