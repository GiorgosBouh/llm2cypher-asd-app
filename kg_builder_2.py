import pandas as pd
import numpy as np
from neo4j import GraphDatabase
import networkx as nx
from node2vec import Node2Vec

# --- Neo4j Aura σύνδεση ---
def connect_to_neo4j(uri="neo4j+s://1f5f8a14.databases.neo4j.io", user="neo4j", password="3xhy4XKQSsSLIT7NI-w9m4Z7Y_WcVnL1hDQkWTMIoMQ"):
    return GraphDatabase.driver(uri, auth=(user, password))

# --- Ανάγνωση και καθαρισμός CSV από GitHub ---
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
        unique_values = df[column].dropna().unique()
        for val in unique_values:
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

# --- Δημιουργία σχέσεων ομοιότητας ---
def create_similarity_relationships(tx, df):
    # Ethnicity
    grouped_ethnicity = df.groupby("Ethnicity")["Case_No"].apply(list)
    for case_list in grouped_ethnicity:
        for i in range(len(case_list)):
            for j in range(i + 1, len(case_list)):
                tx.run("""
                MATCH (c1:Case {id: $id1}), (c2:Case {id: $id2})
                MERGE (c1)-[:SAME_ETHNICITY]->(c2)
                """, id1=int(case_list[i]), id2=int(case_list[j]))

    # Submitter
    grouped_submitter = df.groupby("Who_completed_the_test")["Case_No"].apply(list)
    for case_list in grouped_submitter:
        for i in range(len(case_list)):
            for j in range(i + 1, len(case_list)):
                tx.run("""
                MATCH (c1:Case {id: $id1}), (c2:Case {id: $id2})
                MERGE (c1)-[:SAME_SUBMITTER]->(c2)
                """, id1=int(case_list[i]), id2=int(case_list[j]))

    # QChat similarity (±1)
    similar_qchat = []
    for i, row1 in df.iterrows():
        for j, row2 in df.iloc[i+1:].iterrows():
            if abs(row1["Qchat-10-Score"] - row2["Qchat-10-Score"]) <= 1:
                similar_qchat.append({"id1": int(row1["Case_No"]), "id2": int(row2["Case_No"])} )

    tx.run("""
    UNWIND $similarPairs AS pair
    MATCH (c1:Case {id: pair.id1}), (c2:Case {id: pair.id2})
    MERGE (c1)-[:SIMILAR_QCHAT_SCORE]->(c2)
    """, similarPairs=similar_qchat)

# --- Embeddings με Node2Vec ---
def generate_embeddings(driver):
    G = nx.Graph()

    with driver.session() as session:
        result = session.run("""
            MATCH (c:Case)
            RETURN c.id AS id
        """)
        for record in result:
            G.add_node(str(record["id"]))

        rel_result = session.run("""
            MATCH (c1:Case)-[r:HAS_ANSWER|HAS_DEMOGRAPHIC|SCREENED_FOR|SUBMITTED_BY|SAME_ETHNICITY|SAME_SUBMITTER|SIMILAR_QCHAT_SCORE]->(c2)
            RETURN c1.id AS source, id(c2) AS target
        """)
        for record in rel_result:
            G.add_edge(str(record["source"]), str(record["target"]))

    if len(G.nodes) == 0:
        print("⚠️ Το γράφημα δεν έχει κόμβους!")
        return

    print("⏳ Εκπαίδευση Node2Vec...")
    node2vec = Node2Vec(G, dimensions=64, walk_length=10, num_walks=50, workers=1, seed=42)
    model = node2vec.fit(window=5, min_count=1)

    with driver.session() as session:
        for node_id in G.nodes():
            vec = model.wv[str(node_id)].tolist()
            session.run("""
                MATCH (c:Case {id: toInteger($id)})
                SET c.embedding = $embedding
            """, id=node_id, embedding=vec)

    print("✅ Embeddings αποθηκεύτηκαν!")

# --- Κύρια εκτέλεση ---
def build_graph():
    driver = connect_to_neo4j()
    file_path = "https://raw.githubusercontent.com/GiorgosBouh/llm2cypher-asd-app/main/Toddler_Autism_dataset_July_2018_2.csv"

    try:
        df = parse_csv(file_path)
        print("🧠 First row:", df.iloc[0].to_dict())

        with driver.session() as session:
            print("⏳ Δημιουργία κόμβων...")
            session.execute_write(create_nodes, df)
            print("⏳ Δημιουργία σχέσεων...")
            session.execute_write(create_relationships, df)
            print("⏳ Δημιουργία σχέσεων ομοιότητας...")
            session.execute_write(create_similarity_relationships, df)

        print("⏳ Δημιουργία embeddings...")
        generate_embeddings(driver)

        print("✅ Γράφος δημιουργήθηκε επιτυχώς!")

    except Exception as e:
        print(f"❌ Σφάλμα: {str(e)}")
    finally:
        driver.close()

if __name__ == "__main__":
    build_graph()
