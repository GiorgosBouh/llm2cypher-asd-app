import pandas as pd
import numpy as np
from neo4j import GraphDatabase
import networkx as nx
from node2vec import Node2Vec
from random import shuffle
import traceback
import sys
from random import randint
import os


def connect_to_neo4j(uri="neo4j+s://1f5f8a14.databases.neo4j.io", user="neo4j", password="3xhy4XKQSsSLIT7NI-w9m4Z7Y_WcVnL1hDQkWTMIoMQ"):
    print(f"🌐 Connecting to Neo4j Aura: {uri}", flush=True)
    return GraphDatabase.driver(uri, auth=(user, password))

def parse_csv(file_path):
    df = pd.read_csv(file_path, sep=";", encoding="utf-8-sig")
    df.columns = [col.strip() for col in df.columns]
    df = df.apply(lambda col: col.str.strip() if col.dtypes == 'object' else col)

    numeric_cols = ['Case_No', 'A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'A8', 'A9', 'A10', 'Age_Mons', 'Qchat-10-Score']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col].astype(str).str.replace(",", "."), errors='coerce')

    print("✅ Καθαρίστηκαν οι στήλες:", df.columns.tolist(), flush=True)
    return df.dropna()

def create_nodes(tx, df):
    for q in [f"A{i}" for i in range(1, 11)]:
        tx.run("MERGE (:BehaviorQuestion {name: $q})", q=q)

    for column in ["Sex", "Ethnicity", "Jaundice", "Family_mem_with_ASD"]:
        for val in df[column].dropna().unique():
            tx.run("MERGE (:DemographicAttribute {type: $type, value: $val})", type=column, val=val)

    for val in df["Who_completed_the_test"].dropna().unique():
        tx.run("MERGE (:SubmitterType {type: $val})", val=val)

def create_relationships(tx, df):
    case_data = []
    answer_data, demo_data, submitter_data = [], [], []

    for _, row in df.iterrows():
        case_id = int(row["Case_No"])
        upload_id = str(case_id)
        case_data.append({"id": case_id, "upload_id": upload_id})

        for q in [f"A{i}" for i in range(1, 11)]:
            answer_data.append({"upload_id": upload_id, "q": q, "val": int(row[q])})

        for col in ["Sex", "Ethnicity", "Jaundice", "Family_mem_with_ASD"]:
            demo_data.append({"upload_id": upload_id, "type": col, "val": row[col]})

        submitter_data.append({"upload_id": upload_id, "val": row["Who_completed_the_test"]})

    tx.run("UNWIND $data as row MERGE (c:Case {id: row.id}) SET c.upload_id = row.upload_id, c.embedding = null", data=case_data)
    tx.run("""
        UNWIND $data as row
        MATCH (q:BehaviorQuestion {name: row.q})
        MATCH (c:Case {upload_id: row.upload_id})
        MERGE (c)-[:HAS_ANSWER {value: row.val}]->(q)
    """, data=answer_data)
    tx.run("""
        UNWIND $data as row
        MATCH (d:DemographicAttribute {type: row.type, value: row.val})
        MATCH (c:Case {upload_id: row.upload_id})
        MERGE (c)-[:HAS_DEMOGRAPHIC]->(d)
    """, data=demo_data)
    tx.run("""
        UNWIND $data as row
        MATCH (s:SubmitterType {type: row.val})
        MATCH (c:Case {upload_id: row.upload_id})
        MERGE (c)-[:SUBMITTED_BY]->(s)
    """, data=submitter_data)
def create_similarity_relationships(tx, df, max_pairs=10000):
    pairs = set()
    
    # 1. Συμπεριφορική ομοιότητα (A1-A10)
    for i, row1 in df.iterrows():
        for j, row2 in df.iloc[i+1:].iterrows():
            if sum(row1[f'A{k}'] == row2[f'A{k}'] for k in range(1,11)) >= 7:
                pairs.add((int(row1['Case_No']), int(row2['Case_No'])))

    # 2. Δημογραφική ομοιότητα (διορθωμένο groupby)
    demo_cols = ['Sex', 'Ethnicity', 'Jaundice', 'Family_mem_with_ASD']
    for col in demo_cols:
        # Σωστή χρήση groupby με ένα μόνο κριτήριο
        grouped = df.groupby(col)['Case_No'].apply(list)
        for case_list in grouped:
            for i in range(len(case_list)):
                for j in range(i+1, len(case_list)):
                    pairs.add((int(case_list[i]), int(case_list[j])))

    # Εφαρμογή ορίου και τυχαιοποίηση
    pair_list = list(pairs)[:max_pairs]
    shuffle(pair_list)

    # Εισαγωγή σχέσεων στη Neo4j
    tx.run("""
        UNWIND $batch AS pair
        MATCH (c1:Case {id: pair.id1}), (c2:Case {id: pair.id2})
        MERGE (c1)-[:SIMILAR_TO]->(c2)
    """, batch=[{'id1':x, 'id2':y} for x,y in pair_list])

def generate_embeddings(driver):
    temp_folder_path = os.path.join(os.getcwd(), 'node2vec_temp')
    os.makedirs(temp_folder_path, exist_ok=True)
    G = nx.Graph()
    
    # Επιτάχυνση φόρτωσης γραφήματος με single query
    with driver.session() as session:
        result = session.run("""
            MATCH (c:Case)
            OPTIONAL MATCH (c)-[r:HAS_ANSWER|HAS_DEMOGRAPHIC|SUBMITTED_BY|GRAPH_SIMILARITY]->(other)
            WHERE c.id IS NOT NULL AND other.id IS NOT NULL
            RETURN c.id AS node_id, collect(DISTINCT other.id) AS neighbors
        """)
        
        for record in result:
            node_id = str(record["node_id"])
            G.add_node(node_id)
            for neighbor in record["neighbors"]:
                G.add_edge(node_id, str(neighbor))

    # Έλεγχος γραφήματος
    if len(G.nodes) < 10:  # Αύξηση ελάχιστου ορίου για καλύτερα embeddings
        raise ValueError(f"❌ Not enough nodes ({len(G.nodes)}) for meaningful embeddings")

    print(f"📊 Graph stats: {len(G.nodes)} nodes, {len(G.edges)} edges")
    
    # Βελτιστοποιημένοι παράμετροι Node2Vec
    node2vec = Node2Vec(
        G,
        dimensions=128,       # Συνέπεια με το generate_case_embedding.py
        walk_length=30,       # Αυξημένο για καλύτερη εξερεύνηση γραφήματος
        num_walks=200,        # Περισσότερα walks για σταθερότερα αποτελέσματα
        workers=4,            # Αξιοποίηση πολυπύρηνων επεξεργαστών
        p=1.0,                # Παράμετρος p για BFS
        q=0.5,                # Παράμετρος q για DFS
        temp_folder=os.path.join(os.getcwd(), 'node2vec_temp')  # Προσωρινός φάκελος
    )

    try:
        model = node2vec.fit(
            window=10,
            min_count=1,
            batch_words=128
        )
        
        # Επιτάχυνση αποθήκευσης με batch query
        with driver.session() as session:
            batch = []
            for node_id in G.nodes():
                embedding = model.wv[str(node_id)].tolist()
                batch.append({"node_id": int(node_id), "embedding": embedding})
                
                if len(batch) >= 1000:  # Batch ενημερώσεων
                    session.run("""
                        UNWIND $batch AS item
                        MATCH (c:Case {id: item.node_id})
                        SET c.embedding = item.embedding
                    """, {"batch": batch})
                    batch = []
            
            if batch:  # Τελευταίο batch
                session.run("""
                    UNWIND $batch AS item
                    MATCH (c:Case {id: item.node_id})
                    SET c.embedding = item.embedding
                """, {"batch": batch})

        print(f"✅ Successfully saved embeddings for {len(G.nodes)} nodes")
        return True

    except Exception as e:
        print(f"❌ Failed to generate embeddings: {str(e)}")
        return False

def build_graph():
    driver = connect_to_neo4j()
    file_path = "https://raw.githubusercontent.com/GiorgosBouh/llm2cypher-asd-app/main/Toddler_Autism_dataset_July_2018_2.csv"

    try:
        df = parse_csv(file_path)
        print("🧠 First row:", df.iloc[0].to_dict(), flush=True)

        with driver.session() as session:
            print("🧹 Διαγραφή όλων των κόμβων και σχέσεων...", flush=True)
            session.run("MATCH (n) DETACH DELETE n")

            print("⏳ Δημιουργία κόμβων...", flush=True)
            session.execute_write(create_nodes, df)

            print("⏳ Δημιουργία σχέσεων...", flush=True)
            session.execute_write(create_relationships, df)

            print("⏳ Δημιουργία σχέσεων ομοιότητας...", flush=True)
            session.execute_write(create_similarity_relationships, df)

        print("⏳ Δημιουργία embeddings...", flush=True)
        generate_embeddings(driver)

        print("✅ Ολοκληρώθηκε επιτυχώς!", flush=True)
        sys.exit(0)

    except Exception as e:
        print(f"❌ Σφάλμα: {str(e)}", flush=True)
        traceback.print_exc()
        sys.exit(1)

    finally:
        driver.close()

if __name__ == "__main__":
    build_graph()