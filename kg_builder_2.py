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
    # Δημιουργία κόμβων ερωτήσεων συμπεριφοράς (A1-A10)
    for q in [f"A{i}" for i in range(1, 11)]:
        tx.run("MERGE (:BehaviorQuestion {name: $q})", q=q)

    # Δημιουργία δημογραφικών κόμβων
    for column in ["Sex", "Ethnicity", "Jaundice", "Family_mem_with_ASD"]:
        for val in df[column].dropna().unique():
            tx.run("MERGE (:DemographicAttribute {type: $type, value: $val})", type=column, val=val)

    # Δημιουργία κόμβων υποβληθέντων από
    for val in df["Who_completed_the_test"].dropna().unique():
        tx.run("MERGE (:SubmitterType {type: $val})", val=val)

def create_relationships(tx, df):
    # Προετοιμασία δεδομένων για μαζική εισαγωγή
    case_data = []
    answer_data, demo_data, submitter_data = [], [], []

    for _, row in df.iterrows():
        case_id = int(row["Case_No"])
        upload_id = str(case_id)
        case_data.append({"id": case_id, "upload_id": upload_id})

        # Σχέσεις απαντήσεων (HAS_ANSWER)
        for q in [f"A{i}" for i in range(1, 11)]:
            answer_data.append({"upload_id": upload_id, "q": q, "val": int(row[q])})

        # Σχέσεις δημογραφικών (HAS_DEMOGRAPHIC)
        for col in ["Sex", "Ethnicity", "Jaundice", "Family_mem_with_ASD"]:
            demo_data.append({"upload_id": upload_id, "type": col, "val": row[col]})

        # Σχέσεις υποβληθέντων από (SUBMITTED_BY)
        submitter_data.append({"upload_id": upload_id, "val": row["Who_completed_the_test"]})

    # Μαζική εισαγωγή Cases
    tx.run("""
        UNWIND $data as row 
        MERGE (c:Case {id: row.id}) 
        SET c.upload_id = row.upload_id, c.embedding = null
    """, data=case_data)

    # Μαζική εισαγωγή HAS_ANSWER σχέσεων
    tx.run("""
        UNWIND $data as row
        MATCH (q:BehaviorQuestion {name: row.q})
        MATCH (c:Case {upload_id: row.upload_id})
        MERGE (c)-[:HAS_ANSWER {value: row.val}]->(q)
    """, data=answer_data)

    # Μαζική εισαγωγή HAS_DEMOGRAPHIC σχέσεων
    tx.run("""
        UNWIND $data as row
        MATCH (d:DemographicAttribute {type: row.type, value: row.val})
        MATCH (c:Case {upload_id: row.upload_id})
        MERGE (c)-[:HAS_DEMOGRAPHIC]->(d)
    """, data=demo_data)

    # Μαζική εισαγωγή SUBMITTED_BY σχέσεων
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

    # 2. Δημογραφική ομοιότητα
    demo_cols = ['Sex', 'Ethnicity', 'Jaundice', 'Family_mem_with_ASD']
    for col in demo_cols:
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
    
    print("⏳ Φόρτωση γράφου από Neo4j...", flush=True)
    
    # Βήμα 1: Φόρτωση όλων των κόμβων Case με συνεπή αναπαράσταση ID
    with driver.session() as session:
        # Φόρτωση όλων των Case κόμβων ως strings
        cases = session.run("MATCH (c:Case) RETURN c.id AS id")
        case_ids = [f"Case_{record['id']}" for record in cases]
        G.add_nodes_from(case_ids, type="Case")
        print(f"📊 Φορτώθηκαν {len(case_ids)} κόμβοι Case", flush=True)
        
        # Βήμα 2: Ειδική φόρτωση SIMILAR_TO σχέσεων
        similar_relations = session.run("""
            MATCH (c1:Case)-[r:SIMILAR_TO]->(c2:Case)
            RETURN c1.id AS source_id, c2.id AS target_id
        """)
        
        sim_count = 0
        for record in similar_relations:
            source = f"Case_{record['source_id']}"
            target = f"Case_{record['target_id']}"
            if source in G and target in G:  # Έλεγχος ύπαρξης κόμβων
                G.add_edge(source, target, type="SIMILAR_TO")
                sim_count += 1
        
        print(f"📊 Φορτώθηκαν {sim_count} σχέσεις SIMILAR_TO", flush=True)
        
        # Βήμα 3: Φόρτωση άλλων τύπων σχέσεων
        other_relations = session.run("""
            MATCH (c:Case)-[r:HAS_ANSWER|HAS_DEMOGRAPHIC|SUBMITTED_BY]->(n)
            RETURN c.id AS case_id, type(r) AS rel_type,
                   coalesce(n.name, n.type, n.value, toString(id(n))) AS target_name,
                   labels(n) AS target_labels
        """)
        
        other_count = 0
        for record in other_relations:
            source = f"Case_{record['case_id']}"
            target = f"{record['target_labels'][0]}_{record['target_name']}"
            G.add_node(target)
            G.add_edge(source, target, type=record['rel_type'])
            other_count += 1
        
        print(f"📊 Φορτώθηκαν {other_count} άλλες σχέσεις", flush=True)
    
    print(f"📈 Τελικός γράφος: {len(G.nodes)} κόμβοι, {len(G.edges)} ακμές", flush=True)
    
    if len(G.edges) == 0:
        raise ValueError("❌ Ο γράφος είναι άδειος! Έλεγξε τα δεδομένα.")
    
    # Δημιουργία embeddings μόνο για κόμβους Case
    case_nodes = [n for n in G.nodes if G.nodes[n].get('type') == 'Case']
    print(f"🔍 Θα δημιουργηθούν embeddings για {len(case_nodes)} κόμβους Case", flush=True)
    
    node2vec = Node2Vec(
        G,
        dimensions=128,
        walk_length=30,
        num_walks=200,
        workers=4,
        p=1.0,
        q=0.5,
        temp_folder=temp_folder_path
    )
    
    try:
        model = node2vec.fit(
            window=10,
            min_count=1,
            batch_words=128
        )
        
        # Αποθήκευση embeddings
        with driver.session() as session:
            batch = []
            for node in case_nodes:
                case_id = int(node.split('_')[1])
                try:
                    embedding = model.wv[node].tolist()
                    batch.append({"case_id": case_id, "embedding": embedding})
                    
                    if len(batch) >= 500:
                        session.run("""
                            UNWIND $batch AS item
                            MATCH (c:Case {id: item.case_id})
                            SET c.embedding = item.embedding
                        """, {"batch": batch})
                        batch = []
                
                except KeyError:
                    print(f"⚠️ Χωρίς embedding: {node}", flush=True)
                    continue
            
            if batch:
                session.run("""
                    UNWIND $batch AS item
                    MATCH (c:Case {id: item.case_id})
                    SET c.embedding = item.embedding
                """, {"batch": batch})
        
        print(f"✅ Αποθηκεύτηκαν embeddings για {len(batch)} κόμβους", flush=True)
        return True
    
    except Exception as e:
        print(f"❌ Σφάλμα: {str(e)}", flush=True)
        traceback.print_exc()
        return False
    finally:
        if os.path.exists(temp_folder_path):
            shutil.rmtree(temp_folder_path)

def build_graph():
    driver = connect_to_neo4j()
    file_path = "https://raw.githubusercontent.com/GiorgosBouh/llm2cypher-asd-app/main/Toddler_Autism_dataset_July_2018_2.csv"

    try:
        # Φόρτωση και προετοιμασία δεδομένων
        print("⏳ Φόρτωση και καθαρισμός δεδομένων...", flush=True)
        df = parse_csv(file_path)
        print("🧠 Δείγμα δεδομένων:", df.iloc[0].to_dict(), flush=True)

        # Δημιουργία γράφου στη Neo4j
        with driver.session() as session:
            print("🧹 Διαγραφή παλιού γράφου...", flush=True)
            session.run("MATCH (n) DETACH DELETE n")

            print("⏳ Δημιουργία νέων κόμβων...", flush=True)
            session.execute_write(create_nodes, df)

            print("⏳ Δημιουργία βασικών σχέσεων...", flush=True)
            session.execute_write(create_relationships, df)

            print("⏳ Δημιουργία σχέσεων ομοιότητας...", flush=True)
            session.execute_write(create_similarity_relationships, df)

        # Δημιουργία και αποθήκευση embeddings
        print("⏳ Δημιουργία graph embeddings...", flush=True)
        if not generate_embeddings(driver):
            raise RuntimeError("Failed to generate embeddings")

        print("✅ Ολοκληρώθηκε επιτυχώς η δημιουργία γράφου!", flush=True)
        sys.exit(0)

    except Exception as e:
        print(f"❌ Κρίσιμο σφάλμα: {str(e)}", flush=True)
        traceback.print_exc()
        sys.exit(1)

    finally:
        driver.close()

if __name__ == "__main__":
    build_graph()