print("=== RUNNING kg_builder_2.py from Git repo ===", flush=True)
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
import shutil  # Προσθήκη της λείπεις βιβλιοθήκης


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

    print("✅ Καθαριστηκαν οι στήλες:", df.columns.tolist(), flush=True)
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

    tx.run("""
        UNWIND $data as row 
        MERGE (c:Case {id: row.id}) 
        SET c.upload_id = row.upload_id, c.embedding = null
    """, data=case_data)

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
    
    with driver.session() as session:
        # Φόρτωση όλων των Case κόμβων
        cases = session.run("MATCH (c:Case) RETURN c.id AS id")
        case_ids = [f"Case_{record['id']}" for record in cases]
        G.add_nodes_from(case_ids, type="Case")
        print(f"📊 Φορτώθηκαν {len(case_ids)} κόμβοι Case", flush=True)
        
        # Βελτιστοποιημένο query για φόρτωση σχέσεων
        relationships = session.run("""
            MATCH (c:Case)-[r]->(n)
            RETURN c.id AS source_id, 
                   type(r) AS rel_type,
                   CASE 
                     WHEN n:BehaviorQuestion THEN 'Q_' + n.name
                     WHEN n:DemographicAttribute THEN 'D_' + n.type + '_' + replace(n.value, ' ', '_')
                     WHEN n:SubmitterType THEN 'S_' + replace(n.type, ' ', '_')
                     WHEN n:Case THEN 'Case_' + toString(n.id)
                     ELSE 'Node_' + toString(id(n))
                   END AS target_id
        """)
        
        edge_count = 0
        for record in relationships:
            source = f"Case_{record['source_id']}"
            target = record['target_id']
            G.add_node(target)
            G.add_edge(source, target, type=record['rel_type'])
            edge_count += 1
        
        print(f"📊 Φορτώθηκαν {edge_count} ακμές", flush=True)
    
    print(f"📈 Τελικός γράφος: {len(G.nodes)} κόμβοι, {len(G.edges)} ακμές", flush=True)
    
    if len(G.edges) == 0:
        raise ValueError("❌ Ο γράφος είναι άδειος! Έλεγξε τα δεδομένα στη Neo4j.")
    
    # Δημιουργία embeddings μόνο για κόμβους Case
    case_nodes = [n for n in G.nodes if G.nodes[n].get('type') == 'Case']
    print(f"🔍 Θα δημιουργηθούν embeddings για {len(case_nodes)} κόμβους Case", flush=True)
    
    # Ρύθμιση Node2Vec για καλύτερη απόδοση
    node2vec = Node2Vec(
        G,
        dimensions=128,
        walk_length=30,
        num_walks=100,  # Μειωμένο για ταχύτερη εκτέλεση
        workers=4,
        p=1.0,
        q=0.5,
        temp_folder=temp_folder_path,
        quiet=True  # Λιγότερο output
    )
    
    try:
        model = node2vec.fit(
            window=10,
            min_count=1,
            batch_words=128
        )
        
        # Αποθήκευση embeddings με batch
        with driver.session() as session:
            batch_size = 500
            batch = []
            saved_count = 0
            
            for node in case_nodes:
                case_id = int(node.split('_')[1])
                try:
                    embedding = model.wv[node].tolist()
                    batch.append({"case_id": case_id, "embedding": embedding})
                    
                    if len(batch) >= batch_size:
                        session.run("""
                            UNWIND $batch AS item
                            MATCH (c:Case {id: item.case_id})
                            SET c.embedding = item.embedding
                        """, {"batch": batch})
                        saved_count += len(batch)
                        batch = []
                        print(f"✅ Αποθηκεύτηκαν {saved_count}/{len(case_nodes)} embeddings", flush=True)
                
                except KeyError:
                    print(f"⚠️ Χωρίς embedding: {node}", flush=True)
                    continue
            
            if batch:
                session.run("""
                    UNWIND $batch AS item
                    MATCH (c:Case {id: item.case_id})
                    SET c.embedding = item.embedding
                """, {"batch": batch})
                saved_count += len(batch)
        
        print(f"✅ Ολοκληρώθηκε η αποθήκευση {saved_count} embeddings", flush=True)
        return True
    
    except Exception as e:
        print(f"❌ Σφάλμα: {str(e)}", flush=True)
        traceback.print_exc()
        return False
    finally:
        # Καθαρισμός προσωρινού φακέλου με έλεγχο σφαλμάτων
        try:
            if os.path.exists(temp_folder_path):
                shutil.rmtree(temp_folder_path)
        except Exception as e:
            print(f"⚠️ Σφάλμα καθαρισμού temp folder: {str(e)}", flush=True)

def build_graph():
    driver = connect_to_neo4j()
    file_path = "https://raw.githubusercontent.com/GiorgosBouh/llm2cypher-asd-app/main/Toddler_Autism_dataset_July_2018_2.csv"

    try:
        print("⏳ Φόρτωση και καθαρισμός δεδομένων...", flush=True)
        df = parse_csv(file_path)
        print("🧠 Δείγμα δεδομένων:", df.iloc[0].to_dict(), flush=True)

        with driver.session() as session:
            print("🧹 Διαγραφή παλιού γράφου...", flush=True)
            session.run("MATCH (n) DETACH DELETE n")

            print("⏳ Δημιουργία νέων κόμβων...", flush=True)
            session.execute_write(create_nodes, df)

            print("⏳ Δημιουργία βασικών σχέσεων...", flush=True)
            session.execute_write(create_relationships, df)

            print("⏳ Δημιουργία σχέσεων ομοιότητας...", flush=True)
            session.execute_write(create_similarity_relationships, df)

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