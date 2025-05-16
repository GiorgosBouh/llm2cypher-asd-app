import pandas as pd
import numpy as np
from neo4j import GraphDatabase
import os
import sys
from dotenv import load_dotenv
import logging
import traceback
import networkx as nx
from node2vec import Node2Vec

# Load environment variables
load_dotenv()

# Logging setup
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def connect_to_neo4j():
    uri = os.getenv("NEO4J_URI")
    user = os.getenv("NEO4J_USER")
    password = os.getenv("NEO4J_PASSWORD")
    return GraphDatabase.driver(uri, auth=(user, password))

def parse_csv(file_path):
    df = pd.read_csv(file_path, sep=";", encoding="utf-8-sig")
    df.columns = [col.strip() for col in df.columns]
    df = df.apply(lambda col: col.str.strip() if col.dtypes == 'object' else col)

    numeric_cols = ['Case_No', 'A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'A8', 'A9', 'A10']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col].astype(str).str.replace(",", "."), errors='coerce')

    logger.info("✅ Successfully parsed and cleaned data")
    return df.dropna()

def create_nodes(driver, df):
    with driver.session() as session:
        for q in [f"A{i}" for i in range(1, 11)]:
            session.run("MERGE (:BehaviorQuestion {name: $q})", q=q)

        for column in ["Sex", "Ethnicity", "Jaundice", "Family_mem_with_ASD"]:
            for val in df[column].dropna().unique():
                session.run("MERGE (:DemographicAttribute {type: $type, value: $val})", type=column, val=val)

        for val in df["Who_completed_the_test"].dropna().unique():
            session.run("MERGE (:SubmitterType {type: $val})", val=val)

def create_relationships(driver, df):
    for index, row in df.iterrows():
        with driver.session() as session:
            case_id = int(row["Case_No"])
            upload_id = str(case_id)
            session.run("MERGE (c:Case {id: $id}) SET c.upload_id = $uid, c.embedding = null", id=case_id, uid=upload_id)

            for q in [f"A{i}" for i in range(1, 11)]:
                session.run("""
                    MATCH (q:BehaviorQuestion {name: $q})
                    MATCH (c:Case {upload_id: $uid})
                    MERGE (c)-[:HAS_ANSWER {value: $val}]->(q)
                """, q=q, val=int(row[q]), uid=upload_id)

            for col in ["Sex", "Ethnicity", "Jaundice", "Family_mem_with_ASD"]:
                session.run("""
                    MATCH (d:DemographicAttribute {type: $type, value: $val})
                    MATCH (c:Case {upload_id: $uid})
                    MERGE (c)-[:HAS_DEMOGRAPHIC]->(d)
                """, type=col, val=row[col], uid=upload_id)

            session.run("""
                MATCH (s:SubmitterType {type: $val})
                MATCH (c:Case {upload_id: $uid})
                MERGE (c)-[:SUBMITTED_BY]->(s)
            """, val=row["Who_completed_the_test"], uid=upload_id)

def create_similarity_relationships(driver, df):
    pairs = set()
    for i, row1 in df.iterrows():
        for j, row2 in df.iloc[i+1:].iterrows():
            shared = sum(row1[f"A{k}"] == row2[f"A{k}"] for k in range(1, 11))
            if shared >= 7:
                pairs.add((int(row1['Case_No']), int(row2['Case_No']), shared / 10.0))

    logger.info(f"🔗 Creating {len(pairs)} SIMILAR_TO relationships")
    batch = []
    for id1, id2, weight in pairs:
        batch.append({'id1': id1, 'id2': id2, 'weight': weight})
        if len(batch) >= 100:
            with driver.session() as session:
                session.run("""
                    UNWIND $batch AS pair
                    MATCH (a:Case {id: pair.id1})
                    MATCH (b:Case {id: pair.id2})
                    MERGE (a)-[:SIMILAR_TO {type: 'answer', weight: pair.weight}]->(b)
                """, batch=batch)
            batch = []
    if batch:
        with driver.session() as session:
            session.run("""
                UNWIND $batch AS pair
                MATCH (a:Case {id: pair.id1})
                MATCH (b:Case {id: pair.id2})
                MERGE (a)-[:SIMILAR_TO {type: 'answer', weight: pair.weight}]->(b)
            """, batch=batch)

def generate_embeddings(driver):
    G = nx.Graph()
    with driver.session() as session:
        result = session.run("""
            MATCH (c:Case)
            OPTIONAL MATCH (c)-[r:HAS_ANSWER|HAS_DEMOGRAPHIC|SUBMITTED_BY|SIMILAR_TO]->(n)
            RETURN c.id AS case_id, collect(DISTINCT n.id) AS neighbors
        """)
        for record in result:
            node_id = str(record["case_id"])
            G.add_node(node_id)
            for neighbor in record["neighbors"]:
                if neighbor:
                    G.add_edge(node_id, str(neighbor))

    if len(G.nodes) < 10:
        raise ValueError("Graph too small for embeddings")

    node2vec = Node2Vec(G, dimensions=128, walk_length=30, num_walks=200, workers=2)
    model = node2vec.fit(window=10, min_count=1)

    logger.info("🧠 Saving embeddings to Neo4j")
    batch = []
    for node in G.nodes:
        emb = model.wv[node].tolist()
        batch.append({'id': int(node), 'embedding': emb})
        if len(batch) >= 100:
            with driver.session() as session:
                session.run("""
                    UNWIND $batch AS item
                    MATCH (c:Case {id: item.id})
                    SET c.embedding = item.embedding
                """, batch=batch)
            batch = []
    if batch:
        with driver.session() as session:
            session.run("""
                UNWIND $batch AS item
                MATCH (c:Case {id: item.id})
                SET c.embedding = item.embedding
            """, batch=batch)

def build_graph():
    driver = connect_to_neo4j()
    file_path = os.getenv("DATA_URL", "https://raw.githubusercontent.com/GiorgosBouh/llm2cypher-asd-app/main/Toddler_Autism_dataset_July_2018_2.csv")

    try:
        df = parse_csv(file_path)
        logger.info("🧠 First row: %s", df.iloc[0].to_dict())

        logger.info("🧹 Deleting existing graph...")
        with driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")

        logger.info("⏳ Creating nodes...")
        create_nodes(driver, df)

        logger.info("⏳ Creating relationships...")
        create_relationships(driver, df)

        logger.info("⏳ Creating similarity relationships...")
        create_similarity_relationships(driver, df)

        logger.info("⏳ Generating embeddings...")
        generate_embeddings(driver)

        logger.info("✅ Graph build completed successfully!")
        sys.exit(0)

    except Exception as e:
        logger.error(f"❌ Error: {str(e)}")
        traceback.print_exc()
        sys.exit(1)

    finally:
        driver.close()

if __name__ == "__main__":
    build_graph()
