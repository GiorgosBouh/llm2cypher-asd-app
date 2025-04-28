import os
import uuid
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import plotly.express as px
import networkx as nx
from neo4j import GraphDatabase
from dotenv import load_dotenv
from node2vec import Node2Vec
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve, average_precision_score, classification_report
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from contextlib import contextmanager
import logging
from openai import OpenAI
from typing import Optional, Tuple

# === Configuration ===
class Config:
    EMBEDDING_DIM = 64
    RANDOM_STATE = 42
    TEST_SIZE = 0.2
    N_ESTIMATORS = 100

# === Logger ===
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# === Load env ===
load_dotenv()

NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USER = os.getenv("NEO4J_USER")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# === Neo4j Service ===
class Neo4jService:
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    @contextmanager
    def session(self):
        with self.driver.session() as session:
            yield session

    def close(self):
        self.driver.close()

# === Services ===
neo4j_service = Neo4jService(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)
client = OpenAI(api_key=OPENAI_API_KEY)

# === Helper Functions ===
def safe_neo4j_operation(func):
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            st.error(f"Neo4j error: {str(e)}")
            logger.error(f"Neo4j error: {str(e)}")
            return None
    return wrapper

@safe_neo4j_operation
def insert_case(row, upload_id):
    queries = []

    queries.append(("CREATE (c:Case {upload_id: $upload_id})", {"upload_id": upload_id}))

    for q in [f"A{i}" for i in range(1, 11)]:
        queries.append((
            "MATCH (q:BehaviorQuestion {name: $q}) "
            "MATCH (c:Case {upload_id: $upload_id}) "
            "CREATE (c)-[:HAS_ANSWER {value: $val}]->(q)",
            {"q": q, "val": int(row[q]), "upload_id": upload_id}
        ))

    for k in ["Sex", "Ethnicity", "Jaundice", "Family_mem_with_ASD"]:
        queries.append((
            "MATCH (d:DemographicAttribute {type: $k, value: $v}) "
            "MATCH (c:Case {upload_id: $upload_id}) "
            "CREATE (c)-[:HAS_DEMOGRAPHIC]->(d)",
            {"k": k, "v": row[k], "upload_id": upload_id}
        ))

    queries.append((
        "MATCH (s:SubmitterType {type: $who}) "
        "MATCH (c:Case {upload_id: $upload_id}) "
        "CREATE (c)-[:SUBMITTED_BY]->(s)",
        {"who": row["Who_completed_the_test"], "upload_id": upload_id}
    ))

    with neo4j_service.session() as session:
        for query, params in queries:
            session.run(query, **params)

@safe_neo4j_operation
def generate_embedding(upload_id):
    """Generate node2vec embedding for the inserted case"""
    with neo4j_service.session() as session:
        G = nx.Graph()

        nodes = session.run("MATCH (c:Case) RETURN c.upload_id AS id")
        for record in nodes:
            G.add_node(record["id"])

        edges = session.run("""
            MATCH (c1:Case)-[r]->(c2)
            RETURN c1.upload_id AS source, c2
        """)
        for record in edges:
            if record["source"] and record["c2"]:
                G.add_edge(record["source"], str(record["c2"].id))

        if len(G.nodes) < 2:
            return None

        node2vec = Node2Vec(G, dimensions=Config.EMBEDDING_DIM, seed=Config.RANDOM_STATE)
        model = node2vec.fit()

        vec = model.wv[upload_id]
        session.run("MATCH (c:Case {upload_id: $upload_id}) SET c.embedding_vector = $embedding", 
                    upload_id=upload_id, embedding=vec.tolist())
        return vec

@safe_neo4j_operation
def extract_training_data():
    with neo4j_service.session() as session:
        result = session.run("""
            MATCH (c:Case)-[:SCREENED_FOR]->(t:ASD_Trait)
            WHERE c.embedding_vector IS NOT NULL
            RETURN c.embedding_vector AS embedding, t.value AS label
        """)
        data = result.data()
    if not data:
        return pd.DataFrame(), pd.Series()
    X = np.array([d["embedding"] for d in data])
    y = np.array([1 if d["label"].lower() == "yes" else 0 for d in data])
    return pd.DataFrame(X), pd.Series(y)

def train_classifier():
    X, y = extract_training_data()
    if X.empty:
        return None
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=Config.TEST_SIZE, random_state=Config.RANDOM_STATE)

    pipe = Pipeline([
        ('smote', SMOTE(random_state=Config.RANDOM_STATE)),
        ('rf', RandomForestClassifier(n_estimators=Config.N_ESTIMATORS, random_state=Config.RANDOM_STATE))
    ])
    pipe.fit(X_train, y_train)
    return pipe.named_steps['rf']

def train_anomaly_detector():
    X, _ = extract_training_data()
    if X.empty:
        return None
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    model = IsolationForest(random_state=Config.RANDOM_STATE, contamination=0.1)
    model.fit(X_scaled)
    return model, scaler

def nl_to_cypher(question: str) -> Optional[str]:
    prompt = f"""
    Schema:
    - (:Case {{upload_id: string}})
    - (:BehaviorQuestion)
    - (:ASD_Trait)
    - (:DemographicAttribute)
    - (:SubmitterType)

    Relationships:
    - HAS_ANSWER
    - HAS_DEMOGRAPHIC
    - SUBMITTED_BY
    - SCREENED_FOR

    Translate this into Cypher:

    Q: {question}
    """
    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        cypher_query = response.choices[0].message.content.strip()
        return cypher_query.replace("```cypher", "").replace("```", "").strip()
    except Exception as e:
        st.error(f"OpenAI API error: {e}")
        logger.error(f"OpenAI API error: {e}")
        return None

# === Streamlit UI ===
st.set_page_config(layout="wide")
st.title("🧠 NeuroGraph ASD Prediction")

st.sidebar.title("🔧 Settings")

st.sidebar.markdown("**Quick Queries:**")
example_questions = {
    "Πόσα παιδιά έχουν ASD traits;": "How many children have ASD traits?",
    "Πόσα παιδιά είχαν ιστορικό ίκτερου;": "How many children had jaundice?",
    "Πόσα κορίτσια έχουν ASD traits;": "How many girls have ASD traits?",
    "Πόσα παιδιά με οικογενειακό ιστορικό ASD υπάρχουν;": "How many children have family history of ASD?",
    "Πόσα τεστ συμπληρώθηκαν από γιατρό;": "How many tests were completed by doctor?"
}
selected_question = st.sidebar.selectbox("Διάλεξε ερώτηση:", list(example_questions.keys()))

if st.sidebar.button("💬 Μετατροπή σε Cypher"):
    query = nl_to_cypher(example_questions[selected_question])
    st.code(query, language="cypher")
    if st.button("▶️ Εκτέλεση ερωτήματος"):
        with neo4j_service.session() as session:
            results = session.run(query).data()
            st.dataframe(pd.DataFrame(results))

# === Upload CSV ===
st.header("📄 Upload a Single Case")
uploaded_file = st.file_uploader("Upload CSV (only 1 row)", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file, delimiter=";")
    if df.shape[0] != 1:
        st.error("Please upload exactly ONE row!")
        st.stop()

    st.subheader("📊 Preview:")
    st.dataframe(df.T)

    row = df.iloc[0]
    upload_id = str(uuid.uuid4())

    with st.spinner("Inserting into graph..."):
        insert_case(row, upload_id)

    with st.spinner("Generating embedding..."):
        embedding = generate_embedding(upload_id)
        if embedding is None:
            st.error("Failed to generate embedding")
            st.stop()

    # === Train Classifier
    model = train_classifier()
    if model:
        proba = model.predict_proba([embedding])[0][1]
        prediction = "YES (ASD Traits)" if proba >= 0.5 else "NO (Control)"
        st.success(f"Prediction: {prediction} (Confidence: {proba:.2%})")

    # === Train Anomaly Detector
    iso_forest, scaler = train_anomaly_detector()
    if iso_forest:
        scaled_emb = scaler.transform([embedding])
        anomaly_score = iso_forest.decision_function(scaled_emb)[0]
        is_anomaly = iso_forest.predict(scaled_emb)[0] == -1
        if is_anomaly:
            st.warning(f"Anomaly detected! (score: {anomaly_score:.3f})")
        else:
            st.success(f"Normal case (score: {anomaly_score:.3f})")