# --- main_app.py (διορθωμενο) ---
import streamlit as st

# MUST be the first Streamlit command
st.set_page_config(layout="wide")
from neo4j import GraphDatabase
from openai import OpenAI
from dotenv import load_dotenv
import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score, roc_curve, precision_recall_curve,
    average_precision_score, classification_report
)
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from collections import Counter
import uuid
import numpy as np
import time
import matplotlib.pyplot as plt
import plotly.express as px
from contextlib import contextmanager
import logging
from typing import Optional, Tuple
from sklearn.preprocessing import StandardScaler

# === Configuration ===
class Config:
    NODE2VEC_EMBEDDING_DIM = 64
    NODE2VEC_ITERATIONS = 10
    RANDOM_STATE = 42
    TEST_SIZE = 0.2
    N_ESTIMATORS = 100
    SMOTE_RATIO = 'auto'
    MIN_CASES_FOR_ANOMALY_DETECTION = 10

# === Logging Setup ===
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# === Load environment variables ===
load_dotenv()

required_env_vars = ["NEO4J_URI", "NEO4J_USER", "NEO4J_PASSWORD", "OPENAI_API_KEY"]
missing_vars = [var for var in required_env_vars if not os.getenv(var)]
if missing_vars:
    st.error(f"Missing required environment variables: {', '.join(missing_vars)}")
    st.stop()

NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USER = os.getenv("NEO4J_USER")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

class Neo4jService:
    def __init__(self, uri: str, user: str, password: str):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    @contextmanager
    def session(self):
        with self.driver.session() as session:
            yield session

    def close(self):
        self.driver.close()

@st.cache_resource
def get_neo4j_service():
    return Neo4jService(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)

@st.cache_resource
def get_openai_client():
    return OpenAI(api_key=OPENAI_API_KEY)

neo4j_service = get_neo4j_service()
client = get_openai_client()

def safe_neo4j_operation(func):
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            st.error(f"Neo4j operation failed: {str(e)}")
            logger.error(f"Neo4j operation failed: {str(e)}")
            return None
    return wrapper

@safe_neo4j_operation
def insert_user_case(row: pd.Series, upload_id: str) -> None:
    queries = [("CREATE (c:Case {upload_id: $upload_id})", {"upload_id": upload_id})]
    for i in range(1, 11):
        q = f"A{i}"
        val = int(row[q])
        queries.append(("""
            MATCH (q:BehaviorQuestion {name: $q})
            MATCH (c:Case {upload_id: $upload_id})
            CREATE (c)-[:HAS_ANSWER {value: $val}]->(q)
            """, {"q": q, "val": val, "upload_id": upload_id}))

    demo = {"Sex": row["Sex"], "Ethnicity": row["Ethnicity"], "Jaundice": row["Jaundice"], "Family_mem_with_ASD": row["Family_mem_with_ASD"]}
    for k, v in demo.items():
        queries.append(("""
            MATCH (d:DemographicAttribute {type: $k, value: $v})
            MATCH (c:Case {upload_id: $upload_id})
            CREATE (c)-[:HAS_DEMOGRAPHIC]->(d)
            """, {"k": k, "v": v, "upload_id": upload_id}))

    queries.append(("""
        MATCH (s:SubmitterType {type: $who})
        MATCH (c:Case {upload_id: $upload_id})
        CREATE (c)-[:SUBMITTED_BY]->(s)
        """, {"who": row["Who_completed_the_test"], "upload_id": upload_id}))

    with neo4j_service.session() as session:
        for query, params in queries:
            session.run(query, **params)
        logger.info(f"Inserted case {upload_id}")

@safe_neo4j_operation
def extract_user_embedding(upload_id: str) -> Optional[np.ndarray]:
    with neo4j_service.session() as session:
        result = session.run("MATCH (c:Case {upload_id: $upload_id}) RETURN c.embedding AS embedding", upload_id=upload_id)
        record = result.single()
        return np.array([record["embedding"]]) if record and record["embedding"] is not None else None

@safe_neo4j_operation
def extract_training_data() -> Tuple[pd.DataFrame, pd.Series]:
    with neo4j_service.session() as session:
        result = session.run("""
            MATCH (c:Case)-[:SCREENED_FOR]->(t:ASD_Trait)
            WHERE c.embedding IS NOT NULL
            RETURN c.embedding AS embedding, t.value AS label
        """)
        records = result.data()

    if not records:
        return pd.DataFrame(), pd.Series()

    X = [[r["embedding"]] for r in records]
    y = [1 if r["label"] == "Yes" else 0 for r in records]
    return pd.DataFrame(X), pd.Series(y)

@st.cache_resource(show_spinner="Training model...")
def train_asd_detection_model():
    X, y = extract_training_data()
    if X.empty:
        st.warning("No training data available.")
        return None

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=Config.TEST_SIZE, random_state=Config.RANDOM_STATE, stratify=y)
    pipeline = Pipeline([
        ('smote', SMOTE(random_state=Config.RANDOM_STATE)),
        ('classifier', RandomForestClassifier(n_estimators=Config.N_ESTIMATORS, random_state=Config.RANDOM_STATE, class_weight='balanced'))
    ])
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    st.subheader("Model Evaluation")
    st.metric("ROC AUC", f"{roc_auc_score(y_test, y_proba):.3f}")
    st.metric("Avg Precision", f"{average_precision_score(y_test, y_proba):.3f}")

    return pipeline.named_steps['classifier']

# === Streamlit UI ===
st.title("🧠 NeuroCypher ASD")
st.sidebar.markdown(f"🔗 Connected to: `{NEO4J_URI}`")

st.header("💬 Natural Language to Cypher")
question = st.text_input("Ask a question:")

if question:
    cypher_query = nl_to_cypher(question)
    if cypher_query:
        st.code(cypher_query, language="cypher")

st.header("🤖 ASD Detection Model")
if st.button("🔄 Train/Refresh Model"):
    model = train_asd_detection_model()
    if model:
        st.success("Model trained successfully!")
        st.session_state['asd_model'] = model

st.header("📄 Upload New Case")
uploaded_file = st.file_uploader("Upload a CSV", type="csv")

def validate_csv(df: pd.DataFrame) -> bool:
    required_columns = [f"A{i}" for i in range(1, 11)] + ["Sex", "Ethnicity", "Jaundice", "Family_mem_with_ASD", "Who_completed_the_test"]
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        st.error(f"Missing columns: {', '.join(missing)}")
        return False
    return True

if uploaded_file:
    df = pd.read_csv(uploaded_file, delimiter=";")
    if validate_csv(df) and len(df) == 1:
        row = df.iloc[0]
        upload_id = str(uuid.uuid4())

        with st.spinner("Inserting case into graph..."):
            insert_user_case(row, upload_id)

        with st.spinner("Skipping embedding generation (already exists)..."):
            time.sleep(1)

        with st.spinner("Verifying embedding..."):
            embedding = extract_user_embedding(upload_id)
            if embedding is None:
                st.error("Embedding not found.")
                st.stop()
            st.success("Embedding found!")
            st.write(embedding)
