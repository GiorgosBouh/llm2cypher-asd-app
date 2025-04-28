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
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# === Load environment variables ===
load_dotenv()

# === Validate Environment Variables ===
required_env_vars = ["NEO4J_URI", "NEO4J_USER", "NEO4J_PASSWORD", "OPENAI_API_KEY"]
missing_vars = [var for var in required_env_vars if not os.getenv(var)]
if missing_vars:
    st.error(f"Missing required environment variables: {', '.join(missing_vars)}")
    st.stop()

# === Credentials ===
NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USER = os.getenv("NEO4J_USER")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# === Neo4j Service Class ===
class Neo4jService:
    def __init__(self, uri: str, user: str, password: str):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    @contextmanager
    def session(self):
        with self.driver.session() as session:
            yield session

    def close(self):
        self.driver.close()

# === Initialize Services ===
@st.cache_resource
def get_neo4j_service():
    return Neo4jService(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)

@st.cache_resource
def get_openai_client():
    return OpenAI(api_key=OPENAI_API_KEY)

neo4j_service = get_neo4j_service()
client = get_openai_client()

# === Helper Functions ===
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
    queries = []
    params = {"upload_id": upload_id}

    queries.append(("CREATE (c:Case {upload_id: $upload_id})", params))

    for i in range(1, 11):
        q = f"A{i}"
        val = int(row[q])
        queries.append((
            """
            MATCH (q:BehaviorQuestion {name: $q})
            MATCH (c:Case {upload_id: $upload_id})
            CREATE (c)-[:HAS_ANSWER {value: $val}]->(q)
            """,
            {"q": q, "val": val, "upload_id": upload_id}
        ))

    demo = {
        "Sex": row["Sex"],
        "Ethnicity": row["Ethnicity"],
        "Jaundice": row["Jaundice"],
        "Family_mem_with_ASD": row["Family_mem_with_ASD"]
    }
    for k, v in demo.items():
        queries.append((
            """
            MATCH (d:DemographicAttribute {type: $k, value: $v})
            MATCH (c:Case {upload_id: $upload_id})
            CREATE (c)-[:HAS_DEMOGRAPHIC]->(d)
            """,
            {"k": k, "v": v, "upload_id": upload_id}
        ))

    queries.append((
        """
        MATCH (s:SubmitterType {type: $who})
        MATCH (c:Case {upload_id: $upload_id})
        CREATE (c)-[:SUBMITTED_BY]->(s)
        """,
        {"who": row["Who_completed_the_test"], "upload_id": upload_id}
    ))

    with neo4j_service.session() as session:
        for query, params in queries:
            session.run(query, **params)
        logger.info(f"Successfully inserted case {upload_id}")

@safe_neo4j_operation
def extract_user_embedding(upload_id: str) -> Optional[np.ndarray]:
    with neo4j_service.session() as session:
        result = session.run(
            "MATCH (c:Case {upload_id: $upload_id}) RETURN c.embedding AS embedding",
            upload_id=upload_id
        )
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
    logger.info(f"Extracted {len(X)} training samples")
    return pd.DataFrame(X), pd.Series(y)

@safe_neo4j_operation
def get_existing_embeddings() -> Optional[np.ndarray]:
    with neo4j_service.session() as session:
        result = session.run("""
            MATCH (c:Case)
            WHERE c.embedding IS NOT NULL
            RETURN c.embedding AS embedding
        """)
        embeddings = [record["embedding"] for record in result]
        return np.array(embeddings) if embeddings else None

@st.cache_resource(show_spinner="Training ASD detection model...")
def train_asd_detection_model() -> Optional[RandomForestClassifier]:
    X, y = extract_training_data()
    if X.empty:
        st.warning("No training data available")
        return None

    st.subheader("📊 Class Distribution")
    st.write(Counter(y))

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=Config.TEST_SIZE,
        stratify=y,
        random_state=Config.RANDOM_STATE
    )

    pipeline = Pipeline([
        ('smote', SMOTE(random_state=Config.RANDOM_STATE, sampling_strategy='auto')),
        ('classifier', RandomForestClassifier(
            n_estimators=Config.N_ESTIMATORS,
            random_state=Config.RANDOM_STATE,
            class_weight='balanced'
        ))
    ])

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    st.subheader("📈 Model Evaluation")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("ROC AUC", f"{roc_auc_score(y_test, y_proba):.3f}")
        st.metric("Average Precision", f"{average_precision_score(y_test, y_proba):.3f}")
    with col2:
        st.metric("F1 Score", f"{classification_report(y_test, y_pred, output_dict=True)['1']['f1-score']:.3f}")
        st.metric("Accuracy", f"{classification_report(y_test, y_pred, output_dict=True)['accuracy']:.3f}")

    return pipeline.named_steps['classifier']

@st.cache_resource(show_spinner="Training Isolation Forest...")
def train_isolation_forest() -> Optional[Tuple[IsolationForest, StandardScaler]]:
    embeddings = get_existing_embeddings()
    if embeddings is None or len(embeddings) < Config.MIN_CASES_FOR_ANOMALY_DETECTION:
        st.warning(f"Anomaly detection requires at least {Config.MIN_CASES_FOR_ANOMALY_DETECTION} existing cases.")
        return None

    scaler = StandardScaler()
    embeddings_scaled = scaler.fit_transform(embeddings)

    contamination_rate = 0.05 if len(embeddings) < 50 else 0.1

    iso_forest = IsolationForest(
        random_state=Config.RANDOM_STATE,
        contamination=contamination_rate
    )
    iso_forest.fit(embeddings_scaled)

    return iso_forest, scaler

# === Streamlit UI ===
st.title("🧠 NeuroCypher ASD")

# === File Upload Section ===
st.header("📄 Upload New Case")
uploaded_file = st.file_uploader("Upload CSV for single child prediction", type="csv")

def validate_csv(df: pd.DataFrame) -> bool:
    required_columns = [f"A{i}" for i in range(1, 11)] + [
        "Sex", "Ethnicity", "Jaundice",
        "Family_mem_with_ASD", "Who_completed_the_test"
    ]
    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        st.error(f"Missing required columns: {', '.join(missing_cols)}")
        return False
    return True

if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file, delimiter=";")

        if not validate_csv(df):
            st.stop()

        if len(df) != 1:
            st.error("Please upload exactly one row (one child)")
            st.stop()

        st.subheader("👀 CSV Preview")
        st.dataframe(df.T)

        row = df.iloc[0]
        upload_id = str(uuid.uuid4())

        with st.spinner("Inserting case into graph..."):
            insert_user_case(row, upload_id)

        with st.spinner("Verifying embedding..."):
            embedding = extract_user_embedding(upload_id)
            if embedding is None:
                st.error("Failed to generate embedding")
                st.stop()

            st.subheader("🧠 Case Embedding")
            st.write(embedding)

    except Exception as e:
        st.error(f"❌ Error processing file: {e}")
        logger.error(f"❌ Exception during upload processing: {e}")
