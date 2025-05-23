import streamlit as st
from neo4j import GraphDatabase
from openai import OpenAI
from dotenv import load_dotenv
import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_predict
from sklearn.metrics import (
    roc_auc_score, roc_curve, precision_recall_curve,
    average_precision_score, classification_report, 
    precision_score, recall_score, f1_score, confusion_matrix
)
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import make_pipeline as make_imb_pipeline
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
from node2vec import Node2Vec
import networkx as nx
import tempfile
import shutil
import seaborn as sns
from sklearn.inspection import permutation_importance
import subprocess
import sys
from sklearn.impute import SimpleImputer
from xgboost import XGBClassifier
from imblearn.pipeline import Pipeline as ImbPipeline
import json

def call_embedding_generator(upload_id: str) -> bool:
    """Generate embedding for a single case using subprocess with enhanced error handling"""
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        builder_path = os.path.join(script_dir, "generate_case_embedding.py")
        
        if not os.path.exists(builder_path):
            st.error(f"❌ Embedding generator script not found at: {builder_path}")
            return False

        # Prepare environment variables
        env = os.environ.copy()
        env.update({
            "NEO4J_URI": os.getenv("NEO4J_URI"),
            "NEO4J_USER": os.getenv("NEO4J_USER"),
            "NEO4J_PASSWORD": os.getenv("NEO4J_PASSWORD"),
            "PYTHONPATH": os.path.dirname(script_dir)
        })

        # Run the process with timeout
        result = subprocess.run(
            [sys.executable, builder_path, upload_id],
            env=env,
            capture_output=True,
            text=True,
            timeout=Config.EMBEDDING_GENERATION_TIMEOUT
        )
        
        if result.returncode != 0:
            error_msg = result.stderr or "Unknown error (no stderr output)"
            st.error(f"❌ Embedding generation failed with error:\n{error_msg}")
            logger.error(f"Embedding generation failed for {upload_id}: {error_msg}")
            return False
            
        return True
        
    except subprocess.TimeoutExpired:
        st.error("❌ Embedding generation timed out")
        logger.error(f"Embedding generation timeout for {upload_id}")
        return False
    except Exception as e:
        st.error(f"❌ Fatal error calling embedding script: {str(e)}")
        logger.exception(f"Fatal error generating embedding for {upload_id}")
        return False
        
# === Configuration ===
class Config:
    EMBEDDING_DIM = 128
    RANDOM_STATE = np.random.randint(0, 1000)
    TEST_SIZE = 0.3
    N_ESTIMATORS = 100
    SMOTE_RATIO = 'auto'
    MIN_CASES_FOR_ANOMALY_DETECTION = 10
    NODE2VEC_WALK_LENGTH = 20
    NODE2VEC_NUM_WALKS = 100
    NODE2VEC_WORKERS = 2
    NODE2VEC_P = 1
    NODE2VEC_Q = 1
    EMBEDDING_BATCH_SIZE = 50
    MAX_RELATIONSHIPS = 100000
    EMBEDDING_GENERATION_TIMEOUT = 300
    LEAKAGE_CHECK = True
    SMOTE_K_NEIGHBORS = 5  # Added SMOTE configuration

# === Logging Setup ===
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# === Environment Setup ===
load_dotenv()

# Validate Environment Variables
required_env_vars = ["NEO4J_URI", "NEO4J_USER", "NEO4J_PASSWORD", "OPENAI_API_KEY"]
missing_vars = [var for var in required_env_vars if not os.getenv(var)]
if missing_vars:
    st.error(f"Missing required environment variables: {', '.join(missing_vars)}")
    st.stop()

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
    if 'neo4j_driver' in st.session_state:
        st.session_state.neo4j_driver.close()
    return Neo4jService(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)

@st.cache_resource
def get_openai_client():
    return OpenAI(api_key=OPENAI_API_KEY)

neo4j_service = get_neo4j_service()
client = get_openai_client()

# === Helper Functions ===
def safe_neo4j_operation(func):
    """Decorator for Neo4j operations with error handling"""
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            st.error(f"Neo4j operation failed: {str(e)}")
            logger.error(f"Neo4j operation failed: {str(e)}")
            return None
    return wrapper

@safe_neo4j_operation
def check_embedding_dimensions():
    with neo4j_service.session() as session:
        result = session.run("""
            MATCH (c:Case) WHERE c.embedding IS NOT NULL
            RETURN c.id AS case_id, size(c.embedding) AS embedding_length
        """)
        wrong_dims = [(r["case_id"], r["embedding_length"]) for r in result if r["embedding_length"] != 128]
        if wrong_dims:
            st.warning(f"⚠️ Cases with wrong embedding size: {wrong_dims}")
        else:
            st.success("✅ All embeddings have correct size (128).")

@safe_neo4j_operation
def find_cases_missing_labels() -> list:
    with neo4j_service.session() as session:
        result = session.run("""
            MATCH (c:Case)
            WHERE NOT (c)-[:SCREENED_FOR]->(:ASD_Trait)
            RETURN c.id AS case_id
        """)
        missing_cases = [record["case_id"] for record in result]
        if missing_cases:
            st.warning(f"⚠️ Cases missing SCREENED_FOR label: {missing_cases}")
        else:
            st.success("✅ All cases have SCREENED_FOR labels.")
        return missing_cases

@safe_neo4j_operation
def refresh_screened_for_labels(csv_url: str):
    """
    Atomically removes all existing SCREENED_FOR relationships
    και ξαναδημιουργεί τις σωστές βάσει του CSV.
    """

    df = pd.read_csv(csv_url, delimiter=";", encoding='utf-8-sig')
    df.columns = [col.strip() for col in df.columns]

    if "Case_No" not in df.columns or "Class_ASD_Traits" not in df.columns:
        st.error("❌ Το CSV πρέπει να περιέχει τις στήλες 'Case_No' και 'Class_ASD_Traits'")
        return

    with neo4j_service.session() as session:
        # Διαγραφή όλων των σχέσεων
        session.run("""
            MATCH (c:Case)-[r:SCREENED_FOR]->(:ASD_Trait)
            DELETE r
        """)
        logger.info("✅ Παλιές σχέσεις SCREENED_FOR διαγράφηκαν.")

        # Επανεγγραφή νέων σχέσεων με batch
        tx = session.begin_transaction()

        for _, row in df.iterrows():
            case_id = int(row["Case_No"])
            label = str(row["Class_ASD_Traits"]).strip().capitalize()  # "Yes" ή "No"

            if label in ["Yes", "No"]:
                tx.run("""
                    MATCH (c:Case {id: $case_id})
                    MERGE (t:ASD_Trait {label: $label})
                    MERGE (c)-[:SCREENED_FOR]->(t)
                """, case_id=case_id, label=label)

        tx.commit()
        logger.info("✅ Νέες σχέσεις SCREENED_FOR δημιουργήθηκαν με βάση το CSV.")
        st.success("✅ Ολοκληρώθηκε η ανανέωση των σχέσεων SCREENED_FOR.")        

# === Data Insertion ===
@safe_neo4j_operation
def insert_user_case(row: pd.Series, upload_id: str) -> str:
    queries = []

    queries.append((
        "MERGE (c:Case {upload_id: $upload_id}) SET c.id = $id, c.embedding = null",
        {"upload_id": upload_id, "id": int(row["Case_No"])}
    ))

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

    # Add ASD_Trait relationship if Class_ASD_Traits exists
    if "Class_ASD_Traits" in row:
        label = str(row["Class_ASD_Traits"]).strip().lower()
        if label in ["yes", "no"]:
         
            queries.append((
                """
                MATCH (c:Case {id: $id})
                MERGE (t:ASD_Trait {label: $label})
                MERGE (c)-[:SCREENED_FOR]->(t)
                """,
                {"id": int(row["Case_No"]), "label": label.capitalize()}
            ))

    with neo4j_service.session() as session:
        for query, params in queries:
            session.run(query, **params)
        logger.info(f"✅ Inserted new case with upload_id {upload_id}")

    return upload_id

@safe_neo4j_operation
def remove_screened_for_labels():
    with neo4j_service.session() as session:
        session.run("""
            MATCH (c:Case)-[r:SCREENED_FOR]->(:ASD_Trait)
            DELETE r
        """)
        logger.info("✅ SCREENED_FOR relationships removed to prevent leakage.")

# === Graph Embeddings Generation ===
@safe_neo4j_operation
def generate_embedding_for_case(upload_id: str) -> bool:
    """Generate embedding for a single case using subprocess"""
    try:
        # Get the directory of the current script
        script_dir = os.path.dirname(os.path.abspath(__file__))
        builder_path = os.path.join(script_dir, "generate_case_embedding.py")
        
        if not os.path.exists(builder_path):
            st.error(f"❌ Embedding generator script not found at: {builder_path}")
            return False

        # Run the embedding generator as a subprocess
        result = subprocess.run(
            [sys.executable, builder_path, upload_id],
            capture_output=True,
            text=True
        )
        
        if result.returncode != 0:
            st.error(f"❌ Embedding generation failed with error:\n{result.stderr}")
            return False
            
        return True
        
    except Exception as e:
        st.error(f"❌ Error generating embedding: {str(e)}")
        return False

# === Natural Language to Cypher ===
def nl_to_cypher(question: str) -> Optional[str]:
    """Translates natural language to Cypher using OpenAI"""
    prompt = f"""
    You are a Cypher expert working with a Neo4j Knowledge Graph about toddlers and autism.

    Schema:
    - (:Case {{id: int}})
    - (:BehaviorQuestion {{name: string}})
    - (:ASD_Trait {{value: 'Yes' | 'No'}})
    - (:DemographicAttribute {{type: 'Sex' | 'Ethnicity' | 'Jaundice' | 'Family_mem_with_ASD', value: string}})
    - (:SubmitterType {{type: string}})

    Relationships:
    - (:Case)-[:HAS_ANSWER {{value: int}}]->(:BehaviorQuestion)
    - (:Case)-[:HAS_DEMOGRAPHIC]->(:DemographicAttribute)
    - (:Case)-[:SCREENED_FOR]->(:ASD_Trait)
    - (:Case)-[:SUBMITTED_BY]->(:SubmitterType)

    Rules:
    1. Always use `toLower()` for case-insensitive comparisons
    2. Interpret 'f' as 'female' and 'm' as 'male' for Sex
    3. Never use SCREENED_FOR relationships in training queries

    Translate this question to Cypher:
    Q: {question}

    Return ONLY the Cypher query.
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

# === Embedding Extraction ===
@safe_neo4j_operation
def extract_user_embedding(upload_id: str) -> Optional[np.ndarray]:
    """Safely extracts embedding for a specific case"""
    with neo4j_service.session() as session:
        result = session.run(
            "MATCH (c:Case {upload_id: $upload_id}) RETURN c.embedding AS embedding",
            upload_id=upload_id
        )
        record = result.single()
        
        if record and record["embedding"] is not None:
            return np.array(record["embedding"]).reshape(1, -1)
        
        exists = session.run(
            "MATCH (c:Case {upload_id: $upload_id}) RETURN count(c) > 0 AS exists",
            upload_id=upload_id
        ).single()["exists"]
        
        if not exists:
            st.error(f"❌ Case with upload_id {upload_id} not found")
        else:
            st.error(f"❌ No embedding found for case {upload_id}. Please regenerate embeddings.")
        
        return None

# === Training Data Preparation ===
@safe_neo4j_operation
def check_label_consistency(df: pd.DataFrame, neo4j_service) -> None:
    inconsistent_cases = []
    with neo4j_service.session() as session:
        for _, row in df.iterrows():
            case_id = int(row["Case_No"])
            csv_label = str(row["Class_ASD_Traits"]).strip().lower()

            record = session.run("""
                MATCH (c:Case {id: $case_id})-[r:SCREENED_FOR]->(t:ASD_Trait)
                RETURN t.label AS graph_label
            """, case_id=case_id).single()

            graph_label = record["graph_label"].strip().lower() if record and record["graph_label"] else None

            if graph_label is None:
                st.warning(f"⚠️ Case_No {case_id} έχει ετικέτα '{csv_label}' στο CSV, αλλά δεν έχει ετικέτα στον γράφο. Δημιουργία σχέσης...")
                session.run("""
                    MATCH (c:Case {id: $case_id})
                    MERGE (t:ASD_Trait {label: $label})
                    MERGE (c)-[:SCREENED_FOR]->(t)
                """, case_id=case_id, label=csv_label.capitalize())
            elif graph_label != csv_label:
                inconsistent_cases.append((case_id, csv_label, graph_label))

    if inconsistent_cases:
        st.error("❌ Βρέθηκαν ασυμφωνίες μεταξύ CSV και Neo4j ετικετών (Class_ASD_Traits vs SCREENED_FOR):")
        for case_id, csv_label, graph_label in inconsistent_cases:
            st.error(f"- Case_No {case_id}: CSV='{csv_label}' | Neo4j='{graph_label}'")
        st.stop()

@safe_neo4j_operation
def extract_training_data_from_csv(file_path: str) -> Tuple[pd.DataFrame, pd.Series]:
    try:
        df = pd.read_csv(file_path, delimiter=";", encoding='utf-8-sig')
        df.columns = [col.strip().replace('\r', '') for col in df.columns]
        df.columns = [col.strip() for col in df.columns]

        # Έλεγχος συνέπειας ετικετών CSV με γράφο
        check_label_consistency(df, neo4j_service)

        required_cols = ["Case_No", "Class_ASD_Traits"]
        missing = [col for col in required_cols if col not in df.columns]
        if missing:
            st.error(f"❌ Missing required columns: {', '.join(missing)}")
            st.write("📋 Found columns in CSV:", df.columns.tolist())
            return pd.DataFrame(), pd.Series()

        numeric_cols = [f"A{i}" for i in range(1, 11)] + ["Case_No"]
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        embeddings = []
        valid_ids = []
        with neo4j_service.session() as session:
            for case_no in df["Case_No"]:
                result = session.run("""
                    MATCH (c:Case {id: $id})
                    WHERE c.embedding IS NOT NULL
                    RETURN c.embedding AS embedding
                """, id=int(case_no))
                record = result.single()
                if record and record["embedding"]:
                    embeddings.append(record["embedding"])
                    valid_ids.append(case_no)

        df_filtered = df[df["Case_No"].isin(valid_ids)].copy()
        y = df_filtered["Class_ASD_Traits"].apply(
            lambda x: 1 if str(x).strip().lower() == "yes" else 0
        )

        assert len(embeddings) == len(y), f"⚠️ Embeddings: {len(embeddings)}, Labels: {len(y)}"

        X = pd.DataFrame(embeddings[:len(y)])

        if X.isna().any().any():
            st.warning(f"⚠️ Found {X.isna().sum().sum()} NaN values in embeddings - applying imputation")
            X = X.fillna(X.mean())

        return X, y

    except Exception as e:
        st.error(f"Data extraction failed: {str(e)}")
        return pd.DataFrame(), pd.Series()
        
@safe_neo4j_operation
def extract_training_data_from_csv(file_path: str) -> Tuple[pd.DataFrame, pd.Series]:
    """Extracts training data with leakage protection and NaN handling"""
    try:
        df = pd.read_csv(file_path, delimiter=";", encoding='utf-8-sig')
        df.columns = [col.strip().replace('\r', '') for col in df.columns]
        df.columns = [col.strip() for col in df.columns]
        check_label_consistency(df, neo4j_service)

        required_cols = ["Case_No", "Class_ASD_Traits"]
        missing = [col for col in required_cols if col not in df.columns]
        if missing:
            st.error(f"❌ Missing required columns: {', '.join(missing)}")
            st.write("📋 Found columns in CSV:", df.columns.tolist())
            return pd.DataFrame(), pd.Series()

        numeric_cols = [f"A{i}" for i in range(1, 11)] + ["Case_No"]
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        embeddings = []
        valid_ids = []
        with neo4j_service.session() as session:
            for case_no in df["Case_No"]:
                result = session.run("""
                    MATCH (c:Case {id: $id})
                    WHERE c.embedding IS NOT NULL
                    RETURN c.embedding AS embedding
                """, id=int(case_no))
                record = result.single()
                if record and record["embedding"]:
                    embeddings.append(record["embedding"])
                    valid_ids.append(case_no)

        df_filtered = df[df["Case_No"].isin(valid_ids)].copy()
        y = df_filtered["Class_ASD_Traits"].apply(
            lambda x: 1 if str(x).strip().lower() == "yes" else 0
        )

        assert len(embeddings) == len(y), f"⚠️ Embeddings: {len(embeddings)}, Labels: {len(y)}"

        X = pd.DataFrame(embeddings[:len(y)])

        if X.isna().any().any():
            st.warning(f"⚠️ Found {X.isna().sum().sum()} NaN values in embeddings - applying imputation")
            X = X.fillna(X.mean())

        return X, y

    except Exception as e:
        st.error(f"Data extraction failed: {str(e)}")
        return pd.DataFrame(), pd.Series()

# === Model Evaluation ===
def analyze_embedding_correlations(X: pd.DataFrame, csv_url: str):
    st.subheader("📌 Feature–Embedding Correlation Analysis")
    try:
        df = pd.read_csv(csv_url, delimiter=";", encoding='utf-8-sig')
        df.columns = [col.strip() for col in df.columns]

        if "Case_No" not in df.columns:
            st.error("Το αρχείο πρέπει να περιέχει στήλη 'Case_No'")
            return

        features = [f"A{i}" for i in range(1, 11)] + ["Sex", "Ethnicity", "Jaundice", "Family_mem_with_ASD"]
        df = df[features]
        df = pd.get_dummies(df, drop_first=True)

        if df.shape[0] != X.shape[0]:
            df = df.iloc[:X.shape[0]]

        corr = pd.DataFrame(index=df.columns, columns=X.columns)

        for feat in df.columns:
            for dim in X.columns:
                corr.at[feat, dim] = np.corrcoef(df[feat], X[dim])[0, 1]

        corr = corr.astype(float)

        fig, ax = plt.subplots(figsize=(12, 6))
        sns.heatmap(corr, cmap="coolwarm", center=0, annot=False)
        ax.set_title("Συσχέτιση Χαρακτηριστικών με Embedding Διαστάσεις")
        st.pyplot(fig)

    except Exception as e:
        st.error(f"❌ Correlation analysis failed: {str(e)}")

def plot_combined_curves(y_true, y_proba):
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))

    fpr, tpr, _ = roc_curve(y_true, y_proba)
    ax[0].plot(fpr, tpr, label=f"AUC = {roc_auc_score(y_true, y_proba):.3f}")
    ax[0].plot([0, 1], [0, 1], 'k--')
    ax[0].set_xlabel("False Positive Rate")
    ax[0].set_ylabel("True Positive Rate")
    ax[0].set_title("ROC Curve")
    ax[0].legend()

    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    ax[1].plot(recall, precision, label=f"AP = {average_precision_score(y_true, y_proba):.3f}")
    ax[1].set_xlabel("Recall")
    ax[1].set_ylabel("Precision")
    ax[1].set_title("Precision-Recall Curve")
    ax[1].legend()

    st.pyplot(fig)

def show_permutation_importance(model, X_test, y_test):
    try:
        result = permutation_importance(model, X_test, y_test, n_repeats=10, random_state=42)
        importance_df = pd.DataFrame({
            "Feature": [f"Dim_{i}" for i in range(X_test.shape[1])],
            "Importance": result.importances_mean
        }).sort_values(by="Importance", ascending=False)

        st.subheader("📊 Permutation Importance")
        st.bar_chart(importance_df.set_index("Feature").head(15))
    except Exception as e:
        st.warning(f"Could not calculate permutation importance: {str(e)}")

def evaluate_model(model, X_test, y_test):
    """Comprehensive model evaluation"""
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    st.subheader("📉 probability distribution forecast")
    fig, ax = plt.subplots()
    ax.hist(y_proba, bins=20, color='skyblue', edgecolor='black')
    ax.set_xlabel("ASD Traits probability")
    ax.set_ylabel("No of cases")
    st.pyplot(fig)

    if roc_auc_score(y_test, y_proba) > 0.98:
        st.warning("""
        🚨 Suspiciously high performance detected. Possible causes:
        1. Data leakage in embeddings
        2. Test set contains training data
        3. Label contamination in graph
        """)

    st.subheader("📊 Model Evaluation Metrics")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("ROC AUC", f"{roc_auc_score(y_test, y_proba):.3f}")
        st.metric("Precision", f"{precision_score(y_test, y_pred):.3f}")
    with col2:
        st.metric("Recall", f"{recall_score(y_test, y_pred):.3f}")
        st.metric("F1 Score", f"{f1_score(y_test, y_pred):.3f}")

    st.subheader("Confusion Matrix")
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    st.pyplot(fig)

    st.subheader("📈 Performance Curves")
    plot_combined_curves(y_test, y_proba)

    show_permutation_importance(model, X_test, y_test)

    csv_url = "https://raw.githubusercontent.com/GiorgosBouh/llm2cypher-asd-app/main/Toddler_Autism_dataset_July_2018_2.csv"
    analyze_embedding_correlations(X_test, csv_url)


# === Model Training ===
@st.cache_resource(show_spinner="Training ASD detection model...")
def train_asd_detection_model(cache_key: str) -> Optional[dict]:
    try:
        csv_url = "https://raw.githubusercontent.com/GiorgosBouh/llm2cypher-asd-app/main/Toddler_Autism_dataset_July_2018_2.csv"

        # 1. Αφαίρεση SCREENED_FOR σχέσεων για αποφυγή leakage
        remove_screened_for_labels()

        # 2. Αναγέννηση embeddings χωρίς τις ετικέτες (labels)
        result = subprocess.run(
            [sys.executable, "kg_builder_2.py"],
            capture_output=True,
            text=True,
            check=True
        )
        if result.returncode != 0:
            st.error(f"❌ Failed to generate embeddings:\n{result.stderr}")
            return None

        # 3. Φόρτωση embeddings και labels από CSV (χωρίς leakage)
        X_raw, y = extract_training_data_from_csv(csv_url)
        X = X_raw.copy()
        X.columns = [f"Dim_{i}" for i in range(X.shape[1])]

        if X.empty or y.empty:
            st.error("⚠️ No valid training data available")
            return None

        # 4. Train/test split (προστασία από leakage)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=Config.TEST_SIZE,
            stratify=y,
            random_state=Config.RANDOM_STATE
        )

        # 5. Υπολογισμός βαρών κλάσεων
        neg = sum(y_train == 0)
        pos = sum(y_train == 1)
        scale_pos_weight = neg / pos if pos != 0 else 1

        # 6. Pipeline με SMOTE και XGBoost
        pipeline = ImbPipeline([
            ('smote', SMOTE(
                sampling_strategy='auto',
                k_neighbors=min(Config.SMOTE_K_NEIGHBORS, pos - 1),
                random_state=Config.RANDOM_STATE
            )),
            ('xgb', XGBClassifier(
                n_estimators=Config.N_ESTIMATORS,
                use_label_encoder=False,
                eval_metric='logloss',
                random_state=Config.RANDOM_STATE,
                scale_pos_weight=scale_pos_weight
            ))
        ])

        # 7. Cross-validation για να ελεγχθεί η απόδοση χωρίς leakage
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=Config.RANDOM_STATE)
        y_proba = cross_val_predict(
            pipeline, X_train, y_train,
            cv=cv,
            method='predict_proba',
            n_jobs=-1
        )[:, 1]

        # 8. Εκπαίδευση στο πλήρες training set
        pipeline.fit(X_train, y_train)

        # 9. Αξιολόγηση στο test set
        test_proba = pipeline.predict_proba(X_test)[:, 1]
        test_pred = pipeline.predict(X_test)

        st.subheader("📊 Cross-Validation Results (Training Set)")
        st.write(f"Mean CV ROC AUC: {roc_auc_score(y_train, y_proba):.3f}")
        
        st.subheader("📊 Test Set Results")
        st.write(f"Test ROC AUC: {roc_auc_score(y_test, test_proba):.3f}")

        # 10. Επανέφερε τις SCREENED_FOR ετικέτες στον γράφο (μετά το training)
        reinsert_labels_from_csv(csv_url)

        return {
            "model": pipeline,
            "X_test": X_test,
            "y_test": y_test
        }

    except subprocess.CalledProcessError as cpe:
        st.error(f"❌ Subprocess failed: {cpe.stderr}")
        logger.error(f"Subprocess error during embedding generation: {cpe.stderr}")
        return None
    except Exception as e:
        st.error(f"❌ Error training model: {e}")
        logger.error(f"Training error: {e}", exc_info=True)
        return None

# === Anomaly Detection ===
@safe_neo4j_operation
def get_existing_embeddings() -> Optional[np.ndarray]:
    """Returns all case embeddings for anomaly detection"""
    with neo4j_service.session() as session:
        result = session.run("""
            MATCH (c:Case)
            WHERE c.embedding IS NOT NULL
            RETURN c.embedding AS embedding
        """)
        embeddings = [record["embedding"] for record in result]
        return np.array(embeddings) if embeddings else None

@st.cache_resource(show_spinner="Training Isolation Forest...")
def train_isolation_forest(cache_key: str) -> Optional[Tuple[IsolationForest, StandardScaler]]:
    """Trains anomaly detection model"""
    embeddings = get_existing_embeddings()
    if embeddings is None or len(embeddings) < Config.MIN_CASES_FOR_ANOMALY_DETECTION:
        st.warning(f"⚠️ Need at least {Config.MIN_CASES_FOR_ANOMALY_DETECTION} cases for anomaly detection")
        return None

    scaler = StandardScaler()
    embeddings_scaled = scaler.fit_transform(embeddings)

    contamination = min(0.1, 5.0 / len(embeddings))
    iso_forest = IsolationForest(
        contamination=contamination,
        random_state=Config.RANDOM_STATE
    )
    iso_forest.fit(embeddings_scaled)

    return iso_forest, scaler

@safe_neo4j_operation
def reinsert_labels_from_csv(csv_url: str):
    df = pd.read_csv(csv_url, delimiter=";", encoding='utf-8-sig')
    df.columns = [col.strip() for col in df.columns]

    if "Case_No" not in df.columns or "Class_ASD_Traits" not in df.columns:
        st.error("❌ Το CSV δεν περιέχει τις στήλες 'Case_No' και 'Class_ASD_Traits'")
        return

    with neo4j_service.session() as session:
        for _, row in df.iterrows():
            case_id = int(row["Case_No"])
            label = str(row["Class_ASD_Traits"]).strip().lower()
            if label in ["yes", "no"]:
                session.run("""
                    MATCH (c:Case {id: $case_id})
                    MERGE (t:ASD_Trait {label: $label})
                    MERGE (c)-[:SCREENED_FOR]->(t)
                """, case_id=case_id, label=label.capitalize())

# === Streamlit UI ===
def main():
    st.title("🧠 NeuroCypher ASD")
    st.markdown("""
        <i>Autism Spectrum Disorder detection using graph embeddings</i>
        """, unsafe_allow_html=True)

    st.sidebar.markdown(f"🔗 Connected to: `{os.getenv('NEO4J_URI')}`")
    st.sidebar.markdown("""
---
### 📘 About This Project

This project was developed by [Dr. Georgios Bouchouras](https://giorgosbouh.github.io/github-portfolio/), in collaboration with Dimitrios Doumanas MSc, and Dr. Konstantinos Kotis  
at the [Intelligent Systems Research Laboratory (i-Lab), University of the Aegean](https://i-lab.aegean.gr/).

It is part of the postdoctoral research project:

**"Development of Intelligent Systems for the Early Detection and Management of Developmental Disorders: Combining Biomechanics and Artificial Intelligence"**  
by Dr. Bouchouras under the supervision of Dr. Kotis.

---
### 🧪 What This App Does

This interactive application functions as a GraphRAG-powered intelligent agent designed for the early 
detection of Autism Spectrum Disorder traits in toddlers. It integrates a Neo4j knowledge graph, 
machine learning, and natural language interfaces powered by GPT-4. The app allows you to:

- 🧠 Train a machine learning model to detect ASD traits using graph embeddings
- 📤 Upload your own toddler screening data from the Q-Chat-10 questionnaire and other demographics
- 🔗 Automatically connect the uploaded case to a knowledge graph
- 🌐 Generate a graph-based embedding for the new case
- 🔍 Predict whether the case shows signs of Autism Spectrum Disorder (ASD)
- 🕵️ Run anomaly detection to check for anomalies
- 💬 Ask natural language questions and receive Cypher queries with results, using GPT4 based NLP-to-Cypher translation

---
### 📥 Download Example CSV

To get started, [download this example CSV](https://raw.githubusercontent.com/GiorgosBouh/llm2cypher-asd-app/main/Toddler_Autism_dataset_July_2018_3_test_39.csv)  
to format your own screening case correctly. 
Also, [read this description](https://raw.githubusercontent.com/GiorgosBouh/llm2cypher-asd-app/main/Toddler_data_description.docx) for further information about the dataset.
""")

    # Initialize session state variables safely
    if "active_tab" not in st.session_state:
        st.session_state.active_tab = "Model Training"
    if "case_inserted" not in st.session_state:
        st.session_state.case_inserted = False
    if "last_upload_id" not in st.session_state:
        st.session_state.last_upload_id = None
    if "last_case_no" not in st.session_state:
        st.session_state.last_case_no = None
    if "model_trained" not in st.session_state:
        st.session_state.model_trained = False
    if "model_results" not in st.session_state:
        st.session_state.model_results = None
    if "saved_embedding_case1" not in st.session_state:
        st.session_state.saved_embedding_case1 = None
    if "last_cypher_query" not in st.session_state:
        st.session_state.last_cypher_query = None
    if "last_cypher_results" not in st.session_state:
        st.session_state.last_cypher_results = None
    if "preset_question" not in st.session_state:
        st.session_state.preset_question = ""

    # Create tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Model Training",
        "🌐 Graph Embeddings",
        "📤 Upload New Case",
        "💬 NLP to Cypher"
    ])

    # === Tab 1: Model Training ===
    with tab1:
        st.header("🤖 ASD Detection Model")

        missing_labels = find_cases_missing_labels()
        if missing_labels:
            st.warning(f"⚠️ There are {len(missing_labels)} cases without SCREENED_FOR label. Please fix the data before proceeding.")
        else:
            st.success("✅ All cases have SCREENED_FOR labels.")

        if st.button("🔄 Train/Refresh Model"):
            with st.spinner("Training model with leakage protection..."):
                results = train_asd_detection_model(cache_key=str(uuid.uuid4()))
                if results:
                    st.session_state.model_results = results
                    st.session_state.model_trained = True
                    st.success("✅ Training completed successfully.")
                    evaluate_model(
                        results["model"],
                        results["X_test"],
                        results["y_test"]
                    )

        if st.session_state.get("model_trained") and st.session_state.get("model_results"):
            evaluate_model(
                st.session_state.model_results["model"],
                st.session_state.model_results["X_test"],
                st.session_state.model_results["y_test"]
            )

            with st.expander("🧪 Compare old vs new embeddings (Case 1)"):
                if st.button("📤 Save current embedding of Case 1"):
                    with neo4j_service.session() as session:
                        result = session.run("MATCH (c:Case {id: 1}) RETURN c.embedding AS emb").single()
                        if result and result["emb"]:
                            st.session_state.saved_embedding_case1 = result["emb"]
                            st.success("✅ Saved current embedding of Case 1")

                if st.button("📥 Compare to current embedding of Case 1"):
                    with neo4j_service.session() as session:
                        result = session.run("MATCH (c:Case {id: 1}) RETURN c.embedding AS emb").single()
                        if result and result["emb"]:
                            new_emb = result["emb"]
                            old_emb = st.session_state.saved_embedding_case1
                            if old_emb:
                                from numpy.linalg import norm
                                diff = norm(np.array(old_emb) - np.array(new_emb))
                                st.write(f"📏 Difference (L2 norm) between saved and current embedding: `{diff:.4f}`")
                                if diff < 1e-3:
                                    st.warning("⚠️ Embedding is (almost) identical — rebuild had no effect.")
                                else:
                                    st.success("✅ Embedding changed — rebuild updated the graph.")
                            else:
                                st.error("❌ No saved embedding found. Click 'Save current embedding' first.")

    # === Tab 2: Graph Embeddings ===
    with tab2:
        st.header("🌐 Graph Embeddings")
        st.warning("⚠️ Don't push this button unless you are the developer!")
        st.info("ℹ️ This function is for the developer only")
        if st.button("🔁 Recalculate All Embeddings"):
            with st.spinner("Running full graph rebuild and embedding generation..."):
                result = subprocess.run(
                    [sys.executable, "kg_builder_2.py"],
                    capture_output=True,
                    text=True
                )
                if result.returncode == 0:
                    st.success("✅ Embeddings recalculated and updated in the graph!")
                else:
                    st.error("❌ Failed to run kg_builder_2.py")
                    st.code(result.stderr)

    # === Tab 3: Upload New Case ===
    with tab3:
        st.header("📄 Upload New Case (Prediction Only - No Graph Storage)")

        # ========== INSTRUCTIONS SECTION ==========
        with st.container(border=True):
            st.subheader("📝 Before You Upload", anchor=False)

            cols = st.columns([1, 3])
            with cols[0]:
                st.image("https://cdn-icons-png.flaticon.com/512/2965/2965300.png", width=80)
            with cols[1]:
                st.markdown("""
                **Please follow these steps carefully:**
                1. Download the example CSV template
                2. Review the data format instructions
                3. Prepare your case data accordingly
                """)

            st.markdown("---")
            st.markdown("### 🛠️ Required Resources")

            col1, col2 = st.columns(2)
            with col1:
                with st.container(border=True):
                    st.markdown("**📥 Example CSV Template**")
                    st.markdown("Download and use this template to format your data:")
                    st.markdown("[Download Example CSV](https://raw.githubusercontent.com/GiorgosBouh/llm2cypher-asd-app/main/Toddler_Autism_dataset_July_2018_3_test_39.csv)")

            with col2:
                with st.container(border=True):
                    st.markdown("**📋 Data Format Instructions**")
                    st.markdown("Read the detailed documentation:")
                    st.markdown("[View Instructions Document](https://raw.githubusercontent.com/GiorgosBouh/llm2cypher-asd-app/main/Toddler_data_description.docx)")

            st.markdown("---")
            st.markdown("""
            **❗ Important Notes:**
            - Ensure all required columns are present
            - Upload exactly one case at a time
            - Values must match the specified formats
            - This upload **does NOT** save data to the graph — prediction only!
            """)
        # ========== END INSTRUCTIONS SECTION ==========

        uploaded_file = st.file_uploader(
            "**Upload your prepared CSV file**",
            type="csv",
            key="case_uploader",
            help="Ensure your file follows the template format before uploading"
        )

        if uploaded_file:
            try:
                df = pd.read_csv(uploaded_file, delimiter=";")
                st.subheader("📊 Uploaded CSV Preview")
                st.dataframe(
                    df.style.format(
                        {
                            "Case_No": "{:.0f}",
                            **{f"A{i}": "{:.0f}" for i in range(1, 11)}
                        }
                    ),
                    use_container_width=True,
                    hide_index=True
                )

                required_cols = [
                    "Case_No", "A1", "A2", "A3", "A4", "A5",
                    "A6", "A7", "A8", "A9", "A10",
                    "Sex", "Ethnicity", "Jaundice",
                    "Family_mem_with_ASD", "Who_completed_the_test"
                ]

                # Check required columns
                missing = [col for col in required_cols if col not in df.columns]
                if missing:
                    st.error(f"❌ Missing required columns: {', '.join(missing)}")
                    st.stop()

                if len(df) != 1:
                    st.error("❌ Please upload exactly one case")
                    st.stop()

                row = df.iloc[0]
                case_data = row.to_dict()

                if "model_results" not in st.session_state or st.session_state.model_results is None:
                    st.error("⚠️ Model not trained yet. Please train the model first.")
                    st.stop()

                # Temporary upload_id for embedding generation
                temp_upload_id = "temp_upload_" + str(uuid.uuid4())

                # Call subprocess generate_case_embedding.py with upload_id and case_data JSON
                with st.spinner("Generating embedding for prediction..."):
                    cmd = [
                        sys.executable,
                        "generate_case_embedding.py",
                        temp_upload_id,
                        json.dumps(case_data)
                    ]
                    proc = subprocess.run(cmd, capture_output=True, text=True)

                    if proc.returncode != 0:
                        st.error(f"❌ Embedding generation failed: {proc.stderr.strip()}")
                        st.stop()

                    embedding_json = proc.stdout.strip()
                    try:
                        embedding = np.array(json.loads(embedding_json))
                    except Exception as e:
                        st.error(f"❌ Failed to parse embedding JSON: {str(e)}")
                        st.stop()

                st.subheader("🧠 Case Embedding (Temporary)")
                st.write(embedding)

                st.subheader("🧪 Embedding Diagnostics")
                st.text("📦 Embedding vector:")
                st.write(embedding.tolist())

                if np.isnan(embedding).any():
                    st.error("❌ Embedding contains NaN values.")
                    st.stop()
                else:
                    st.success("✅ Embedding is valid (no NaNs)")

                # *** Prediction ***
                model = st.session_state.model_results["model"]
                embedding_reshaped = embedding.reshape(1, -1)  # Important for correct shape
                proba = model.predict_proba(embedding_reshaped)[0][1]
                prediction = "ASD Traits Detected" if proba >= 0.5 else "Typical Development"

                st.subheader("🔍 Prediction Result")
                col1, col2 = st.columns(2)
                col1.metric("Prediction", prediction)
                col2.metric("Confidence", f"{max(proba, 1-proba):.1%}")

                # Distance from training mean
                X_train = st.session_state.model_results["X_test"]
                train_mean = X_train.mean().values
                dist = np.linalg.norm(embedding - train_mean)
                st.text(f"📏 Distance from train mean: {dist:.4f}")
                if dist > 5.0:
                    st.warning("⚠️ Embedding far from training distribution. Prediction may be unreliable.")

                # Anomaly Detection
                with st.spinner("Running anomaly detection..."):
                    iso_result = train_isolation_forest(cache_key=temp_upload_id)
                    if iso_result:
                        iso_forest, scaler = iso_result
                        embedding_scaled = scaler.transform(embedding_reshaped)
                        anomaly_score = iso_forest.decision_function(embedding_scaled)[0]
                        is_anomaly = iso_forest.predict(embedding_scaled)[0] == -1

                        st.subheader("🕵️ Anomaly Detection")
                        if is_anomaly:
                            st.warning(f"⚠️ Anomaly detected (score: {anomaly_score:.3f})")
                        else:
                            st.success(f"✅ Normal case (score: {anomaly_score:.3f})")

                st.success("✅ Prediction completed successfully!")

            except Exception as e:
                st.error(f"❌ Error processing file: {str(e)}")
                logger.exception("Upload case error:")

    # === Tab 4: NLP to Cypher ===
    with tab4:
        st.header("💬 Natural Language to Cypher")
        with st.expander("ℹ️ What can I ask? (Dataset Description & Examples)"):
            st.markdown("""
            ### 📚 Dataset Overview
            This knowledge graph contains screening data for toddlers to help detect potential signs of Autism Spectrum Disorder (ASD).

            #### ✅ Node Types:
            - **Case**: A toddler who was screened
            - **BehaviorQuestion**: A question from the Q-Chat-10 questionnaire
            - **DemographicAttribute**: Characteristics like `Sex`, `Ethnicity`, `Jaundice`, `Family_mem_with_ASD`
            - **SubmitterType**: Who completed the questionnaire (e.g., Parent, Health worker)
            - **ASD_Trait**: Whether the case was labeled as showing ASD traits (`Yes` or `No`)

            #### 🔗 Relationships:
            - `HAS_ANSWER`: A case's answer to a behavioral question
            - `HAS_DEMOGRAPHIC`: Links a case to demographic attributes
            - `SUBMITTED_BY`: Who submitted the test
            - `SCREENED_FOR`: Final ASD classification
            """)

            st.markdown("### 🧠 Example Questions (Click to use)")
            example_questions = [
                "How many male toddlers have ASD traits?",
                "List all ethnicities with more than 5 cases",
                "How many cases answered '1' for both A1 and A2?"
            ]

            for q in example_questions:
                if st.button(q, key=f"example_{q}"):
                    st.session_state.preset_question = q
                    st.session_state.last_cypher_results = None  # Reset previous results

        default_question = st.session_state.get("preset_question", "")
        question = st.text_input("Ask about the data:", value=default_question, key="nlp_question_input")

        if st.button("▶️ Execute Query", key="execute_cypher_button"):
            if question.strip():
                cypher = nl_to_cypher(question)
                if cypher:
                    st.session_state.last_cypher_query = cypher
                    try:
                        with neo4j_service.session() as session:
                            results = session.run(cypher).data()
                            st.session_state.last_cypher_results = results
                    except Exception as e:
                        st.error(f"Query failed: {str(e)}")
                        st.session_state.last_cypher_results = None
            else:
                st.warning("Please enter a question.")

        if st.session_state.last_cypher_query:
            st.code(st.session_state.last_cypher_query, language="cypher")

        if st.session_state.last_cypher_results is not None:
            if len(st.session_state.last_cypher_results) > 0:
                st.dataframe(pd.DataFrame(st.session_state.last_cypher_results))
            else:
                st.info("No results found.")

if __name__ == "__main__":
    main()