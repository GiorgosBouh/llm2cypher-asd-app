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
    EMBEDDING_GENERATION_TIMEOUT = 600  # Increased timeout to 10 minutes
    LEAKAGE_CHECK = True
    SMOTE_K_NEIGHBORS = 5

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
            st.warning(f"⚠️ Cases missing SCREENED_FOR label: {len(missing_cases)} cases")
        else:
            st.success("✅ All cases have SCREENED_FOR labels.")
        return missing_cases

@safe_neo4j_operation
def refresh_screened_for_labels(csv_url: str):
    """Refresh all SCREENED_FOR relationships from CSV"""
    try:
        df = pd.read_csv(csv_url, delimiter=";", encoding='utf-8-sig')
        df.columns = [col.strip() for col in df.columns]

        if "Case_No" not in df.columns or "Class_ASD_Traits" not in df.columns:
            st.error("CSV must contain 'Case_No' and 'Class_ASD_Traits' columns")
            return

        with neo4j_service.session() as session:
            # Remove all existing relationships
            session.run("""
                MATCH (c:Case)-[r:SCREENED_FOR]->(:ASD_Trait)
                DELETE r
            """)
            
            # Batch create new relationships
            batch_size = 100
            total_cases = len(df)
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for i in range(0, total_cases, batch_size):
                batch = df.iloc[i:i+batch_size]
                queries = []
                params = []
                
                for _, row in batch.iterrows():
                    case_id = int(row["Case_No"])
                    label = str(row["Class_ASD_Traits"]).strip().capitalize()
                    if label in ["Yes", "No"]:
                        queries.append("""
                            MATCH (c:Case {id: $case_id_%d})
                            MERGE (t:ASD_Trait {label: $label_%d})
                            MERGE (c)-[:SCREENED_FOR]->(t)
                        """ % (case_id, case_id))
                        params.extend([{"case_id_%d" % case_id: case_id, "label_%d" % case_id: label}])
                
                if queries:
                    # Execute all queries in a single transaction
                    tx = session.begin_transaction()
                    try:
                        for query, param in zip(queries, params):
                            tx.run(query, **param)
                        tx.commit()
                    except Exception as e:
                        tx.rollback()
                        raise e
                
                # Update progress
                progress = min((i + batch_size) / total_cases, 1.0)
                progress_bar.progress(progress)
                status_text.text(f"Processing cases {i+1} to {min(i+batch_size, total_cases)} of {total_cases}")
                
        progress_bar.empty()
        status_text.empty()
        st.success(f"✅ Successfully updated {total_cases} SCREENED_FOR relationships")
        
    except Exception as e:
        st.error(f"Error refreshing labels: {str(e)}")
        logger.error(f"Error in refresh_screened_for_labels: {str(e)}", exc_info=True)

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
        # Remove all label-related relationships and properties
        session.run("""
            MATCH (c:Case)-[r:SCREENED_FOR]->(:ASD_Trait)
            DELETE r
        """)
        # Also remove any label-related properties that might be cached
        session.run("""
            MATCH (c:Case)
            REMOVE c.predicted_label
            REMOVE c.label_probability
        """)
        logger.info("✅ All label-related data removed for training")

# === Graph Embeddings Generation ===
@safe_neo4j_operation
def generate_embedding_for_case(upload_id: str) -> bool:
    """Generate embedding for a single case using subprocess"""
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        builder_path = os.path.join(script_dir, "generate_case_embedding.py")
        
        if not os.path.exists(builder_path):
            st.error(f"❌ Embedding generator script not found at: {builder_path}")
            return False

        # Run the embedding generator as a subprocess with timeout
        result = subprocess.run(
            [sys.executable, builder_path, upload_id],
            capture_output=True,
            text=True,
            timeout=Config.EMBEDDING_GENERATION_TIMEOUT
        )
        
        if result.returncode != 0:
            st.error(f"❌ Embedding generation failed with error:\n{result.stderr}")
            return False
            
        return True
        
    except subprocess.TimeoutExpired:
        st.error(f"❌ Embedding generation timed out after {Config.EMBEDDING_GENERATION_TIMEOUT} seconds")
        return False
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
                st.warning(f"⚠️ Case_No {case_id} has label '{csv_label}' in CSV but not in graph. Creating relationship...")
                session.run("""
                    MATCH (c:Case {id: $case_id})
                    MERGE (t:ASD_Trait {label: $label})
                    MERGE (c)-[:SCREENED_FOR]->(t)
                """, case_id=case_id, label=csv_label.capitalize())
            elif graph_label != csv_label:
                inconsistent_cases.append((case_id, csv_label, graph_label))

    if inconsistent_cases:
        st.error("❌ Found inconsistencies between CSV and Neo4j labels (Class_ASD_Traits vs SCREENED_FOR):")
        for case_id, csv_label, graph_label in inconsistent_cases:
            st.error(f"- Case_No {case_id}: CSV='{csv_label}' | Neo4j='{graph_label}'")
        st.stop()

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
            st.error("File must contain 'Case_No' column")
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
        ax.set_title("Feature-Embedding Dimension Correlation")
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
    
    st.subheader("📉 Probability Distribution Forecast")
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

        # 1. Strict label removal from the graph to prevent leakage
        remove_screened_for_labels()
        st.info("✅ SCREENED_FOR relationships temporarily removed for leakage prevention.")

        # 2. Generate embeddings with strict isolation (no labels considered for embedding generation)
        script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "kg_builder_2.py")
        if not os.path.exists(script_path):
            raise FileNotFoundError(f"kg_builder_2.py not found at {script_path}")

        st.info("🔄 Generating graph embeddings without label information...")
        result = subprocess.run(
            [sys.executable, script_path, "--no-labels"],
            capture_output=True,
            text=True,
            timeout=Config.EMBEDDING_GENERATION_TIMEOUT
        )
        
        if result.returncode != 0:
            error_msg = result.stderr or "Unknown error - check logs for details"
            raise RuntimeError(f"Embedding generation failed: {error_msg}")
        st.success("✅ Embeddings generated successfully (without label leakage).")

        # 3. Extract training data (embeddings and labels) from the CSV and graph
        st.info("📊 Extracting training data from graph and CSV...")
        X_raw, y = extract_training_data_from_csv(csv_url)
        X = X_raw.copy()
        X.columns = [f"Dim_{i}" for i in range(X.shape[1])]

        if X.empty or y.empty:
            st.error("⚠️ No valid training data available after extraction.")
            return None
        st.success(f"✅ Extracted {len(X)} cases for training.")

        # 4. Train/test split (stratified to maintain class proportions)
        st.info("🔪 Splitting data into training and test sets...")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=Config.TEST_SIZE,
            stratify=y,
            random_state=Config.RANDOM_STATE
        )
        st.success(f"✅ Train set size: {len(X_train)}, Test set size: {len(X_test)}")
        st.write(f"Class distribution in train set: {Counter(y_train)}")
        st.write(f"Class distribution in test set: {Counter(y_test)}")

        # 5. Calculate class weights for imbalance
        neg = sum(y_train == 0)
        pos = sum(y_train == 1)
        scale_pos_weight = neg / pos if pos != 0 else 1.0
        st.info(f"⚖️ Calculated scale_pos_weight: {scale_pos_weight:.2f}")

        # 6. Pipeline with StandardScaler and XGBoost
        pipeline = ImbPipeline([
            ('scaler', StandardScaler()),
            ('xgb', XGBClassifier(
                n_estimators=Config.N_ESTIMATORS,
                use_label_encoder=False,
                eval_metric='logloss',
                random_state=Config.RANDOM_STATE,
                scale_pos_weight=scale_pos_weight
            ))
        ])

        # 7. Cross-validation for robust performance estimation
        st.info("🔬 Performing Stratified K-Fold Cross-Validation...")
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=Config.RANDOM_STATE)
        cv_scores = []
        cv_y_true = []
        cv_y_proba = []

        for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X_train, y_train)):
            st.text(f"  - Processing CV Fold {fold_idx + 1}/5...")
            X_fold_train, y_fold_train = X_train.iloc[train_idx], y_train.iloc[train_idx]
            X_fold_val, y_fold_val = X_train.iloc[val_idx], y_train.iloc[val_idx]

            # Apply SMOTE only to the training data of the current fold
            k_neighbors = min(Config.SMOTE_K_NEIGHBORS, sum(y_fold_train == 1) - 1)
            if k_neighbors < 1:
                st.warning(f"⚠️ SMOTE k_neighbors adjusted to 1 for fold {fold_idx+1} due to very small minority class.")
                k_neighbors = 1 
            
            smote = SMOTE(
                sampling_strategy='auto',
                k_neighbors=k_neighbors, 
                random_state=Config.RANDOM_STATE
            )
            
            # Apply SMOTE and then fit the pipeline
            X_res, y_res = smote.fit_resample(X_fold_train, y_fold_train)
            
            temp_pipeline = ImbPipeline([
                ('scaler', StandardScaler()),
                ('xgb', XGBClassifier(
                    n_estimators=Config.N_ESTIMATORS,
                    use_label_encoder=False,
                    eval_metric='logloss',
                    random_state=Config.RANDOM_STATE,
                    scale_pos_weight=scale_pos_weight
                ))
            ])
            temp_pipeline.fit(X_res, y_res)
            
            y_proba_fold = temp_pipeline.predict_proba(X_fold_val)[:, 1]
            cv_scores.append(roc_auc_score(y_fold_val, y_proba_fold))
            cv_y_true.extend(y_fold_val.tolist())
            cv_y_proba.extend(y_proba_fold.tolist())

        st.success(f"✅ Cross-Validation complete. Mean CV ROC AUC: {np.mean(cv_scores):.3f} ± {np.std(cv_scores):.3f}")
        st.subheader("📊 Cross-Validation Performance (on Training Set)")
        plot_combined_curves(cv_y_true, cv_y_proba)

        # 8. Final model training on the entire training set
        st.info("🏋️‍♀️ Training final model on full training set...")
        
        final_smote = SMOTE(
            sampling_strategy='auto',
            k_neighbors=min(Config.SMOTE_K_NEIGHBORS, sum(y_train == 1) - 1),
            random_state=Config.RANDOM_STATE
        )
        X_train_res, y_train_res = final_smote.fit_resample(X_train, y_train)

        pipeline.fit(X_train_res, y_train_res)
        st.success("✅ Final model trained.")

        # 9. Reinsert labels into the graph (after training is complete)
        st.info("🔄 Re-inserting SCREENED_FOR relationships into the graph...")
        reinsert_labels_from_csv(csv_url)
        st.success("✅ SCREENED_FOR relationships re-inserted.")

        return {
            "model": pipeline,
            "X_test": X_test,
            "y_test": y_test,
            "cv_scores": cv_scores
        }

    except subprocess.TimeoutExpired:
        st.error(f"❌ Embedding generation timed out after {Config.EMBEDDING_GENERATION_TIMEOUT} seconds.")
        logger.error(f"Embedding generation subprocess timed out.")
        return None
    except FileNotFoundError as e:
        st.error(f"❌ Required script not found: {e}")
        logger.error(f"File not found: {e}")
        return None
    except RuntimeError as e:
        st.error(f"❌ Error during embedding generation or graph operation: {e}")
        logger.error(f"Runtime error during embedding generation: {e}")
        return None
    except Exception as e:
        logger.error(f"🚨 Model training failed: {str(e)}", exc_info=True)
        st.error(f"❌ An unexpected error occurred during model training: {str(e)}")
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
        st.error("❌ CSV must contain 'Case_No' and 'Class_ASD_Traits' columns")
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
""")

    # Initialize session state variables
    if "active_tab" not in st.session_state:
        st.session_state.active_tab = "Model Training"
    if "model_trained" not in st.session_state:
        st.session_state.model_trained = False
    if "model_results" not in st.session_state:
        st.session_state.model_results = None

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

        # First check for missing labels
        missing_labels = find_cases_missing_labels()
        if missing_labels:
            st.warning(f"⚠️ There are {len(missing_labels)} cases without SCREENED_FOR label.")
            
            # Add button to fix missing labels
            if st.button("🔄 Fix Missing Labels from CSV"):
                csv_url = "https://raw.githubusercontent.com/GiorgosBouh/llm2cypher-asd-app/main/Toddler_Autism_dataset_July_2018_2.csv"
                with st.spinner("Refreshing labels from CSV..."):
                    refresh_screened_for_labels(csv_url)
                    st.rerun()  # Refresh the UI
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

    # === Tab 2: Graph Embeddings ===
    with tab2:
        st.header("🌐 Graph Embeddings")
        st.warning("⚠️ Developer only - may take several minutes")
        
        if st.button("🔁 Recalculate All Embeddings"):
            with st.spinner("Running full graph rebuild and embedding generation..."):
                result = subprocess.run(
                    [sys.executable, "kg_builder_2.py"],
                    capture_output=True,
                    text=True,
                    timeout=Config.EMBEDDING_GENERATION_TIMEOUT
                )
                if result.returncode == 0:
                    st.success("✅ Embeddings recalculated and updated in the graph!")
                else:
                    st.error("❌ Failed to run kg_builder_2.py")
                    st.code(result.stderr)

    # === Tab 3: Upload New Case ===
    with tab3:
        st.header("📄 Upload New Case (Prediction Only)")
        
        uploaded_file = st.file_uploader(
            "Upload your prepared CSV file",
            type="csv",
            key="case_uploader"
        )

        if uploaded_file:
            try:
                df = pd.read_csv(uploaded_file, delimiter=";")
                st.subheader("📊 Uploaded CSV Preview")
                st.dataframe(df)

                required_cols = ["Case_No", "A1", "A2", "A3", "A4", "A5",
                                "A6", "A7", "A8", "A9", "A10",
                                "Sex", "Ethnicity", "Jaundice",
                                "Family_mem_with_ASD", "Who_completed_the_test"]
                
                missing = [col for col in required_cols if col not in df.columns]
                if missing:
                    st.error(f"❌ Missing required columns: {', '.join(missing)}")
                    st.stop()

                if len(df) != 1:
                    st.error("❌ Please upload exactly one case")
                    st.stop()

                row = df.iloc[0]
                temp_upload_id = "temp_" + str(uuid.uuid4())

                # Generate embedding
                with st.spinner("Generating embedding..."):
                    if not generate_embedding_for_case(temp_upload_id):
                        st.error("Failed to generate embedding")
                        st.stop()

                embedding = extract_user_embedding(temp_upload_id)
                if embedding is None:
                    st.error("Failed to extract embedding")
                    st.stop()

                # Make prediction
                if st.session_state.model_results:
                    model = st.session_state.model_results["model"]
                    proba = model.predict_proba(embedding)[0][1]
                    prediction = "ASD Traits Detected" if proba >= 0.5 else "Typical Development"
                    
                    st.subheader("🔍 Prediction Result")
                    col1, col2 = st.columns(2)
                    col1.metric("Prediction", prediction)
                    col2.metric("Confidence", f"{max(proba, 1-proba):.1%}")

            except Exception as e:
                st.error(f"❌ Error processing file: {str(e)}")

    # === Tab 4: NLP to Cypher ===
    with tab4:
        st.header("💬 Natural Language to Cypher")
        with st.expander("ℹ️ What can I ask? (Dataset Description & Examples)"):
            st.markdown("""
            ### 📚 Dataset Overview
            This knowledge graph contains screening data for toddlers to help detect potential signs of Autism Spectrum Disorder (ASD).

            #### ✅ Node Types:
            - **Case**: A toddler who was screened.
            - **BehaviorQuestion**: A question from the Q-Chat-10 questionnaire:
                - **A1**: Does your child look at you when you call his/her name?
                - **A2**: How easy is it for you to get eye contact with your child?
                - **A3**: Does your child point to indicate that s/he wants something?
                - **A4**: Does your child point to share interest with you?
                - **A5**: Does your child pretend?
                - **A6**: Does your child follow where you're looking?
                - **A7**: If you or someone else in the family is visibly upset, does your child show signs of wanting to comfort them?
                - **A8**: Would you describe your child's first words as normal in their development?
                - **A9**: Does your child use simple gestures such as waving to say goodbye?
                - **A10**: Does your child stare at nothing with no apparent purpose?

            - **DemographicAttribute**: Characteristics like `Sex`, `Ethnicity`, `Jaundice`, `Family_mem_with_ASD`.
            - **SubmitterType**: Who completed the questionnaire (e.g., Parent, Health worker).
            - **ASD_Trait**: Whether the case was labeled as showing ASD traits (`Yes` or `No`).

            #### 🔗 Relationships:
            - `HAS_ANSWER`: A case's answer to a behavioral question.
            - `HAS_DEMOGRAPHIC`: Links a case to demographic attributes.
            - `SUBMITTED_BY`: Who submitted the test.
            - `SCREENED_FOR`: Final ASD classification.
            """)

            st.markdown("### 🧠 Example Questions (Click to use)")
            example_questions = [
                "How many male toddlers have ASD traits?",
                "List all ethnicities with more than 5 cases.",
                "How many cases answered '1' for both A1 and A2?"
            ]

            for q in example_questions:
                if st.button(q, key=f"example_{q}"):
                    st.session_state["preset_question"] = q

        default_question = st.session_state.get("preset_question", "")
        question = st.text_input("Ask about the data:", value=default_question)

        if question:
            cypher = nl_to_cypher(question)
            if cypher:
                st.code(cypher, language="cypher")
                if st.button("▶️ Execute Query"):
                    with neo4j_service.session() as session:
                        try:
                            results = session.run(cypher).data()
                            if results:
                                st.dataframe(pd.DataFrame(results))
                            else:
                                st.info("No results found")
                        except Exception as e:
                            st.error(f"Query failed: {str(e)}")

if __name__ == "__main__":
    main()