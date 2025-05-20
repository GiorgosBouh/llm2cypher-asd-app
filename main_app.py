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
from sklearn.feature_selection import SelectKBest, f_classif
from io import StringIO

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
    SMOTE_K_NEIGHBORS = 5
    FEATURE_SELECTION_K = 50
    MAX_FILE_SIZE_MB = 5

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
def call_embedding_generator(upload_id: str) -> bool:
    """Generate embedding for a single case using subprocess with enhanced error handling"""
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        builder_path = os.path.join(script_dir, "generate_case_embedding.py")
        
        if not os.path.exists(builder_path):
            st.error(f"❌ Embedding generator script not found at: {builder_path}")
            return False

        env = os.environ.copy()
        env.update({
            "NEO4J_URI": os.getenv("NEO4J_URI"),
            "NEO4J_USER": os.getenv("NEO4J_USER"),
            "NEO4J_PASSWORD": os.getenv("NEO4J_PASSWORD"),
            "PYTHONPATH": os.path.dirname(script_dir)
        })

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
def extract_training_data_from_graph(_) -> Tuple[pd.DataFrame, pd.Series]:
    """Extracts X (embeddings) and y (ASD_Trait) directly from Neo4j"""
    with neo4j_service.session() as session:
        result = session.run("""
            MATCH (c:Case)-[:SCREENED_FOR]->(t:ASD_Trait)
            WHERE c.embedding IS NOT NULL
            RETURN c.embedding AS embedding, 
                   t.value = "Yes" AS label
        """)
        
        records = result.data()
        if not records:
            return pd.DataFrame(), pd.Series()
        
        X = pd.DataFrame([r["embedding"] for r in records])
        y = pd.Series([r["label"] for r in records])
        return X, y

@safe_neo4j_operation
def validate_labels(csv_url: str) -> bool:
    """Checks if CSV labels match graph labels"""
    try:
        df = pd.read_csv(csv_url, delimiter=";")
        with neo4j_service.session() as session:
            mismatches = session.run("""
                MATCH (c:Case)-[:SCREENED_FOR]->(t:ASD_Trait)
                WHERE c.id IN $case_ids
                AND (
                    (t.value = "Yes" AND NOT c.id IN $csv_yes) OR
                    (t.value = "No" AND NOT c.id IN $csv_no)
                )
                RETURN count(c) AS mismatches
            """, {
                "case_ids": df["Case_No"].tolist(),
                "csv_yes": df[df["Class_ASD_Traits"] == "Yes"]["Case_No"].tolist(),
                "csv_no": df[df["Class_ASD_Traits"] == "No"]["Case_No"].tolist()
            }).single()["mismatches"]
            
            if mismatches > 0:
                st.error(f"❌ {mismatches} cases have inconsistent labels between CSV and graph!")
                return False
            return True
    except Exception as e:
        st.error(f"Label validation failed: {str(e)}")
        return False

# === Model Evaluation ===
def analyze_embedding_correlations(X: pd.DataFrame, csv_url: str):
    st.subheader("📌 Feature–Embedding Correlation Analysis")
    try:
        df = pd.read_csv(csv_url, delimiter=";")
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
    
    st.subheader("📉 Probability Distribution Forecast")
    fig, ax = plt.subplots()
    ax.hist(y_proba, bins=20, color='skyblue', edgecolor='black')
    ax.set_xlabel("ASD Traits Probability")
    ax.set_ylabel("Number of Cases")
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
        # Use data from graph (not CSV)
        X, y = extract_training_data_from_graph(None)
        
        if X.empty or y.empty:
            st.error("⚠️ No valid training data with embeddings and labels found")
            return None

        # Split with stratification
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=Config.TEST_SIZE,
            stratify=y,
            random_state=Config.RANDOM_STATE
        )

        # Calculate class weights
        scale_pos_weight = sum(y_train == 0) / sum(y_train == 1)

        # Pipeline with SMOTE + Feature Selection + XGBoost
        pipeline = ImbPipeline([
            ('smote', SMOTE(
                k_neighbors=min(Config.SMOTE_K_NEIGHBORS, sum(y_train == 1) - 1),
                random_state=Config.RANDOM_STATE
            )),
            ('feature_select', SelectKBest(f_classif, k=Config.FEATURE_SELECTION_K)),
            ('xgb', XGBClassifier(
                n_estimators=Config.N_ESTIMATORS,
                scale_pos_weight=scale_pos_weight,
                eval_metric='aucpr',
                random_state=Config.RANDOM_STATE
            ))
        ])

        # Cross-validation
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=Config.RANDOM_STATE)
        y_proba = cross_val_predict(
            pipeline, X_train, y_train,
            cv=cv, method='predict_proba', n_jobs=-1
        )[:, 1]

        # Final training
        pipeline.fit(X_train, y_train)

        return {
            "model": pipeline,
            "X_test": X_test,
            "y_test": y_test,
            "cv_auc": roc_auc_score(y_train, y_proba)
        }

    except Exception as e:
        st.error(f"❌ Model training failed: {str(e)}")
        logger.exception("Training error")
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
    """Reinserts SCREENED_FOR labels from CSV"""
    df = pd.read_csv(csv_url, delimiter=";")
    df.columns = [col.strip() for col in df.columns]

    if "Case_No" not in df.columns or "Class_ASD_Traits" not in df.columns:
        st.error("❌ CSV is missing required columns 'Case_No' and 'Class_ASD_Traits'")
        return

    with neo4j_service.session() as session:
        for _, row in df.iterrows():
            case_id = int(row["Case_No"])
            label = str(row["Class_ASD_Traits"]).strip().lower()
            if label in ["yes", "no"]:
                session.run("""
                    MATCH (c:Case {id: $case_id})
                    MERGE (t:ASD_Trait {value: $label})
                    MERGE (c)-[:SCREENED_FOR]->(t)
                """, case_id=case_id, label=label.capitalize())

# === File Upload Handling ===
def handle_file_upload():
    """Handles file upload with proper validation and preview"""
    uploaded_file = st.file_uploader(
        "**Upload your prepared CSV file**",
        type="csv",
        key="case_uploader"
    )
    
    if uploaded_file is not None:
        try:
            # Check file size
            if uploaded_file.size > Config.MAX_FILE_SIZE_MB * 1024 * 1024:
                st.error(f"❌ File too large (max {Config.MAX_FILE_SIZE_MB}MB)")
                return None

            # Read CSV
            df = pd.read_csv(uploaded_file, delimiter=";")
            st.session_state.uploaded_file = uploaded_file
            st.session_state.uploaded_data = df

            # Display preview
            st.subheader("📊 Uploaded CSV Preview")
            st.dataframe(df.style.format({"Case_No": "{:.0f}"}), use_container_width=True)

            # Validate required columns
            required_cols = [
                "Case_No", "A1", "A2", "A3", "A4", "A5", 
                "A6", "A7", "A8", "A9", "A10",
                "Sex", "Ethnicity", "Jaundice", 
                "Family_mem_with_ASD", "Who_completed_the_test"
            ]
            
            if not all(col in df.columns for col in required_cols):
                missing = [col for col in required_cols if col not in df.columns]
                st.error(f"❌ Missing required columns: {', '.join(missing)}")
                return None

            if len(df) != 1:
                st.error("❌ Please upload exactly one case")
                return None

            return df.iloc[0]

        except Exception as e:
            st.error(f"❌ Error reading CSV file: {str(e)}")
            return None
    return None

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

- 🧠 Train a machine learning model to detect ASD traits using graph embeddings.
- 📤 Upload your own toddler screening data from the Q-Chat-10 questionnaire and other demographics.
- 🔗 Automatically connect the uploaded case to a knowledge graph.
- 🌐 Generate a graph-based embedding for the new case.
- 🔍 Predict whether the case shows signs of Autism Spectrum Disorder (ASD).
- 🕵️ Run anomaly detection to check for anomalies.
- 💬 Ask natural language questions and receive Cypher queries with results, using GPT4 based NLP-to-Cypher translation

---
### 📥 Download Example CSV

To get started, [download this example CSV](https://raw.githubusercontent.com/GiorgosBouh/llm2cypher-asd-app/main/Toddler_Autism_dataset_July_2018_3_test_39.csv)  
to format your own screening case correctly. 
Also, [read this description](https://raw.githubusercontent.com/GiorgosBouh/llm2cypher-asd-app/main/Toddler_data_description.docx) for further informations about the dataset.
""")

    # Initialize session state
    if "active_tab" not in st.session_state:
        st.session_state.active_tab = "Model Training"
    if "case_inserted" not in st.session_state:
        st.session_state.case_inserted = False
    if "last_upload_id" not in st.session_state:
        st.session_state.last_upload_id = None
    if "model_trained" not in st.session_state:
        st.session_state.model_trained = False

    # Create tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Model Training", 
        "🌐 Graph Embeddings", 
        "📤 Upload New Case", 
        "💬 NLP to Cypher"
    ])

    with tab1:
        st.header("🤖 ASD Detection Model")

        if st.button("🔄 Train/Refresh Model"):
            with st.spinner("Training model with leakage protection..."):
                results = train_asd_detection_model(str(uuid.uuid4()))
                if results:
                    st.session_state.model_results = results
                    st.session_state.model_trained = True
                    evaluate_model(
                        results["model"],
                        results["X_test"],
                        results["y_test"]
                    )
                    st.success("✅ Model trained successfully!")

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
                        old_emb = st.session_state.get("saved_embedding_case1")
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

    with tab3:
        st.header("📄 Upload New Case")
        
        # Reset case_inserted if new file uploaded
        if "uploaded_file" in st.session_state and st.session_state.uploaded_file is not None:
            if st.session_state.uploaded_file != st.session_state.get("last_uploaded_file"):
                st.session_state.case_inserted = False
                st.session_state.last_uploaded_file = st.session_state.uploaded_file

        # Handle file upload
        row = handle_file_upload()
        
        if row is not None and not st.session_state.case_inserted:
            if st.button("🚀 Process Case"):
                with st.spinner("Inserting case into graph..."):
                    upload_id = str(uuid.uuid4())
                    if insert_user_case(row, upload_id):
                        st.session_state.last_upload_id = upload_id
                        
                        with st.spinner("Generating embedding..."):
                            if call_embedding_generator(upload_id):
                                st.session_state.case_inserted = True
                                st.success("✅ Case processed successfully!")
                                st.balloons()

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