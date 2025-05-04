import streamlit as st
from neo4j import GraphDatabase
from openai import OpenAI
from dotenv import load_dotenv
import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score, roc_curve, precision_recall_curve,
    average_precision_score, classification_report, 
    precision_score, recall_score, f1_score, confusion_matrix
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
from node2vec import Node2Vec
import networkx as nx
import tempfile
import shutil
import seaborn as sns
from sklearn.inspection import permutation_importance

# === Configuration ===
class Config:
    EMBEDDING_DIM = 64
    RANDOM_STATE = 42
    TEST_SIZE = 0.3  # Increased for better validation
    N_ESTIMATORS = 100
    SMOTE_RATIO = 'auto'
    MIN_CASES_FOR_ANOMALY_DETECTION = 10
    NODE2VEC_WALK_LENGTH = 20
    NODE2VEC_NUM_WALKS = 50
    NODE2VEC_WORKERS = 1
    NODE2VEC_P = 1
    NODE2VEC_Q = 1
    EMBEDDING_BATCH_SIZE = 50
    MAX_RELATIONSHIPS = 100000
    EMBEDDING_GENERATION_TIMEOUT = 300
    LEAKAGE_CHECK = True  # Enable rigorous leakage checks

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
# ✅ Load variables for use
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
    """Inserts a user case into Neo4j with leakage protection"""
    queries = []
    params = {"upload_id": upload_id, "id": int(row["Case_No"])}

    # Create Case node
    queries.append(("CREATE (c:Case {upload_id: $upload_id, id: $id})", params))

    # Add answers to behavioral questions
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

    # Add demographic attributes
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

    # Add submitter information
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
    import subprocess, sys, os

    builder_path = os.path.join(os.path.dirname(__file__), "generate_case_embedding.py")
    if not os.path.exists(builder_path):
        st.error(f"❌ Δεν βρέθηκε το αρχείο: {builder_path}")
        return False

    try:
        proc = subprocess.run(
            [sys.executable, builder_path, upload_id],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True
        )
        st.text(proc.stdout)

        if proc.returncode != 0:
            st.error("❌ Η δημιουργία embedding για το case απέτυχε")
            return False
        return True
    except Exception as e:
        st.error(f"❌ Σφάλμα: {e}")
        return False

    try:
        st.info(f"📄 Εκκίνηση: `{builder_path}`")
        print(f"[DEBUG] Running kg_builder_2.py at: {builder_path}")
        proc = subprocess.Popen(
            [sys.executable, builder_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True
        )

        output_lines = []
        for line in proc.stdout:
            output_lines.append(line)
            status_text.text(line.strip())
            st.text(line.strip())  # εμφάνιση σε Streamlit
            if not hasattr(progress_bar, "_current_progress"):
                progress_bar._current_progress = 10
            else:
                 progress_bar._current_progress = min(progress_bar._current_progress + 5, 95)

            progress_bar.progress(progress_bar._current_progress)

        proc.wait()
        if proc.returncode != 0:
            status_text.error("❌ Ο builder απέτυχε (return code != 0)")
            st.code("".join(output_lines), language="bash")
            return False
        if proc.returncode != 0:
            #Εκτύπωσε το πλήρες output αν απέτυχε
            st.error("❌ Ο builder απέτυχε (return code != 0)")
            st.code("".join(output_lines), language="bash")  # ✅ Προσθέτει εμφανές log
            return False

        progress_bar.progress(100)
        status_text.text("✅ Embeddings generated and stored!")
        return True

    except Exception as e:
        status_text.error(f"❌ Σφάλμα: {e}")
        return False

    finally:
        time.sleep(1)
        status_text.empty()
        progress_bar.empty()
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
        
        # Verify if case exists
        exists = session.run(
            "MATCH (c:Case {upload_id: $upload_id}) RETURN count(c) > 0 AS exists",
            upload_id=upload_id
        ).single()["exists"]
        
        if not exists:
            st.error(f"❌ Case with upload_id {upload_id} not found")
        else:
            st.error(f"❌ No embedding found for case {upload_id}. Please regenerate embeddings.")
        
        return None

from sklearn.impute import SimpleImputer

# === Training Data Preparation ===
@safe_neo4j_operation
def extract_training_data_from_csv(file_path: str) -> Tuple[pd.DataFrame, pd.Series]:
    """Extracts training data with leakage protection and NaN handling"""
    try:
        df = pd.read_csv(file_path, delimiter=";", encoding='utf-8-sig')

        # 🔧 Καθαρισμός στηλών (αφαιρεί και κρυφά και περιττά κενά)
        df.columns = [col.strip().replace('\r', '') for col in df.columns]
        df.columns = [col.strip() for col in df.columns]  # τελική μορφή

        # Έλεγχος για βασικές στήλες
        required_cols = ["Case_No", "Class_ASD_Traits"]
        missing = [col for col in required_cols if col not in df.columns]
        if missing:
            st.error(f"❌ Missing required columns: {', '.join(missing)}")
            st.write("📋 Found columns in CSV:", df.columns.tolist())
            return pd.DataFrame(), pd.Series()

        # Μετατροπή αριθμητικών πεδίων
        numeric_cols = [f"A{i}" for i in range(1, 11)] + ["Case_No"]
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        # Απόσπαση embeddings από Neo4j
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

        print("📋 Available Case_Nos in CSV:", df["Case_No"].tolist()[:5])
        print("📦 Valid Case_Nos with embeddings:", valid_ids[:5])
        print("📊 Total matched embeddings:", len(embeddings))

        # Φιλτράρισμα μόνο για έγκυρες περιπτώσεις
        df_filtered = df[df["Case_No"].isin(valid_ids)].copy()

        # Απόσπαση ετικετών
        y = df_filtered["Class_ASD_Traits"].apply(
            lambda x: 1 if str(x).strip().lower() == "yes" else 0
        )

        assert len(embeddings) == len(y), f"⚠️ Embeddings: {len(embeddings)}, Labels: {len(y)}"

        # Δημιουργία X
        X = pd.DataFrame(embeddings[:len(y)])

        # Imputation αν υπάρχουν NaN
        if X.isna().any().any():
            st.warning(f"⚠️ Found {X.isna().sum().sum()} NaN values in embeddings - applying imputation")
            X = X.fillna(X.mean())

        return X, y

    except Exception as e:
        st.error(f"Data extraction failed: {str(e)}")
        return pd.DataFrame(), pd.Series()

# === Model Evaluation ===
def analyze_embedding_correlations(X: pd.DataFrame, csv_url: str):
    """Συσχετίζει κάθε διάσταση embedding με τα αρχικά χαρακτηριστικά (A1–A10, δημογραφικά)"""
    st.subheader("📌 Feature–Embedding Correlation Analysis")

    try:
        df = pd.read_csv(csv_url, delimiter=";", encoding='utf-8-sig')
        df.columns = [col.strip() for col in df.columns]

        # Κρατάμε μόνο όσα Case_No υπάρχουν στο X
        if "Case_No" not in df.columns:
            st.error("Το αρχείο πρέπει να περιέχει στήλη 'Case_No'")
            return

        if len(X) != len(df):
            st.warning("⚠️ Μήκος X και CSV δεν ταιριάζουν — προσπαθώ best effort")

        # Επιλογή χαρακτηριστικών
        features = [f"A{i}" for i in range(1, 11)] + ["Sex", "Ethnicity", "Jaundice", "Family_mem_with_ASD"]

        df = df[features]
        df = pd.get_dummies(df, drop_first=True)  # μετατροπή κατηγορικών σε αριθμητικά

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

    # ROC Curve
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    ax[0].plot(fpr, tpr, label=f"AUC = {roc_auc_score(y_true, y_proba):.3f}")
    ax[0].plot([0, 1], [0, 1], 'k--')
    ax[0].set_xlabel("False Positive Rate")
    ax[0].set_ylabel("True Positive Rate")
    ax[0].set_title("ROC Curve")
    ax[0].legend()

    # Precision-Recall Curve
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
# === Model Evaluation ===

def evaluate_model(model, X_test, y_test):
    """Comprehensive model evaluation"""
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    # 📊 Κατανομή πιθανοτήτων
    st.subheader("📉 Κατανομή πιθανοτήτων πρόβλεψης")
    fig, ax = plt.subplots()
    ax.hist(y_proba, bins=20, color='skyblue', edgecolor='black')
    ax.set_xlabel("Πιθανότητα ASD Traits")
    ax.set_ylabel("Αριθμός Δειγμάτων")
    st.pyplot(fig)

    # Check for suspicious performance
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

    # Confusion Matrix
    st.subheader("Confusion Matrix")
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    st.pyplot(fig)

    # Feature Importance (Gini)
    st.subheader("🔍 Feature Importance (Gini)")
    try:
        importances = pd.Series(
            model.named_steps['classifier'].feature_importances_,
            index=[f"Dim_{i}" for i in range(X_test.shape[1])]
        ).sort_values(ascending=False)
        st.bar_chart(importances.head(15))
    except Exception as e:
        st.warning(f"Could not plot feature importance: {str(e)}")

    # Performance Curves
    st.subheader("📈 Performance Curves")
    plot_combined_curves(y_test, y_proba)

    # Permutation Importance
    show_permutation_importance(model, X_test, y_test)

    # Feature-to-Embedding Correlation Analysis
    csv_url = "https://raw.githubusercontent.com/GiorgosBouh/llm2cypher-asd-app/main/Toddler_Autism_dataset_July_2018_2.csv"
    analyze_embedding_correlations(X_test, csv_url)

# === Model Training ===
@st.cache_resource(show_spinner="Training ASD detection model...")
def train_asd_detection_model() -> Optional[dict]:
    """Trains the ASD detection model with leakage protection"""
    try:
        csv_url = "https://raw.githubusercontent.com/GiorgosBouh/llm2cypher-asd-app/main/Toddler_Autism_dataset_July_2018_2.csv"

        # 1️⃣ Αφαίρεση SCREENED_FOR για προστασία από leakage
        remove_screened_for_labels()

        # 2️⃣ Φόρτωση και καθαρισμός δεδομένων
        X_raw, y = extract_training_data_from_csv(csv_url)
        X = X_raw.copy()
        X.columns = [f"Dim_{i}" for i in range(X.shape[1])]
        if X.empty or y.empty:
            st.error("⚠️ No valid training data available")
            return None

        # 3️⃣ Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=Config.TEST_SIZE,
            stratify=y,
            random_state=Config.RANDOM_STATE
        )

        # 4️⃣ Pipeline with imputation + SMOTE + classifier
        pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='mean')),
            ('smote', SMOTE(random_state=Config.RANDOM_STATE)),
            ('classifier', RandomForestClassifier(
                n_estimators=Config.N_ESTIMATORS,
                random_state=Config.RANDOM_STATE,
                class_weight='balanced'
            ))
        ])

        pipeline.fit(X_train, y_train)

        # 5️⃣ Επανατοποθέτηση labels ΜΟΝΟ για τις περιπτώσεις με embeddings
        reinsert_labels_from_csv(csv_url)

        # 6️⃣ Επιστροφή
        return {
            "model": pipeline,
            "X_test": X_test,
            "y_test": y_test
        }

    except Exception as e:
        st.error(f"❌ Error training model: {e}")
        logger.error(f"Training error: {e}", exc_info=True)
        return None
# === Anomaly Detection ===
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
def train_isolation_forest() -> Optional[Tuple[IsolationForest, StandardScaler]]:
    """Trains anomaly detection model"""
    embeddings = get_existing_embeddings()
    if embeddings is None or len(embeddings) < Config.MIN_CASES_FOR_ANOMALY_DETECTION:
        st.warning(f"⚠️ Need at least {Config.MIN_CASES_FOR_ANOMALY_DETECTION} cases for anomaly detection")
        return None

    scaler = StandardScaler()
    embeddings_scaled = scaler.fit_transform(embeddings)

    contamination = min(0.1, 5.0 / len(embeddings))  # Dynamic contamination rate
    iso_forest = IsolationForest(
        contamination=contamination,
        random_state=Config.RANDOM_STATE
    )
    iso_forest.fit(embeddings_scaled)

    return iso_forest, scaler
@safe_neo4j_operation
def reinsert_labels_from_csv(csv_url: str):
    """Επανατοποθέτηση SCREENED_FOR labels από CSV"""
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
                    MERGE (t:ASD_Trait {value: $label})
                    MERGE (c)-[:SCREENED_FOR]->(t)
                """, case_id=case_id, label=label.capitalize())
# === Streamlit UI ===
def main():
    st.title("🧠 NeuroCypher ASD")
    st.markdown("""
        <i>Autism Spectrum Disorder detection using graph embeddings</i>
        """, unsafe_allow_html=True)

    # Sidebar
    st.sidebar.markdown(f"🔗 Connected to: `{os.getenv('NEO4J_URI')}`")

    # Tab system
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Model Training", 
        "🌐 Graph Embeddings", 
        "📤 Upload New Case", 
        "💬 NLP to Cypher"
    ])

    # === Model Training Tab ===
    with tab1:
        st.header("🤖 ASD Detection Model")

        if st.button("🔄 Train/Refresh Model"):
            with st.spinner("Training model with leakage protection..."):
                results = train_asd_detection_model()

                if results:
                    st.session_state.model_results = results
                    st.success("✅ Model trained successfully!")

                    # ✅ Αξιολόγηση μοντέλου
                    evaluate_model(
                        results["model"],
                        results["X_test"],
                        results["y_test"]
                    )

                    # ✅ Αυτόματη επανασύνδεση ετικετών
                    with st.spinner("Reattaching labels to cases..."):
                        csv_url = "https://raw.githubusercontent.com/GiorgosBouh/llm2cypher-asd-app/main/Toddler_Autism_dataset_July_2018_2.csv"
                        reinsert_labels_from_csv(csv_url)
                        st.success("🎯 Labels reinserted automatically after training!")

    # === Graph Embeddings Tab ===
    with tab2:
        st.header("🌐 Graph Embeddings")
        if st.button("🔁 Recalculate All Embeddings"):
             st.info("Αυτή η λειτουργία έχει απενεργοποιηθεί. Για embeddings, τρέξε το kg_builder_2.py.")

    # === File Upload Tab ===
    with tab3:
        st.header("📄 Upload New Case")
        uploaded_file = st.file_uploader(
            "Upload CSV for single case prediction", 
            type="csv",
            key="case_uploader"
        )

        if uploaded_file:
            try:
                df = pd.read_csv(uploaded_file, delimiter=";")
                required_cols = [
                    "Case_No", "A1", "A2", "A3", "A4", "A5", 
                    "A6", "A7", "A8", "A9", "A10",
                    "Sex", "Ethnicity", "Jaundice", 
                    "Family_mem_with_ASD", "Who_completed_the_test"
                ]

                if not all(col in df.columns for col in required_cols):
                    st.error("❌ Missing required columns in CSV")
                    st.stop()

                if len(df) != 1:
                    st.error("Please upload exactly one case")
                    st.stop()

                # Process file
                upload_id = str(uuid.uuid4())
                row = df.iloc[0]

                # 1. Insert case
                with st.spinner("Inserting case into graph..."):
                    upload_id = insert_user_case(row, upload_id)
                    st.session_state.last_upload_id = upload_id

                # 2. Generate embedding just for this case
                with st.spinner("Generating graph embedding for new case..."):
                    from generate_case_embedding import generate_embedding_for_case
                    success = generate_embedding_for_case(neo4j_service.driver, upload_id)
                    if not success:
                        st.error("❌ Failed to generate embedding for new case")
                        st.stop()           

                # 3. Extract embedding
                with st.spinner("Extracting case embedding..."):
                    embedding = extract_user_embedding(upload_id)
                    if embedding is None:
                        st.stop()
                    st.session_state.current_embedding = embedding

                st.subheader("🧠 Case Embedding")
                st.write(embedding)

                # 4. Make prediction if model exists
                if "model_results" in st.session_state:
                    model = st.session_state.model_results["model"]
                    proba = model.predict_proba(embedding)[0][1]
                    prediction = "ASD Traits Detected" if proba >= 0.5 else "Typical Development"

                    st.subheader("🔍 Prediction Result")
                    col1, col2 = st.columns(2)
                    col1.metric("Prediction", prediction)
                    col2.metric("Confidence", f"{max(proba, 1-proba):.1%}")

                    # Show probability distribution
                    fig = px.bar(
                        x=["Typical", "ASD Traits"],
                        y=[1-proba, proba],
                        title="Prediction Probabilities"
                    )
                    st.plotly_chart(fig)

                # 5. Anomaly detection
                with st.spinner("Running anomaly detection..."):
                    iso_result = train_isolation_forest()
                    if iso_result:
                        iso_forest, scaler = iso_result
                        embedding_scaled = scaler.transform(embedding)
                        anomaly_score = iso_forest.decision_function(embedding_scaled)[0]
                        is_anomaly = iso_forest.predict(embedding_scaled)[0] == -1

                        st.subheader("🕵️ Anomaly Detection")
                        if is_anomaly:
                            st.warning(f"⚠️ Anomaly detected (score: {anomaly_score:.3f})")
                        else:
                            st.success(f"✅ Normal case (score: {anomaly_score:.3f})")

            except Exception as e:
                st.error(f"❌ Error processing file: {str(e)}")
                logger.error(f"Upload error: {str(e)}", exc_info=True)

    # === NLP to Cypher Tab ===
    with tab4:
        st.header("💬 Natural Language to Cypher")
        question = st.text_input("Ask about the data:")

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

# === App entrypoint ===
if __name__ == "__main__":
    main()