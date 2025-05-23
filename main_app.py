The problem of your Streamlit app "stacking on loading" often indicates that a long-running operation is blocking the main thread, causing the UI to freeze. In a Streamlit application, this is typically due to:

1.  **Synchronous Long Computations**: Operations like training a model, running complex Neo4j queries, or executing external scripts (subprocesses) can take a lot of time. If these are not handled asynchronously or with proper feedback, the app appears unresponsive.
2.  **Inefficient Database Interactions**: Slow Neo4j queries or large data transfers can cause delays.
3.  **Subprocess Issues**: If `subprocess.run` calls to `kg_builder_2.py` or `generate_case_embedding.py` hang, crash, or take too long, the main app will wait indefinitely or until a timeout occurs.
4.  **Caching Misuse**: While `@st.cache_resource` is used, if the `cache_key` always changes (e.g., `str(uuid.uuid4())`), the cached function will always re-run, effectively defeating the caching purpose for static operations and leading to repeated long waits.

I've made the following key adjustments to your code to address these potential issues and improve robustness and clarity:

---

### Summary of Fixes & Improvements

1.  **Robust Embedding Extraction for Training Data**:
    * In `extract_training_data_from_csv`, the process of matching CSV `Case_No` with Neo4j embeddings has been made more robust. Instead of relying on sequential order, it now explicitly collects embeddings along with their `Case_No` from Neo4j and then merges them correctly with the filtered DataFrame, ensuring `X` (embeddings) and `y` (labels) are perfectly aligned. This reduces the risk of data misalignment that could lead to unexpected model behavior or errors.
2.  **Streamlined `Neo4jService` Initialization**:
    * Removed `st.session_state.neo4j_driver.close()` from `get_neo4j_service`. Streamlit's `@st.cache_resource` is designed to manage the lifecycle of such resources. Explicitly closing the driver there could lead to issues where the driver is prematurely closed or reopened unnecessarily by the caching mechanism.
3.  **XGBoost `early_stopping_rounds` Clarification**:
    * Removed `early_stopping_rounds=10` from the `XGBClassifier` initialization within the `train_asd_detection_model` function. This parameter is only effective when an `eval_set` is passed to the `fit` method, which was not consistently done for the final model training pipeline. If you intend to use early stopping for the final model, you'll need to explicitly provide an evaluation set during `pipeline.fit()`.
4.  **Improved Anomaly Detection Caching**:
    * Modified the `cache_key` for `train_isolation_forest` to be based on the number of existing embeddings (`len(embeddings)`). This ensures the Isolation Forest model is *actually cached* and only retrained if the number of cases in your Neo4j graph changes, rather than retraining on every run or on every new prediction attempt for a temporary case.
5.  **"Upload New Case" Clarity**:
    * Clarified the `generate_case_embedding.py` subprocess call's role. The "Upload New Case" tab explicitly states "Prediction Only - No Graph Storage". The subprocess call to `generate_case_embedding.py` is assumed to handle the generation of the embedding *without persisting the new case data in your main Neo4j graph for that specific flow*. This is crucial to maintain the promise of "No Graph Storage" for uploaded cases. If `generate_case_embedding.py` *does* write to the graph, you should review its implementation to align with this promise.
6.  **Minor Refinements**:
    * Added comments for better understanding of certain logic blocks.
    * Ensured consistent variable usage.

---

### Full Code

```python
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
from imblearn.pipeline import make_pipeline as make_imb_pipeline # Kept for potential future use or consistency, though ImbPipeline is used directly
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
from node2vec import Node2Vec # Imported but not used in the provided snippet
import networkx as nx # Imported but not used in the provided snippet
import tempfile # Imported but not used in the provided snippet
import shutil # Imported but not used in the provided snippet
import seaborn as sns
from sklearn.inspection import permutation_importance
import subprocess
import sys
from sklearn.impute import SimpleImputer # Imported but not used in the provided snippet
from xgboost import XGBClassifier
from imblearn.pipeline import Pipeline as ImbPipeline
import json

# === Configuration ===
class Config:
    EMBEDDING_DIM = 128
    # Using np.random.randint directly for RANDOM_STATE ensures it's different every run if not cached,
    # but for reproducibility across app restarts (without clearing cache), a fixed seed might be better.
    # For now, keeping it as is, as it's passed to models.
    RANDOM_STATE = np.random.randint(0, 1000) 
    TEST_SIZE = 0.3
    N_ESTIMATORS = 100
    SMOTE_RATIO = 'auto' # Defined but 'auto' is used in SMOTE, which is equivalent
    MIN_CASES_FOR_ANOMALY_DETECTION = 10
    NODE2VEC_WALK_LENGTH = 20 # Defined but used by kg_builder_2.py, not directly here
    NODE2VEC_NUM_WALKS = 100 # Defined but used by kg_builder_2.py, not directly here
    NODE2VEC_WORKERS = 2 # Defined but used by kg_builder_2.py, not directly here
    NODE2VEC_P = 1 # Defined but used by kg_builder_2.py, not directly here
    NODE2VEC_Q = 1 # Defined but used by kg_builder_2.py, not directly here
    EMBEDDING_BATCH_SIZE = 50 # Defined but used by kg_builder_2.py, not directly here
    MAX_RELATIONSHIPS = 100000 # Defined but not used in this snippet
    EMBEDDING_GENERATION_TIMEOUT = 300
    LEAKAGE_CHECK = True
    SMOTE_K_NEIGHBORS = 3

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
    # Streamlit's @st.cache_resource handles resource lifecycle.
    # Explicitly closing a driver potentially managed by the cache might cause issues.
    # The cache ensures a single driver instance across reruns unless its arguments change.
    # if 'neo4j_driver' in st.session_state:
    #     st.session_state.neo4j_driver.close()
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
            logger.error(f"Neo4j operation failed: {str(e)}", exc_info=True) # Log full traceback
            return None
    return wrapper

@safe_neo4j_operation
def check_embedding_dimensions():
    with neo4j_service.session() as session:
        result = session.run("""
            MATCH (c:Case) WHERE c.embedding IS NOT NULL
            RETURN c.id AS case_id, size(c.embedding) AS embedding_length
        """)
        wrong_dims = [(r["case_id"], r["embedding_length"]) for r in result if r["embedding_length"] != Config.EMBEDDING_DIM]
        if wrong_dims:
            st.warning(f"⚠️ Cases with wrong embedding size ({Config.EMBEDDING_DIM}): {wrong_dims}")
        else:
            st.success(f"✅ All embeddings have correct size ({Config.EMBEDDING_DIM}).")

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
    try:
        df = pd.read_csv(csv_url, delimiter=";", encoding='utf-8-sig')
        df.columns = [col.strip() for col in df.columns]

        if "Case_No" not in df.columns or "Class_ASD_Traits" not in df.columns:
            st.error("❌ Το CSV πρέπει να περιέχει τις στήλες 'Case_No' και 'Class_ASD_Traits'")
            return False

        with neo4j_service.session() as session:
            # Delete all existing relationships
            session.run("""
                MATCH (c:Case)-[r:SCREENED_FOR]->(:ASD_Trait)
                DELETE r
            """)
            logger.info("✅ Παλιές σχέσεις SCREENED_FOR διαγράφηκαν.")

            # Recreate new relationships in batches for efficiency
            # It's better to use UNWIND for batching in Neo4j rather than a Python loop with individual transactions.
            data_to_create = []
            for _, row in df.iterrows():
                case_id = int(row["Case_No"])
                label = str(row["Class_ASD_Traits"]).strip().capitalize()
                if label in ["Yes", "No"]:
                    data_to_create.append({"case_id": case_id, "label": label})
            
            # Use UNWIND for efficient batching of relationship creation
            session.run("""
                UNWIND $data AS item
                MATCH (c:Case {id: item.case_id})
                MERGE (t:ASD_Trait {label: item.label})
                MERGE (c)-[:SCREENED_FOR]->(t)
            """, data={"data": data_to_create})

            logger.info("✅ Νέες σχέσεις SCREENED_FOR δημιουργήθηκαν με βάση το CSV.")
            st.success("✅ Ολοκληρώθηκε η ανανέωση των σχέσεων SCREENED_FOR.")
            return True
    except Exception as e:
        st.error(f"Failed to refresh SCREENED_FOR labels: {e}")
        logger.error(f"Failed to refresh SCREENED_FOR labels: {e}", exc_info=True)
        return False

# === Data Insertion ===
# Note: insert_user_case is not used by the "Upload New Case" tab's primary prediction flow,
# which instead calls generate_case_embedding.py via subprocess.
@safe_neo4j_operation
def insert_user_case(row: pd.Series, upload_id: str) -> str:
    queries = []

    # MERGE the Case by upload_id, setting id and embedding to null.
    # This assumes 'Case_No' is the permanent ID, while 'upload_id' is temporary for the session.
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

    # Add ASD_Trait relationship if Class_ASD_Traits exists (e.g., for training data insertion)
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
def generate_embedding(upload_id: str, script_path: str, case_data_json: Optional[str] = None) -> Tuple[bool, Optional[str]]:
    """Generates embedding for a case using a subprocess."""
    try:
        if not os.path.exists(script_path):
            st.error(f"❌ Embedding generator script not found at: {script_path}")
            return False, None

        env = os.environ.copy()
        env.update({
            "NEO4J_URI": os.getenv("NEO4J_URI"),
            "NEO4J_USER": os.getenv("NEO4J_USER"),
            "NEO4J_PASSWORD": os.getenv("NEO4J_PASSWORD"),
            "PYTHONPATH": os.path.dirname(script_path)
        })

        cmd = [sys.executable, script_path, upload_id]
        if case_data_json:
            cmd.append(case_data_json) # Pass case data as JSON string for temporary use

        result = subprocess.run(
            cmd,
            env=env,
            capture_output=True,
            text=True,
            timeout=Config.EMBEDDING_GENERATION_TIMEOUT
        )

        if result.returncode != 0:
            error_msg = result.stderr or "Unknown error (no stderr output)"
            st.error(f"❌ Embedding generation failed with error:\n{error_msg}")
            logger.error(f"Embedding generation failed for {upload_id}: {error_msg}")
            return False, None

        # Assuming the script outputs the embedding JSON to stdout
        return True, result.stdout.strip()

    except subprocess.TimeoutExpired:
        st.error("❌ Embedding generation timed out")
        logger.error(f"Embedding generation timeout for {upload_id}")
        return False, None
    except Exception as e:
        st.error(f"❌ Fatal error calling embedding script: {str(e)}")
        logger.exception(f"Fatal error generating embedding for {upload_id}")
        return False, None

# These functions are wrappers to call the main generate_embedding
# 'call_embedding_generator' and 'generate_embedding_for_case' seem redundant, consolidating to one.
# The 'upload_id' parameter is critical for the subprocess to identify the case.
# If 'generate_case_embedding.py' is expected to insert data, this upload_id (e.g., temp_upload_...) is used.
def generate_embedding_via_subprocess(upload_id: str, case_data_json: Optional[str] = None) -> Tuple[bool, Optional[str]]:
    """Generate embedding for a single case using subprocess"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    builder_path = os.path.join(script_dir, "generate_case_embedding.py")
    return generate_embedding(upload_id, builder_path, case_data_json)


# === Natural Language to Cypher ===
def nl_to_cypher(question: str) -> Optional[str]:
    """Translates natural language to Cypher using OpenAI"""
    prompt = f"""
    You are a Cypher expert working with a Neo4j Knowledge Graph about toddlers and autism.

    Schema:
    - (:Case {{id: int}})
    - (:BehaviorQuestion {{name: string}})
    - (:ASD_Trait {{label: 'Yes' | 'No'}})
    - (:DemographicAttribute {{type: 'Sex' | 'Ethnicity' | 'Jaundice' | 'Family_mem_with_ASD', value: string}})
    - (:SubmitterType {{type: string}})

    Relationships:
    - (:Case)-[:HAS_ANSWER {{value: int}}]->(:BehaviorQuestion)
    - (:Case)-[:HAS_DEMOGRAPHIC]->(:DemographicAttribute)
    - (:Case)-[:SCREENED_FOR]->(:ASD_Trait)
    - (:Case)-[:SUBMITTED_BY]->(:SubmitterType)

    Rules:
    1. Always use `toLower()` for case-insensitive comparisons where appropriate (e.g., for string values like 'Sex', 'Ethnicity').
    2. Interpret 'f' as 'female' and 'm' as 'male' for Sex.
    3. Never use SCREENED_FOR relationships in training queries (This rule is for model training, not for general NL2Cypher queries).
    4. If the question asks for a count of cases, return the count directly.
    5. Be mindful of property names. For ASD_Trait, the property is `label`, not `value`.

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
        logger.error(f"OpenAI API error: {e}", exc_info=True)
        return None

# === Embedding Extraction ===
# This function is not used in the "Upload New Case" prediction flow,
# as the embedding is returned directly by the subprocess call.
@safe_neo4j_operation
def extract_user_embedding(upload_id: str) -> Optional[np.ndarray]:
    """Safely extracts embedding for a specific case identified by upload_id."""
    with neo4j_service.session() as session:
        result = session.run(
            "MATCH (c:Case {upload_id: $upload_id}) RETURN c.embedding AS embedding",
            upload_id=upload_id
        )
        record = result.single()
        
        if record and record["embedding"] is not None:
            return np.array(record["embedding"]).reshape(1, -1)
        
        # Check if the case exists at all, even without an embedding
        exists = session.run(
            "MATCH (c:Case {upload_id: $upload_id}) RETURN count(c) > 0 AS exists",
            upload_id=upload_id
        ).single()["exists"]
        
        if not exists:
            st.error(f"❌ Case with upload_id {upload_id} not found.")
        else:
            st.error(f"❌ No embedding found for case {upload_id}. Please ensure embedding generation completed.")
        
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
                # If no label in graph, create it based on CSV
                st.warning(f"⚠️ Case_No {case_id} has label '{csv_label}' in CSV, but no label in graph. Creating relationship...")
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
        df_raw = pd.read_csv(file_path, delimiter=";", encoding='utf-8-sig')
        df_raw.columns = [col.strip().replace('\r', '') for col in df_raw.columns]
        df_raw.columns = [col.strip() for col in df_raw.columns]

        # Check column presence
        required_cols = ["Case_No", "Class_ASD_Traits"]
        missing = [col for col in required_cols if col not in df_raw.columns]
        if missing:
            st.error(f"❌ Missing required columns: {', '.join(missing)}")
            st.write("📋 Found columns in CSV:", df_raw.columns.tolist())
            return pd.DataFrame(), pd.Series()

        # Ensure Case_No is numeric for consistency
        df_raw["Case_No"] = pd.to_numeric(df_raw["Case_No"], errors='coerce')
        df_raw.dropna(subset=["Case_No"], inplace=True)
        df_raw["Case_No"] = df_raw["Case_No"].astype(int)

        # Check consistency of labels between CSV and graph
        check_label_consistency(df_raw.copy(), neo4j_service)

        # Extract embeddings and align them with Case_No from Neo4j
        neo4j_embeddings_data = []
        with neo4j_service.session() as session:
            # Fetch all embeddings with their IDs
            result = session.run("""
                MATCH (c:Case)
                WHERE c.embedding IS NOT NULL
                RETURN c.id AS case_id, c.embedding AS embedding
            """)
            for record in result:
                neo4j_embeddings_data.append({
                    "Case_No": record["case_id"],
                    "embedding": np.array(record["embedding"])
                })

        if not neo4j_embeddings_data:
            st.error("❌ No embeddings found in Neo4j for training.")
            return pd.DataFrame(), pd.Series()

        embeddings_df = pd.DataFrame(neo4j_embeddings_data)
        embeddings_df.set_index("Case_No", inplace=True)
        
        # Merge CSV data with embeddings based on Case_No
        df_merged = pd.merge(
            df_raw,
            embeddings_df,
            left_on="Case_No",
            right_index=True,
            how="inner" # Only keep cases that have both CSV data and an embedding
        )

        if df_merged.empty:
            st.error("⚠️ No common cases with embeddings found after merging CSV data.")
            return pd.DataFrame(), pd.Series()

        # Prepare X (embeddings) and y (labels)
        X = pd.DataFrame(df_merged["embedding"].tolist())
        X.columns = [f"Dim_{i}" for i in range(X.shape[1])] # Rename columns for clarity
        y = df_merged["Class_ASD_Traits"].apply(
            lambda x: 1 if str(x).strip().lower() == "yes" else 0
        )
        
        # Ensure X and y are perfectly aligned by index
        X.index = df_merged["Case_No"]
        y.index = df_merged["Case_No"]
        
        # Impute NaNs if any, though embeddings typically shouldn't have them if generation is robust
        if X.isna().any().any():
            st.warning(f"⚠️ Found {X.isna().sum().sum()} NaN values in embeddings - applying imputation")
            imputer = SimpleImputer(strategy='mean')
            X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns, index=X.index)

        assert len(X) == len(y), "Mismatch between number of embeddings and labels after extraction."

        return X, y

    except Exception as e:
        st.error(f"Data extraction failed: {str(e)}")
        logger.error(f"Data extraction failed: {str(e)}", exc_info=True)
        return pd.DataFrame(), pd.Series()

# === Model Evaluation ===
def analyze_embedding_correlations(X: pd.DataFrame, csv_url: str):
    st.subheader("📌 Feature–Embedding Correlation Analysis")
    try:
        df_features = pd.read_csv(csv_url, delimiter=";", encoding='utf-8-sig')
        df_features.columns = [col.strip() for col in df_features.columns]

        if "Case_No" not in df_features.columns:
            st.error("The CSV file must contain a 'Case_No' column for correlation analysis.")
            return

        # Ensure Case_No is numeric and handle potential errors
        df_features["Case_No"] = pd.to_numeric(df_features["Case_No"], errors='coerce')
        df_features.dropna(subset=["Case_No"], inplace=True)
        df_features["Case_No"] = df_features["Case_No"].astype(int)

        # Align X (embeddings) with df_features (original features) using Case_No
        # This assumes X's index is Case_No, as set in extract_training_data_from_csv
        df_combined = pd.merge(
            df_features,
            X,
            left_on="Case_No",
            right_index=True,
            how="inner"
        )
        
        if df_combined.empty:
            st.warning("No common cases found for correlation analysis after merging features and embeddings.")
            return

        features_for_correlation = [f"A{i}" for i in range(1, 11)] + ["Sex", "Ethnicity", "Jaundice", "Family_mem_with_ASD"]
        
        # Select and one-hot encode the original features
        df_original_features_processed = df_combined[features_for_correlation].copy()
        df_original_features_processed = pd.get_dummies(df_original_features_processed, drop_first=True)

        # Select the embedding dimensions from the combined DataFrame
        embedding_cols = [col for col in df_combined.columns if col.startswith("Dim_")]
        X_embeddings_aligned = df_combined[embedding_cols]

        if df_original_features_processed.empty or X_embeddings_aligned.empty:
            st.warning("Not enough data to perform correlation analysis after processing features.")
            return

        # Calculate correlation matrix
        corr = pd.DataFrame(index=df_original_features_processed.columns, columns=X_embeddings_aligned.columns)

        for feat in df_original_features_processed.columns:
            for dim in X_embeddings_aligned.columns:
                # Ensure both series have same length for correlation
                if len(df_original_features_processed[feat]) == len(X_embeddings_aligned[dim]):
                    corr.at[feat, dim] = np.corrcoef(df_original_features_processed[feat], X_embeddings_aligned[dim])[0, 1]
                else:
                    corr.at[feat, dim] = np.nan # Or handle discrepancy

        corr = corr.astype(float)
        corr.dropna(axis=0, how='all', inplace=True) # Drop rows with all NaNs if any
        corr.dropna(axis=1, how='all', inplace=True) # Drop columns with all NaNs if any

        if corr.empty:
            st.warning("Correlation matrix is empty. Not enough common data or features for analysis.")
            return

        fig, ax = plt.subplots(figsize=(max(10, len(corr.columns) / 2), max(6, len(corr.index) / 3)))
        sns.heatmap(corr, cmap="coolwarm", center=0, annot=False, ax=ax)
        ax.set_title("Correlation of Original Features with Embedding Dimensions")
        st.pyplot(fig)

    except Exception as e:
        st.error(f"❌ Correlation analysis failed: {str(e)}")
        logger.error(f"Correlation analysis failed: {str(e)}", exc_info=True)

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
        # Ensure X_test columns are named if they aren't already
        if not all(isinstance(col, str) and col.startswith("Dim_") for col in X_test.columns):
            X_test.columns = [f"Dim_{i}" for i in range(X_test.shape[1])]

        result = permutation_importance(model, X_test, y_test, n_repeats=10, random_state=42, n_jobs=-1)
        importance_df = pd.DataFrame({
            "Feature": X_test.columns,
            "Importance": result.importances_mean
        }).sort_values(by="Importance", ascending=False)

        st.subheader("📊 Permutation Importance")
        st.bar_chart(importance_df.set_index("Feature").head(15))
    except Exception as e:
        st.warning(f"Could not calculate permutation importance: {str(e)}")
        logger.error(f"Permutation importance failed: {str(e)}", exc_info=True)


def evaluate_model(model, X_test, y_test):
    """Comprehensive model evaluation"""
    if X_test.empty or y_test.empty:
        st.warning("No test data available for evaluation.")
        return

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    st.subheader("📉 Probability Distribution Forecast")
    fig, ax = plt.subplots()
    ax.hist(y_proba, bins=20, color='skyblue', edgecolor='black')
    ax.set_xlabel("ASD Traits Probability")
    ax.set_ylabel("Number of Cases")
    st.pyplot(fig)

    auc_score = roc_auc_score(y_test, y_proba)
    if auc_score > 0.98:
        st.warning(f"""
        🚨 Suspiciously high performance detected (ROC AUC: {auc_score:.3f}). Possible causes:
        1. **Data Leakage**: Features used in the model might implicitly contain information about the target variable from the graph generation process.
        2. **Test Set Contamination**: The test set might inadvertently contain data or patterns too similar to the training data.
        3. **Label Contamination**: Labels might have been used during graph embedding generation, leading to an overly optimistic model.
        """)

    st.subheader("📊 Model Evaluation Metrics")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("ROC AUC", f"{auc_score:.3f}")
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

    # This CSV URL is specific to the original features for correlation analysis
    csv_url_original_features = "https://raw.githubusercontent.com/GiorgosBouh/llm2cypher-asd-app/main/Toddler_Autism_dataset_July_2018_2.csv"
    analyze_embedding_correlations(X_test, csv_url_original_features)

# === Model Training ===
@st.cache_resource(show_spinner="Training ASD detection model...")
def train_asd_detection_model(cache_key: str) -> Optional[dict]:
    """
    Trains the ASD detection model.
    The `cache_key` parameter is explicitly passed to allow manual cache invalidation,
    ensuring a fresh training run when triggered by the button.
    """
    try:
        csv_url = "https://raw.githubusercontent.com/GiorgosBouh/llm2cypher-asd-app/main/Toddler_Autism_dataset_July_2018_2.csv"
        
        # 1. Refresh labels from CSV
        with st.spinner("Refreshing SCREENED_FOR labels..."):
            if not refresh_screened_for_labels(csv_url):
                return None
            missing_cases = find_cases_missing_labels()
            if missing_cases:
                st.error(f"❌ {len(missing_cases)} cases still missing labels after refresh. Please check CSV data.")
                return None

        # 2. Remove labels temporarily for safe embedding generation (to prevent leakage)
        with st.spinner("Removing labels temporarily for safe embedding generation..."):
            remove_screened_for_labels()
            
        # 3. Rebuild graph embeddings (calls external kg_builder_2.py)
        with st.spinner("Rebuilding graph embeddings. This may take a while..."):
            result = subprocess.run(
                [sys.executable, "kg_builder_2.py"],
                capture_output=True,
                text=True,
                timeout=600  # 10 minutes timeout
            )
            if result.returncode != 0:
                st.error(f"❌ Failed to generate embeddings (kg_builder_2.py exited with error):\n{result.stderr}")
                logger.error(f"kg_builder_2.py failed: {result.stderr}")
                return None
            else:
                logger.info(f"kg_builder_2.py stdout: {result.stdout}")

        # 4. Restore labels after embedding generation
        with st.spinner("Restoring labels after embedding generation..."):
            if not reinsert_labels_from_csv(csv_url):
                return None
            missing_cases = find_cases_missing_labels()
            if missing_cases:
                st.error(f"❌ {len(missing_cases)} cases missing labels after reinsertion. Please check CSV data.")
                return None

        # 5. Load embeddings and labels from CSV
        with st.spinner("Loading training data..."):
            X_raw, y = extract_training_data_from_csv(csv_url)
            if X_raw.empty or y.empty:
                st.error("⚠️ No valid training data available after extraction.")
                return None
                
            # Check for leakage after loading data, before splitting
            if Config.LEAKAGE_CHECK:
                labeled_cases_in_graph = set()
                with neo4j_service.session() as session:
                    result = session.run("""
                        MATCH (c:Case)-[:SCREENED_FOR]->(:ASD_Trait)
                        RETURN c.id AS case_id
                    """)
                    labeled_cases_in_graph = {r["case_id"] for r in result}
                
                # Check if any case_id in X_raw (which is now indexed by Case_No) was part of the initial labeled set
                # This check ensures that the data used for training and testing did not have labels
                # present during the embedding generation phase if that's the intended leakage prevention.
                # The labels are removed, then embeddings generated, then labels restored.
                # So if any case_id from X_raw.index was in labeled_cases_in_graph *before* the removal,
                # it's just a verification that the labels were indeed restored properly.
                # The *critical* leakage check is about *how embeddings were generated*.
                # If embeddings were generated when labels were present, that's the actual leakage.
                # The current workflow of remove_labels -> generate_embeddings -> restore_labels attempts to prevent this.
                # This `if any(case_id in labeled_cases_in_graph for case_id in X_raw.index)` check
                # would actually always be true if reinsert_labels_from_csv worked.
                # It's more of a check for *missing* labels after reinsertion, already covered.
                # The `LEAKAGE_CHECK` constant might refer to a deeper check on how kg_builder_2.py handles labels internally.
                pass # Keeping the pass as the primary leakage prevention is the remove/reinsert cycle.

            X = X_raw # X_raw is already a DataFrame with appropriate column names
            # X.columns = [f"Dim_{i}" for i in range(X.shape[1])] # This should already be handled in extract_training_data_from_csv

        # 6. Train/test split with stratification
        with st.spinner("Splitting dataset..."):
            X_train, X_test, y_train, y_test = train_test_split(
                X, y,
                test_size=Config.TEST_SIZE,
                stratify=y,
                random_state=Config.RANDOM_STATE
            )

        # 7. Calculate class weights
        neg = sum(y_train == 0)
        pos = sum(y_train == 1)
        scale_pos_weight = neg / pos if pos > 0 else 1 # Avoid division by zero if no positive samples

        # 8. Configure pipeline with SMOTE and XGBoost
        with st.spinner("Configuring model pipeline..."):
            # Ensure k_neighbors is less than the number of minority samples (pos)
            smote_k = min(Config.SMOTE_K_NEIGHBORS, pos - 1) if pos > 1 else 1 # smote_k must be at least 1
            if smote_k == 0: # If pos is 1, k_neighbors becomes 0, SMOTE fails. Handle this case.
                st.warning("Not enough positive samples for SMOTE k_neighbors. SMOTE will be skipped.")
                pipeline = ImbPipeline([
                    ('xgb', XGBClassifier(
                        n_estimators=Config.N_ESTIMATORS,
                        use_label_encoder=False, # Deprecated warning, but still works for now
                        eval_metric='logloss',
                        random_state=Config.RANDOM_STATE,
                        scale_pos_weight=scale_pos_weight
                        # early_stopping_rounds removed as it requires eval_set in pipeline.fit()
                    ))
                ])
            else:
                pipeline = ImbPipeline([
                    ('smote', SMOTE(
                        sampling_strategy='auto',
                        k_neighbors=smote_k,
                        random_state=Config.RANDOM_STATE
                    )),
                    ('xgb', XGBClassifier(
                        n_estimators=Config.N_ESTIMATORS,
                        use_label_encoder=False, # Deprecated warning, but still works for now
                        eval_metric='logloss',
                        random_state=Config.RANDOM_STATE,
                        scale_pos_weight=scale_pos_weight
                        # early_stopping_rounds removed as it requires eval_set in pipeline.fit()
                    ))
                ])

        # 9. Cross-validation (only on training set)
        with st.spinner("Running cross-validation..."):
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=Config.RANDOM_STATE)
            y_proba_cv = cross_val_predict(
                pipeline, X_train, y_train,
                cv=cv,
                method='predict_proba',
                n_jobs=-1
            )[:, 1]

        # 10. Train final model on the entire training set
        with st.spinner("Training final model..."):
            # If early_stopping_rounds is desired, add an eval_set here:
            # pipeline.fit(X_train, y_train, xgb__eval_set=[(X_val, y_val)], xgb__verbose=False)
            pipeline.fit(X_train, y_train)

        # 11. Evaluation
        with st.spinner("Evaluating model..."):
            test_proba = pipeline.predict_proba(X_test)[:, 1]
            test_pred = pipeline.predict(X_test)
            
            # Performance metrics
            cv_auc = roc_auc_score(y_train, y_proba_cv)
            test_auc = roc_auc_score(y_test, test_proba)
            
            st.subheader("📊 Cross-Validation Results (Training Set)")
            st.write(f"Mean CV ROC AUC: {cv_auc:.3f}")
            
            st.subheader("📊 Test Set Results")
            st.write(f"Test ROC AUC: {test_auc:.3f}")

            # Performance validation
            if test_auc > 0.98: # Threshold for suspiciously high performance
                st.warning("""
                🚨 Suspiciously high performance detected. Possible causes:
                1. Data leakage in embeddings: Embeddings might have been generated using information that should not be available during prediction (e.g., labels).
                2. Test set contains training data: Accidental overlap between training and test sets.
                3. Label contamination in graph: Labels in the graph itself might have influenced embedding generation.
                """)

        return {
            "model": pipeline,
            "X_test": X_test,
            "y_test": y_test,
            "cv_auc": cv_auc,
            "test_auc": test_auc
        }

    except subprocess.TimeoutExpired:
        st.error("❌ Subprocess (embedding generation) timed out.")
        logger.error("Embedding generation timeout during training process.")
        return None
    except subprocess.CalledProcessError as cpe:
        st.error(f"❌ Subprocess failed during training: {cpe.stderr}")
        logger.error(f"Subprocess error during training: {cpe.stderr}", exc_info=True)
        return None
    except Exception as e:
        st.error(f"❌ Error training model: {str(e)}")
        logger.error(f"Training error: {str(e)}", exc_info=True)
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
def train_isolation_forest(num_embeddings: int) -> Optional[Tuple[IsolationForest, StandardScaler]]:
    """
    Trains anomaly detection model.
    The cache_key is based on `num_embeddings` to trigger retraining only when the number of cases changes.
    """
    embeddings = get_existing_embeddings()
    if embeddings is None or len(embeddings) < Config.MIN_CASES_FOR_ANOMALY_DETECTION:
        st.warning(f"⚠️ Need at least {Config.MIN_CASES_FOR_ANOMALY_DETECTION} cases with embeddings for anomaly detection. Found {len(embeddings) if embeddings is not None else 0}.")
        return None

    scaler = StandardScaler()
    embeddings_scaled = scaler.fit_transform(embeddings)

    # Contamination should be estimated or based on domain knowledge.
    # A small fraction of the total cases is a common starting point.
    contamination = min(0.1, 5.0 / len(embeddings)) 
    
    iso_forest = IsolationForest(
        contamination=contamination,
        random_state=Config.RANDOM_STATE,
        n_jobs=-1 # Use all available cores
    )
    iso_forest.fit(embeddings_scaled)

    return iso_forest, scaler

@safe_neo4j_operation
def reinsert_labels_from_csv(csv_url: str) -> bool:
    """Reinserts SCREENED_FOR relationships based on CSV data."""
    try:
        df = pd.read_csv(csv_url, delimiter=";", encoding='utf-8-sig')
        df.columns = [col.strip() for col in df.columns]

        if "Case_No" not in df.columns or "Class_ASD_Traits" not in df.columns:
            st.error("❌ The CSV must contain 'Case_No' and 'Class_ASD_Traits' columns.")
            return False

        with neo4j_service.session() as session:
            data_to_create = []
            for _, row in df.iterrows():
                case_id = int(row["Case_No"])
                label = str(row["Class_ASD_Traits"]).strip().lower()
                if label in ["yes", "no"]:
                    data_to_create.append({"case_id": case_id, "label": label.capitalize()})
            
            # Use UNWIND for efficient batching
            session.run("""
                UNWIND $data AS item
                MATCH (c:Case {id: item.case_id})
                MERGE (t:ASD_Trait {label: item.label})
                MERGE (c)-[:SCREENED_FOR]->(t)
            """, data={"data": data_to_create})
        logger.info("✅ SCREENED_FOR relationships reinserted from CSV.")
        return True
    except Exception as e:
        st.error(f"Failed to reinsert labels from CSV: {e}")
        logger.error(f"Failed to reinsert labels from CSV: {e}", exc_info=True)
        return False

# === Streamlit UI ===
def main():
    st.set_page_config(layout="wide", page_title="NeuroCypher ASD")
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

**"Development of Intelligent Systems for the Early Detection and Management of Developmental Disorders: Combining Biomechanics and Artificial Intelligence"** by Dr. Bouchouras under the supervision of Dr. Kotis.

---
### 🧪 What This App Does

This interactive application functions as a GraphRAG-powered intelligent agent designed for the early 
detection of Autism Spectrum Disorder traits in toddlers. It integrates a Neo4j knowledge graph, 
machine learning, and natural language interfaces powered by GPT-4. The app allows you to:

- 🧠 **Train a machine learning model** to detect ASD traits using graph embeddings.
- 📤 **Upload your own toddler screening data** from the Q-Chat-10 questionnaire and other demographics.
- 🔗 Automatically connect the uploaded case to a knowledge graph **(temporarily for prediction)**.
- 🌐 **Generate a graph-based embedding** for the new case.
- 🔍 **Predict** whether the case shows signs of Autism Spectrum Disorder (ASD).
- 🕵️ **Run anomaly detection** to check if a case is unusual compared to existing data.
- 💬 **Ask natural language questions** and receive Cypher queries with results, using GPT-4 based NLP-to-Cypher translation.

---
### 📥 Download Example CSV

To get started, [download this example CSV](https://raw.githubusercontent.com/GiorgosBouh/llm2cypher-asd-app/main/Toddler_Autism_dataset_July_2018_3_test_39.csv)  
to format your own screening case correctly. 
Also, [read this description](https://raw.githubusercontent.com/GiorgosBouh/llm2cypher-asd-app/main/Toddler_data_description.docx) for further informations about the dataset.
""")

    # Initialize session state variables safely
    # Using st.session_state.get() provides a default if the key doesn't exist.
    st.session_state.active_tab = st.session_state.get("active_tab", "Model Training")
    st.session_state.case_inserted = st.session_state.get("case_inserted", False)
    st.session_state.last_upload_id = st.session_state.get("last_upload_id", None)
    st.session_state.last_case_no = st.session_state.get("last_case_no", None)
    st.session_state.model_trained = st.session_state.get("model_trained", False)
    st.session_state.model_results = st.session_state.get("model_results", None)
    st.session_state.saved_embedding_case1 = st.session_state.get("saved_embedding_case1", None)
    st.session_state.last_cypher_query = st.session_state.get("last_cypher_query", None)
    st.session_state.last_cypher_results = st.session_state.get("last_cypher_results", None)
    st.session_state.preset_question = st.session_state.get("preset_question", "")

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

        missing_labels_initial_check = find_cases_missing_labels()
        if missing_labels_initial_check:
            st.warning(f"⚠️ Υπάρχουν {len(missing_labels_initial_check)} περιπτώσεις χωρίς SCREENED_FOR ετικέτα. Παρακαλώ διορθώστε τα δεδομένα πριν προχωρήσετε.")
        else:
            st.success("✅ Όλες οι περιπτώσεις έχουν SCREENED_FOR ετικέτα.")

        # The cache_key=str(uuid.uuid4()) forces a re-run of the cached function every time this button is clicked.
        # This is desired for a "Train/Refresh" button to ensure new training.
        if st.button("🔄 Train/Refresh Model"):
            with st.spinner("Training model with leakage protection... This can take several minutes."):
                results = train_asd_detection_model(cache_key=str(uuid.uuid4()))
                if results:
                    st.session_state.model_results = results
                    st.session_state.model_trained = True
                    st.success("✅ Training completed successfully. Evaluating model...")
                    # Evaluate model immediately after successful training
                    evaluate_model(
                        results["model"],
                        results["X_test"],
                        results["y_test"]
                    )
                else:
                    st.error("❌ Model training failed. Check logs for details.")
                    st.session_state.model_trained = False
                    st.session_state.model_results = None
        
        # Display evaluation metrics if model is already trained from a previous run or current run
        if st.session_state.get("model_trained") and st.session_state.get("model_results"):
            st.markdown("---")
            st.subheader("Current Model Performance")
            # Re-evaluate model if session state has results (e.g. after a page refresh, retaining results)
            evaluate_model(
                st.session_state.model_results["model"],
                st.session_state.model_results["X_test"],
                st.session_state.model_results["y_test"]
            )

            st.markdown("---")
            with st.expander("🧪 Compare old vs new embeddings (Case 1)"):
                st.markdown("This feature helps verify if `kg_builder_2.py` correctly generates new embeddings. Saving Case 1's embedding then re-running `kg_builder_2.py` and comparing will show if the embedding changed.")
                col_save, col_compare = st.columns(2)
                with col_save:
                    if st.button("📤 Save current embedding of Case 1"):
                        with neo4j_service.session() as session:
                            result = session.run("MATCH (c:Case {id: 1}) RETURN c.embedding AS emb").single()
                            if result and result["emb"]:
                                st.session_state.saved_embedding_case1 = result["emb"]
                                st.success("✅ Saved current embedding of Case 1")
                            else:
                                st.warning("⚠️ Case 1 not found or has no embedding.")

                with col_compare:
                    if st.button("📥 Compare to saved embedding of Case 1"):
                        if st.session_state.saved_embedding_case1 is None:
                            st.error("❌ No saved embedding found. Click 'Save current embedding' first.")
                        else:
                            with neo4j_service.session() as session:
                                result = session.run("MATCH (c:Case {id: 1}) RETURN c.embedding AS emb").single()
                                if result and result["emb"]:
                                    new_emb = np.array(result["emb"])
                                    old_emb = np.array(st.session_state.saved_embedding_case1)
                                    from numpy.linalg import norm
                                    diff = norm(old_emb - new_emb)
                                    st.write(f"📏 Difference (L2 norm) between saved and current embedding: `{diff:.4f}`")
                                    if diff < 1e-6: # Use a small epsilon for floating point comparison
                                        st.info("ℹ️ Embedding is (almost) identical – recent rebuild may not have changed it significantly.")
                                    else:
                                        st.success("✅ Embedding changed – rebuild successfully updated the graph.")
                                else:
                                    st.error("❌ Current embedding for Case 1 not found in graph.")

    # === Tab 2: Graph Embeddings ===
    with tab2:
        st.header("🌐 Graph Embeddings")
        st.warning("⚠️ **Developer Only**: This button triggers a full graph rebuild and embedding recalculation for all existing cases. This is typically done as part of the initial setup or for major graph updates. It can be a long-running process.")
        st.info("ℹ️ **Purpose**: Ensures all cases in the Neo4j graph have up-to-date embeddings generated by `kg_builder_2.py`.")
        
        if st.button("🔁 Recalculate All Embeddings in Graph"):
            st.session_state.model_trained = False # Invalidate model if embeddings are recalculated
            st.session_state.model_results = None
            st.session_state.saved_embedding_case1 = None # Invalidate saved embedding for comparison

            with st.spinner("Running full graph rebuild and embedding generation... This can take a considerable amount of time depending on graph size."):
                result = subprocess.run(
                    [sys.executable, "kg_builder_2.py"],
                    capture_output=True,
                    text=True,
                    timeout=1200 # Increased timeout for full rebuild (20 minutes)
                )
                if result.returncode == 0:
                    st.success("✅ All embeddings recalculated and updated in the graph!")
                    logger.info("Full graph rebuild (kg_builder_2.py) successful.")
                else:
                    st.error(f"❌ Failed to run kg_builder_2.py:\n{result.stderr}")
                    logger.error(f"Full graph rebuild (kg_builder_2.py) failed: {result.stderr}", exc_info=True)
                
            check_embedding_dimensions() # Check consistency after recalculation

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
                1. Download the example CSV template.
                2. Review the data format instructions for each column.
                3. Prepare your single case data according to the template.
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
            - Ensure all required columns are present and correctly named.
            - Upload exactly **one case** at a time per CSV file.
            - Values must match the specified formats (e.g., A1-A10 as integers, Sex as 'm'/'f').
            - **This upload process is for prediction only; the uploaded data is NOT permanently saved to the Neo4j graph.**
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
                    st.error("❌ Please upload exactly one case (one row in the CSV file).")
                    st.stop()

                row = df.iloc[0]
                case_data = row.to_dict()

                if not st.session_state.get("model_trained") or st.session_state.model_results is None:
                    st.error("⚠️ The ASD detection model has not been trained yet. Please go to the 'Model Training' tab and train the model first.")
                    st.stop()

                # Temporary upload_id for embedding generation. This ID will be used by the subprocess
                # to identify this temporary case if it interacts with Neo4j temporarily.
                temp_upload_id = "temp_prediction_" + str(uuid.uuid4())

                # Call subprocess generate_case_embedding.py with upload_id and case_data JSON
                # The generate_case_embedding.py script is expected to take the case_data as JSON
                # and return the embedding as JSON to stdout. It should NOT permanently store data.
                with st.spinner("Generating embedding for prediction... This involves temporary graph interaction."):
                    success, embedding_json_str = generate_embedding_via_subprocess(temp_upload_id, json.dumps(case_data))

                    if not success or embedding_json_str is None:
                        st.error("❌ Embedding generation failed. Check logs for details.")
                        st.stop()

                    try:
                        embedding = np.array(json.loads(embedding_json_str))
                    except json.JSONDecodeError as e:
                        st.error(f"❌ Failed to parse embedding JSON from subprocess output: {e}")
                        st.stop()
                    except Exception as e:
                        st.error(f"❌ Unexpected error processing embedding: {e}")
                        st.stop()

                st.subheader("🧠 Case Embedding (Temporary)")
                st.write(embedding)

                st.subheader("🧪 Embedding Diagnostics")
                st.text("📦 Embedding vector (first 50 dimensions):")
                st.write(embedding[:50].tolist()) # Display only first 50 for brevity

                if np.isnan(embedding).any():
                    st.error("❌ Generated embedding contains NaN values. Prediction may be unreliable.")
                    st.stop()
                else:
                    st.success("✅ Embedding is valid (no NaNs).")

                # *** Prediction ***
                model = st.session_state.model_results["model"]
                embedding_reshaped = embedding.reshape(1, -1)  # Reshape for single sample prediction
                proba = model.predict_proba(embedding_reshaped)[0][1]
                prediction = "ASD Traits Detected" if proba >= 0.5 else "Typical Development"

                st.subheader("🔍 Prediction Result")
                col1, col2 = st.columns(2)
                col1.metric("Prediction", prediction)
                col2.metric("Confidence", f"{proba:.1%}" if prediction == "ASD Traits Detected" else f"{1-proba:.1%}")

                # Distance from training mean
                X_train_for_mean = st.session_state.model_results["X_test"] # Using X_test from training as a sample of the data distribution
                if not X_train_for_mean.empty:
                    train_mean = X_train_for_mean.mean().values
                    dist = np.linalg.norm(embedding - train_mean)
                    st.text(f"📏 Euclidean Distance from Training Data Mean: {dist:.4f}")
                    # A heuristic threshold for warning
                    if dist > 5.0: # This threshold may need adjustment based on data
                        st.warning("⚠️ The generated embedding for this case is significantly distant from the mean of the training data. The prediction might be less reliable for this out-of-distribution case.")
                else:
                    st.info("ℹ️ Cannot calculate distance from training data mean as training data is not available.")

                # Anomaly Detection
                with st.spinner("Running anomaly detection..."):
                    num_existing_embeddings = len(get_existing_embeddings()) if get_existing_embeddings() is not None else 0
                    iso_result = train_isolation_forest(num_existing_embeddings) # Pass num_embeddings as cache key
                    if iso_result:
                        iso_forest, scaler = iso_result
                        embedding_scaled = scaler.transform(embedding_reshaped)
                        anomaly_score = iso_forest.decision_function(embedding_scaled)[0]
                        is_anomaly = iso_forest.predict(embedding_scaled)[0] == -1

                        st.subheader("🕵️ Anomaly Detection")
                        if is_anomaly:
                            st.warning(f"⚠️ This case is detected as an **anomaly** (Isolation Forest score: {anomaly_score:.3f}). It appears unusual compared to the existing data in the graph.")
                        else:
                            st.success(f"✅ This case is considered **normal** (Isolation Forest score: {anomaly_score:.3f}) based on the existing data distribution.")
                    else:
                        st.info("ℹ️ Anomaly detection could not be performed due to insufficient existing embeddings or an error during model training.")

                st.success("✅ Prediction and anomaly detection completed successfully!")

            except Exception as e:
                st.error(f"❌ Error processing uploaded file or generating prediction: {str(e)}")
                logger.exception("Upload case error:")

    # === Tab 4: NLP to Cypher ===
    with tab4:
        st.header("💬 Natural Language to Cypher")
        with st.expander("ℹ️ What can I ask? (Dataset Description & Examples)"):
            st.markdown("""
            ### 📚 Dataset Overview
            This knowledge graph contains screening data for toddlers to help detect potential signs of Autism Spectrum Disorder (ASD).

            #### ✅ Node Types:
            - **Case**: A toddler who was screened.
            - **BehaviorQuestion**: A question from the Q-Chat-10 questionnaire. Attributes: `name` (e.g., "A1").
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

            - **ASD_Trait**: Whether the case was labeled as showing ASD traits. Attributes: `label` (`'Yes'` or `'No'`).
            - **DemographicAttribute**: Characteristics like `Sex`, `Ethnicity`, `Jaundice`, `Family_mem_with_ASD`. Attributes: `type` (e.g., "Sex"), `value` (e.g., "m", "female", "White-European").
            - **SubmitterType**: Who completed the questionnaire. Attributes: `type` (e.g., "Parent", "Health worker").

            #### 🔗 Relationships:
            - `HAS_ANSWER`: A case's answer to a behavioral question. Has a `value` property (int: 0-10).
            - `HAS_DEMOGRAPHIC`: Links a case to demographic attributes.
            - `SUBMITTED_BY`: Who submitted the test.
            - `SCREENED_FOR`: Final ASD classification for a case.

            ### 🧠 Example Questions (Click to use)
            - How many male toddlers have ASD traits?
            - List all ethnicities with more than 5 cases.
            - How many cases answered '1' for both A1 and A2?
            - What is the average A5 score for cases with ASD traits?
            - Which submitter type has the most cases?
            """)

            # Use a single button row for example questions
            st.markdown("<div style='display: flex; flex-wrap: wrap; gap: 10px;'>", unsafe_allow_html=True)
            example_questions = [
                "How many male toddlers have ASD traits?",
                "List all ethnicities with more than 5 cases.",
                "How many cases answered '1' for both A1 and A2?",
                "What is the average A5 score for cases with ASD traits?",
                "Which submitter type has the most cases?"
            ]
            for q in example_questions:
                if st.button(q, key=f"example_{q}"):
                    st.session_state.preset_question = q
                    st.session_state.last_cypher_results = None  # Reset previous results
                    st.session_state.last_cypher_query = None # Reset previous query
                    st.experimental_rerun() # Rerun to update the text_input with the preset question
            st.markdown("</div>", unsafe_allow_html=True)


        default_question = st.session_state.get("preset_question", "")
        question = st.text_input("Ask about the data:", value=default_question, key="nlp_question_input")

        if st.button("▶️ Generate & Execute Cypher Query", key="execute_cypher_button"):
            if question.strip():
                with st.spinner("Translating question to Cypher and executing..."):
                    cypher = nl_to_cypher(question)
                    if cypher:
                        st.session_state.last_cypher_query = cypher
                        try:
                            with neo4j_service.session() as session:
                                results = session.run(cypher).data()
                                st.session_state.last_cypher_results = results
                                st.success("✅ Query executed successfully!")
                        except Exception as e:
                            st.error(f"❌ Query failed: {str(e)}")
                            logger.error(f"Cypher query execution failed: {str(e)}", exc_info=True)
                            st.session_state.last_cypher_results = None
                    else:
                        st.error("❌ Could not translate your question to a Cypher query.")
                        st.session_state.last_cypher_query = None
                        st.session_state.last_cypher_results = None
            else:
                st.warning("Please enter a question.")

        # Display results only if a query was run
        if st.session_state.last_cypher_query:
            st.markdown("---")
            st.subheader("Generated Cypher Query")
            st.code(st.session_state.last_cypher_query, language="cypher")

        if st.session_state.last_cypher_results is not None:
            st.subheader("Query Results")
            if len(st.session_state.last_cypher_results) > 0:
                st.dataframe(pd.DataFrame(st.session_state.last_cypher_results), use_container_width=True)
            else:
                st.info("No results found for this query.")


if __name__ == "__main__":
    main()

```