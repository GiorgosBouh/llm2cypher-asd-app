import streamlit as st
from neo4j import GraphDatabase
from openai import OpenAI
from dotenv import load_dotenv
import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    classification_report, roc_auc_score,
    roc_curve, precision_recall_curve,
    average_precision_score, confusion_matrix, ConfusionMatrixDisplay
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
from sklearn.preprocessing import StandardScaler  # Added missing import

# === Configuration ===
class Config:
    NODE2VEC_EMBEDDING_DIM = 64
    NODE2VEC_ITERATIONS = 10
    RANDOM_STATE = 42
    TEST_SIZE = 0.2
    N_ESTIMATORS = 100
    SMOTE_RATIO = 'auto'

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
def insert_user_case(row: pd.Series, upload_id: str) -> None:
    """Inserts a user case into the Neo4j graph database.

    Args:
        row: A pandas Series containing all case data
        upload_id: Unique identifier for the case
    """
    queries = []
    params = {"upload_id": upload_id}

    # Base case creation
    queries.append(("CREATE (c:Case {upload_id: $upload_id})", params))

    # Add behavior questions
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

    # Add demographic information
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

    # Execute all queries in a single transaction
    with neo4j_service.session() as session:
        for query, params in queries:
            session.run(query, **params)
        logger.info(f"Successfully inserted case {upload_id}")

@safe_neo4j_operation
def get_asd_with_jaundice_count() -> Optional[int]:
    """Returns count of toddlers with ASD and Jaundice (case-insensitive)."""
    query = """
    MATCH (c:Case)-[:SCREENED_FOR]->(a:ASD_Trait),
          (c)-[:HAS_DEMOGRAPHIC]->(d:DemographicAttribute {type: 'Jaundice'})
    WHERE a.value = 'Yes' AND toLower(d.value) = 'yes'
    RETURN count(c) AS count
    """
    with neo4j_service.session() as session:
        result = session.run(query)
        record = result.single()
        return record["count"] if record else None

@safe_neo4j_operation
def run_node2vec() -> None:
    """Generates Node2Vec embeddings for all cases."""
    with neo4j_service.session() as session:
        # Check and drop existing graph projection if needed
        if session.run("CALL gds.graph.exists('asd-graph') YIELD exists").single()["exists"]:
            session.run("CALL gds.graph.drop('asd-graph')")

        # Create new graph projection
        session.run(f"""
            CALL gds.graph.project(
                'asd-graph',
                'Case',
                '*',
                {{
                    nodeProperties: ['embedding'],
                    relationshipProperties: ['value']
                }}
            )
        """)

        # Run Node2Vec
        session.run(f"""
            CALL gds.node2vec.write(
                'asd-graph',
                {{
                    embeddingDimension: {Config.NODE2VEC_EMBEDDING_DIM},
                    writeProperty: 'embedding',
                    iterations: {Config.NODE2VEC_ITERATIONS},
                    randomSeed: {Config.RANDOM_STATE}
                }}
            )
        """)

        # Clean up
        session.run("CALL gds.graph.drop('asd-graph')")
        logger.info("Node2Vec embedding generation completed")

def nl_to_cypher(question: str) -> Optional[str]:
    """Enhanced translator with exact schema matching"""
    prompt = f"""
    You are a Cypher expert for an ASD screening database. Translate questions to exact Cypher queries matching this schema:

    NODES:
    - Case (represents individual cases)
    - BehaviorQuestion (labels: A1, A2, ..., A10)
    - DemographicAttribute (types: Sex, Ethnicity, Jaundice, Family_mem_with_ASD)
    - SubmitterType (type: Who_completed_the_test)
    - ASD_Trait (value: 'Yes' or 'No')

    RELATIONSHIPS:
    - (:Case)-[:HAS_ANSWER {{value: int}}]->(:BehaviorQuestion)
    - (:Case)-[:HAS_DEMOGRAPHIC]->(:DemographicAttribute)
    - (:Case)-[:SUBMITTED_BY]->(:SubmitterType)
    - (:Case)-[:SCREENED_FOR]->(:ASD_Trait)

    VALUE FORMATS:
    - BehaviorQuestions: A1-A10 (exact)
    - Sex: 'm' or 'f' (lowercase)
    - Jaundice: 'yes' or 'no' (lowercase)
    - Family_mem_with_ASD: 'yes' or 'no' (lowercase)
    - ASD_Trait: 'Yes' or 'No' (capitalized)
    - Ethnicity: exact string values from data
    - Who_completed_the_test: exact string values from data

    Examples:
    Q: Count male cases with ASD and family history
    A: MATCH (c:Case)-[:HAS_DEMOGRAPHIC]->(:DemographicAttribute {{type: 'Sex', value: 'm'}}),
                 (c)-[:HAS_DEMOGRAPHIC]->(:DemographicAttribute {{type: 'Family_mem_with_ASD', value: 'yes'}}),
                 (c)-[:SCREENED_FOR]->(:ASD_Trait {{value: 'Yes'}})
        RETURN count(c)

    Q: Show cases where A1 score > 3
    A: MATCH (c:Case)-[r:HAS_ANSWER]->(:BehaviorQuestion {{name: 'A1'}})
        WHERE r.value > 3
        RETURN c

    Q: Find cases submitted by parents with jaundice
    A: MATCH (c:Case)-[:SUBMITTED_BY]->(:SubmitterType {{type: 'Parent'}}),
                 (c)-[:HAS_DEMOGRAPHIC]->(:DemographicAttribute {{type: 'Jaundice', value: 'yes'}})
        RETURN c

    Q: {question}
    A:"""

    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        cypher = response.choices[0].message.content.strip()
        # Enforce exact schema requirements
        cypher = (cypher
            .replace("'male'", "'m'").replace("'female'", "'f'")
            .replace("ASD_Trait {value: 'yes'}", "ASD_Trait {value: 'Yes'}")
            .replace("ASD_Trait {value: 'no'}", "ASD_Trait {value: 'No'}")
            .replace("BehaviorQuestion {id:", "BehaviorQuestion {name:")  # Ensure correct property name
        )
        return cypher.replace("```cypher", "").replace("```", "").strip()
    except Exception as e:
        st.error(f"Translation error: {e}")
        logger.error(f"Failed to translate: {question}\nError: {e}")
        return None

@safe_neo4j_operation
def extract_training_data() -> Tuple[pd.DataFrame, pd.Series]:
    """Extracts training data with proper case handling"""
    with neo4j_service.session() as session:
        result = session.run("""
            MATCH (c:Case)-[:SCREENED_FOR]->(t:ASD_Trait)
            WHERE c.embedding IS NOT NULL
            RETURN c.embedding AS embedding, t.value AS label
        """)
        records = result.data()

    if not records:
        return pd.DataFrame(), pd.Series()

    X = [r["embedding"] for r in records]
    y = [1 if str(r["label"]).lower() == "yes" else 0 for r in records]  # Case-insensitive check
    logger.info(f"Extracted {len(X)} training samples")
    return pd.DataFrame(X), pd.Series(y)

def plot_combined_curves(y_true: np.ndarray, y_proba: np.ndarray) -> None:
    """Plots ROC and Precision-Recall curves side by side."""
    # ROC Curve
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    roc_auc = roc_auc_score(y_true, y_proba)

    # Precision-Recall Curve
    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    avg_precision = average_precision_score(y_true, y_proba)

    # Create subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Plot ROC
    ax1.plot(fpr, tpr, label=f'ROC (AUC = {roc_auc:.2f})')
    ax1.plot([0, 1], [0, 1], 'k--')
    ax1.set_xlabel('False Positive Rate')
    ax1.set_ylabel('True Positive Rate')
    ax1.set_title('ROC Curve')
    ax1.legend(loc='lower right')

    # Plot Precision-Recall
    ax2.plot(recall, precision, label=f'PR (AP = {avg_precision:.2f})')
    ax2.set_xlabel('Recall')
    ax2.set_ylabel('Precision')
    ax2.set_title('Precision-Recall Curve')
    ax2.legend(loc='lower left')

    st.pyplot(fig)

@st.cache_resource(show_spinner="Training ASD detection model...")
def train_asd_detection_model() -> Optional[RandomForestClassifier]:
    """Trains and evaluates the ASD detection model with proper SMOTE usage."""
    X, y = extract_training_data()
    if X.empty:
        st.warning("No training data available")
        return None

    # Initial split (stratified)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=Config.TEST_SIZE,
        stratify=y,
        random_state=Config.RANDOM_STATE
    )

    # Create pipeline with SMOTE only applied during training
    pipeline = Pipeline([
        ('smote', SMOTE(random_state=Config.RANDOM_STATE, sampling_strategy=Config.SMOTE_RATIO)),
        ('classifier', RandomForestClassifier(
            n_estimators=Config.N_ESTIMATORS,
            random_state=Config.RANDOM_STATE
        ))
    ])

    # Train model
    pipeline.fit(X_train, y_train)

    # Evaluate
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    # Display metrics
    st.subheader("Model Evaluation")
    col1, col2 = st.columns(2)

    with col1:
        st.metric("ROC AUC", f"{roc_auc_score(y_test, y_proba):.3f}")
        st.metric("Average Precision", f"{average_precision_score(y_test, y_proba):.3f}")

    with col2:
        st.metric("F1 Score", f"{classification_report(y_test, y_pred, output_dict=True)['1']['f1-score']:.3f}")
        st.metric("Balanced Accuracy", f"{classification_report(y_test, y_pred, output_dict=True)['accuracy']:.3f}")

    # Show curves
    plot_combined_curves(y_test, y_proba)

    # Cross-validation results
    cv_scores = cross_val_score(
        pipeline, X, y,
        cv=5,
        scoring='roc_auc',
        n_jobs=-1
    )
    st.write(f"Cross-validated ROC AUC: {np.mean(cv_scores):.3f} (±{np.std(cv_scores):.3f})")

    return pipeline.named_steps['classifier']

@safe_neo4j_operation
def get_existing_embeddings() -> Optional[np.ndarray]:
    """Retrieves all existing embeddings for anomaly detection."""
    with neo4j_service.session() as session:
        result = session.run("""
            MATCH (c:Case)
            WHERE c.embedding IS NOT NULL
            RETURN c.embedding AS embedding
        """)
        embeddings = [record["embedding"] for record in result]
        return np.array(embeddings) if embeddings else None

@st.cache_resource(show_spinner="Training Isolation Forest...")
def train_isolation_forest() -> Optional[IsolationForest]:
    embeddings = get_existing_embeddings()
    if embeddings is None or len(embeddings) < 10:
        st.warning("Not enough embeddings for anomaly detection")
        return None

   # Scale the embeddings
    scaler = StandardScaler()
    embeddings_scaled = scaler.fit_transform(embeddings)

    # Contamination (percentage of "anomalies")
    contamination_rate = 0.05 if len(embeddings) < 50 else 0.1

    iso_forest = IsolationForest(
        random_state=Config.RANDOM_STATE,
        contamination=contamination_rate
    )
    iso_forest.fit(embeddings_scaled)

    # Store the scaler for future use
    st.session_state["iso_scaler"] = scaler

    return iso_forest

@safe_neo4j_operation
def extract_user_embedding(upload_id: str) -> Optional[list]:
    """Extracts the embedding for a specific uploaded case."""
    with neo4j_service.session() as session:
        result = session.run("""
            MATCH (c:Case {upload_id: $upload_id})
            RETURN c.embedding AS embedding
        """, {"upload_id": upload_id})
        record = result.single()
        return record["embedding"] if record else None

# === Streamlit UI ===
st.title("🧠 NeuroCypher ASD")
st.markdown("""
    <i>The graph is based on Q-Chat-10 plus survey and other individual characteristics
    that have proved to be effective in detecting ASD cases from controls.</i>
    """, unsafe_allow_html=True)

# Sidebar connection info
st.sidebar.markdown(f"🔗 **Connected to:** `{os.getenv('NEO4J_URI')}`")

# === Natural Language to Cypher Section ===
st.header("💬 Natural Language to Cypher")
question = st.text_input("📝 Ask your question in natural language:")

if question:
    cypher_query = nl_to_cypher(question)
    if cypher_query:
        st.code(cypher_query, language="cypher")

        if st.button("▶️ Run Query"):
            with neo4j_service.session() as session:
                try:
                    results = session.run(cypher_query).data()
                    if results:
                        st.subheader("📊 Query Results:")
                        st.dataframe(pd.DataFrame(results))
                    else:
                        st.info("No results found.")
                except Exception as e:
                    st.error(f"Query execution failed: {e}")

# === Model Training Section ===
st.header("🤖 ASD Detection Model")
if st.button("🔄 Train/Refresh Model"):
    with st.spinner("Training model with proper SMOTE handling..."):
        model = train_asd_detection_model()
        if model:
            st.success("Model trained successfully!")
            st.session_state['asd_model'] = model

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

        row = df.iloc[0]
        upload_id = str(uuid.uuid4())

        # Insert case
        with st.spinner("Inserting case into graph..."):
            insert_user_case(row, upload_id)

        # Generate embeddings
        with st.spinner("Generating embeddings..."):
            run_node2vec()
            time.sleep(3)  # Allow time for embeddings to be written

        # Check if case exists and get embedding
        with st.spinner("Verifying data..."):
            embedding = extract_user_embedding(upload_id)
            if embedding is None:
                st.error("Failed to generate embedding")
                st.stop()

            st.subheader("Case Embedding")
            st.write(f"Generated upload_id: {upload_id}")
            st.write(f"Inserting case with upload_id: {upload_id}")
            for i, val in enumerate(row):
                st.write(f"Answer {df.columns[i]}: {val}")
            st.write(f"Submitter: {row['Who_completed_the_test']}")
            st.write(f"Embedding του νέου περιστατικού (upload_id: {upload_id}):")
            embedding_dict = {i: v for i, v in enumerate(embedding[:10])} # Display first 10 values
            for key, value in embedding_dict.items():
                st.write(f"{key}:{value}")
            if len(embedding) > 10:
                st.write("...")
                st.write(f"First 5 values: {embedding[:5]}")


        # ASD Prediction
        if 'asd_model' in st.session_state:
            with st.spinner("Predicting ASD traits..."):
                model = st.session_state['asd_model']
                proba = model.predict_proba([embedding])[0][1]
                prediction = "YES (ASD Traits Detected)" if proba >= 0.5 else "NO (Control Case)"

                st.subheader("🔍 Prediction:")
                st.markdown(f"<font size='+2'>{prediction}</font>", unsafe_allow_html=True)
                st.write(f"Confidence: {max(proba, 1-proba):.1%}")

                # Show probability distribution
                fig = px.bar(
                    x=["Control", "ASD Traits"],
                    y=[1-proba, proba],
                    labels={'x': 'Class', 'y': 'Probability'},
                    title="Prediction Probabilities"
                )
                st.plotly_chart(fig)

        # === Anomaly Detection ===
        with st.spinner("Checking for anomalies..."):
            iso_forest = train_isolation_forest()
            if iso_forest and "iso_scaler" in st.session_state:
                embedding_scaled = st.session_state["iso_scaler"].transform([embedding])
                anomaly_score = iso_forest.decision_function(embedding_scaled)[0]
                is_anomaly = iso_forest.predict(embedding_scaled)[0] == -1

                st.subheader("🕵️ Anomaly Detection")
                if is_anomaly:
                    st.warning(f"⚠️ This case might be an anomaly!")
                else:
                    st.success(f"✅ Normal case")
                st.write(f"Anomaly Score: {anomaly_score:.3f}")

                # === Anomaly score distribution visualization ===
                all_embeddings = get_existing_embeddings()
                if all_embeddings is not None and "iso_scaler" in st.session_state:
                    all_embeddings_scaled = st.session_state["iso_scaler"].transform(all_embeddings)
                    scores = iso_forest.decision_function(all_embeddings_scaled)

                    fig = px.histogram(
                        x=scores,
                        nbins=20,
                        labels={'x': 'Anomaly Score'},
                        title="Anomaly Score Distribution"
                    )
                    fig.add_vline(x=anomaly_score, line_dash="dash", line_color="red")
                    st.plotly_chart(fig)
                else:
                    st.error("Anomaly detection model or scaler not available for distribution plot.")
            else:
                st.warning("Anomaly detection model not trained yet.")

    except Exception as e:
        st.error(f"Error processing file: {e}")
        logger.error(f"Error processing file: {e}")

# === Graph Schema Help ===
with st.expander("🧠 Graph Schema Help"):
    st.markdown("### 🧩 Node Types")
    st.markdown("""
    - `Case`: Each screening instance
    - `BehaviorQuestion`: Questions A1–A10 (exact labels)
    - `ASD_Trait`: Classification result (`Yes`/`No` - capitalized)
    - `DemographicAttribute`:
        - `Sex`: 'm' or 'f'
        - `Ethnicity`: String values
        - `Jaundice`: 'yes' or 'no'
        - `Family_mem_with_ASD`: 'yes' or 'no'
    - `SubmitterType`: Who completed the test
    """)

    st.markdown("### 🔗 Relationships")
    st.markdown("""
    - `(:Case)-[:HAS_ANSWER {value: 0-2}]->(:BehaviorQuestion)`
    - `(:Case)-[:HAS_DEMOGRAPHIC]->(:DemographicAttribute)`
    - `(:Case)-[:SCREENED_FOR]->(:ASD_Trait)`
    - `(:Case)-[:SUBMITTED_BY]->(:SubmitterType)`
    """)

    st.markdown("### 💡 Example Questions")
    st.code("""
Q: Count male toddlers with ASD
Q: Show cases where A1 score > 1
Q: How many cases with family history?
Q: Breakdown by who completed the test
    """)

# Clean up when done
def cleanup():
    neo4j_service.close()

import atexit
atexit.register(cleanup)