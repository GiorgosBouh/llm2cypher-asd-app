import streamlit as st

# MUST be the first Streamlit command
st.set_page_config(layout="wide")
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
from sklearn.preprocessing import StandardScaler
import shap  # Moved to top with other imports

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
    """Inserts a user case into the Neo4j graph database."""
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
def run_node2vec() -> None:
    """Generates Node2Vec embeddings for all cases."""
    with neo4j_service.session() as session:
        # Check and drop existing graph projection if needed
        if session.run("CALL gds.graph.exists('asd-graph') YIELD exists").single()["exists"]:
            session.run("CALL gds.graph.drop('asd-graph')")

        # Create new graph projection with explicit property types
        try:
            session.run(f"""
                CALL gds.graph.project(
                    'asd-graph',
                    'Case',
                    '*',
                    {{
                        nodeProperties: [
                            'embedding', // Existing embedding
                            {{ property: 'A1', type: 'float' }},
                            {{ property: 'A2', type: 'float' }},
                            {{ property: 'A3', type: 'float' }},
                            {{ property: 'A4', type: 'float' }},
                            {{ property: 'A5', type: 'float' }},
                            {{ property: 'A6', type: 'float' }},
                            {{ property: 'A7', type: 'float' }},
                            {{ property: 'A8', type: 'float' }},
                            {{ property: 'A9', type: 'float' }},
                            {{ property: 'A10', type: 'float' }},
                            {{ property: 'Age_Mons', type: 'float' }},
                            {{ property: 'Qchat-10-Score', type: 'float' }},
                            {{ property: 'Sex', type: 'string' }},
                            {{ property: 'Ethnicity', type: 'string' }},
                            {{ property: 'Jaundice', type: 'string' }},
                            {{ property: 'Family_mem_with_ASD', type: 'string' }},
                            {{ property: 'Who_completed_the_test', type: 'string' }}
                            // Add 'upload_id' if it's consistently a string
                        ],
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

        except Exception as e:
            st.error(f"Error during Node2Vec embedding generation: {e}")
            logger.error(f"Error during Node2Vec embedding generation: {e}")
            raise

def nl_to_cypher(question: str) -> Optional[str]:
    """Translates natural language to Cypher using OpenAI."""
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
    Please make sure:
    - All value matching (e.g., 'Yes', 'No', 'Female', etc.) is case-insensitive using `toLower()`
    - Interpret 'f' as 'female' and 'm' as 'male' where relevant (e.g., Sex)
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

@safe_neo4j_operation
def extract_user_embedding(upload_id: str) -> Optional[np.ndarray]:
    """Extracts the embedding for a specific case."""
    with neo4j_service.session() as session:
        result = session.run(
            "MATCH (c:Case {upload_id: $upload_id}) RETURN c.embedding AS embedding",
            upload_id=upload_id
        )
        record = result.single()
        return np.array(record["embedding"]) if record and record["embedding"] else None

@safe_neo4j_operation
def extract_training_data() -> Tuple[pd.DataFrame, pd.Series]:
    """Extracts training data from Neo4j."""
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
    y = [1 if r["label"] == "Yes" else 0 for r in records]
    logger.info(f"Extracted {len(X)} training samples")
    return pd.DataFrame(X), pd.Series(y)

def plot_combined_curves(y_true: np.ndarray, y_proba: np.ndarray) -> None:
    """Plots ROC and Precision-Recall curves side by side."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # ROC Curve
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    roc_auc = roc_auc_score(y_true, y_proba)
    ax1.plot(fpr, tpr, label=f'ROC (AUC = {roc_auc:.2f})')
    ax1.plot([0, 1], [0, 1], 'k--')
    ax1.set_xlabel('False Positive Rate')
    ax1.set_ylabel('True Positive Rate')
    ax1.set_title('ROC Curve')
    ax1.legend(loc='lower right')

    # Precision-Recall Curve
    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    avg_precision = average_precision_score(y_true, y_proba)
    ax2.plot(recall, precision, label=f'PR (AP = {avg_precision:.2f})')
    ax2.set_xlabel('Recall')
    ax2.set_ylabel('Precision')
    ax2.set_title('Precision-Recall Curve')
    ax2.legend(loc='lower left')

    st.pyplot(fig)

@st.cache_resource(show_spinner="Training ASD detection model...")
def train_asd_detection_model() -> Optional[RandomForestClassifier]:
    X, y = extract_training_data()
    if X.empty:
        st.warning("No training data available")
        return None

    st.subheader("📊 Class Distribution")
    st.write(Counter(y))
    st.markdown("""
    - **`0`** 🟢 → No ASD Traits
    - **`1`** 🔴 → ASD Traits
    """)

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
            random_state=Config.RANDOM_STATE
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

    # SHAP explainability
    st.subheader("🧠 Feature Importance (SHAP Values)")
    try:
        explainer = shap.Explainer(pipeline.named_steps['classifier'])
        shap_values = explainer(X_train)
        shap.summary_plot(shap_values, X_train, plot_type="bar", show=False)
        st.pyplot(bbox_inches='tight')
    except Exception as e:
        st.error(f"❌ SHAP analysis failed: {e}")
        logger.error(f"SHAP error: {e}")

    # Evaluation curves
    plot_combined_curves(y_test, y_proba)

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

    scaler = StandardScaler()
    embeddings_scaled = scaler.fit_transform(embeddings)
    contamination_rate = 0.05 if len(embeddings) < 50 else 0.1

    iso_forest = IsolationForest(
        random_state=Config.RANDOM_STATE,
        contamination=contamination_rate
    )
    iso_forest.fit(embeddings_scaled)
    .state["iso_scaler"] = scaler
    return iso_forest

def validate_csv(df: pd.DataFrame) -> bool:
    """Validates uploaded CSV file structure."""
    required_columns = [f"A{i}" for i in range(1, 11)] + [
        "Sex", "Ethnicity", "Jaundice",
        "Family_mem_with_ASD", "Who_completed_the_test"
    ]
    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        st.error(f"Missing required columns: {', '.join(missing_cols)}")
        return False
    return True

# === Streamlit UI ===
st.title("🧠 NeuroCypher ASD")
st.markdown("""
    <i>The graph is based on Q-Chat-10 plus survey and other individual characteristics
    that have proved to be effective in detecting ASD cases from controls.</i>
    """, unsafe_allow_html=True)

# Sidebar connection info
st.sidebar.markdown(f"🔗 **Connected to:** `{os.getenv('NEO4J_URI')}`")

with st.expander("ℹ️ Help: KG Schema & Example Questions", expanded=False):
    st.markdown("### 🧠 Knowledge Graph Schema")
    st.code("""
(:Case)-[:HAS_ANSWER {value: int}]->(:BehaviorQuestion {name: string})
(:Case)-[:HAS_DEMOGRAPHIC]->(:DemographicAttribute {type: string, value: string})
(:Case)-[:SUBMITTED_BY]->(:SubmitterType {type: string})
(:Case)-[:SCREENED_FOR]->(:ASD_Trait {value: 'Yes' | 'No'})
""", language="cypher")

    st.markdown("### 💡 Example Questions")
    st.markdown("""
- How many children were diagnosed with ASD?
- Count cases where A1 equals 1 and ASD is Yes.
- How many female toddlers have ASD traits?
- What is the most common ethnicity among non-ASD children?
- How many children have jaundice and ASD?
""")

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

        # Insert case
        with st.spinner("Inserting case into graph..."):
            insert_user_case(row, upload_id)

        # Generate embeddings
        with st.spinner("Generating embeddings..."):
            run_node2vec()
            time.sleep(3)

        # Check if case exists and get embedding
        with st.spinner("Verifying data..."):
            embedding = extract_user_embedding(upload_id)
            if embedding is None:
                st.error("Failed to generate embedding")
                st.stop()

            # Retrieve qchat_score from graph
            with neo4j_service.session() as session:
                record = session.run("""
                    MATCH (c:Case {upload_id: $upload_id})
                    RETURN c.qchat_score AS score
                """, upload_id=upload_id).single()

            qchat_score = record.get("score") if record else None
            if qchat_score is None:
                try:
                    qchat_score = sum(int(row[f"A{i}"]) for i in range(1, 11))
                    st.info("ℹ️ Q-chat score was missing from the graph. Calculated from uploaded data.")
                except Exception as e:
                    st.error("❌ Failed to calculate Q-chat score from data.")
                    logger.error(f"Fallback qchat_score calculation failed: {e}")

        # Display the score
        if qchat_score is not None:
            st.subheader("🧶 Q-Chat-10 Score")
            st.write(f"Score: **{qchat_score} / 10**")

            if qchat_score <= 3:
                st.success("✅ Based on Q-Chat-10, this child is not expected to show ASD traits (Score ≤ 3).")
            else:
                st.warning("⚠️ Based on Q-Chat-10, this child may present ASD traits (Score > 3).")
        else:
            st.error("⚠️ Q-chat score not found for this case.")

        # ASD Prediction
        if 'asd_model' in st.session_state:
            with st.spinner("Predicting ASD traits..."):
                model = st.session_state['asd_model']
                proba = model.predict_proba([embedding])[0][1]

                st.subheader("🛠️ Prediction Threshold")
                threshold = st.slider("Select prediction threshold", min_value=0.3, max_value=0.9, value=0.5, step=0.01)

                prediction = "YES (ASD Traits Detected)" if proba >= threshold else "NO (Control Case)"

                st.subheader("🔍 Prediction Result")
                col1, col2 = st.columns(2)
                col1.metric("Prediction", prediction)
                col2.metric("Confidence", f"{proba:.1%}" if prediction == "YES (ASD Traits Detected)" else f"{1 - proba:.1%}")

                fig = px.bar(
                    x=["Control", "ASD Traits"],
                    y=[1 - proba, proba],
                    labels={'x': 'Class', 'y': 'Probability'},
                    title="Prediction Probabilities"
                )
                st.plotly_chart(fig)

        # Anomaly Detection
        with st.spinner("Checking for anomalies..."):
            iso_forest = train_isolation_forest()
            if iso_forest and "iso_scaler" in st.session_state:
                embedding_scaled = st.session_state["iso_scaler"].transform([embedding])
                anomaly_score = iso_forest.decision_function(embedding_scaled)[0]
                is_anomaly = iso_forest.predict(embedding_scaled)[0] == -1

                st.subheader("🕵️ Anomaly Detection")
                if is_anomaly:
                    st.warning(f"⚠️ Anomaly detected (score: {anomaly_score:.3f})")
                else:
                    st.success(f"✅ Normal case (score: {anomaly_score:.3f})")

                all_embeddings = get_existing_embeddings()
                if all_embeddings is not None:
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
                st.error("Anomaly detection model or scaler not available.")

    except Exception as e:
        st.error(f"Error processing file: {e}")
        logger.error(f"Error processing file: {e}")