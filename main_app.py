import streamlit as st
from neo4j import GraphDatabase
from openai import OpenAI
from dotenv import load_dotenv
import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, roc_auc_score,
    roc_curve, precision_recall_curve,
    average_precision_score
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
from typing import Optional, Tuple, List, Dict, Any
from sklearn.preprocessing import StandardScaler
from datetime import datetime

# MUST be the first Streamlit command
st.set_page_config(
    layout="wide",
    page_title="NeuroCypher ASD",
    page_icon="🧠"
)

# === Configuration ===
class Config:
    NODE2VEC_EMBEDDING_DIM = 64
    NODE2VEC_ITERATIONS = 10
    RANDOM_STATE = 42
    TEST_SIZE = 0.2
    N_ESTIMATORS = 100
    SMOTE_RATIO = 'auto'
    MAX_QUERY_RETRIES = 3
    QUERY_RETRY_DELAY = 1

# === Logging Setup ===
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("neurocypher.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# === Helper Decorators ===
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

def log_execution_time(func):
    """Decorator to log function execution time"""
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        logger.info(f"Function {func.__name__} executed in {end_time - start_time:.2f} seconds")
        return result
    return wrapper

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
        self._verify_connection()

    def _verify_connection(self):
        """Verify the Neo4j connection is working"""
        try:
            with self.driver.session() as session:
                session.run("RETURN 1")
        except Exception as e:
            logger.error(f"Failed to connect to Neo4j: {e}")
            raise ConnectionError(f"Could not connect to Neo4j: {e}")

    @contextmanager
    def session(self):
        """Context manager for Neo4j sessions"""
        session = self.driver.session()
        try:
            yield session
        except Exception as e:
            logger.error(f"Session error: {e}")
            raise
        finally:
            session.close()

    def close(self):
        """Close the Neo4j driver"""
        self.driver.close()

    @safe_neo4j_operation
    def execute_query(self, query: str, params: Dict[str, Any] = None, retries: int = Config.MAX_QUERY_RETRIES) -> List[Dict[str, Any]]:
        """Execute a query with retry logic"""
        params = params or {}
        last_error = None
        
        for attempt in range(retries):
            try:
                with self.session() as session:
                    result = session.run(query, **params)
                    return [dict(record) for record in result]
            except Exception as e:
                last_error = e
                logger.warning(f"Query attempt {attempt + 1} failed: {e}")
                if attempt < retries - 1:
                    time.sleep(Config.QUERY_RETRY_DELAY)
        
        logger.error(f"Query failed after {retries} attempts")
        raise last_error if last_error else Exception("Unknown query error")

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
@safe_neo4j_operation
@log_execution_time
def extract_user_embedding(upload_id: str) -> Optional[np.ndarray]:
    """Extracts the embedding for a specific case."""
    result = neo4j_service.execute_query(
        "MATCH (c:Case {upload_id: $upload_id}) RETURN c.embedding AS embedding",
        {"upload_id": upload_id}
    )
    return np.array(result[0]["embedding"]) if result and result[0]["embedding"] else None

@safe_neo4j_operation
@log_execution_time
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

    # Execute all queries
    for query, params in queries:
        neo4j_service.execute_query(query, params)
    
    logger.info(f"Successfully inserted case {upload_id}")

@safe_neo4j_operation
@log_execution_time
def run_node2vec() -> None:
    """Generates Node2Vec embeddings for all cases."""
    # Check and drop existing graph projection if needed
    if neo4j_service.execute_query("CALL gds.graph.exists('asd-graph') YIELD exists")[0]["exists"]:
        neo4j_service.execute_query("CALL gds.graph.drop('asd-graph')")

    # Create new graph projection
    neo4j_service.execute_query(f"""
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
    neo4j_service.execute_query(f"""
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
    neo4j_service.execute_query("CALL gds.graph.drop('asd-graph')")
    logger.info("Node2Vec embedding generation completed")

@safe_neo4j_operation
@log_execution_time
def extract_training_data() -> Tuple[pd.DataFrame, pd.Series]:
    """Extracts training data from Neo4j without score information."""
    records = neo4j_service.execute_query("""
        MATCH (c:Case)-[:SCREENED_FOR]->(t:ASD_Trait)
        WHERE c.embedding IS NOT NULL
        RETURN c.embedding AS embedding, t.value AS label
    """)

    if not records:
        return pd.DataFrame(), pd.Series()

    X = np.array([np.array(r["embedding"]) for r in records])
    y = np.array([1 if r["label"] == "Yes" else 0 for r in records])
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
    plt.close(fig)

@st.cache_resource(show_spinner="Training ASD detection model...")
@log_execution_time
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
        ('smote', SMOTE(random_state=Config.RANDOM_STATE)),
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

    # Evaluation curves
    plot_combined_curves(y_test, y_proba)

    return pipeline.named_steps['classifier']

@safe_neo4j_operation
@log_execution_time
def get_existing_embeddings() -> Optional[np.ndarray]:
    """Retrieves all existing embeddings for anomaly detection."""
    records = neo4j_service.execute_query("""
        MATCH (c:Case)
        WHERE c.embedding IS NOT NULL
        RETURN c.embedding AS embedding
    """)
    embeddings = [record["embedding"] for record in records]
    return np.array(embeddings) if embeddings else None

@st.cache_resource(show_spinner="Training Isolation Forest...")
@log_execution_time
def train_isolation_forest() -> Optional[Tuple[IsolationForest, StandardScaler]]:
    """Trains and returns both the Isolation Forest model and its scaler"""
    embeddings = get_existing_embeddings()
    if embeddings is None or len(embeddings) < 10:
        st.warning("Not enough embeddings for anomaly detection (need at least 10)")
        return None

    scaler = StandardScaler()
    embeddings_scaled = scaler.fit_transform(embeddings)
    contamination_rate = 0.05 if len(embeddings) < 50 else 0.1

    iso_forest = IsolationForest(
        random_state=Config.RANDOM_STATE,
        contamination=contamination_rate,
        n_estimators=100
    )
    iso_forest.fit(embeddings_scaled)
    
    logger.info("Isolation Forest trained successfully")
    return iso_forest, scaler  # Return both model and scaler

def validate_csv(df: pd.DataFrame) -> bool:
    """Validates uploaded CSV file structure without checking score."""
    required_columns = [f"A{i}" for i in range(1, 11)] + [
        "Sex", "Ethnicity", "Jaundice",
        "Family_mem_with_ASD", "Who_completed_the_test"
    ]
    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        st.error(f"Missing required columns: {', '.join(missing_cols)}")
        return False
    
    # Validate behavior question values (should be 0 or 1)
    for i in range(1, 11):
        col = f"A{i}"
        if not all(df[col].isin([0, 1])):
            st.error(f"Column {col} should contain only 0 or 1 values")
            return False
            
    return True

@log_execution_time
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

    Please convert the following natural language question to a Cypher query:
    "{question}"

    Requirements:
    - All value matching (e.g., 'Yes', 'No', 'Female', etc.) should be case-insensitive using `toLower()`
    - Interpret 'f' as 'female' and 'm' as 'male' where relevant (e.g., Sex)
    - Return only the Cypher query, no additional explanation
    """
    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=500
        )
        cypher_query = response.choices[0].message.content.strip()
        return cypher_query.replace("```cypher", "").replace("```", "").strip()
    except Exception as e:
        st.error(f"OpenAI API error: {e}")
        logger.error(f"OpenAI API error: {e}")
        return None

# === Streamlit UI ===
def main():
    st.title("🧠 NeuroCypher ASD")
    st.markdown("""
        <i>The graph is based on Q-Chat-10 plus survey and other individual characteristics
        that have proved to be effective in detecting ASD cases from controls.</i>
        """, unsafe_allow_html=True)

    # Sidebar connection info
    st.sidebar.markdown(f"🔗 **Connected to:** `{os.getenv('NEO4J_URI')}`")
    st.sidebar.markdown(f"📅 **Last updated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

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
    question = st.text_input("📝 Ask your question in natural language:", key="nl_question")

    if question:
        cypher_query = nl_to_cypher(question)
        if cypher_query:
            st.code(cypher_query, language="cypher")

            if st.button("▶️ Run Query", key="run_query"):
                try:
                    results = neo4j_service.execute_query(cypher_query)
                    if results:
                        st.subheader("📊 Query Results:")
                        st.dataframe(pd.DataFrame(results))
                    else:
                        st.info("No results found.")
                except Exception as e:
                    st.error(f"Query execution failed: {e}")

    # === Model Training Section ===
    st.header("🤖 ASD Detection Model")
    if st.button("🔄 Train/Refresh Model", key="train_model"):
        with st.spinner("Training model..."):
            model = train_asd_detection_model()
            if model:
                st.success("Model trained successfully!")
                st.session_state['asd_model'] = model

    # === File Upload Section ===
    st.header("📄 Upload New Case")
    uploaded_file = st.file_uploader("Upload CSV for single child prediction", type="csv", key="file_uploader")

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
                time.sleep(3)  # Give time for embeddings to generate

            # Check if case exists and get embedding
            with st.spinner("Verifying data..."):
                embedding = extract_user_embedding(upload_id)
                if embedding is None:
                    st.error("Failed to generate embedding")
                    st.stop()

            # ASD Prediction
            if 'asd_model' in st.session_state:
                with st.spinner("Predicting ASD traits..."):
                    model = st.session_state['asd_model']
                    proba = model.predict_proba([embedding])[0][1]

                    st.subheader("🛠️ Prediction Threshold")
                    threshold = st.slider(
                        "Select prediction threshold",
                        min_value=0.3,
                        max_value=0.9,
                        value=0.5,
                        step=0.01,
                        key="threshold_slider"
                    )

                    prediction = "YES (ASD Traits Detected)" if proba >= threshold else "NO (Control Case)"

                    st.subheader("🔍 Prediction Result")
                    col1, col2 = st.columns(2)
                    col1.metric("Prediction", prediction)
                    col2.metric(
                        "Confidence", 
                        f"{proba:.1%}" if prediction == "YES (ASD Traits Detected)" else f"{1 - proba:.1%}"
                    )

                    fig = px.bar(
                        x=["Control", "ASD Traits"],
                        y=[1 - proba, proba],
                        labels={'x': 'Class', 'y': 'Probability'},
                        title="Prediction Probabilities"
                    )
                    st.plotly_chart(fig)

            # Anomaly Detection
            with st.spinner("Checking for anomalies..."):
                anomaly_model = train_isolation_forest()
                
                if anomaly_model is not None:
                    iso_forest, scaler = anomaly_model
                    embedding_scaled = scaler.transform([embedding])
                    anomaly_score = iso_forest.decision_function(embedding_scaled)[0]
                    is_anomaly = iso_forest.predict(embedding_scaled)[0] == -1

                    st.subheader("🕵️ Anomaly Detection")
                    if is_anomaly:
                        st.warning(f"⚠️ Anomaly detected (score: {anomaly_score:.3f})")
                        st.markdown("""
                        **Interpretation:**  
                        This case appears unusual compared to others in our database.  
                        Please review carefully as it may represent:
                        - A rare presentation of ASD traits
                        - Uncommon demographic combinations
                        - Potentially incomplete or unusual data
                        """)
                    else:
                        st.success(f"✅ Normal case (score: {anomaly_score:.3f})")
                        st.markdown("This case appears typical compared to others in our database.")

                    # Show distribution of anomaly scores
                    all_embeddings = get_existing_embeddings()
                    if all_embeddings is not None:
                        all_embeddings_scaled = scaler.transform(all_embeddings)
                        scores = iso_forest.decision_function(all_embeddings_scaled)

                        fig = px.histogram(
                            x=scores,
                            nbins=20,
                            labels={'x': 'Anomaly Score'},
                            title="Anomaly Score Distribution",
                            color_discrete_sequence=['#636EFA']
                        )
                        fig.add_vline(
                            x=anomaly_score, 
                            line_dash="dash", 
                            line_color="red",
                            annotation_text="Current Case",
                            annotation_position="top"
                        )
                        fig.update_layout(showlegend=False)
                        st.plotly_chart(fig)
                else:
                    st.warning("""
                    Anomaly detection requires at least 10 existing cases in the database.
                    Currently we don't have enough data to perform this analysis.
                    """)

        except pd.errors.EmptyDataError:
            st.error("The uploaded file is empty or corrupt")
            logger.error("Empty file uploaded")
        except pd.errors.ParserError:
            st.error("Could not parse the CSV file. Please check the format.")
            logger.error("CSV parsing error")
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")
            logger.error(f"File processing error: {str(e)}")

if __name__ == "__main__":
    main()