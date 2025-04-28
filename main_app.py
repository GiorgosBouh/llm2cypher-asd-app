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
from sklearn.decomposition import PCA

# === Configuration ===
class Config:
    EMBEDDING_DIM = 16  # Reduced from original Node2Vec dimension
    RANDOM_STATE = 42
    TEST_SIZE = 0.2
    N_ESTIMATORS = 100
    SMOTE_RATIO = 'auto'
    MIN_CASES_FOR_ANOMALY_DETECTION = 10
    PRE_COMPUTED_PCA_COMPONENTS = None  # Will be initialized if available
    FEATURE_SCALER = None  # Will be initialized if available

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

def get_feature_vector(answers: dict, demographics: dict, submitter_type: str) -> np.ndarray:
    """Convert case data to a numerical feature vector."""
    feature_vector = []
    
    # Add answers (A1-A10)
    for i in range(1, 11):
        feature_vector.append(answers.get(f"A{i}", 0))
    
    # Add demographics
    # Sex: 0 for male, 1 for female
    sex = demographics.get("Sex", "").lower()
    feature_vector.append(1 if sex in ["f", "female"] else 0)
    
    # Ethnicity: encoded categories
    ethnicity = demographics.get("Ethnicity", "").lower()
    ethnicities = ["asian", "hispanic", "black", "white", "middle eastern", "mixed", "south asian"]
    eth_encoded = ethnicities.index(ethnicity) if ethnicity in ethnicities else len(ethnicities)
    feature_vector.append(eth_encoded)
    
    # Jaundice: 0 for no, 1 for yes
    jaundice = demographics.get("Jaundice", "").lower()
    feature_vector.append(1 if jaundice == "yes" else 0)
    
    # Family history: 0 for no, 1 for yes
    family = demographics.get("Family_mem_with_ASD", "").lower()
    feature_vector.append(1 if family == "yes" else 0)
    
    # Submitter type: encoded categories
    submitter_types = ["family member", "health care professional", "self", "others", "school and/or child care staff"]
    submitter = submitter_type.lower() if submitter_type else ""
    sub_encoded = submitter_types.index(submitter) if submitter in submitter_types else len(submitter_types)
    feature_vector.append(sub_encoded)
    
    return np.array(feature_vector)

@safe_neo4j_operation
def generate_case_embedding(upload_id: str) -> None:
    """Generates an embedding for a case based on its features."""
    with neo4j_service.session() as session:
        # Get all answers for the case
        result = session.run("""
            MATCH (c:Case {upload_id: $upload_id})-[r:HAS_ANSWER]->(q:BehaviorQuestion)
            RETURN q.name AS question, r.value AS value
            ORDER BY q.name
        """, upload_id=upload_id)
        answers = {record["question"]: record["value"] for record in result}
        
        # Get demographics for the case
        result = session.run("""
            MATCH (c:Case {upload_id: $upload_id})-[:HAS_DEMOGRAPHIC]->(d:DemographicAttribute)
            RETURN d.type AS type, d.value AS value
        """, upload_id=upload_id)
        demographics = {record["type"]: record["value"] for record in result}
        
        # Get submitter type
        result = session.run("""
            MATCH (c:Case {upload_id: $upload_id})-[:SUBMITTED_BY]->(s:SubmitterType)
            RETURN s.type AS type
        """, upload_id=upload_id)
        submitter_type = result.single()["type"] if result else None
        
        # Convert to feature vector
        feature_vector = get_feature_vector(answers, demographics, submitter_type)
        
        # Generate embedding
        if Config.PRE_COMPUTED_PCA_COMPONENTS is not None and Config.FEATURE_SCALER is not None:
            # Scale features using pre-computed scaler
            features_scaled = Config.FEATURE_SCALER.transform([feature_vector])
            # Project using pre-computed PCA components
            embedding = np.dot(features_scaled, Config.PRE_COMPUTED_PCA_COMPONENTS.T)[0]
        else:
            # Fallback to simple approach - use raw features (truncated or padded)
            embedding = feature_vector[:Config.EMBEDDING_DIM]
            if len(embedding) < Config.EMBEDDING_DIM:
                embedding = np.pad(embedding, (0, Config.EMBEDDING_DIM - len(embedding)))
        
        # Store the embedding in the graph
        session.run("""
            MATCH (c:Case {upload_id: $upload_id})
            SET c.embedding = $embedding
        """, upload_id=upload_id, embedding=embedding.tolist())
        
        logger.info(f"Generated embedding for case {upload_id}")

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
    Interpret 'f' as 'female' and 'm' as 'male' where relevant (e.g., Sex).
Always use `toLower()` for case-insensitive value matching (e.g., toLower(d.value) = 'yes')

    Translate the following natural language question to Cypher:

    Q: {question}

    Only return the Cypher query.
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
        return np.array(record["embedding"]) if record and record["embedding"] is not None else None

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
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    roc_auc = roc_auc_score(y_true, y_proba)

    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    avg_precision = average_precision_score(y_true, y_proba)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    ax1.plot(fpr, tpr, label=f'ROC (AUC = {roc_auc:.2f})')
    ax1.plot([0, 1], [0, 1], 'k--')
    ax1.set_xlabel('False Positive Rate')
    ax1.set_ylabel('True Positive Rate')
    ax1.set_title('ROC Curve')
    ax1.legend(loc='lower right')

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

# === Initialize PCA Components (if available) ===
@st.cache_resource
def initialize_pca_components():
    """Initialize PCA components if we have training data available"""
    try:
        X, _ = extract_training_data()
        if len(X) > Config.EMBEDDING_DIM:
            # Get raw features for all cases
            with neo4j_service.session() as session:
                result = session.run("""
                    MATCH (c:Case)-[r:HAS_ANSWER]->(q:BehaviorQuestion)
                    WITH c, q.name AS question, r.value AS value
                    ORDER BY c, question
                    RETURN c, collect(value) AS answers
                """)
                all_answers = {record["c"].id: record["answers"] for record in result}
                
                result = session.run("""
                    MATCH (c:Case)-[:HAS_DEMOGRAPHIC]->(d:DemographicAttribute)
                    RETURN c, d.type AS type, d.value AS value
                """)
                all_demographics = {}
                for record in result:
                    if record["c"].id not in all_demographics:
                        all_demographics[record["c"].id] = {}
                    all_demographics[record["c"].id][record["type"]] = record["value"]
                
                result = session.run("""
                    MATCH (c:Case)-[:SUBMITTED_BY]->(s:SubmitterType)
                    RETURN c, s.type AS type
                """)
                all_submitters = {record["c"].id: record["type"] for record in result}
                
            # Create feature matrix
            features = []
            for case_id in all_answers.keys():
                answers = all_answers.get(case_id, [0]*10)
                demographics = all_demographics.get(case_id, {})
                submitter = all_submitters.get(case_id, "")
                feature_vector = get_feature_vector(
                    {f"A{i+1}": answers[i] for i in range(10)},
                    demographics,
                    submitter
                )
                features.append(feature_vector)
            
            features = np.array(features)
            
            # Fit scaler and PCA
            scaler = StandardScaler().fit(features)
            pca = PCA(n_components=Config.EMBEDDING_DIM).fit(scaler.transform(features))
            
            return pca.components_, scaler
    except Exception as e:
        logger.error(f"Error initializing PCA components: {e}")
    return None, None

# Initialize PCA components if possible
Config.PRE_COMPUTED_PCA_COMPONENTS, Config.FEATURE_SCALER = initialize_pca_components()

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
    with st.spinner("Training model..."):
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

# === CSV Upload ===
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

        with st.spinner("Generating embedding..."):
            generate_case_embedding(upload_id)
            embedding = extract_user_embedding(upload_id)
            if embedding is None:
                st.error("Failed to generate embedding")
                st.stop()

            st.subheader("🧠 Case Embedding")
            st.write(embedding)

        # === ASD Prediction ===
        if 'asd_model' in st.session_state:
            with st.spinner("Predicting ASD traits..."):
                model = st.session_state['asd_model']
                proba = model.predict_proba([embedding])[0][1]
                prediction = "YES (ASD Traits Detected)" if proba >= 0.5 else "NO (Control Case)"

                st.subheader("🔍 Prediction Result")
                col1, col2 = st.columns(2)
                col1.metric("Prediction", prediction)
                col2.metric("Confidence", f"{max(proba, 1 - proba):.1%}")

                fig = px.bar(
                    x=["Control", "ASD Traits"],
                    y=[1 - proba, proba],
                    labels={'x': 'Class', 'y': 'Probability'},
                    title="Prediction Probabilities"
                )
                st.plotly_chart(fig, key=f"prediction_bar_{upload_id}")

        # === Anomaly Detection ===
        with st.spinner("Checking for anomalies..."):
            iso_forest_scaler = train_isolation_forest()
            if iso_forest_scaler:
                iso_forest, scaler = iso_forest_scaler
                embedding_scaled = scaler.transform([embedding])
                anomaly_score = iso_forest.decision_function(embedding_scaled)[0]
                is_anomaly = iso_forest.predict(embedding_scaled)[0] == -1

                st.subheader("🕵️ Anomaly Detection")
                if is_anomaly:
                    st.warning(f"⚠️ Anomaly detected (score: {anomaly_score:.3f})")
                else:
                    st.success(f"✅ Normal case (score: {anomaly_score:.3f})")

                all_embeddings = get_existing_embeddings()
                if all_embeddings is not None:
                    all_embeddings_scaled = scaler.transform(all_embeddings)
                    scores = iso_forest.decision_function(all_embeddings_scaled)

                    fig = px.histogram(
                        x=scores,
                        nbins=20,
                        labels={'x': 'Anomaly Score'},
                        title="Anomaly Score Distribution"
                    )
                    fig.add_vline(x=anomaly_score, line_dash="dash", line_color="red")
                    st.plotly_chart(fig, key=f"anomaly_hist_{upload_id}")
            else:
                st.info("Anomaly detection model not trained yet or insufficient data.")

    except Exception as e:
        st.error(f"❌ Error processing file: {e}")
        logger.error(f"❌ Exception during upload processing: {e}")