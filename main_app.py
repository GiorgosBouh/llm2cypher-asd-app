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
from node2vec import Node2Vec
import networkx as nx
import tempfile
import shutil

# === Configuration ===
class Config:
    EMBEDDING_DIM = 64
    RANDOM_STATE = 42
    TEST_SIZE = 0.2
    N_ESTIMATORS = 100
    SMOTE_RATIO = 'auto'
    MIN_CASES_FOR_ANOMALY_DETECTION = 10
    NODE2VEC_WALK_LENGTH = 20  # Reduced for faster processing
    NODE2VEC_NUM_WALKS = 50    # Reduced for faster processing
    NODE2VEC_WORKERS = 1       # Single worker for reliability
    NODE2VEC_P = 1
    NODE2VEC_Q = 1
    EMBEDDING_BATCH_SIZE = 50  # Smaller batches for reliability
    MAX_RELATIONSHIPS = 100000 # Safety limit for large graphs

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
    params = {"upload_id": upload_id, "id": int(row["Case_No"])}

    queries.append(("CREATE (c:Case {upload_id: $upload_id, id: $id})", params))

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
def generate_embedding_for_node(upload_id: str) -> bool:
    """Generate embedding only for the new case node."""
    with neo4j_service.session() as session:
        # Build local subgraph around this case node
        result = session.run("""
            MATCH (c:Case {upload_id: $upload_id})-[r]-(n)
            RETURN elementId(c) AS case_id, collect(elementId(n)) AS neighbors
        """, upload_id=upload_id)
        
        record = result.single()
        if not record:
            return False

        case_id = record["case_id"]
        neighbors = record["neighbors"]

        G = nx.Graph()
        G.add_node(case_id)
        for neighbor in neighbors:
            G.add_node(neighbor)
            G.add_edge(case_id, neighbor)

        # Generate Node2Vec
        node2vec = Node2Vec(
            G,
            dimensions=Config.EMBEDDING_DIM,
            walk_length=10,
            num_walks=20,
            workers=1,
            seed=Config.RANDOM_STATE,
            quiet=True
        )
        model = node2vec.fit(window=5, min_count=1, batch_words=4, epochs=1)

        if case_id not in model.wv:
            return False

        embedding = model.wv[case_id].tolist()
        session.run("""
            MATCH (c:Case {upload_id: $upload_id})
            SET c.embedding = $embedding
        """, upload_id=upload_id, embedding=embedding)

        return True

@safe_neo4j_operation
def generate_graph_embeddings() -> bool:
    """Generates graph embeddings using Node2Vec with proper resource handling."""
    progress_bar = st.progress(0)
    status_text = st.empty()
    temp_dir = None
    model = None
    
    try:
        # Step 0: Remove label leakage
        with neo4j_service.session() as session:
            session.run("MATCH (c:Case)-[r:SCREENED_FOR]->(:ASD_Trait) DELETE r")
            logger.info("🔒 Removed SCREENED_FOR relationships to prevent label leakage")
        # Create temp directory
        temp_dir = tempfile.mkdtemp(prefix="node2vec_")
        logger.info(f"Created temp directory at {temp_dir}")
        
        # Step 1: Extract graph structure
        status_text.text("Step 1/4: Extracting graph structure...")
        progress_bar.progress(10)
        
        with neo4j_service.session() as session:
            # Get nodes first
            node_result = session.run("MATCH (n)RETURN elementId(n) AS node_id")
            node_ids = [record["node_id"] for record in node_result]
            
            if not node_ids:
                st.error("No nodes found in the graph")
                return False
                
            # Get relationships with safety limit
            rel_result = session.run(f"""
                MATCH (n)-[r]->(m)
                RETURN id(n) as source, id(m) as target
                LIMIT {Config.MAX_RELATIONSHIPS}
            """)
            edges = [(record["source"], record["target"]) for record in rel_result]

        # Step 2: Create NetworkX graph
        status_text.text(f"Step 2/4: Creating graph ({len(node_ids)} nodes, {len(edges)} edges)...")
        progress_bar.progress(30)
        
        G = nx.Graph()
        G.add_nodes_from(node_ids)
        
        # Add edges in chunks to avoid memory issues
        chunk_size = 10000
        for i in range(0, len(edges), chunk_size):
            G.add_edges_from(edges[i:i+chunk_size])
            progress = 30 + int(20 * min(i/len(edges), 1))
            progress_bar.progress(progress)
        
        # Remove self-loops if any
        G.remove_edges_from(nx.selfloop_edges(G))
        
        # Step 3: Generate embeddings
        status_text.text("Step 3/4: Generating embeddings... (This may take 3-5 minutes)")
        progress_bar.progress(50)
        
        # Initialize Node2Vec with conservative parameters
        node2vec = Node2Vec(
            G,
            dimensions=Config.EMBEDDING_DIM,
            walk_length=Config.NODE2VEC_WALK_LENGTH,
            num_walks=Config.NODE2VEC_NUM_WALKS,
            workers=Config.NODE2VEC_WORKERS,
            p=Config.NODE2VEC_P,
            q=Config.NODE2VEC_Q,
            seed=Config.RANDOM_STATE,
            temp_folder=temp_dir,
            quiet=True
        )
        
        # Train model with timeout
        start_time = time.time()
        timeout = 600  # 10 minutes timeout
        
        model = node2vec.fit(
            window=5,
            min_count=1,
            batch_words=4,
            epochs=1  # ✅ Use `epochs` instead of `iter`
    )
        
        if time.time() - start_time > timeout:
            raise TimeoutError("Embedding generation timed out after 10 minutes")
        
        # Step 4: Store embeddings
        status_text.text("Step 4/4: Storing embeddings...")
        progress_bar.progress(70)
        
        with neo4j_service.session() as session:
            total_nodes = len(node_ids)
            processed = 0
            
            for i in range(0, total_nodes, Config.EMBEDDING_BATCH_SIZE):
                batch = node_ids[i:i + Config.EMBEDDING_BATCH_SIZE]
                queries = []
                
                for node_id in batch:
                    if str(node_id) in model.wv:
                        embedding = model.wv[str(node_id)].tolist()
                        queries.append((
                            """
                            MATCH (n)
                            WHERE id(n) = $node_id
                            SET n.embedding = $embedding
                            """,
                            {"node_id": node_id, "embedding": embedding}
                        ))
                
                # Execute batch
                for query, params in queries:
                    session.run(query, **params)
                
                processed += len(batch)
                progress = 70 + int(30 * (processed / total_nodes))
                progress_bar.progress(min(progress, 99))
        
        status_text.text("Graph embeddings generated successfully!")
        progress_bar.progress(100)
        logger.info("Successfully generated and stored graph embeddings")
        return True
        
    except Exception as e:
        status_text.error(f"Error generating embeddings: {str(e)}")
        logger.error(f"Embedding generation failed: {str(e)}", exc_info=True)
        return False
    finally:
        # Clean up resources
        if temp_dir and os.path.exists(temp_dir):
            try:
                shutil.rmtree(temp_dir)
                logger.info(f"Removed temp directory {temp_dir}")
            except Exception as e:
                logger.error(f"Error removing temp directory: {str(e)}")
        
        time.sleep(2)
        status_text.empty()
        progress_bar.empty()

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
def extract_user_embedding() -> Optional[np.ndarray]:
    """Extracts the embedding for the most recently uploaded case."""
    upload_id = st.session_state.get("last_upload_id")

    if not upload_id:
        st.error("❌ No upload ID found in session state.")
        return None

    with neo4j_service.session() as session:
        result = session.run(
            "MATCH (c:Case {upload_id: $upload_id}) RETURN c.embedding AS embedding",
            upload_id=upload_id
        )
        record = result.single()
        if record and record["embedding"] is not None:
            return np.array(record["embedding"]).reshape(1, -1)
        return None

@safe_neo4j_operation
def extract_training_data_from_csv(file_path: str) -> Tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(file_path, delimiter=";", encoding='utf-8-sig')
    df.columns = [col.strip() for col in df.columns]
   
   # Αντικατέστησε κόμμα με τελεία σε όλα τα string πεδία
    df = df.applymap(lambda x: str(x).replace(",", ".") if isinstance(x, str) else x)

    # Μετατροπή αριθμητικών στηλών σε float
    numeric_cols = [f"A{i}" for i in range(1, 11)] + ["Case_No", "Age_Mons", "Qchat-10-Score"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "Class_ASD_Traits" not in df.columns or "Case_No" not in df.columns:
        st.error("CSV must contain columns 'Class_ASD_Traits' and 'Case_No'")
        return pd.DataFrame(), pd.Series()

    y = df["Class_ASD_Traits"].apply(lambda x: 1 if str(x).strip().lower() == "yes" else 0)
    embeddings = []

    with neo4j_service.session() as session:
        for case_no in df["Case_No"]:
            result = session.run(
                "MATCH (c:Case {id: $id}) RETURN c.embedding AS embedding",
                id=int(case_no)
            )
            record = result.single()
            if record and record["embedding"]:
                embeddings.append(record["embedding"])
            else:
                logger.warning(f"No embedding found for case ID {case_no}")
    
    if not embeddings:
        st.error("⚠️ No embeddings found for any of the cases.")
        return pd.DataFrame(), pd.Series()

    return pd.DataFrame(embeddings), y

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


@st.cache_resource(show_spinner="Training ASD detection model (with embeddings)...")
def train_asd_detection_model() -> Optional[RandomForestClassifier]:
    try:
        # URL του CSV με τις ετικέτες
        csv_url = "https://raw.githubusercontent.com/GiorgosBouh/llm2cypher-asd-app/main/Toddler_Autism_dataset_July_2018_2.csv"

        # 📥 Φόρτωσε τα embeddings και τις ετικέτες από το Neo4j και το CSV
        X, y = extract_training_data_from_csv(csv_url)

        if X.empty or y.empty:
            st.error("⚠️ Δεν υπάρχουν embeddings ή labels για εκπαίδευση.")
            return None

        # 📊 Εμφάνισε κατανομή τάξεων
        st.subheader("📊 Class Distribution")
        st.write(Counter(y))

        # ✂️ Split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=Config.TEST_SIZE,
            stratify=y,
            random_state=Config.RANDOM_STATE
        )

        # 🧪 Χτίσε Pipeline
        pipeline = Pipeline([
            ('smote', SMOTE(random_state=Config.RANDOM_STATE, sampling_strategy='auto')),
            ('classifier', RandomForestClassifier(
                n_estimators=Config.N_ESTIMATORS,
                random_state=Config.RANDOM_STATE,
                class_weight='balanced'
            ))
        ])

        # 🧠 Εκπαίδευσε
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        y_proba = pipeline.predict_proba(X_test)[:, 1]

        # 📈 Αξιολόγηση
        st.subheader("📈 Model Evaluation")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("ROC AUC", f"{roc_auc_score(y_test, y_proba):.3f}")
            st.metric("Average Precision", f"{average_precision_score(y_test, y_proba):.3f}")
        with col2:
            st.metric("F1 Score", f"{classification_report(y_test, y_pred, output_dict=True)['1']['f1-score']:.3f}")
            st.metric("Accuracy", f"{classification_report(y_test, y_pred, output_dict=True)['accuracy']:.3f}")

        # 📉 Καμπύλες
        plot_combined_curves(y_test, y_proba)

        return pipeline.named_steps['classifier']

    except Exception as e:
        st.error(f"❌ Error training model: {e}")
        logger.error(f"❌ Error in train_asd_detection_model: {e}", exc_info=True)
        return None
@safe_neo4j_operation
def get_existing_embeddings() -> Optional[np.ndarray]:
    """Returns all existing case node embeddings from the graph."""
    with neo4j_service.session() as session:
        result = session.run("MATCH (c:Case) WHERE c.embedding IS NOT NULL RETURN c.embedding AS embedding")
        embeddings = [record["embedding"] for record in result if record["embedding"] is not None]

        if not embeddings:
            return None
        return np.array(embeddings)

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

# === Graph Embeddings Section ===
st.header("🌐 Graph Embeddings")
if st.button("🔁 Recalculate All Embeddings (Full Graph)"):
    with st.spinner("Re-generating embeddings for the entire graph..."):
        success = generate_graph_embeddings()
        if success:
            st.success("✅ All embeddings updated successfully!")
        else:
            st.error("❌ Failed to regenerate graph embeddings.")

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
        upload_id = str(uuid.uuid4())  # ✅ Δημιουργία μοναδικού ID για το νέο case
        st.session_state["last_upload_id"] = upload_id

        with st.spinner("Inserting case into graph..."):
            insert_user_case(row, upload_id)

        with st.spinner("Generating embedding for new case..."):
            if generate_embedding_for_node(upload_id):  # ✅ ΜΟΝΟ το νέο case
                embedding = extract_user_embedding()
                if embedding is None:
                    st.error("Failed to generate embedding for the new case")
                    st.stop()

                st.subheader("🧠 Graph Embedding")
                st.write(embedding)

                # === ASD Prediction ===
                if 'asd_model' in st.session_state:
                    with st.spinner("Predicting ASD traits..."):
                        model = st.session_state['asd_model']
                        if len(embedding.shape) == 1:
                            embedding = embedding.reshape(1, -1)

                        st.subheader("🧪 DEBUG: Embedding Inspection")
                        st.write("✅ Embedding Shape:", embedding.shape)
                        st.write("✅ Embedding Preview:", embedding.tolist())

                        st.write("✅ Model Expected Features:", model.n_features_in_)
                        if model.n_features_in_ != embedding.shape[1]:
                            st.error(f"❌ Feature Mismatch: Model expects {model.n_features_in_} features, got {embedding.shape[1]}")
                            st.stop()

                        all_embeddings = get_existing_embeddings()
                        if all_embeddings is not None:
                            from sklearn.metrics.pairwise import cosine_similarity
                            sim = cosine_similarity(embedding, all_embeddings)
                            st.write("🔍 Max Similarity to Existing Embeddings:", np.max(sim))
                            st.write("🔍 Mean Similarity to Existing Embeddings:", np.mean(sim))

                        proba = model.predict_proba(embedding)[0][1]
                        prediction = "YES (ASD Traits Detected)" if proba >= 0.5 else "NO (Control Case)"

                        st.subheader("🔍 Prediction Result")
                        col1, col2 = st.columns(2)
                        col1.metric("Prediction", prediction)
                        col2.metric("Confidence", f"{max(proba, 1 - proba):.1%}")

                        case_key = st.session_state.get("last_upload_id", "default_key")
                        fig = px.bar(
                            x=["Control", "ASD Traits"],
                            y=[1 - proba, proba],
                            labels={'x': 'Class', 'y': 'Probability'},
                            title="Prediction Probabilities"
                        )
                        st.plotly_chart(fig, key=f"prediction_bar_{case_key}")

                # === Anomaly Detection ===
                with st.spinner("Checking for anomalies..."):
                    iso_forest_scaler = train_isolation_forest()
                    if iso_forest_scaler:
                        iso_forest, scaler = iso_forest_scaler
                        if len(embedding.shape) == 1:
                            embedding = embedding.reshape(1, -1)
                        embedding_scaled = scaler.transform(embedding)
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
                            st.plotly_chart(fig, key=f"anomaly_hist_{case_key}")
                    else:
                        st.info("Anomaly detection model not trained yet or insufficient data.")
            else:
                st.error("❌ Failed to generate embedding for the new case.")
    except Exception as e:
        st.error(f"❌ Error processing file: {e}")
        logger.error(f"❌ Exception during upload processing: {e}")