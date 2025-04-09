import streamlit as st
from neo4j import GraphDatabase
from openai import OpenAI
from dotenv import load_dotenv
import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, roc_curve
from imblearn.over_sampling import SMOTE  # Import SMOTE for balancing classes
import uuid
import numpy as np
import time  # Import the time module
import matplotlib.pyplot as plt # Import matplotlib for plotting

st.sidebar.markdown(f"🔗 **Connected to:** `{os.getenv('NEO4J_URI')}`")

# === Load environment variables ===
load_dotenv()

# === Credentials ===
NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USER = os.getenv("NEO4J_USER")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# === Connect to Neo4j ===
@st.cache_resource
def get_driver():
    return GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

driver = get_driver()

# === Define Helper Functions ===
def insert_user_case(row, upload_id):
    with driver.session() as session:
        st.info(f"Inserting case with upload_id: {upload_id}")
        session.run("CREATE (c:Case {upload_id: $upload_id})", upload_id=upload_id)
        for i in range(1, 11):  # Assuming you have questions A1 to A10
            q = f"A{i}"
            val = int(row[q])  # Get the answer from the CSV file
            session.run("""
                MATCH (q:BehaviorQuestion {name: $q})
                MATCH (c:Case {upload_id: $upload_id})
                CREATE (c)-[:HAS_ANSWER {value: $val}]->(q)
            """, q=q, val=val, upload_id=upload_id)
            st.info(f"  - Answer {q}: {val}")

        # Insert demographic information
        demo = {
            "Sex": row["Sex"],
            "Ethnicity": row["Ethnicity"],
            "Jaundice": row["Jaundice"],
            "Family_mem_with_ASD": row["Family_mem_with_ASD"]
        }
        for k, v in demo.items():
            session.run("""
                MATCH (d:DemographicAttribute {type: $k, value: $v})
                MATCH (c:Case {upload_id: $upload_id})
                CREATE (c)-[:HAS_DEMOGRAPHIC]->(d)
            """, k=k, v=v, upload_id=upload_id)
            st.info(f"  - Demographic {k}: {v}")

        session.run("""
            MATCH (s:SubmitterType {type: $who})
            MATCH (c:Case {upload_id: $upload_id})
            CREATE (c)-[:SUBMITTED_BY]->(s)
        """, who=row["Who_completed_the_test"], upload_id=upload_id)
        st.info(f"  - Submitter: {row['Who_completed_the_test']}")

# === Generate Embeddings using Node2Vec ===
def run_node2vec():
    with driver.session() as session:
        st.info("Starting Node2Vec embedding generation...")
        # Check if the graph exists
        result = session.run("CALL gds.graph.exists('asd-graph') YIELD exists").single()
        if result and result['exists']:
            # Drop the graph if it exists
            st.info("  - Existing 'asd-graph' found, dropping it.")
            session.run("CALL gds.graph.drop('asd-graph') YIELD graphName")

        # Create the graph projection
        st.info("  - Creating graph projection 'asd-graph'.")
        session.run("""
            CALL gds.graph.project(
                'asd-graph',
                'Case',
                '*'
            )
            YIELD graphName, nodeCount, relationshipCount
        """)
        # Run Node2Vec
        st.info("  - Running Node2Vec algorithm.")
        session.run("""
            CALL gds.node2vec.write(
                'asd-graph',
                {
                    nodeLabels: ['Case'],
                    embeddingDimension: 128,
                    writeProperty: 'embedding',
                    iterations: 50,
                    randomSeed: 42
                }
            )
            YIELD nodeCount, writeMillis
        """)
        # Clean up the projected graph
        st.info("  - Dropping graph projection 'asd-graph'.")
        session.run("CALL gds.graph.drop('asd-graph')")
        st.info("Node2Vec embedding generation finished.")

# === NL to Cypher Transformation ===
def nl_to_cypher(question):
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

    Translate the following natural language question to Cypher, ensuring that you use the correct values and capitalization as described in the schema.

    Q: {question}

    Only return the Cypher query.
    """
    try:
        response = client.chat.completions.create(
            model="gpt-4o-2024-08-06",
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        llm_response = response.choices[0].message.content.strip()
        cypher_query = llm_response.replace("```cypher", "").replace("```", "").strip()
        return cypher_query
    except Exception as e:
        st.error(f"Error with OpenAI API: {e}")
        return None

# === Extract User Embedding ===
def extract_user_embedding(upload_id):
    with driver.session() as session:
        res = session.run("""
            MATCH (c:Case {upload_id: $upload_id})
            RETURN c.embedding AS embedding
        """, upload_id=upload_id)
        record = res.single()
        if record and record["embedding"] is not None:
            st.info(f"Embedding extracted for upload_id: {upload_id} - First 5 values: {record['embedding'][:5]}")
            return record["embedding"]
        else:
            st.warning(f"No embedding found for upload_id: {upload_id}")
            return None

# === Extract Training Data for ML Model ===
def extract_training_data():
    with driver.session() as session:
        result = session.run("""
            MATCH (c:Case)-[:SCREENED_FOR]->(t:ASD_Trait)
            WHERE c.embedding IS NOT NULL
            RETURN c.embedding AS embedding, t.value AS label
        """)
        records = result.data()

    X = [r["embedding"] for r in records]
    y = [1 if r["label"] == "Yes" else 0 for r in records]
    if X:
        st.info(f"Extracted {len(X)} data points for training the ASD detection model.")
    else:
        st.warning("⚠️ No data points with embeddings found for training the ASD detection model.")
    return pd.DataFrame(X), pd.Series(y)

# === Train ML Model for ASD Detection with SMOTE ===
def train_asd_detection_model():
    X, y = extract_training_data()
    if not X.empty:
        # Apply SMOTE to balance the classes
        smote = SMOTE(random_state=42)
        X_resampled, y_resampled = smote.fit_resample(X, y)
        st.info(f"Resampled dataset size: {len(X_resampled)}")

        X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, stratify=y_resampled, test_size=0.2, random_state=42)
        clf = RandomForestClassifier(n_estimators=100, random_state=42)
        clf.fit(X_train, y_train)

        y_pred_proba = clf.predict_proba(X_test)[:, 1]
        auc_roc = roc_auc_score(y_test, y_pred_proba)
        st.subheader("📊 Model Evaluation Results")
        st.write(f"AUC-ROC: **{auc_roc:.4f}**")

        y_pred = clf.predict(X_test)
        report = classification_report(y_test, y_pred, output_dict=True)
        st.subheader("📊 Classification Report (Precision, Recall, F1)")
        st.write(pd.DataFrame(report).transpose())

        # Plot ROC Curve
        fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
        fig, ax = plt.subplots()
        ax.plot(fpr, tpr, label=f'AUC = {auc_roc:.2f}')
        ax.plot([0, 1], [0, 1], linestyle='--')
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('ROC Curve')
        ax.legend(loc='lower right')
        st.pyplot(fig)

        st.info("ASD detection model trained and evaluated with AUC-ROC.")
        return clf
    else:
        st.warning("⚠️ Not enough data to train the ASD detection model.")
        return None

# === Get Existing Embeddings for Anomaly Detection ===
def get_existing_embeddings():
    with driver.session() as session:
        result = session.run("""
            MATCH (c:Case)
            WHERE c.embedding IS NOT NULL
            RETURN c.embedding AS embedding
        """)
        embeddings = [record["embedding"] for record in result]

        if not embeddings:  # Αν δεν υπάρχουν embeddings
            st.warning("⚠️ No embeddings found for anomaly detection.")

        return np.array(embeddings) if embeddings else None

# === Anomaly Detection with Isolation Forest ===
def detect_anomalies_with_isolation_forest(upload_id, iso_forest_model):
    # Extract the embedding for the new case
    new_embedding = extract_user_embedding(upload_id)
    if new_embedding:
        new_embedding_reshaped = np.array(new_embedding).reshape(1, -1)  # Reshape for prediction

        # Predict whether the new case is an anomaly
        anomaly_prediction = iso_forest_model.predict(new_embedding_reshaped)[0]

        if anomaly_prediction == -1:
            st.warning(f"⚠️ This case might be an anomaly!")
        else:
            st.success(f"✅ This case is likely normal.")
    else:
        st.error(f"❌ No embedding found for the new case with upload_id: {upload_id}")

# === Main Streamlit Logic ===
st.title("🧠 NeuroCypher ASD")
st.markdown(
    "<i>The graph is based on Q-Chat-10 plus survey and other individuals characteristics that have proved to be effective in detecting the ASD cases from controls in behaviour science.</i>",
    unsafe_allow_html=True,
)
# === Full Code Update ===

# === Check if User Case Exists ===
def check_user_case_exists(upload_id):
    with driver.session() as session:
        result = session.run("""
            MATCH (c:Case {upload_id: $upload_id})
            RETURN c
        """, upload_id=upload_id)
        return result.peek() is not None

# === Train Isolation Forest on Existing Embeddings ===
def train_isolation_forest(embeddings):
    if embeddings is not None and embeddings.shape[0] > 0:
        iso_forest = IsolationForest(random_state=42)
        iso_forest.fit(embeddings)
        st.info(f"Isolation Forest model trained on {embeddings.shape[0]} embeddings.")
        return iso_forest
    else:
        st.warning("⚠️ No embeddings available for training Isolation Forest.")
        return None

# === Main Streamlit Logic ===
st.title("🧠 NeuroCypher ASD")
st.markdown(
    "<i>The graph is based on Q-Chat-10 plus survey and other individuals characteristics that have proved to be effective in detecting the ASD cases from controls in behaviour science.</i>",
    unsafe_allow_html=True,
)

# === 2. Natural Language to Cypher Section ===
st.header("💬 Natural Language to Cypher")
question = st.text_input("📝 Ask your question in natural language:")

openai_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=openai_key)

if question:
    # Transform natural language to Cypher
    cypher_query = nl_to_cypher(question)
    if cypher_query:
        st.code(cypher_query, language="cypher")

        # === Execute Cypher Query and Display Results ===
        if st.button("▶️ Run Query"):
            with driver.session() as session:
                try:
                    results = session.run(cypher_query).data()
                    if results:
                        st.subheader("📊 Query Results:")
                        st.write(pd.DataFrame(results))
                    else:
                        st.info("No results found.")
                except Exception as e:
                    st.error(f"Neo4j error: {e}")
    else:
        st.error("❌ Failed to generate Cypher query.")

# === 3. ML Model Evaluation (Precision, Recall, F1, AUC-ROC) ===
st.subheader("📊 Model Evaluation on Existing Graph Data")

clf = train_asd_detection_model()  # Train the model

# === Upload CSV for 1 Child ASD Prediction ===
st.subheader("📄 Upload CSV for 1 Child ASD Prediction1")
uploaded_file = st.file_uploader("Upload CSV", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file, delimiter=";")
    if len(df) != 1:
        st.error("❌ Please upload exactly one row (one child).")
        st.stop()

    row = df.iloc[0]

    # Δημιουργία μοναδικού ID για το νέο περιστατικό
    upload_id = str(uuid.uuid4())
    st.info(f"Generated upload_id: {upload_id}")  # Add this line

    # Εισαγωγή των νέων δεδομένων στον γράφο
    with st.spinner("📥 Inserting into graph..."):
        insert_user_case(row, upload_id)

    # Δημιουργία embeddings για το νέο περιστατικό χρησιμοποιώντας το Node2Vec
    with st.spinner("🔄 Generating embeddings..."):
        run_node2vec()
        time.sleep(5)  # Wait for embeddings to be written

    # Πρόβλεψη των χαρακτηριστικών ASD για το νέο περιστατικό
    with st.spinner("🔮 Predicting ASD Traits..."):
        # Check if the user case exists
        if not check_user_case_exists(upload_id):
            st.error(f"❌ Could not find Case with upload_id: {upload_id} in the graph.")
        else:
            new_embedding = extract_user_embedding(upload_id)
            st.write(f"**Embedding του νέου περιστατικού (upload_id: {upload_id}):**")
            st.write(new_embedding) # <--- Η προσθήκη για εμφάνιση του embedding
            if new_embedding and clf:
                new_embedding_reshaped = np.array(new_embedding).reshape(1, -1)  # Reshape for prediction
                # Χρησιμοποιήστε έναν προεκπαιδευμένο ταξινομητή για την πρόβλεψη των χαρακτηριστικών ASD
                prediction = clf.predict(new_embedding_reshaped)[0]
                label = "YES (ASD Traits Detected)" if prediction == 1 else "NO (Control Case)"
                st.success(f"🔍 Prediction: **{label}**")
            elif not new_embedding:
                st.error("❌ No embedding found for the new Case.")
            else:st.warning("⚠️ ASD prediction model not trained yet.")

    # --- Ανίχνευση Ανωμαλιών με Isolation Forest ---
    with st.spinner("🧐 Detecting Anomalies (Isolation Forest)..."):
        existing_embeddings = get_existing_embeddings()
        iso_forest_model = train_isolation_forest(existing_embeddings)
        if iso_forest_model:
            detect_anomalies_with_isolation_forest(upload_id, iso_forest_model)
        else:
            st.warning("❌ Could not detect anomalies as the Isolation Forest model could not be trained.")