import streamlit as st
from neo4j import GraphDatabase
from openai import OpenAI
from dotenv import load_dotenv
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import numpy as np
import pandas as pd
import uuid

# === Load environment variables ===
load_dotenv()

# === Neo4j credentials from .env ===
NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USER = os.getenv("NEO4J_USER")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

# === Title, Subtitle & Attribution ===
st.title("🧠 NeuroCypher ASD")
st.markdown(
    "<i>The graph is based on Q-Chat-10 plus survey and other individuals characteristics that have proved to be effective in detecting the ASD cases from controls in behaviour science.</i>",
    unsafe_allow_html=True,
)
st.markdown("""
---
**Made in the Intelligent Systems Laboratory of the University of the Aegean by Bouchouras G., Doumanas D., Kotis K. (2025)**  
""")

# === ML Functions ===
def train_asd_detection_model():
    """Train the model on the graph data embeddings and ASD labels"""
    X, y = extract_training_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True)

    st.write(pd.DataFrame(report).transpose())  # Show classification report in Streamlit
    return clf

def extract_training_data():
    """Extract embeddings and corresponding ASD labels from the Neo4j database"""
    # Get embeddings and labels (1 for ASD, 0 for Control)
    with driver.session() as session:
        result = session.run("""
            MATCH (c:Case)-[:SCREENED_FOR]->(t:ASD_Trait)
            WHERE c.embedding IS NOT NULL
            RETURN c.embedding AS embedding, t.value AS label
        """)
        records = result.data()

    X = [r["embedding"] for r in records]
    y = [1 if r["label"] == "Yes" else 0 for r in records]
    return pd.DataFrame(X), pd.Series(y)

def insert_user_case(row, upload_id):
    """Insert the uploaded case into Neo4j"""
    with driver.session() as session:
        session.run("CREATE (c:Case {upload_id: $upload_id})", upload_id=upload_id)
        # Insert answers to behavior questions
        for i in range(1, 11):
            q = f"A{i}"
            val = int(row[q])
            session.run("""
                MATCH (q:BehaviorQuestion {name: $q})
                MATCH (c:Case {upload_id: $upload_id})
                CREATE (c)-[:HAS_ANSWER {value: $val}]->(q)
            """, q=q, val=val, upload_id=upload_id)

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

        session.run("""
            MATCH (s:SubmitterType {type: $who})
            MATCH (c:Case {upload_id: $upload_id})
            CREATE (c)-[:SUBMITTED_BY]->(s)
        """, who=row["Who_completed_the_test"], upload_id=upload_id)

def extract_user_embedding(upload_id):
    """Extract embedding for the uploaded case"""
    with driver.session() as session:
        res = session.run("""
            MATCH (c:Case {upload_id: $upload_id})
            RETURN c.embedding AS embedding
        """, upload_id=upload_id)
        record = res.single()
        return [record["embedding"]] if record else None

# === Anomaly Detection ===
def detect_anomalies_for_new_case(upload_id):
    """Detect if the new uploaded case is an anomaly in relation to existing cases"""
    new_embedding = extract_user_embedding(upload_id)

    if new_embedding:
        if any(pd.isna(val) for val in new_embedding[0]):
            st.error("❌ The embedding contains NaN values. Please check the input data.")
        else:
            new_embedding_reshaped = new_embedding[0].reshape(1, -1)

            # Calculate distances (Euclidean or cosine similarity)
            from sklearn.metrics.pairwise import euclidean_distances
            existing_embeddings = get_existing_embeddings()  # Function to retrieve existing embeddings
            distances = euclidean_distances(new_embedding_reshaped, existing_embeddings)
            st.write(f"Distances to existing cases: {distances}")

            threshold = 2.0  # Can adjust the threshold for anomaly detection
            if np.min(distances) > threshold:
                st.warning("⚠️ This case might be an anomaly!")
            else:
                st.success("✅ This case is similar to existing cases.")
    else:
        st.error("❌ No embedding found for the new Case.")

# === Get Existing Embeddings ===
def get_existing_embeddings():
    """Retrieve existing embeddings from the Neo4j database for anomaly detection"""
    with driver.session() as session:
        result = session.run("MATCH (c:Case) WHERE c.embedding IS NOT NULL RETURN c.embedding AS embedding")
        embeddings = [record["embedding"] for record in result]
    return np.array(embeddings)

# === Main Process for Handling Uploaded File ===
st.subheader("📄 Upload CSV for 1 Child ASD Prediction")
uploaded_file = st.file_uploader("Upload CSV", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file, delimiter=";")  # Ensure correct delimiter is used
    if len(df) != 1:
        st.error("❌ Please upload exactly one row (one child).")
        st.stop()

    row = df.iloc[0]
    upload_id = str(uuid.uuid4())

    # Insert new case into the graph and generate its embedding
    with st.spinner("📥 Inserting into graph..."):
        insert_user_case(row, upload_id)

    with st.spinner("🔄 Generating embedding..."):
        run_node2vec()

    # === Predict ASD for the new uploaded case ===
    with st.spinner("🔮 Predicting ASD Traits..."):
        clf = train_asd_detection_model()  # Train the model on the existing data first
        predict_asd_for_new_case(upload_id, clf)

    # === Anomaly Detection for the new uploaded case ===
    with st.spinner("🔍 Detecting Anomalies..."):
        detect_anomalies_for_new_case(upload_id)