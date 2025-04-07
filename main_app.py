import streamlit as st
from neo4j import GraphDatabase
from openai import OpenAI
from dotenv import load_dotenv
import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import uuid

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

# === App UI ===
st.markdown("Ask a question in natural language and get answers from Neo4j using OpenAI.")

# === Optional: Prompt Schema Visualizer ===
with st.expander("🧠 Graph Schema Help"):
    st.markdown("### 🧩 Node Types")
    st.markdown("""
    - `Case`: Each screening instance  
    - `BehaviorQuestion`: Questions A1–A10  
    - `ASD_Trait`: Classification result (`Yes` / `No`)  
    - `DemographicAttribute`: Sex, Ethnicity, Jaundice, etc.  
    - `SubmitterType`: Who completed the test  
    """)

    st.markdown("### 🔗 Relationships")
    st.markdown("""
    - `(:Case)-[:HAS_ANSWER]->(:BehaviorQuestion)`  
    - `(:Case)-[:HAS_DEMOGRAPHIC]->(:DemographicAttribute)`  
    - `(:Case)-[:SCREENED_FOR]->(:ASD_Trait)`  
    - `(:Case)-[:SUBMITTED_BY]->(:SubmitterType)`  
    """)

    st.markdown("### 💡 Example Questions")
    st.code("""
Q: How many toddlers have ASD traits?
Q: How many answered 1 to question A3?
Q: How many male toddlers with jaundice?
Q: Who completed the test most often?
    """)

# === Log Translation Function ===
def log_translation(nl_question, cypher_query, result, success=True):
    with open("nl_to_cypher_log.csv", "a") as f:
        f.write(f'"{nl_question}","{cypher_query}","{str(result)}",{success}\n')

# === Embedding Functions ===
def prepare_graph_for_embeddings():
    with driver.session() as session:
        session.run("CALL gds.graph.drop('asd-graph', false)")
        session.run("""
            CALL gds.graph.project(
                'asd-graph',
                {
                    Case: {},
                    BehaviorQuestion: {},
                    DemographicAttribute: {},
                    ASD_Trait: {},
                    SubmitterType: {}
                },
                {
                    HAS_ANSWER: { type: 'HAS_ANSWER', orientation: 'UNDIRECTED' },
                    HAS_DEMOGRAPHIC: { type: 'HAS_DEMOGRAPHIC', orientation: 'UNDIRECTED' },
                    SCREENED_FOR: { type: 'SCREENED_FOR', orientation: 'UNDIRECTED' },
                    SUBMITTED_BY: { type: 'SUBMITTED_BY', orientation: 'UNDIRECTED' }
                }
            )
        """)

def run_node2vec():
    with driver.session() as session:
        session.run("""
            CALL gds.node2vec.write(
                'asd-graph',
                {
                    nodeLabels: ['Case'],
                    embeddingDimension: 64,
                    writeProperty: 'embedding',
                    iterations: 10,
                    randomSeed: 42
                }
            )
        """)

def extract_user_embedding(upload_id):
    with driver.session() as session:
        res = session.run("""
            MATCH (c:Case {upload_id: $upload_id})
            RETURN c.embedding AS embedding
        """, upload_id=upload_id)
        record = res.single()
        return record["embedding"] if record else None

# === Prediction Process ===
st.subheader("📄 Upload CSV for 1 Child ASD Prediction")
uploaded_file = st.file_uploader("Upload CSV", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file, delimiter=";")  # Ensure the correct delimiter is used
    if len(df) != 1:
        st.error("❌ Please upload exactly one row (one child).")
        st.stop()

    row = df.iloc[0]
    upload_id = str(uuid.uuid4())

    with st.spinner("📥 Inserting into graph..."):
        insert_user_case(row, upload_id)

    with st.spinner("🔄 Generating embedding..."):
        run_node2vec()

    # === Check if the new embedding is valid ===
    with st.spinner("🔮 Predicting..."):
        new_embedding = extract_user_embedding(upload_id)
        
        if new_embedding:
            # Ensure the embedding is valid before proceeding
            if any(pd.isna(val) for val in new_embedding):
                st.error("❌ The embedding contains NaN values. Please check the input data.")
            else:
                # Reshape the embedding into 2D if necessary
                new_embedding_reshaped = np.array(new_embedding).reshape(1, -1)
                
                # Perform the prediction
                prediction = clf.predict(new_embedding_reshaped)[0]
                label = "YES (ASD Traits Detected)" if prediction == 1 else "NO (Control Case)"
                st.success(f"🔍 Prediction: **{label}**")
        else:
            st.error("❌ No embedding found for the new Case.")