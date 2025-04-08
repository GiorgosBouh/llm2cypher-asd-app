import streamlit as st
from neo4j import GraphDatabase
from openai import OpenAI
from dotenv import load_dotenv
import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import uuid
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
import time  # Import the time module

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
        session.run("CREATE (c:Case {upload_id: $upload_id})", upload_id=upload_id)
        for i in range(1, 11):  # Assuming you have questions A1 to A10
            q = f"A{i}"
            val = int(row[q])  # Get the answer from the CSV file
            session.run("""
                MATCH (q:BehaviorQuestion {name: $q})
                MATCH (c:Case {upload_id: $upload_id})
                CREATE (c)-[:HAS_ANSWER {value: $val}]->(q)
            """, q=q, val=val, upload_id=upload_id)

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

        session.run("""
            MATCH (s:SubmitterType {type: $who})
            MATCH (c:Case {upload_id: $upload_id})
            CREATE (c)-[:SUBMITTED_BY]->(s)
        """, who=row["Who_completed_the_test"], upload_id=upload_id)

# === Generate Embeddings using Node2Vec ===
# === Generate Embeddings using Node2Vec ===
def run_node2vec():
    with driver.session() as session:
        # Check if the graph exists
        result = session.run("CALL gds.graph.exists('asd-graph') YIELD exists").single()
        if result and result['exists']:
            # Drop the graph if it exists
            session.run("CALL gds.graph.drop('asd-graph') YIELD graphName")

        # Create the graph projection
        session.run("""
            CALL gds.graph.project(
                'asd-graph',
                'Case',
                '*'
            )
            YIELD graphName, nodeCount, relationshipCount
        """)
        # Run Node2Vec
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
            YIELD nodeCount, writeMillis
        """)
        # Clean up the projected graph
        session.run("CALL gds.graph.drop('asd-graph')")

# === Check if User Case Exists ===
def check_user_case_exists(upload_id):
    with driver.session() as session:
        result = session.run("""
            MATCH (c:Case {upload_id: $upload_id})
            RETURN c
        """, upload_id=upload_id)
        return result.peek() is not None

# === Extract User Embedding ===
def extract_user_embedding(upload_id):
    with driver.session() as session:
        res = session.run("""
            MATCH (c:Case {upload_id: $upload_id})
            RETURN c.embedding AS embedding
        """, upload_id=upload_id)
        record = res.single()
        return record["embedding"] if record else None

# === Predict ASD for New Case ===
def predict_asd_for_new_case(upload_id, clf):
    new_embedding = extract_user_embedding(upload_id)
    if new_embedding:
        new_embedding_reshaped = np.array(new_embedding).reshape(1, -1)
        prediction = clf.predict(new_embedding_reshaped)[0]
        label = "YES (ASD Traits Detected)" if prediction == 1 else "NO (Control Case)"
        st.success(f"🔍 Prediction: **{label}**")
    else:
        st.error("❌ No embedding found for the new Case.")

# === Anomaly Detection ===
def detect_anomalies_for_new_case(upload_id):
    new_embedding = extract_user_embedding(upload_id)
    if new_embedding:
        new_embedding_reshaped = new_embedding[0].reshape(1, -1)
        distances = euclidean_distances(new_embedding_reshaped, get_existing_embeddings())
        threshold = 2.0
        if np.min(distances) > threshold:
            st.warning("⚠️ This case might be an anomaly!")
        else:
            st.success("✅ This case is similar to existing cases.")
    else:
        st.error("❌ No embedding found for the new Case.")

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
    return pd.DataFrame(X), pd.Series(y)

# === Train ML Model for ASD Detection ===
def train_asd_detection_model():
    X, y = extract_training_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True)
    st.write(pd.DataFrame(report).transpose())

    return clf

# === Get Existing Embeddings for Anomaly Detection ===
def get_existing_embeddings():
    with driver.session() as session:
        result = session.run("""
            MATCH (c:Case)
            WHERE c.embedding IS NOT NULL
            RETURN c.embedding AS embedding
        """)
        embeddings = [record["embedding"] for record in result]
    return np.array(embeddings)

# === Main Streamlit Logic ===
st.title("🧠 NeuroCypher ASD")
st.markdown(
    "<i>The graph is based on Q-Chat-10 plus survey and other individuals characteristics that have proved to be effective in detecting the ASD cases from controls in behaviour science.</i>",
    unsafe_allow_html=True,
)

# === 1. Graph Schema Section ===
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

# === 2. Natural Language to Cypher Section ===
st.header("💬 Natural Language to Cypher")
question = st.text_input("📝 Ask your question in natural language:")

openai_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=openai_key)

# === Prompt engineering with schema ===
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

Translate this question to Cypher:
Q: {question}

Only return the Cypher query.
"""

if question:
    question = question.strip().replace("```cypher", "").replace("```", "").strip()
    with st.spinner("💬 Thinking..."):
        try:
            response = client.chat.completions.create(
                model="gpt-4o-2024-08-06",
                messages=[{"role": "user", "content": prompt}],
                temperature=0
            )
            cypher_query = response.choices[0].message.content.strip()
            st.code(cypher_query, language="cypher")
        except Exception as e:
            st.error(f"OpenAI error: {e}")
            st.stop()

# === 3. ML Model Evaluation (Precision, Recall, F1) ===
st.subheader("📊 Model Evaluation on Existing Graph Data")

clf = train_asd_detection_model()  # Train the model
st.write("🔍 Model Evaluation Results (Precision, Recall, F1)")

# === Upload CSV for 1 Child ASD Prediction ===
st.subheader("📄 Upload CSV for 1 Child ASD Prediction")
uploaded_file = st.file_uploader("Upload CSV", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file, delimiter=";")  # Ensure the correct delimiter is used
    if len(df) != 1:
        st.error("❌ Please upload exactly one row (one child).")
        st.stop()

    row = df.iloc[0]

    # Δημιουργία μοναδικού ID για το νέο περιστατικό
    upload_id = str(uuid.uuid4())  # Δημιουργεί ένα μοναδικό αναγνωριστικό ID για το νέο περιστατικό

    # Εισαγωγή των νέων δεδομένων στον γράφο
    with st.spinner("📥 Inserting into graph..."):
        insert_user_case(row, upload_id)

    # Δημιουργία embeddings για το νέο περιστατικό χρησιμοποιώντας το Node2Vec
    with st.spinner("🔄 Generating embeddings..."):
        run_node2vec()

    # Wait a bit to ensure embeddings are written (you might need to adjust the sleep duration)
    time.sleep(5)  # Wait for 5 seconds

    # Πρόβλεψη των χαρακτηριστικών ASD για το νέο περιστατικό
    with st.spinner("🔮 Predicting ASD Traits..."):
        # Check if the user case exists
        if not check_user_case_exists(upload_id):
            st.error(f"❌ Could not find Case with upload_id: {upload_id} in the graph.")
        else:
            new_embedding = extract_user_embedding(upload_id)
            if new_embedding:
                new_embedding_reshaped = np.array(new_embedding).reshape(1, -1)  # Reshape for prediction
                # Χρησιμοποιήστε έναν προεκπαιδευμένο ταξινομητή για την πρόβλεψη των χαρακτηριστικών ASD
                prediction = clf.predict(new_embedding_reshaped)[0]
                label = "YES (ASD Traits Detected)" if prediction == 1 else "NO (Control Case)"
                st.success(f"🔍 Prediction: **{label}**")
                # --- Προσθέστε την κλήση για την ανίχνευση ανωμαλιών εδώ ---
            with st.spinner("🧐 Detecting Anomalies..."):
                detect_anomalies_for_new_case(upload_id)
            else:
                st.error("❌ No embedding found for the new Case.")