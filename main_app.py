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

def train_asd_detection_model():
    X, y = extract_training_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True)
    st.write(pd.DataFrame(report).transpose())

    return clf

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

clf = train_asd_detection_model()  # Train the model
st.write("🔍 Model Evaluation Results (Precision, Recall, F1)")

# === 4. Upload CSV for New Case ===
st.subheader("📄 Upload CSV for 1 Child ASD Prediction")
uploaded_file = st.file_uploader("Upload CSV", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file, delimiter=";")
    if len(df) != 1:
        st.error("❌ Please upload exactly one row (one child).")
        st.stop()

    row = df.iloc[0]
    upload_id = str(uuid.uuid4())

    with st.spinner("📥 Inserting into graph..."):
        insert_user_case(row, upload_id)

    with st.spinner("🔄 Generating embedding..."):
        run_node2vec()

    # === 5. Predict ASD for New Case ===
    with st.spinner("🔮 Predicting ASD Traits..."):
        predict_asd_for_new_case(upload_id, clf)

    # === 6. Anomaly Detection ===
    with st.spinner("🔍 Detecting Anomalies..."):
        detect_anomalies_for_new_case(upload_id)

# Display results for new CSV file
def predict_asd_for_new_case(upload_id, clf):
    new_embedding = extract_user_embedding(upload_id)
    if new_embedding:
        new_embedding_reshaped = new_embedding[0].reshape(1, -1)
        prediction = clf.predict(new_embedding_reshaped)[0]
        label = "YES (ASD Traits Detected)" if prediction == 1 else "NO (Control Case)"
        st.success(f"🔍 Prediction: **{label}**")
    else:
        st.error("❌ No embedding found for the new Case.")

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