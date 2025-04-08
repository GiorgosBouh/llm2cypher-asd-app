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

# === Input fields ===
openai_key = os.getenv("OPENAI_API_KEY")
question = st.text_input("📝 Ask your question in natural language")

if not question:
    st.stop()

# === Initialize OpenAI client ===
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

Examples:
Q: How many toddlers have ASD traits?
A: MATCH (c:Case)-[:SCREENED_FOR]->(:ASD_Trait {{value: 'Yes'}}) RETURN count(DISTINCT c) AS total

Q: How many male toddlers?
A: MATCH (c:Case)-[:HAS_DEMOGRAPHIC]->(:DemographicAttribute {{type: 'Sex', value: 'm'}}) RETURN count(DISTINCT c) AS male_cases

Q: How many female toddlers with family history of ASD?
A: MATCH (c:Case)
      -[:HAS_DEMOGRAPHIC]->(:DemographicAttribute {{type: 'Sex', value: 'f'}}),
      (c)-[:HAS_DEMOGRAPHIC]->(:DemographicAttribute {{type: 'Family_mem_with_ASD', value: 'yes'}})
   RETURN count(DISTINCT c) AS total

Now, translate this user question into Cypher:
Q: {question}

Only return the Cypher query, no explanation, no markdown.
"""

# === Generate Cypher query ===
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

# === Run query on Neo4j ===
with st.spinner("📡 Querying Neo4j..."):
    try:
        driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
        with driver.session() as session:
            result = session.run(cypher_query)
            records = [record.data() for record in result]

        if records:
            st.success("✅ Results:")
            st.json(records)
        else:
            st.warning("No results found.")

    except Exception as e:
        st.error(f"Neo4j error: {e}")
# === ML Functions ===
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import numpy as np
import pandas as pd

# === Train ML model using existing graph embeddings ===
def train_asd_detection_model():
    # Extract embeddings and labels (1 for ASD, 0 for Control)
    X, y = extract_training_data()

    # Train the Random Forest Classifier
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)

    # Evaluate the model
    y_pred = clf.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True)

    # Display the results
    st.write(pd.DataFrame(report).transpose())

    return clf

# === Use embeddings from Neo4j to predict ASD for new file ===
def predict_asd_for_new_case(upload_id, clf):
    # Get the embedding for the new case
    new_embedding = extract_user_embedding(upload_id)

    if new_embedding:
        # Check if the embedding is valid
        if any(pd.isna(val) for val in new_embedding[0]):
            st.error("❌ The embedding contains NaN values. Please check the input data.")
        else:
            # Reshape the embedding into 2D if necessary
            new_embedding_reshaped = new_embedding[0].reshape(1, -1)

            # Perform the prediction
            prediction = clf.predict(new_embedding_reshaped)[0]
            label = "YES (ASD Traits Detected)" if prediction == 1 else "NO (Control Case)"
            st.success(f"🔍 Prediction: **{label}**")
    else:
        st.error("❌ No embedding found for the new Case.")

# === Anomaly Detection ===
def detect_anomalies_for_new_case(upload_id):
    # Get the embedding for the new case
    new_embedding = extract_user_embedding(upload_id)

    if new_embedding:
        # Check if embedding contains NaN values
        if any(pd.isna(val) for val in new_embedding[0]):
            st.error("❌ The embedding contains NaN values. Please check the input data.")
        else:
            # Reshape into 2D array for anomaly detection
            new_embedding_reshaped = new_embedding[0].reshape(1, -1)

            # Calculate the distance between the new case and other cases (can use cosine similarity, Euclidean distance, etc.)
            # For simplicity, use Euclidean distance
            from sklearn.metrics.pairwise import euclidean_distances

            # Get the embeddings of the existing data (use embeddings stored in the Neo4j graph)
            existing_embeddings = get_existing_embeddings()  # This function needs to be defined to get existing embeddings

            # Compute the distance
            distances = euclidean_distances(new_embedding_reshaped, existing_embeddings)
            st.write(f"Distances to existing cases: {distances}")

            # Check if the new case is significantly different (define a threshold for anomaly detection)
            threshold = 2.0  # Example threshold; this can be adjusted
            if np.min(distances) > threshold:
                st.warning("⚠️ This case might be an anomaly!")
            else:
                st.success("✅ This case is similar to existing cases.")
    else:
        st.error("❌ No embedding found for the new Case.")

# === Process for handling uploaded file ===
st.subheader("📄 Upload CSV for 1 Child ASD Prediction")
uploaded_file = st.file_uploader("Upload CSV", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file, delimiter=";")  # Ensure correct delimiter is used
    if len(df) != 1:
        st.error("❌ Please upload exactly one row (one child).")
        st.stop()

    row = df.iloc[0]
    upload_id = str(uuid.uuid4())

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