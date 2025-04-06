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

# === NATURAL LANGUAGE TO CYPHER ===
st.header("💬 Natural Language to Cypher")
question = st.text_input("Ask a question in natural language:")

if question:
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

    client = OpenAI(api_key=OPENAI_API_KEY)

    with st.spinner("💡 Translating to Cypher..."):
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

    with st.spinner("📡 Executing query in Neo4j..."):
        try:
            with driver.session() as session:
                result = session.run(cypher_query)
                records = [record.data() for record in result]

            if records:
                st.success("✅ Results:")
                st.json(records)
            else:
                st.warning("⚠️ No results found.")
        except Exception as e:
            st.error(f"Neo4j error: {e}")

# === ML SECTION ===
st.markdown("---")
st.header("🧠 Predict Autism Traits")

# Check GDS support
GDS_SUPPORTED = True
with driver.session() as session:
    try:
        session.run("CALL gds.debug.sysInfo()")
    except Exception as e:
        GDS_SUPPORTED = False

if not GDS_SUPPORTED:
    st.warning("⚠️ Node2Vec is not supported on Neo4j AuraDB Free. Please switch to Neo4j Desktop or Aura Professional to enable graph-based machine learning features.")
    st.stop()

# Continue with embedding + ML prediction if GDS is supported

# Helper functions
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

def extract_training_data():
    with driver.session() as session:
        result = session.run("""
            MATCH (c:Case)-[:SCREENED_FOR]->(t:ASD_Trait)
            WHERE exists(c.embedding)
            RETURN c.embedding AS embedding, t.value AS label
        """)
        records = result.data()
    X = [r["embedding"] for r in records]
    y = [1 if r["label"] == "Yes" else 0 for r in records]
    return pd.DataFrame(X), pd.Series(y)

def insert_user_case(row, upload_id):
    with driver.session() as session:
        session.run("CREATE (c:Case {upload_id: $upload_id})", upload_id=upload_id)
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
        """, who=row["Who_completed"], upload_id=upload_id)

def extract_user_embedding(upload_id):
    with driver.session() as session:
        res = session.run("""
            MATCH (c:Case {upload_id: $upload_id})
            RETURN c.embedding AS embedding
        """, upload_id=upload_id)
        record = res.single()
        return [record["embedding"]] if record else None

# === Model Evaluation on Existing Graph Data ===
st.subheader("📊 Model Evaluation on Existing Graph Data")
prepare_graph_for_embeddings()
run_node2vec()
X, y = extract_training_data()
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
report = classification_report(y_test, y_pred, output_dict=True)
st.write(pd.DataFrame(report).transpose())

# === Upload CSV and Predict ASD ===
st.subheader("📄 Upload CSV for 1 Child ASD Prediction")
uploaded_file = st.file_uploader("Upload CSV", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    if len(df) != 1:
        st.error("❌ Please upload exactly one row (one child).")
        st.stop()

    row = df.iloc[0]
    upload_id = str(uuid.uuid4())

    with st.spinner("📥 Inserting into graph..."):
        insert_user_case(row, upload_id)

    with st.spinner("🔄 Generating embedding..."):
        run_node2vec()

    with st.spinner("🔮 Predicting..."):
        new_embedding = extract_user_embedding(upload_id)
        if new_embedding:
            prediction = clf.predict(new_embedding)[0]
            label = "YES (ASD Traits Detected)" if prediction == 1 else "NO (Control Case)"
            st.success(f"🔍 Prediction: **{label}**")
        else:
            st.error("❌ No embedding found for the new Case.").
