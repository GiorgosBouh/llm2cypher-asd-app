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
    - `Child`: Placeholder for individual child (if modeled)  
    - `ScreeningResult`: Alternative or extended result node  
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
    question = question.strip().replace("```cypher", "").replace("```", "").strip()
    prompt = f"""
You are a Cypher expert working with a Neo4j Knowledge Graph about toddlers and autism.

Schema:
- (:Case {{id: int}})
- (:BehaviorQuestion {{name: string}})
- (:ASD_Trait {{value: 'Yes' | 'No'}})
- (:DemographicAttribute {{type: 'Sex' | 'Ethnicity' | 'Jaundice' | 'Family_mem_with_ASD', value: string}})
- (:SubmitterType {{type: string}})
- (:Child)
- (:ScreeningResult)

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
            cypher_query = cypher_query.replace("```cypher", "").replace("```", "").strip()
            st.code(cypher_query, language="cypher")
        except Exception as e:
            st.error(f"OpenAI error: {e}")
            st.stop()

    with st.spinner("📱 Executing query in Neo4j..."):
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
# After initializing driver
driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

# Define the function BEFORE using it
def is_gds_supported(driver):
    try:
        with driver.session(database="neo4j") as session:
            result = session.run("CALL gds.version()")
            version = result.single()[0]
            st.info(f"✅ GDS version detected: {version}")
            return True
    except Exception as e:
        st.error(f"GDS test failed: {e}")
        return False

# Now it's safe to call it
if not is_gds_supported(driver):
    st.warning("⚠️ GDS is not available. Please ensure it is installed in the 'neo4j' database.")
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
                    SubmitterType: {},
                    Child: {},
                    ScreeningResult: {}
                },
                {
                    HAS_ANSWER: { type: 'HAS_ANSWER', orientation: 'UNDIRECTED' },
                    HAS_DEMOGRAPHIC: { type: 'HAS_DEMOGRAPHIC', orientation: 'UNDIRECTED' },
                    SCREENED_FOR: { type: 'SCREENED_FOR', orientation: 'UNDIRECTED' },
                    SUBMITTED_BY: { type: 'SUBMITTED_BY', orientation: 'UNDIRECTED' }
                }
            )
        """)

# (rest of your code remains unchanged)

