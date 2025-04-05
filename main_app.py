import streamlit as st
from neo4j import GraphDatabase
from openai import OpenAI
from dotenv import load_dotenv
import os

# === Load environment variables ===
load_dotenv()

# === Neo4j credentials from .env ===
NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USER = os.getenv("NEO4J_USER")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

# === App UI ===
st.title("🧠 Ask the Autism Knowledge Graph")
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