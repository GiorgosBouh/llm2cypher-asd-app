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

openai_key = st.text_input("🔑 Enter your OpenAI API Key", type="password", value=os.getenv("OPENAI_API_KEY"))
question = st.text_input("📝 Ask your question in natural language")

# === Guard conditions ===
if not openai_key:
    st.warning("Please enter your OpenAI API key.")
    st.stop()

if not question:
    st.stop()

# === Initialize OpenAI client ===
client = OpenAI(api_key=openai_key)

# === Generate Cypher query from natural language ===
with st.spinner("💬 Thinking..."):
    prompt = f"""You are a Cypher expert working with this Neo4j knowledge graph about toddlers and autism.

Schema:
- (:Case)-[:HAS_DEMOGRAPHIC]->(:DemographicAttribute {{type, value}})
- (:Case)-[:SCREENED_FOR]->(:ASD_Trait {{value}})
- (:Case)-[:HAS_ANSWER {{value}}]->(:BehaviorQuestion {{name}})
- (:Case)-[:SUBMITTED_BY]->(:SubmitterType)

Relevant Values:
- Sex: 'm', 'f'
- Ethnicity: 'middle eastern', 'White European', 'Hispanic', 'black', 'asian', 'south asian', 'Native Indian', 'Others', 'Latino', 'mixed', 'Pacifica'
- Jaundice: 'yes', 'no'
- Family History: 'yes', 'no'
- Submitter: 'family member', 'Health Care Professional', 'Health care professional', 'Self', 'Others'
- ASD Trait Class: 'Yes', 'No'

Examples:
Q: How many toddlers have ASD traits?
A: MATCH (c:Case)-[:SCREENED_FOR]->(:ASD_Trait {{value: 'Yes'}}) RETURN count(DISTINCT c) AS total

Q: How many male toddlers?
A: MATCH (c:Case)-[:HAS_DEMOGRAPHIC]->(:DemographicAttribute {{type: 'Sex', value: 'm'}}) RETURN count(DISTINCT c) AS male_cases

Q: How many female toddlers with family history of ASD?
A: MATCH (c:Case)-[:HAS_DEMOGRAPHIC]->(:DemographicAttribute {{type: 'Sex', value: 'f'}}),
             (c)-[:HAS_DEMOGRAPHIC]->(:DemographicAttribute {{type: 'Family_mem_with_ASD', value: 'yes'}})
   RETURN count(DISTINCT c) AS total

Now, translate this user question into Cypher:
Q: {question}

Only return the Cypher query, no explanation.
"""

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

# === Run Cypher query ===
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