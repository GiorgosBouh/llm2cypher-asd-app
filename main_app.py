import streamlit as st
import openai
from neo4j import GraphDatabase
from dotenv import load_dotenv
import os
load_dotenv()
# === Neo4j Settings ===

NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USER = os.getenv("NEO4J_USER")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# === Streamlit UI ===
st.set_page_config(page_title="LLM to Cypher (ASD KG)", layout="centered")
st.title("🧠 Ask Questions about the Autism Knowledge Graph")

api_key = st.text_input("🔑 Enter your OpenAI API Key", type="password")

question = st.text_input("📝 Ask your question in natural language")

run = st.button("🔍 Submit Question")

if run and api_key and question:
    openai.api_key = api_key

    with st.spinner("💡 Translating your question to Cypher..."):
        prompt = f"""You are an expert in Cypher for Neo4j. Translate the following question into a Cypher query only (no explanation, no markdown, no formatting):

{question}
"""

        try:
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[
                    {"role": "user", "content": prompt}
                ],
                temperature=0
            )
            cypher = response["choices"][0]["message"]["content"].strip()
            st.code(cypher, language="cypher")
        except Exception as e:
            st.error(f"OpenAI error: {e}")
            st.stop()

    with st.spinner("📦 Executing Cypher query on Neo4j..."):
        try:
            driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
            with driver.session() as session:
                result = session.run(cypher)
                records = [dict(r) for r in result]
            driver.close()

            if records:
                st.success("📊 Results:")
                st.json(records)
            else:
                st.info("🔍 No results found.")

        except Exception as e:
            st.error(f"Neo4j error: {e}")
