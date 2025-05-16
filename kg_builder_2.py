import pandas as pd
import numpy as np
from neo4j import GraphDatabase
import networkx as nx
from node2vec import Node2Vec
import os
import logging
from dotenv import load_dotenv

# Φόρτωση environment variables
load_dotenv()

# Ρύθμιση logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class GraphBuilder:
    def __init__(self):
        self.EMBEDDING_DIM = 256  # Τώρα ταιριάζει με τα υπόλοιπα scripts
        self.MIN_NODES = 20  # Αυξημένο ελάχιστο όριο
        self.BATCH_SIZE = 500  # Βελτιστοποίηση batch processing

    def connect_to_neo4j(self):
        """Βελτιωμένη σύνδεση με επιπλέον παραμέτρους"""
        return GraphDatabase.driver(
            os.getenv("NEO4J_URI"),
            auth=(os.getenv("NEO4J_USER"), os.getenv("NEO4J_PASSWORD")),
            max_connection_lifetime=7200,
            max_connection_pool_size=50,
            connection_timeout=30
        )

    def parse_csv(self, file_path):
        """Ενισχυμένο parsing με έλεγχο δεδομένων"""
        try:
            df = pd.read_csv(file_path, sep=";", encoding="utf-8-sig")
            df.columns = [col.strip() for col in df.columns]
            
            # Έλεγχος υποχρεωτικών στηλών
            required_cols = ['Case_No', 'A1', 'A2', 'A3', 'A4', 'A5', 
                           'A6', 'A7', 'A8', 'A9', 'A10', 'Class_ASD_Traits']
            missing = [col for col in required_cols if col not in df.columns]
            if missing:
                raise ValueError(f"Missing required columns: {missing}")

            # Μετατροπή αριθμητικών
            numeric_cols = ['Case_No'] + [f'A{i}' for i in range(1,11)]
            for col in numeric_cols:
                df[col] = pd.to_numeric(df[col].astype(str).str.replace(",", "."), errors='coerce')

            # Έλεγχος για κενές τιμές
            if df.isnull().values.any():
                logger.warning("Data contains missing values - applying imputation")
                df = df.fillna(df.mean(numeric_only=True))

            logger.info("✅ Successfully parsed and cleaned data")
            return df

        except Exception as e:
            logger.error(f"Failed to parse CSV: {str(e)}")
            raise

    def create_graph_structure(self, driver, df):
        """Ολοκληρωμένη δημιουργία γράφου με transaction management"""
        with driver.session() as session:
            try:
                # Διαγραφή παλιού γράφου
                session.run("MATCH (n) DETACH DELETE n")
                
                # Δημιουργία κόμβων
                self._create_base_nodes(session, df)
                
                # Δημιουργία σχέσεων
                self._create_case_relationships(session, df)
                
                # Ενισχυμένες σχέσεις ομοιότητας
                self._create_enhanced_similarity_relations(session, df)
                
                logger.info("✅ Graph structure created successfully")
                return True
                
            except Exception as e:
                logger.error(f"Graph creation failed: {str(e)}")
                return False

    def _create_base_nodes(self, session, df):
        """Δημιουργία βασικών κόμβων"""
        # Behavior Questions
        session.run("""
            UNWIND range(1,10) AS i
            MERGE (:BehaviorQuestion {name: 'A' + toString(i)})
        """)
        
        # Demographic Attributes
        demo_cols = ["Sex", "Ethnicity", "Jaundice", "Family_mem_with_ASD"]
        for col in demo_cols:
            session.run(f"""
                UNWIND $values AS val
                MERGE (:DemographicAttribute {{type: '{col}', value: val}})
            """, values=df[col].unique().tolist())
        
        # Submitter Types
        session.run("""
            UNWIND $values AS val
            MERGE (:SubmitterType {type: val})
        """, values=df["Who_completed_the_test"].unique().tolist())

    def _create_case_relationships(self, session, df):
        """Δημιουργία σχέσεων περιπτώσεων"""
        # Batch creation for better performance
        cases = df[['Case_No']].to_dict('records')
        session.run("""
            UNWIND $cases AS case
            MERGE (c:Case {id: case.Case_No})
            SET c.upload_id = toString(case.Case_No)
        """, cases=cases)
        
        # Answers relationships
        answer_data = []
        for _, row in df.iterrows():
            for q in [f'A{i}' for i in range(1,11)]:
                answer_data.append({
                    'case_id': int(row['Case_No']),
                    'question': q,
                    'value': int(row[q])
                })
        
        session.run("""
            UNWIND $answers AS ans
            MATCH (c:Case {id: ans.case_id})
            MATCH (q:BehaviorQuestion {name: ans.question})
            MERGE (c)-[:HAS_ANSWER {value: ans.value}]->(q)
        """, answers=answer_data)
        
        # Demographic relationships
        demo_data = []
        demo_cols = ["Sex", "Ethnicity", "Jaundice", "Family_mem_with_ASD"]
        for _, row in df.iterrows():
            for col in demo_cols:
                demo_data.append({
                    'case_id': int(row['Case_No']),
                    'type': col,
                    'value': row[col]
                })
        
        session.run("""
            UNWIND $demos AS demo
            MATCH (c:Case {id: demo.case_id})
            MATCH (d:DemographicAttribute {type: demo.type, value: demo.value})
            MERGE (c)-[:HAS_DEMOGRAPHIC]->(d)
        """, demos=demo_data)
        
        # Submitter relationships
        submitter_data = df[['Case_No', 'Who_completed_the_test']].to_dict('records')
        session.run("""
            UNWIND $submitters AS sub
            MATCH (c:Case {id: sub.Case_No})
            MATCH (s:SubmitterType {type: sub.Who_completed_the_test})
            MERGE (c)-[:SUBMITTED_BY]->(s)
        """, submitters=submitter_data)

    def _create_enhanced_similarity_relations(self, session, df):
        """Ενισχυμένες σχέσεις ομοιότητας"""
        # 1. Answer-based similarity
        session.run("""
            MATCH (c1:Case)-[r1:HAS_ANSWER]->(q)<-[r2:HAS_ANSWER]-(c2:Case)
            WHERE c1 <> c2 AND r1.value = r2.value
            WITH c1, c2, count(q) AS shared_answers
            WHERE shared_answers >= 7
            MERGE (c1)-[:SIMILAR_TO {type: 'answer', weight: shared_answers/10.0}]->(c2)
        """)
        
        # 2. Demographic similarity
        demo_cols = ["Sex", "Ethnicity", "Jaundice", "Family_mem_with_ASD"]
        for col in demo_cols:
            session.run(f"""
                MATCH (c1:Case)-[:HAS_DEMOGRAPHIC]->(d:DemographicAttribute {{type: '{col}'}})
                MATCH (c2:Case)-[:HAS_DEMOGRAPHIC]->(d)
                WHERE c1 <> c2
                MERGE (c1)-[:SIMILAR_TO {{type: 'demographic', category: '{col}'}}]->(c2)
            """)

    def generate_graph_embeddings(self, driver):
        """Ενισχυμένη δημιουργία embeddings"""
        try:
            # Φόρτωση γράφου με ενισχυμένο query
            with driver.session() as session:
                result = session.run("""
                    MATCH (c:Case)
                    OPTIONAL MATCH (c)-[r]->(other)
                    WHERE other:BehaviorQuestion OR other:DemographicAttribute OR other:SubmitterType OR other:Case
                    RETURN c.id AS node_id, 
                           collect(DISTINCT other.id) AS neighbors,
                           labels(c) AS labels
                """)
                
                G = nx.Graph()
                for record in result:
                    node_id = str(record["node_id"])
                    G.add_node(node_id, labels=record["labels"])
                    for neighbor in record["neighbors"]:
                        if neighbor:
                            G.add_edge(node_id, str(neighbor))

            # Έλεγχος γράφου
            if len(G.nodes) < self.MIN_NODES:
                raise ValueError(f"Insufficient nodes for embedding ({len(G.nodes)} < {self.MIN_NODES})")

            logger.info(f"📊 Graph loaded: {len(G.nodes)} nodes, {len(G.edges)} edges")

            # Βελτιστοποιημένο Node2Vec
            node2vec = Node2Vec(
                G,
                dimensions=self.EMBEDDING_DIM,
                walk_length=30,
                num_walks=200,
                workers=4,
                p=1.0,
                q=0.5,
                temp_folder=os.path.join(os.getcwd(), 'node2vec_temp')
            )

            model = node2vec.fit(
                window=10,
                min_count=1,
                batch_words=2000,
                epochs=50
            )

            # Αποθήκευση embeddings με batch processing
            with driver.session() as session:
                batch = []
                for node_id in G.nodes():
                    if node_id.isdigit():  # Βεβαιωθείτε ότι είναι Case κόμβος
                        embedding = model.wv[node_id].tolist()
                        batch.append({"node_id": int(node_id), "embedding": embedding})
                        
                        if len(batch) >= self.BATCH_SIZE:
                            session.run("""
                                UNWIND $batch AS item
                                MATCH (c:Case {id: item.node_id})
                                SET c.embedding = item.embedding,
                                    c.embedding_generated = true,
                                    c.embedding_version = '2.1'
                            """, {"batch": batch})
                            batch = []
                
                if batch:
                    session.run("""
                        UNWIND $batch AS item
                        MATCH (c:Case {id: item.node_id})
                        SET c.embedding = item.embedding,
                            c.embedding_generated = true,
                            c.embedding_version = '2.1'
                    """, {"batch": batch})

            logger.info(f"✅ Saved embeddings for {len(G.nodes)} nodes")
            return True

        except Exception as e:
            logger.error(f"Embedding generation failed: {str(e)}")
            return False

    def build_complete_graph(self):
        """Ολοκληρωμένη διαδικασία δημιουργίας γράφου"""
        driver = self.connect_to_neo4j()
        try:
            # Φόρτωση δεδομένων
            file_path = os.getenv("DATA_URL", "https://raw.githubusercontent.com/GiorgosBouh/llm2cypher-asd-app/main/Toddler_Autism_dataset_July_2018_2.csv")
            df = self.parse_csv(file_path)
            
            # Δημιουργία γράφου
            if not self.create_graph_structure(driver, df):
                raise RuntimeError("Failed to create graph structure")
            
            # Δημιουργία embeddings
            if not self.generate_graph_embeddings(driver):
                raise RuntimeError("Failed to generate embeddings")
            
            logger.info("🎉 Graph construction completed successfully!")
            return True
            
        except Exception as e:
            logger.error(f"Graph build failed: {str(e)}")
            return False
        finally:
            driver.close()

if __name__ == "__main__":
    builder = GraphBuilder()
    success = builder.build_complete_graph()
    sys.exit(0 if success else 1)