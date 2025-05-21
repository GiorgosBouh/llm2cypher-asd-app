import pandas as pd
import numpy as np
from neo4j import GraphDatabase
import networkx as nx
from node2vec import Node2Vec
from random import shuffle
import traceback
import sys
import os
import shutil
import logging
import tempfile
from typing import Dict, List, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

class GraphBuilder:
    def __init__(self):
        # Configuration aligned with other scripts
        self.EMBEDDING_DIM = 128
        self.NODE2VEC_PARAMS = {
            "walk_length": 30,
            "num_walks": 100,
            "p": 1.0,
            "q": 0.5,
            "workers": 4,
            "window": 10,
            "min_count": 1,
            "batch_words": 128
        }
        self.MAX_SIMILAR_PAIRS = 10000
        self.MIN_SIMILAR_ANSWERS = 7  # For similarity relationships
        self.BATCH_SIZE = 500

    def connect_to_neo4j(self) -> GraphDatabase.driver:
        """Create Neo4j driver with environment variables"""
        return GraphDatabase.driver(
            os.getenv("NEO4J_URI", "neo4j+s://1f5f8a14.databases.neo4j.io"),
            auth=(
                os.getenv("NEO4J_USER", "neo4j"),
                os.getenv("NEO4J_PASSWORD", "3xhy4XKQSsSLIT7NI-w9m4Z7Y_WcVnL1hDQkWTMIoMQ")
            ),
            max_connection_lifetime=7200,
            max_connection_pool_size=50
        )

    def parse_csv(self, file_path: str) -> pd.DataFrame:
        """Load and clean the dataset"""
        try:
            df = pd.read_csv(file_path, sep=";", encoding="utf-8-sig")
            df.columns = [col.strip() for col in df.columns]
            
            # Convert numeric columns
            numeric_cols = ['Case_No', 'A1', 'A2', 'A3', 'A4', 'A5', 
                            'A6', 'A7', 'A8', 'A9', 'A10', 'Age_Mons', 'Qchat-10-Score']
            for col in numeric_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(
                        df[col].astype(str).str.replace(",", "."), 
                        errors='coerce'
                    )
            
            # Convert Case_No to int to avoid float keys
            df['Case_No'] = df['Case_No'].astype(int)
            
            # Clean and standardize Class_ASD_Traits values: capitalize first letter, strip spaces
            df['Class_ASD_Traits'] = df['Class_ASD_Traits'].astype(str).str.strip().str.capitalize()
            
            logger.info("✅ Cleaned columns: %s", df.columns.tolist())
            return df.dropna()
        except Exception as e:
            logger.error("❌ Failed to parse CSV: %s", str(e))
            raise

    def create_nodes(self, tx, df: pd.DataFrame) -> None:
        """Create all nodes in the graph"""
        # Behavior Questions
        for q in [f"A{i}" for i in range(1, 11)]:
            tx.run("MERGE (:BehaviorQuestion {name: $q})", q=q)
        
        # Demographic Attributes
        demo_cols = ["Sex", "Ethnicity", "Jaundice", "Family_mem_with_ASD"]
        for col in demo_cols:
            for val in df[col].dropna().unique():
                tx.run(
                    "MERGE (:DemographicAttribute {type: $type, value: $val})",
                    type=col, val=val
                )
        
        # Submitter Types
        for val in df["Who_completed_the_test"].dropna().unique():
            tx.run("MERGE (:SubmitterType {type: $val})", val=val)
        
        # Create ASD_Trait nodes explicitly for "Yes" and "No"
        tx.run("MERGE (:ASD_Trait {label: 'Yes'})")
        tx.run("MERGE (:ASD_Trait {label: 'No'})")

    def create_relationships(self, tx, df: pd.DataFrame) -> None:
        """Create all relationships between nodes"""
        # Prepare batch data
        case_data = []
        answer_data = []
        demo_data = []
        submitter_data = []
        
        for _, row in df.iterrows():
            case_id = int(row["Case_No"])
            upload_id = str(case_id)
            
            # Normalize asd_trait label to capitalized form, safe to ignore if invalid
            raw_trait = str(row.get("Class_ASD_Traits", "")).strip().lower()
            if raw_trait == "yes":
                asd_trait = "Yes"
            elif raw_trait == "no":
                asd_trait = "No"
            else:
                asd_trait = None
            
            case_data.append({
                "id": case_id, 
                "upload_id": upload_id,
                "asd_trait": asd_trait
            })
            
            # Answers to behavior questions
            for q in [f"A{i}" for i in range(1, 11)]:
                answer_data.append({
                    "upload_id": upload_id,
                    "q": q,
                    "val": int(row[q])
                })
            
            # Demographic attributes
            demo_cols = ["Sex", "Ethnicity", "Jaundice", "Family_mem_with_ASD"]
            for col in demo_cols:
                demo_data.append({
                    "upload_id": upload_id,
                    "type": col,
                    "val": row[col]
                })
            
            # Submitter information
            submitter_data.append({
                "upload_id": upload_id,
                "val": row["Who_completed_the_test"]
            })
        
        # Create all nodes and relationships in batches
        # First create Case nodes with ASD_Trait relationship if trait is valid
        tx.run("""
            UNWIND $data as row 
            MERGE (c:Case {id: row.id}) 
            SET c.upload_id = row.upload_id, c.embedding = null
            WITH c, row
            CALL apoc.do.when(
                row.asd_trait IS NOT NULL,
                'MERGE (t:ASD_Trait {label: row.asd_trait}) MERGE (c)-[:SCREENED_FOR]->(t)',
                '',
                {c:c, row:row}
            ) YIELD value
            RETURN count(*) 
        """, data=case_data)
        
        # Create HAS_ANSWER relationships
        tx.run("""
            UNWIND $data as row
            MATCH (q:BehaviorQuestion {name: row.q})
            MATCH (c:Case {upload_id: row.upload_id})
            MERGE (c)-[:HAS_ANSWER {value: row.val}]->(q)
        """, data=answer_data)
        
        # Create HAS_DEMOGRAPHIC relationships
        tx.run("""
            UNWIND $data as row
            MATCH (d:DemographicAttribute {type: row.type, value: row.val})
            MATCH (c:Case {upload_id: row.upload_id})
            MERGE (c)-[:HAS_DEMOGRAPHIC]->(d)
        """, data=demo_data)
        
        # Create SUBMITTED_BY relationships
        tx.run("""
            UNWIND $data as row
            MATCH (s:SubmitterType {type: row.val})
            MATCH (c:Case {upload_id: row.upload_id})
            MERGE (c)-[:SUBMITTED_BY]->(s)
        """, data=submitter_data)

    def create_similarity_relationships(self, tx, df: pd.DataFrame) -> None:
        """Create similarity relationships between cases"""
        pairs = set()
        
        # Behavioral similarity (shared answers)
        for i, row1 in df.iterrows():
            for j, row2 in df.iloc[i+1:].iterrows():
                if sum(row1[f'A{k}'] == row2[f'A{k}'] for k in range(1,11)) >= self.MIN_SIMILAR_ANSWERS:
                    pairs.add((int(row1['Case_No']), int(row2['Case_No'])))
        
        # Demographic similarity
        demo_cols = ['Sex', 'Ethnicity', 'Jaundice', 'Family_mem_with_ASD']
        for col in demo_cols:
            grouped = df.groupby(col)['Case_No'].apply(list)
            for case_list in grouped:
                for i in range(len(case_list)):
                    for j in range(i+1, len(case_list)):
                        pairs.add((int(case_list[i]), int(case_list[j])))
        
        # Apply limit and shuffle
        pair_list = list(pairs)[:self.MAX_SIMILAR_PAIRS]
        shuffle(pair_list)
        
        # Create relationships with weights
        tx.run("""
            UNWIND $batch AS pair
            MATCH (c1:Case {id: pair.id1}), (c2:Case {id: pair.id2})
            MERGE (c1)-[r:SIMILAR_TO]->(c2)
            SET r.weight = pair.weight
        """, batch=[{
            'id1': x,
            'id2': y,
            'weight': self._calculate_similarity_weight(x, y, df)
        } for x,y in pair_list])

    def _calculate_similarity_weight(self, id1: int, id2: int, df: pd.DataFrame) -> float:
        """Calculate similarity weight between two cases"""
        row1 = df[df['Case_No'] == id1].iloc[0]
        row2 = df[df['Case_No'] == id2].iloc[0]
        
        # Behavioral similarity (70% weight)
        answer_sim = sum(row1[f'A{i}'] == row2[f'A{i}'] for i in range(1,11)) / 10.0
        
        # Demographic similarity (30% weight)
        demo_cols = ['Sex', 'Ethnicity', 'Jaundice', 'Family_mem_with_ASD']
        demo_sim = sum(row1[col] == row2[col] for col in demo_cols) / len(demo_cols)
        
        return 0.7 * answer_sim + 0.3 * demo_sim

    def generate_embeddings(self, driver: GraphDatabase.driver) -> bool:
        """Generate and store node embeddings"""
        temp_dir = tempfile.mkdtemp()
        try:
            # Build networkx graph from Neo4j data
            G = self._build_networkx_graph(driver)
            
            # Generate embeddings using Node2Vec
            model = self._run_node2vec(G, temp_dir)
            
            # Store embeddings in Neo4j
            return self._store_embeddings(driver, model, G)
        finally:
            self._cleanup_temp_dir(temp_dir)

    def _build_networkx_graph(self, driver: GraphDatabase.driver) -> nx.Graph:
        """Construct the networkx graph from Neo4j data"""
        G = nx.Graph()
        
        with driver.session() as session:
            # Load all Case nodes
            cases = session.run("MATCH (c:Case) RETURN c.id AS id")
            case_ids = [f"Case_{record['id']}" for record in cases]
            G.add_nodes_from(case_ids, type="Case")
            logger.info("📊 Loaded %d Case nodes", len(case_ids))
            
            # Load all relationships
            relationships = session.run("""
                MATCH (c:Case)-[r]->(n)
                RETURN c.id AS source_id, 
                       type(r) AS rel_type,
                       CASE 
                         WHEN n:BehaviorQuestion THEN 'Q_' + n.name
                         WHEN n:DemographicAttribute THEN 'D_' + n.type + '_' + replace(n.value, ' ', '_')
                         WHEN n:SubmitterType THEN 'S_' + replace(n.type, ' ', '_')
                         WHEN n:Case THEN 'Case_' + toString(n.id)
                         ELSE 'Node_' + toString(id(n))
                       END AS target_id,
                       r.value AS value
            """)
            
            edge_count = 0
            for record in relationships:
                source = f"Case_{record['source_id']}"
                target = record['target_id']
                G.add_node(target)
                
                # Add edge with properties
                edge_attrs = {"type": record['rel_type']}
                if record['value'] is not None:
                    edge_attrs["value"] = record['value']
                
                G.add_edge(source, target, **edge_attrs)
                edge_count += 1
            
            logger.info("📊 Loaded %d edges", edge_count)
        
        if len(G.edges) == 0:
            raise ValueError("Empty graph - check Neo4j data")
        
        return G

    def _run_node2vec(self, G: nx.Graph, temp_dir: str) -> Node2Vec:
        """Run Node2Vec algorithm on the graph"""
        case_nodes = [n for n in G.nodes if G.nodes[n].get('type') == 'Case']
        logger.info("🔍 Generating embeddings for %d Case nodes", len(case_nodes))
        
        # Initialize Node2Vec with the correct parameters
        node2vec = Node2Vec(
            G,
            dimensions=self.EMBEDDING_DIM,
            walk_length=self.NODE2VEC_PARAMS['walk_length'],
            num_walks=self.NODE2VEC_PARAMS['num_walks'],
            p=self.NODE2VEC_PARAMS['p'],
            q=self.NODE2VEC_PARAMS['q'],
            workers=self.NODE2VEC_PARAMS['workers'],
            temp_folder=temp_dir,
            quiet=True
        )
        
        # Fit the model with the remaining parameters
        return node2vec.fit(
            window=self.NODE2VEC_PARAMS['window'],
            min_count=self.NODE2VEC_PARAMS['min_count'],
            batch_words=self.NODE2VEC_PARAMS['batch_words']
        )

    def _store_embeddings(self, driver: GraphDatabase.driver, model: Node2Vec, G: nx.Graph) -> bool:
        """Store generated embeddings back to Neo4j"""
        case_nodes = [n for n in G.nodes if G.nodes[n].get('type') == 'Case']
        saved_count = 0
        
        with driver.session() as session:
            batch = []
            for node in case_nodes:
                try:
                    case_id = int(node.split('_')[1])
                    embedding = model.wv[node].tolist()
                    batch.append({"case_id": case_id, "embedding": embedding})
                    
                    if len(batch) >= self.BATCH_SIZE:
                        session.run("""
                            UNWIND $batch AS item
                            MATCH (c:Case {id: item.case_id})
                            SET c.embedding = item.embedding,
                                c.embedding_version = 2.1
                        """, {"batch": batch})
                        saved_count += len(batch)
                        batch = []
                        logger.info("✅ Saved %d/%d embeddings", saved_count, len(case_nodes))
                
                except KeyError:
                    logger.warning("⚠️ No embedding for %s", node)
                    continue
            
            if batch:
                session.run("""
                    UNWIND $batch AS item
                    MATCH (c:Case {id: item.case_id})
                    SET c.embedding = item.embedding,
                        c.embedding_version = 2.1
                """, {"batch": batch})
                saved_count += len(batch)
        
        logger.info("✅ Completed saving %d embeddings", saved_count)
        return saved_count > 0

    def _cleanup_temp_dir(self, temp_dir: str) -> None:
        """Clean up temporary directory"""
        try:
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
        except Exception as e:
            logger.warning("⚠️ Error cleaning temp directory: %s", str(e))

    def build_graph(self) -> None:
        """Main graph building workflow"""
        driver = None
        try:
            driver = self.connect_to_neo4j()
            df = self.parse_csv(
                "https://raw.githubusercontent.com/GiorgosBouh/llm2cypher-asd-app/main/Toddler_Autism_dataset_July_2018_2.csv"
            )
            
            with driver.session() as session:
                logger.info("🧹 Deleting old graph...")
                session.run("MATCH (n) DETACH DELETE n")
                
                logger.info("⏳ Creating nodes...")
                session.execute_write(self.create_nodes, df)
                
                logger.info("⏳ Creating all relationships...")
                session.execute_write(self.create_relationships, df)
                
                logger.info("⏳ Creating similarity relationships...")
                session.execute_write(self.create_similarity_relationships, df)
            
            logger.info("⏳ Generating embeddings...")
            if not self.generate_embeddings(driver):
                raise RuntimeError("Embedding generation failed")
            
            logger.info("✅ Graph built successfully!")
            sys.exit(0)
            
        except Exception as e:
            logger.critical("❌ Critical error: %s", str(e), exc_info=True)
            sys.exit(1)
        finally:
            if driver:
                driver.close()

if __name__ == "__main__":
    GraphBuilder().build_graph()