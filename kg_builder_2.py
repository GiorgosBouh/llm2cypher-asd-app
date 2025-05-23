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
from dotenv import load_dotenv # ADDED: For consistent env var loading

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# ADDED: Load environment variables here for standalone script execution
load_dotenv()

class GraphBuilder:
    def __init__(self):
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
        # --- CRITICAL FIX: REMOVED HARDCODED DEFAULTS ---
        # Ensure NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD are set in your .env file
        # and are NOT hardcoded here for security and correct database connection.
        return GraphDatabase.driver(
            os.getenv("NEO4J_URI"),
            auth=(
                os.getenv("NEO4J_USER"),
                os.getenv("NEO4J_PASSWORD")
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
            # df.dropna() handles rows where Case_No might have become NaN due to 'coerce'
            df['Case_No'] = df['Case_No'].astype(int)
            
            # Clean and standardize Class_ASD_Traits values: capitalize first letter, strip spaces
            df['Class_ASD_Traits'] = df['Class_ASD_Traits'].astype(str).str.strip().str.capitalize()
            
            logger.info("✅ Cleaned columns: %s", df.columns.tolist())
            # --- IMPROVEMENT: df.dropna() will drop any row with any NaN. 
            # If you want to keep rows with NaNs in non-critical columns,
            # use df.dropna(subset=['Case_No', 'Class_ASD_Traits', ...])
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
            upload_id = str(case_id) # Using Case_No as upload_id for initial build
            
            raw_trait = str(row.get("Class_ASD_Traits", "")).strip().lower()
            if raw_trait == "yes":
                asd_trait = "Yes"
            elif raw_trait == "no":
                asd_trait = "No"
            else:
                asd_trait = None # If label is neither Yes nor No, set to None
            
            case_data.append({
                "id": case_id, 
                "upload_id": upload_id,
                "asd_trait": asd_trait
            })
            
            # Answers to behavior questions
            for q in [f"A{i}" for i in range(1, 11)]:
                answer_val = pd.to_numeric(row.get(q, np.nan), errors='coerce') # Handle potential NaNs in A1-A10
                answer_data.append({
                    "upload_id": upload_id,
                    "q": q,
                    "val": int(answer_val) if not pd.isna(answer_val) else -1 # Default to -1 or specific value if NaN
                })
            
            # Demographic attributes
            demo_cols = ["Sex", "Ethnicity", "Jaundice", "Family_mem_with_ASD"]
            for col in demo_cols:
                val = str(row.get(col, "")).strip() 
                demo_data.append({
                    "upload_id": upload_id,
                    "type": col,
                    "val": val
                })
            
            # Submitter information
            submitter_val = str(row.get("Who_completed_the_test", "")).strip()
            submitter_data.append({
                "upload_id": upload_id,
                "val": submitter_val
            })
        
        # Create all nodes and relationships in batches
        logger.info(f"Preparing to create/merge {len(case_data)} Case nodes and their SCREENED_FOR relationships.")
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
            RETURN count(*) AS createdCases
        """, data=case_data)
        
        logger.info(f"Preparing to create/merge {len(answer_data)} HAS_ANSWER relationships.")
        tx.run("""
            UNWIND $data as row
            MATCH (q:BehaviorQuestion {name: row.q})
            MATCH (c:Case {upload_id: row.upload_id})
            MERGE (c)-[:HAS_ANSWER {value: row.val}]->(q)
        """, data=answer_data)
        
        logger.info(f"Preparing to create/merge {len(demo_data)} HAS_DEMOGRAPHIC relationships.")
        tx.run("""
            UNWIND $data as row
            MATCH (d:DemographicAttribute {type: row.type, value: row.val})
            MATCH (c:Case {upload_id: row.upload_id})
            MERGE (c)-[:HAS_DEMOGRAPHIC]->(d)
        """, data=demo_data)
        
        logger.info(f"Preparing to create/merge {len(submitter_data)} SUBMITTED_BY relationships.")
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
        if all(f'A{k}' in df.columns for k in range(1, 11)):
            for i, row1 in df.iterrows():
                for j, row2 in df.iloc[i+1:].iterrows():
                    shared_answers_count = sum(
                        1 for k in range(1, 11) 
                        if not pd.isna(row1.get(f'A{k}')) and not pd.isna(row2.get(f'A{k}')) and row1[f'A{k}'] == row2[f'A{k}']
                    )
                    if shared_answers_count >= self.MIN_SIMILAR_ANSWERS:
                        pairs.add(tuple(sorted((int(row1['Case_No']), int(row2['Case_No']))))) 
        else:
            logger.warning("Skipping behavioral similarity: A1-A10 columns not all present in DataFrame.")
            
        # Demographic similarity
        demo_cols = ['Sex', 'Ethnicity', 'Jaundice', 'Family_mem_with_ASD']
        actual_demo_cols = [col for col in demo_cols if col in df.columns]

        for col in actual_demo_cols:
            grouped = df.groupby(col)['Case_No'].apply(list)
            for case_list in grouped:
                for i in range(len(case_list)):
                    for j in range(i+1, len(case_list)):
                        pairs.add(tuple(sorted((int(case_list[i]), int(case_list[j])))))
        
        # Apply limit and shuffle
        pair_list = list(pairs)
        shuffle(pair_list)
        pair_list = pair_list[:self.MAX_SIMILAR_PAIRS]
        
        if not pair_list: # ADDED: Check if pair_list is empty
            logger.info("No similarity pairs generated. Skipping SIMILAR_TO relationship creation.")
            return

        similarity_batch_data = []
        for x, y in pair_list:
            weight = self._calculate_similarity_weight(x, y, df)
            similarity_batch_data.append({'id1': x, 'id2': y, 'weight': weight})

        logger.info(f"Preparing to create/merge {len(similarity_batch_data)} SIMILAR_TO relationships.")
        tx.run("""
            UNWIND $batch AS pair
            MATCH (c1:Case {id: pair.id1}), (c2:Case {id: pair.id2})
            MERGE (c1)-[r:SIMILAR_TO]->(c2)
            SET r.weight = pair.weight
        """, batch=similarity_batch_data)


    def _calculate_similarity_weight(self, id1: int, id2: int, df: pd.DataFrame) -> float:
        """Calculate similarity weight between two cases"""
        row1 = df[df['Case_No'] == id1]
        row2 = df[df['Case_No'] == id2]

        if row1.empty or row2.empty:
            logger.warning(f"Case_No {id1} or {id2} not found in DataFrame for similarity calculation. Returning 0.0.")
            return 0.0

        row1 = row1.iloc[0]
        row2 = row2.iloc[0]
        
        # Behavioral similarity (70% weight)
        answer_sim_score = 0.0
        answer_cols = [f'A{i}' for i in range(1, 11)]
        valid_answer_comparisons = 0
        for col in answer_cols:
            if col in row1 and col in row2 and not pd.isna(row1[col]) and not pd.isna(row2[col]):
                if row1[col] == row2[col]:
                    answer_sim_score += 1
                valid_answer_comparisons += 1
        
        answer_sim = answer_sim_score / valid_answer_comparisons if valid_answer_comparisons > 0 else 0.0
        
        # Demographic similarity (30% weight)
        demo_cols = ['Sex', 'Ethnicity', 'Jaundice', 'Family_mem_with_ASD']
        demo_sim_score = 0.0
        valid_demo_comparisons = 0
        for col in demo_cols:
            if col in row1 and col in row2 and not pd.isna(row1[col]) and not pd.isna(row2[col]):
                if str(row1[col]).strip() == str(row2[col]).strip(): 
                    demo_sim_score += 1
                valid_demo_comparisons += 1
        
        demo_sim = demo_sim_score / valid_demo_comparisons if valid_demo_comparisons > 0 else 0.0
        
        return 0.7 * answer_sim + 0.3 * demo_sim

    def generate_embeddings(self, driver: GraphDatabase.driver) -> bool:
        """Generate and store node embeddings"""
        temp_dir = tempfile.mkdtemp()
        try:
            G = self._build_networkx_graph(driver)
            
            if not G.nodes:
                logger.warning("No nodes found in graph to generate embeddings. Skipping.")
                return False
            if not G.edges:
                logger.warning("No edges found in graph to generate embeddings. Node2Vec works best with edges. Proceeding but embeddings might be trivial.")
                # You might want to raise ValueError here if edges are strictly required.
            
            model = self._run_node2vec(G, temp_dir)
            
            return self._store_embeddings(driver, model, G)
        except ValueError as ve:
            logger.error("❌ Graph building failed for embeddings: %s", str(ve))
            return False
        except Exception as e:
            logger.error("❌ Error during embedding generation: %s", str(e), exc_info=True)
            return False
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
            
            # Load all relationships connecting to/from Case nodes
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
                       r.value AS value,
                       r.weight AS weight 
            """)
            
            edge_count = 0
            for record in relationships:
                source = f"Case_{record['source_id']}"
                target = record['target_id']
                
                if target not in G:
                    G.add_node(target, type='Generic') 
                
                edge_attrs = {"type": record['rel_type']}
                if record['value'] is not None:
                    edge_attrs["value"] = record['value']
                
                if record['weight'] is not None:
                    edge_attrs["weight"] = record['weight']
                else:
                    edge_attrs["weight"] = 1.0 # Default weight for non-similar edges

                G.add_edge(source, target, **edge_attrs)
                edge_count += 1
            
            logger.info("📊 Loaded %d edges", edge_count)
        
        if len(G.nodes) == 0:
            raise ValueError("NetworkX graph is empty after loading from Neo4j. No nodes found.")
        if len(G.edges) == 0:
            logger.warning("NetworkX graph has no edges. Node2Vec requires edges for meaningful embeddings. Check graph creation.")
            
        return G

    def _run_node2vec(self, G: nx.Graph, temp_dir: str) -> Node2Vec:
        """Run Node2Vec algorithm on the graph"""
        case_nodes = [n for n in G.nodes if G.nodes[n].get('type') == 'Case']
        logger.info("🔍 Generating embeddings for %d Case nodes", len(case_nodes))
        
        node2vec = Node2Vec(
            G,
            dimensions=self.EMBEDDING_DIM,
            walk_length=self.NODE2VEC_PARAMS['walk_length'],
            num_walks=self.NODE2VEC_PARAMS['num_walks'],
            p=self.NODE2VEC_PARAMS['p'],
            q=self.NODE2VEC_PARAMS['q'],
            workers=self.NODE2VEC_PARAMS['workers'],
            temp_folder=temp_dir,
            quiet=True,
            weighted=True, # ADDED: Ensure weights are used
            weight_key='weight' # ADDED: Specify weight key
        )
        
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
                    if node in model.wv: 
                        embedding = model.wv[node].tolist()
                        if self._validate_embedding_data(embedding): # Renamed helper for clarity
                            batch.append({"case_id": case_id, "embedding": embedding})
                        else:
                            logger.warning(f"⚠️ Invalid embedding data generated for {node}. Skipping save.")
                            continue
                    else:
                        logger.warning(f"⚠️ Node2Vec did not generate embedding for {node}. Skipping save.")
                        continue
                    
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
                    logger.warning("⚠️ No embedding for %s in model.wv. Skipping save.", node)
                    continue
                except Exception as e:
                    logger.error(f"Error processing embedding for {node}: {str(e)}", exc_info=True)
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

    def _validate_embedding_data(self, embedding: List[float]) -> bool: # Renamed helper for clarity
        """Helper to validate embedding list."""
        if not embedding or len(embedding) != self.EMBEDDING_DIM:
            logger.warning(f"Embedding invalid: Length {len(embedding)} != {self.EMBEDDING_DIM} or empty.")
            return False
        if any(np.isnan(x) for x in embedding):
            logger.warning("Embedding contains NaN values.")
            return False
        if np.all(np.array(embedding) == 0):
            logger.warning("Embedding is all zeros.")
            return False
        return True


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
                raise RuntimeError("Embedding generation failed or no embeddings were generated/stored.")
            
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