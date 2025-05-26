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
from dotenv import load_dotenv
import argparse

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

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
        # Initialize no_labels_flag here so it always exists
        self.no_labels_flag = False 

    def connect_to_neo4j(self) -> GraphDatabase.driver:
        """Create Neo4j driver with environment variables"""
        uri = os.getenv("NEO4J_URI")
        user = os.getenv("NEO4J_USER")
        password = os.getenv("NEO4J_PASSWORD")
        
        # Validate environment variables
        if not uri or not user or not password:
            raise ValueError("Missing required Neo4j environment variables: NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD")
            
        password_masked = '*' * len(password) if password else 'N/A'
        logger.info(f"kg_builder_2.py connecting to: URI='{uri}', User='{user}', Pass='{password_masked}'")
        
        return GraphDatabase.driver(
            uri,
            auth=(user, password),
            max_connection_lifetime=7200,
            max_connection_pool_size=50
        )

    def parse_csv(self, file_path: str) -> pd.DataFrame:
        try:
            df = pd.read_csv(file_path, sep=";", encoding="utf-8-sig")
            df.columns = [col.strip() for col in df.columns]
            
            # Clean and convert numeric columns
            numeric_cols = ['Case_No', 'A1', 'A2', 'A3', 'A4', 'A5', 
                            'A6', 'A7', 'A8', 'A9', 'A10', 'Age_Mons', 'Qchat-10-Score']
            for col in numeric_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(
                        df[col].astype(str).str.replace(",", "."), 
                        errors='coerce'
                    )
            
            # Ensure Case_No is integer
            df['Case_No'] = df['Case_No'].astype(int)
            
            # Clean Class_ASD_Traits column
            if 'Class_ASD_Traits' in df.columns:
                df['Class_ASD_Traits'] = df['Class_ASD_Traits'].astype(str).str.strip().str.capitalize()
            
            logger.info("✅ Cleaned columns: %s", df.columns.tolist())
            logger.info(f"✅ Loaded {len(df)} cases from CSV")
            
            # Drop rows with missing Case_No
            df = df.dropna(subset=['Case_No'])
            return df
            
        except Exception as e:
            logger.error("❌ Failed to parse CSV: %s", str(e))
            raise

    def create_nodes(self, tx, df: pd.DataFrame) -> None:
        """Create all required nodes in the graph"""
        try:
            # Create BehaviorQuestion nodes
            for q in [f"A{i}" for i in range(1, 11)]:
                tx.run("MERGE (:BehaviorQuestion {name: $q})", q=q)
            
            # Create DemographicAttribute nodes
            demo_cols = ["Sex", "Ethnicity", "Jaundice", "Family_mem_with_ASD"]
            for col in demo_cols:
                if col in df.columns:
                    unique_values = df[col].dropna().unique()
                    for val in unique_values:
                        val_str = str(val).strip()
                        if val_str and val_str.lower() not in ['nan', 'none', '']:
                            tx.run(
                                "MERGE (:DemographicAttribute {type: $type, value: $val})",
                                type=col, val=val_str
                            )
            
            # Create SubmitterType nodes
            if "Who_completed_the_test" in df.columns:
                unique_submitters = df["Who_completed_the_test"].dropna().unique()
                for val in unique_submitters:
                    val_str = str(val).strip()
                    if val_str and val_str.lower() not in ['nan', 'none', '']:
                        tx.run("MERGE (:SubmitterType {type: $val})", val=val_str)
            
            # Create ASD_Trait nodes
            tx.run("MERGE (:ASD_Trait {label: 'Yes'})")
            tx.run("MERGE (:ASD_Trait {label: 'No'})")
            
            logger.info("✅ Created all node types")
            
        except Exception as e:
            logger.error(f"❌ Error creating nodes: {str(e)}")
            raise

    def create_relationships(self, tx, df: pd.DataFrame, include_labels: bool = True) -> None:
        """Create relationships between nodes"""
        try:
            case_data = []
            answer_data = []
            demo_data = []
            submitter_data = []
            
            for _, row in df.iterrows():
                case_id = int(row["Case_No"])
                upload_id = str(case_id)  # Using case_id as upload_id for consistency
                
                # Handle ASD Trait based on include_labels flag
                asd_trait = None
                if include_labels and "Class_ASD_Traits" in row:
                    raw_trait = str(row["Class_ASD_Traits"]).strip().lower()
                    if raw_trait == "yes":
                        asd_trait = "Yes"
                    elif raw_trait == "no":
                        asd_trait = "No"
                
                case_data.append({
                    "id": case_id, 
                    "upload_id": upload_id,
                    "asd_trait": asd_trait
                })
                
                # Create answer relationships
                for i in range(1, 11):
                    q = f"A{i}"
                    if q in row:
                        answer_val = pd.to_numeric(row[q], errors='coerce')
                        answer_data.append({
                            "upload_id": upload_id,
                            "q": q,
                            "val": int(answer_val) if not pd.isna(answer_val) else 0
                        })
                
                # Create demographic relationships
                demo_cols = ["Sex", "Ethnicity", "Jaundice", "Family_mem_with_ASD"]
                for col in demo_cols:
                    if col in row:
                        val = str(row[col]).strip() 
                        if val and val.lower() not in ['nan', 'none', '']:
                            demo_data.append({
                                "upload_id": upload_id,
                                "type": col,
                                "val": val
                            })
                
                # Create submitter relationship
                if "Who_completed_the_test" in row:
                    submitter_val = str(row["Who_completed_the_test"]).strip()
                    if submitter_val and submitter_val.lower() not in ['nan', 'none', '']:
                        submitter_data.append({
                            "upload_id": upload_id,
                            "val": submitter_val
                        })
            
            # Create Case nodes and SCREENED_FOR relationships
            logger.info(f"Creating {len(case_data)} Case nodes (include_labels={include_labels})")
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
            
            # Create HAS_ANSWER relationships
            if answer_data:
                logger.info(f"Creating {len(answer_data)} HAS_ANSWER relationships")
                tx.run("""
                    UNWIND $data as row
                    MATCH (q:BehaviorQuestion {name: row.q})
                    MATCH (c:Case {upload_id: row.upload_id})
                    MERGE (c)-[:HAS_ANSWER {value: row.val}]->(q)
                """, data=answer_data)
            
            # Create HAS_DEMOGRAPHIC relationships
            if demo_data:
                logger.info(f"Creating {len(demo_data)} HAS_DEMOGRAPHIC relationships")
                tx.run("""
                    UNWIND $data as row
                    MATCH (d:DemographicAttribute {type: row.type, value: row.val})
                    MATCH (c:Case {upload_id: row.upload_id})
                    MERGE (c)-[:HAS_DEMOGRAPHIC]->(d)
                """, data=demo_data)
            
            # Create SUBMITTED_BY relationships
            if submitter_data:
                logger.info(f"Creating {len(submitter_data)} SUBMITTED_BY relationships")
                tx.run("""
                    UNWIND $data as row
                    MATCH (s:SubmitterType {type: row.val})
                    MATCH (c:Case {upload_id: row.upload_id})
                    MERGE (c)-[:SUBMITTED_BY]->(s)
                """, data=submitter_data)
                
            logger.info("✅ Created all relationships")
            
        except Exception as e:
            logger.error(f"❌ Error creating relationships: {str(e)}")
            raise

    def create_similarity_relationships(self, tx, df: pd.DataFrame) -> None:
        """Create similarity relationships between cases"""
        try:
            pairs = set()
            
            # Behavioral similarity
            if all(f'A{k}' in df.columns for k in range(1, 11)):
                logger.info("Creating behavioral similarity pairs...")
                for i, row1 in df.iterrows():
                    for j, row2 in df.iloc[i+1:].iterrows():
                        shared_answers_count = sum(
                            1 for k in range(1, 11) 
                            if (not pd.isna(row1.get(f'A{k}')) and 
                                not pd.isna(row2.get(f'A{k}')) and 
                                row1[f'A{k}'] == row2[f'A{k}'])
                        )
                        if shared_answers_count >= self.MIN_SIMILAR_ANSWERS:
                            pair = tuple(sorted((int(row1['Case_No']), int(row2['Case_No']))))
                            pairs.add(pair)
            else:
                logger.warning("Skipping behavioral similarity: A1-A10 columns not all present")
                
            # Demographic similarity
            demo_cols = ['Sex', 'Ethnicity', 'Jaundice', 'Family_mem_with_ASD']
            actual_demo_cols = [col for col in demo_cols if col in df.columns]
            
            for col in actual_demo_cols:
                grouped = df.groupby(col)['Case_No'].apply(list)
                for case_list in grouped:
                    if len(case_list) > 1:
                        for i in range(len(case_list)):
                            for j in range(i+1, len(case_list)):
                                pair = tuple(sorted((int(case_list[i]), int(case_list[j]))))
                                pairs.add(pair)
            
            # Limit pairs and shuffle
            pair_list = list(pairs)
            shuffle(pair_list)
            pair_list = pair_list[:self.MAX_SIMILAR_PAIRS]
            
            if not pair_list:
                logger.info("No similarity pairs generated")
                return

            # Calculate weights and create relationships
            similarity_batch_data = []
            for x, y in pair_list:
                weight = self._calculate_similarity_weight(x, y, df)
                similarity_batch_data.append({'id1': x, 'id2': y, 'weight': weight})

            logger.info(f"Creating {len(similarity_batch_data)} SIMILAR_TO relationships")
            tx.run("""
                UNWIND $batch AS pair
                MATCH (c1:Case {id: pair.id1}), (c2:Case {id: pair.id2})
                MERGE (c1)-[r:SIMILAR_TO]->(c2)
                SET r.weight = pair.weight
            """, batch=similarity_batch_data)
            
            logger.info("✅ Created similarity relationships")
            
        except Exception as e:
            logger.error(f"❌ Error creating similarity relationships: {str(e)}")
            raise

    def _calculate_similarity_weight(self, id1: int, id2: int, df: pd.DataFrame) -> float:
        """Calculate similarity weight between two cases"""
        try:
            row1 = df[df['Case_No'] == id1]
            row2 = df[df['Case_No'] == id2]

            if row1.empty or row2.empty:
                logger.warning(f"Case_No {id1} or {id2} not found in DataFrame")
                return 0.0

            row1 = row1.iloc[0]
            row2 = row2.iloc[0]
            
            # Answer similarity
            answer_sim_score = 0.0
            answer_cols = [f'A{i}' for i in range(1, 11)]
            valid_answer_comparisons = 0
            
            for col in answer_cols:
                if (col in row1 and col in row2 and 
                    not pd.isna(row1[col]) and not pd.isna(row2[col])):
                    if row1[col] == row2[col]:
                        answer_sim_score += 1
                    valid_answer_comparisons += 1
            
            answer_sim = answer_sim_score / valid_answer_comparisons if valid_answer_comparisons > 0 else 0.0
            
            # Demographic similarity
            demo_cols = ['Sex', 'Ethnicity', 'Jaundice', 'Family_mem_with_ASD']
            demo_sim_score = 0.0
            valid_demo_comparisons = 0
            
            for col in demo_cols:
                if (col in row1 and col in row2 and 
                    not pd.isna(row1[col]) and not pd.isna(row2[col])):
                    if str(row1[col]).strip() == str(row2[col]).strip(): 
                        demo_sim_score += 1
                    valid_demo_comparisons += 1
            
            demo_sim = demo_sim_score / valid_demo_comparisons if valid_demo_comparisons > 0 else 0.0
            
            return 0.7 * answer_sim + 0.3 * demo_sim
            
        except Exception as e:
            logger.error(f"Error calculating similarity weight: {str(e)}")
            return 0.0

    def generate_embeddings(self, driver: GraphDatabase.driver) -> bool:
        """Generate embeddings for all cases"""
        temp_dir = tempfile.mkdtemp()
        try:
            G = self._build_networkx_graph(driver)
            
            if not G.nodes:
                logger.warning("No nodes found in graph")
                return False
            if not G.edges:
                logger.warning("No edges found in graph - embeddings might be trivial")
            
            model = self._run_node2vec(G, temp_dir)
            return self._store_embeddings(driver, model, G)
            
        except Exception as e:
            logger.error(f"❌ Error during embedding generation: {str(e)}", exc_info=True)
            return False
        finally:
            self._cleanup_temp_dir(temp_dir)

    def _build_networkx_graph(self, driver: GraphDatabase.driver) -> nx.Graph:
        """Build NetworkX graph from Neo4j"""
        try:
            G = nx.Graph()
            
            with driver.session() as session:
                # Load Case nodes
                cases = session.run("MATCH (c:Case) RETURN c.id AS id")
                case_ids = [f"Case_{record['id']}" for record in cases]
                G.add_nodes_from(case_ids, type="Case")
                logger.info(f"📊 Loaded {len(case_ids)} Case nodes")
                
                # Build relationships query
                query = """
                    MATCH (c:Case)-[r]->(n)
                    """
                if self.no_labels_flag:
                    query += "WHERE type(r) <> 'SCREENED_FOR'\n"
                
                query += """
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
                """
                
                relationships = session.run(query)
                
                edge_count = 0
                for record in relationships:
                    source = f"Case_{record['source_id']}"
                    target = record['target_id']
                    
                    if target not in G:
                        G.add_node(target, type='Generic') 
                    
                    edge_attrs = {"type": record['rel_type']}
                    if record['value'] is not None:
                        edge_attrs["value"] = record['value']
                    
                    weight = record.get('weight', 1.0)
                    if weight is None or not np.isfinite(weight) or weight <= 0:
                        weight = 1.0
                    edge_attrs["weight"] = float(weight)

                    G.add_edge(source, target, **edge_attrs)
                    edge_count += 1
                
                logger.info(f"📊 Loaded {edge_count} edges")
            
            if len(G.nodes) == 0:
                raise ValueError("NetworkX graph is empty - no nodes found")
                
            return G
            
        except Exception as e:
            logger.error(f"Error building NetworkX graph: {str(e)}")
            raise

    def _run_node2vec(self, G: nx.Graph, temp_dir: str) -> Node2Vec:
        """Run Node2Vec algorithm on the graph"""
        try:
            case_nodes = [n for n in G.nodes if G.nodes[n].get('type') == 'Case']
            logger.info(f"🔍 Generating embeddings for {len(case_nodes)} Case nodes")
            
            # Try with weighted graph first
            try:
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
            except TypeError:
                # Fallback for older Node2Vec versions
                logger.info("Using legacy Node2Vec initialization")
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
            
            model = node2vec.fit(
                window=self.NODE2VEC_PARAMS['window'],
                min_count=self.NODE2VEC_PARAMS['min_count'],
                batch_words=self.NODE2VEC_PARAMS['batch_words']
            )
            
            return model
            
        except Exception as e:
            logger.error(f"Error running Node2Vec: {str(e)}")
            raise

    def _store_embeddings(self, driver: GraphDatabase.driver, model: Node2Vec, G: nx.Graph) -> bool:
        """Store generated embeddings back to Neo4j"""
        try:
            case_nodes = [n for n in G.nodes if G.nodes[n].get('type') == 'Case']
            saved_count = 0
            
            with driver.session() as session:
                batch = []
                for node in case_nodes:
                    try:
                        case_id = int(node.split('_')[1])
                        if node in model.wv: 
                            embedding = model.wv[node].tolist()
                            if self._validate_embedding_data(embedding): 
                                batch.append({"case_id": case_id, "embedding": embedding})
                            else:
                                logger.warning(f"⚠️ Invalid embedding for {node}")
                                continue
                        else:
                            logger.warning(f"⚠️ No embedding generated for {node}")
                            continue
                        
                        if len(batch) >= self.BATCH_SIZE:
                            session.run("""
                                UNWIND $batch AS item
                                MATCH (c:Case {id: item.case_id})
                                SET c.embedding = item.embedding,
                                    c.embedding_version = 2.1
                            """, batch=batch)
                            saved_count += len(batch)
                            batch = []
                            logger.info(f"✅ Saved {saved_count}/{len(case_nodes)} embeddings")
                    
                    except Exception as e:
                        logger.error(f"Error processing embedding for {node}: {str(e)}")
                        continue
                
                # Save remaining batch
                if batch:
                    session.run("""
                        UNWIND $batch AS item
                        MATCH (c:Case {id: item.case_id})
                        SET c.embedding = item.embedding,
                            c.embedding_version = 2.1
                    """, batch=batch)
                    saved_count += len(batch)
            
            logger.info(f"✅ Completed saving {saved_count} embeddings")
            return saved_count > 0
            
        except Exception as e:
            logger.error(f"Error storing embeddings: {str(e)}")
            return False

    def _validate_embedding_data(self, embedding: List[float]) -> bool:
        """Validate embedding data"""
        if not embedding or len(embedding) != self.EMBEDDING_DIM:
            return False
        if any(np.isnan(x) or not np.isfinite(x) for x in embedding):
            return False
        if np.all(np.array(embedding) == 0):
            return False
        return True

    def _cleanup_temp_dir(self, temp_dir: str) -> None:
        """Clean up temporary directory"""
        try:
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
        except Exception as e:
            logger.warning(f"⚠️ Error cleaning temp directory: {str(e)}")

    def build_full_graph(self) -> None:
        """Build the entire graph from scratch"""
        self.no_labels_flag = False
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
                
                logger.info("⏳ Creating relationships...")
                session.execute_write(self.create_relationships, df, include_labels=True)
                
                logger.info("⏳ Creating similarity relationships...")
                session.execute_write(self.create_similarity_relationships, df)
            
            logger.info("⏳ Generating embeddings...")
            if not self.generate_embeddings(driver):
                raise RuntimeError("Embedding generation failed")
            
            logger.info("✅ Full graph built successfully!")
            
        except Exception as e:
            logger.critical(f"❌ Critical error during full graph build: {str(e)}", exc_info=True)
            raise
        finally:
            if driver:
                driver.close()

    def generate_embeddings_only(self) -> None:
        """Generate embeddings for existing graph"""
        self.no_labels_flag = True
        driver = None
        try:
            driver = self.connect_to_neo4j()
            
            logger.info("⏳ Generating embeddings for existing graph (excluding labels)...")
            if not self.generate_embeddings(driver):
                raise RuntimeError("Embedding generation failed")
            
            logger.info("✅ Embeddings generated successfully!")
            
        except Exception as e:
            logger.critical(f"❌ Critical error during embeddings generation: {str(e)}", exc_info=True)
            raise
        finally:
            if driver:
                driver.close()

# Main execution
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build Neo4j Knowledge Graph for ASD data")
    parser.add_argument('--no-labels', action='store_true', 
                      help='Generate embeddings without label information')
    parser.add_argument('--build-full-graph', action='store_true',
                        help='Delete and rebuild entire graph')
    parser.add_argument('--generate-embeddings-only', action='store_true',
                        help='Generate embeddings for existing graph')
    args = parser.parse_args()

    builder = GraphBuilder()
    builder.no_labels_flag = args.no_labels

    try:
        if args.build_full_graph:
            logger.info("Running in --build-full-graph mode")
            builder.build_full_graph()
        elif args.generate_embeddings_only:
            logger.info("Running in --generate-embeddings-only mode")
            builder.generate_embeddings_only()
        elif args.no_labels:
            logger.info("Running in --no-labels mode")
            builder.generate_embeddings_only()
        else:
            logger.info("No arguments provided. Running full graph build")
            builder.build_full_graph()
        
        sys.exit(0)
    except Exception as e:
        logger.error(f"Execution failed: {e}")
        sys.exit(1)