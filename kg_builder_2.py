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
import argparse
import json
from env_utils import load_project_env

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

load_project_env()

class GraphBuilder:
    def __init__(
        self,
        include_similarity: bool = True,
        include_demographics: bool = True,
        include_behavior_patterns: bool = False,
        csv_path: str | None = None,
        weight_profile: Dict[str, float] | None = None,
    ):
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
        self.include_similarity = include_similarity
        self.include_demographics = include_demographics
        self.include_behavior_patterns = include_behavior_patterns
        self.csv_path = csv_path or "https://raw.githubusercontent.com/GiorgosBouh/llm2cypher-asd-app/main/Toddler_Autism_dataset_July_2018_2.csv"
        self.weight_profile = self._build_weight_profile(weight_profile)

    def _build_weight_profile(self, override: Dict[str, float] | None = None) -> Dict[str, float]:
        profile = {
            "answer_edge_weight": 1.0,
            "demographic_edge_weight": 1.0,
            "submitter_edge_weight": 1.0,
            "pattern_edge_weight": 1.8,
            "complexity_edge_weight": 1.5,
            "similarity_answer_component": 0.7,
            "similarity_demo_component": 0.3,
            "similarity_global_scale": 1.0,
            "demo_family_yes_multiplier": 1.0,
            "demo_jaundice_yes_multiplier": 1.0,
            "demo_male_multiplier": 1.0,
        }
        if override:
            profile.update(override)
        return profile

    def _demographic_edge_weight(self, demo_type: str, demo_value: str) -> float:
        weight = float(self.weight_profile["demographic_edge_weight"])
        demo_value_normalized = str(demo_value).strip().lower()

        if demo_type == "Family_mem_with_ASD" and demo_value_normalized == "yes":
            weight *= float(self.weight_profile["demo_family_yes_multiplier"])
        elif demo_type == "Jaundice" and demo_value_normalized == "yes":
            weight *= float(self.weight_profile["demo_jaundice_yes_multiplier"])
        elif demo_type == "Sex" and demo_value_normalized in {"m", "male"}:
            weight *= float(self.weight_profile["demo_male_multiplier"])

        return float(weight)

    def connect_to_neo4j(self) -> GraphDatabase.driver:
        """Create Neo4j driver with environment variables"""
        uri = os.getenv("NEO4J_URI")
        user = os.getenv("NEO4J_USER")
        password = os.getenv("NEO4J_PASSWORD")
        
        # Validate environment variables
        if not uri or not user or not password:
            raise ValueError("Missing required Neo4j environment variables: NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD")
            
        logger.info("kg_builder_2.py connecting to Neo4j with configured environment variables")
        
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
            if self.include_demographics:
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
            if self.include_demographics and "Who_completed_the_test" in df.columns:
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

    def ensure_schema(self, tx) -> None:
        """Create indexes/constraints needed for acceptable graph build performance."""
        statements = [
            """
            CREATE CONSTRAINT case_id_unique IF NOT EXISTS
            FOR (c:Case) REQUIRE c.id IS UNIQUE
            """,
            """
            CREATE CONSTRAINT case_upload_id_unique IF NOT EXISTS
            FOR (c:Case) REQUIRE c.upload_id IS UNIQUE
            """,
            """
            CREATE CONSTRAINT behavior_question_name_unique IF NOT EXISTS
            FOR (q:BehaviorQuestion) REQUIRE q.name IS UNIQUE
            """,
            """
            CREATE CONSTRAINT demographic_attribute_unique IF NOT EXISTS
            FOR (d:DemographicAttribute) REQUIRE (d.type, d.value) IS UNIQUE
            """,
            """
            CREATE CONSTRAINT submitter_type_unique IF NOT EXISTS
            FOR (s:SubmitterType) REQUIRE s.type IS UNIQUE
            """,
            """
            CREATE CONSTRAINT asd_trait_label_unique IF NOT EXISTS
            FOR (t:ASD_Trait) REQUIRE t.label IS UNIQUE
            """,
            """
            CREATE CONSTRAINT behavior_pattern_unique IF NOT EXISTS
            FOR (p:BehaviorPattern) REQUIRE (p.pattern, p.intensity) IS UNIQUE
            """,
            """
            CREATE CONSTRAINT complexity_pattern_unique IF NOT EXISTS
            FOR (p:ComplexityPattern) REQUIRE p.level IS UNIQUE
            """,
        ]

        for statement in statements:
            tx.run(statement).consume()

        logger.info("✅ Ensured Neo4j schema constraints")

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
                if self.include_demographics:
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
                if self.include_demographics and "Who_completed_the_test" in row:
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
                    MERGE (c)-[r:HAS_ANSWER]->(q)
                    SET r.value = row.val,
                        r.weight = row.weight
                """, data=[
                    {
                        **row,
                        "weight": float(self.weight_profile["answer_edge_weight"]),
                    }
                    for row in answer_data
                ])
            
            # Create HAS_DEMOGRAPHIC relationships
            if demo_data:
                logger.info(f"Creating {len(demo_data)} HAS_DEMOGRAPHIC relationships")
                tx.run("""
                    UNWIND $data as row
                    MATCH (d:DemographicAttribute {type: row.type, value: row.val})
                    MATCH (c:Case {upload_id: row.upload_id})
                    MERGE (c)-[r:HAS_DEMOGRAPHIC]->(d)
                    SET r.weight = row.weight
                """, data=[
                    {
                        **row,
                        "weight": self._demographic_edge_weight(row["type"], row["val"]),
                    }
                    for row in demo_data
                ])
            
            # Create SUBMITTED_BY relationships
            if submitter_data:
                logger.info(f"Creating {len(submitter_data)} SUBMITTED_BY relationships")
                tx.run("""
                    UNWIND $data as row
                    MATCH (s:SubmitterType {type: row.val})
                    MATCH (c:Case {upload_id: row.upload_id})
                    MERGE (c)-[r:SUBMITTED_BY]->(s)
                    SET r.weight = row.weight
                """, data=[
                    {
                        **row,
                        "weight": float(self.weight_profile["submitter_edge_weight"]),
                    }
                    for row in submitter_data
                ])
                
            logger.info("✅ Created all relationships")
            
        except Exception as e:
            logger.error(f"❌ Error creating relationships: {str(e)}")
            raise

    def create_behavior_pattern_relationships(self, tx, df: pd.DataFrame) -> None:
        """Create optional behavior-pattern and complexity nodes for ablation experiments."""
        try:
            if not self.include_behavior_patterns:
                logger.info("Skipping behavior pattern node creation")
                return

            pattern_rows = []
            complexity_rows = []

            for _, row in df.iterrows():
                case_id = int(row["Case_No"])
                upload_id = str(case_id)
                answers = {}
                for i in range(1, 11):
                    value = pd.to_numeric(row.get(f"A{i}"), errors="coerce")
                    answers[f"A{i}"] = int(value) if not pd.isna(value) else 0

                complexity_score = sum(1 for value in answers.values() if value == 1)
                if complexity_score >= 5:
                    complexity_level = "HIGH"
                elif complexity_score >= 2:
                    complexity_level = "MODERATE"
                else:
                    complexity_level = "LOW"

                complexity_rows.append({
                    "upload_id": upload_id,
                    "level": complexity_level,
                    "score": int(complexity_score),
                })

                communication = answers["A1"] + answers["A2"] + answers["A8"]
                social = answers["A3"] + answers["A4"] + answers["A6"] + answers["A7"]
                activity = answers["A5"] + answers["A9"] + answers["A10"]

                if communication >= 2:
                    pattern_rows.append({
                        "upload_id": upload_id,
                        "pattern_group": "communication",
                        "intensity": int(communication),
                        "weight": 1.8,
                    })
                if social >= 2:
                    pattern_rows.append({
                        "upload_id": upload_id,
                        "pattern_group": "social",
                        "intensity": int(social),
                        "weight": 1.8,
                    })
                if activity >= 2:
                    pattern_rows.append({
                        "upload_id": upload_id,
                        "pattern_group": "activity",
                        "intensity": int(activity),
                        "weight": 1.8,
                    })

            if complexity_rows:
                logger.info(f"Creating {len(complexity_rows)} HAS_COMPLEXITY relationships")
                tx.run("""
                    UNWIND $rows AS row
                    MATCH (c:Case {upload_id: row.upload_id})
                    MERGE (p:ComplexityPattern {level: row.level})
                    MERGE (c)-[r:HAS_COMPLEXITY]->(p)
                    SET r.weight = row.weight,
                        r.complexity_score = row.score
                """, rows=[
                    {
                        **row,
                        "weight": float(self.weight_profile["complexity_edge_weight"]),
                    }
                    for row in complexity_rows
                ])

            if pattern_rows:
                logger.info(f"Creating {len(pattern_rows)} HAS_PATTERN relationships")
                tx.run("""
                    UNWIND $rows AS row
                    MATCH (c:Case {upload_id: row.upload_id})
                    MERGE (p:BehaviorPattern {pattern: row.pattern_group, intensity: row.intensity})
                    MERGE (c)-[r:HAS_PATTERN]->(p)
                    SET r.weight = row.weight
                """, rows=[
                    {
                        **row,
                        "weight": float(self.weight_profile["pattern_edge_weight"]),
                    }
                    for row in pattern_rows
                ])

            logger.info("✅ Created optional behavior pattern relationships")

        except Exception as e:
            logger.error(f"❌ Error creating behavior pattern relationships: {str(e)}")
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
            
            similarity_answer_component = float(self.weight_profile["similarity_answer_component"])
            similarity_demo_component = float(self.weight_profile["similarity_demo_component"])
            similarity_global_scale = float(self.weight_profile["similarity_global_scale"])

            return similarity_global_scale * (
                similarity_answer_component * answer_sim +
                similarity_demo_component * demo_sim
            )
            
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
                             WHEN n:BehaviorPattern THEN 'P_' + n.pattern + '_' + toString(n.intensity)
                             WHEN n:ComplexityPattern THEN 'C_' + n.level
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
            df = self.parse_csv(self.csv_path)
            
            with driver.session() as session:
                logger.info("⏳ Ensuring schema constraints...")
                session.execute_write(self.ensure_schema)

                logger.info("🧹 Deleting old graph...")
                session.run("MATCH (n) DETACH DELETE n")
                
                logger.info("⏳ Creating nodes...")
                session.execute_write(self.create_nodes, df)
                
                logger.info("⏳ Creating relationships...")
                session.execute_write(self.create_relationships, df, include_labels=True)
                
                if self.include_behavior_patterns:
                    logger.info("⏳ Creating behavior pattern relationships...")
                    session.execute_write(self.create_behavior_pattern_relationships, df)

                if self.include_similarity:
                    logger.info("⏳ Creating similarity relationships...")
                    session.execute_write(self.create_similarity_relationships, df)
                else:
                    logger.info("Skipping similarity relationships")
            
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
    parser.add_argument('--exclude-similarity', action='store_true',
                        help='Exclude SIMILAR_TO edges from graph construction')
    parser.add_argument('--exclude-demographics', action='store_true',
                        help='Exclude demographic and submitter context from graph construction')
    parser.add_argument('--include-behavior-patterns', action='store_true',
                        help='Include behavior pattern and complexity nodes for ablation experiments')
    parser.add_argument('--csv-path', default=None,
                        help='Override the default CSV source used for graph construction')
    parser.add_argument('--weight-profile-json', default=None,
                        help='Optional JSON string overriding graph edge weights for experiments')
    args = parser.parse_args()

    weight_profile = None
    if args.weight_profile_json:
        weight_profile = json.loads(args.weight_profile_json)

    builder = GraphBuilder(
        include_similarity=not args.exclude_similarity,
        include_demographics=not args.exclude_demographics,
        include_behavior_patterns=args.include_behavior_patterns,
        csv_path=args.csv_path,
        weight_profile=weight_profile,
    )
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
