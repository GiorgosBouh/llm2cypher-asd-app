import numpy as np
import networkx as nx
from node2vec import Node2Vec
from neo4j import GraphDatabase
import sys
import os
import time
import tempfile
import logging
from typing import List, Optional, Tuple, Dict, Any
import shutil
import json
from dotenv import load_dotenv
import pandas as pd

# Set random seeds for deterministic results
np.random.seed(42)
import random
random.seed(42)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

class EmbeddingGenerator:
    def __init__(self):
        self.EMBEDDING_DIM = 128
        self.MIN_CONNECTIONS = 3  # Reduced for better coverage
        self.MIN_SIMILARITY = 2   # Reduced similarity threshold
        
        # Updated Node2Vec parameters for deterministic results
        self.NODE2VEC_WALK_LENGTH = 10
        self.NODE2VEC_NUM_WALKS = 50
        self.NODE2VEC_P = 0.5
        self.NODE2VEC_Q = 2.0
        self.NODE2VEC_WORKERS = 1  # Set to 1 for deterministic results
        self.NODE2VEC_WINDOW = 10
        self.RANDOM_SEED = 42

    def get_driver(self):
        """Initialize Neo4j driver using environment variables"""
        uri = os.getenv("NEO4J_URI")
        user = os.getenv("NEO4J_USER")
        password = os.getenv("NEO4J_PASSWORD")
        
        if not uri or not user or not password:
            raise ValueError("Missing Neo4j environment variables")
            
        return GraphDatabase.driver(
            uri,
            auth=(user, password),
            connection_timeout=900,
            max_connection_lifetime=7200,
            max_connection_pool_size=50
        )

    def validate_embedding(self, embedding: List[float]) -> bool:
        """Validate a generated embedding list"""
        if not embedding or len(embedding) != self.EMBEDDING_DIM:
            logger.warning(f"Invalid embedding length: {len(embedding) if embedding else 0} != {self.EMBEDDING_DIM}")
            return False
            
        if any(np.isnan(x) or not np.isfinite(x) for x in embedding):
            logger.warning("Embedding contains NaN or infinite values")
            return False
            
        if np.all(np.array(embedding) == 0):
            logger.warning("Embedding is all zeros")
            return False
            
        return True

    def insert_temporary_case(self, driver, upload_id: str, case_data: Dict[str, Any]) -> bool:
        """Insert temporary Case node and related nodes/relationships"""
        logger.info(f"Inserting temporary case with upload_id {upload_id}")
        
        with driver.session() as session:
            try:
                # Check if case already exists
                existing = session.run(
                    "MATCH (c:Case {upload_id: $upload_id}) RETURN c.id AS id",
                    upload_id=upload_id
                ).single()
                
                if existing:
                    logger.info(f"Case {upload_id} already exists, skipping insertion")
                    return True

                # Create the Case node
                case_no = int(case_data.get("Case_No", -1))
                session.run("""
                    CREATE (c:Case {upload_id: $upload_id, id: $case_no})
                """, upload_id=upload_id, case_no=case_no)
                
                # Connect to BehaviorQuestion nodes (HAS_ANSWER)
                for i in range(1, 11):
                    question_label = f"A{i}"
                    raw_val = case_data.get(question_label, "")
                    
                    try:
                        # Handle string values that might be comma-separated decimals
                        val_str = str(raw_val).strip().replace(",", ".")
                        val = float(val_str) if val_str else 0
                        if np.isfinite(val):
                            answer_value = int(val)
                        else:
                            logger.warning(f"Non-finite value for {question_label}: {raw_val}")
                            answer_value = 0
                    except (ValueError, TypeError) as e:
                        logger.warning(f"Failed to convert {question_label}: {raw_val} -> {e}")
                        answer_value = 0

                    session.run("""
                        MATCH (c:Case {upload_id: $upload_id}), (q:BehaviorQuestion {name: $question})
                        MERGE (c)-[r:HAS_ANSWER]->(q)
                        SET r.value = $value
                    """, upload_id=upload_id, question=question_label, value=answer_value)

                # Connect to DemographicAttribute nodes (HAS_DEMOGRAPHIC)
                demographic_fields = ["Sex", "Ethnicity", "Jaundice", "Family_mem_with_ASD"]
                for field in demographic_fields:
                    val = str(case_data.get(field, "")).strip()
                    if val and val.lower() not in ['nan', 'none', '']:
                        session.run("""
                            MATCH (c:Case {upload_id: $upload_id}), 
                                  (d:DemographicAttribute {type: $field, value: $val})
                            MERGE (c)-[:HAS_DEMOGRAPHIC]->(d)
                        """, upload_id=upload_id, field=field, val=val)

                # Connect to SubmitterType node (SUBMITTED_BY)
                submitter = str(case_data.get("Who_completed_the_test", "")).strip()
                if submitter and submitter.lower() not in ['nan', 'none', '']:
                    session.run("""
                        MATCH (c:Case {upload_id: $upload_id}), (s:SubmitterType {type: $submitter})
                        MERGE (c)-[:SUBMITTED_BY]->(s)
                    """, upload_id=upload_id, submitter=submitter)

                logger.info(f"Temporary case {upload_id} inserted successfully")
                return True

            except Exception as e:
                logger.error(f"Failed to insert temporary case {upload_id}: {str(e)}", exc_info=True)
                return False

    def delete_temporary_case(self, driver, upload_id: str) -> None:
        """Delete the temporary case node and all its relationships"""
        logger.info(f"Deleting temporary case with upload_id {upload_id}")
        with driver.session() as session:
            try:
                session.run("""
                    MATCH (c:Case {upload_id: $upload_id})
                    DETACH DELETE c
                """, upload_id=upload_id)
                logger.info(f"Temporary case {upload_id} deleted successfully")
            except Exception as e:
                logger.error(f"Failed to delete temporary case {upload_id}: {str(e)}")

    def build_base_graph(self, driver, upload_id: str) -> Optional[Tuple[nx.Graph, str]]:
        """Build base graph for the case with enhanced behavioral pattern recognition"""
        with driver.session() as session:
            try:
                # Get comprehensive case information with enhanced behavioral pattern capture
                result = session.run("""
                    MATCH (c:Case {upload_id: $upload_id})
                    OPTIONAL MATCH (c)-[r:HAS_ANSWER]->(q:BehaviorQuestion)
                    WHERE q IS NOT NULL AND r IS NOT NULL
                    OPTIONAL MATCH (c)-[d:HAS_DEMOGRAPHIC]->(dem:DemographicAttribute)
                    WHERE dem IS NOT NULL
                    OPTIONAL MATCH (c)-[s:SUBMITTED_BY]->(sub:SubmitterType)
                    WHERE sub IS NOT NULL
                    
                    RETURN c.id AS case_id_num,
                        collect(DISTINCT {
                            question: q.name, 
                            answer_value: r.value,
                            node_id: id(q)
                        }) AS behavior_answers,
                        collect(DISTINCT {
                            demo_type: dem.type,
                            demo_value: dem.value,
                            node_id: id(dem)
                        }) AS demographics,
                        collect(DISTINCT {
                            submitter_type: sub.type,
                            node_id: id(sub)
                        }) AS submitters
                """, upload_id=upload_id).single()

                if not result or result["case_id_num"] is None:
                    logger.error(f"Case {upload_id} not found in graph")
                    return None

                case_id_num = result['case_id_num']
                case_node_name = f"Case_{case_id_num}"
                G = nx.Graph()
                G.add_node(case_node_name, type="Case", id=case_id_num)

                # Enhanced behavioral pattern processing
                behavior_answers = result.get('behavior_answers', [])
                
                # Create behavioral complexity indicators
                behavioral_responses = 0
                q_chat_answers = {}
                
                for answer in behavior_answers:
                    question = answer.get('question', '')
                    value = answer.get('answer_value', 0)
                    
                    if question and question.startswith('A'):
                        q_chat_answers[question] = value
                        behavioral_responses += 1  # Count all responses
                        
                        # Create individual question nodes with enhanced weighting
                        q_node_name = f"Q_{question}"
                        G.add_node(q_node_name, type="BehaviorQuestion", question=question)
                        
                        # Enhanced edge weighting based on behavioral significance
                        if question in ['A1', 'A2', 'A3', 'A4', 'A6', 'A7', 'A8', 'A9']:  # Standard questions
                            edge_weight = 1.5  # Important behavioral indicators
                        elif question == 'A10':  # Different question type
                            edge_weight = 1.5  # Equally important
                        else:
                            edge_weight = 1.0
                            
                        G.add_edge(case_node_name, q_node_name, 
                                 type="HAS_ANSWER", 
                                 value=value, 
                                 weight=edge_weight,
                                 behavioral_response=value)

                # Add behavioral complexity patterns (not risk-based)
                complexity_score = len([a for a in behavior_answers if a.get('answer_value', 0) == 1])
                complexity_level = "HIGH" if complexity_score >= 5 else "MODERATE" if complexity_score >= 2 else "LOW"
                complexity_node_name = f"BehaviorComplexity_{complexity_level}"
                G.add_node(complexity_node_name, type="ComplexityPattern", level=complexity_level, score=complexity_score)
                G.add_edge(case_node_name, complexity_node_name, 
                         type="HAS_COMPLEXITY", 
                         weight=1.5,  # Neutral weight - not biased toward risk
                         complexity_score=complexity_score)

                # Add enhanced behavioral pattern combinations (neutral approach)
                if len(q_chat_answers) >= 10:  # Ensure we have all answers
                    # Communication patterns (A1, A2, A8) - neutral detection
                    comm_responses = [q_chat_answers.get(q, 0) for q in ['A1', 'A2', 'A8']]
                    if sum(comm_responses) >= 2:  # Any consistent pattern (high or low)
                        comm_node = f"Pattern_Communication_{sum(comm_responses)}"
                        G.add_node(comm_node, type="BehaviorPattern", pattern="communication", intensity=sum(comm_responses))
                        G.add_edge(case_node_name, comm_node, type="HAS_PATTERN", weight=1.8)
                    
                    # Social interaction patterns (A3, A4, A6, A7) - neutral detection
                    social_responses = [q_chat_answers.get(q, 0) for q in ['A3', 'A4', 'A6', 'A7']]
                    if sum(social_responses) >= 2:  # Any consistent pattern
                        social_node = f"Pattern_Social_{sum(social_responses)}"
                        G.add_node(social_node, type="BehaviorPattern", pattern="social", intensity=sum(social_responses))
                        G.add_edge(case_node_name, social_node, type="HAS_PATTERN", weight=1.8)
                    
                    # Activity/behavioral patterns (A5, A9, A10) - neutral detection
                    activity_responses = [q_chat_answers.get(q, 0) for q in ['A5', 'A9', 'A10']]
                    if sum(activity_responses) >= 2:  # Any consistent pattern
                        activity_node = f"Pattern_Activity_{sum(activity_responses)}"
                        G.add_node(activity_node, type="BehaviorPattern", pattern="activity", intensity=sum(activity_responses))
                        G.add_edge(case_node_name, activity_node, type="HAS_PATTERN", weight=1.8)

                # Add demographic connections with enhanced weights
                demographics = result.get('demographics', [])
                for demo in demographics:
                    demo_type = demo.get('demo_type', '')
                    demo_value = demo.get('demo_value', '')
                    
                    if demo_type and demo_value:
                        demo_node_name = f"D_{demo_type}_{demo_value.replace(' ', '_')}"
                        G.add_node(demo_node_name, type="DemographicAttribute", 
                                 demo_type=demo_type, demo_value=demo_value)
                        
                        # Enhanced weights for demographic factors
                        demo_weight = 1.0
                        if demo_type == "Family_mem_with_ASD" and demo_value.lower() == "yes":
                            demo_weight = 2.0  # Family history is important
                        elif demo_type == "Jaundice" and demo_value.lower() == "yes":
                            demo_weight = 1.5  # Jaundice is a mild factor
                        elif demo_type == "Sex" and demo_value.lower() in ["m", "male"]:
                            demo_weight = 1.3  # Male gender is a factor
                            
                        G.add_edge(case_node_name, demo_node_name, 
                                 type="HAS_DEMOGRAPHIC", weight=demo_weight)

                # Add submitter connections
                submitters = result.get('submitters', [])
                for submitter in submitters:
                    sub_type = submitter.get('submitter_type', '')
                    if sub_type:
                        sub_node_name = f"S_{sub_type.replace(' ', '_')}"
                        G.add_node(sub_node_name, type="SubmitterType", submitter_type=sub_type)
                        G.add_edge(case_node_name, sub_node_name, type="SUBMITTED_BY", weight=1.0)

                # Ensure minimum connectivity for embedding generation
                if len(G.edges(case_node_name)) == 0:
                    logger.warning("Case has no connections - adding default Q-Chat structure")
                    for i in range(1, 11):
                        q_node = f"Q_A{i}"
                        if q_node not in G:
                            G.add_node(q_node, type="BehaviorQuestion")
                        G.add_edge(case_node_name, q_node, type="HAS_ANSWER", value=0, weight=0.1)

                logger.info(f"Built enhanced behavioral graph with {len(G.nodes)} nodes and {len(G.edges)} edges")
                logger.info(f"Behavioral responses captured: {behavioral_responses}/10")
                return G, case_node_name

            except Exception as e:
                logger.error(f"Error building base graph: {str(e)}", exc_info=True)
                return None

    def augment_with_similarity(self, driver, G: nx.Graph, case_node_name: str) -> None:
        """Augment graph with enhanced behavioral pattern similarity"""
        try:
            original_case_id = int(case_node_name.split('_')[1])
            
            with driver.session() as session:
                # Find cases with similar behavioral patterns (neutral approach)
                similar_results = session.run("""
                    MATCH (c:Case {id: $case_id})-[r:HAS_ANSWER]->(q:BehaviorQuestion)
                    WITH c, collect({question: q.name, value: r.value}) as case_answers
                    
                    MATCH (other:Case)-[r2:HAS_ANSWER]->(q2:BehaviorQuestion)
                    WHERE c <> other AND other.embedding IS NOT NULL AND other.id IS NOT NULL
                    
                    WITH c, case_answers, other, 
                         collect({question: q2.name, value: r2.value}) as other_answers
                    
                    // Calculate behavioral pattern similarity (neutral approach)
                    WITH c, other, case_answers, other_answers,
                         // Count exact matches in behavioral responses
                         size([x IN case_answers WHERE x IN other_answers]) as exact_matches,
                         // Count total behavioral complexity
                         size([x IN case_answers WHERE x.value = 1]) as case_complexity,
                         size([y IN other_answers WHERE y.value = 1]) as other_complexity
                    
                    // Calculate similarity based on behavioral pattern overlap
                    WITH c, other, exact_matches, case_complexity, other_complexity,
                         // Calculate pattern similarity
                         CASE 
                             WHEN exact_matches >= 7 THEN 0.9  // Very similar patterns
                             WHEN exact_matches >= 5 THEN 0.7  // Moderately similar
                             WHEN abs(case_complexity - other_complexity) <= 1 THEN 0.6  // Similar complexity
                             ELSE toFloat(exact_matches) / 10.0  // Partial similarity
                         END AS similarity_score
                    
                    WHERE similarity_score >= 0.3  // Minimum similarity threshold
                    
                    RETURN other.id AS similar_id,
                           similarity_score,
                           exact_matches, case_complexity, other_complexity
                    ORDER BY similarity_score DESC
                    LIMIT 15
                """, case_id=original_case_id)

                similarity_count = 0
                for record in similar_results:
                    similar_id = record['similar_id']
                    similar_node_name = f"Case_{similar_id}"
                    weight = max(0.2, record['similarity_score'])

                    if similar_node_name not in G:
                        G.add_node(similar_node_name, type="Case", id=similar_id)
                    
                    if not G.has_edge(case_node_name, similar_node_name):
                        G.add_edge(case_node_name, similar_node_name, 
                                 type="SIMILAR_TO", 
                                 weight=weight,
                                 exact_matches=record['exact_matches'],
                                 pattern_similarity=record['similarity_score'])
                        similarity_count += 1

                logger.info(f"Added {similarity_count} behavioral pattern similarity connections")

        except Exception as e:
            logger.error(f"Error in behavioral similarity augmentation: {str(e)}")

    def generate_embedding(self, G: nx.Graph, case_node_name: str) -> Optional[List[float]]:
        """Generate embedding with enhanced behavioral pattern preservation"""
        temp_dir = None
        try:
            # Critical: Leakage check - ensure no SCREENED_FOR relationships
            screened_for_edges = [(u, v) for u, v, data in G.edges(data=True) 
                                if data.get('type') == 'SCREENED_FOR']
            if screened_for_edges:
                raise ValueError(f"❌ LABEL LEAKAGE: Found {len(screened_for_edges)} SCREENED_FOR relationships in graph")

            temp_dir = tempfile.mkdtemp()

            # Enhanced edge weighting for behaviorally important connections
            for u, v, data in G.edges(data=True):
                weight = data.get("weight", 1.0)
                edge_type = data.get("type", "")
                
                # Enhance weights based on behavioral significance, not risk
                if edge_type == "HAS_ANSWER":
                    # All behavioral responses are important for pattern recognition
                    data["weight"] = max(1.5, weight)  # Boost all behavioral connections
                elif edge_type == "HAS_COMPLEXITY":
                    data["weight"] = max(1.8, weight)  # Complexity patterns important
                elif edge_type == "HAS_PATTERN":
                    data["weight"] = max(1.8, weight)  # Behavioral patterns important
                elif edge_type == "SIMILAR_TO":
                    # Weight based on pattern similarity
                    pattern_sim = data.get("pattern_similarity", 0.5)
                    data["weight"] = max(0.5, min(2.5, pattern_sim * 2.0))
                elif edge_type == "HAS_DEMOGRAPHIC":
                    # Keep demographic weights as they are (already enhanced appropriately)
                    pass
                
                if not isinstance(data["weight"], (float, int)) or not np.isfinite(data["weight"]) or data["weight"] <= 0:
                    data["weight"] = 1.0

            # Check connectivity
            if case_node_name not in G:
                logger.error(f"Case node '{case_node_name}' not found in graph")
                return None

            degree = G.degree(case_node_name)
            if degree < self.MIN_CONNECTIONS:
                logger.warning(f"Node '{case_node_name}' has only {degree} connections")
                # Return a simple default embedding based on available connections
                if degree == 0:
                    return [0.1] * self.EMBEDDING_DIM
                
            # Set deterministic random seeds
            np.random.seed(self.RANDOM_SEED)
            random.seed(self.RANDOM_SEED)

            # Enhanced Node2Vec with behavioral pattern optimization
            try:
                node2vec = Node2Vec(
                    G,
                    dimensions=self.EMBEDDING_DIM,
                    walk_length=self.NODE2VEC_WALK_LENGTH,
                    num_walks=self.NODE2VEC_NUM_WALKS,
                    workers=self.NODE2VEC_WORKERS,
                    p=self.NODE2VEC_P,  # Lower p = more local exploration (behavioral patterns)
                    q=self.NODE2VEC_Q,  # Higher q = less return to previous nodes
                    temp_folder=temp_dir,
                    quiet=True
                )
            except Exception as e:
                logger.warning(f"Node2Vec initialization failed: {e}. Using conservative parameters.")
                node2vec = Node2Vec(
                    G,
                    dimensions=self.EMBEDDING_DIM,
                    walk_length=5,
                    num_walks=20,
                    workers=1,
                    p=1.0,
                    q=1.0,
                    temp_folder=temp_dir,
                    quiet=True
                )

            # Train model with deterministic settings
            model = node2vec.fit(
                window=self.NODE2VEC_WINDOW, 
                min_count=1,
                sg=1,  # Skip-gram for better pattern capture
                epochs=20,  # More epochs for better pattern learning
                seed=self.RANDOM_SEED,
                workers=1  # Deterministic training
            )

            if case_node_name not in model.wv:
                logger.error(f"No embedding generated for '{case_node_name}'")
                return None

            embedding = model.wv[case_node_name].tolist()

            # Validate embedding
            if not all(np.isfinite(embedding)):
                logger.error(f"Embedding contains non-finite values")
                return None

            norm = np.linalg.norm(embedding)
            if norm == 0:
                logger.warning("Zero-norm embedding - using raw vector")
                return embedding if self.validate_embedding(embedding) else None

            # Enhanced L2 normalization for better model training
            normalized_embedding = (np.array(embedding) / norm).tolist()
            
            # Additional validation
            if not self.validate_embedding(normalized_embedding):
                logger.error("Generated embedding failed validation")
                return None
                
            logger.info(f"Successfully generated behavioral pattern embedding (norm: {norm:.4f})")
            return normalized_embedding

        except Exception as e:
            logger.error(f"Enhanced embedding generation failed: {str(e)}", exc_info=True)
            return None

        finally:
            if temp_dir and os.path.exists(temp_dir):
                shutil.rmtree(temp_dir, ignore_errors=True)

    def generate_embedding_for_case(self, upload_id: str, case_data: Dict[str, Any]) -> Optional[List[float]]:
        """Main workflow for generating embedding for a new case"""
        driver = None
        try:
            driver = self.get_driver()

            # Step 1: Insert temporary case
            if not self.insert_temporary_case(driver, upload_id, case_data):
                return None

            # Step 2: Build base graph
            graph_result = self.build_base_graph(driver, upload_id)
            if not graph_result:
                return None

            G, case_node_name = graph_result

            # Step 3: Augment with similarity if needed
            if len(G.edges(case_node_name)) < self.MIN_CONNECTIONS:
                self.augment_with_similarity(driver, G, case_node_name)

            # Step 4: Generate embedding
            embedding = self.generate_embedding(G, case_node_name)

            # Step 5: Store embedding in the temporary node
            if embedding and self.validate_embedding(embedding):
                with driver.session() as session:
                    session.run("""
                        MATCH (c:Case {upload_id: $upload_id})
                        SET c.embedding = $embedding,
                            c.embedding_version = 2.1,
                            c.last_embedding_update = timestamp()
                    """, upload_id=upload_id, embedding=embedding)
                
                logger.info(f"Successfully generated and stored embedding for {upload_id}")
                return embedding
            else:
                logger.error("Failed to generate valid embedding")
                return None

        except Exception as e:
            logger.critical(f"Fatal error in embedding generation workflow: {str(e)}", exc_info=True)
            return None
        finally:
            if driver:
                driver.close()


if __name__ == "__main__":
    try:
        if len(sys.argv) < 3:
            raise ValueError("Usage: generate_case_embedding.py <upload_id> <case_data_json>")
        
        upload_id = sys.argv[1]
        case_data_json = sys.argv[2]
        
        try:
            case_data = json.loads(case_data_json)
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON data: {e}")
            sys.exit(1)

        logger.info(f"Starting embedding generation for upload_id: {upload_id}")
        generator = EmbeddingGenerator()
        embedding = generator.generate_embedding_for_case(upload_id, case_data)

        if embedding is not None and generator.validate_embedding(embedding):
            logger.info("Embedding generated successfully")
            print(json.dumps(embedding))
            sys.exit(0)
        else:
            logger.error("Embedding generation failed")
            sys.exit(1)
        
    except Exception as e:
        logger.critical(f"Fatal error in main: {str(e)}", exc_info=True)
        sys.exit(1)