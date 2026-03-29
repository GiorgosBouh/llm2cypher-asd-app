The application has now moved to: http://195.251.143.63:8501/.


NeuroCypher ASD — Knowledge Graph + ML + LLM for ASD Screening

NeuroCypher ASD is a graph-augmented machine learning framework that combines a symbolic graph representation in Neo4j, Node2Vec graph embeddings, conventional machine-learning models (XGBoost + Isolation Forest), and an LLM-powered natural-language query interface for Autism Spectrum Disorder (ASD) screening analysis in toddlers. In its current form, the repository implements a pipeline in which graph structure is used to produce learned features for downstream prediction and exploration. It should be understood as graph-enhanced or weakly neurosymbolic integration rather than a fully bidirectional neurosymbolic reasoning system.

Paper/technical report: NeuroCypher ASD: A Knowledge Graph and Machine Learning Framework for ASD Screening in Toddlers. Please cite if you use this repo.

⸻

Key Features
	•	Graph-native data model (Neo4j) for Q-Chat-10, demographics, and ASD traits; weighted relations are heuristic and can be studied with the included sensitivity experiments.
	•	Node2Vec embeddings as ML features to capture latent relational structure.
	•	Supervised classification (XGBoost) and unsupervised anomaly detection (Isolation Forest) with leakage-safe training protocol.
	•	Natural-language to Cypher interface (GPT-4 or compatible) to query the graph without writing Cypher.
	•	Streamlit UI for training, case uploads/predictions, and NL graph querying.

⸻

Results (reference)

On the public Autistic Spectrum Disorder Screening Data for Toddlers dataset (1,054 samples; 18 attributes), the framework achieved:

ROC AUC ≈ 0.678
F1 ≈ 0.765

under a stratified 70/30 split with class imbalance handled through class weighting and XGBoost `scale_pos_weight`.

⸻

Architecture
	1.	Ingest & Validate → clean and normalize CSV, parse numerics, impute missing values.
	2.	KG Construction → create nodes for cases, Q-Chat items, demographics, ASD traits; connect with weighted edges.
	3.	Embeddings → Node2Vec over the KG (label edges removed during embedding to prevent leakage).
	4.	ML → XGBoost classifier (supervised) and Isolation Forest (anomaly detection).
	5.	Interpretation & NL Querying → SHAP/correlation-based interpretability artifacts and an LLM interface that translates natural language into Cypher; Streamlit UI for interaction.

⸻

Getting Started

Prerequisites:
	•	Python 3.9+
	•	Neo4j (Aura or local)
	•	LLM API key (e.g., OPENAI_API_KEY)
	•	(Optional) Streamlit for the UI

Installation:
Clone the repo, create a virtual environment, and install dependencies with pip install -r requirements.txt.

Environment:
Create a .env file in the project root with NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD, and OPENAI_API_KEY.

Data:
Download the Autism Screening for Toddlers dataset from Kaggle and place it under data/. Ensure the CSV headers match the Q-Chat-10 mapping (A1–A10) and demographics.

⸻

Usage
	1.	Build the Knowledge Graph with the build_kg.py script.
	2.	Generate Node2Vec embeddings with the generate_embeddings.py script.
	3.	Train and evaluate models with the train.py script.
	4.	Run the Streamlit app with streamlit run app.py.

Upload CSV cases for inference, see anomaly scores, and run natural language queries.

⸻

Natural Language to Cypher (Example)

NL query: How many male toddlers in the dataset were screened positive for ASD traits?

Generated Cypher (conceptual): Match male cases and count how many are linked to ASD traits.

⸻

Reproducibility and Anti-Leakage Checklist
	•	Excluded total Q-Chat score (deterministic label mapping).
	•	Removed :SCREENED_FOR edges before embeddings and restored after.
	•	Handled class imbalance through class weighting and XGBoost `scale_pos_weight`.
	•	Fixed random seeds for reproducibility.

Terminology Note
	•	This repository contains symbolic graph modeling, graph-derived learned features, conventional ML, and an LLM query interface.
	•	It does not currently implement explicit symbolic reasoning over model predictions, iterative feedback from the predictor back into the knowledge graph, or a bidirectional neuro-symbolic training loop.
	•	Claims such as “context-aware” or “neurosymbolic” should therefore be interpreted conservatively and tied to graph-augmented feature construction and query support, not to full neuro-symbolic reasoning.

⸻

Roadmap
	•	Add GNN embeddings (GCN, GAT, GraphSAGE).
	•	Expand multimodal features and learner comparisons.
	•	Extend NL-to-Cypher templates and safety checks.
	•	For stronger neurosymbolic integration, future work would need explicit symbolic constraints or rules that influence prediction, tighter coupling between symbolic inference and learned models, and evaluation showing that this coupling changes outcomes beyond standard graph-embedding pipelines.

⸻

Citation

If you use this code or ideas, please cite:

Bouchouras, Georgios; Doumanas, Dimitrios; Kotis, Konstantinos (2025). NeuroCypher ASD: A Knowledge Graph and Machine Learning Framework for ASD Screening in Toddlers. Preprint/Technical Report.

⸻

License

Specify your license here (e.g., MIT). If the dataset license differs, follow its terms.

⸻

Disclaimer

This software is for research and educational purposes only and is not a medical device. Predictions are screening aids and must not be used as standalone diagnoses.
