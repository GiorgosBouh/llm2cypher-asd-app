The application has now moved to: http://195.251.143.85:8501.

NeuroCypher ASD — Knowledge Graph + ML + LLM for ASD Screening

NeuroCypher ASD is a neurosymbolic AI framework that fuses Knowledge Graphs (Neo4j), graph embeddings (Node2Vec), Machine Learning (XGBoost + Isolation Forest), and an LLM-powered natural-language interface to support explainable screening analysis for Autism Spectrum Disorder (ASD) in toddlers. It models Q-Chat-10 responses, demographics, and outcomes as a graph, enabling transparent queries, predictive modeling, and anomaly detection through a Streamlit web UI.

Paper/technical report: NeuroCypher ASD: A Knowledge Graph and Machine Learning Framework for ASD Screening in Toddlers. Please cite if you use this repo.

⸻

Key Features
	•	Graph-native data model (Neo4j) for Q-Chat-10, demographics, and ASD traits; weighted relations for clinical salience.
	•	Node2Vec embeddings as ML features to capture latent relational structure.
	•	Supervised classification (XGBoost) and unsupervised anomaly detection (Isolation Forest) with leakage-safe training protocol.
	•	Natural-language to Cypher interface (GPT-4 or compatible) to query the graph without writing Cypher.
	•	Streamlit UI for training, case uploads/predictions, and NL graph querying.

⸻

Results (reference)

On the public Autistic Spectrum Disorder Screening Data for Toddlers dataset (1,054 samples; 18 attributes), the framework achieved:

ROC AUC ≈ 0.678
F1 ≈ 0.765

under a stratified 70/30 split with SMOTE applied inside CV folds.

⸻

Architecture
	1.	Ingest & Validate → clean and normalize CSV, parse numerics, impute missing values.
	2.	KG Construction → create nodes for cases, Q-Chat items, demographics, ASD traits; connect with weighted edges.
	3.	Embeddings → Node2Vec over the KG (label edges removed during embedding to prevent leakage).
	4.	ML → XGBoost classifier (supervised) and Isolation Forest (anomaly detection).
	5.	Explainability & NL Querying → LLM translates natural language into Cypher; Streamlit UI for interaction.

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
	•	Applied SMOTE only inside CV training folds.
	•	Fixed random seeds for reproducibility.

⸻

Roadmap
	•	Add GNN embeddings (GCN, GAT, GraphSAGE).
	•	Expand multimodal features and learner comparisons.
	•	Extend NL-to-Cypher templates and safety checks.

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