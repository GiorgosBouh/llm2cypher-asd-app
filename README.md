# NeuroCypher ASD

This repository contains the NeuroCypher ASD application, the experiment scripts used in the study, the packaged result artifacts, and the manuscript assets available in the project workspace.

NeuroCypher ASD is a graph-enhanced decision-support project for ASD screening in toddlers. The tabular model provides the primary prediction, while the knowledge graph, anomaly support, and natural-language interface provide context, exploration, and interpretability support.

## What Is Included

Included in the repository:

- The current Streamlit application code under `app/` plus the root `main_app.py` entrypoint.
- The graph construction, embedding, experiment, safety, and packaging scripts used for the paper.
- Packaged manuscript artifacts under `revision_results/`.
- The specific experiment run folders referenced by those packaged outputs under `results/`.
- Manuscript assets under `manuscript/springer_revision_assets/`, including Springer files, tables, figures, and bibliography files present in the working project folder.
- The toddler ASD dataset files used in this project workspace.
- Model bundle artifacts used by the application for tabular prediction and anomaly support.

This snapshot intentionally excludes local caches, `__pycache__`, and `.env` credentials.

## Repository Structure

```text
repo_revised/
├── app/
│   ├── data/
│   ├── models/
│   ├── services/
│   ├── ui/
│   └── utils/
├── artifacts/
├── manuscript/
│   └── springer_revision_assets/
├── results/
│   ├── baseline_run_20260324_143442/
│   ├── ablation_run_20260324_132809/
│   ├── weight_sensitivity_run_20260324_134356/
│   └── interpretability_run_20260324_140612/
├── revision_results/
├── main_app.py
├── kg_builder_2.py
├── generate_case_embedding.py
├── experiments_baselines.py
├── experiments_ablation.py
├── experiments_interpretability.py
├── experiments_weight_sensitivity.py
├── prepare_revision_results.py
├── evaluation_safety.py
├── env_utils.py
├── requirements.txt
└── README.md
```

## Paper Map

The sections below link the main parts of the paper to the corresponding repository files.

### Section: Dataset Description and Preprocessing

Relevant files:

- `Toddler_Autism_dataset_July_2018_2.csv`
- `Toddler_Autism_dataset_July_2018_2_no_qchat_score.csv`
- `Toddler_data_description.pdf`
- `experiments_baselines.py`
- `app/data/schema_validation.py`
- `app/data/preprocessing.py`

Notes:

- `experiments_baselines.py` loads, normalizes, splits, and evaluates the dataset under the revised leakage-safer protocol.
- `app/data/schema_validation.py` and `app/data/preprocessing.py` implement the schema validation and app-side preprocessing used for single-case inference.
- The dataset description files document the raw screening variables and their meanings.

Note:

- `Qchat-10-Score` is excluded from predictive modeling because it is deterministically derived from the questionnaire responses and would introduce leakage.

### Section: Knowledge Graph Construction and Embedding Generation

Relevant files:

- `kg_builder_2.py`
- `generate_case_embedding.py`
- `evaluation_safety.py`
- `app/services/graph_context_service.py`
- `app/services/neo4j_service.py`

Notes:

- `kg_builder_2.py` builds the Neo4j graph and generates Node2Vec-based case embeddings.
- `generate_case_embedding.py` supports single-case embedding generation for uploaded cases.
- `evaluation_safety.py` contains helpers for stratified split-first evaluation, safety auditing, and inductive graph embedding refresh.
- `graph_context_service.py` implements the application-facing graph contextualization layer used after prediction.

### Section: Graph Variants and Edge Weight Design

Relevant files:

- `experiments_ablation.py`
- `experiments_weight_sensitivity.py`
- `kg_builder_2.py`
- `revision_results/table_ablations.tex`
- `revision_results/table_weight_sensitivity.tex`

Notes:

- `experiments_ablation.py` evaluates graph structure variants such as full graph, simplified graph, no similarity edges, no demographic context, and anomaly-gated settings.
- `experiments_weight_sensitivity.py` evaluates heuristic, uniform, and perturbed weighting schemes.
- `kg_builder_2.py` is the builder that encodes the graph schema and optional graph construction flags used by these analyses.

### Section: Predictive Evaluation Design

Relevant files:

- `experiments_baselines.py`
- `evaluation_safety.py`
- `results/baseline_run_20260324_143442/`
- `revision_results/table_baselines.tex`

Notes:

- `experiments_baselines.py` implements the split-first, deterministic, stratified predictive comparison for raw tabular, graph-only, and hybrid settings.
- `evaluation_safety.py` contains split and safety utilities used to reduce leakage and keep train/test handling explicit.
- The `results/baseline_run_20260324_143442/` folder contains the exact exported outputs for the retained packaged baseline artifacts.

### Section: Interpretability and Representation Analysis

Relevant files:

- `experiments_interpretability.py`
- `revision_results/table_global_shap_importance.csv`
- `revision_results/table_embedding_feature_mapping.csv`
- `revision_results/table_case_level_explanations.csv`
- `revision_results/table_kg_explanation_example.csv`
- `revision_results/figure_shap_global_bar.png`
- `revision_results/figure_shap_beeswarm.png`

Notes:

- `experiments_interpretability.py` exports the SHAP-based global and case-level analyses and the approximate embedding-to-feature bridge.
- The revision artifacts provide the manuscript-facing interpretability tables and figures.

Related application file:

- `app/services/shap_service.py` provides local/global explainability for the deployed tabular-first Streamlit application, but it is separate from the paper-facing graph-embedding interpretability script.

### Section: Anomaly Detection and Decision-Support Layer

Relevant files:

- `app/services/anomaly_service.py`
- `app/ui/anomaly_view.py`
- `artifacts/anomaly_model_bundle.pkl`
- `experiments_ablation.py`

Notes:

- `anomaly_service.py` trains and applies the Isolation Forest support layer.
- `anomaly_view.py` renders the anomaly interpretation in the Streamlit app.
- `experiments_ablation.py` also includes the anomaly-gated experimental setting used in the paper.

### Section: Natural-Language Querying Interface

Relevant files:

- `app/services/nl_to_cypher_service.py`
- `app/ui/nl_query_view.py`
- `app/services/neo4j_service.py`

Notes:

- `nl_to_cypher_service.py` implements the natural-language-to-Cypher layer, combining rule-based routing and LLM-backed query generation with read-only safety constraints.
- `nl_query_view.py` renders the generated query and the returned rows.
- `neo4j_service.py` provides the database access wrapper used by the graph-related application services.

### Section: Data Integrity and Clinical Interpretation

Relevant files:

- `evaluation_safety.py`
- `experiments_baselines.py`
- `experiments_ablation.py`
- `README.md`
- `revision_results/INDEX.md`

Notes:

- `evaluation_safety.py` captures the safety-oriented evaluation logic used in the project.
- The baseline and ablation scripts operationalize the leakage-safer split-first methodology described in the paper.
- This repository README and the packaged revision index make the provenance of results explicit and conservative.

### Results Section: Reference Tabular, Graph-Only, and Hybrid Baselines

Relevant files:

- `experiments_baselines.py`
- `results/baseline_run_20260324_143442/`
- `revision_results/table_baselines.csv`
- `revision_results/table_baselines.md`
- `revision_results/table_baselines.tex`

Notes:

- `experiments_baselines.py` evaluates raw tabular, graph-only, and hybrid models under a common split and CV protocol.
- The included baseline results folder contains the referenced experiment outputs.
- The packaged `revision_results/table_baselines.*` files are manuscript-facing exports derived from those runs.

### Results Section: Graph Ablation Analysis

Relevant files:

- `experiments_ablation.py`
- `results/ablation_run_20260324_132809/`
- `revision_results/table_ablations.csv`
- `revision_results/table_ablations.md`
- `revision_results/table_ablations.tex`

Notes:

- `experiments_ablation.py` evaluates graph variants such as full graph, simplified graph, no similarity edges, no demographic context, and anomaly-gated variants.
- The packaged ablation tables in `revision_results/` are the stable manuscript-facing outputs.

### Results Section: Weight Sensitivity

Relevant files:

- `experiments_weight_sensitivity.py`
- `results/weight_sensitivity_run_20260324_134356/`
- `revision_results/table_weight_sensitivity.csv`
- `revision_results/table_weight_sensitivity.md`
- `revision_results/table_weight_sensitivity.tex`

Notes:

- `experiments_weight_sensitivity.py` evaluates whether heuristic vs uniform vs perturbed graph weights materially change graph-only performance.
- The results folder contains the referenced run outputs.
- The packaged revision tables are the manuscript-ready versions.

### Results Section: Interpretability, SHAP Analysis, and Clinical Utility

Relevant files:

- `experiments_interpretability.py`
- `app/services/shap_service.py`
- `results/interpretability_run_20260324_140612/`
- `revision_results/table_global_shap_importance.csv`
- `revision_results/table_embedding_feature_mapping.csv`
- `revision_results/table_case_level_explanations.csv`
- `revision_results/table_kg_explanation_example.csv`
- `revision_results/figure_shap_global_bar.png`
- `revision_results/figure_shap_beeswarm.png`

Notes:

- `experiments_interpretability.py` runs the graph-model interpretability analyses and exports SHAP-related artifacts.
- `app/services/shap_service.py` is the app-side explanation layer for local and global explainability.
- The packaged revision outputs provide manuscript-facing SHAP tables and figures.

### Results Section: Anomaly Detection and Decision Support

Primary files:

- `experiments_ablation.py`
- `app/services/anomaly_service.py`
- `app/ui/anomaly_view.py`
- `artifacts/anomaly_model_bundle.pkl`

What these files do:

- `experiments_ablation.py` contains the anomaly-gated evaluation variant referenced in the revised paper.
- `anomaly_service.py` trains and applies the Isolation Forest-based anomaly support layer.
- `anomaly_view.py` renders the anomaly analysis in the Streamlit application.

### Application

Primary files:

- `main_app.py`
- `app/main_app.py`
- `app/ui/`
- `app/services/`
- `artifacts/tabular_model_bundle.pkl`
- `artifacts/anomaly_model_bundle.pkl`

What the app does:

- Accepts single-case CSV upload or manual questionnaire entry.
- Runs tabular-first prediction.
- Builds graph context around the case.
- Provides explainability, anomaly support, natural-language querying, and PDF report export.

## Packaged Artifacts

The `revision_results/` folder contains the packaged outputs used for tables and figures.

Key files:

- `revision_results/INDEX.md`: artifact index and provenance summary.
- `revision_results/artifact_manifest.json`: machine-readable artifact manifest.
- `revision_results/manuscript_methods_results_revision.tex`: packaged methods/results text asset present in the project workspace.
- `revision_results/table_baselines.tex`
- `revision_results/table_ablations.tex`
- `revision_results/table_weight_sensitivity.tex`
- `revision_results/table_local_explanation_example.tex`
- `revision_results/table_shap_top_dimensions.tex`
- `revision_results/figure_shap_global_bar.png`
- `revision_results/figure_shap_beeswarm.png`

To refresh these packaged outputs after rerunning experiments:

```bash
python3 prepare_revision_results.py
```

to refresh the packaged manuscript-facing outputs.

## Manuscript Assets

The repository includes manuscript-related files under:

```text
manuscript/springer_revision_assets/
```

This folder preserves the manuscript assets that were present in the project workspace, including:

- `sn-article.tex`
- `sn-bibliography.bib`
- `sn-jnl.cls`
- `sn-basic.bst`
- packaged table `.tex` files
- manuscript figures and screenshots
- `user-manual.pdf`

Note:

- The current working repository contained more than one manuscript-related TeX file. This curated snapshot preserves the manuscript assets found in the folder, but you should verify that the exact submission draft you intend to upload to the journal is the one you want to keep as authoritative.
- If your newest paper text exists outside the project folder or was edited separately, replace or add that exact submission `.tex` file under `manuscript/` before pushing to GitHub.

## Reproducibility Workflow

### Environment setup

Create and activate a Python environment, then install dependencies:

```bash
pip install -r requirements.txt
```

You will also need:

- Neo4j credentials in environment variables.
- An OpenAI API key if you want to use the natural-language query interface.

Expected environment variables:

- `NEO4J_URI`
- `NEO4J_USER`
- `NEO4J_PASSWORD`
- `NEO4J_DB`
- `OPENAI_API_KEY`
- `OPENAI_MODEL` (optional)

### Run the app

```bash
python3 -m streamlit run main_app.py
```

### Rebuild the full graph

```bash
python3 kg_builder_2.py --build-full-graph
```

### Refresh label-free embeddings

```bash
python3 kg_builder_2.py --no-labels
```

### Run the revised baseline experiments

```bash
python3 experiments_baselines.py
```

### Run the graph ablation study

```bash
python3 experiments_ablation.py
```

### Run weight sensitivity analysis

```bash
python3 experiments_weight_sensitivity.py
```

### Run interpretability analysis

```bash
python3 experiments_interpretability.py
```

### Package manuscript-facing outputs

```bash
python3 prepare_revision_results.py
```

## Data Integrity Notes

The repository follows leakage-aware evaluation decisions:

- `Qchat-10-Score` is not used as a predictive feature.
- Label-related graph leakage is reduced by split-first evaluation and label-free embedding refresh for held-out cases.
- Train/test and CV procedures are deterministic and stratified.
- The tabular model is treated as a strong reference baseline, not as evidence of independent diagnostic reasoning.
- The graph layer is better understood as a contextual and exploratory support layer than as a superior standalone classifier on this dataset.

## Repository Contents at a Glance

The repository makes it possible to inspect:

- Which scripts were used for the baseline, ablation, interpretability, and weight-sensitivity experiments.
- Which packaged tables and figures were produced for the manuscript.
- How the app is structured and how prediction, graph context, anomaly support, and natural-language querying are implemented.
- How the project encodes its leakage-prevention decisions.
- Which result folders correspond to the packaged manuscript artifacts.

## Repository Scope

This repository contains:

- The Streamlit application
- Graph construction and embedding scripts
- Experiment scripts for baselines, ablations, interpretability, and weight sensitivity
- Packaged manuscript-facing tables and figures
- Manuscript assets present in the project workspace

The repository is intended as a code and artifact companion to the NeuroCypher ASD paper. It does not claim full clinical reproducibility beyond the included data, environment configuration, and external service dependencies.

## Citation

If this repository is used in academic work, cite the corresponding NeuroCypher ASD paper.
