# NeuroCypher ASD: Repository for Paper, App, and Reproducibility Artifacts

This repository snapshot was prepared as a publication-facing version of the NeuroCypher ASD project. Its purpose is to make the code, packaged experiment outputs, manuscript-facing tables/figures, and the deployed application code available in one place so that readers can inspect what was implemented and how the reported results were produced.

The repository supports three complementary uses:

1. Reproducing the main experiment scripts used for the revised paper.
2. Inspecting the Streamlit application and its graph-assisted screening workflow.
3. Tracing manuscript tables and figures back to the scripts and result artifacts that generated them.

This is a graph-enhanced decision-support repository for ASD screening in toddlers. It should be interpreted conservatively: the tabular model provides the primary prediction, while the knowledge graph, anomaly support, and natural-language interface provide contextualization, exploration, and interpretability support.

## What Is Included

This curated `repo_revised` snapshot contains:

- The current Streamlit application code under `app/` plus the root `main_app.py` entrypoint.
- The graph construction, embedding, experiment, safety, and packaging scripts used for the paper.
- Packaged revision artifacts under `revision_results/` for manuscript-ready tables and figures.
- The specific experiment run folders referenced by the packaged revision outputs under `results/`.
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

## Paper-to-Code Map

This section maps the main methodological components of the paper to the code and artifacts in this repository.

### 1. Dataset Description and Preprocessing

Primary files:

- `Toddler_Autism_dataset_July_2018_2.csv`
- `Toddler_Autism_dataset_July_2018_2_no_qchat_score.csv`
- `Toddler_data_description.pdf`
- `experiments_baselines.py`
- `app/data/schema_validation.py`
- `app/data/preprocessing.py`

What these files do:

- `experiments_baselines.py` loads, normalizes, splits, and evaluates the dataset under the revised leakage-safer protocol.
- `app/data/schema_validation.py` and `app/data/preprocessing.py` implement the schema validation and app-side preprocessing used for single-case inference.
- The dataset description files document the raw screening variables and their meanings.

Important methodological note:

- `Qchat-10-Score` is excluded from predictive modeling because it is deterministically derived from the questionnaire responses and would introduce leakage.

### 2. Knowledge Graph Construction and Embeddings

Primary files:

- `kg_builder_2.py`
- `generate_case_embedding.py`
- `evaluation_safety.py`
- `app/services/graph_context_service.py`
- `app/services/neo4j_service.py`

What these files do:

- `kg_builder_2.py` builds the Neo4j graph and generates Node2Vec-based case embeddings.
- `generate_case_embedding.py` supports single-case embedding generation for uploaded cases.
- `evaluation_safety.py` contains helpers for stratified split-first evaluation, safety auditing, and inductive graph embedding refresh.
- `graph_context_service.py` implements the application-facing graph contextualization layer used after prediction.

### 3. Reference Tabular, Graph-Only, and Hybrid Baselines

Primary files:

- `experiments_baselines.py`
- `results/baseline_run_20260324_143442/`
- `revision_results/table_baselines.csv`
- `revision_results/table_baselines.md`
- `revision_results/table_baselines.tex`

What these files do:

- `experiments_baselines.py` evaluates raw tabular, graph-only, and hybrid models under a common split and CV protocol.
- The included baseline results folder contains the referenced experiment outputs.
- The packaged `revision_results/table_baselines.*` files are manuscript-facing exports derived from those runs.

### 4. Graph Ablation Analysis

Primary files:

- `experiments_ablation.py`
- `results/ablation_run_20260324_132809/`
- `revision_results/table_ablations.csv`
- `revision_results/table_ablations.md`
- `revision_results/table_ablations.tex`

What these files do:

- `experiments_ablation.py` evaluates graph variants such as full graph, simplified graph, no similarity edges, no demographic context, and anomaly-gated variants.
- The packaged ablation tables in `revision_results/` are the stable manuscript-facing outputs.

### 5. Weight Sensitivity Analysis

Primary files:

- `experiments_weight_sensitivity.py`
- `results/weight_sensitivity_run_20260324_134356/`
- `revision_results/table_weight_sensitivity.csv`
- `revision_results/table_weight_sensitivity.md`
- `revision_results/table_weight_sensitivity.tex`

What these files do:

- `experiments_weight_sensitivity.py` evaluates whether heuristic vs uniform vs perturbed graph weights materially change graph-only performance.
- The results folder contains the referenced run outputs.
- The packaged revision tables are the manuscript-ready versions.

### 6. Interpretability and SHAP Analysis

Primary files:

- `experiments_interpretability.py`
- `app/services/shap_service.py`
- `results/interpretability_run_20260324_140612/`
- `revision_results/table_global_shap_importance.csv`
- `revision_results/table_embedding_feature_mapping.csv`
- `revision_results/table_case_level_explanations.csv`
- `revision_results/table_kg_explanation_example.csv`
- `revision_results/figure_shap_global_bar.png`
- `revision_results/figure_shap_beeswarm.png`

What these files do:

- `experiments_interpretability.py` runs the graph-model interpretability analyses and exports SHAP-related artifacts.
- `app/services/shap_service.py` is the app-side explanation layer for local and global explainability.
- The packaged revision outputs provide manuscript-facing SHAP tables and figures.

### 7. Anomaly Detection

Primary files:

- `app/services/anomaly_service.py`
- `app/ui/anomaly_view.py`
- `artifacts/anomaly_model_bundle.pkl`

What these files do:

- `anomaly_service.py` trains and applies the Isolation Forest-based anomaly support layer.
- `anomaly_view.py` renders the anomaly analysis in the Streamlit application.

### 8. Natural-Language Query Interface

Primary files:

- `app/services/nl_to_cypher_service.py`
- `app/ui/nl_query_view.py`
- `app/services/neo4j_service.py`

What these files do:

- `nl_to_cypher_service.py` translates user questions into read-only Cypher, using rule-based intent handling plus LLM-backed generation.
- `nl_query_view.py` renders the generated query and result rows.
- `neo4j_service.py` is the thin Neo4j access layer used by the app and graph services.

### 9. The Streamlit Application

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

## Packaged Revision Artifacts

The `revision_results/` folder is the most important manuscript-facing directory for reviewers or readers who want to inspect the exact packaged outputs used for tables and figures.

Key files:

- `revision_results/INDEX.md`: artifact index and provenance summary.
- `revision_results/artifact_manifest.json`: machine-readable artifact manifest.
- `revision_results/manuscript_methods_results_revision.tex`: packaged methods/results text asset present in the working project.
- `revision_results/table_baselines.tex`
- `revision_results/table_ablations.tex`
- `revision_results/table_weight_sensitivity.tex`
- `revision_results/table_local_explanation_example.tex`
- `revision_results/table_shap_top_dimensions.tex`
- `revision_results/figure_shap_global_bar.png`
- `revision_results/figure_shap_beeswarm.png`

If you regenerate experiments, rerun:

```bash
python3 prepare_revision_results.py
```

to refresh the packaged manuscript-facing outputs.

## Manuscript Assets

The repository includes manuscript-related files under:

```text
manuscript/springer_revision_assets/
```

This folder preserves the manuscript assets that existed in the working project at the time this snapshot was created, including:

- `sn-article.tex`
- `sn-bibliography.bib`
- `sn-jnl.cls`
- `sn-basic.bst`
- packaged table `.tex` files
- manuscript figures and screenshots
- `user-manual.pdf`

Important note:

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

## Data Integrity and Leakage Safeguards

The revised project is organized around leakage-aware evaluation decisions. These are important enough to restate clearly here:

- `Qchat-10-Score` is not used as a predictive feature.
- Label-related graph leakage is reduced by split-first evaluation and label-free embedding refresh for held-out cases.
- Train/test and CV procedures are deterministic and stratified.
- The tabular model should be interpreted as a strong reference baseline, not as evidence of independent diagnostic reasoning.
- The graph layer is better understood as a contextual and exploratory support layer than as a superior standalone classifier on this dataset.

## What Readers Can Verify from This Repository

By inspecting this snapshot, a reviewer or reader can verify:

- Which scripts were used for the baseline, ablation, interpretability, and weight-sensitivity experiments.
- Which packaged tables and figures were produced for the revised manuscript.
- How the app is structured and how prediction, graph context, anomaly support, and natural-language querying are implemented.
- How the project encodes its leakage-prevention decisions.
- Which result folders correspond to the packaged revision artifacts.

## Suggested GitHub Positioning

If you want to cite this repository in the paper, a conservative description would be:

> The repository includes the Streamlit application, graph construction scripts, experiment scripts for the revised evaluation, packaged manuscript-facing result artifacts, and manuscript asset files needed to inspect the reported methodology and results.

That wording is code-aligned and does not overclaim full clinical reproducibility beyond the included data, environment, and external service dependencies.

## Practical Notes Before You Push

- Do not commit `.env` or live credentials.
- Keep `revision_results/` because it is the easiest manuscript-facing entry point for reviewers.
- Keep the specific `results/` run folders already included here, because they provide direct provenance for the packaged outputs.
- Verify which manuscript `.tex` file is your final submission draft before pushing.

## Citation

If you use or refer to this repository, cite the corresponding NeuroCypher ASD paper and describe this repository as the code and artifact companion to the revised manuscript.
