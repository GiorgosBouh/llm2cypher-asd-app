# AGENTS.md

## Repository Map
- `main_app.py`: Streamlit app, model training UI, upload-time prediction flow, anomaly detection, NL-to-Cypher.
- `kg_builder_2.py`: full Neo4j graph build and batch Node2Vec embedding generation.
- `generate_case_embedding.py`: single-case embedding generation for uploaded cases.
- `experiments_baselines.py`: raw-tabular vs graph-feature baseline comparisons.
- `experiments_ablation.py`: graph component ablation study.
- `experiments_interpretability.py`: SHAP and approximate embedding-to-feature mapping.
- `experiments_weight_sensitivity.py`: structured weight robustness study.
- `prepare_revision_results.py`: packages latest experiment outputs into `revision_results/`.
- `README.md`: external-facing project description; keep claims conservative and code-aligned.

## Run Commands
- App: `streamlit run main_app.py`
- Full graph build: `python3 kg_builder_2.py --build-full-graph`
- Label-free embedding refresh: `python3 kg_builder_2.py --no-labels`
- Baselines: `python3 experiments_baselines.py`
- Ablations: `python3 experiments_ablation.py`
- Interpretability: `python3 experiments_interpretability.py`
- Weight sensitivity: `python3 experiments_weight_sensitivity.py`
- Package manuscript artifacts: `python3 prepare_revision_results.py`

## Working Rules
- Do not break the Streamlit app path in `main_app.py`.
- Prefer new modular experiment scripts over large refactors in app code.
- Preserve leakage prevention:
  - do not use `Qchat-10-Score` as a predictive feature
  - do not include `SCREENED_FOR` edges when generating training embeddings
  - keep train/test and CV evaluation stratified and deterministic
- Keep random seeds fixed unless there is a strong scientific reason to change them.
- Document scientific assumptions in code/docstrings when adding heuristics, weights, or interpretation layers.
- Keep repository-facing claims conservative: this is a graph-enhanced ML pipeline, not a fully bidirectional neurosymbolic system.

## Coding Conventions
- Follow existing Python style and keep changes local.
- Default to standalone scripts for experiments and manuscript support.
- Export experiment outputs to `results/` and packaged revision artifacts to `revision_results/`.
- Use machine-readable outputs first: CSV/JSON, plus Markdown/LaTeX when useful for the paper.

## Done Means
- Code runs or is syntax-validated if the environment lacks dependencies.
- Existing app behavior is preserved unless the task explicitly changes it.
- New experiments or analysis write reproducible outputs to disk.
- README/comments are updated if claims or assumptions change.
