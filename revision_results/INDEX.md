# Revision Results

Generated at: `2026-03-24T14:38:02.602345Z`

This folder packages manuscript-facing artifacts from the latest available experiment outputs under `results/`.

## Artifacts

| Output File | Category | Status | Source Run | Producer | Description |
| --- | --- | --- | --- | --- | --- |
| `table_baselines.csv` | table | present | `baseline_run_20260324_143442` | `experiments_baselines.py` | Baseline comparison table for raw tabular, graph-embedding, and combined models. |
| `table_baselines.md` | table | present | `baseline_run_20260324_143442` | `experiments_baselines.py` | Markdown version of the baseline comparison table. |
| `table_baselines.tex` | table | present | `baseline_run_20260324_143442` | `experiments_baselines.py` | LaTeX-ready baseline comparison table. |
| `table_ablations.csv` | table | present | `ablation_run_20260324_132809` | `experiments_ablation.py` | Ablation table comparing full, reduced, and baseline configurations. |
| `table_ablations.md` | table | present | `ablation_run_20260324_132809` | `experiments_ablation.py` | Markdown version of the ablation table. |
| `table_ablations.tex` | table | present | `ablation_run_20260324_132809` | `experiments_ablation.py` | LaTeX-ready ablation table. |
| `table_weight_sensitivity.csv` | table | present | `weight_sensitivity_run_20260324_134356` | `experiments_weight_sensitivity.py` | Weight sensitivity table comparing heuristic, uniform, and perturbed weight configurations. |
| `table_weight_sensitivity.md` | table | present | `weight_sensitivity_run_20260324_134356` | `experiments_weight_sensitivity.py` | Markdown version of the weight sensitivity table. |
| `table_weight_sensitivity.tex` | table | present | `weight_sensitivity_run_20260324_134356` | `experiments_weight_sensitivity.py` | LaTeX-ready weight sensitivity table. |
| `table_global_shap_importance.csv` | table | present | `interpretability_run_20260324_140612` | `experiments_interpretability.py` | Global SHAP importance over embedding dimensions for the main graph classifier. |
| `figure_shap_global_bar.png` | figure | present | `interpretability_run_20260324_140612` | `experiments_interpretability.py` | Publication-friendly global SHAP bar plot. |
| `figure_shap_beeswarm.png` | figure | present | `interpretability_run_20260324_140612` | `experiments_interpretability.py` | SHAP beeswarm plot for the graph classifier. |
| `table_embedding_feature_mapping.csv` | table | present | `interpretability_run_20260324_140612` | `experiments_interpretability.py` | Approximate mapping from important embedding dimensions back to Q-Chat and demographic variables. |
| `table_case_level_explanations.csv` | table | present | `interpretability_run_20260324_140612` | `experiments_interpretability.py` | Case-level SHAP explanations for selected holdout examples. |
| `table_kg_explanation_example.csv` | table | present | `derived_from_packaged_outputs` | `prepare_revision_results.py` | Derived example linking one local SHAP explanation to approximate Q-Chat/demographic feature mappings. |

## How To Produce Missing Artifacts

1. Run `python3 experiments_baselines.py`
2. Run `python3 experiments_ablation.py`
3. Run `python3 experiments_weight_sensitivity.py`
4. Run `python3 experiments_interpretability.py`
5. Re-run `python3 prepare_revision_results.py`

## Notes

- `table_kg_explanation_example.csv` is a derived packaging artifact created from case-level SHAP explanations plus the approximate embedding-feature mapping.
- The embedding-to-feature bridge remains approximate and should be described that way in the manuscript.
