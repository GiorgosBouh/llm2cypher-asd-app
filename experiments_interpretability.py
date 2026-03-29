"""
Interpretability pipeline for the NeuroCypher ASD graph-embedding classifier.

What this script provides:
1. Model-level SHAP explanations for the XGBoost classifier trained on graph embeddings
2. Case-level SHAP explanations for selected holdout examples
3. An approximate mapping from important embedding dimensions back to original
   Q-Chat / demographic features via correlation analysis
4. Exportable tables and plots for manuscript use

Important limitation:
The classifier is trained on latent graph embeddings, so SHAP explanations are
directly valid for embedding dimensions only. Any mapping from embedding
dimensions back to Q-Chat items or demographic variables is approximate rather
than causal or exact. This script makes that distinction explicit in its
artifacts and metadata.

Outputs:
- global_shap_importance.csv
- case_level_explanations.csv
- embedding_feature_mapping.csv
- holdout_predictions.csv
- interpretability_summary.json
- shap_summary_bar.png
- shap_summary_beeswarm.png

Examples:
    python3 experiments_interpretability.py
    python3 experiments_interpretability.py --top-k-dims 20 --top-k-cases 10
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from env_utils import load_project_env
from evaluation_safety import build_inductive_graph_embeddings, build_safety_audit, stratified_split
from sklearn.model_selection import train_test_split

from experiments_baselines import (
    DEFAULT_CSV_NAME,
    DEFAULT_RESULTS_DIR,
    RANDOM_STATE,
    TEST_SIZE,
    RAW_CATEGORICAL_FEATURES,
    RAW_NUMERIC_FEATURES,
    available_columns,
    build_model,
    compute_metrics,
    fetch_graph_embeddings,
    load_dataset,
    predicted_scores,
)

try:
    import shap
except ImportError as exc:  # pragma: no cover - environment dependent
    raise ImportError(
        "The `shap` package is required for experiments_interpretability.py. "
        "Install dependencies from requirements.txt before running this script."
    ) from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run SHAP-based interpretability analysis for the graph classifier.")
    parser.add_argument(
        "--csv-path",
        default=str(Path(__file__).resolve().parent / DEFAULT_CSV_NAME),
        help="Path to the labeled toddler ASD CSV.",
    )
    parser.add_argument(
        "--results-dir",
        default=DEFAULT_RESULTS_DIR,
        help="Base directory where interpretability result subfolders will be created.",
    )
    parser.add_argument(
        "--neo4j-uri",
        default=os.getenv("NEO4J_URI"),
        help="Neo4j URI. Defaults to NEO4J_URI from the environment.",
    )
    parser.add_argument(
        "--neo4j-user",
        default=os.getenv("NEO4J_USER", "neo4j"),
        help="Neo4j username. Defaults to NEO4J_USER from the environment.",
    )
    parser.add_argument(
        "--neo4j-password",
        default=os.getenv("NEO4J_PASSWORD"),
        help="Neo4j password. Defaults to NEO4J_PASSWORD from the environment.",
    )
    parser.add_argument(
        "--neo4j-database",
        default=os.getenv("NEO4J_DB", "neo4j"),
        help="Neo4j database name. Defaults to NEO4J_DB or 'neo4j'.",
    )
    parser.add_argument(
        "--top-k-dims",
        type=int,
        default=15,
        help="Number of top embedding dimensions to keep in summary tables.",
    )
    parser.add_argument(
        "--top-k-cases",
        type=int,
        default=5,
        help="Number of holdout cases to export detailed case-level explanations for.",
    )
    parser.add_argument(
        "--mapping-top-features",
        type=int,
        default=5,
        help="Number of original features to keep per important embedding dimension.",
    )
    return parser.parse_args()


def ensure_results_dir(base_dir: Path) -> Path:
    run_name = datetime.utcnow().strftime("interpretability_run_%Y%m%d_%H%M%S")
    run_dir = base_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def raw_feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    numeric_cols = available_columns(df, RAW_NUMERIC_FEATURES)
    categorical_cols = available_columns(df, RAW_CATEGORICAL_FEATURES)

    parts: List[pd.DataFrame] = []
    if numeric_cols:
        numeric_df = df[numeric_cols].copy()
        numeric_df = numeric_df.fillna(numeric_df.median(numeric_only=True))
        parts.append(numeric_df)

    if categorical_cols:
        cat_df = df[categorical_cols].copy()
        cat_df = cat_df.fillna("missing")
        cat_df = pd.get_dummies(cat_df, dummy_na=False, drop_first=False)
        parts.append(cat_df)

    if not parts:
        raise ValueError("No raw features available for approximate mapping analysis.")

    matrix = pd.concat(parts, axis=1)
    matrix.columns = [str(col) for col in matrix.columns]
    return matrix


def train_graph_classifier(
    train_merged: pd.DataFrame,
    test_merged: pd.DataFrame,
) -> Tuple[Any, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    graph_columns = [col for col in train_merged.columns if col.startswith("emb_")]
    if not graph_columns:
        raise ValueError("No graph embedding columns found.")

    X_train = train_merged[graph_columns].copy()
    y_train = train_merged["target"].copy()
    X_test = test_merged[graph_columns].copy()
    y_test = test_merged["target"].copy()

    model = build_model("xgboost", y_train)
    model.fit(X_train, y_train)
    return model, X_train, X_test, y_train, y_test


def shap_values_to_array(shap_values: Any) -> np.ndarray:
    values = getattr(shap_values, "values", shap_values)
    if isinstance(values, list):
        values = values[0]
    values = np.asarray(values)
    if values.ndim == 3:
        values = values[:, :, 0]
    return values


def build_global_importance_table(
    shap_matrix: np.ndarray,
    X_test: pd.DataFrame,
    top_k_dims: int,
) -> pd.DataFrame:
    mean_abs_shap = np.mean(np.abs(shap_matrix), axis=0)
    mean_feature_value = np.mean(X_test.to_numpy(), axis=0)

    df = pd.DataFrame(
        {
            "embedding_dim": X_test.columns,
            "mean_abs_shap": mean_abs_shap,
            "mean_feature_value": mean_feature_value,
        }
    ).sort_values("mean_abs_shap", ascending=False)

    return df.head(top_k_dims).reset_index(drop=True)


def build_case_level_table(
    shap_matrix: np.ndarray,
    X_test: pd.DataFrame,
    merged_test: pd.DataFrame,
    y_test: pd.Series,
    y_score: np.ndarray,
    top_k_dims: int,
    top_k_cases: int,
) -> pd.DataFrame:
    selected_indices = np.argsort(np.abs(y_score - 0.5))[::-1][:top_k_cases]
    rows: List[Dict[str, Any]] = []

    for idx in selected_indices:
        case_features = X_test.iloc[idx]
        case_shap = shap_matrix[idx]
        top_dims = np.argsort(np.abs(case_shap))[::-1][:top_k_dims]

        for rank, dim_idx in enumerate(top_dims, start=1):
            rows.append(
                {
                    "case_id": int(merged_test.iloc[idx]["Case_No"]),
                    "rank": rank,
                    "embedding_dim": X_test.columns[dim_idx],
                    "shap_value": float(case_shap[dim_idx]),
                    "abs_shap_value": float(abs(case_shap[dim_idx])),
                    "embedding_value": float(case_features.iloc[dim_idx]),
                    "predicted_probability": float(y_score[idx]),
                    "true_label": int(y_test.iloc[idx]),
                }
            )

    return pd.DataFrame(rows)


def build_holdout_prediction_table(
    merged_test: pd.DataFrame,
    y_test: pd.Series,
    y_score: np.ndarray,
) -> pd.DataFrame:
    prediction_df = merged_test[["Case_No"]].copy()
    prediction_df["true_label"] = y_test.to_numpy()
    prediction_df["predicted_probability"] = y_score
    prediction_df["predicted_label"] = (y_score >= 0.5).astype(int)
    prediction_df["prediction_margin"] = np.abs(y_score - 0.5)
    return prediction_df.sort_values("prediction_margin", ascending=False).reset_index(drop=True)


def build_embedding_feature_mapping(
    merged: pd.DataFrame,
    important_dims: Sequence[str],
    mapping_top_features: int,
) -> pd.DataFrame:
    raw_df = raw_feature_matrix(merged)
    embedding_df = merged[list(important_dims)].copy()

    rows: List[Dict[str, Any]] = []
    for dim in important_dims:
        dim_values = embedding_df[dim]
        correlations: List[Tuple[str, float]] = []

        for feature_name in raw_df.columns:
            feature_values = raw_df[feature_name]
            if feature_values.nunique(dropna=False) <= 1:
                continue

            corr = np.corrcoef(dim_values, feature_values)[0, 1]
            if np.isfinite(corr):
                correlations.append((feature_name, float(corr)))

        correlations.sort(key=lambda item: abs(item[1]), reverse=True)
        for rank, (feature_name, corr) in enumerate(correlations[:mapping_top_features], start=1):
            feature_group = "demographic_or_submitter"
            if feature_name.startswith("A") or feature_name == "Age_Mons":
                feature_group = "qchat_or_age"

            rows.append(
                {
                    "embedding_dim": dim,
                    "mapping_rank": rank,
                    "approx_feature": feature_name,
                    "approx_feature_group": feature_group,
                    "correlation": corr,
                    "absolute_correlation": abs(corr),
                    "mapping_note": "Approximate association only; embedding dimensions do not map one-to-one to clinical variables.",
                }
            )

    return pd.DataFrame(rows)


def save_shap_plots(explainer: Any, shap_matrix: np.ndarray, X_test: pd.DataFrame, run_dir: Path, top_k_dims: int) -> None:
    display_df = X_test.iloc[:, :].copy()

    plt.figure()
    shap.summary_plot(
        shap_matrix,
        display_df,
        plot_type="bar",
        max_display=top_k_dims,
        show=False,
    )
    plt.tight_layout()
    plt.savefig(run_dir / "shap_summary_bar.png", dpi=200, bbox_inches="tight")
    plt.close()

    plt.figure()
    shap.summary_plot(
        shap_matrix,
        display_df,
        max_display=top_k_dims,
        show=False,
    )
    plt.tight_layout()
    plt.savefig(run_dir / "shap_summary_beeswarm.png", dpi=200, bbox_inches="tight")
    plt.close()


def main() -> None:
    load_project_env()
    np.random.seed(RANDOM_STATE)

    args = parse_args()
    csv_path = Path(args.csv_path).resolve()
    run_dir = ensure_results_dir(Path(args.results_dir).resolve())
    repo_root = Path(__file__).resolve().parent

    df = load_dataset(csv_path)
    raw_numeric = available_columns(df, RAW_NUMERIC_FEATURES)
    raw_categorical = available_columns(df, RAW_CATEGORICAL_FEATURES)
    raw_feature_columns = raw_numeric + raw_categorical
    df, safety_audit = build_safety_audit(df, raw_feature_columns)
    train_df, test_df = stratified_split(df)
    train_merged, test_merged, graph_meta = build_inductive_graph_embeddings(
        repo_root=repo_root,
        args=args,
        train_df=train_df,
        test_df=test_df,
        csv_columns=[column for column in df.columns if column != "target"],
        extra_builder_args=["--include-behavior-patterns"],
    )

    if train_merged.empty or test_merged.empty:
        raise ValueError("No overlap between split dataset rows and leakage-safe Neo4j embeddings.")

    model, X_train, X_test, y_train, y_test = train_graph_classifier(train_merged, test_merged)
    y_score = predicted_scores(model, X_test.to_numpy())
    metrics = compute_metrics(y_test, y_score)

    explainer = shap.TreeExplainer(model)
    shap_raw = explainer(X_test)
    shap_matrix = shap_values_to_array(shap_raw)

    global_importance_df = build_global_importance_table(shap_matrix, X_test, args.top_k_dims)

    case_level_df = build_case_level_table(
        shap_matrix=shap_matrix,
        X_test=X_test,
        merged_test=test_merged,
        y_test=y_test,
        y_score=y_score,
        top_k_dims=args.top_k_dims,
        top_k_cases=args.top_k_cases,
    )
    holdout_prediction_df = build_holdout_prediction_table(test_merged, y_test, y_score)

    important_dims = global_importance_df["embedding_dim"].tolist()
    mapping_df = build_embedding_feature_mapping(
        merged=train_merged,
        important_dims=important_dims,
        mapping_top_features=args.mapping_top_features,
    )

    global_importance_df.to_csv(run_dir / "global_shap_importance.csv", index=False)
    case_level_df.to_csv(run_dir / "case_level_explanations.csv", index=False)
    mapping_df.to_csv(run_dir / "embedding_feature_mapping.csv", index=False)
    holdout_prediction_df.to_csv(run_dir / "holdout_predictions.csv", index=False)

    save_shap_plots(explainer, shap_matrix, X_test, run_dir, args.top_k_dims)

    summary = {
        "generated_at_utc": datetime.utcnow().isoformat() + "Z",
        "csv_path": str(csv_path),
        "results_dir": str(run_dir),
        "rows_with_embeddings": int(len(train_merged) + len(test_merged)),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "duplicate_rows_removed_before_split": safety_audit.duplicate_rows_removed,
        "deterministic_target_from_answers": safety_audit.deterministic_target_from_answers,
        "deterministic_target_note": safety_audit.deterministic_rule_note,
        "graph_protocol": graph_meta["graph_protocol"],
        "metrics": {key: float(value) for key, value in metrics.items()},
        "top_k_dims": args.top_k_dims,
        "top_k_cases": args.top_k_cases,
        "mapping_top_features": args.mapping_top_features,
        "limitation_note": (
            "SHAP explanations are exact for embedding features in the trained XGBoost model. "
            "The mapping from embeddings back to Q-Chat or demographic variables is correlation-based "
            "and should be treated as approximate interpretability support rather than exact mechanistic attribution."
        ),
    }

    with open(run_dir / "interpretability_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Saved interpretability outputs to: {run_dir}")
    print(pd.DataFrame([summary["metrics"]]).to_string(index=False))


if __name__ == "__main__":
    main()
