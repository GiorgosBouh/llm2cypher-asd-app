"""
Run ablation experiments for the ASD graph-based pipeline.

This script evaluates explicit, reproducible ablation conditions for the
Neo4j + Node2Vec + XGBoost workflow. It is designed to support paper-ready
component contribution analysis without changing the production Streamlit app.

Implemented ablation conditions:
1. `current_graph_pipeline`
   Production-like graph: Q-Chat answers + demographics + similarity edges.
2. `full_graph_pipeline`
   Adds optional behavior-pattern and complexity nodes/edges.
3. `no_similarity_edges`
   Full graph without `SIMILAR_TO` relationships.
4. `no_behavior_patterns`
   Full graph without behavior-pattern / complexity augmentation.
5. `no_demographic_context`
   Full graph without demographic or submitter context.
6. `simplified_graph_pipeline`
   Questions only: no demographics, no similarity, no behavior-pattern nodes.
7. `full_graph_with_anomaly_gate`
   Full graph plus the existing Isolation Forest workflow used as a reject option.
8. `no_graph_baseline`
   Raw tabular XGBoost baseline for direct comparison.

Outputs:
- `ablation_results.csv`
- `ablation_results.json`
- `ablation_results.md`
- `ablation_results.tex`
- `ablation_cv_fold_metrics.csv`
- `ablation_conditions.json`
- `run_metadata.json`

Examples:
    python3 experiments_ablation.py
    python3 experiments_ablation.py --csv-path Toddler_Autism_dataset_July_2018_2.csv

Notes:
- Graph conditions rebuild the Neo4j graph repeatedly. Run this against a
  research database, not a production instance.
- Embeddings are regenerated in label-free mode after each graph rebuild to
  match the repository's leakage-prevention approach.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from env_utils import load_project_env
from evaluation_safety import build_inductive_graph_embeddings, build_safety_audit, stratified_split
from sklearn.ensemble import IsolationForest
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold, train_test_split

from experiments_baselines import (
    CV_FOLDS,
    DEFAULT_CSV_NAME,
    DEFAULT_RESULTS_DIR,
    RANDOM_STATE,
    TEST_SIZE,
    RAW_CATEGORICAL_FEATURES,
    RAW_NUMERIC_FEATURES,
    available_columns,
    build_model,
    build_preprocessor,
    compute_metrics,
    fetch_graph_embeddings,
    load_dataset,
    predicted_scores,
)


@dataclass(frozen=True)
class AblationCondition:
    name: str
    description: str
    include_similarity: bool
    include_demographics: bool
    include_behavior_patterns: bool
    use_anomaly_gate: bool
    is_graph_condition: bool

    def builder_args(self) -> List[str]:
        args: List[str] = []
        if not self.include_similarity:
            args.append("--exclude-similarity")
        if not self.include_demographics:
            args.append("--exclude-demographics")
        if self.include_behavior_patterns:
            args.append("--include-behavior-patterns")
        return args


GRAPH_CONDITIONS: List[AblationCondition] = [
    AblationCondition(
        name="current_graph_pipeline",
        description="Production-like graph with questions, demographics, submitter context, and similarity edges.",
        include_similarity=True,
        include_demographics=True,
        include_behavior_patterns=False,
        use_anomaly_gate=False,
        is_graph_condition=True,
    ),
    AblationCondition(
        name="full_graph_pipeline",
        description="Extended graph with demographics, similarity edges, and behavior-pattern / complexity nodes.",
        include_similarity=True,
        include_demographics=True,
        include_behavior_patterns=True,
        use_anomaly_gate=False,
        is_graph_condition=True,
    ),
    AblationCondition(
        name="no_similarity_edges",
        description="Full graph without case-to-case similarity edges.",
        include_similarity=False,
        include_demographics=True,
        include_behavior_patterns=True,
        use_anomaly_gate=False,
        is_graph_condition=True,
    ),
    AblationCondition(
        name="no_behavior_patterns",
        description="Graph without behavior-pattern and complexity nodes.",
        include_similarity=True,
        include_demographics=True,
        include_behavior_patterns=False,
        use_anomaly_gate=False,
        is_graph_condition=True,
    ),
    AblationCondition(
        name="no_demographic_context",
        description="Graph without demographic or submitter context.",
        include_similarity=True,
        include_demographics=False,
        include_behavior_patterns=True,
        use_anomaly_gate=False,
        is_graph_condition=True,
    ),
    AblationCondition(
        name="simplified_graph_pipeline",
        description="Questions-only graph with no demographics, no similarity edges, and no behavior-pattern nodes.",
        include_similarity=False,
        include_demographics=False,
        include_behavior_patterns=False,
        use_anomaly_gate=False,
        is_graph_condition=True,
    ),
    AblationCondition(
        name="full_graph_with_anomaly_gate",
        description="Full graph pipeline with Isolation Forest used as a reject option during evaluation.",
        include_similarity=True,
        include_demographics=True,
        include_behavior_patterns=True,
        use_anomaly_gate=True,
        is_graph_condition=True,
    ),
    AblationCondition(
        name="no_graph_baseline",
        description="Raw tabular XGBoost baseline with no graph features.",
        include_similarity=False,
        include_demographics=False,
        include_behavior_patterns=False,
        use_anomaly_gate=False,
        is_graph_condition=False,
    ),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run ablation experiments for the ASD graph pipeline.")
    parser.add_argument(
        "--csv-path",
        default=str(Path(__file__).resolve().parent / DEFAULT_CSV_NAME),
        help="Path to the labeled toddler ASD CSV.",
    )
    parser.add_argument(
        "--results-dir",
        default=DEFAULT_RESULTS_DIR,
        help="Base directory where ablation result subfolders will be created.",
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
    return parser.parse_args()


def ensure_results_dir(base_dir: Path) -> Path:
    run_name = datetime.utcnow().strftime("ablation_run_%Y%m%d_%H%M%S")
    run_dir = base_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def safe_compute_metrics(y_true: pd.Series, y_score: np.ndarray) -> Dict[str, float]:
    if len(y_true) == 0:
        return {
            "roc_auc": float("nan"),
            "f1": float("nan"),
            "precision": float("nan"),
            "recall": float("nan"),
            "ap": float("nan"),
        }
    if len(np.unique(y_true)) < 2:
        y_pred = (y_score >= 0.5).astype(int)
        return {
            "roc_auc": float("nan"),
            "f1": float(f1_score(y_true, y_pred, zero_division=0)),
            "precision": float(precision_score(y_true, y_pred, zero_division=0)),
            "recall": float(recall_score(y_true, y_pred, zero_division=0)),
            "ap": float("nan"),
        }
    return compute_metrics(y_true, y_score)


def run_builder(script_path: Path, csv_path: Path, condition: AblationCondition, mode: str) -> None:
    args = [sys.executable, str(script_path)]
    if mode == "build":
        args.append("--build-full-graph")
        args.extend(["--csv-path", str(csv_path)])
    elif mode == "embed":
        args.append("--no-labels")
    else:
        raise ValueError(f"Unsupported builder mode: {mode}")

    args.extend(condition.builder_args())
    result = subprocess.run(args, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        stderr = result.stderr.strip() or "No stderr returned."
        raise RuntimeError(f"{condition.name} ({mode}) failed: {stderr}")


def prepare_graph_embeddings(
    repo_root: Path,
    csv_path: Path,
    args: argparse.Namespace,
    df: pd.DataFrame,
    condition: AblationCondition,
) -> pd.DataFrame:
    run_builder(repo_root / "kg_builder_2.py", csv_path, condition, mode="build")
    run_builder(repo_root / "kg_builder_2.py", csv_path, condition, mode="embed")

    embedding_df = fetch_graph_embeddings(args, df["Case_No"].tolist())
    merged = df.merge(embedding_df, on="Case_No", how="inner").copy()
    graph_columns = [col for col in merged.columns if col.startswith("emb_")]
    if not graph_columns:
        raise ValueError(f"No embeddings available for condition {condition.name}.")
    return merged


def fit_anomaly_detector(X_train_encoded: np.ndarray) -> IsolationForest:
    contamination = min(0.1, 5.0 / len(X_train_encoded))
    model = IsolationForest(
        contamination=contamination,
        n_estimators=100,
        random_state=RANDOM_STATE,
    )
    model.fit(X_train_encoded)
    return model


def summarize_fold_metrics(rows: Sequence[Dict[str, Any]]) -> Dict[str, float]:
    def stats(key: str) -> Tuple[float, float]:
        values = [row[key] for row in rows]
        return float(np.nanmean(values)), float(np.nanstd(values, ddof=0))

    roc_mean, roc_std = stats("roc_auc")
    f1_mean, f1_std = stats("f1")
    p_mean, p_std = stats("precision")
    r_mean, r_std = stats("recall")
    ap_mean, ap_std = stats("ap")
    cov_mean, cov_std = stats("coverage")
    rej_mean, rej_std = stats("rejection_rate")

    return {
        "cv_roc_auc_mean": roc_mean,
        "cv_roc_auc_std": roc_std,
        "cv_f1_mean": f1_mean,
        "cv_f1_std": f1_std,
        "cv_precision_mean": p_mean,
        "cv_precision_std": p_std,
        "cv_recall_mean": r_mean,
        "cv_recall_std": r_std,
        "cv_ap_mean": ap_mean,
        "cv_ap_std": ap_std,
        "cv_coverage_mean": cov_mean,
        "cv_coverage_std": cov_std,
        "cv_rejection_rate_mean": rej_mean,
        "cv_rejection_rate_std": rej_std,
    }


def evaluate_condition(
    dataset: pd.DataFrame,
    feature_columns: Sequence[str],
    condition: AblationCondition,
    feature_type: str,
    raw_numeric: Sequence[str],
    raw_categorical: Sequence[str],
    graph_columns: Sequence[str],
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    X = dataset.loc[:, list(feature_columns)].copy()
    y = dataset["target"].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    fold_rows: List[Dict[str, Any]] = []

    for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X_train, y_train), start=1):
        X_fold_train = X_train.iloc[train_idx]
        X_fold_val = X_train.iloc[val_idx]
        y_fold_train = y_train.iloc[train_idx]
        y_fold_val = y_train.iloc[val_idx]

        preprocessor = build_preprocessor(raw_numeric, raw_categorical, graph_columns)
        X_fold_train_enc = preprocessor.fit_transform(X_fold_train)
        X_fold_val_enc = preprocessor.transform(X_fold_val)

        classifier = build_model("xgboost", y_fold_train)
        classifier.fit(X_fold_train_enc, y_fold_train)
        y_score = predicted_scores(classifier, X_fold_val_enc)

        coverage = 1.0
        rejection_rate = 0.0
        y_eval = y_fold_val
        score_eval = y_score

        if condition.use_anomaly_gate:
            anomaly_model = fit_anomaly_detector(X_fold_train_enc)
            keep_mask = anomaly_model.predict(X_fold_val_enc) == 1
            coverage = float(np.mean(keep_mask))
            rejection_rate = 1.0 - coverage
            y_eval = y_fold_val.loc[keep_mask]
            score_eval = y_score[keep_mask]

        metrics = safe_compute_metrics(y_eval, score_eval)
        metrics.update(
            {
                "fold": fold_idx,
                "coverage": coverage,
                "rejection_rate": rejection_rate,
                "condition_name": condition.name,
                "feature_type": feature_type,
            }
        )
        fold_rows.append(metrics)

    summary = summarize_fold_metrics(fold_rows)

    final_preprocessor = build_preprocessor(raw_numeric, raw_categorical, graph_columns)
    X_train_enc = final_preprocessor.fit_transform(X_train)
    X_test_enc = final_preprocessor.transform(X_test)

    classifier = build_model("xgboost", y_train)
    classifier.fit(X_train_enc, y_train)
    holdout_scores = predicted_scores(classifier, X_test_enc)

    holdout_coverage = 1.0
    holdout_rejection_rate = 0.0
    y_holdout_eval = y_test
    holdout_score_eval = holdout_scores

    if condition.use_anomaly_gate:
        anomaly_model = fit_anomaly_detector(X_train_enc)
        keep_mask = anomaly_model.predict(X_test_enc) == 1
        holdout_coverage = float(np.mean(keep_mask))
        holdout_rejection_rate = 1.0 - holdout_coverage
        y_holdout_eval = y_test.loc[keep_mask]
        holdout_score_eval = holdout_scores[keep_mask]

    holdout_metrics = safe_compute_metrics(y_holdout_eval, holdout_score_eval)

    row = {
        "condition_name": condition.name,
        "description": condition.description,
        "model_family": "xgboost",
        "feature_type": feature_type,
        "rows_used": len(dataset),
        "include_similarity": condition.include_similarity if condition.is_graph_condition else False,
        "include_demographics": condition.include_demographics if condition.is_graph_condition else False,
        "include_behavior_patterns": condition.include_behavior_patterns if condition.is_graph_condition else False,
        "use_anomaly_gate": condition.use_anomaly_gate,
        "holdout_roc_auc": float(holdout_metrics["roc_auc"]),
        "holdout_f1": float(holdout_metrics["f1"]),
        "holdout_precision": float(holdout_metrics["precision"]),
        "holdout_recall": float(holdout_metrics["recall"]),
        "holdout_ap": float(holdout_metrics["ap"]),
        "holdout_coverage": holdout_coverage,
        "holdout_rejection_rate": holdout_rejection_rate,
        **summary,
    }
    return row, fold_rows


def evaluate_condition_fixed_split(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    condition: AblationCondition,
    feature_type: str,
    raw_numeric: Sequence[str],
    raw_categorical: Sequence[str],
    graph_columns: Sequence[str],
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    feature_columns = list(graph_columns) if feature_type == "graph_embedding" else list(raw_numeric) + list(raw_categorical)
    X_train = train_df.loc[:, feature_columns].copy()
    y_train = train_df["target"].copy()
    X_test = test_df.loc[:, feature_columns].copy()
    y_test = test_df["target"].copy()

    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    fold_rows: List[Dict[str, Any]] = []

    for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X_train, y_train), start=1):
        X_fold_train = X_train.iloc[train_idx]
        X_fold_val = X_train.iloc[val_idx]
        y_fold_train = y_train.iloc[train_idx]
        y_fold_val = y_train.iloc[val_idx]

        preprocessor = build_preprocessor(raw_numeric, raw_categorical, graph_columns)
        X_fold_train_enc = preprocessor.fit_transform(X_fold_train)
        X_fold_val_enc = preprocessor.transform(X_fold_val)

        classifier = build_model("xgboost", y_fold_train)
        classifier.fit(X_fold_train_enc, y_fold_train)
        y_score = predicted_scores(classifier, X_fold_val_enc)

        coverage = 1.0
        rejection_rate = 0.0
        y_eval = y_fold_val
        score_eval = y_score

        if condition.use_anomaly_gate:
            anomaly_model = fit_anomaly_detector(X_fold_train_enc)
            keep_mask = anomaly_model.predict(X_fold_val_enc) == 1
            coverage = float(np.mean(keep_mask))
            rejection_rate = 1.0 - coverage
            y_eval = y_fold_val.loc[keep_mask]
            score_eval = y_score[keep_mask]

        metrics = safe_compute_metrics(y_eval, score_eval)
        metrics.update(
            {
                "fold": fold_idx,
                "coverage": coverage,
                "rejection_rate": rejection_rate,
                "condition_name": condition.name,
                "feature_type": feature_type,
            }
        )
        fold_rows.append(metrics)

    summary = summarize_fold_metrics(fold_rows)

    final_preprocessor = build_preprocessor(raw_numeric, raw_categorical, graph_columns)
    X_train_enc = final_preprocessor.fit_transform(X_train)
    X_test_enc = final_preprocessor.transform(X_test)

    classifier = build_model("xgboost", y_train)
    classifier.fit(X_train_enc, y_train)
    holdout_scores = predicted_scores(classifier, X_test_enc)

    holdout_coverage = 1.0
    holdout_rejection_rate = 0.0
    y_holdout_eval = y_test
    holdout_score_eval = holdout_scores

    if condition.use_anomaly_gate:
        anomaly_model = fit_anomaly_detector(X_train_enc)
        keep_mask = anomaly_model.predict(X_test_enc) == 1
        holdout_coverage = float(np.mean(keep_mask))
        holdout_rejection_rate = 1.0 - holdout_coverage
        y_holdout_eval = y_test.loc[keep_mask]
        holdout_score_eval = holdout_scores[keep_mask]

    holdout_metrics = safe_compute_metrics(y_holdout_eval, holdout_score_eval)

    row = {
        "condition_name": condition.name,
        "description": condition.description,
        "model_family": "xgboost",
        "feature_type": feature_type,
        "rows_used": len(train_df) + len(test_df),
        "include_similarity": condition.include_similarity if condition.is_graph_condition else False,
        "include_demographics": condition.include_demographics if condition.is_graph_condition else False,
        "include_behavior_patterns": condition.include_behavior_patterns if condition.is_graph_condition else False,
        "use_anomaly_gate": condition.use_anomaly_gate,
        "holdout_roc_auc": float(holdout_metrics["roc_auc"]),
        "holdout_f1": float(holdout_metrics["f1"]),
        "holdout_precision": float(holdout_metrics["precision"]),
        "holdout_recall": float(holdout_metrics["recall"]),
        "holdout_ap": float(holdout_metrics["ap"]),
        "holdout_coverage": holdout_coverage,
        "holdout_rejection_rate": holdout_rejection_rate,
        **summary,
    }
    return row, fold_rows


def markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "No results were produced.\n"

    cols = list(df.columns)
    header = "| " + " | ".join(cols) + " |"
    divider = "| " + " | ".join(["---"] * len(cols)) + " |"
    lines = [header, divider]
    for _, row in df.iterrows():
        values = [str(row[col]) for col in cols]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines) + "\n"


def write_outputs(
    run_dir: Path,
    summary_rows: Sequence[Dict[str, Any]],
    fold_rows: Sequence[Dict[str, Any]],
    conditions: Sequence[AblationCondition],
    metadata: Dict[str, Any],
) -> None:
    summary_df = pd.DataFrame(summary_rows)
    fold_df = pd.DataFrame(fold_rows)

    if not summary_df.empty:
        summary_df = summary_df.sort_values(
            by=["holdout_roc_auc", "holdout_f1", "cv_roc_auc_mean"],
            ascending=False,
            na_position="last",
        )

    summary_df.to_csv(run_dir / "ablation_results.csv", index=False)
    fold_df.to_csv(run_dir / "ablation_cv_fold_metrics.csv", index=False)

    with open(run_dir / "ablation_results.json", "w", encoding="utf-8") as f:
        json.dump(summary_df.to_dict(orient="records"), f, indent=2)

    with open(run_dir / "ablation_conditions.json", "w", encoding="utf-8") as f:
        json.dump([condition.__dict__ for condition in conditions], f, indent=2)

    with open(run_dir / "run_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    with open(run_dir / "ablation_results.md", "w", encoding="utf-8") as f:
        f.write("# Ablation Results\n\n")
        f.write(markdown_table(summary_df))

    with open(run_dir / "ablation_results.tex", "w", encoding="utf-8") as f:
        if summary_df.empty:
            f.write("% No results were produced.\n")
        else:
            f.write(summary_df.to_latex(index=False, float_format=lambda x: f"{x:.4f}"))


def main() -> None:
    load_project_env()
    np.random.seed(RANDOM_STATE)

    args = parse_args()
    repo_root = Path(__file__).resolve().parent
    csv_path = Path(args.csv_path).resolve()
    run_dir = ensure_results_dir(Path(args.results_dir).resolve())

    df = load_dataset(csv_path)
    raw_numeric = available_columns(df, RAW_NUMERIC_FEATURES)
    raw_categorical = available_columns(df, RAW_CATEGORICAL_FEATURES)
    raw_feature_columns = raw_numeric + raw_categorical
    df, safety_audit = build_safety_audit(df, raw_feature_columns)
    train_df, test_df = stratified_split(df)

    summary_rows: List[Dict[str, Any]] = []
    fold_rows: List[Dict[str, Any]] = []
    graph_cache: Dict[Tuple[bool, bool, bool], Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]] = {}
    skipped_conditions: List[str] = []

    for condition in GRAPH_CONDITIONS:
        if condition.is_graph_condition:
            cache_key = (
                condition.include_similarity,
                condition.include_demographics,
                condition.include_behavior_patterns,
            )
            if cache_key not in graph_cache:
                graph_cache[cache_key] = build_inductive_graph_embeddings(
                    repo_root=repo_root,
                    args=args,
                    train_df=train_df,
                    test_df=test_df,
                    csv_columns=[column for column in df.columns if column != "target"],
                    extra_builder_args=condition.builder_args(),
                )
            merged_train, merged_test, _ = graph_cache[cache_key]
            graph_columns = [col for col in merged_train.columns if col.startswith("emb_")]
            row, folds = evaluate_condition_fixed_split(
                train_df=merged_train,
                test_df=merged_test,
                condition=condition,
                feature_type="graph_embedding",
                raw_numeric=[],
                raw_categorical=[],
                graph_columns=graph_columns,
            )
        else:
            if safety_audit.deterministic_target_from_answers:
                skipped_conditions.append(condition.name)
                continue
            row, folds = evaluate_condition_fixed_split(
                train_df=train_df,
                test_df=test_df,
                condition=condition,
                feature_type="raw_tabular",
                raw_numeric=raw_numeric,
                raw_categorical=raw_categorical,
                graph_columns=[],
            )

        summary_rows.append(row)
        fold_rows.extend(folds)

    metadata = {
        "generated_at_utc": datetime.utcnow().isoformat() + "Z",
        "csv_path": str(csv_path),
        "results_dir": str(run_dir),
        "random_state": RANDOM_STATE,
        "test_size": TEST_SIZE,
        "cv_folds": CV_FOLDS,
        "neo4j_uri_present": bool(args.neo4j_uri),
        "raw_numeric_features": raw_numeric,
        "raw_categorical_features": raw_categorical,
        "duplicate_rows_removed_before_split": safety_audit.duplicate_rows_removed,
        "deterministic_target_from_answers": safety_audit.deterministic_target_from_answers,
        "deterministic_target_note": safety_audit.deterministic_rule_note,
        "skipped_conditions": skipped_conditions,
        "graph_protocol": "train-only graph build with per-test-case embedding inference",
        "conditions_run": [condition.name for condition in GRAPH_CONDITIONS],
    }

    write_outputs(run_dir, summary_rows, fold_rows, GRAPH_CONDITIONS, metadata)

    print(f"Saved ablation results to: {run_dir}")
    summary_df = pd.DataFrame(summary_rows)
    if not summary_df.empty:
        print(
            summary_df[
                [
                    "condition_name",
                    "feature_type",
                    "holdout_roc_auc",
                    "holdout_f1",
                    "holdout_coverage",
                ]
            ].to_string(index=False)
        )


if __name__ == "__main__":
    main()
