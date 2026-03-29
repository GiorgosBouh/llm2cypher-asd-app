"""
Structured sensitivity analysis for hand-crafted graph edge weights.

This script evaluates whether manually chosen graph edge weights materially
affect downstream ASD classification performance. It does not perform exhaustive
hyperparameter search. Instead, it runs a small, explicit suite of robustness
checks around a heuristic baseline profile.

Study design:
- Fixed graph topology with all weighted edge families enabled
  (answers, demographics, similarity, behavior patterns, complexity)
- Deterministic 70/30 stratified train/test split
- Deterministic 5-fold stratified CV on the training split
- Explicit weight configurations:
  * heuristic baseline
  * uniform weights
  * all manual weights -10%
  * all manual weights +10%
  * all manual weights -20%
  * all manual weights +20%
  * neutralized similarity family weights
  * neutralized demographic-risk weights

Outputs:
- weight_sensitivity_results.csv
- weight_sensitivity_results.json
- weight_sensitivity_results.md
- weight_sensitivity_results.tex
- weight_sensitivity_cv_fold_metrics.csv
- weight_configurations.json
- sensitivity_summary.json
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
from env_utils import load_project_env
from evaluation_safety import build_inductive_graph_embeddings, build_safety_audit, stratified_split
from sklearn.model_selection import StratifiedKFold, train_test_split

from experiments_baselines import (
    CV_FOLDS,
    DEFAULT_CSV_NAME,
    DEFAULT_RESULTS_DIR,
    RANDOM_STATE,
    RAW_CATEGORICAL_FEATURES,
    RAW_NUMERIC_FEATURES,
    TEST_SIZE,
    available_columns,
    build_model,
    compute_metrics,
    fetch_graph_embeddings,
    load_dataset,
    predicted_scores,
)


BASE_WEIGHT_PROFILE: Dict[str, float] = {
    "answer_edge_weight": 1.5,
    "demographic_edge_weight": 1.0,
    "submitter_edge_weight": 1.0,
    "pattern_edge_weight": 1.8,
    "complexity_edge_weight": 1.5,
    "similarity_answer_component": 0.7,
    "similarity_demo_component": 0.3,
    "similarity_global_scale": 1.0,
    "demo_family_yes_multiplier": 2.0,
    "demo_jaundice_yes_multiplier": 1.5,
    "demo_male_multiplier": 1.3,
}


@dataclass(frozen=True)
class WeightConfiguration:
    name: str
    description: str
    profile: Dict[str, float]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a structured sensitivity analysis for graph edge weights.")
    parser.add_argument(
        "--csv-path",
        default=str(Path(__file__).resolve().parent / DEFAULT_CSV_NAME),
        help="Path to the labeled toddler ASD CSV.",
    )
    parser.add_argument(
        "--results-dir",
        default=DEFAULT_RESULTS_DIR,
        help="Base directory where sensitivity result subfolders will be created.",
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
    run_name = datetime.utcnow().strftime("weight_sensitivity_run_%Y%m%d_%H%M%S")
    run_dir = base_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def scaled_profile(base: Dict[str, float], scale: float) -> Dict[str, float]:
    profile = deepcopy(base)
    for key in [
        "answer_edge_weight",
        "demographic_edge_weight",
        "submitter_edge_weight",
        "pattern_edge_weight",
        "complexity_edge_weight",
        "similarity_global_scale",
        "demo_family_yes_multiplier",
        "demo_jaundice_yes_multiplier",
        "demo_male_multiplier",
    ]:
        profile[key] = float(profile[key]) * scale
    return profile


def build_weight_configurations() -> List[WeightConfiguration]:
    uniform = {
        "answer_edge_weight": 1.0,
        "demographic_edge_weight": 1.0,
        "submitter_edge_weight": 1.0,
        "pattern_edge_weight": 1.0,
        "complexity_edge_weight": 1.0,
        "similarity_answer_component": 0.5,
        "similarity_demo_component": 0.5,
        "similarity_global_scale": 1.0,
        "demo_family_yes_multiplier": 1.0,
        "demo_jaundice_yes_multiplier": 1.0,
        "demo_male_multiplier": 1.0,
    }

    similarity_neutral = deepcopy(BASE_WEIGHT_PROFILE)
    similarity_neutral["similarity_answer_component"] = 0.5
    similarity_neutral["similarity_demo_component"] = 0.5

    demographic_neutral = deepcopy(BASE_WEIGHT_PROFILE)
    demographic_neutral["demo_family_yes_multiplier"] = 1.0
    demographic_neutral["demo_jaundice_yes_multiplier"] = 1.0
    demographic_neutral["demo_male_multiplier"] = 1.0

    return [
        WeightConfiguration(
            name="heuristic_baseline",
            description="Heuristic baseline profile reflecting manually chosen edge-family weights.",
            profile=deepcopy(BASE_WEIGHT_PROFILE),
        ),
        WeightConfiguration(
            name="uniform_weights",
            description="All edge families set to uniform weight; similarity split neutralized to 0.5/0.5.",
            profile=uniform,
        ),
        WeightConfiguration(
            name="all_weights_minus_10",
            description="Global -10% perturbation of manual weight magnitudes.",
            profile=scaled_profile(BASE_WEIGHT_PROFILE, 0.9),
        ),
        WeightConfiguration(
            name="all_weights_plus_10",
            description="Global +10% perturbation of manual weight magnitudes.",
            profile=scaled_profile(BASE_WEIGHT_PROFILE, 1.1),
        ),
        WeightConfiguration(
            name="all_weights_minus_20",
            description="Global -20% perturbation of manual weight magnitudes.",
            profile=scaled_profile(BASE_WEIGHT_PROFILE, 0.8),
        ),
        WeightConfiguration(
            name="all_weights_plus_20",
            description="Global +20% perturbation of manual weight magnitudes.",
            profile=scaled_profile(BASE_WEIGHT_PROFILE, 1.2),
        ),
        WeightConfiguration(
            name="neutral_similarity_family",
            description="Similarity composition neutralized from 0.7/0.3 to 0.5/0.5 while other weights remain heuristic.",
            profile=similarity_neutral,
        ),
        WeightConfiguration(
            name="neutral_demographic_risk_weights",
            description="Family-history, jaundice, and male-sex demographic multipliers set to 1.0.",
            profile=demographic_neutral,
        ),
    ]


def run_builder(repo_root: Path, csv_path: Path, config: WeightConfiguration, mode: str) -> None:
    cmd = [
        sys.executable,
        str(repo_root / "kg_builder_2.py"),
        "--include-behavior-patterns",
        "--weight-profile-json",
        json.dumps(config.profile, sort_keys=True),
    ]

    if mode == "build":
        cmd.extend(["--build-full-graph", "--csv-path", str(csv_path)])
    elif mode == "embed":
        cmd.append("--no-labels")
    else:
        raise ValueError(f"Unsupported builder mode: {mode}")

    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        stderr = result.stderr.strip() or "No stderr returned."
        raise RuntimeError(f"{config.name} ({mode}) failed: {stderr}")


def prepare_embeddings(repo_root: Path, csv_path: Path, args: argparse.Namespace, df: pd.DataFrame, config: WeightConfiguration) -> pd.DataFrame:
    run_builder(repo_root, csv_path, config, "build")
    run_builder(repo_root, csv_path, config, "embed")
    embedding_df = fetch_graph_embeddings(args, df["Case_No"].tolist())
    merged = df.merge(embedding_df, on="Case_No", how="inner").copy()
    graph_columns = [col for col in merged.columns if col.startswith("emb_")]
    if not graph_columns:
        raise ValueError(f"No embeddings found for configuration {config.name}.")
    return merged


def run_cv(X_train: pd.DataFrame, y_train: pd.Series) -> List[Dict[str, float]]:
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    fold_rows: List[Dict[str, float]] = []

    for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X_train, y_train), start=1):
        X_fold_train = X_train.iloc[train_idx]
        y_fold_train = y_train.iloc[train_idx]
        X_fold_val = X_train.iloc[val_idx]
        y_fold_val = y_train.iloc[val_idx]

        model = build_model("xgboost", y_fold_train)
        model.fit(X_fold_train, y_fold_train)
        y_score = predicted_scores(model, X_fold_val.to_numpy())
        metrics = compute_metrics(y_fold_val, y_score)
        metrics["fold"] = fold_idx
        fold_rows.append(metrics)

    return fold_rows


def summarize_cv(fold_rows: Sequence[Dict[str, float]]) -> Dict[str, float]:
    return {
        "cv_roc_auc_mean": float(np.mean([row["roc_auc"] for row in fold_rows])),
        "cv_roc_auc_std": float(np.std([row["roc_auc"] for row in fold_rows], ddof=0)),
        "cv_f1_mean": float(np.mean([row["f1"] for row in fold_rows])),
        "cv_f1_std": float(np.std([row["f1"] for row in fold_rows], ddof=0)),
        "cv_precision_mean": float(np.mean([row["precision"] for row in fold_rows])),
        "cv_precision_std": float(np.std([row["precision"] for row in fold_rows], ddof=0)),
        "cv_recall_mean": float(np.mean([row["recall"] for row in fold_rows])),
        "cv_recall_std": float(np.std([row["recall"] for row in fold_rows], ddof=0)),
        "cv_ap_mean": float(np.mean([row["ap"] for row in fold_rows])),
        "cv_ap_std": float(np.std([row["ap"] for row in fold_rows], ddof=0)),
    }


def evaluate_configuration(merged: pd.DataFrame, config: WeightConfiguration) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    graph_columns = [col for col in merged.columns if col.startswith("emb_")]
    X = merged[graph_columns].copy()
    y = merged["target"].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    fold_rows = run_cv(X_train, y_train)
    cv_summary = summarize_cv(fold_rows)

    model = build_model("xgboost", y_train)
    model.fit(X_train, y_train)
    y_score = predicted_scores(model, X_test.to_numpy())
    holdout = compute_metrics(y_test, y_score)

    row = {
        "configuration_name": config.name,
        "description": config.description,
        "rows_used": len(merged),
        "holdout_roc_auc": float(holdout["roc_auc"]),
        "holdout_f1": float(holdout["f1"]),
        "holdout_precision": float(holdout["precision"]),
        "holdout_recall": float(holdout["recall"]),
        "holdout_ap": float(holdout["ap"]),
        **cv_summary,
    }

    for fold_row in fold_rows:
        fold_row["configuration_name"] = config.name

    return row, fold_rows


def evaluate_configuration_fixed_split(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    config: WeightConfiguration,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    graph_columns = [col for col in train_df.columns if col.startswith("emb_")]
    X_train = train_df[graph_columns].copy()
    y_train = train_df["target"].copy()
    X_test = test_df[graph_columns].copy()
    y_test = test_df["target"].copy()

    fold_rows = run_cv(X_train, y_train)
    cv_summary = summarize_cv(fold_rows)

    model = build_model("xgboost", y_train)
    model.fit(X_train, y_train)
    y_score = predicted_scores(model, X_test.to_numpy())
    holdout = compute_metrics(y_test, y_score)

    row = {
        "configuration_name": config.name,
        "description": config.description,
        "rows_used": len(train_df) + len(test_df),
        "holdout_roc_auc": float(holdout["roc_auc"]),
        "holdout_f1": float(holdout["f1"]),
        "holdout_precision": float(holdout["precision"]),
        "holdout_recall": float(holdout["recall"]),
        "holdout_ap": float(holdout["ap"]),
        **cv_summary,
    }

    for fold_row in fold_rows:
        fold_row["configuration_name"] = config.name

    return row, fold_rows


def classify_sensitivity(delta_auc: float, delta_f1: float) -> str:
    if delta_auc <= 0.01 and delta_f1 <= 0.02:
        return "robust"
    if delta_auc <= 0.03 and delta_f1 <= 0.05:
        return "moderately_sensitive"
    return "sensitive"


def build_summary(results_df: pd.DataFrame) -> Dict[str, Any]:
    baseline_row = results_df.loc[results_df["configuration_name"] == "heuristic_baseline"].iloc[0]
    comparison_rows = []

    for _, row in results_df.iterrows():
        delta_auc = abs(float(row["holdout_roc_auc"]) - float(baseline_row["holdout_roc_auc"]))
        delta_f1 = abs(float(row["holdout_f1"]) - float(baseline_row["holdout_f1"]))
        comparison_rows.append(
            {
                "configuration_name": row["configuration_name"],
                "delta_holdout_roc_auc_vs_baseline": delta_auc,
                "delta_holdout_f1_vs_baseline": delta_f1,
                "sensitivity_label": classify_sensitivity(delta_auc, delta_f1),
            }
        )

    worst_auc = max(item["delta_holdout_roc_auc_vs_baseline"] for item in comparison_rows)
    worst_f1 = max(item["delta_holdout_f1_vs_baseline"] for item in comparison_rows)

    overall_label = classify_sensitivity(worst_auc, worst_f1)
    return {
        "baseline_configuration": "heuristic_baseline",
        "overall_sensitivity_label": overall_label,
        "max_delta_holdout_roc_auc": worst_auc,
        "max_delta_holdout_f1": worst_f1,
        "comparison_rows": comparison_rows,
        "interpretation_note": (
            "Sensitivity labels are heuristic summaries for manuscript convenience: "
            "robust if max delta ROC AUC <= 0.01 and max delta F1 <= 0.02; "
            "moderately sensitive if max delta ROC AUC <= 0.03 and max delta F1 <= 0.05; "
            "otherwise sensitive."
        ),
    }


def markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "No results were produced.\n"

    cols = list(df.columns)
    header = "| " + " | ".join(cols) + " |"
    divider = "| " + " | ".join(["---"] * len(cols)) + " |"
    lines = [header, divider]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(str(row[col]) for col in cols) + " |")
    return "\n".join(lines) + "\n"


def write_outputs(
    run_dir: Path,
    results_df: pd.DataFrame,
    fold_df: pd.DataFrame,
    configs: Sequence[WeightConfiguration],
    summary: Dict[str, Any],
    metadata: Dict[str, Any],
) -> None:
    results_df.to_csv(run_dir / "weight_sensitivity_results.csv", index=False)
    fold_df.to_csv(run_dir / "weight_sensitivity_cv_fold_metrics.csv", index=False)

    with open(run_dir / "weight_sensitivity_results.json", "w", encoding="utf-8") as f:
        json.dump(results_df.to_dict(orient="records"), f, indent=2)

    with open(run_dir / "weight_configurations.json", "w", encoding="utf-8") as f:
        json.dump(
            [
                {
                    "name": config.name,
                    "description": config.description,
                    "profile": config.profile,
                }
                for config in configs
            ],
            f,
            indent=2,
        )

    with open(run_dir / "sensitivity_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    with open(run_dir / "run_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    with open(run_dir / "weight_sensitivity_results.md", "w", encoding="utf-8") as f:
        f.write("# Weight Sensitivity Results\n\n")
        f.write(markdown_table(results_df))

    with open(run_dir / "weight_sensitivity_results.tex", "w", encoding="utf-8") as f:
        if results_df.empty:
            f.write("% No results were produced.\n")
        else:
            f.write(results_df.to_latex(index=False, float_format=lambda x: f"{x:.4f}"))


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
    configs = build_weight_configurations()

    result_rows: List[Dict[str, Any]] = []
    fold_rows: List[Dict[str, Any]] = []

    for config in configs:
        builder_args = [
            "--include-behavior-patterns",
            "--weight-profile-json",
            json.dumps(config.profile, sort_keys=True),
        ]
        merged_train, merged_test, _ = build_inductive_graph_embeddings(
            repo_root=repo_root,
            args=args,
            train_df=train_df,
            test_df=test_df,
            csv_columns=[column for column in df.columns if column != "target"],
            extra_builder_args=builder_args,
        )
        row, config_fold_rows = evaluate_configuration_fixed_split(merged_train, merged_test, config)
        result_rows.append(row)
        fold_rows.extend(config_fold_rows)

    results_df = pd.DataFrame(result_rows).sort_values(
        by=["holdout_roc_auc", "holdout_f1", "cv_roc_auc_mean"],
        ascending=False,
    )
    fold_df = pd.DataFrame(fold_rows)
    summary = build_summary(results_df)
    metadata = {
        "generated_at_utc": datetime.utcnow().isoformat() + "Z",
        "csv_path": str(csv_path),
        "results_dir": str(run_dir),
        "random_state": RANDOM_STATE,
        "test_size": TEST_SIZE,
        "cv_folds": CV_FOLDS,
        "duplicate_rows_removed_before_split": safety_audit.duplicate_rows_removed,
        "deterministic_target_from_answers": safety_audit.deterministic_target_from_answers,
        "deterministic_target_note": safety_audit.deterministic_rule_note,
        "study_note": (
            "This sensitivity analysis uses a fixed graph topology with all weighted edge families enabled "
            "so that only weight choices vary across runs. Graph embeddings are evaluated with a "
            "train-only graph build and per-test-case embedding inference."
        ),
        "neo4j_uri_present": bool(args.neo4j_uri),
    }

    write_outputs(run_dir, results_df, fold_df, configs, summary, metadata)

    print(f"Saved weight sensitivity results to: {run_dir}")
    print(results_df[["configuration_name", "holdout_roc_auc", "holdout_f1", "cv_roc_auc_mean"]].to_string(index=False))
    print(f"Overall sensitivity label: {summary['overall_sensitivity_label']}")


if __name__ == "__main__":
    main()
