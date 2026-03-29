"""
Run reproducible baseline experiments for the NeuroCypher ASD repository.

This script compares:
1. Logistic Regression on raw tabular features
2. Random Forest on raw tabular features
3. XGBoost on raw tabular features
4. XGBoost on graph embeddings stored in Neo4j
5. XGBoost on combined raw tabular + graph features

Default behavior:
- Reads the local toddler ASD CSV in the repository root
- Uses a deterministic stratified 70/30 train/test split
- Uses 5-fold stratified cross-validation on the training split
- Excludes `Qchat-10-Score` to follow the project's leakage-avoidance policy
- Writes CSV, JSON, Markdown, and LaTeX-ready outputs under `results/`

Examples:
    python3 experiments_baselines.py
    python3 experiments_baselines.py --refresh-graph-embeddings
    python3 experiments_baselines.py --csv-path Toddler_Autism_dataset_July_2018_2.csv

Graph models:
- Require Neo4j connection details via environment variables
- Reuse the repository's existing label-free embedding generation approach
- Can optionally refresh graph embeddings with `--refresh-graph-embeddings`
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
from evaluation_safety import build_inductive_graph_embeddings, build_safety_audit, stratified_split
from env_utils import load_project_env
from neo4j import GraphDatabase
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier


RANDOM_STATE = 42
TEST_SIZE = 0.30
CV_FOLDS = 5

DEFAULT_CSV_NAME = "Toddler_Autism_dataset_July_2018_2.csv"
DEFAULT_RESULTS_DIR = "results"

TARGET_COL = "Class_ASD_Traits"
ID_COL = "Case_No"
LEAKAGE_COLS = {"Qchat-10-Score", TARGET_COL, ID_COL}

RAW_NUMERIC_FEATURES = [f"A{i}" for i in range(1, 11)] + ["Age_Mons"]
RAW_CATEGORICAL_FEATURES = [
    "Sex",
    "Ethnicity",
    "Jaundice",
    "Family_mem_with_ASD",
    "Who_completed_the_test",
]

REFERENCE_BASELINE_NOTE = (
    "Reference baseline only: the published ASD label is deterministically derived from questionnaire answers, "
    "so raw-questionnaire performance should not be interpreted as independent generalization evidence."
)

XGBOOST_TUNING_CANDIDATES: Dict[str, List[Dict[str, Any]]] = {
    "raw_tabular_reference": [
        {"n_estimators": 300, "max_depth": 3, "learning_rate": 0.05, "subsample": 0.9, "colsample_bytree": 0.9, "min_child_weight": 1, "reg_lambda": 1.0},
        {"n_estimators": 500, "max_depth": 4, "learning_rate": 0.03, "subsample": 0.9, "colsample_bytree": 0.8, "min_child_weight": 1, "reg_lambda": 1.0},
    ],
    "graph_embedding": [
        {"n_estimators": 200, "max_depth": 2, "learning_rate": 0.03, "subsample": 0.8, "colsample_bytree": 0.6, "min_child_weight": 8, "reg_lambda": 3.0},
        {"n_estimators": 300, "max_depth": 3, "learning_rate": 0.03, "subsample": 0.8, "colsample_bytree": 0.7, "min_child_weight": 6, "reg_lambda": 3.0},
        {"n_estimators": 500, "max_depth": 2, "learning_rate": 0.02, "subsample": 0.7, "colsample_bytree": 0.6, "min_child_weight": 10, "reg_lambda": 5.0},
        {"n_estimators": 250, "max_depth": 4, "learning_rate": 0.05, "subsample": 0.9, "colsample_bytree": 0.7, "min_child_weight": 4, "reg_lambda": 2.0},
    ],
    "raw_plus_graph": [
        {"n_estimators": 300, "max_depth": 3, "learning_rate": 0.05, "subsample": 0.9, "colsample_bytree": 0.8, "min_child_weight": 1, "reg_lambda": 1.0},
        {"n_estimators": 500, "max_depth": 4, "learning_rate": 0.03, "subsample": 0.9, "colsample_bytree": 0.7, "min_child_weight": 2, "reg_lambda": 1.5},
        {"n_estimators": 700, "max_depth": 3, "learning_rate": 0.02, "subsample": 0.8, "colsample_bytree": 0.7, "min_child_weight": 2, "reg_lambda": 2.0},
    ],
}


@dataclass
class ExperimentResult:
    model_family: str
    feature_type: str
    rows_used: int
    holdout_roc_auc: float
    holdout_f1: float
    holdout_precision: float
    holdout_recall: float
    holdout_ap: float
    cv_roc_auc_mean: float
    cv_roc_auc_std: float
    cv_f1_mean: float
    cv_f1_std: float
    cv_precision_mean: float
    cv_precision_std: float
    cv_recall_mean: float
    cv_recall_std: float
    cv_ap_mean: float
    cv_ap_std: float
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return self.__dict__.copy()


def make_one_hot_encoder() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run baseline experiments for NeuroCypher ASD.")
    parser.add_argument(
        "--csv-path",
        default=str(Path(__file__).resolve().parent / DEFAULT_CSV_NAME),
        help="Path to the labeled toddler ASD CSV.",
    )
    parser.add_argument(
        "--results-dir",
        default=DEFAULT_RESULTS_DIR,
        help="Base directory where results subfolders will be created.",
    )
    parser.add_argument(
        "--refresh-graph-embeddings",
        action="store_true",
        help="Regenerate label-free graph embeddings in Neo4j before running graph models.",
    )
    parser.add_argument(
        "--skip-graph-models",
        action="store_true",
        help="Run only raw tabular baselines.",
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
    run_name = datetime.utcnow().strftime("baseline_run_%Y%m%d_%H%M%S")
    run_dir = base_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def normalize_label(value: Any) -> int:
    return 1 if str(value).strip().lower() == "yes" else 0


def load_dataset(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path, sep=";", encoding="utf-8-sig")
    df.columns = [col.strip() for col in df.columns]

    numeric_cols = [ID_COL] + RAW_NUMERIC_FEATURES
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col].astype(str).str.replace(",", "."), errors="coerce")

    if ID_COL not in df.columns or TARGET_COL not in df.columns:
        raise ValueError(f"Dataset must contain '{ID_COL}' and '{TARGET_COL}'.")

    df = df.dropna(subset=[ID_COL, TARGET_COL]).copy()
    df[ID_COL] = df[ID_COL].astype(int)
    df[TARGET_COL] = df[TARGET_COL].astype(str).str.strip().str.capitalize()
    df["target"] = df[TARGET_COL].map(normalize_label)
    return df


def available_columns(df: pd.DataFrame, columns: Sequence[str]) -> List[str]:
    return [col for col in columns if col in df.columns]


def build_preprocessor(
    raw_numeric: Sequence[str],
    raw_categorical: Sequence[str],
    graph_columns: Sequence[str],
) -> ColumnTransformer:
    transformers: List[Tuple[str, Any, Sequence[str]]] = []

    if raw_numeric:
        transformers.append(
            (
                "num",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                list(raw_numeric),
            )
        )

    if raw_categorical:
        transformers.append(
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", make_one_hot_encoder()),
                    ]
                ),
                list(raw_categorical),
            )
        )

    if graph_columns:
        transformers.append(
            (
                "graph",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                    ]
                ),
                list(graph_columns),
            )
        )

    if not transformers:
        raise ValueError("At least one feature block must be provided.")

    return ColumnTransformer(transformers=transformers, remainder="drop", sparse_threshold=0.0)


def build_model(model_family: str, y_train: pd.Series, model_params: Optional[Dict[str, Any]] = None) -> Any:
    positives = int((y_train == 1).sum())
    negatives = int((y_train == 0).sum())
    scale_pos_weight = (negatives / positives) if positives else 1.0
    model_params = model_params or {}

    if model_family == "logistic_regression":
        return LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            solver="lbfgs",
        )
    if model_family == "random_forest":
        return RandomForestClassifier(
            n_estimators=400,
            random_state=RANDOM_STATE,
            class_weight="balanced",
            min_samples_leaf=2,
            n_jobs=-1,
        )
    if model_family == "xgboost":
        defaults = {
            "n_estimators": 300,
            "max_depth": 4,
            "learning_rate": 0.05,
            "subsample": 0.9,
            "colsample_bytree": 0.9,
            "min_child_weight": 1,
            "reg_lambda": 1.0,
        }
        defaults.update(model_params)
        return XGBClassifier(
            eval_metric="logloss",
            random_state=RANDOM_STATE,
            scale_pos_weight=scale_pos_weight,
            use_label_encoder=False,
            n_jobs=1,
            verbosity=0,
            **defaults,
        )
    raise ValueError(f"Unsupported model family: {model_family}")


def predicted_scores(model: Any, features: np.ndarray) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(features)[:, 1]
    if hasattr(model, "decision_function"):
        raw_scores = model.decision_function(features)
        return 1.0 / (1.0 + np.exp(-raw_scores))
    raise ValueError("Model does not expose predict_proba or decision_function.")


def compute_metrics(y_true: pd.Series, y_score: np.ndarray) -> Dict[str, float]:
    y_pred = (y_score >= 0.5).astype(int)
    return {
        "roc_auc": roc_auc_score(y_true, y_score),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "ap": average_precision_score(y_true, y_score),
    }


def run_cv(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    preprocessor: ColumnTransformer,
    model_family: str,
    model_params: Optional[Dict[str, Any]] = None,
) -> Tuple[Dict[str, float], List[Dict[str, float]]]:
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    fold_results: List[Dict[str, float]] = []

    for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X_train, y_train), start=1):
        X_fold_train = X_train.iloc[train_idx]
        X_fold_val = X_train.iloc[val_idx]
        y_fold_train = y_train.iloc[train_idx]
        y_fold_val = y_train.iloc[val_idx]

        fold_preprocessor = clone(preprocessor)
        X_fold_train_enc = fold_preprocessor.fit_transform(X_fold_train)
        X_fold_val_enc = fold_preprocessor.transform(X_fold_val)

        model = build_model(model_family, y_fold_train, model_params=model_params)
        model.fit(X_fold_train_enc, y_fold_train)
        y_score = predicted_scores(model, X_fold_val_enc)
        metrics = compute_metrics(y_fold_val, y_score)
        metrics["fold"] = fold_idx
        fold_results.append(metrics)

    summary = {
        "cv_roc_auc_mean": float(np.mean([row["roc_auc"] for row in fold_results])),
        "cv_roc_auc_std": float(np.std([row["roc_auc"] for row in fold_results], ddof=0)),
        "cv_f1_mean": float(np.mean([row["f1"] for row in fold_results])),
        "cv_f1_std": float(np.std([row["f1"] for row in fold_results], ddof=0)),
        "cv_precision_mean": float(np.mean([row["precision"] for row in fold_results])),
        "cv_precision_std": float(np.std([row["precision"] for row in fold_results], ddof=0)),
        "cv_recall_mean": float(np.mean([row["recall"] for row in fold_results])),
        "cv_recall_std": float(np.std([row["recall"] for row in fold_results], ddof=0)),
        "cv_ap_mean": float(np.mean([row["ap"] for row in fold_results])),
        "cv_ap_std": float(np.std([row["ap"] for row in fold_results], ddof=0)),
    }
    return summary, fold_results


def tune_xgboost_config(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    preprocessor: ColumnTransformer,
    feature_type: str,
) -> Tuple[Optional[Dict[str, Any]], Dict[str, float]]:
    candidates = XGBOOST_TUNING_CANDIDATES.get(feature_type, [])
    if not candidates:
        return None, {}

    best_params: Optional[Dict[str, Any]] = None
    best_summary: Optional[Dict[str, float]] = None

    for candidate in candidates:
        summary, _ = run_cv(X_train, y_train, preprocessor, "xgboost", model_params=candidate)
        if best_summary is None:
            best_params = candidate
            best_summary = summary
            continue
        current_key = (summary["cv_roc_auc_mean"], summary["cv_ap_mean"], summary["cv_f1_mean"])
        best_key = (best_summary["cv_roc_auc_mean"], best_summary["cv_ap_mean"], best_summary["cv_f1_mean"])
        if current_key > best_key:
            best_params = candidate
            best_summary = summary

    return best_params, best_summary or {}


def evaluate_experiment(
    dataset: pd.DataFrame,
    feature_columns: Sequence[str],
    preprocessor: ColumnTransformer,
    model_family: str,
    feature_type: str,
    notes: str = "",
) -> Tuple[ExperimentResult, List[Dict[str, float]]]:
    X = dataset.loc[:, list(feature_columns)].copy()
    y = dataset["target"].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    cv_summary, fold_rows = run_cv(X_train, y_train, preprocessor, model_family)

    final_preprocessor = clone(preprocessor)
    X_train_enc = final_preprocessor.fit_transform(X_train)
    X_test_enc = final_preprocessor.transform(X_test)

    final_model = build_model(model_family, y_train)
    final_model.fit(X_train_enc, y_train)
    holdout_scores = predicted_scores(final_model, X_test_enc)
    holdout_metrics = compute_metrics(y_test, holdout_scores)

    result = ExperimentResult(
        model_family=model_family,
        feature_type=feature_type,
        rows_used=len(dataset),
        holdout_roc_auc=float(holdout_metrics["roc_auc"]),
        holdout_f1=float(holdout_metrics["f1"]),
        holdout_precision=float(holdout_metrics["precision"]),
        holdout_recall=float(holdout_metrics["recall"]),
        holdout_ap=float(holdout_metrics["ap"]),
        cv_roc_auc_mean=cv_summary["cv_roc_auc_mean"],
        cv_roc_auc_std=cv_summary["cv_roc_auc_std"],
        cv_f1_mean=cv_summary["cv_f1_mean"],
        cv_f1_std=cv_summary["cv_f1_std"],
        cv_precision_mean=cv_summary["cv_precision_mean"],
        cv_precision_std=cv_summary["cv_precision_std"],
        cv_recall_mean=cv_summary["cv_recall_mean"],
        cv_recall_std=cv_summary["cv_recall_std"],
        cv_ap_mean=cv_summary["cv_ap_mean"],
        cv_ap_std=cv_summary["cv_ap_std"],
        notes=notes,
    )

    for row in fold_rows:
        row["model_family"] = model_family
        row["feature_type"] = feature_type

    return result, fold_rows


def evaluate_fixed_split_experiment(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_columns: Sequence[str],
    preprocessor: ColumnTransformer,
    model_family: str,
    feature_type: str,
    notes: str = "",
    tune_xgboost: bool = False,
) -> Tuple[ExperimentResult, List[Dict[str, float]]]:
    X_train = train_df.loc[:, list(feature_columns)].copy()
    y_train = train_df["target"].copy()
    X_test = test_df.loc[:, list(feature_columns)].copy()
    y_test = test_df["target"].copy()

    model_params: Optional[Dict[str, Any]] = None
    tuning_note = ""
    if model_family == "xgboost" and tune_xgboost:
        model_params, tuned_summary = tune_xgboost_config(X_train, y_train, preprocessor, feature_type)
        if model_params:
            tuning_note = f" CV-guided XGBoost params: {json.dumps(model_params, sort_keys=True)}."
        cv_summary, fold_rows = run_cv(X_train, y_train, preprocessor, model_family, model_params=model_params)
    else:
        cv_summary, fold_rows = run_cv(X_train, y_train, preprocessor, model_family, model_params=model_params)

    final_preprocessor = clone(preprocessor)
    X_train_enc = final_preprocessor.fit_transform(X_train)
    X_test_enc = final_preprocessor.transform(X_test)

    final_model = build_model(model_family, y_train, model_params=model_params)
    final_model.fit(X_train_enc, y_train)
    holdout_scores = predicted_scores(final_model, X_test_enc)
    holdout_metrics = compute_metrics(y_test, holdout_scores)

    result = ExperimentResult(
        model_family=model_family,
        feature_type=feature_type,
        rows_used=len(train_df) + len(test_df),
        holdout_roc_auc=float(holdout_metrics["roc_auc"]),
        holdout_f1=float(holdout_metrics["f1"]),
        holdout_precision=float(holdout_metrics["precision"]),
        holdout_recall=float(holdout_metrics["recall"]),
        holdout_ap=float(holdout_metrics["ap"]),
        cv_roc_auc_mean=cv_summary["cv_roc_auc_mean"],
        cv_roc_auc_std=cv_summary["cv_roc_auc_std"],
        cv_f1_mean=cv_summary["cv_f1_mean"],
        cv_f1_std=cv_summary["cv_f1_std"],
        cv_precision_mean=cv_summary["cv_precision_mean"],
        cv_precision_std=cv_summary["cv_precision_std"],
        cv_recall_mean=cv_summary["cv_recall_mean"],
        cv_recall_std=cv_summary["cv_recall_std"],
        cv_ap_mean=cv_summary["cv_ap_mean"],
        cv_ap_std=cv_summary["cv_ap_std"],
        notes=(notes + tuning_note).strip(),
    )

    for row in fold_rows:
        row["model_family"] = model_family
        row["feature_type"] = feature_type

    return result, fold_rows


def get_neo4j_driver(args: argparse.Namespace):
    if not args.neo4j_uri or not args.neo4j_user or not args.neo4j_password:
        return None
    return GraphDatabase.driver(args.neo4j_uri, auth=(args.neo4j_user, args.neo4j_password))


def remove_screened_for_labels(driver: Any, database: str) -> None:
    with driver.session(database=database) as session:
        session.run(
            """
            MATCH (:Case)-[r:SCREENED_FOR]->(:ASD_Trait)
            DELETE r
            """
        )


def reinsert_labels_from_csv(driver: Any, database: str, df: pd.DataFrame) -> None:
    rows = []
    for _, row in df.iterrows():
        label = str(row[TARGET_COL]).strip().lower()
        if label in {"yes", "no"}:
            rows.append({"case_id": int(row[ID_COL]), "label": label.capitalize()})

    with driver.session(database=database) as session:
        session.run("MERGE (:ASD_Trait {label: 'Yes'})")
        session.run("MERGE (:ASD_Trait {label: 'No'})")
        session.run(
            """
            UNWIND $rows AS row
            MATCH (c:Case {id: row.case_id})
            MATCH (t:ASD_Trait {label: row.label})
            MERGE (c)-[:SCREENED_FOR]->(t)
            """,
            rows=rows,
        )


def refresh_graph_embeddings(script_path: Path, args: argparse.Namespace, df: pd.DataFrame) -> None:
    driver = get_neo4j_driver(args)
    if driver is None:
        raise ValueError("Neo4j credentials are required to refresh graph embeddings.")

    env = os.environ.copy()
    env.update(
        {
            "NEO4J_URI": args.neo4j_uri,
            "NEO4J_USER": args.neo4j_user,
            "NEO4J_PASSWORD": args.neo4j_password,
            "NEO4J_DB": args.neo4j_database,
        }
    )

    try:
        remove_screened_for_labels(driver, args.neo4j_database)
        result = subprocess.run(
            [sys.executable, str(script_path), "--no-labels"],
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            stderr = result.stderr.strip() or "No stderr was returned."
            raise RuntimeError(f"Embedding refresh failed: {stderr}")
    finally:
        reinsert_labels_from_csv(driver, args.neo4j_database, df)
        driver.close()


def fetch_graph_embeddings(args: argparse.Namespace, case_ids: Sequence[int]) -> pd.DataFrame:
    driver = get_neo4j_driver(args)
    if driver is None:
        raise ValueError("Neo4j credentials are required for graph-based experiments.")

    case_id_rows = [{"case_id": int(case_id)} for case_id in case_ids]
    with driver.session(database=args.neo4j_database) as session:
        records = session.run(
            """
            UNWIND $rows AS row
            MATCH (c:Case {id: row.case_id})
            RETURN c.id AS case_id, c.embedding AS embedding
            """,
            rows=case_id_rows,
        )
        rows = [record.data() for record in records]

    driver.close()

    embeddings = []
    for row in rows:
        embedding = row.get("embedding")
        if embedding is None:
            continue
        if len(embedding) != 128:
            continue
        embeddings.append({"Case_No": int(row["case_id"]), **{f"emb_{i}": float(value) for i, value in enumerate(embedding)}})

    if not embeddings:
        raise ValueError("No graph embeddings were found in Neo4j for the requested cases.")

    return pd.DataFrame(embeddings)


def write_outputs(
    run_dir: Path,
    summary_rows: Sequence[ExperimentResult],
    fold_rows: Sequence[Dict[str, Any]],
    metadata: Dict[str, Any],
) -> None:
    summary_df = pd.DataFrame([row.to_dict() for row in summary_rows])
    fold_df = pd.DataFrame(fold_rows)

    if not summary_df.empty:
        sort_cols = ["holdout_roc_auc", "holdout_f1", "cv_roc_auc_mean"]
        summary_df = summary_df.sort_values(by=sort_cols, ascending=False)

    summary_df.to_csv(run_dir / "comparison_table.csv", index=False)
    fold_df.to_csv(run_dir / "cv_fold_metrics.csv", index=False)

    with open(run_dir / "comparison_table.json", "w", encoding="utf-8") as f:
        json.dump(summary_df.to_dict(orient="records"), f, indent=2)

    with open(run_dir / "run_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    markdown = summary_df.to_markdown(index=False) if not summary_df.empty else "No results were produced."
    with open(run_dir / "comparison_table.md", "w", encoding="utf-8") as f:
        f.write("# Baseline Comparison\n\n")
        f.write(markdown)
        f.write("\n")

    latex = summary_df.to_latex(index=False, float_format=lambda x: f"{x:.4f}") if not summary_df.empty else "% No results were produced.\n"
    with open(run_dir / "comparison_table.tex", "w", encoding="utf-8") as f:
        f.write(latex)


def main() -> None:
    load_project_env()
    np.random.seed(RANDOM_STATE)

    args = parse_args()
    repo_root = Path(__file__).resolve().parent
    csv_path = Path(args.csv_path).resolve()
    results_dir = ensure_results_dir(Path(args.results_dir).resolve())

    df = load_dataset(csv_path)
    raw_numeric = available_columns(df, RAW_NUMERIC_FEATURES)
    raw_categorical = available_columns(df, RAW_CATEGORICAL_FEATURES)

    if not raw_numeric and not raw_categorical:
        raise ValueError("No raw tabular features were found in the dataset.")

    raw_feature_columns = raw_numeric + raw_categorical
    df, safety_audit = build_safety_audit(df, raw_feature_columns)
    train_df, test_df = stratified_split(df)
    raw_preprocessor = build_preprocessor(raw_numeric, raw_categorical, [])

    summary_rows: List[ExperimentResult] = []
    fold_rows: List[Dict[str, Any]] = []
    skipped_experiments: List[str] = []

    raw_experiments = [("xgboost", "raw_tabular_reference")]

    raw_status_note = "Reference tabular baseline completed."
    for model_family, feature_type in raw_experiments:
        raw_notes = "Deduplicated before split; preprocessing and CV fit only on training data."
        if safety_audit.deterministic_target_from_answers:
            raw_notes = f"{raw_notes} {REFERENCE_BASELINE_NOTE}"
        result, folds = evaluate_fixed_split_experiment(
            train_df=train_df,
            test_df=test_df,
            feature_columns=raw_feature_columns,
            preprocessor=raw_preprocessor,
            model_family=model_family,
            feature_type=feature_type,
            notes=raw_notes,
            tune_xgboost=True,
        )
        summary_rows.append(result)
        fold_rows.extend(folds)

    graph_status_note = "Graph experiments were skipped."
    if not args.skip_graph_models:
        try:
            merged_train, merged_test, graph_meta = build_inductive_graph_embeddings(
                repo_root=repo_root,
                args=args,
                train_df=train_df,
                test_df=test_df,
                csv_columns=[column for column in df.columns if column != "target"],
            )
            graph_columns = [col for col in merged_train.columns if col.startswith("emb_")]

            if not graph_columns:
                raise ValueError("Embedding columns were not found after inductive graph inference.")

            graph_preprocessor = build_preprocessor([], [], graph_columns)
            graph_result, graph_folds = evaluate_fixed_split_experiment(
                train_df=merged_train,
                test_df=merged_test,
                feature_columns=graph_columns,
                preprocessor=graph_preprocessor,
                model_family="xgboost",
                feature_type="graph_embedding",
                notes="Leakage-safe protocol: deduplicate, split first, build graph on training set only, infer unlabeled test embeddings from the train graph.",
                tune_xgboost=True,
            )
            summary_rows.append(graph_result)
            fold_rows.extend(graph_folds)

            combined_columns = raw_feature_columns + graph_columns
            combined_preprocessor = build_preprocessor(raw_numeric, raw_categorical, graph_columns)
            combined_notes = "Leakage-safe protocol: train-only graph fit plus unlabeled test embedding refresh."
            if safety_audit.deterministic_target_from_answers:
                combined_notes = f"{combined_notes} {REFERENCE_BASELINE_NOTE}"
            combined_result, combined_folds = evaluate_fixed_split_experiment(
                train_df=merged_train,
                test_df=merged_test,
                feature_columns=combined_columns,
                preprocessor=combined_preprocessor,
                model_family="xgboost",
                feature_type="raw_plus_graph",
                notes=combined_notes,
                tune_xgboost=True,
            )
            summary_rows.append(combined_result)
            fold_rows.extend(combined_folds)
            graph_status_note = (
                "Graph and hybrid experiments completed with inductive embeddings. "
                f"Train rows with embeddings: {graph_meta['train_rows_with_embeddings']}; "
                f"test rows with embeddings: {graph_meta['test_rows_with_embeddings']}."
            )
        except Exception as exc:
            graph_status_note = f"Graph experiments were skipped: {exc}"

    metadata = {
        "generated_at_utc": datetime.utcnow().isoformat() + "Z",
        "csv_path": str(csv_path),
        "results_dir": str(results_dir),
        "random_state": RANDOM_STATE,
        "test_size": TEST_SIZE,
        "cv_folds": CV_FOLDS,
        "raw_numeric_features": raw_numeric,
        "raw_categorical_features": raw_categorical,
        "excluded_for_leakage": sorted(LEAKAGE_COLS),
        "duplicate_rows_removed_before_split": safety_audit.duplicate_rows_removed,
        "deterministic_target_from_answers": safety_audit.deterministic_target_from_answers,
        "deterministic_target_note": safety_audit.deterministic_rule_note,
        "raw_status": raw_status_note,
        "graph_status": graph_status_note,
        "refresh_graph_embeddings": bool(args.refresh_graph_embeddings),
        "skip_graph_models": bool(args.skip_graph_models),
        "skipped_experiments": skipped_experiments,
    }

    write_outputs(results_dir, summary_rows, fold_rows, metadata)

    print(f"Saved results to: {results_dir}")
    print(raw_status_note)
    print(graph_status_note)
    if summary_rows:
        summary_df = pd.DataFrame([row.to_dict() for row in summary_rows])
        print(summary_df[["model_family", "feature_type", "holdout_roc_auc", "holdout_f1", "cv_roc_auc_mean"]].to_string(index=False))


if __name__ == "__main__":
    main()
