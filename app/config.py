from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from env_utils import load_project_env

load_project_env()

ROOT_DIR = Path(__file__).resolve().parent.parent
ARTIFACT_DIR = ROOT_DIR / "artifacts"
ARTIFACT_DIR.mkdir(exist_ok=True)

DEFAULT_DATASET_PATH = ROOT_DIR / "Toddler_Autism_dataset_July_2018_2.csv"
TABULAR_MODEL_ARTIFACT = ARTIFACT_DIR / "tabular_model_bundle.pkl"
ANOMALY_MODEL_ARTIFACT = ARTIFACT_DIR / "anomaly_model_bundle.pkl"

CASE_ID_COLUMN = "Case_No"
TARGET_COLUMN = "Class_ASD_Traits"
LEAKY_COLUMNS = {"Qchat-10-Score"}
QUESTION_COLUMNS = [f"A{i}" for i in range(1, 11)]
NUMERIC_COLUMNS = QUESTION_COLUMNS + ["Age_Mons"]
CATEGORICAL_COLUMNS = [
    "Sex",
    "Ethnicity",
    "Jaundice",
    "Family_mem_with_ASD",
    "Who_completed_the_test",
]
FEATURE_COLUMNS = NUMERIC_COLUMNS + CATEGORICAL_COLUMNS
OPTIONAL_COLUMNS = [CASE_ID_COLUMN, TARGET_COLUMN, *LEAKY_COLUMNS]

NEO4J_URI = os.getenv("NEO4J_URI") or "neo4j+s://1f5f8a14.databases.neo4j.io"
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "")
NEO4J_DB = os.getenv("NEO4J_DB", "neo4j")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")


@dataclass(frozen=True)
class AppConfig:
    """Central application settings."""

    app_title: str = "NeuroCypher ASD"
    random_state: int = 42
    test_size: float = 0.30
    cv_folds: int = 5
    similar_cases_k: int = 5
    shap_background_size: int = 200
    confidence_high: float = 0.85
    confidence_medium: float = 0.65
    anomaly_contamination: float = 0.08
    read_only_cypher: bool = True
