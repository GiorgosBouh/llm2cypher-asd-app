from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.pipeline import Pipeline

from app.config import ANOMALY_MODEL_ARTIFACT, AppConfig
from app.data.preprocessing import build_tabular_preprocessor
from app.data.schema_validation import feature_dataframe_from_case, training_columns
from app.utils.logging_utils import get_logger

logger = get_logger(__name__)


@dataclass
class AnomalyBundle:
    pipeline: Pipeline


@dataclass
class AnomalyResult:
    score: float
    is_anomalous: bool


class AnomalyService:
    """Secondary anomaly detection based on tabular features only."""

    def __init__(self, config: AppConfig, artifact_path: Path | None = None):
        self.config = config
        self.artifact_path = artifact_path or ANOMALY_MODEL_ARTIFACT

    def train(self, df: pd.DataFrame) -> AnomalyBundle:
        prepared = training_columns(df)
        X = prepared.drop(columns=["Class_ASD_Traits"], errors="ignore")
        pipeline = Pipeline(
            steps=[
                ("preprocessor", build_tabular_preprocessor()),
                (
                    "model",
                    IsolationForest(
                        contamination=self.config.anomaly_contamination,
                        random_state=self.config.random_state,
                        n_estimators=300,
                    ),
                ),
            ]
        )
        pipeline.fit(X)
        bundle = AnomalyBundle(pipeline=pipeline)
        with self.artifact_path.open("wb") as file_obj:
            pickle.dump(bundle, file_obj)
        return bundle

    def load(self) -> AnomalyBundle:
        if not self.artifact_path.exists():
            raise FileNotFoundError(
                f"Missing anomaly artifact at {self.artifact_path}. Train support models from Admin / Rebuild first."
            )
        with self.artifact_path.open("rb") as file_obj:
            return pickle.load(file_obj)

    def score_case(self, case_data: dict[str, object]) -> AnomalyResult:
        bundle = self.load()
        case_df = feature_dataframe_from_case(case_data)
        score = float(bundle.pipeline.decision_function(case_df)[0])
        flag = int(bundle.pipeline.predict(case_df)[0]) == -1
        return AnomalyResult(score=score, is_anomalous=flag)
