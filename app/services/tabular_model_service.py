from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from sklearn.pipeline import Pipeline

from app.config import (
    AppConfig,
    FEATURE_COLUMNS,
    TABULAR_MODEL_ARTIFACT,
)
from app.data.preprocessing import build_tabular_preprocessor
from app.data.schema_validation import feature_dataframe_from_case, training_columns
from app.utils.logging_utils import get_logger
from app.utils.text_summaries import confidence_summary

try:
    from xgboost import XGBClassifier
except Exception:  # pragma: no cover - runtime fallback
    XGBClassifier = None

logger = get_logger(__name__)


@dataclass
class PredictionResult:
    predicted_label: str
    probability_yes: float
    probability_no: float
    confidence: str


@dataclass
class ModelBundle:
    pipeline: Pipeline
    feature_columns: list[str]
    model_name: str
    metrics: dict[str, float]
    train_sample: pd.DataFrame


class TabularModelService:
    """Owns the primary ASD prediction pipeline based on raw tabular features only."""

    def __init__(self, config: AppConfig, artifact_path: Path | None = None):
        self.config = config
        self.artifact_path = artifact_path or TABULAR_MODEL_ARTIFACT

    def _build_estimator(self):
        if XGBClassifier is not None:
            logger.info("Using XGBoost as primary tabular predictor.")
            return XGBClassifier(
                n_estimators=400,
                max_depth=4,
                learning_rate=0.03,
                subsample=0.9,
                colsample_bytree=0.8,
                reg_lambda=1.0,
                random_state=self.config.random_state,
                eval_metric="logloss",
            )

        logger.warning("XGBoost unavailable; falling back to RandomForestClassifier.")
        return RandomForestClassifier(
            n_estimators=300,
            random_state=self.config.random_state,
            class_weight="balanced_subsample",
        )

    def build_pipeline(self) -> Pipeline:
        return Pipeline(
            steps=[
                ("preprocessor", build_tabular_preprocessor()),
                ("model", self._build_estimator()),
            ]
        )

    def train(self, df: pd.DataFrame) -> ModelBundle:
        prepared = training_columns(df)
        if prepared.empty:
            raise ValueError("Training dataframe is empty after preprocessing.")
        if "Class_ASD_Traits" not in prepared.columns:
            raise ValueError("Training dataframe is missing the target column.")

        deduplicated = prepared.drop_duplicates().reset_index(drop=True)
        X = deduplicated[FEATURE_COLUMNS].copy()
        y = deduplicated["Class_ASD_Traits"].astype(int)

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=self.config.test_size,
            random_state=self.config.random_state,
            stratify=y,
        )

        pipeline = self.build_pipeline()
        pipeline.fit(X_train, y_train)

        y_proba = pipeline.predict_proba(X_test)[:, 1]
        y_pred = (y_proba >= 0.5).astype(int)

        cv = StratifiedKFold(
            n_splits=self.config.cv_folds,
            shuffle=True,
            random_state=self.config.random_state,
        )
        cv_proba = cross_val_predict(pipeline, X_train, y_train, cv=cv, method="predict_proba")[:, 1]

        metrics = {
            "holdout_roc_auc": float(roc_auc_score(y_test, y_proba)),
            "holdout_f1": float(f1_score(y_test, y_pred)),
            "holdout_precision": float(precision_score(y_test, y_pred)),
            "holdout_recall": float(recall_score(y_test, y_pred)),
            "holdout_ap": float(average_precision_score(y_test, y_proba)),
            "cv_roc_auc": float(roc_auc_score(y_train, cv_proba)),
        }

        bundle = ModelBundle(
            pipeline=pipeline,
            feature_columns=list(FEATURE_COLUMNS),
            model_name=type(pipeline.named_steps["model"]).__name__,
            metrics=metrics,
            train_sample=X_train.head(self.config.shap_background_size).copy(),
        )
        self.save(bundle)
        return bundle

    def save(self, bundle: ModelBundle) -> None:
        with self.artifact_path.open("wb") as file_obj:
            pickle.dump(bundle, file_obj)
        logger.info("Saved tabular model bundle to %s", self.artifact_path)

    def load(self) -> ModelBundle:
        if not self.artifact_path.exists():
            raise FileNotFoundError(
                f"Missing tabular model artifact at {self.artifact_path}. Train the model from Admin / Rebuild first."
            )
        with self.artifact_path.open("rb") as file_obj:
            return pickle.load(file_obj)

    def predict_case(self, case_data: dict[str, Any]) -> PredictionResult:
        bundle = self.load()
        features_df = feature_dataframe_from_case(case_data)
        probability_yes = float(bundle.pipeline.predict_proba(features_df)[0, 1])
        probability_no = 1.0 - probability_yes
        predicted_label = "Yes" if probability_yes >= 0.5 else "No"
        confidence = confidence_summary(
            probability_yes,
            high=self.config.confidence_high,
            medium=self.config.confidence_medium,
        )
        return PredictionResult(
            predicted_label=predicted_label,
            probability_yes=probability_yes,
            probability_no=probability_no,
            confidence=confidence,
        )
