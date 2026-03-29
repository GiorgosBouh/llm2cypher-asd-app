from __future__ import annotations

import argparse

import pandas as pd

from app.config import AppConfig, DEFAULT_DATASET_PATH
from app.services.anomaly_service import AnomalyService
from app.services.tabular_model_service import TabularModelService
from app.utils.logging_utils import get_logger

logger = get_logger(__name__)


def train_from_csv(csv_path: str) -> None:
    """Train and persist the primary tabular model and the secondary anomaly model."""
    df = pd.read_csv(csv_path, sep=";", encoding="utf-8-sig")
    config = AppConfig()

    tabular_service = TabularModelService(config)
    bundle = tabular_service.train(df)
    logger.info("Saved tabular model with metrics: %s", bundle.metrics)

    anomaly_service = AnomalyService(config)
    anomaly_service.train(df)
    logger.info("Saved anomaly support model.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the tabular-first NeuroCypher predictor.")
    parser.add_argument("--csv", default=str(DEFAULT_DATASET_PATH), help="Path to the training CSV file.")
    args = parser.parse_args()
    train_from_csv(args.csv)


if __name__ == "__main__":
    main()
