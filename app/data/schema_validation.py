from __future__ import annotations

import io
from typing import Dict, Iterable, Tuple

import pandas as pd

from app.config import CASE_ID_COLUMN, FEATURE_COLUMNS, OPTIONAL_COLUMNS, TARGET_COLUMN


def _normalize_columns(columns: Iterable[str]) -> list[str]:
    return [str(column).strip().replace("\r", "") for column in columns]


def read_csv_bytes(data: bytes) -> pd.DataFrame:
    """Read CSV bytes using the dataset delimiter conventions."""
    return pd.read_csv(io.BytesIO(data), sep=";", encoding="utf-8-sig")


def standardize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize column names and primitive string values."""
    normalized = df.copy()
    normalized.columns = _normalize_columns(normalized.columns)
    for column in normalized.columns:
        if normalized[column].dtype == object:
            normalized[column] = normalized[column].astype(str).str.strip()
    return normalized


def validate_case_dataframe(df: pd.DataFrame) -> Tuple[bool, list[str]]:
    """Validate that a dataframe can be used for single-case inference."""
    missing = [column for column in FEATURE_COLUMNS if column not in df.columns]
    return (len(missing) == 0, missing)


def coerce_feature_types(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce feature dtypes safely for training and inference."""
    prepared = standardize_dataframe(df)
    for column in FEATURE_COLUMNS:
        if column not in prepared.columns:
            prepared[column] = pd.NA

    for column in [*FEATURE_COLUMNS, CASE_ID_COLUMN]:
        if column in prepared.columns and (column.startswith("A") or column in {"Age_Mons", CASE_ID_COLUMN}):
            prepared[column] = pd.to_numeric(prepared[column], errors="coerce")

    if TARGET_COLUMN in prepared.columns:
        prepared[TARGET_COLUMN] = (
            prepared[TARGET_COLUMN]
            .astype(str)
            .str.strip()
            .str.lower()
            .map({"yes": 1, "no": 0})
        )

    return prepared


def extract_single_case(df: pd.DataFrame) -> Dict[str, object]:
    """Extract a single-case dict in canonical feature order."""
    if len(df) != 1:
        raise ValueError("Expected exactly one row for single-case inference.")
    valid, missing = validate_case_dataframe(df)
    if not valid:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")

    standardized = coerce_feature_types(df)
    record = standardized.iloc[0].to_dict()
    return {column: record.get(column) for column in [*FEATURE_COLUMNS, *OPTIONAL_COLUMNS] if column in record}


def training_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Return only the feature columns plus target and id when present."""
    standardized = standardize_dataframe(df)
    available = [column for column in [CASE_ID_COLUMN, *FEATURE_COLUMNS, TARGET_COLUMN] if column in standardized.columns]
    return coerce_feature_types(standardized[available])


def feature_dataframe_from_case(case_data: Dict[str, object]) -> pd.DataFrame:
    """Create a one-row feature dataframe with stable column ordering."""
    row = {column: case_data.get(column) for column in FEATURE_COLUMNS}
    return pd.DataFrame([row], columns=FEATURE_COLUMNS)
