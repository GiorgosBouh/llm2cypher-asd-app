from __future__ import annotations

from typing import Optional

import pandas as pd
import plotly.express as px


def shap_bar_chart(shap_df: pd.DataFrame, title: str):
    """Render a Plotly bar chart for SHAP importances."""
    if shap_df.empty:
        return None
    plot_df = shap_df.head(15).copy()
    return px.bar(
        plot_df,
        x="importance",
        y="feature",
        orientation="h",
        title=title,
        labels={"importance": "Mean |SHAP|", "feature": "Feature"},
    )


def similar_cases_chart(similar_cases: pd.DataFrame) -> Optional[object]:
    """Create a small chart for similar-case similarity scores."""
    if similar_cases.empty:
        return None
    plot_df = similar_cases.copy()
    plot_df["case_label"] = plot_df["case_id"].astype(str)
    return px.bar(
        plot_df,
        x="similarity_score",
        y="case_label",
        color="label",
        orientation="h",
        title="Most Similar Cases",
        labels={"case_label": "Case ID", "similarity_score": "Similarity score"},
    )
