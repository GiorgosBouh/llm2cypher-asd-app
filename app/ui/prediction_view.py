from __future__ import annotations

import streamlit as st

from app.services.tabular_model_service import PredictionResult


def _predicted_label_probability(result: PredictionResult) -> float:
    return result.probability_yes if result.predicted_label == "Yes" else result.probability_no


def render_prediction(result: PredictionResult) -> None:
    """Render the primary prediction tab."""
    st.subheader("Primary tabular prediction")
    st.metric("Predicted ASD trait label", result.predicted_label)
    st.info(
        f"The primary tabular model classified this case as {result.predicted_label}. "
        f"The probability of the predicted label was {_predicted_label_probability(result):.1%}, "
        f"while the model-estimated probability for ASD traits = Yes was {result.probability_yes:.1%}. "
        f"The confidence summary was {result.confidence}."
    )
    st.warning(
        "This application is a decision-support tool. The tabular model provides the primary prediction; "
        "the graph is used for explanation, context, and exploration. It is not a standalone diagnostic instrument."
    )
