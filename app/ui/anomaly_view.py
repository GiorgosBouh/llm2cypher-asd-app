from __future__ import annotations

import streamlit as st

from app.services.anomaly_service import AnomalyResult
from app.utils.text_summaries import anomaly_plain_language


def render_anomaly(result: AnomalyResult) -> None:
    """Render the secondary anomaly-support tab."""
    st.subheader("Anomaly / flags")
    st.metric("Anomaly score", f"{result.score:.3f}")
    st.metric("Unusual profile?", "Yes" if result.is_anomalous else "No")
    st.warning("Anomaly detection is supportive and does not determine the ASD prediction.")
    st.info(anomaly_plain_language(result.score, result.is_anomalous))
