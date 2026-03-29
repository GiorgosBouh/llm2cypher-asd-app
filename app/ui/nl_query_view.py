from __future__ import annotations

import pandas as pd
import streamlit as st

from app.services.nl_to_cypher_service import NLQueryResult


def render_nl_query(result: NLQueryResult) -> None:
    """Render natural-language-to-Cypher output."""
    if result.source:
        st.caption(f"Query source: {result.source}")
    if result.error:
        st.error(result.error)
    if result.cypher:
        st.code(result.cypher, language="cypher")
    st.info(result.explanation)
    if result.rows:
        st.dataframe(pd.DataFrame(result.rows), use_container_width=True)
