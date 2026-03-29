from __future__ import annotations

import pandas as pd
import streamlit as st

from app.services.graph_context_service import GraphContextResult
from app.utils.plotting import similar_cases_chart

FRIENDLY_BEHAVIOR = {
    "A1": "response to name",
    "A2": "eye contact",
    "A3": "pointing to ask for something",
    "A4": "pointing to share interest",
    "A5": "pretend play",
    "A6": "following another person's gaze",
    "A7": "trying to comfort others",
    "A8": "early word use",
    "A9": "simple gestures",
    "A10": "staring without clear purpose",
}

FRIENDLY_DEMOGRAPHIC = {
    "Family_mem_with_ASD": "family history of ASD",
    "Jaundice": "history of jaundice",
    "Ethnicity": "ethnicity",
    "Sex": "sex",
}


def _friendly_behavior(value: str) -> str:
    return FRIENDLY_BEHAVIOR.get(str(value), str(value))


def _friendly_demographic(value: str) -> str:
    return FRIENDLY_DEMOGRAPHIC.get(str(value), str(value))


def _render_key_insights(context: GraphContextResult) -> None:
    insight_lines: list[str] = []
    if not context.similar_cases.empty:
        top_case = context.similar_cases.iloc[0]
        insight_lines.append(
            f"The closest similar case in the dataset is case #{int(top_case['case_id'])} "
            f"with a similarity score of {float(top_case['similarity_score']):.1f}."
        )
    if not context.shared_behaviors.empty:
        top_behavior = context.shared_behaviors.iloc[0]
        insight_lines.append(
            f"The strongest shared answer is {_friendly_behavior(top_behavior['question'])} "
            f"(seen in {int(top_behavior['supporting_cases'])} similar cases)."
        )
    if not context.shared_demographics.empty:
        top_demo = context.shared_demographics.iloc[0]
        insight_lines.append(
            f"The most common shared background detail is {_friendly_demographic(top_demo['demographic_type'])} = "
            f"{top_demo['demographic_value']} (seen in {int(top_demo['supporting_cases'])} similar cases)."
        )

    if insight_lines:
        st.markdown("**Simple summary**")
        for line in insight_lines[:3]:
            st.write(f"- {line}")


def _render_neighborhood_snapshot(subgraph: dict) -> None:
    edges = subgraph.get("edges", [])
    if not edges:
        st.write("No neighborhood edges were retrieved.")
        return

    edge_df = pd.DataFrame(edges)
    relationship_view = edge_df[["relationship", "target"]].rename(columns={"target": "connected_to"})
    relationship_view["relationship"] = relationship_view["relationship"].replace(
        {
            "HAS_ANSWER": "Answer",
            "HAS_DEMOGRAPHIC": "Background detail",
            "SUBMITTED_BY": "Completed by",
        }
    )
    relationship_view = relationship_view.rename(
        columns={"relationship": "Type of link", "connected_to": "Connected item"}
    )
    st.markdown("**What this case is linked to**")
    st.dataframe(relationship_view, use_container_width=True, hide_index=True)


def render_graph_context(context: GraphContextResult, subgraph: dict) -> None:
    """Render graph-backed contextual exploration."""
    st.subheader("Similar cases and shared patterns")
    st.info(context.neighborhood_summary)
    _render_key_insights(context)

    st.markdown("**Most similar cases**")
    if context.similar_cases.empty:
        st.write("No similar cases were found.")
    else:
        chart = similar_cases_chart(context.similar_cases)
        if chart is not None:
            st.plotly_chart(chart, use_container_width=True)
        st.dataframe(context.similar_cases, use_container_width=True)

    st.markdown("**Shared answers**")
    shared_behaviors = context.shared_behaviors.head(10).copy()
    if not shared_behaviors.empty:
        shared_behaviors["question"] = shared_behaviors["question"].map(_friendly_behavior)
        shared_behaviors = shared_behaviors.rename(
            columns={
                "question": "Answer area",
                "answer_value": "Answer",
                "supporting_cases": "How many similar cases share it",
            }
        )
    st.dataframe(shared_behaviors, use_container_width=True, hide_index=True)

    st.markdown("**Shared background details**")
    shared_demographics = context.shared_demographics.head(10).copy()
    if not shared_demographics.empty:
        shared_demographics["demographic_type"] = shared_demographics["demographic_type"].map(_friendly_demographic)
        shared_demographics = shared_demographics.rename(
            columns={
                "demographic_type": "Detail",
                "demographic_value": "Value",
                "supporting_cases": "How many similar cases share it",
            }
        )
    st.dataframe(shared_demographics, use_container_width=True, hide_index=True)

    _render_neighborhood_snapshot(subgraph)

    with st.expander("Technical graph data"):
        st.json(subgraph)
