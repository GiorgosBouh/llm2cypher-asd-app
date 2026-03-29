from __future__ import annotations
import streamlit as st

from app.services.shap_service import LocalExplanation

FRIENDLY_LABELS = {
    "A1": "Response to name",
    "A2": "Eye contact",
    "A3": "Pointing to ask for something",
    "A4": "Pointing to share interest",
    "A5": "Pretend play",
    "A6": "Following another person's gaze",
    "A7": "Trying to comfort others",
    "A8": "Early word use",
    "A9": "Simple gestures",
    "A10": "Staring without clear purpose",
    "Age Mons": "Age in months",
    "Sex": "Sex",
    "Ethnicity": "Ethnicity",
    "Jaundice": "History of jaundice",
    "Family mem with ASD": "Family history of ASD",
    "Who completed the test": "Who completed the test",
}


def _format_feature_name(name: str) -> str:
    label = str(name).split("__", 1)[-1]
    label = label.replace("_", " ")
    return FRIENDLY_LABELS.get(label, label)


def _direction_label(value: float) -> str:
    return "Pushes the result more toward ASD traits" if value > 0 else "Pushes the result less toward ASD traits"


def _strength_label(value: float) -> str:
    abs_value = abs(float(value))
    if abs_value >= 0.5:
        return "Strong"
    if abs_value >= 0.2:
        return "Medium"
    return "Small"


def render_explainability(global_importance, local_explanation: LocalExplanation) -> None:
    """Render SHAP-based global and local explainability."""
    st.subheader("Why the app gave this result")
    st.info(local_explanation.plain_language)

    local_display = local_explanation.contributions.head(10).copy()
    local_display["feature"] = local_display["feature"].map(_format_feature_name)
    local_display["direction_text"] = local_display["shap_value"].map(_direction_label)
    local_display["strength_text"] = local_display["abs_value"].map(_strength_label)
    local_display = local_display.rename(
        columns={
            "feature": "What the app looked at",
            "direction_text": "What effect it had",
            "strength_text": "How important it was",
        }
    )[["What the app looked at", "What effect it had", "How important it was"]]
    st.markdown("**What mattered most for this result**")
    st.caption("This table explains, in simple words, which answers or details had the biggest effect on this result.")
    st.dataframe(local_display, use_container_width=True, hide_index=True)

    global_display = global_importance.head(15).copy()
    global_display["feature"] = global_display["feature"].map(_format_feature_name)
    global_display = global_display.rename(columns={"feature": "What the app usually looks at", "importance": "Usual importance"})

    st.markdown("**What usually matters most overall**")
    st.caption("These are the things the app tends to pay the most attention to across many cases.")
    for feature in global_display["What the app usually looks at"].head(5):
        st.write(f"- {feature}")
