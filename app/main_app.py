from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from uuid import uuid4

import pandas as pd
import streamlit as st

from app.config import AppConfig, DEFAULT_DATASET_PATH
from app.data.schema_validation import (
    extract_single_case,
    read_csv_bytes,
    standardize_dataframe,
    validate_case_dataframe,
)
from app.models.train_tabular_model import train_from_csv
from app.services.anomaly_service import AnomalyService
from app.services.graph_context_service import GraphContextService
from app.services.neo4j_service import Neo4jService
from app.services.nl_to_cypher_service import NLToCypherService
from app.services.shap_service import ShapService
from app.services.tabular_model_service import TabularModelService
from app.ui.anomaly_view import render_anomaly
from app.ui.explainability_view import render_explainability
from app.ui.graph_view import render_graph_context
from app.ui.nl_query_view import render_nl_query
from app.ui.prediction_view import render_prediction
from app.utils.logging_utils import get_logger
from app.utils.pdf_report import build_pdf_report

logger = get_logger(__name__)

st.set_page_config(page_title="NeuroCypher ASD", page_icon="🧠", layout="wide")

QUESTION_ITEMS = {
    "A1": {
        "text": "Does your child look at you when you call his/her name?",
        "zero": "0 = Looks at the caller / behavior is typically present",
        "one": "1 = Rarely or never looks at the caller",
    },
    "A2": {
        "text": "How easy is it for you to get eye contact with your child?",
        "zero": "0 = Eye contact is easy to obtain",
        "one": "1 = Eye contact is difficult to obtain",
    },
    "A3": {
        "text": "Does your child point to indicate that he/she wants something?",
        "zero": "0 = Uses pointing to request",
        "one": "1 = Rarely or never uses pointing to request",
    },
    "A4": {
        "text": "Does your child point to share interest with you?",
        "zero": "0 = Uses pointing to share interest",
        "one": "1 = Rarely or never points to share interest",
    },
    "A5": {
        "text": "Does your child pretend?",
        "zero": "0 = Engages in pretend play",
        "one": "1 = Rarely or never engages in pretend play",
    },
    "A6": {
        "text": "Does your child follow where you are looking?",
        "zero": "0 = Follows gaze or direction of attention",
        "one": "1 = Rarely or never follows gaze",
    },
    "A7": {
        "text": "If someone is visibly upset, does your child try to comfort them?",
        "zero": "0 = Shows signs of comfort or concern",
        "one": "1 = Rarely or never shows comfort",
    },
    "A8": {
        "text": "Would you describe your child's first words as typical?",
        "zero": "0 = Typical or mostly typical first words",
        "one": "1 = Unusual first words or no speech yet",
    },
    "A9": {
        "text": "Does your child use simple gestures?",
        "zero": "0 = Uses gestures such as waving goodbye",
        "one": "1 = Rarely or never uses simple gestures",
    },
    "A10": {
        "text": "Does your child stare at nothing with no apparent purpose?",
        "zero": "0 = No or minimal such behavior",
        "one": "1 = Behavior is present",
    },
}

ETHNICITY_OPTIONS = [
    "Hispanic",
    "Latino",
    "Native Indian",
    "Others",
    "Pacifica",
    "White European",
    "asian",
    "black",
    "middle eastern",
    "mixed",
    "south asian",
]

RESPONDENT_OPTIONS = [
    "family member",
    "Health Care Professional",
    "Health care professional",
    "Others",
    "Self",
]

SAMPLE_CASE_TEMPLATE = pd.DataFrame(
    [
        {
            "Case_No": 1001,
            "A1": 0,
            "A2": 0,
            "A3": 1,
            "A4": 0,
            "A5": 1,
            "A6": 0,
            "A7": 0,
            "A8": 1,
            "A9": 0,
            "A10": 0,
            "Age_Mons": 36,
            "Sex": "m",
            "Ethnicity": "White European",
            "Jaundice": "no",
            "Family_mem_with_ASD": "yes",
            "Who_completed_the_test": "family member",
        }
    ]
)


def _render_binary_question(feature: str, column) -> int:
    item = QUESTION_ITEMS[feature]
    with column:
        st.markdown(f"**{feature}**")
        st.caption(item["text"])
        return st.radio(
            f"{feature} score",
            options=[0, 1],
            format_func=lambda value: item["zero"] if value == 0 else item["one"],
            index=0,
            horizontal=False,
            key=f"form_{feature}",
            label_visibility="collapsed",
        )


@st.cache_resource
def get_services():
    config = AppConfig()
    neo4j_service = Neo4jService()
    return {
        "config": config,
        "tabular": TabularModelService(config),
        "shap": ShapService(),
        "anomaly": AnomalyService(config),
        "neo4j": neo4j_service,
        "graph": GraphContextService(neo4j_service),
        "nlq": NLToCypherService(neo4j_service),
    }


def _store_case(case_data: dict) -> None:
    normalized_case = dict(case_data)
    normalized_case.setdefault("upload_id", f"streamlit-{uuid4()}")
    st.session_state["current_case"] = normalized_case


def _get_case() -> dict | None:
    return st.session_state.get("current_case")


def _store_prediction(prediction) -> None:
    st.session_state["prediction_result"] = prediction


def _get_prediction():
    return st.session_state.get("prediction_result")


def _store_graph_context(context) -> None:
    st.session_state["graph_context"] = context


def _get_graph_context():
    return st.session_state.get("graph_context")


def _sample_case_csv_bytes() -> bytes:
    return SAMPLE_CASE_TEMPLATE.to_csv(sep=";", index=False).encode("utf-8-sig")


def _build_report_bytes(services: dict) -> bytes:
    case_data = _get_case()
    prediction = _get_prediction()
    graph_context = _get_graph_context()
    anomaly_result = st.session_state.get("anomaly_result")

    if not case_data or not prediction:
        raise ValueError("Prediction results are not available yet.")

    local_explanation = None
    global_importance = None
    try:
        bundle = services["tabular"].load()
        global_importance = services["shap"].global_importance(bundle)
        local_explanation = services["shap"].local_explanation(
            bundle=bundle,
            case_data=case_data,
            predicted_label=prediction.predicted_label,
            probability=prediction.probability_yes,
        )
    except Exception as exc:
        logger.warning("PDF report will be generated without SHAP details: %s", exc)

    return build_pdf_report(
        case_data=case_data,
        prediction=prediction,
        graph_context=graph_context,
        local_explanation=local_explanation,
        global_importance=global_importance,
        anomaly_result=anomaly_result,
    )


def _render_report_download(services: dict) -> None:
    prediction = _get_prediction()
    if not prediction:
        return

    try:
        report_bytes = _build_report_bytes(services)
        st.download_button(
            "Download report",
            data=report_bytes,
            file_name="neurocypher_asd_report_2026.pdf",
            mime="application/pdf",
            use_container_width=True,
        )
        sections = ["prediction summary"]
        if _get_graph_context() is not None:
            sections.append("graph context")
        if st.session_state.get("anomaly_result") is not None:
            sections.append("anomaly support")
        sections.append("available explanations")
        st.caption(f"The report currently includes the submitted case, {', '.join(sections)}.")
    except Exception as exc:
        st.warning(f"PDF report is currently unavailable: {exc}")


def render_upload_tab(services: dict) -> None:
    st.header("Upload / Select Case")
    st.write("Upload a CSV with exactly one case, or populate the single-case form.")
    st.caption("Accepted file type: CSV with `;` as delimiter and one row only.")
    st.download_button(
        "Download sample CSV template",
        data=_sample_case_csv_bytes(),
        file_name="neurocypher_case_template.csv",
        mime="text/csv",
    )
    st.caption(
        "Template guidance: use binary values for A1-A10, where for A1-A9 the value `1` indicates the less typical "
        "response pattern, while for A10 the value `1` indicates presence of the behavior."
    )

    uploaded = st.file_uploader("Single-case CSV", type=["csv"])
    if uploaded is not None:
        try:
            df = standardize_dataframe(read_csv_bytes(uploaded.getvalue()))
            case_data = extract_single_case(df)
            _store_case(case_data)
            st.success("Case loaded from CSV.")
            st.dataframe(pd.DataFrame([case_data]), use_container_width=True)
        except Exception as exc:
            st.error(str(exc))

    with st.form("single_case_form"):
        form_values = {}
        meta_cols = st.columns(2)
        form_values["Case_No"] = meta_cols[0].number_input("Case_No", min_value=1, value=1, step=1)
        form_values["Age_Mons"] = meta_cols[1].number_input("Age_Mons", min_value=1, value=36, step=1)

        st.markdown("**Q-Chat-10 Items**")
        st.caption(
            "Binary scoring follows the dataset convention. For A1-A9, 1 denotes the less typical response pattern; "
            "for A10, 1 denotes presence of the behavior."
        )
        question_cols = st.columns(2)
        for idx, feature in enumerate([f"A{i}" for i in range(1, 11)]):
            form_values[feature] = _render_binary_question(feature, question_cols[idx % 2])

        demo_cols = st.columns(2)
        form_values["Sex"] = demo_cols[0].selectbox(
            "Sex",
            options=["m", "f"],
            format_func=lambda value: "Male (m)" if value == "m" else "Female (f)",
        )
        form_values["Ethnicity"] = demo_cols[1].selectbox("Ethnicity", options=ETHNICITY_OPTIONS)
        form_values["Jaundice"] = demo_cols[0].selectbox(
            "Jaundice",
            options=["no", "yes"],
            format_func=lambda value: "No" if value == "no" else "Yes",
        )
        form_values["Family_mem_with_ASD"] = demo_cols[1].selectbox(
            "Family_mem_with_ASD",
            options=["no", "yes"],
            format_func=lambda value: "No" if value == "no" else "Yes",
        )
        form_values["Who_completed_the_test"] = st.selectbox(
            "Who_completed_the_test",
            options=RESPONDENT_OPTIONS,
        )
        submitted = st.form_submit_button("Use this case")
        if submitted:
            df = pd.DataFrame([form_values])
            valid, missing = validate_case_dataframe(df)
            if not valid:
                st.error(f"Missing required columns: {', '.join(missing)}")
            else:
                case_data = extract_single_case(df)
                _store_case(case_data)
                st.success("Case stored for prediction.")
                st.dataframe(pd.DataFrame([case_data]), use_container_width=True)

    current_case = _get_case()
    if current_case:
        st.info("Current case preview")
        st.dataframe(pd.DataFrame([current_case]), use_container_width=True)
        st.markdown("### Case report")
        if _get_prediction():
            st.caption(
                "Next steps: run `Prediction` first, then run `Anomaly / Flags`, and then return here to download the report. "
                "Any available graph context and explanations will be included automatically."
            )
            _render_report_download(services)
        else:
            st.caption(
                "After you store a case, go to `Prediction` and run the prediction first. "
                "Then go to `Anomaly / Flags`, run the anomaly analysis, and return here to download the report."
            )


def render_prediction_tab(services: dict) -> None:
    st.header("Prediction")
    case_data = _get_case()
    if not case_data:
        st.info("Load or enter a case in the first tab.")
        return

    if st.button("Run tabular prediction", type="primary"):
        try:
            prediction = services["tabular"].predict_case(case_data)
            _store_prediction(prediction)
        except Exception as exc:
            st.error(str(exc))
            return

        try:
            context = services["graph"].build_context(
                case_data,
                prediction_label=prediction.predicted_label,
                risk_score=prediction.probability_yes,
                top_k=services["config"].similar_cases_k,
            )
            _store_graph_context(context)
        except Exception as exc:
            if "ConstraintValidationFailed" in str(exc):
                retry_case = dict(case_data)
                retry_case["upload_id"] = f"streamlit-{uuid4()}"
                _store_case(retry_case)
                try:
                    context = services["graph"].build_context(
                        retry_case,
                        prediction_label=prediction.predicted_label,
                        risk_score=prediction.probability_yes,
                        top_k=services["config"].similar_cases_k,
                    )
                    _store_graph_context(context)
                except Exception as retry_exc:
                    st.warning(
                        "Prediction completed, but graph context could not be refreshed because Neo4j rejected the "
                        f"temporary case node: {retry_exc}"
                    )
            else:
                st.warning(f"Prediction completed, but graph context is unavailable: {exc}")

    prediction = _get_prediction()
    if prediction:
        render_prediction(prediction)


def render_explainability_tab(services: dict) -> None:
    st.header("Why this prediction?")
    case_data = _get_case()
    prediction = _get_prediction()
    if not case_data or not prediction:
        st.info("Run prediction first.")
        return

    try:
        bundle = services["tabular"].load()
        global_importance = services["shap"].global_importance(bundle)
        local_explanation = services["shap"].local_explanation(
            bundle=bundle,
            case_data=case_data,
            predicted_label=prediction.predicted_label,
            probability=prediction.probability_yes,
        )
        render_explainability(global_importance, local_explanation)
    except Exception as exc:
        st.error(str(exc))


def render_graph_tab(services: dict) -> None:
    st.header("Graph context")
    context = _get_graph_context()
    if not context:
        st.info("Run prediction first to populate graph context.")
        return
    subgraph = services["graph"].get_case_subgraph(context.upload_id)
    render_graph_context(context, subgraph)


def render_nl_tab(services: dict) -> None:
    st.header("Ask the Graph")
    st.caption("NL graph explorer version: deterministic-intents v2")
    st.caption(
        "Ask questions about behavioral answers, demographics, ASD labels, and case retrieval. "
        "Counts, case lists, comparisons, common demographics, and common behavior-pattern questions are handled more deterministically."
    )
    st.markdown("**Example questions**")
    st.markdown(
        "- `How many toddlers have A1 = 1?`\n"
        "- `How many male toddlers were screened positive for ASD traits?`\n"
        "- `Show 10 cases where A1 = 1 and A5 = 1.`\n"
        "- `Compare A1 = 1 vs A1 = 0 by ASD label.`\n"
        "- `Top shared demographics for cases with A5 = 1.`\n"
        "- `Show the most common behavior patterns among ASD-positive toddlers.`\n"
        "- `ASD label distribution for cases with A3 = 1.`\n"
        "- `Top co-occurring answers with A2 = 1.`\n"
        "- `Respondent breakdown for cases with A4 = 1.`\n"
        "- `Breakdown by sex for cases with A6 = 1.`\n"
        "- `Show similar cases to 25.`"
    )
    question = st.text_input("Natural-language question")
    if st.button("Generate and run Cypher"):
        result = services["nlq"].ask(question)
        render_nl_query(result)


def render_anomaly_tab(services: dict) -> None:
    st.header("Anomaly / Flags")
    case_data = _get_case()
    if not case_data:
        st.info("Load a case first.")
        return
    if st.button("Run anomaly support analysis"):
        try:
            result = services["anomaly"].score_case(case_data)
            st.session_state["anomaly_result"] = result
        except Exception as exc:
            st.error(str(exc))
    if "anomaly_result" in st.session_state:
        render_anomaly(st.session_state["anomaly_result"])


def render_admin_tab(services: dict) -> None:
    st.header("Admin / Rebuild")
    st.write("Prediction is handled by the tabular model. Graph rebuilds are contextual support operations.")

    if st.button("Retrain tabular + anomaly models"):
        try:
            train_from_csv(str(DEFAULT_DATASET_PATH))
            st.success("Tabular predictor and anomaly support model retrained.")
        except Exception as exc:
            st.error(str(exc))

    if st.button("Rebuild graph"):
        script = Path(__file__).resolve().parent.parent / "kg_builder_2.py"
        try:
            subprocess.run([sys.executable, str(script), "--build-full-graph"], check=True)
            st.success("Graph rebuild completed.")
        except Exception as exc:
            st.error(f"Graph rebuild failed: {exc}")

    if st.button("Health checks"):
        try:
            checks = services["graph"].health_checks()
            st.json(checks)
        except Exception as exc:
            st.error(str(exc))


def main() -> None:
    services = get_services()
    st.title("NeuroCypher ASD")
    st.caption(
        "Prediction comes from a tabular ML model on questionnaire and demographic data. "
        "The knowledge graph is used as a contextual and explanatory layer for decision-support."
    )
    st.markdown(
        """
        NeuroCypher ASD is a clinical decision-support and research application for autism screening data.
        It combines questionnaire-based machine learning predictions with knowledge-graph context,
        explainability, anomaly support, and natural-language graph querying.
        """
    )
    st.markdown(
        """
        <div style="
            margin: 0.75rem 0 1.25rem 0;
            padding: 1rem 1.15rem;
            border: 1px solid #d8e1e8;
            border-radius: 14px;
            background: linear-gradient(135deg, #f8fafc 0%, #eef4f7 100%);
            box-shadow: 0 6px 18px rgba(15, 23, 42, 0.05);
            font-size: 0.95rem;
            line-height: 1.6;
            color: #243447;
        ">
            <div style="
                font-size: 0.78rem;
                font-weight: 700;
                letter-spacing: 0.08em;
                text-transform: uppercase;
                color: #607487;
                margin-bottom: 0.35rem;
            ">Developed at i-Lab · 2026</div>
            <div>
                Developed by
                <a href="https://giorgosbouh.github.io/github-portfolio/" target="_blank" style="color: #0f5c7a; text-decoration: none; font-weight: 600;">Dr. Georgios Bouchouras</a>,
                Dimitrios Doumanas, MSc, and Assoc. Prof. Konstantinos Kotis.
            </div>
            <div style="color: #4b5f71;">
                Research Lab:
                <a href="https://i-lab.aegean.gr" target="_blank" style="color: #0f5c7a; text-decoration: none; font-weight: 600;">Intelligent Systems Research Laboratory (i-Lab)</a>,
                University of the Aegean.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    tabs = st.tabs(
        [
            "Upload / Select Case",
            "Prediction",
            "Why this prediction?",
            "Graph Context",
            "Ask the Graph",
            "Anomaly / Flags",
            "Admin / Rebuild",
        ]
    )
    with tabs[0]:
        render_upload_tab(services)
    with tabs[1]:
        render_prediction_tab(services)
    with tabs[2]:
        render_explainability_tab(services)
    with tabs[3]:
        render_graph_tab(services)
    with tabs[4]:
        render_nl_tab(services)
    with tabs[5]:
        render_anomaly_tab(services)
    with tabs[6]:
        render_admin_tab(services)

if __name__ == "__main__":
    main()
