from __future__ import annotations

from io import BytesIO
from typing import Any

import pandas as pd

TITLE_COLOR_HEX = "#17324D"
BODY_COLOR_HEX = "#1F2D3A"
MUTED_COLOR_HEX = "#5B6B79"
ACCENT_FILL_HEX = "#EAF1F6"
ROW_ALT_FILL_HEX = "#F8FBFD"
GRID_COLOR_HEX = "#D7E0E8"

REPORTLAB_IMPORT_ERROR: Exception | None = None

try:
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_LEFT
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import mm
    from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
except Exception as exc:  # pragma: no cover - runtime dependency guard
    REPORTLAB_IMPORT_ERROR = exc

FEATURE_LABELS = {
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

BEHAVIOR_PHRASES = {
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

DEMOGRAPHIC_PHRASES = {
    "Family_mem_with_ASD": "family history of ASD",
    "Jaundice": "history of jaundice",
    "Ethnicity": "ethnicity",
    "Sex": "sex",
}


def _styles():
    base = getSampleStyleSheet()
    title_color = colors.HexColor(TITLE_COLOR_HEX)
    body_color = colors.HexColor(BODY_COLOR_HEX)
    muted_color = colors.HexColor(MUTED_COLOR_HEX)
    return {
        "title": ParagraphStyle(
            "ReportTitle",
            parent=base["Title"],
            fontName="Helvetica-Bold",
            fontSize=22,
            leading=28,
            textColor=title_color,
            alignment=TA_LEFT,
            spaceAfter=6,
        ),
        "subtitle": ParagraphStyle(
            "ReportSubtitle",
            parent=base["BodyText"],
            fontName="Helvetica",
            fontSize=11,
            leading=15,
            textColor=muted_color,
            spaceAfter=3,
        ),
        "section": ParagraphStyle(
            "SectionHeading",
            parent=base["Heading2"],
            fontName="Helvetica-Bold",
            fontSize=14,
            leading=18,
            textColor=title_color,
            spaceBefore=4,
            spaceAfter=8,
        ),
        "subsection": ParagraphStyle(
            "SubsectionHeading",
            parent=base["Heading3"],
            fontName="Helvetica-Bold",
            fontSize=11.5,
            leading=14,
            textColor=title_color,
            spaceBefore=3,
            spaceAfter=5,
        ),
        "body": ParagraphStyle(
            "BodyCopy",
            parent=base["BodyText"],
            fontName="Helvetica",
            fontSize=10.2,
            leading=14,
            textColor=body_color,
            spaceAfter=6,
        ),
        "muted": ParagraphStyle(
            "MutedCopy",
            parent=base["BodyText"],
            fontName="Helvetica",
            fontSize=9.3,
            leading=12.5,
            textColor=muted_color,
            spaceAfter=5,
        ),
        "table": ParagraphStyle(
            "TableCopy",
            parent=base["BodyText"],
            fontName="Helvetica",
            fontSize=8.9,
            leading=11.2,
            textColor=body_color,
        ),
        "table_header": ParagraphStyle(
            "TableHeader",
            parent=base["BodyText"],
            fontName="Helvetica-Bold",
            fontSize=8.8,
            leading=11,
            textColor=title_color,
        ),
    }


def _format_feature_name(name: str) -> str:
    label = str(name).split("__", 1)[-1].replace("_", " ")
    return FEATURE_LABELS.get(label, label)


def _format_value(value: Any) -> str:
    if pd.isna(value):
        return ""
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def _predicted_label_probability(prediction) -> float:
    return prediction.probability_yes if prediction.predicted_label == "Yes" else prediction.probability_no


def _effect_direction_text(value: float) -> str:
    return "Pushes the result more toward ASD traits" if float(value) > 0 else "Pushes the result less toward ASD traits"


def _importance_level(value: float) -> str:
    abs_value = abs(float(value))
    if abs_value >= 0.5:
        return "High"
    if abs_value >= 0.2:
        return "Medium"
    return "Low"


def _similarity_level(score: float) -> str:
    score = float(score)
    if score >= 9:
        return "Very close match"
    if score >= 7:
        return "Close match"
    if score >= 5:
        return "Moderately similar"
    return "Limited similarity"


def _friendly_answer_text(question: str, value: Any) -> str:
    area = BEHAVIOR_PHRASES.get(str(question), str(question))
    value_text = "present" if str(value) == "1" else "not present / more typical"
    if str(question) == "A10":
        value_text = "present" if str(value) == "1" else "not present"
    return f"{area} = {value_text}"


def _friendly_demographic_text(demo_type: str, demo_value: Any) -> str:
    label = DEMOGRAPHIC_PHRASES.get(str(demo_type), str(demo_type))
    return f"{label}: {demo_value}"


def _humanize_graph_context(graph_context) -> str:
    parts: list[str] = []
    if hasattr(graph_context, "similar_cases") and not graph_context.similar_cases.empty:
        top_case = graph_context.similar_cases.iloc[0]
        parts.append(
            f"The closest similar case found in the dataset was case #{int(top_case['case_id'])}, "
            f"which appears to be a {_similarity_level(top_case['similarity_score']).lower()}."
        )
    if hasattr(graph_context, "shared_behaviors") and not graph_context.shared_behaviors.empty:
        top_behavior = graph_context.shared_behaviors.iloc[0]
        parts.append(
            f"The strongest shared answer pattern was {_friendly_answer_text(top_behavior['question'], top_behavior['answer_value'])}, "
            f"seen in {int(top_behavior['supporting_cases'])} similar cases."
        )
    if hasattr(graph_context, "shared_demographics") and not graph_context.shared_demographics.empty:
        top_demo = graph_context.shared_demographics.iloc[0]
        parts.append(
            f"The most common shared background detail was {_friendly_demographic_text(top_demo['demographic_type'], top_demo['demographic_value'])}, "
            f"seen in {int(top_demo['supporting_cases'])} similar cases."
        )
    if not parts:
        return "The graph-based context layer did not retrieve strong similar-case information for this case."
    return " ".join(parts)


def _paragraph(text: str, style) -> Paragraph:
    safe_text = str(text).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace("\n", "<br/>")
    return Paragraph(safe_text, style)


def _make_table(rows: list[list[str]], col_widths: list[float], styles: dict[str, ParagraphStyle]) -> Table:
    accent_fill = colors.HexColor(ACCENT_FILL_HEX)
    row_alt_fill = colors.HexColor(ROW_ALT_FILL_HEX)
    grid_color = colors.HexColor(GRID_COLOR_HEX)
    title_color = colors.HexColor(TITLE_COLOR_HEX)
    data = []
    for row_idx, row in enumerate(rows):
        style = styles["table_header"] if row_idx == 0 else styles["table"]
        data.append([_paragraph(cell, style) for cell in row])

    table = Table(data, colWidths=col_widths, repeatRows=1, hAlign="LEFT")
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), accent_fill),
                ("TEXTCOLOR", (0, 0), (-1, 0), title_color),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("ALIGN", (0, 0), (-1, -1), "LEFT"),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("GRID", (0, 0), (-1, -1), 0.5, grid_color),
                ("BOX", (0, 0), (-1, -1), 0.6, grid_color),
                ("LEFTPADDING", (0, 0), (-1, -1), 6),
                ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ]
            + [
                ("BACKGROUND", (0, row_idx), (-1, row_idx), row_alt_fill if row_idx % 2 == 0 else colors.white)
                for row_idx in range(1, len(rows))
            ]
        )
    )
    return table


def _prepare_case_dataframe(case_data: dict[str, Any]) -> pd.DataFrame:
    ordered_fields = [
        "Case_No",
        "Age_Mons",
        "Sex",
        "Ethnicity",
        "Jaundice",
        "Family_mem_with_ASD",
        "Who_completed_the_test",
        "A1",
        "A2",
        "A3",
        "A4",
        "A5",
        "A6",
        "A7",
        "A8",
        "A9",
        "A10",
        "upload_id",
    ]
    rows = []
    for field in ordered_fields:
        if field in case_data:
            label = _format_feature_name(field.replace("_", " "))
            if field == "Age_Mons":
                label = "Age in months"
            elif field == "Case_No":
                label = "Case number"
            elif field == "upload_id":
                label = "Upload ID"
            rows.append({"Field": label, "Value": _format_value(case_data.get(field))})
    return pd.DataFrame(rows)


def _prepare_local_explanation(local_explanation) -> pd.DataFrame:
    local_df = local_explanation.contributions.head(10).copy()
    local_df["feature"] = local_df["feature"].map(_format_feature_name)
    local_df["direction"] = local_df["shap_value"].map(_effect_direction_text)
    local_df["importance_level"] = local_df["abs_value"].map(_importance_level)
    return local_df.rename(
        columns={
            "feature": "Feature",
            "direction": "Effect on result",
            "importance_level": "Importance level",
        }
    )[["Feature", "Effect on result", "Importance level"]]


def _prepare_global_importance(global_importance: pd.DataFrame) -> pd.DataFrame:
    global_df = global_importance.head(12).copy()
    global_df["feature"] = global_df["feature"].map(_format_feature_name)
    global_df["importance_level"] = global_df["importance"].map(_importance_level)
    return global_df.rename(columns={"feature": "Feature", "importance_level": "Usual importance"})[
        ["Feature", "Usual importance"]
    ]


def _prepare_similar_cases(graph_context) -> pd.DataFrame:
    similar_cases = graph_context.similar_cases.copy()
    if similar_cases.empty:
        return similar_cases
    similar_cases["closeness"] = similar_cases["similarity_score"].map(_similarity_level)
    similar_cases["shared_answers"] = similar_cases["shared_answers"].map(_format_value)
    similar_cases["shared_demographics"] = similar_cases["shared_demographics"].map(_format_value)
    similar_cases = similar_cases.rename(
        columns={
            "case_id": "Case ID",
            "label": "ASD label",
            "closeness": "How close it is",
            "shared_answers": "Shared answers",
            "shared_demographics": "Shared demographics",
        }
    )
    return similar_cases[["Case ID", "ASD label", "How close it is", "Shared answers", "Shared demographics"]]


def _prepare_shared_behaviors(graph_context) -> pd.DataFrame:
    shared_behaviors = graph_context.shared_behaviors.copy()
    if shared_behaviors.empty:
        return shared_behaviors
    shared_behaviors["question"] = shared_behaviors["question"].map(_format_feature_name)
    shared_behaviors["answer_value"] = shared_behaviors["answer_value"].map(_format_value)
    shared_behaviors["supporting_cases"] = shared_behaviors["supporting_cases"].map(_format_value)
    return shared_behaviors.rename(
        columns={
            "question": "Behavior area",
            "answer_value": "Answer",
            "supporting_cases": "Supporting cases",
        }
    )[["Behavior area", "Answer", "Supporting cases"]]


def build_pdf_report(
    *,
    case_data: dict[str, Any],
    prediction,
    graph_context=None,
    local_explanation=None,
    global_importance: pd.DataFrame | None = None,
    anomaly_result=None,
) -> bytes:
    if REPORTLAB_IMPORT_ERROR is not None:
        raise RuntimeError(
            "PDF export requires the `reportlab` package in the same Python environment used by Streamlit."
        ) from REPORTLAB_IMPORT_ERROR

    styles = _styles()
    buffer = BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        leftMargin=20 * mm,
        rightMargin=20 * mm,
        topMargin=20 * mm,
        bottomMargin=20 * mm,
        title="NeuroCypher ASD Report",
        author="NeuroCypher ASD",
    )

    story = []

    story.append(_paragraph("NeuroCypher ASD", styles["title"]))
    story.append(_paragraph("Clinical decision-support report", styles["subtitle"]))
    story.append(_paragraph("2026", styles["subtitle"]))
    story.append(Spacer(1, 8))

    story.append(_paragraph("Case Summary", styles["section"]))
    summary_lines = [
        f"Case number: {case_data.get('Case_No', 'N/A')}",
        f"Age in months: {case_data.get('Age_Mons', 'N/A')}",
        f"Predicted ASD trait label: {prediction.predicted_label}",
        f"Probability of the predicted label: {_predicted_label_probability(prediction):.1%}",
        f"Probability of ASD traits = Yes: {prediction.probability_yes:.1%}",
        f"Confidence summary: {prediction.confidence}",
    ]
    for line in summary_lines:
        story.append(_paragraph(line, styles["body"]))

    story.append(Spacer(1, 6))
    story.append(_paragraph("Prepared by", styles["subsection"]))
    for line in [
        "Dr. Georgios Bouchouras",
        "Dimitrios Doumanas, MSc",
        "Assoc. Prof. Konstantinos Kotis",
        "Intelligent Systems Research Laboratory (i-Lab), University of the Aegean",
    ]:
        story.append(_paragraph(line, styles["body"]))

    story.append(Spacer(1, 8))
    story.append(_paragraph("Disclaimer", styles["subsection"]))
    story.append(
        _paragraph(
            "This report is generated by a research-oriented decision-support application. "
            "It does not constitute a medical diagnosis and should not be used as a standalone clinical instrument.",
            styles["muted"],
        )
    )

    story.append(PageBreak())
    story.append(_paragraph("Analysis Overview", styles["section"]))
    story.append(
        _paragraph(
            (
                f"The primary tabular model classified this case as {prediction.predicted_label}. "
                f"The probability of the predicted label was {_predicted_label_probability(prediction):.1%}, "
                f"while the model-estimated probability for ASD traits = Yes was {prediction.probability_yes:.1%}. "
                f"The confidence summary was {prediction.confidence}."
            ),
            styles["body"],
        )
    )
    if graph_context is not None:
        story.append(_paragraph("Graph context summary", styles["subsection"]))
        story.append(_paragraph(_humanize_graph_context(graph_context), styles["body"]))
    if anomaly_result is not None:
        story.append(_paragraph("Anomaly support", styles["subsection"]))
        story.append(
            _paragraph(
                (
                    f"Anomaly score: {anomaly_result.score:.3f}. "
                    f"Unusual profile flag: {'Yes' if anomaly_result.is_anomalous else 'No'}."
                ),
                styles["body"],
            )
        )

    story.append(PageBreak())
    story.append(_paragraph("Submitted Case Data", styles["section"]))
    case_df = _prepare_case_dataframe(case_data)
    case_rows = [case_df.columns.tolist(), *case_df.astype(str).values.tolist()]
    story.append(_make_table(case_rows, [52 * mm, 98 * mm], styles))

    if local_explanation is not None and hasattr(local_explanation, "contributions"):
        story.append(PageBreak())
        story.append(_paragraph("Local Explainability Summary", styles["section"]))
        story.append(_paragraph("Top feature-level drivers for the current prediction.", styles["muted"]))
        story.append(_paragraph("Plain-language explanation", styles["subsection"]))
        story.append(_paragraph(getattr(local_explanation, "plain_language", "Not available."), styles["body"]))
        story.append(_paragraph("How to read this section", styles["subsection"]))
        story.append(
            _paragraph(
                "Effect on result tells you whether a feature pulled the result more toward ASD traits or less toward ASD traits. "
                "Importance level is a simple interpretation of how strongly that feature affected this case: High, Medium, or Low.",
                styles["body"],
            )
        )
        local_df = _prepare_local_explanation(local_explanation)
        local_rows = [local_df.columns.tolist(), *local_df.astype(str).values.tolist()]
        story.append(_make_table(local_rows, [48 * mm, 74 * mm, 28 * mm], styles))

    if global_importance is not None:
        story.append(PageBreak())
        story.append(_paragraph("Global Feature Importance", styles["section"]))
        story.append(_paragraph("What the model usually pays most attention to across many cases.", styles["muted"]))
        story.append(_paragraph("How to read this section", styles["subsection"]))
        story.append(
            _paragraph(
                "Usual importance does not mean good or bad. It simply shows how often a feature matters in the model overall. "
                "High means the model relies on it more often; Low means it usually has a smaller role.",
                styles["body"],
            )
        )
        global_df = _prepare_global_importance(global_importance)
        global_rows = [global_df.columns.tolist(), *global_df.astype(str).values.tolist()]
        story.append(_make_table(global_rows, [102 * mm, 48 * mm], styles))

    if graph_context is not None:
        story.append(PageBreak())
        story.append(_paragraph("Most Similar Cases", styles["section"]))
        story.append(_paragraph("Comparable cases retrieved from the graph context layer.", styles["muted"]))
        story.append(_paragraph("How to read this section", styles["subsection"]))
        story.append(
            _paragraph(
                "How close it is is a simplified interpretation of the graph similarity score. "
                "Shared answers shows how many questionnaire responses were the same, and Shared demographics shows how many background details matched.",
                styles["body"],
            )
        )
        similar_df = _prepare_similar_cases(graph_context)
        if similar_df.empty:
            story.append(_paragraph("No similar cases were retrieved.", styles["body"]))
        else:
            similar_rows = [similar_df.columns.tolist(), *similar_df.astype(str).values.tolist()]
            story.append(_make_table(similar_rows, [18 * mm, 24 * mm, 42 * mm, 28 * mm, 38 * mm], styles))

        story.append(Spacer(1, 10))
        story.append(_paragraph("Shared Behavioral Patterns", styles["section"]))
        story.append(
            _paragraph(
                "Recurring answer patterns among similar retrieved cases. Supporting cases tells you how often each pattern appeared.",
                styles["muted"],
            )
        )
        shared_df = _prepare_shared_behaviors(graph_context)
        if shared_df.empty:
            story.append(_paragraph("No shared behavioral patterns were retrieved.", styles["body"]))
        else:
            shared_rows = [shared_df.columns.tolist(), *shared_df.astype(str).values.tolist()]
            story.append(_make_table(shared_rows, [86 * mm, 22 * mm, 42 * mm], styles))

    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()
