from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import Any

import pandas as pd

from app.config import CATEGORICAL_COLUMNS, QUESTION_COLUMNS
from app.services.neo4j_service import Neo4jService
from app.utils.logging_utils import get_logger

logger = get_logger(__name__)


@dataclass
class GraphContextResult:
    upload_id: str
    similar_cases: pd.DataFrame
    shared_behaviors: pd.DataFrame
    shared_demographics: pd.DataFrame
    neighborhood_summary: str


class GraphContextService:
    """Graph-backed contextual reasoning without using the graph as primary predictor."""

    def __init__(self, neo4j_service: Neo4jService):
        self.neo4j = neo4j_service

    def upsert_case_context(self, case_data: dict[str, Any], prediction_label: str, risk_score: float) -> str:
        upload_id = str(case_data.get("upload_id") or f"streamlit-{uuid.uuid4()}")
        payload = {key: case_data.get(key) for key in [*QUESTION_COLUMNS, "Age_Mons", *CATEGORICAL_COLUMNS, "Case_No"]}
        payload.update({"upload_id": upload_id, "predicted_label": prediction_label, "risk_score": risk_score})

        self.neo4j.run_write(
            """
            MERGE (c:Case {upload_id: $upload_id})
            REMOVE c.id
            SET c.temp_case = true,
                c.source = 'streamlit_app',
                c.predicted_label = $predicted_label,
                c.risk_score = $risk_score,
                c.input_case_no = toInteger($Case_No)
            """,
            payload,
        )

        for question in QUESTION_COLUMNS:
            value = payload.get(question)
            if value is None or str(value).lower() == "nan":
                continue
            self.neo4j.run_write(
                """
                MATCH (c:Case {upload_id: $upload_id})
                MERGE (q:BehaviorQuestion {name: $question})
                MERGE (c)-[r:HAS_ANSWER]->(q)
                SET r.value = toInteger($value)
                """,
                {"upload_id": upload_id, "question": question, "value": value},
            )

        for demographic in ["Sex", "Ethnicity", "Jaundice", "Family_mem_with_ASD"]:
            demo_value = payload.get(demographic)
            if demo_value is None or not str(demo_value).strip() or str(demo_value).lower() == "nan":
                continue
            self.neo4j.run_write(
                """
                MATCH (c:Case {upload_id: $upload_id})
                MERGE (d:DemographicAttribute {type: $demo_type, value: $demo_value})
                MERGE (c)-[:HAS_DEMOGRAPHIC]->(d)
                """,
                {"upload_id": upload_id, "demo_type": demographic, "demo_value": str(demo_value)},
            )

        submitter = payload.get("Who_completed_the_test")
        if submitter is not None and str(submitter).strip() and str(submitter).lower() != "nan":
            self.neo4j.run_write(
                """
                MATCH (c:Case {upload_id: $upload_id})
                MERGE (s:SubmitterType {type: $submitter})
                MERGE (c)-[:SUBMITTED_BY]->(s)
                """,
                {"upload_id": upload_id, "submitter": str(submitter)},
            )
        return upload_id

    def get_similar_cases(self, upload_id: str, top_k: int = 5) -> pd.DataFrame:
        rows = self.neo4j.run_read(
            """
            MATCH (u:Case {upload_id: $upload_id})-[ua:HAS_ANSWER]->(q:BehaviorQuestion)<-[ca:HAS_ANSWER]-(c:Case)
            WHERE coalesce(c.upload_id, '') <> $upload_id AND coalesce(c.temp_case, false) = false
            WITH u, c, sum(CASE WHEN ua.value = ca.value THEN 1 ELSE 0 END) AS shared_answers
            OPTIONAL MATCH (u)-[:HAS_DEMOGRAPHIC]->(ud:DemographicAttribute)<-[:HAS_DEMOGRAPHIC]-(c)
            WITH c, shared_answers, count(ud) AS shared_demographics
            OPTIONAL MATCH (c)-[:SCREENED_FOR]->(t:ASD_Trait)
            RETURN coalesce(c.id, -1) AS case_id,
                   coalesce(t.label, 'No ASD label recorded') AS label,
                   shared_answers,
                   shared_demographics,
                   (shared_answers * 1.0 + shared_demographics * 0.5) AS similarity_score
            ORDER BY similarity_score DESC, case_id ASC
            LIMIT $top_k
            """,
            {"upload_id": upload_id, "top_k": top_k},
        )
        return pd.DataFrame(rows)

    def get_shared_behavioral_traits(self, upload_id: str) -> pd.DataFrame:
        rows = self.neo4j.run_read(
            """
            MATCH (u:Case {upload_id: $upload_id})-[ua:HAS_ANSWER]->(q:BehaviorQuestion)
            MATCH (c:Case)-[ca:HAS_ANSWER]->(q)
            WHERE coalesce(c.upload_id, '') <> $upload_id AND coalesce(c.temp_case, false) = false AND ua.value = ca.value
            RETURN q.name AS question, ua.value AS answer_value, count(DISTINCT c) AS supporting_cases
            ORDER BY supporting_cases DESC, question ASC
            """,
            {"upload_id": upload_id},
        )
        return pd.DataFrame(rows)

    def get_shared_demographic_traits(self, upload_id: str) -> pd.DataFrame:
        rows = self.neo4j.run_read(
            """
            MATCH (u:Case {upload_id: $upload_id})-[:HAS_DEMOGRAPHIC]->(d:DemographicAttribute)
            MATCH (c:Case)-[:HAS_DEMOGRAPHIC]->(d)
            WHERE coalesce(c.upload_id, '') <> $upload_id AND coalesce(c.temp_case, false) = false
            RETURN d.type AS demographic_type, d.value AS demographic_value, count(DISTINCT c) AS supporting_cases
            ORDER BY supporting_cases DESC, demographic_type ASC
            """,
            {"upload_id": upload_id},
        )
        return pd.DataFrame(rows)

    def get_case_subgraph(self, upload_id: str) -> dict[str, list[dict[str, Any]]]:
        rows = self.neo4j.run_read(
            """
            MATCH (u:Case {upload_id: $upload_id})-[r]->(n)
            RETURN labels(u) AS source_labels,
                   coalesce(u.upload_id, toString(u.id)) AS source_id,
                   type(r) AS relationship,
                   labels(n) AS target_labels,
                   CASE
                       WHEN 'DemographicAttribute' IN labels(n) THEN n.type + ': ' + n.value
                       WHEN 'SubmitterType' IN labels(n) THEN n.type
                       ELSE coalesce(n.name, n.label, toString(n.id))
                   END AS target_id
            """,
            {"upload_id": upload_id},
        )
        node_map: dict[str, dict[str, Any]] = {}
        node_map[upload_id] = {"id": upload_id, "labels": ["Case"]}
        for row in rows:
            node_map[row["source_id"]] = {"id": row["source_id"], "labels": row["source_labels"]}
            node_map[row["target_id"]] = {"id": row["target_id"], "labels": row["target_labels"]}
        edges = [
            {
                "source": row["source_id"],
                "relationship": row["relationship"],
                "target": row["target_id"],
                "target_labels": row["target_labels"],
            }
            for row in rows
        ]
        return {"nodes": list(node_map.values()), "edges": edges}

    def get_case_pattern_summary(self, upload_id: str) -> str:
        behaviors = self.get_shared_behavioral_traits(upload_id)
        demographics = self.get_shared_demographic_traits(upload_id)
        similar_cases = self.get_similar_cases(upload_id, top_k=3)

        if behaviors.empty and demographics.empty:
            return "No strong graph context was retrieved for this case."

        summary_parts: list[str] = []
        if not behaviors.empty:
            top_behavior = behaviors.iloc[0]
            summary_parts.append(
                f"the strongest shared behavioral signal is {top_behavior['question']} = {top_behavior['answer_value']} "
                f"({int(top_behavior['supporting_cases'])} supporting cases)"
            )
        if not demographics.empty:
            top_demo = demographics.iloc[0]
            summary_parts.append(
                f"the most shared demographic trait is {top_demo['demographic_type']} = {top_demo['demographic_value']} "
                f"({int(top_demo['supporting_cases'])} supporting cases)"
            )
        if not similar_cases.empty:
            top_case = similar_cases.iloc[0]
            summary_parts.append(
                f"the closest retrieved case is #{int(top_case['case_id'])} "
                f"with similarity score {float(top_case['similarity_score']):.1f}"
            )

        return "Graph context indicates that " + "; ".join(summary_parts) + "."

    def build_context(self, case_data: dict[str, Any], prediction_label: str, risk_score: float, top_k: int) -> GraphContextResult:
        upload_id = self.upsert_case_context(case_data, prediction_label, risk_score)
        similar_cases = self.get_similar_cases(upload_id, top_k=top_k)
        shared_behaviors = self.get_shared_behavioral_traits(upload_id)
        shared_demographics = self.get_shared_demographic_traits(upload_id)
        summary = self.get_case_pattern_summary(upload_id)
        return GraphContextResult(
            upload_id=upload_id,
            similar_cases=similar_cases,
            shared_behaviors=shared_behaviors,
            shared_demographics=shared_demographics,
            neighborhood_summary=summary,
        )

    def health_checks(self) -> dict[str, Any]:
        case_count = self.neo4j.run_read("MATCH (c:Case) RETURN count(c) AS count")[0]["count"]
        relation_count = self.neo4j.run_read("MATCH ()-[r]->() RETURN count(r) AS count")[0]["count"]
        return {"case_count": case_count, "relationship_count": relation_count}
