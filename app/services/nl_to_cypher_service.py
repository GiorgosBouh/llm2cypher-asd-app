from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

from openai import OpenAI

from app.config import OPENAI_API_KEY, OPENAI_MODEL
from app.services.neo4j_service import Neo4jService
from app.utils.logging_utils import get_logger

logger = get_logger(__name__)


READ_ONLY_BLOCKLIST = re.compile(
    r"\b(CREATE|MERGE|DELETE|DETACH|SET|REMOVE|DROP|LOAD\s+CSV|CALL\s+dbms|CALL\s+apoc)\b",
    re.IGNORECASE,
)

INVALID_SCHEMA_PATTERNS = [
    (re.compile(r"\bage_group\b", re.IGNORECASE), "There is no `Case.age_group` property in this graph."),
    (re.compile(r"\battribute\b", re.IGNORECASE), "Use `DemographicAttribute.type` rather than `attribute`."),
]

QUERY_START_PATTERN = re.compile(r"^\s*(MATCH|OPTIONAL MATCH|WITH|UNWIND|RETURN)\b", re.IGNORECASE)


@dataclass
class NLQueryResult:
    cypher: str | None
    rows: list[dict]
    explanation: str
    error: str | None = None
    source: str | None = None


class NLToCypherService:
    """Generate and execute schema-aware, read-only Cypher from natural language."""

    def __init__(self, neo4j_service: Neo4jService):
        self.neo4j = neo4j_service
        self.client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

    def _normalize_question(self, question: str) -> str:
        normalized = re.sub(r"\s+", " ", question.strip())
        synonym_replacements = [
            (r"\bversus\b", " vs "),
            (r"\bv\.s\.\b", " vs "),
            (r"\bcompared to\b", " compare "),
            (r"\bpositive cases\b", " asd positive "),
            (r"\bnegative cases\b", " asd negative "),
            (r"\bmost frequent\b", " most common "),
            (r"\bfrequent\b", " common "),
            (r"\bshared demographics\b", " demographics "),
            (r"\bdemographic profile\b", " demographics "),
            (r"\bbehavioral patterns\b", " behavior patterns "),
            (r"\bbehavioural patterns\b", " behavior patterns "),
            (r"\bbehaviour\b", " behavior "),
        ]
        for pattern, replacement in synonym_replacements:
            normalized = re.sub(pattern, replacement, normalized, flags=re.IGNORECASE)
        return re.sub(r"\s+", " ", normalized).strip()

    def _extract_cypher(self, raw_text: str) -> str:
        text = (raw_text or "").strip()
        text = text.replace("```cypher", "```").strip()
        if text.startswith("```") and text.endswith("```"):
            text = text[3:-3].strip()
        if text.lower().startswith("cypher:"):
            text = text.split(":", 1)[1].strip()

        lines = [line.rstrip() for line in text.splitlines() if line.strip()]
        for idx, line in enumerate(lines):
            if QUERY_START_PATTERN.match(line):
                return "\n".join(lines[idx:]).strip()
        return text

    def _behavior_conditions(self, question: str) -> list[tuple[str, int]]:
        matches = re.findall(r"\b(a(?:10|[1-9]))\s*=\s*([01])\b", question, flags=re.IGNORECASE)
        return [(name.upper(), int(value)) for name, value in matches]

    def _count_limit(self, question: str, default: int = 10) -> int:
        match = re.search(r"\b(?:show|list|display|return)\s+(\d+)\b", question, flags=re.IGNORECASE)
        return int(match.group(1)) if match else default

    def _validate_schema_semantics(self, cypher: str) -> None:
        if re.search(r"\bDemographicAttribute\s*\{\s*type\s*:\s*['\"]A(?:10|[1-9])['\"]", cypher, flags=re.IGNORECASE):
            raise ValueError("A1-A10 are behavior answers, not demographic attribute types.")
        if re.search(r"\bd\.(?:type|value)\s*=\s*['\"]A(?:10|[1-9])['\"]", cypher, flags=re.IGNORECASE):
            raise ValueError("A1-A10 should be matched through BehaviorQuestion.name, not demographic fields.")

    def _has_any(self, text: str, phrases: list[str]) -> bool:
        return any(phrase in text for phrase in phrases)

    def _parse_compare_behavior(self, question: str) -> Optional[str]:
        match = re.search(
            r"\bcompare\s+(a(?:10|[1-9]))\s*=\s*([01])\s*(?:vs|and)\s*\1\s*=\s*([01])\s+by\s+asd label\b",
            question,
            flags=re.IGNORECASE,
        )
        if not match:
            return None
        item = match.group(1).upper()
        left_value = int(match.group(2))
        right_value = int(match.group(3))
        return f"""
MATCH (c:Case)-[r:HAS_ANSWER]->(:BehaviorQuestion {{name: '{item}'}})
MATCH (c)-[:SCREENED_FOR]->(t:ASD_Trait)
WHERE r.value IN [{left_value}, {right_value}] AND coalesce(c.temp_case, false) = false
RETURN r.value AS answer_value,
       t.label AS asd_label,
       count(DISTINCT c) AS case_count
ORDER BY answer_value ASC, asd_label ASC
""".strip()

    def _parse_top_demographics_for_behavior(self, question: str) -> Optional[str]:
        match = re.search(
            r"\b(?:top|most common|common)\s+demographics\b.*\bfor cases with\s+(a(?:10|[1-9]))\s*=\s*([01])\b",
            question,
            flags=re.IGNORECASE,
        )
        if not match:
            match = re.search(
                r"\b(?:top|most common|common)\s+demographics\b.*\bamong cases with\s+(a(?:10|[1-9]))\s*=\s*([01])\b",
                question,
                flags=re.IGNORECASE,
            )
        if not match:
            return None
        item = match.group(1).upper()
        value = int(match.group(2))
        limit = self._count_limit(question, default=10)
        return f"""
MATCH (c:Case)-[r:HAS_ANSWER]->(:BehaviorQuestion {{name: '{item}'}})
MATCH (c)-[:HAS_DEMOGRAPHIC]->(d:DemographicAttribute)
WHERE r.value = {value} AND coalesce(c.temp_case, false) = false
RETURN d.type AS demographic_type,
       d.value AS demographic_value,
       count(DISTINCT c) AS case_count
ORDER BY case_count DESC, demographic_type ASC, demographic_value ASC
LIMIT {limit}
""".strip()

    def _parse_common_behavior_patterns(self, question: str) -> Optional[str]:
        q = question.lower()
        if not (
            self._has_any(q, ["most common behavior patterns", "common behavior patterns", "behavior patterns"])
            and self._has_any(q, ["asd positive", "screened positive", "positive for asd", "asd-positive"])
        ):
            return None
        limit = self._count_limit(question, default=15)
        return f"""
MATCH (c:Case)-[:SCREENED_FOR]->(:ASD_Trait {{label: 'Yes'}})
MATCH (c)-[r:HAS_ANSWER]->(q:BehaviorQuestion)
WHERE coalesce(c.temp_case, false) = false
RETURN q.name AS question,
       r.value AS answer_value,
       count(DISTINCT c) AS case_count
ORDER BY case_count DESC, question ASC, answer_value DESC
LIMIT {limit}
""".strip()

    def _parse_asd_distribution_for_behavior(self, question: str) -> Optional[str]:
        match = re.search(
            r"\b(?:asd label distribution|distribution by asd label|asd breakdown|breakdown by asd label)\b.*\b(a(?:10|[1-9]))\s*=\s*([01])\b",
            question,
            flags=re.IGNORECASE,
        )
        if not match:
            match = re.search(
                r"\bfor cases with\s+(a(?:10|[1-9]))\s*=\s*([01])\b.*\b(?:asd label distribution|distribution by asd label|asd breakdown|breakdown by asd label)\b",
                question,
                flags=re.IGNORECASE,
            )
        if not match:
            return None
        item = match.group(1).upper()
        value = int(match.group(2))
        return f"""
MATCH (c:Case)-[r:HAS_ANSWER]->(:BehaviorQuestion {{name: '{item}'}})
MATCH (c)-[:SCREENED_FOR]->(t:ASD_Trait)
WHERE r.value = {value} AND coalesce(c.temp_case, false) = false
RETURN t.label AS asd_label,
       count(DISTINCT c) AS case_count
ORDER BY case_count DESC, asd_label ASC
""".strip()

    def _parse_top_cooccurring_behaviors(self, question: str) -> Optional[str]:
        match = re.search(
            r"\b(?:top|most common|common)\s+(?:cooccurring|co-occurring|associated)\s+(?:answers|behaviors|questions)\b.*\b(?:with|for)\s+(a(?:10|[1-9]))\s*=\s*([01])\b",
            question,
            flags=re.IGNORECASE,
        )
        if not match:
            return None
        item = match.group(1).upper()
        value = int(match.group(2))
        limit = self._count_limit(question, default=15)
        return f"""
MATCH (c:Case)-[r0:HAS_ANSWER]->(:BehaviorQuestion {{name: '{item}'}})
MATCH (c)-[r:HAS_ANSWER]->(q:BehaviorQuestion)
WHERE r0.value = {value} AND q.name <> '{item}' AND coalesce(c.temp_case, false) = false
RETURN q.name AS question,
       r.value AS answer_value,
       count(DISTINCT c) AS case_count
ORDER BY case_count DESC, question ASC, answer_value DESC
LIMIT {limit}
""".strip()

    def _parse_submitter_breakdown(self, question: str) -> Optional[str]:
        q = question.lower()
        if not self._has_any(q, ["submitter breakdown", "submitted by", "who completed the test", "respondent breakdown"]):
            return None
        conditions = self._behavior_conditions(question)
        if not conditions:
            return None
        item, value = conditions[0]
        return f"""
MATCH (c:Case)-[r:HAS_ANSWER]->(:BehaviorQuestion {{name: '{item}'}})
MATCH (c)-[:SUBMITTED_BY]->(s:SubmitterType)
WHERE r.value = {value} AND coalesce(c.temp_case, false) = false
RETURN s.type AS submitter_type,
       count(DISTINCT c) AS case_count
ORDER BY case_count DESC, submitter_type ASC
""".strip()

    def _parse_demographic_breakdown(self, question: str) -> Optional[str]:
        q = question.lower()
        if not self._has_any(q, ["breakdown by sex", "breakdown by ethnicity", "sex breakdown", "ethnicity breakdown"]):
            return None
        conditions = self._behavior_conditions(question)
        if not conditions:
            return None
        demo_type = "Sex" if "sex" in q else "Ethnicity"
        item, value = conditions[0]
        return f"""
MATCH (c:Case)-[r:HAS_ANSWER]->(:BehaviorQuestion {{name: '{item}'}})
MATCH (c)-[:HAS_DEMOGRAPHIC]->(d:DemographicAttribute {{type: '{demo_type}'}})
WHERE r.value = {value} AND coalesce(c.temp_case, false) = false
RETURN d.value AS demographic_value,
       count(DISTINCT c) AS case_count
ORDER BY case_count DESC, demographic_value ASC
""".strip()

    def _parse_similar_cases_for_case(self, question: str) -> Optional[str]:
        match = re.search(
            r"\b(?:similar cases to|cases similar to|neighbors of|similar to case)\s+(\d+)\b",
            question,
            flags=re.IGNORECASE,
        )
        if not match:
            return None
        case_id = int(match.group(1))
        limit = self._count_limit(question, default=10)
        return f"""
MATCH (c:Case {{id: {case_id}}})-[r:SIMILAR_TO]-(other:Case)
OPTIONAL MATCH (other)-[:SCREENED_FOR]->(t:ASD_Trait)
WHERE coalesce(other.temp_case, false) = false
RETURN other.id AS case_id,
       coalesce(t.label, 'No ASD label recorded') AS asd_label,
       r.weight AS similarity_weight
ORDER BY similarity_weight DESC, case_id ASC
LIMIT {limit}
""".strip()

    def _build_rule_based_query(self, question: str) -> Optional[str]:
        q = self._normalize_question(question)
        q_lower = q.lower()
        conditions = self._behavior_conditions(q)

        compare_query = self._parse_compare_behavior(q)
        if compare_query:
            return compare_query

        top_demographics_query = self._parse_top_demographics_for_behavior(q)
        if top_demographics_query:
            return top_demographics_query

        common_patterns_query = self._parse_common_behavior_patterns(q)
        if common_patterns_query:
            return common_patterns_query

        asd_distribution_query = self._parse_asd_distribution_for_behavior(q)
        if asd_distribution_query:
            return asd_distribution_query

        cooccurring_query = self._parse_top_cooccurring_behaviors(q)
        if cooccurring_query:
            return cooccurring_query

        submitter_query = self._parse_submitter_breakdown(q)
        if submitter_query:
            return submitter_query

        demographic_breakdown_query = self._parse_demographic_breakdown(q)
        if demographic_breakdown_query:
            return demographic_breakdown_query

        similar_cases_query = self._parse_similar_cases_for_case(q)
        if similar_cases_query:
            return similar_cases_query

        count_like = self._has_any(q_lower, ["how many", "count", "number of"])
        list_like = self._has_any(q_lower, ["show", "list", "display", "return"])
        positive_label = self._has_any(q_lower, ["screened positive", "positive for asd", "asd traits yes", "asd positive", "asd-positive"])
        negative_label = self._has_any(q_lower, ["screened negative", "negative for asd", "asd traits no", "asd negative", "asd-negative"])
        male = bool(re.search(r"\bmale\b", q_lower))
        female = bool(re.search(r"\bfemale\b", q_lower))

        if count_like and conditions:
            match_lines = []
            where_parts = []
            for idx, (item, value) in enumerate(conditions, start=1):
                rel = f"r{idx}"
                match_lines.append(f"MATCH (c:Case)-[{rel}:HAS_ANSWER]->(:BehaviorQuestion {{name: '{item}'}})")
                where_parts.append(f"{rel}.value = {value}")
            query = "\n".join(match_lines)
            query += f"\nWHERE {' AND '.join(where_parts)} AND coalesce(c.temp_case, false) = false"
            query += "\nRETURN count(DISTINCT c) AS case_count"
            return query

        if list_like and conditions:
            match_lines = []
            where_parts = []
            for idx, (item, value) in enumerate(conditions, start=1):
                rel = f"r{idx}"
                match_lines.append(f"MATCH (c:Case)-[{rel}:HAS_ANSWER]->(:BehaviorQuestion {{name: '{item}'}})")
                where_parts.append(f"{rel}.value = {value}")
            limit = self._count_limit(q)
            query = "\n".join(match_lines)
            query += f"\nWHERE {' AND '.join(where_parts)} AND coalesce(c.temp_case, false) = false"
            query += (
                "\nRETURN coalesce(c.id, c.input_case_no) AS case_id, "
                "coalesce(c.upload_id, toString(c.id)) AS upload_id"
            )
            query += "\nORDER BY case_id ASC"
            query += f"\nLIMIT {limit}"
            return query

        if count_like and (male or female or positive_label or negative_label):
            parts = []
            if male or female:
                parts.append("MATCH (c:Case)-[:HAS_DEMOGRAPHIC]->(sex:DemographicAttribute {type: 'Sex'})")
            if positive_label:
                parts.append("MATCH (c)-[:SCREENED_FOR]->(:ASD_Trait {label: 'Yes'})")
            if negative_label:
                parts.append("MATCH (c)-[:SCREENED_FOR]->(:ASD_Trait {label: 'No'})")

            where_parts = []
            if male:
                where_parts.append("toLower(sex.value) IN ['m', 'male']")
            if female:
                where_parts.append("toLower(sex.value) IN ['f', 'female']")
            where_parts.append("coalesce(c.temp_case, false) = false")

            query = "\n".join(parts) if parts else ""
            if where_parts:
                query += f"\nWHERE {' AND '.join(where_parts)}"
            query += "\nRETURN count(DISTINCT c) AS case_count"
            return query if query.strip() else None

        return None

    def _schema_prompt(self, question: str) -> str:
        return f"""
You translate natural-language questions into one read-only Cypher query for a Neo4j graph used in ASD screening research.

Actual graph schema:

Node labels and properties:
- Case: id, upload_id, temp_case, source, predicted_label, risk_score, input_case_no
- BehaviorQuestion: name
- DemographicAttribute: type, value
- SubmitterType: type
- ASD_Trait: label

Relationships and properties:
- (Case)-[HAS_ANSWER]->(BehaviorQuestion), with relationship property: value
- (Case)-[:HAS_DEMOGRAPHIC]->(DemographicAttribute)
- (Case)-[:SUBMITTED_BY]->(SubmitterType)
- (Case)-[:SCREENED_FOR]->(ASD_Trait)
- (Case)-[:SIMILAR_TO]->(Case)

Important interpretation rules:
- Q-Chat items A1 to A10 are stored as BehaviorQuestion nodes with names 'A1' ... 'A10'.
- The answer itself is stored on the HAS_ANSWER relationship as r.value, and is numeric 0 or 1.
- A1-A10 are not demographic attributes.
- There is no Case.age_group property. The dataset already concerns toddlers, so if the user says toddlers, do not filter by age_group.
- Demographic attributes use DemographicAttribute.type and DemographicAttribute.value.
- ASD labels use ASD_Trait.label with values such as 'Yes' or 'No'.
- Sex values may appear as 'm'/'f'; interpret male as m and female as f unless the user explicitly asks otherwise.

Query-writing rules:
- Return one Cypher query only.
- Read-only only.
- Never use CREATE, MERGE, DELETE, DETACH, SET, REMOVE, DROP, CALL dbms, CALL apoc, or LOAD CSV.
- Use DISTINCT when counting cases.
- Use LIMIT when returning case-level rows unless the user explicitly asks for all rows.
- Prefer exact schema-grounded queries over guesswork.

Examples:
Question: how many toddlers with a1=1
Cypher:
MATCH (c:Case)-[r:HAS_ANSWER]->(q:BehaviorQuestion {{name: 'A1'}})
WHERE r.value = 1
RETURN count(DISTINCT c) AS toddler_count

Question: how many male toddlers screened positive for ASD traits
Cypher:
MATCH (c:Case)-[:HAS_DEMOGRAPHIC]->(d:DemographicAttribute {{type: 'Sex'}})
MATCH (c)-[:SCREENED_FOR]->(t:ASD_Trait {{label: 'Yes'}})
WHERE toLower(d.value) IN ['m', 'male']
RETURN count(DISTINCT c) AS positive_male_toddlers

Question: show 10 cases where A1 = 1 and A5 = 1
Cypher:
MATCH (c:Case)-[r1:HAS_ANSWER]->(:BehaviorQuestion {{name: 'A1'}})
MATCH (c)-[r5:HAS_ANSWER]->(:BehaviorQuestion {{name: 'A5'}})
WHERE r1.value = 1 AND r5.value = 1
RETURN coalesce(c.id, c.input_case_no) AS case_id, coalesce(c.upload_id, toString(c.id)) AS upload_id
ORDER BY case_id ASC
LIMIT 10

Question: compare A1=1 vs A1=0 by ASD label
Cypher:
MATCH (c:Case)-[r:HAS_ANSWER]->(:BehaviorQuestion {{name: 'A1'}})
OPTIONAL MATCH (c)-[:SCREENED_FOR]->(t:ASD_Trait)
WHERE r.value IN [0, 1]
RETURN r.value AS answer_value, coalesce(t.label, 'Unknown') AS asd_label, count(DISTINCT c) AS case_count
ORDER BY answer_value ASC, asd_label ASC

Question: top shared demographics for cases with A5=1
Cypher:
MATCH (c:Case)-[r:HAS_ANSWER]->(:BehaviorQuestion {{name: 'A5'}})
MATCH (c)-[:HAS_DEMOGRAPHIC]->(d:DemographicAttribute)
WHERE r.value = 1
RETURN d.type AS demographic_type, d.value AS demographic_value, count(DISTINCT c) AS case_count
ORDER BY case_count DESC, demographic_type ASC, demographic_value ASC
LIMIT 10

Question: show the most common behavior patterns among ASD-positive toddlers
Cypher:
MATCH (c:Case)-[:SCREENED_FOR]->(:ASD_Trait {{label: 'Yes'}})
MATCH (c)-[r:HAS_ANSWER]->(q:BehaviorQuestion)
RETURN q.name AS question, r.value AS answer_value, count(DISTINCT c) AS case_count
ORDER BY case_count DESC, question ASC, answer_value DESC
LIMIT 15

Question: ASD label distribution for cases with A3 = 1
Cypher:
MATCH (c:Case)-[r:HAS_ANSWER]->(:BehaviorQuestion {{name: 'A3'}})
OPTIONAL MATCH (c)-[:SCREENED_FOR]->(t:ASD_Trait)
WHERE r.value = 1
RETURN coalesce(t.label, 'Unknown') AS asd_label, count(DISTINCT c) AS case_count
ORDER BY case_count DESC, asd_label ASC

Question: top co-occurring answers with A2 = 1
Cypher:
MATCH (c:Case)-[r0:HAS_ANSWER]->(:BehaviorQuestion {{name: 'A2'}})
MATCH (c)-[r:HAS_ANSWER]->(q:BehaviorQuestion)
WHERE r0.value = 1 AND q.name <> 'A2'
RETURN q.name AS question, r.value AS answer_value, count(DISTINCT c) AS case_count
ORDER BY case_count DESC, question ASC, answer_value DESC
LIMIT 15

Question: respondent breakdown for cases with A4 = 1
Cypher:
MATCH (c:Case)-[r:HAS_ANSWER]->(:BehaviorQuestion {{name: 'A4'}})
MATCH (c)-[:SUBMITTED_BY]->(s:SubmitterType)
WHERE r.value = 1
RETURN s.type AS submitter_type, count(DISTINCT c) AS case_count
ORDER BY case_count DESC, submitter_type ASC

Question: breakdown by sex for cases with A6 = 1
Cypher:
MATCH (c:Case)-[r:HAS_ANSWER]->(:BehaviorQuestion {{name: 'A6'}})
MATCH (c)-[:HAS_DEMOGRAPHIC]->(d:DemographicAttribute {{type: 'Sex'}})
WHERE r.value = 1
RETURN d.value AS demographic_value, count(DISTINCT c) AS case_count
ORDER BY case_count DESC, demographic_value ASC

Question: show similar cases to 25
Cypher:
MATCH (c:Case {{id: 25}})-[r:SIMILAR_TO]-(other:Case)
OPTIONAL MATCH (other)-[:SCREENED_FOR]->(t:ASD_Trait)
RETURN other.id AS case_id, coalesce(t.label, 'Unknown') AS asd_label, r.weight AS similarity_weight
ORDER BY similarity_weight DESC, case_id ASC
LIMIT 10

Question: {question}
"""

    def generate_query(self, question: str) -> str:
        rule_based = self._build_rule_based_query(question)
        if rule_based:
            return rule_based
        if self.client is None:
            raise RuntimeError("OPENAI_API_KEY is not configured.")

        response = self.client.chat.completions.create(
            model=OPENAI_MODEL,
            temperature=0,
            messages=[
                {"role": "system", "content": "You are a careful Neo4j Cypher generator. Use only the provided schema."},
                {"role": "user", "content": self._schema_prompt(question)},
            ],
        )
        cypher = response.choices[0].message.content or ""
        return self._extract_cypher(cypher)

    def repair_query(self, question: str, invalid_query: str, validation_error: str) -> str:
        if self.client is None:
            raise RuntimeError("OPENAI_API_KEY is not configured.")
        repair_prompt = (
            self._schema_prompt(question)
            + "\nThe previous generated query was invalid for this schema.\n"
            + f"Invalid query:\n{invalid_query}\n\n"
            + f"Validation error:\n{validation_error}\n\n"
            + "Return one corrected read-only Cypher query only."
        )
        response = self.client.chat.completions.create(
            model=OPENAI_MODEL,
            temperature=0,
            messages=[
                {"role": "system", "content": "Correct the Cypher query so it strictly matches the provided schema."},
                {"role": "user", "content": repair_prompt},
            ],
        )
        cypher = response.choices[0].message.content or ""
        return self._extract_cypher(cypher)

    def validate_query(self, cypher: str, admin_mode: bool = False) -> None:
        if not cypher:
            raise ValueError("Empty Cypher query generated.")
        if not QUERY_START_PATTERN.match(cypher):
            raise ValueError("Generated text is not a valid Cypher query.")
        if not admin_mode and READ_ONLY_BLOCKLIST.search(cypher):
            raise ValueError("Blocked non-read-only Cypher query.")
        for pattern, message in INVALID_SCHEMA_PATTERNS:
            if pattern.search(cypher):
                raise ValueError(message)
        self._validate_schema_semantics(cypher)

    def ask(self, question: str, admin_mode: bool = False) -> NLQueryResult:
        cypher = None
        try:
            cypher = self.generate_query(question)
            source = "deterministic" if self._build_rule_based_query(question) else "llm"
            try:
                self.validate_query(cypher, admin_mode=admin_mode)
            except ValueError as validation_exc:
                cypher = self.repair_query(question, cypher, str(validation_exc))
                source = "llm-repaired"
                self.validate_query(cypher, admin_mode=admin_mode)
            rows = self.neo4j.run_read(cypher)
            explanation = (
                f"The graph answered the question using a {source} read-only Cypher query over cases, demographics, and behavioral responses."
                if rows
                else f"The generated {source} read-only Cypher query ran successfully, but it returned no rows."
            )
            return NLQueryResult(cypher=cypher, rows=rows, explanation=explanation, source=source)
        except Exception as exc:
            logger.exception("NL-to-Cypher query failed")
            return NLQueryResult(
                cypher=cypher,
                rows=[],
                explanation="Cypher generation or execution failed gracefully.",
                error=str(exc),
                source="failed",
            )
