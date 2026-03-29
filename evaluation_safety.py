from __future__ import annotations

import json
import os
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import pandas as pd
from neo4j import GraphDatabase
from sklearn.model_selection import train_test_split


RANDOM_STATE = 42
TEST_SIZE = 0.30


@dataclass(frozen=True)
class SafetyAudit:
    duplicate_rows_removed: int
    deterministic_target_from_answers: bool
    deterministic_rule_note: str


def deduplicate_dataset(df: pd.DataFrame, feature_columns: Sequence[str]) -> Tuple[pd.DataFrame, int]:
    dedup_subset = list(feature_columns) + ["target"]
    deduped = df.drop_duplicates(subset=dedup_subset).copy()
    return deduped, int(len(df) - len(deduped))


def detect_deterministic_qchat_rule(df: pd.DataFrame) -> Tuple[bool, str]:
    answer_cols = [f"A{i}" for i in range(1, 11)]
    if not all(col in df.columns for col in answer_cols) or "target" not in df.columns:
        return False, "Q-Chat determinism check skipped because required columns were missing."

    answer_sum = df[answer_cols].apply(pd.to_numeric, errors="coerce").fillna(0).sum(axis=1)
    deterministic = bool(((answer_sum >= 4).astype(int) == df["target"].astype(int)).all())
    if deterministic:
        return True, "Target is exactly reconstructed by sum(A1..A10) >= 4."
    return False, "Target is not exactly reconstructed by sum(A1..A10) >= 4."


def build_safety_audit(df: pd.DataFrame, feature_columns: Sequence[str]) -> Tuple[pd.DataFrame, SafetyAudit]:
    deduped, removed = deduplicate_dataset(df, feature_columns)
    deterministic, note = detect_deterministic_qchat_rule(deduped)
    return deduped, SafetyAudit(
        duplicate_rows_removed=removed,
        deterministic_target_from_answers=deterministic,
        deterministic_rule_note=note,
    )


def stratified_split(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    train_df, test_df = train_test_split(
        df,
        test_size=TEST_SIZE,
        stratify=df["target"],
        random_state=RANDOM_STATE,
    )
    return train_df.copy(), test_df.copy()


def get_neo4j_driver(args: Any):
    if not args.neo4j_uri or not args.neo4j_user or not args.neo4j_password:
        return None
    return GraphDatabase.driver(args.neo4j_uri, auth=(args.neo4j_user, args.neo4j_password))


def _builder_env(args: Any) -> Dict[str, str]:
    env = os.environ.copy()
    env.update(
        {
            "NEO4J_URI": args.neo4j_uri or "",
            "NEO4J_USER": args.neo4j_user or "",
            "NEO4J_PASSWORD": args.neo4j_password or "",
            "NEO4J_DB": args.neo4j_database or "neo4j",
        }
    )
    return env


def cleanup_temporary_case(args: Any, upload_id: str) -> None:
    driver = get_neo4j_driver(args)
    if driver is None:
        return
    try:
        with driver.session(database=args.neo4j_database) as session:
            session.run(
                """
                MATCH (c:Case {upload_id: $upload_id})
                DETACH DELETE c
                """,
                upload_id=upload_id,
            ).consume()
    finally:
        driver.close()


def fetch_embeddings(args: Any, case_ids: Sequence[int]) -> pd.DataFrame:
    driver = get_neo4j_driver(args)
    if driver is None:
        raise ValueError("Neo4j credentials are required for graph-based experiments.")

    case_id_rows = [{"case_id": int(case_id)} for case_id in case_ids]
    try:
        with driver.session(database=args.neo4j_database) as session:
            records = session.run(
                """
                UNWIND $rows AS row
                MATCH (c:Case {id: row.case_id})
                RETURN c.id AS case_id, c.embedding AS embedding
                """,
                rows=case_id_rows,
            )
            rows = [record.data() for record in records]
    finally:
        driver.close()

    embeddings: List[Dict[str, Any]] = []
    for row in rows:
        embedding = row.get("embedding")
        if embedding is None or len(embedding) != 128:
            continue
        embeddings.append(
            {
                "Case_No": int(row["case_id"]),
                **{f"emb_{i}": float(value) for i, value in enumerate(embedding)},
            }
        )

    if not embeddings:
        raise ValueError("No graph embeddings were found in Neo4j for the requested cases.")

    return pd.DataFrame(embeddings)


def _run_builder(
    repo_root: Path,
    args: Any,
    csv_path: Path,
    extra_builder_args: Sequence[str],
    mode: str,
) -> None:
    cmd = [sys_executable(), str(repo_root / "kg_builder_2.py")]
    cmd.extend(extra_builder_args)
    if mode == "build":
        cmd.extend(["--build-full-graph", "--csv-path", str(csv_path)])
    elif mode == "embed":
        cmd.append("--no-labels")
    else:
        raise ValueError(f"Unsupported builder mode: {mode}")

    result = subprocess.run(cmd, env=_builder_env(args), capture_output=True, text=True, check=False)
    if result.returncode != 0:
        stderr = result.stderr.strip() or result.stdout.strip() or "No subprocess output was returned."
        raise RuntimeError(f"Builder failed in mode '{mode}': {stderr}")


def _python_json_safe(value: Any) -> Any:
    if pd.isna(value):
        return None
    if hasattr(value, "item"):
        return value.item()
    return value


def _insert_batch_test_cases(args: Any, test_df: pd.DataFrame, upload_prefix: str) -> None:
    driver = get_neo4j_driver(args)
    if driver is None:
        raise ValueError("Neo4j credentials are required for batch test embedding inference.")

    case_rows: List[Dict[str, Any]] = []
    answer_rows: List[Dict[str, Any]] = []
    demo_rows: List[Dict[str, Any]] = []
    submitter_rows: List[Dict[str, Any]] = []

    for _, row in test_df.iterrows():
        case_id = int(row["Case_No"])
        upload_id = f"{upload_prefix}{case_id}"
        case_rows.append({"case_id": case_id, "upload_id": upload_id})

        for i in range(1, 11):
            question = f"A{i}"
            if question in row:
                raw_value = pd.to_numeric(row[question], errors="coerce")
                answer_rows.append(
                    {
                        "upload_id": upload_id,
                        "question": question,
                        "value": int(raw_value) if not pd.isna(raw_value) else 0,
                    }
                )

        for field in ["Sex", "Ethnicity", "Jaundice", "Family_mem_with_ASD"]:
            if field in row:
                value = str(row[field]).strip()
                if value and value.lower() not in {"nan", "none", ""}:
                    demo_rows.append({"upload_id": upload_id, "field": field, "value": value})

        if "Who_completed_the_test" in row:
            submitter = str(row["Who_completed_the_test"]).strip()
            if submitter and submitter.lower() not in {"nan", "none", ""}:
                submitter_rows.append({"upload_id": upload_id, "submitter": submitter})

    try:
        with driver.session(database=args.neo4j_database) as session:
            session.run(
                """
                UNWIND $rows AS row
                CREATE (c:Case {id: row.case_id, upload_id: row.upload_id})
                SET c.embedding = null
                """,
                rows=case_rows,
            ).consume()

            session.run(
                """
                UNWIND $rows AS row
                MATCH (c:Case {upload_id: row.upload_id})
                MATCH (q:BehaviorQuestion {name: row.question})
                MERGE (c)-[r:HAS_ANSWER]->(q)
                SET r.value = row.value
                """,
                rows=answer_rows,
            ).consume()

            if demo_rows:
                session.run(
                    """
                    UNWIND $rows AS row
                    MATCH (c:Case {upload_id: row.upload_id})
                    MATCH (d:DemographicAttribute {type: row.field, value: row.value})
                    MERGE (c)-[:HAS_DEMOGRAPHIC]->(d)
                    """,
                    rows=demo_rows,
                ).consume()

            if submitter_rows:
                session.run(
                    """
                    UNWIND $rows AS row
                    MATCH (c:Case {upload_id: row.upload_id})
                    MATCH (s:SubmitterType {type: row.submitter})
                    MERGE (c)-[:SUBMITTED_BY]->(s)
                    """,
                    rows=submitter_rows,
                ).consume()
    finally:
        driver.close()


def _cleanup_batch_test_cases(args: Any, upload_prefix: str) -> None:
    driver = get_neo4j_driver(args)
    if driver is None:
        return
    try:
        with driver.session(database=args.neo4j_database) as session:
            session.run(
                """
                MATCH (c:Case)
                WHERE c.upload_id STARTS WITH $upload_prefix
                DETACH DELETE c
                """,
                upload_prefix=upload_prefix,
            ).consume()
    finally:
        driver.close()


def _infer_test_embeddings_batch(
    repo_root: Path,
    args: Any,
    test_df: pd.DataFrame,
    extra_builder_args: Sequence[str],
) -> pd.DataFrame:
    upload_prefix = "eval_batch_"
    try:
        _insert_batch_test_cases(args, test_df, upload_prefix)
        _run_builder(repo_root, args, Path("unused.csv"), extra_builder_args, mode="embed")
        return fetch_embeddings(args, test_df["Case_No"].tolist())
    finally:
        _cleanup_batch_test_cases(args, upload_prefix)


def sys_executable() -> str:
    return os.environ.get("PYTHON_EXECUTABLE_FOR_EVAL", os.sys.executable)


def build_inductive_graph_embeddings(
    repo_root: Path,
    args: Any,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    csv_columns: Sequence[str],
    extra_builder_args: Sequence[str] | None = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    extra_builder_args = list(extra_builder_args or [])
    with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False, encoding="utf-8-sig") as tmp_file:
        train_csv_path = Path(tmp_file.name)
        train_df.loc[:, list(csv_columns)].to_csv(train_csv_path, sep=";", index=False, encoding="utf-8-sig")

    try:
        _run_builder(repo_root, args, train_csv_path, extra_builder_args, mode="build")
        _run_builder(repo_root, args, train_csv_path, extra_builder_args, mode="embed")

        train_embedding_df = fetch_embeddings(args, train_df["Case_No"].tolist())
        test_embedding_df = _infer_test_embeddings_batch(repo_root, args, test_df, extra_builder_args)

        merged_train = train_df.merge(train_embedding_df, on="Case_No", how="inner").copy()
        merged_test = test_df.merge(test_embedding_df, on="Case_No", how="inner").copy()

        metadata = {
            "train_rows_requested": int(len(train_df)),
            "test_rows_requested": int(len(test_df)),
            "train_rows_with_embeddings": int(len(merged_train)),
            "test_rows_with_embeddings": int(len(merged_test)),
            "builder_args": list(extra_builder_args),
            "graph_protocol": "train-only graph build with batch unlabeled test-case insertion and label-free embedding refresh",
        }
        return merged_train, merged_test, metadata
    finally:
        train_csv_path.unlink(missing_ok=True)
