from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Iterator

from neo4j import GraphDatabase

from app.config import NEO4J_DB, NEO4J_PASSWORD, NEO4J_URI, NEO4J_USER
from app.utils.logging_utils import get_logger

logger = get_logger(__name__)


class Neo4jService:
    """Thin wrapper around the Neo4j Bolt driver."""

    def __init__(
        self,
        uri: str = NEO4J_URI,
        user: str = NEO4J_USER,
        password: str = NEO4J_PASSWORD,
        database: str = NEO4J_DB,
    ):
        self._driver = GraphDatabase.driver(uri, auth=(user, password))
        self._database = database

    @contextmanager
    def session(self) -> Iterator[Any]:
        session = self._driver.session(database=self._database) if self._database else self._driver.session()
        try:
            yield session
        finally:
            session.close()

    def close(self) -> None:
        self._driver.close()

    def run_read(self, query: str, parameters: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        with self.session() as session:
            result = session.run(query, parameters or {})
            return [record.data() for record in result]

    def run_write(self, query: str, parameters: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        with self.session() as session:
            result = session.run(query, parameters or {})
            return [record.data() for record in result]
