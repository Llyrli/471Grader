"""Shared Postgres connection helper for the JN Grader task-bank layer.

Connection comes from ``DATABASE_URL`` (e.g. set by docker-compose to
``postgresql://grader:grader@db:5432/jngrader``); falls back to localhost so the
scripts also work outside Docker.
"""

from __future__ import annotations

import os
from pathlib import Path

import psycopg

DEFAULT_DSN = "postgresql://grader:grader@localhost:5432/jngrader"
SCHEMA_PATH = Path(__file__).parent / "db" / "schema.sql"


def dsn() -> str:
    return os.environ.get("DATABASE_URL", DEFAULT_DSN)


def connect() -> psycopg.Connection:
    return psycopg.connect(dsn())


def apply_schema(conn: psycopg.Connection) -> None:
    """Create extension + tables if they don't exist (idempotent)."""
    sql = SCHEMA_PATH.read_text(encoding="utf-8")
    with conn.cursor() as cur:
        cur.execute(sql)
    conn.commit()
