from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any


ANALYSIS_PERIODS = (7, 30, 90, 180)


def now_text() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


class AnalysisCacheStore:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS recent_problem_analysis (
                    days INTEGER PRIMARY KEY,
                    payload TEXT NOT NULL,
                    generated_at TEXT NOT NULL,
                    vector_count INTEGER NOT NULL DEFAULT 0,
                    ticket_count INTEGER NOT NULL DEFAULT 0
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS app_state (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                )
                """
            )

    def get(self, days: int) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT payload, generated_at, vector_count
                FROM recent_problem_analysis
                WHERE days = ?
                """,
                (days,),
            ).fetchone()
        if not row:
            return None
        payload = json.loads(row["payload"])
        payload["generated_at"] = row["generated_at"]
        payload["vector_count"] = int(row["vector_count"])
        return payload

    def save(self, days: int, payload: dict[str, Any], vector_count: int, ticket_count: int) -> dict[str, Any]:
        generated_at = now_text()
        saved_payload = {**payload, "generated_at": generated_at, "vector_count": vector_count}
        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO recent_problem_analysis
                    (days, payload, generated_at, vector_count, ticket_count)
                VALUES (?, ?, ?, ?, ?)
                """,
                (days, json.dumps(saved_payload, ensure_ascii=False), generated_at, vector_count, ticket_count),
            )
        return saved_payload

    def get_state(self, key: str) -> str | None:
        with self._connect() as conn:
            row = conn.execute("SELECT value FROM app_state WHERE key = ?", (key,)).fetchone()
        return str(row["value"]) if row else None

    def set_state(self, key: str, value: str) -> None:
        with self._connect() as conn:
            conn.execute("INSERT OR REPLACE INTO app_state (key, value) VALUES (?, ?)", (key, value))
