from __future__ import annotations

import hashlib
import hmac
import os
import secrets
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Literal


Role = Literal["admin", "user"]


@dataclass(frozen=True)
class CurrentUser:
    id: int
    username: str
    role: Role


class AuthStore:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()
        self._seed_admin()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT NOT NULL UNIQUE,
                    password_hash TEXT NOT NULL,
                    role TEXT NOT NULL CHECK(role IN ('admin', 'user')),
                    active INTEGER NOT NULL DEFAULT 1,
                    created_at TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS auth_sessions (
                    token TEXT PRIMARY KEY,
                    user_id INTEGER NOT NULL,
                    created_at TEXT NOT NULL,
                    expires_at TEXT NOT NULL,
                    FOREIGN KEY(user_id) REFERENCES users(id)
                )
                """
            )

    def _seed_admin(self) -> None:
        with self._connect() as conn:
            count = conn.execute("SELECT COUNT(*) FROM users").fetchone()[0]
            if count:
                return
            conn.execute(
                """
                INSERT INTO users (username, password_hash, role, active, created_at)
                VALUES (?, ?, 'admin', 1, ?)
                """,
                ("admin", self.hash_password("admin"), self._now()),
            )

    def list_users(self) -> list[dict]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT id, username, role, active, created_at
                FROM users
                ORDER BY role, username
                """
            ).fetchall()
        return [dict(row) for row in rows]

    def create_user(self, username: str, password: str, role: Role) -> dict:
        username = username.strip()
        if not username:
            raise ValueError("Username obbligatorio.")
        if len(password) < 4:
            raise ValueError("La password deve avere almeno 4 caratteri.")
        if role not in {"admin", "user"}:
            raise ValueError("Ruolo non valido.")
        try:
            with self._connect() as conn:
                cursor = conn.execute(
                    """
                    INSERT INTO users (username, password_hash, role, active, created_at)
                    VALUES (?, ?, ?, 1, ?)
                    """,
                    (username, self.hash_password(password), role, self._now()),
                )
                user_id = int(cursor.lastrowid)
        except sqlite3.IntegrityError as exc:
            raise ValueError("Username gia' presente.") from exc
        return {
            "id": user_id,
            "username": username,
            "role": role,
            "active": True,
            "created_at": self._now(),
        }

    def authenticate(self, username: str, password: str) -> CurrentUser | None:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT id, username, password_hash, role
                FROM users
                WHERE username = ? AND active = 1
                """,
                (username.strip(),),
            ).fetchone()
        if not row or not self.verify_password(password, row["password_hash"]):
            return None
        return CurrentUser(id=int(row["id"]), username=row["username"], role=row["role"])

    def create_session(self, user_id: int) -> str:
        token = secrets.token_urlsafe(36)
        now = datetime.utcnow()
        expires = now + timedelta(days=7)
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO auth_sessions (token, user_id, created_at, expires_at)
                VALUES (?, ?, ?, ?)
                """,
                (token, user_id, now.isoformat(), expires.isoformat()),
            )
        return token

    def user_for_token(self, token: str) -> CurrentUser | None:
        if not token:
            return None
        now = datetime.utcnow().isoformat()
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT users.id, users.username, users.role
                FROM auth_sessions
                JOIN users ON users.id = auth_sessions.user_id
                WHERE auth_sessions.token = ?
                  AND auth_sessions.expires_at > ?
                  AND users.active = 1
                """,
                (token, now),
            ).fetchone()
        if not row:
            return None
        return CurrentUser(id=int(row["id"]), username=row["username"], role=row["role"])

    def delete_session(self, token: str) -> None:
        with self._connect() as conn:
            conn.execute("DELETE FROM auth_sessions WHERE token = ?", (token,))

    @staticmethod
    def hash_password(password: str) -> str:
        salt = os.urandom(16)
        digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, 120_000)
        return f"pbkdf2_sha256${salt.hex()}${digest.hex()}"

    @staticmethod
    def verify_password(password: str, stored: str) -> bool:
        try:
            method, salt_hex, digest_hex = stored.split("$", 2)
        except ValueError:
            return False
        if method != "pbkdf2_sha256":
            return False
        salt = bytes.fromhex(salt_hex)
        expected = bytes.fromhex(digest_hex)
        actual = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, 120_000)
        return hmac.compare_digest(actual, expected)

    @staticmethod
    def _now() -> str:
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
