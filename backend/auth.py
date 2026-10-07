from __future__ import annotations

import hashlib
import hmac
import os
import secrets
import sqlite3
from contextlib import closing
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
    all_departments: bool = True
    department_ids: tuple[int, ...] = ()


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
        with closing(self._connect()) as conn, conn:
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
            columns = {row[1] for row in conn.execute("PRAGMA table_info(users)").fetchall()}
            if "all_departments" not in columns:
                conn.execute("ALTER TABLE users ADD COLUMN all_departments INTEGER NOT NULL DEFAULT 1")
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS user_departments (
                    user_id INTEGER NOT NULL,
                    department_id INTEGER NOT NULL,
                    PRIMARY KEY(user_id, department_id),
                    FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE
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
        with closing(self._connect()) as conn, conn:
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
        with closing(self._connect()) as conn, conn:
            rows = conn.execute(
                """
                SELECT id, username, role, active, created_at, all_departments
                FROM users
                ORDER BY role, username
                """
            ).fetchall()
            departments = conn.execute(
                "SELECT user_id, department_id FROM user_departments ORDER BY department_id"
            ).fetchall()
        by_user: dict[int, list[int]] = {}
        for row in departments:
            by_user.setdefault(int(row["user_id"]), []).append(int(row["department_id"]))
        users = []
        for row in rows:
            user = dict(row)
            user["all_departments"] = bool(user["all_departments"]) or user["role"] == "admin"
            user["department_ids"] = by_user.get(int(user["id"]), [])
            users.append(user)
        return users

    def create_user(
        self, username: str, password: str, role: Role, all_departments: bool = True, department_ids: list[int] | None = None
    ) -> dict:
        username = username.strip()
        if not username:
            raise ValueError("Username obbligatorio.")
        if len(password) < 4:
            raise ValueError("La password deve avere almeno 4 caratteri.")
        if role not in {"admin", "user"}:
            raise ValueError("Ruolo non valido.")
        if role == "user" and not all_departments and not department_ids:
            raise ValueError("Seleziona almeno un reparto oppure abilita Tutti.")
        try:
            with closing(self._connect()) as conn, conn:
                cursor = conn.execute(
                    """
                    INSERT INTO users (username, password_hash, role, active, created_at, all_departments)
                    VALUES (?, ?, ?, 1, ?, ?)
                    """,
                    (username, self.hash_password(password), role, self._now(), int(all_departments or role == "admin")),
                )
                user_id = int(cursor.lastrowid)
                self._replace_departments(conn, user_id, [] if role == "admin" else department_ids or [])
        except sqlite3.IntegrityError as exc:
            raise ValueError("Username gia' presente.") from exc
        return {
            "id": user_id,
            "username": username,
            "role": role,
            "active": True,
            "created_at": self._now(),
            "all_departments": bool(all_departments or role == "admin"),
            "department_ids": [] if role == "admin" else sorted(set(department_ids or [])),
        }

    def update_departments(self, user_id: int, all_departments: bool, department_ids: list[int]) -> dict:
        with closing(self._connect()) as conn, conn:
            row = conn.execute("SELECT role FROM users WHERE id = ?", (user_id,)).fetchone()
            if not row:
                raise ValueError("Utente non trovato.")
            effective_all = bool(all_departments or row["role"] == "admin")
            if row["role"] == "user" and not effective_all and not department_ids:
                raise ValueError("Seleziona almeno un reparto oppure abilita Tutti.")
            conn.execute("UPDATE users SET all_departments = ? WHERE id = ?", (int(effective_all), user_id))
            self._replace_departments(conn, user_id, [] if effective_all else department_ids)
        return next(user for user in self.list_users() if int(user["id"]) == user_id)

    @staticmethod
    def _replace_departments(conn: sqlite3.Connection, user_id: int, department_ids: list[int]) -> None:
        conn.execute("DELETE FROM user_departments WHERE user_id = ?", (user_id,))
        values = sorted({int(value) for value in department_ids})
        conn.executemany(
            "INSERT INTO user_departments (user_id, department_id) VALUES (?, ?)",
            [(user_id, value) for value in values],
        )

    def authenticate(self, username: str, password: str) -> CurrentUser | None:
        with closing(self._connect()) as conn, conn:
            row = conn.execute(
                """
                SELECT id, username, password_hash, role, all_departments
                FROM users
                WHERE username = ? AND active = 1
                """,
                (username.strip(),),
            ).fetchone()
        if not row or not self.verify_password(password, row["password_hash"]):
            return None
        return self._user_from_row(row)

    def create_session(self, user_id: int) -> str:
        token = secrets.token_urlsafe(36)
        now = datetime.utcnow()
        expires = now + timedelta(days=7)
        with closing(self._connect()) as conn, conn:
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
        with closing(self._connect()) as conn, conn:
            row = conn.execute(
                """
                SELECT users.id, users.username, users.role, users.all_departments
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
        return self._user_from_row(row)

    def _user_from_row(self, row: sqlite3.Row) -> CurrentUser:
        with closing(self._connect()) as conn, conn:
            departments = conn.execute(
                "SELECT department_id FROM user_departments WHERE user_id = ? ORDER BY department_id",
                (int(row["id"]),),
            ).fetchall()
        return CurrentUser(
            id=int(row["id"]),
            username=row["username"],
            role=row["role"],
            all_departments=bool(row["all_departments"]) or row["role"] == "admin",
            department_ids=tuple(int(item["department_id"]) for item in departments),
        )

    def delete_session(self, token: str) -> None:
        with closing(self._connect()) as conn, conn:
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
