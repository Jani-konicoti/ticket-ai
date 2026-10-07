from __future__ import annotations

from dataclasses import dataclass

import bcrypt
import mysql.connector
from sshtunnel import SSHTunnelForwarder

from .models import DatabaseConfig


class PortalAuthUnavailable(RuntimeError):
    pass


@dataclass(frozen=True)
class PortalIdentity:
    staff_id: int
    username: str
    is_admin: bool
    department_ids: tuple[int, ...]


class OsTicketAuthenticator:
    def authenticate(self, config: DatabaseConfig, username: str, password: str) -> PortalIdentity | None:
        if not all((config.ssh_host, config.ssh_user, config.db_user, config.db_name)):
            raise PortalAuthUnavailable("La connessione al database osTicket non e' configurata.")

        try:
            with SSHTunnelForwarder(
                (config.ssh_host, config.ssh_port),
                ssh_username=config.ssh_user,
                ssh_password=config.ssh_password,
                remote_bind_address=(config.db_host, config.db_port),
            ) as tunnel:
                tunnel.start()
                connection = mysql.connector.connect(
                    host="127.0.0.1",
                    port=tunnel.local_bind_port,
                    user=config.db_user,
                    password=config.db_password,
                    database=config.db_name,
                    use_pure=True,
                    connection_timeout=10,
                )
                try:
                    return self._authenticate_connection(connection, username, password)
                finally:
                    connection.close()
        except PortalAuthUnavailable:
            raise
        except Exception as exc:
            raise PortalAuthUnavailable("Impossibile contattare il portale ticket per verificare le credenziali.") from exc

    @staticmethod
    def _authenticate_connection(connection: object, username: str, password: str) -> PortalIdentity | None:
        cursor = connection.cursor()  # type: ignore[attr-defined]
        try:
            cursor.execute(
                """
                SELECT staff_id, username, passwd, backend, isactive, isadmin, dept_id
                FROM ost_staff
                WHERE username = %s
                LIMIT 1
                """,
                (username.strip(),),
            )
            row = cursor.fetchone()
            if not row or not bool(row[4]):
                return None

            backend = str(row[3] or "").strip().lower()
            if backend not in {"", "local"}:
                raise PortalAuthUnavailable(
                    f"L'utente usa il backend di autenticazione '{backend}', non verificabile dal database locale."
                )

            stored_hash = str(row[2] or "")
            if not stored_hash.startswith(("$2a$", "$2b$", "$2y$")):
                raise PortalAuthUnavailable("Il formato password dell'utente osTicket non e' bcrypt.")
            if len(password.encode("utf-8")) > 72:
                return None
            if not bcrypt.checkpw(password.encode("utf-8"), stored_hash.encode("utf-8")):
                return None

            staff_id = int(row[0])
            primary_department = int(row[6] or 0)
            cursor.execute(
                """
                SELECT dept_id
                FROM ost_staff_dept_access
                WHERE staff_id = %s
                ORDER BY dept_id
                """,
                (staff_id,),
            )
            department_ids = {int(item[0]) for item in cursor.fetchall() if int(item[0] or 0) > 0}
            if primary_department > 0:
                department_ids.add(primary_department)
            return PortalIdentity(
                staff_id=staff_id,
                username=str(row[1]),
                is_admin=bool(row[5]),
                department_ids=tuple(sorted(department_ids)),
            )
        finally:
            cursor.close()
