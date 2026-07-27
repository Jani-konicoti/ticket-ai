from __future__ import annotations

import csv
import gc
import json
import logging
import shutil
import sqlite3
import threading
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable
from uuid import uuid4

import faiss
import mysql.connector
import numpy as np
import pandas as pd
import tiktoken
import truststore
from bs4 import BeautifulSoup
from openai import OpenAI
from sshtunnel import SSHTunnelForwarder

from .models import DatabaseConfig


truststore.inject_into_ssl()

EMBEDDING_MODEL = "text-embedding-3-large"
MAX_EMBED_TOKENS = 7200
MAX_EMBED_BATCH_TOKENS = 80000
CHUNK_TOKEN_OVERLAP = 160
CSV_COLUMNS = [
    "id",
    "thread_id",
    "staff_id",
    "user_id",
    "poster",
    "created",
    "title",
    "clean_body",
    "chunk_index",
    "chunk_count",
]
logger = logging.getLogger(__name__)
DEFAULT_QUERY = """
SELECT id, thread_id, staff_id, user_id, poster, created, title, body
FROM ost_thread_entry
WHERE title <> 'Stato modificato'
  AND title <> 'Nuovo collaboratore aggiunto'
  AND title <> 'Ticket aggiornato'
  AND title NOT LIKE '%Nuovo ticket da%'
  AND title NOT LIKE '%Ticket assegnato a%'
  AND LENGTH(body) > 70
""".strip()


def clean_body(html: Any) -> str:
    soup = BeautifulSoup(str(html or ""), "html.parser")
    for tag in soup.find_all(["img", "hr", "style", "script"]):
        tag.decompose()
    for br in soup.find_all("br"):
        br.replace_with("\n")

    text = soup.get_text(separator="\n")
    lines = [line.strip() for line in text.splitlines() if line.strip()]

    cleaned_lines = []
    noise = [
        "informazioni di carattere confidenziale",
        "questo messaggio di posta elettronica",
        "virus",
        "eset",
        "avast",
        "kaspersky",
        "inviato:",
        "da:",
        "to:",
        "from:",
        "subject:",
        "firma",
        "tel",
        "fax",
        "@",
        "www.",
    ]
    for line in lines:
        lowered = line.lower()
        if any(keyword in lowered for keyword in noise):
            continue
        if "il " in lowered and " ha scritto" in lowered:
            continue
        cleaned_lines.append(line)

    return "\n".join(cleaned_lines).strip()


def _now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _strip_query(query: str) -> str:
    return query.strip().rstrip(";")


def _date_filter_query(base_query: str, operator: str) -> str:
    return f"SELECT * FROM ({_strip_query(base_query)}) AS source_query WHERE created {operator} %s ORDER BY created, id"


def _count_query(filtered_query: str) -> str:
    return f"SELECT COUNT(*) FROM ({_strip_query(filtered_query)}) AS count_query"


@dataclass
class JobState:
    id: str
    type: str
    status: str = "queued"
    step: str = "In coda"
    message: str = ""
    total: int = 0
    current: int = 0
    processed: int = 0
    embedded: int = 0
    written: int = 0
    skipped: int = 0
    chunks: int = 0
    attempt: int = 0
    error: str | None = None
    started_at: str = field(default_factory=_now)
    finished_at: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ConfigStore:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.path)

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS app_config (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                )
                """
            )

    def get_config(self) -> DatabaseConfig:
        with self._connect() as conn:
            row = conn.execute("SELECT value FROM app_config WHERE key = 'database_config'").fetchone()
        if not row:
            return DatabaseConfig(query=DEFAULT_QUERY)
        data = json.loads(row[0])
        if not data.get("query"):
            data["query"] = DEFAULT_QUERY
        return DatabaseConfig(**data)

    def save_config(self, config: DatabaseConfig) -> DatabaseConfig:
        with self._connect() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO app_config (key, value) VALUES ('database_config', ?)",
                (config.model_dump_json(),),
            )
        return config


class JobManager:
    def __init__(self) -> None:
        self._jobs: dict[str, JobState] = {}
        self._lock = threading.Lock()
        self._active_job_id: str | None = None

    def start(self, job_type: str, target: Callable[[JobState], None]) -> JobState:
        with self._lock:
            if self._active_job_id:
                active = self._jobs.get(self._active_job_id)
                if active and active.status in {"queued", "running"}:
                    raise RuntimeError(f"Job gia' in esecuzione: {active.id}")

            job = JobState(id=str(uuid4()), type=job_type)
            self._jobs[job.id] = job
            self._active_job_id = job.id

        thread = threading.Thread(target=self._run, args=(job, target), daemon=True)
        thread.start()
        return job

    def _run(self, job: JobState, target: Callable[[JobState], None]) -> None:
        try:
            job.status = "running"
            target(job)
            if job.status != "failed":
                job.status = "completed"
                job.step = "Completato"
                job.finished_at = _now()
        except Exception as exc:
            job.status = "failed"
            job.error = str(exc)
            job.message = "Job interrotto. Dettagli tecnici disponibili nel log backend."
            logger.exception("Index job %s failed", job.id)
            job.finished_at = _now()
        finally:
            with self._lock:
                if self._active_job_id == job.id:
                    self._active_job_id = None

    def get(self, job_id: str) -> JobState | None:
        return self._jobs.get(job_id)

    def latest(self) -> JobState | None:
        if not self._jobs:
            return None
        return list(self._jobs.values())[-1]

    def active(self) -> JobState | None:
        if not self._active_job_id:
            return None
        return self._jobs.get(self._active_job_id)


class VectorIndexBuilder:
    def __init__(
        self,
        faiss_dir: Path,
        openai_api_key: str,
        max_batch_size: int = 80,
        save_every_batches: int = 20,
        batch_pause_seconds: float = 0.15,
        on_complete: Callable[[], None] | None = None,
    ) -> None:
        self._set_output_dir(faiss_dir)
        self.client = OpenAI(api_key=openai_api_key, timeout=90.0, max_retries=2)
        self.max_batch_size = max(1, int(max_batch_size))
        self.save_every_batches = max(1, int(save_every_batches))
        self.batch_pause_seconds = max(0.0, float(batch_pause_seconds))
        self.on_complete = on_complete

    def _set_output_dir(self, faiss_dir: Path) -> None:
        self.faiss_dir = faiss_dir
        self.index_path = faiss_dir / "ticket_index.faiss"
        self.csv_path = faiss_dir / "ticket_data.csv"
        self.ids_path = faiss_dir / "ticket_ids.txt"

    def latest_local_ticket_date(self) -> str | None:
        if not self.csv_path.exists():
            return None
        try:
            df = pd.read_csv(self.csv_path, sep=";", encoding="utf-8", usecols=["created"])
        except Exception:
            df = pd.read_csv(self.csv_path, sep=";", encoding="utf-8", engine="python", usecols=["created"])
        values = pd.to_datetime(df["created"], errors="coerce")
        latest = values.max()
        if pd.isna(latest):
            return None
        return latest.strftime("%Y-%m-%d %H:%M:%S")

    def rebuild(self, config: DatabaseConfig, from_date: str, job: JobState) -> None:
        final_dir = self.faiss_dir
        build_dir = final_dir.parent / f"{final_dir.name}_build"
        self._prepare_build_dir(build_dir, job)
        self._set_output_dir(build_dir)
        try:
            self._prepare_empty_output(job)
            query = _date_filter_query(config.query, ">=")
            self._run_query_to_index(config, query, (from_date,), set(), job, notify_on_complete=False)
            self._publish_build(build_dir, final_dir, job)
            if self.on_complete:
                self.on_complete()
        finally:
            self._set_output_dir(final_dir)

    def resume_rebuild(self, config: DatabaseConfig, from_date: str, job: JobState) -> None:
        final_dir = self.faiss_dir
        build_dir = final_dir.parent / f"{final_dir.name}_build"
        self._set_output_dir(build_dir)
        try:
            existing_count = self._validate_resumable_build(build_dir)
            existing_ids = self._read_existing_ids()
            job.written = existing_count
            job.chunks = existing_count
            resume_from, partial_min, partial_max = self._resume_date_window(build_dir, from_date)
            job.message = (
                f"Ripresa da build parziale: {existing_count} vettori gia' presenti. "
                f"Checkpoint {partial_min or 'sconosciuto'} -> {partial_max or 'sconosciuto'}; riparto da {resume_from}."
            )
            query = _date_filter_query(config.query, ">=")
            self._run_query_to_index(config, query, (resume_from,), existing_ids, job, notify_on_complete=False)
            self._publish_build(build_dir, final_dir, job)
            if self.on_complete:
                self.on_complete()
        finally:
            self._set_output_dir(final_dir)

    def append_until_today(self, config: DatabaseConfig, job: JobState) -> None:
        latest = self.latest_local_ticket_date()
        if not latest:
            raise RuntimeError("Nessun indice locale trovato: esegui prima una ricostruzione da zero.")

        existing_ids = self._read_existing_ids()
        query = _date_filter_query(config.query, ">")
        self._run_query_to_index(config, query, (latest,), existing_ids, job)

    @staticmethod
    def _prepare_build_dir(build_dir: Path, job: JobState) -> None:
        job.step = "Preparazione cartella FAISS"
        if build_dir.exists():
            shutil.rmtree(build_dir)
        build_dir.mkdir(parents=True, exist_ok=True)

    def _prepare_empty_output(self, job: JobState) -> None:
        job.step = "Preparazione cartella FAISS"
        self.faiss_dir.mkdir(parents=True, exist_ok=True)
        for path in [self.index_path, self.csv_path, self.ids_path]:
            if path.exists():
                path.unlink()

        with self.csv_path.open("w", newline="", encoding="utf-8") as file:
            csv.writer(file, delimiter=";").writerow(CSV_COLUMNS)
        self.ids_path.write_text("", encoding="utf-8")

    @staticmethod
    def _validate_resumable_build(build_dir: Path) -> int:
        required = ["ticket_index.faiss", "ticket_data.csv", "ticket_ids.txt"]
        missing = [name for name in required if not (build_dir / name).exists() or (build_dir / name).stat().st_size <= 0]
        if missing:
            raise RuntimeError(f"Nessuna build parziale riprendibile in {build_dir}: mancano {', '.join(missing)}.")

        index = faiss.read_index(str(build_dir / "ticket_index.faiss"))
        ids_count = sum(1 for line in (build_dir / "ticket_ids.txt").read_text(encoding="utf-8", errors="ignore").splitlines() if line.strip())
        if index.ntotal != ids_count:
            raise RuntimeError(
                f"Build parziale incoerente: FAISS contiene {index.ntotal} vettori, ticket_ids.txt contiene {ids_count} righe."
            )
        return int(index.ntotal)

    @staticmethod
    def _resume_date_window(build_dir: Path, requested_from: str) -> tuple[str, str | None, str | None]:
        csv_path = build_dir / "ticket_data.csv"
        dates = pd.read_csv(csv_path, sep=";", encoding="utf-8", usecols=["created"])
        values = pd.to_datetime(dates["created"], errors="coerce").dropna()
        if values.empty:
            return requested_from, None, None

        partial_min = values.min()
        partial_max = values.max()
        resume_from = partial_max.strftime("%Y-%m-%d %H:%M:%S")
        return resume_from, partial_min.strftime("%Y-%m-%d %H:%M:%S"), resume_from

    @staticmethod
    def _move_with_retry(source: Path, destination: Path) -> None:
        last_error: Exception | None = None
        for attempt in range(1, 8):
            try:
                shutil.move(str(source), str(destination))
                return
            except PermissionError as exc:
                last_error = exc
                gc.collect()
                time.sleep(min(0.5 * attempt, 3))
        assert last_error is not None
        raise last_error

    def _run_query_to_index(
        self,
        config: DatabaseConfig,
        query: str,
        params: tuple[Any, ...],
        existing_ids: set[str],
        job: JobState,
        notify_on_complete: bool = True,
    ) -> None:
        max_retries = 3
        for attempt in range(1, max_retries + 1):
            try:
                job.attempt = attempt
                existing_ids.update(self._read_existing_ids())
                self._run_query_once(config, query, params, existing_ids, job)
                if notify_on_complete and self.on_complete:
                    self.on_complete()
                return
            except Exception:
                if attempt >= max_retries:
                    raise
                job.step = "Riprovo connessione"
                job.message = f"Tentativo {attempt}/{max_retries} fallito, nuovo tentativo tra {5 * attempt} secondi."
                time.sleep(5 * attempt)

    def _run_query_once(
        self,
        config: DatabaseConfig,
        query: str,
        params: tuple[Any, ...],
        existing_ids: set[str],
        job: JobState,
    ) -> None:
        job.step = "Connessione SSH"
        with SSHTunnelForwarder(
            (config.ssh_host, config.ssh_port),
            ssh_username=config.ssh_user,
            ssh_password=config.ssh_password,
            remote_bind_address=(config.db_host, config.db_port),
        ) as tunnel:
            tunnel.start()
            job.step = "Connessione database"
            conn = mysql.connector.connect(
                host="127.0.0.1",
                port=tunnel.local_bind_port,
                user=config.db_user,
                password=config.db_password,
                database=config.db_name,
                use_pure=True,
            )
            try:
                self._process_cursor(conn, query, params, existing_ids, config.batch_size, job)
            finally:
                conn.close()

    def _process_cursor(
        self,
        conn: Any,
        query: str,
        params: tuple[Any, ...],
        existing_ids: set[str],
        batch_size: int,
        job: JobState,
    ) -> None:
        job.step = "Conteggio ticket"
        job.current = 0
        job.processed = 0
        job.skipped = 0
        effective_batch_size = max(1, min(int(batch_size or self.max_batch_size), self.max_batch_size))
        cursor_count = conn.cursor()
        cursor_count.execute(_count_query(query), params)
        job.total = int(cursor_count.fetchone()[0] or 0)
        cursor_count.close()

        index = faiss.read_index(str(self.index_path)) if self.index_path.exists() else None

        cursor = conn.cursor()
        cursor.execute(query, params)
        batch_texts: list[str] = []
        batch_meta: list[dict[str, Any]] = []
        pending_meta: list[dict[str, Any]] = []
        batch_tokens = 0
        flush_count = 0

        job.step = "Lettura ticket"
        for row in cursor:
            job.current += 1
            id_, thread_id, staff_id, user_id, poster, created, title, body = row
            ticket_id = str(id_)

            if ticket_id in existing_ids:
                job.skipped += 1
                continue

            clean_text = clean_body(body)
            if not clean_text or len(clean_text) < 30:
                job.skipped += 1
                continue

            chunks = self._split_for_embedding(clean_text)
            chunk_count = len(chunks)
            for chunk_index, chunk in enumerate(chunks, start=1):
                chunk_tokens = self._count_tokens(chunk)
                if batch_texts and (
                    len(batch_texts) >= effective_batch_size or batch_tokens + chunk_tokens > MAX_EMBED_BATCH_TOKENS
                ):
                    index = self._embed_batch(index, batch_texts, batch_meta, pending_meta, existing_ids, job)
                    flush_count += 1
                    if flush_count % self.save_every_batches == 0:
                        self._persist_checkpoint(index, pending_meta, job)
                    batch_texts.clear()
                    batch_meta.clear()
                    batch_tokens = 0

                batch_texts.append(chunk)
                batch_tokens += chunk_tokens
                batch_meta.append(
                    {
                        "id": id_,
                        "thread_id": thread_id,
                        "staff_id": staff_id,
                        "user_id": user_id,
                        "poster": poster,
                        "created": str(created),
                        "title": title,
                        "clean_body": chunk,
                        "chunk_index": chunk_index,
                        "chunk_count": chunk_count,
                    }
                )
            job.processed += 1

            if len(batch_texts) >= effective_batch_size:
                index = self._embed_batch(index, batch_texts, batch_meta, pending_meta, existing_ids, job)
                flush_count += 1
                if flush_count % self.save_every_batches == 0:
                    self._persist_checkpoint(index, pending_meta, job)
                batch_texts.clear()
                batch_meta.clear()
                batch_tokens = 0

        cursor.close()
        if batch_texts:
            index = self._embed_batch(index, batch_texts, batch_meta, pending_meta, existing_ids, job)
        if index is not None and pending_meta:
            self._persist_checkpoint(index, pending_meta, job)

    def _embed_batch(
        self,
        index: faiss.Index | None,
        texts: list[str],
        rows: list[dict[str, Any]],
        pending_rows: list[dict[str, Any]],
        existing_ids: set[str],
        job: JobState,
    ) -> faiss.Index:
        job.step = "Embedding OpenAI"
        response = self.client.embeddings.create(input=texts, model=EMBEDDING_MODEL)
        embeddings = np.array([item.embedding for item in response.data], dtype="float32")
        job.embedded += len(embeddings)

        if index is None:
            index = faiss.IndexFlatL2(embeddings.shape[1])
        elif index.d != embeddings.shape[1]:
            raise RuntimeError(
                f"L'indice esistente ha dimensione {index.d}, ma {EMBEDDING_MODEL} produce {embeddings.shape[1]}. "
                "Svuota FAISS o esegui una ricostruzione da zero."
            )

        job.step = "Scrittura FAISS"
        index.add(embeddings)
        pending_rows.extend(rows)
        for row in rows:
            existing_ids.add(str(row["id"]))
        job.chunks += len(rows)
        job.message = (
            f"Preparati {job.chunks} vettori/chunk; {len(pending_rows)} in attesa del prossimo checkpoint FAISS."
        )
        del response, embeddings
        gc.collect()
        if self.batch_pause_seconds:
            time.sleep(self.batch_pause_seconds)
        return index

    def _persist_checkpoint(self, index: faiss.Index, pending_rows: list[dict[str, Any]], job: JobState) -> None:
        if not pending_rows:
            return

        job.step = "Scrittura FAISS"
        self._write_index_atomically(index)

        with self.csv_path.open("a", newline="", encoding="utf-8") as csv_file, self.ids_path.open("a", encoding="utf-8") as ids_file:
            writer = csv.writer(csv_file, delimiter=";")
            for row in pending_rows:
                writer.writerow([row[column] for column in CSV_COLUMNS])
                ids_file.write(f"{row['id']}\n")
                job.written += 1

        pending_rows.clear()
        job.message = f"Checkpoint salvato: {job.written} vettori/chunk persistiti su FAISS"
        gc.collect()

    def _publish_build(self, build_dir: Path, final_dir: Path, job: JobState) -> None:
        job.step = "Pubblicazione FAISS"
        required = ["ticket_index.faiss", "ticket_data.csv", "ticket_ids.txt"]
        for name in required:
            source = build_dir / name
            if not source.exists() or source.stat().st_size <= 0:
                raise RuntimeError(f"Build incompleta: {name} non esiste o e' vuoto.")

        final_dir.mkdir(parents=True, exist_ok=True)
        for name in required:
            self._replace_with_retry(build_dir / name, final_dir / name)
        job.message = "Nuovo FAISS pubblicato."

    @staticmethod
    def _replace_with_retry(source: Path, destination: Path) -> None:
        last_error: Exception | None = None
        for attempt in range(1, 10):
            try:
                source.replace(destination)
                return
            except PermissionError as exc:
                last_error = exc
                gc.collect()
                time.sleep(min(0.5 * attempt, 4))
        assert last_error is not None
        raise last_error

    def _write_index_atomically(self, index: faiss.Index) -> None:
        self.faiss_dir.mkdir(parents=True, exist_ok=True)
        temp_path = self.index_path.with_name(f"{self.index_path.name}.tmp")
        if temp_path.exists():
            temp_path.unlink()

        faiss.write_index(index, str(temp_path))
        if not temp_path.exists() or temp_path.stat().st_size <= 0:
            if temp_path.exists():
                temp_path.unlink()
            raise RuntimeError("Scrittura FAISS fallita: il file temporaneo e' vuoto.")

        temp_path.replace(self.index_path)

    def _read_existing_ids(self) -> set[str]:
        if not self.ids_path.exists():
            return set()
        return {line.strip() for line in self.ids_path.read_text(encoding="utf-8", errors="ignore").splitlines() if line.strip()}

    @staticmethod
    def _encoding() -> tiktoken.Encoding:
        return tiktoken.encoding_for_model(EMBEDDING_MODEL)

    @classmethod
    def _count_tokens(cls, text: str) -> int:
        return len(cls._encoding().encode(text))

    @classmethod
    def _split_for_embedding(cls, text: str) -> list[str]:
        encoding = cls._encoding()
        tokens = encoding.encode(text)
        if len(tokens) <= MAX_EMBED_TOKENS:
            return [text]

        chunks: list[str] = []
        start = 0
        while start < len(tokens):
            end = min(start + MAX_EMBED_TOKENS, len(tokens))
            chunks.append(encoding.decode(tokens[start:end]))
            if end >= len(tokens):
                break
            start = max(0, end - CHUNK_TOKEN_OVERLAP)
        return chunks
