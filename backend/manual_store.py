from __future__ import annotations

import io
import json
import logging
import shutil
import sqlite3
import threading
import uuid
from contextlib import closing
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import faiss
import pymupdf as fitz
import numpy as np
import pytesseract
import tiktoken
from PIL import Image

from .models import ManualHit
from .openai_service import OpenAIService


logger = logging.getLogger("uvicorn.error")
ACTIVE_STATUSES = {"queued", "processing"}


class ManualRegistry:
    def __init__(self, database_path: Path) -> None:
        self.database_path = database_path
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path, timeout=30)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        return connection

    def _init_db(self) -> None:
        with closing(self._connect()) as connection, connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS manuals (
                    id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    filename TEXT NOT NULL,
                    status TEXT NOT NULL,
                    error TEXT,
                    page_count INTEGER NOT NULL DEFAULT 0,
                    chunk_count INTEGER NOT NULL DEFAULT 0,
                    image_count INTEGER NOT NULL DEFAULT 0,
                    progress_step TEXT NOT NULL DEFAULT '',
                    progress_current INTEGER NOT NULL DEFAULT 0,
                    progress_total INTEGER NOT NULL DEFAULT 0,
                    all_departments INTEGER NOT NULL DEFAULT 1,
                    created_by TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )
            columns = {row[1] for row in connection.execute("PRAGMA table_info(manuals)").fetchall()}
            for name, definition in (
                ("progress_step", "TEXT NOT NULL DEFAULT ''"),
                ("progress_current", "INTEGER NOT NULL DEFAULT 0"),
                ("progress_total", "INTEGER NOT NULL DEFAULT 0"),
            ):
                if name not in columns:
                    connection.execute(f"ALTER TABLE manuals ADD COLUMN {name} {definition}")
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS manual_departments (
                    manual_id TEXT NOT NULL,
                    department_id INTEGER NOT NULL,
                    PRIMARY KEY (manual_id, department_id),
                    FOREIGN KEY (manual_id) REFERENCES manuals(id) ON DELETE CASCADE
                )
                """
            )
            connection.execute(
                """
                UPDATE manuals
                SET status = 'failed', error = 'Elaborazione interrotta dal riavvio del backend.',
                    progress_step = 'Interrotto', updated_at = ?
                WHERE status IN ('queued', 'processing')
                """,
                (self._now(),),
            )

    def create(
        self,
        title: str,
        filename: str,
        created_by: str,
        all_departments: bool,
        department_ids: list[int],
    ) -> dict[str, Any]:
        manual_id = uuid.uuid4().hex
        now = self._now()
        with closing(self._connect()) as connection, connection:
            connection.execute(
                """
                INSERT INTO manuals
                    (id, title, filename, status, all_departments, created_by, created_at, updated_at)
                VALUES (?, ?, ?, 'queued', ?, ?, ?, ?)
                """,
                (manual_id, title.strip(), filename, int(all_departments), created_by, now, now),
            )
            self._replace_departments(connection, manual_id, [] if all_departments else department_ids)
        result = self.get(manual_id)
        assert result is not None
        return result

    def get(self, manual_id: str) -> dict[str, Any] | None:
        with closing(self._connect()) as connection:
            row = connection.execute("SELECT * FROM manuals WHERE id = ?", (manual_id,)).fetchone()
            if row is None:
                return None
            department_rows = connection.execute(
                "SELECT department_id FROM manual_departments WHERE manual_id = ? ORDER BY department_id",
                (manual_id,),
            ).fetchall()
        return self._serialize(row, department_rows)

    def list(self, allowed_department_ids: set[int] | None, include_unready: bool = False) -> list[dict[str, Any]]:
        query = "SELECT * FROM manuals"
        params: tuple[object, ...] = ()
        if not include_unready:
            query += " WHERE status = ?"
            params = ("ready",)
        query += " ORDER BY title COLLATE NOCASE, created_at DESC"
        with closing(self._connect()) as connection:
            rows = connection.execute(query, params).fetchall()
            department_rows = connection.execute(
                "SELECT manual_id, department_id FROM manual_departments ORDER BY department_id"
            ).fetchall()
        departments: dict[str, list[sqlite3.Row]] = {}
        for row in department_rows:
            departments.setdefault(str(row["manual_id"]), []).append(row)
        result = [self._serialize(row, departments.get(str(row["id"]), [])) for row in rows]
        return [manual for manual in result if self.can_access(manual, allowed_department_ids)]

    @staticmethod
    def can_access(manual: dict[str, Any], allowed_department_ids: set[int] | None) -> bool:
        if allowed_department_ids is None or manual["all_departments"]:
            return True
        return bool(set(manual["department_ids"]) & allowed_department_ids)

    def update_status(self, manual_id: str, status: str, **fields: object) -> None:
        allowed_fields = {
            "error", "page_count", "chunk_count", "image_count",
            "progress_step", "progress_current", "progress_total",
        }
        updates = ["status = ?", "updated_at = ?"]
        values: list[object] = [status, self._now()]
        for key, value in fields.items():
            if key in allowed_fields:
                updates.append(f"{key} = ?")
                values.append(value)
        values.append(manual_id)
        with closing(self._connect()) as connection, connection:
            connection.execute(f"UPDATE manuals SET {', '.join(updates)} WHERE id = ?", values)

    def update_permissions(
        self, manual_id: str, all_departments: bool, department_ids: list[int]
    ) -> dict[str, Any]:
        if not all_departments and not department_ids:
            raise ValueError("Seleziona almeno un reparto oppure abilita Tutti.")
        with closing(self._connect()) as connection, connection:
            found = connection.execute("SELECT 1 FROM manuals WHERE id = ?", (manual_id,)).fetchone()
            if not found:
                raise ValueError("Manuale non trovato.")
            connection.execute(
                "UPDATE manuals SET all_departments = ?, updated_at = ? WHERE id = ?",
                (int(all_departments), self._now(), manual_id),
            )
            self._replace_departments(connection, manual_id, [] if all_departments else department_ids)
        result = self.get(manual_id)
        assert result is not None
        return result

    def delete(self, manual_id: str) -> bool:
        with closing(self._connect()) as connection, connection:
            cursor = connection.execute("DELETE FROM manuals WHERE id = ?", (manual_id,))
        return cursor.rowcount > 0

    @staticmethod
    def _replace_departments(
        connection: sqlite3.Connection, manual_id: str, department_ids: list[int]
    ) -> None:
        connection.execute("DELETE FROM manual_departments WHERE manual_id = ?", (manual_id,))
        connection.executemany(
            "INSERT INTO manual_departments (manual_id, department_id) VALUES (?, ?)",
            [(manual_id, value) for value in sorted({int(item) for item in department_ids})],
        )

    @staticmethod
    def _serialize(row: sqlite3.Row, department_rows: list[sqlite3.Row]) -> dict[str, Any]:
        result = dict(row)
        result["all_departments"] = bool(result["all_departments"])
        result["department_ids"] = [int(item["department_id"]) for item in department_rows]
        return result

    @staticmethod
    def _now() -> str:
        return datetime.now().isoformat(timespec="seconds")


class ManualIndexManager:
    def __init__(
        self,
        root: Path,
        registry: ManualRegistry,
        openai_service: OpenAIService,
        embedding_model: str,
        on_complete: Callable[[], None] | None = None,
    ) -> None:
        self.root = root
        self.registry = registry
        self.openai_service = openai_service
        self.embedding_model = embedding_model
        self.on_complete = on_complete
        self.root.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._worker_lock = threading.Lock()
        self._active: set[str] = set()

    def start(self, manual_id: str) -> None:
        with self._lock:
            if manual_id in self._active:
                raise ValueError("Questo manuale e' gia' in elaborazione.")
            self._active.add(manual_id)
        self.registry.update_status(
            manual_id, "queued", error=None, progress_step="In coda", progress_current=0, progress_total=0
        )
        threading.Thread(target=self._run, args=(manual_id,), daemon=True).start()

    def _run(self, manual_id: str) -> None:
        try:
            # OCR and page rendering are deliberately serialized to protect small servers from memory spikes.
            with self._worker_lock:
                previous = self.registry.get(manual_id)
                manual_dir = self.root / manual_id
                backup = self._backup_generated_files(manual_dir)
                built = False
                try:
                    self._build(manual_id)
                    built = True
                except Exception as exc:
                    if backup and previous:
                        self._restore_generated_files(manual_dir, backup)
                        self.registry.update_status(
                            manual_id,
                            "ready",
                            error=f"Ultima reindicizzazione non riuscita: {str(exc)[:1500]}",
                            page_count=previous["page_count"],
                            chunk_count=previous["chunk_count"],
                            image_count=previous["image_count"],
                            progress_step="Indice precedente ripristinato",
                            progress_current=previous["progress_total"],
                            progress_total=previous["progress_total"],
                        )
                        logger.exception("Manual reindex failed; previous index restored for %s", manual_id)
                        return
                    raise
                finally:
                    if built and backup and backup.exists():
                        shutil.rmtree(backup, ignore_errors=True)
        except Exception as exc:
            logger.exception("Manual indexing failed for %s", manual_id)
            self.registry.update_status(
                manual_id, "failed", error=str(exc)[:2000], progress_step="Errore"
            )
        finally:
            with self._lock:
                self._active.discard(manual_id)
            if self.on_complete:
                self.on_complete()

    def _build(self, manual_id: str) -> None:
        manual = self.registry.get(manual_id)
        if manual is None:
            raise ValueError("Manuale non trovato.")
        manual_dir = self.root / manual_id
        pdf_path = manual_dir / "original.pdf"
        if not pdf_path.exists():
            raise FileNotFoundError("PDF originale non trovato.")
        if not self.openai_service.configured():
            raise RuntimeError("OPENAI_API_KEY non configurata.")

        self.registry.update_status(
            manual_id,
            "processing",
            error=None,
            page_count=0,
            chunk_count=0,
            image_count=0,
            progress_step="Apertura PDF",
            progress_current=0,
            progress_total=0,
        )
        self._clear_generated_files(manual_dir)
        pages_dir = manual_dir / "pages"
        images_dir = manual_dir / "images"
        pages_dir.mkdir(parents=True, exist_ok=True)
        images_dir.mkdir(parents=True, exist_ok=True)

        metadata: list[dict[str, Any]] = []
        image_count = 0
        document = fitz.open(pdf_path)
        try:
            if document.page_count > 1500:
                raise ValueError("Il PDF supera il limite di 1500 pagine.")
            total_pages = document.page_count
            logger.info("Manual %s: extraction started, %s pages", manual_id, total_pages)
            self.registry.update_status(
                manual_id,
                "processing",
                progress_step="Estrazione testo e OCR",
                progress_current=0,
                progress_total=total_pages,
            )
            for page_index, page in enumerate(document):
                page_number = page_index + 1
                self.registry.update_status(
                    manual_id,
                    "processing",
                    progress_step=f"Estrazione e OCR pagina {page_number}",
                    progress_current=page_index,
                    progress_total=total_pages,
                )
                page_image = self._render_page(page, pages_dir / f"page-{page_number:04d}.webp")
                text = page.get_text("text").strip()
                page_image_paths: list[str] = []
                ocr_parts: list[str] = []
                seen_xrefs: set[int] = set()
                displayed_images = page.get_image_info(xrefs=True)
                for image_number, image_info in enumerate(displayed_images, start=1):
                    xref = int(image_info.get("xref", 0))
                    bbox = fitz.Rect(image_info["bbox"])
                    if xref <= 0 or bbox.width < 180 or bbox.height < 80:
                        continue
                    if xref in seen_xrefs:
                        continue
                    seen_xrefs.add(xref)
                    try:
                        extracted = document.extract_image(xref)
                        image = Image.open(io.BytesIO(extracted["image"])).convert("RGB")
                        if image.width < 160 or image.height < 90:
                            continue
                        relative = f"images/page-{page_number:04d}-{image_number:02d}.webp"
                        image.save(manual_dir / relative, "WEBP", quality=84, method=4)
                        page_image_paths.append(relative)
                        image_count += 1
                        ocr = self._ocr(image)
                        if ocr:
                            ocr_parts.append(ocr)
                        if len(page_image_paths) >= 12:
                            break
                    except Exception:
                        logger.warning("Unable to extract image %s from page %s", xref, page_number, exc_info=True)

                if len(text) < 80:
                    page_ocr = self._ocr(page_image)
                    if page_ocr and page_ocr not in text:
                        ocr_parts.append(page_ocr)
                combined = self._normalize_text("\n".join([text, *ocr_parts]))
                if combined:
                    for chunk_index, chunk in enumerate(self._chunks(combined), start=1):
                        metadata.append(
                            {
                                "manual_id": manual_id,
                                "manual_title": manual["title"],
                                "page": page_number,
                                "chunk": chunk_index,
                                "text": chunk,
                                "page_image": f"pages/page-{page_number:04d}.webp",
                                "images": page_image_paths,
                            }
                        )
                self.registry.update_status(
                    manual_id,
                    "processing",
                    page_count=page_number,
                    chunk_count=len(metadata),
                    image_count=image_count,
                    progress_step="Estrazione testo e OCR",
                    progress_current=page_number,
                    progress_total=total_pages,
                )
                if page_number == 1 or page_number % 5 == 0 or page_number == total_pages:
                    logger.info(
                        "Manual %s: extracted page %s/%s, %s chunks, %s images",
                        manual_id,
                        page_number,
                        total_pages,
                        len(metadata),
                        image_count,
                    )
        finally:
            document.close()

        if not metadata:
            raise ValueError("Non e' stato possibile estrarre testo o immagini leggibili dal PDF.")

        vectors: list[list[float]] = []
        logger.info("Manual %s: embedding started, %s chunks", manual_id, len(metadata))
        self.registry.update_status(
            manual_id,
            "processing",
            progress_step="Creazione embedding OpenAI",
            progress_current=0,
            progress_total=len(metadata),
        )
        for start in range(0, len(metadata), 50):
            texts = [item["text"] for item in metadata[start : start + 50]]
            vectors.extend(self.openai_service.embed_many(texts, self.embedding_model))
            completed = min(start + len(texts), len(metadata))
            self.registry.update_status(
                manual_id,
                "processing",
                chunk_count=len(metadata),
                progress_step="Creazione embedding OpenAI",
                progress_current=completed,
                progress_total=len(metadata),
            )
            logger.info("Manual %s: embedded %s/%s chunks", manual_id, completed, len(metadata))
        self.registry.update_status(
            manual_id,
            "processing",
            progress_step="Scrittura indice FAISS",
            progress_current=0,
            progress_total=1,
        )
        matrix = np.asarray(vectors, dtype="float32")
        index = faiss.IndexFlatL2(matrix.shape[1])
        index.add(matrix)

        index_tmp = manual_dir / "manual.faiss.tmp"
        metadata_tmp = manual_dir / "metadata.json.tmp"
        faiss.write_index(index, str(index_tmp))
        metadata_tmp.write_text(json.dumps(metadata, ensure_ascii=False), encoding="utf-8")
        index_tmp.replace(manual_dir / "manual.faiss")
        metadata_tmp.replace(manual_dir / "metadata.json")
        self.registry.update_status(
            manual_id,
            "ready",
            error=None,
            page_count=total_pages,
            chunk_count=len(metadata),
            image_count=image_count,
            progress_step="Completato",
            progress_current=1,
            progress_total=1,
        )
        logger.info(
            "Manual %s: completed, %s pages, %s chunks, %s images",
            manual_id,
            total_pages,
            len(metadata),
            image_count,
        )

    @staticmethod
    def _render_page(page: fitz.Page, path: Path) -> Image.Image:
        pixmap = page.get_pixmap(matrix=fitz.Matrix(1.45, 1.45), alpha=False)
        image = Image.frombytes("RGB", (pixmap.width, pixmap.height), pixmap.samples)
        image.save(path, "WEBP", quality=82, method=4)
        return image

    @staticmethod
    def _ocr(image: Image.Image) -> str:
        try:
            return pytesseract.image_to_string(image, lang="ita+eng", timeout=45).strip()
        except pytesseract.TesseractError:
            try:
                return pytesseract.image_to_string(image, lang="eng", timeout=45).strip()
            except Exception:
                return ""
        except Exception:
            return ""

    @staticmethod
    def _normalize_text(text: str) -> str:
        lines = [" ".join(line.split()) for line in text.replace("\x00", " ").splitlines()]
        return "\n".join(line for line in lines if line).strip()

    @staticmethod
    def _chunks(text: str, max_tokens: int = 900, overlap: int = 120) -> list[str]:
        encoding = tiktoken.get_encoding("cl100k_base")
        tokens = encoding.encode(text)
        if not tokens:
            return []
        chunks: list[str] = []
        start = 0
        while start < len(tokens):
            end = min(len(tokens), start + max_tokens)
            chunks.append(encoding.decode(tokens[start:end]).strip())
            if end >= len(tokens):
                break
            start = max(start + 1, end - overlap)
        return [chunk for chunk in chunks if chunk]

    @staticmethod
    def _clear_generated_files(manual_dir: Path) -> None:
        for name in ("pages", "images"):
            shutil.rmtree(manual_dir / name, ignore_errors=True)
        for name in ("manual.faiss", "manual.faiss.tmp", "metadata.json", "metadata.json.tmp"):
            (manual_dir / name).unlink(missing_ok=True)

    @classmethod
    def _backup_generated_files(cls, manual_dir: Path) -> Path | None:
        backup = manual_dir / ".reindex-backup"
        if backup.exists():
            cls._restore_generated_files(manual_dir, backup)
        # Partial output from an interrupted first ingestion is not a usable rollback target.
        if not (manual_dir / "manual.faiss").exists() or not (manual_dir / "metadata.json").exists():
            cls._clear_generated_files(manual_dir)
            return None
        names = ("pages", "images", "manual.faiss", "metadata.json")
        existing = [manual_dir / name for name in names if (manual_dir / name).exists()]
        if not existing:
            return None
        backup.mkdir(parents=True)
        for path in existing:
            shutil.move(str(path), str(backup / path.name))
        return backup

    @classmethod
    def _restore_generated_files(cls, manual_dir: Path, backup: Path) -> None:
        cls._clear_generated_files(manual_dir)
        if not backup.exists():
            return
        for path in backup.iterdir():
            shutil.move(str(path), str(manual_dir / path.name))
        shutil.rmtree(backup, ignore_errors=True)


class ManualSearchStore:
    def __init__(self, root: Path, registry: ManualRegistry) -> None:
        self.root = root
        self.registry = registry
        self._cache: dict[str, tuple[float, faiss.Index, list[dict[str, Any]]]] = {}
        self._lock = threading.Lock()

    def clear(self) -> None:
        with self._lock:
            self._cache.clear()

    def search(
        self, query_embedding: list[float], top_k: int, allowed_department_ids: set[int] | None
    ) -> list[ManualHit]:
        vector = np.asarray([query_embedding], dtype="float32")
        candidates: list[tuple[float, dict[str, Any]]] = []
        manuals = self.registry.list(allowed_department_ids, include_unready=False)
        page_counts = {str(manual["id"]): int(manual["page_count"]) for manual in manuals}
        for manual in manuals:
            index, metadata = self._load(manual["id"])
            if vector.shape[1] != index.d:
                logger.warning("Skipping manual %s: embedding dimension mismatch", manual["id"])
                continue
            search_k = min(int(index.ntotal), max(top_k * 3, 12))
            if not search_k:
                continue
            distances, positions = index.search(vector, search_k)
            for score, position in zip(distances[0], positions[0]):
                if 0 <= position < len(metadata):
                    candidates.append((float(score), metadata[int(position)]))
        candidates.sort(key=lambda item: item[0])

        hits: list[ManualHit] = []
        seen: set[tuple[str, int]] = set()
        for score, item in candidates:
            key = (str(item["manual_id"]), int(item["page"]))
            if key in seen:
                continue
            seen.add(key)
            manual_id = str(item["manual_id"])
            page = int(item["page"])
            body = str(item["text"])
            hits.append(
                ManualHit(
                    rank=len(hits) + 1,
                    score=score,
                    manual_id=manual_id,
                    manual_title=str(item["manual_title"]),
                    page=page,
                    page_count=page_counts.get(manual_id, page),
                    excerpt=body[:700].rstrip() + ("..." if len(body) > 700 else ""),
                    body=body,
                    image_urls=[f"/api/manuals/{manual_id}/assets/{path}" for path in item.get("images", [])],
                    page_image_url=f"/api/manuals/{manual_id}/assets/{item['page_image']}",
                    pdf_url=f"/api/manuals/{manual_id}/pdf?page={page}",
                )
            )
            if len(hits) >= top_k:
                break
        return hits

    def _load(self, manual_id: str) -> tuple[faiss.Index, list[dict[str, Any]]]:
        manual_dir = self.root / manual_id
        index_path = manual_dir / "manual.faiss"
        metadata_path = manual_dir / "metadata.json"
        modified = max(index_path.stat().st_mtime, metadata_path.stat().st_mtime)
        with self._lock:
            cached = self._cache.get(manual_id)
            if cached and cached[0] == modified:
                return cached[1], cached[2]
            index = faiss.read_index(str(index_path))
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            self._cache[manual_id] = (modified, index, metadata)
            return index, metadata
