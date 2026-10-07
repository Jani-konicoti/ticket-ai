from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

import faiss
import numpy as np
import pandas as pd

from backend.index_builder import (
    CSV_COLUMNS,
    DEFAULT_QUERY,
    LEGACY_DEFAULT_QUERY,
    ConfigStore,
    JobState,
    VectorIndexBuilder,
    clean_body,
)
from backend.models import DatabaseConfig
from backend.ticket_store import TicketStore


class TicketStoreShardTests(unittest.TestCase):
    def test_search_merges_base_and_append_shards(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            faiss_dir = Path(temp_dir)
            self._write_index(faiss_dir / "ticket_index.faiss", [[0.0, 0.0], [10.0, 10.0]])
            self._write_index(faiss_dir / "ticket_index.append-202610.faiss", [[1.0, 1.0]])
            (faiss_dir / "ticket_ids.txt").write_text("1\n2\n3\n", encoding="utf-8")
            self._write_metadata(faiss_dir / "ticket_data.csv")

            store = TicketStore(faiss_dir)
            hits = store.search([1.0, 1.0], top_k=2)

            self.assertEqual(store.stats.vectors, 3)
            self.assertEqual([hit["id"] for hit in hits], [3, 1])
            self.assertEqual([hit["rank"] for hit in hits], [1, 2])

    def test_append_uses_monthly_shard_without_loading_base_index(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            faiss_dir = Path(temp_dir)
            self._write_index(faiss_dir / "ticket_index.faiss", [[0.0, 0.0]])
            (faiss_dir / "ticket_ids.txt").write_text("1\n", encoding="utf-8")
            self._write_metadata(faiss_dir / "ticket_data.csv", ticket_ids=(1,))
            builder = VectorIndexBuilder(faiss_dir, "test-key")
            selected_paths: list[Path] = []
            selected_queries: list[str] = []
            selected_params: list[tuple[object, ...]] = []

            def capture_path(_config: object, query: str, params: tuple[object, ...], *_args: object, **_kwargs: object) -> None:
                selected_paths.append(builder.index_path)
                selected_queries.append(query)
                selected_params.append(params)

            builder._run_query_to_index = capture_path  # type: ignore[method-assign]
            builder.append_until_today(DatabaseConfig(), type("Job", (), {})())  # type: ignore[arg-type]

            self.assertEqual(len(selected_paths), 1)
            self.assertRegex(selected_paths[0].name, r"^ticket_index\.append-\d{6}-part000\.faiss$")
            self.assertEqual(builder.index_path, faiss_dir / "ticket_index.faiss")
            self.assertIn("WHERE id > %s", selected_queries[0])
            self.assertEqual(selected_params[0], (1,))

    def test_conversation_cleaning_keeps_question_and_answer(self) -> None:
        entries = [
            {
                "id": 10,
                "thread_id": 7,
                "staff_id": 0,
                "user_id": 3,
                "poster": "Anna",
                "created": "2026-10-06 11:00:00",
                "title": "Notifiche nei log",
                "body": "<p>Il cliente riceve anche una mail?</p><p>Buona giornata.</p><p>Firma molto lunga</p>",
                "entry_type": "M",
            },
            {
                "id": 11,
                "thread_id": 7,
                "staff_id": 4,
                "user_id": 0,
                "poster": "Benedetta",
                "created": "2026-10-06 12:00:00",
                "title": "Notifiche nei log",
                "body": "<p>La notifica viene mostrata soltanto nei log.</p><blockquote>testo precedente</blockquote>",
                "entry_type": "R",
            },
            {
                "id": 12,
                "thread_id": 7,
                "staff_id": 4,
                "user_id": 0,
                "poster": "Sistema",
                "created": "2026-10-06 12:01:00",
                "title": "Notifiche nei log",
                "body": "Ticket trasferito da Segnalazioni a Centro Paghe",
                "entry_type": "N",
            },
        ]

        chunks = VectorIndexBuilder._conversation_chunks(entries)

        self.assertEqual(len(chunks), 1)
        self.assertIn("[Cliente - Anna", chunks[0])
        self.assertIn("Il cliente riceve anche una mail?", chunks[0])
        self.assertIn("[Operatore - Benedetta", chunks[0])
        self.assertIn("soltanto nei log", chunks[0])
        self.assertNotIn("Firma molto lunga", chunks[0])
        self.assertNotIn("testo precedente", chunks[0])
        self.assertNotIn("Ticket trasferito", chunks[0])

    def test_clean_body_stops_before_privacy_disclaimer(self) -> None:
        cleaned = clean_body("<p>Soluzione utile.</p><p>Ai sensi degli artt. 13 e 14 del GDPR...</p><p>Rumore</p>")
        self.assertEqual(cleaned, "Soluzione utile.")

    def test_index_shards_keep_a_stable_metadata_order(self) -> None:
        base = Path("FAISS/ticket_index.faiss")
        build_part = VectorIndexBuilder._next_shard_path(base)
        second_build_part = VectorIndexBuilder._next_shard_path(build_part)
        monthly = Path("FAISS/ticket_index.append-202610-part000.faiss")
        second_monthly_part = VectorIndexBuilder._next_shard_path(monthly)

        self.assertEqual(build_part.name, "ticket_index.append-000000-part001.faiss")
        self.assertEqual(second_build_part.name, "ticket_index.append-000000-part002.faiss")
        self.assertEqual(second_monthly_part.name, "ticket_index.append-202610-part001.faiss")
        self.assertEqual(
            sorted([monthly.name, second_build_part.name, build_part.name, second_monthly_part.name]),
            [build_part.name, second_build_part.name, monthly.name, second_monthly_part.name],
        )

    def test_legacy_default_query_is_migrated(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            store = ConfigStore(Path(temp_dir) / "config.sqlite")
            store.save_config(DatabaseConfig(query=LEGACY_DEFAULT_QUERY))

            migrated = store.get_config()

            self.assertEqual(migrated.query, DEFAULT_QUERY)
            self.assertIn("ost_ticket__cdata", migrated.query)

    def test_cursor_groups_ordered_entries_by_thread(self) -> None:
        rows = [
            (1, 10, 0, 5, "Cliente", "2026-10-01 10:00:00", "Titolo A", "<p>Domanda del cliente abbastanza lunga.</p>", "M"),
            (2, 10, 7, 0, "Tecnico", "2026-10-01 11:00:00", "Titolo A", "<p>Risposta risolutiva del tecnico.</p>", "R"),
            (3, 20, 0, 6, "Altro cliente", "2026-10-02 10:00:00", "Titolo B", "<p>Seconda conversazione indipendente.</p>", "M"),
        ]

        class FakeCursor:
            def __init__(self, values: list[tuple[object, ...]], count_only: bool = False) -> None:
                self.values = values
                self.count_only = count_only

            def execute(self, *_args: object) -> None:
                return None

            def fetchone(self) -> tuple[int]:
                return (len(self.values),)

            def close(self) -> None:
                return None

            def __iter__(self):  # type: ignore[no-untyped-def]
                return iter([] if self.count_only else self.values)

        class FakeConnection:
            def __init__(self, values: list[tuple[object, ...]]) -> None:
                self.values = values
                self.calls = 0

            def cursor(self) -> FakeCursor:
                self.calls += 1
                return FakeCursor(self.values, count_only=self.calls == 1)

        with tempfile.TemporaryDirectory() as temp_dir:
            builder = VectorIndexBuilder(Path(temp_dir), "test-key")
            builder._prepare_empty_output(JobState(id="prepare", type="test"))

            def fake_embed(index, texts, metadata, pending, existing, job):  # type: ignore[no-untyped-def]
                if index is None:
                    index = faiss.IndexFlatL2(2)
                index.add(np.zeros((len(texts), 2), dtype="float32"))
                pending.extend(metadata)
                existing.update(str(row["id"]) for row in metadata)
                job.chunks += len(metadata)
                return index

            builder._embed_batch = fake_embed  # type: ignore[method-assign]
            job = JobState(id="group", type="test")
            builder._process_cursor(FakeConnection(rows), "SELECT", (), set(), 30, job)

            metadata = pd.read_csv(Path(temp_dir) / "ticket_data.csv", sep=";")
            self.assertEqual(job.current, 3)
            self.assertEqual(job.processed, 2)
            self.assertEqual(len(metadata), 2)
            self.assertIn("Domanda del cliente", metadata.iloc[0]["clean_body"])
            self.assertIn("Risposta risolutiva", metadata.iloc[0]["clean_body"])
            self.assertNotIn("Seconda conversazione", metadata.iloc[0]["clean_body"])

    @staticmethod
    def _write_index(path: Path, vectors: list[list[float]]) -> None:
        index = faiss.IndexFlatL2(2)
        index.add(np.asarray(vectors, dtype="float32"))
        faiss.write_index(index, str(path))

    @staticmethod
    def _write_metadata(path: Path, ticket_ids: tuple[int, ...] = (1, 2, 3)) -> None:
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle, delimiter=";")
            writer.writerow(CSV_COLUMNS)
            for ticket_id in ticket_ids:
                writer.writerow(
                    [
                        ticket_id,
                        ticket_id,
                        "",
                        "",
                        "test",
                        "2026-10-01 00:00:00",
                        f"Ticket {ticket_id}",
                        "body",
                        1,
                        1,
                        ticket_id,
                        ticket_id,
                    ]
                )


if __name__ == "__main__":
    unittest.main()
