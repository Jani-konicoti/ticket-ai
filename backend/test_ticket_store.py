from __future__ import annotations

import csv
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import faiss
import bcrypt
import pymupdf as fitz
import numpy as np
import pandas as pd
from PIL import Image

from backend.index_builder import (
    CSV_COLUMNS,
    DEFAULT_QUERY,
    CONVERSATION_DEFAULT_QUERY_V1,
    CONVERSATION_DEFAULT_QUERY_V2,
    LEGACY_DEFAULT_QUERY,
    MAX_EMBED_TOKENS,
    ConfigStore,
    JobState,
    VectorIndexBuilder,
    clean_body,
)
from backend.models import DatabaseConfig
from backend.auth import AuthStore
from backend.ticket_store import TicketStore
from backend.osticket_auth import OsTicketAuthenticator
from backend.manual_store import ManualIndexManager, ManualRegistry, ManualSearchStore


class TicketStoreShardTests(unittest.TestCase):
    def test_manual_index_preserves_pages_images_and_department_permissions(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            registry = ManualRegistry(root / "app.sqlite")
            manual = registry.create("Manuale paghe", "paghe.pdf", "admin", False, [10])
            manual_dir = root / "manuals" / manual["id"]
            manual_dir.mkdir(parents=True)

            image_buffer = io.BytesIO()
            Image.new("RGB", (320, 120), "white").save(image_buffer, format="PNG")
            document = fitz.open()
            page = document.new_page()
            page.insert_text((72, 72), "Procedura chiusura mensile: selezionare Conferma e verificare lo stato.")
            page.insert_image(fitz.Rect(72, 100, 392, 220), stream=image_buffer.getvalue())
            document.save(manual_dir / "original.pdf")
            document.close()

            class FakeOpenAI:
                @staticmethod
                def configured() -> bool:
                    return True

                @staticmethod
                def embed_many(texts: list[str], _model: str) -> list[list[float]]:
                    return [[float(len(text) % 5), 1.0] for text in texts]

            manager = ManualIndexManager(
                root / "manuals", registry, FakeOpenAI(), "test-model"  # type: ignore[arg-type]
            )
            with patch.object(ManualIndexManager, "_ocr", return_value="Pulsante conferma"):
                manager._build(str(manual["id"]))

            indexed = registry.get(str(manual["id"]))
            self.assertEqual(indexed["status"], "ready")  # type: ignore[index]
            self.assertEqual(indexed["page_count"], 1)  # type: ignore[index]
            self.assertEqual(indexed["image_count"], 1)  # type: ignore[index]
            self.assertTrue((manual_dir / "pages" / "page-0001.webp").exists())
            self.assertEqual(registry.list({99}), [])
            self.assertEqual(len(registry.list({10})), 1)

            hits = ManualSearchStore(root / "manuals", registry).search([1.0, 1.0], 3, {10})
            self.assertEqual(len(hits), 1)
            self.assertEqual(hits[0].manual_title, "Manuale paghe")
            self.assertEqual(hits[0].page, 1)
            self.assertTrue(hits[0].image_urls)

    def test_failed_manual_reindex_restores_previous_files(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            registry = ManualRegistry(root / "app.sqlite")
            manual = registry.create("Manuale", "manuale.pdf", "admin", True, [])
            manual_dir = root / "manuals" / manual["id"]
            manual_dir.mkdir(parents=True)
            (manual_dir / "original.pdf").write_bytes(b"%PDF-test")
            (manual_dir / "manual.faiss").write_bytes(b"indice precedente")
            (manual_dir / "metadata.json").write_text("[]", encoding="utf-8")
            registry.update_status(str(manual["id"]), "ready", page_count=4, chunk_count=9, image_count=2)

            manager = ManualIndexManager(root / "manuals", registry, object(), "model")  # type: ignore[arg-type]
            with (
                patch.object(manager, "_build", side_effect=RuntimeError("servizio non disponibile")),
                patch("backend.manual_store.logger.exception"),
            ):
                manager._run(str(manual["id"]))

            restored = registry.get(str(manual["id"]))
            self.assertEqual(restored["status"], "ready")  # type: ignore[index]
            self.assertEqual(restored["chunk_count"], 9)  # type: ignore[index]
            self.assertIn("non riuscita", restored["error"])  # type: ignore[index]
            self.assertEqual((manual_dir / "manual.faiss").read_bytes(), b"indice precedente")

    def test_interrupted_initial_ingestion_is_not_treated_as_valid_backup(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            manual_dir = Path(temp_dir) / "manual"
            partial_pages = manual_dir / "pages"
            partial_pages.mkdir(parents=True)
            (partial_pages / "page-0001.webp").write_bytes(b"partial")

            backup = ManualIndexManager._backup_generated_files(manual_dir)

            self.assertIsNone(backup)
            self.assertFalse(partial_pages.exists())

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
            self.assertEqual(hits[0]["ticket_number"], 3)
            self.assertEqual(hits[0]["ticket_url"], "https://ticket.centropaghe.it/scp/tickets.php?id=3")

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

    def test_search_exposes_public_number_and_internal_ticket_link(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            faiss_dir = Path(temp_dir)
            self._write_index(faiss_dir / "ticket_index.faiss", [[1.0, 1.0]])
            (faiss_dir / "ticket_ids.txt").write_text("7475088\n", encoding="utf-8")
            self._write_metadata(faiss_dir / "ticket_data.csv", ticket_ids=(7475088,))
            metadata = pd.read_csv(faiss_dir / "ticket_data.csv", sep=";")
            metadata.loc[0, "ticket_id"] = 1865112
            metadata.loc[0, "ticket_number"] = 1865081
            metadata.to_csv(faiss_dir / "ticket_data.csv", sep=";", index=False)

            hit = TicketStore(faiss_dir).search([1.0, 1.0], top_k=1)[0]

            self.assertEqual(hit["id"], 1865112)
            self.assertEqual(hit["ticket_number"], 1865081)
            self.assertEqual(hit["ticket_url"], "https://ticket.centropaghe.it/scp/tickets.php?id=1865112")

    def test_conversation_cleaning_keeps_question_and_answer(self) -> None:
        entries = [
            {
                "id": 10,
                "thread_id": 7,
                "ticket_id": 100,
                "ticket_number": 99,
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
                "ticket_id": 100,
                "ticket_number": 99,
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
                "ticket_id": 100,
                "ticket_number": 99,
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
        self.assertIn("Ticket #99", chunks[0])
        self.assertIn("Il cliente riceve anche una mail?", chunks[0])
        self.assertIn("[Operatore - Benedetta", chunks[0])
        self.assertIn("soltanto nei log", chunks[0])
        self.assertNotIn("Firma molto lunga", chunks[0])
        self.assertNotIn("testo precedente", chunks[0])
        self.assertNotIn("Ticket trasferito", chunks[0])

    def test_conversation_chunks_never_exceed_embedding_limit(self) -> None:
        entries = [
            {
                "id": 10,
                "thread_id": 7,
                "ticket_id": 100,
                "ticket_number": 99,
                "staff_id": 0,
                "user_id": 3,
                "poster": "Anna",
                "created": "2026-10-06 11:00:00",
                "title": "Titolo molto lungo " * 2_000,
                "body": "Dettaglio tecnico utile per la risoluzione. " * 5_000,
                "entry_type": "M",
            }
        ]

        chunks = VectorIndexBuilder._conversation_chunks(entries)

        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(VectorIndexBuilder._count_tokens(chunk) <= MAX_EMBED_TOKENS for chunk in chunks))

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

    def test_conversation_v1_query_is_migrated_with_ticket_identifiers(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            store = ConfigStore(Path(temp_dir) / "config.sqlite")
            store.save_config(DatabaseConfig(query=CONVERSATION_DEFAULT_QUERY_V1))

            migrated = store.get_config()

            self.assertEqual(migrated.query, DEFAULT_QUERY)
            self.assertIn("t.ticket_id", migrated.query)
            self.assertIn("t.number AS ticket_number", migrated.query)

    def test_conversation_v2_query_is_migrated_with_department_fields(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            store = ConfigStore(Path(temp_dir) / "config.sqlite")
            store.save_config(DatabaseConfig(query=CONVERSATION_DEFAULT_QUERY_V2))

            migrated = store.get_config()

            self.assertEqual(migrated.query, DEFAULT_QUERY)
            self.assertIn("d.name AS department_name", migrated.query)
            self.assertIn("t.source AS ticket_source", migrated.query)

    def test_cursor_groups_ordered_entries_by_thread(self) -> None:
        rows = [
            (1, 10, 0, 5, "Cliente", "2026-10-01 10:00:00", "Titolo A", "<p>Domanda del cliente abbastanza lunga.</p>", "M", 101, 1001, 8, "Paghe", "Email"),
            (2, 10, 7, 0, "Tecnico", "2026-10-01 11:00:00", "Titolo A", "<p>Risposta risolutiva del tecnico.</p>", "R", 101, 1001, 8, "Paghe", "Email"),
            (3, 20, 0, 6, "Altro cliente", "2026-10-02 10:00:00", "Titolo B", "<p>Seconda conversazione indipendente.</p>", "M", 202, 2002, 3, "Fiscale", "Web"),
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
            self.assertEqual(metadata.iloc[0]["ticket_id"], 101)
            self.assertEqual(metadata.iloc[0]["ticket_number"], 1001)
            self.assertEqual(metadata.iloc[0]["department_name"], "Paghe")
            self.assertEqual(metadata.iloc[0]["ticket_source"], "Email")

    def test_department_filter_and_permissions_exclude_other_tickets(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            faiss_dir = Path(temp_dir)
            self._write_index(faiss_dir / "ticket_index.faiss", [[1.0, 1.0], [1.1, 1.1]])
            (faiss_dir / "ticket_ids.txt").write_text("1\n2\n", encoding="utf-8")
            self._write_metadata(faiss_dir / "ticket_data.csv", ticket_ids=(1, 2))
            metadata = pd.read_csv(faiss_dir / "ticket_data.csv", sep=";")
            metadata.loc[0, ["department_id", "department_name", "ticket_source"]] = [20, "Zeta", "Email"]
            metadata.loc[1, ["department_id", "department_name", "ticket_source"]] = [10, "Alfa", "Web"]
            metadata.to_csv(faiss_dir / "ticket_data.csv", sep=";", index=False)

            store = TicketStore(faiss_dir)
            hits = store.search([1.0, 1.0], top_k=10, department_ids={10})
            options = store.available_filters({10, 20})

            self.assertEqual([hit["department_id"] for hit in hits], [10])
            self.assertEqual([item["name"] for item in options["departments"]], ["Alfa", "Zeta"])

    def test_user_department_permissions_are_persisted(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            store = AuthStore(Path(temp_dir) / "auth.sqlite")
            user = store.create_user("operatore", "segreta", "user", False, [8, 3])
            authenticated = store.authenticate("operatore", "segreta")

            self.assertFalse(user["all_departments"])
            self.assertEqual(user["department_ids"], [3, 8])
            self.assertIsNotNone(authenticated)
            self.assertEqual(authenticated.department_ids, (3, 8))  # type: ignore[union-attr]

    def test_manual_view_session_is_scoped_to_one_manual(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            store = AuthStore(Path(temp_dir) / "auth.sqlite")
            user = store.create_user("operatore", "segreta", "user", False, [8])
            token = store.create_manual_view_session(int(user["id"]), "manuale-a")

            authorized = store.user_for_manual_view_session(token, "manuale-a")

            self.assertIsNotNone(authorized)
            self.assertEqual(authorized.username, "operatore")  # type: ignore[union-attr]
            self.assertIsNone(store.user_for_manual_view_session(token, "manuale-b"))

    def test_osticket_user_is_created_and_departments_are_resynced(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            store = AuthStore(Path(temp_dir) / "auth.sqlite")
            created = store.upsert_osticket_user("jani.konicoti", 418, False, (136, 140))
            updated = store.upsert_osticket_user("jani.konicoti", 418, False, (136, 150))

            self.assertEqual(created.auth_source, "osticket")
            self.assertEqual(created.external_staff_id, 418)
            self.assertEqual(created.department_ids, (136, 140))
            self.assertEqual(updated.department_ids, (136, 150))
            self.assertEqual(len([user for user in store.list_users() if user["username"] == "jani.konicoti"]), 1)

    def test_osticket_bcrypt_login_reads_primary_and_additional_departments(self) -> None:
        password = "portale-segreto"
        stored_hash = bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt(rounds=4)).decode("ascii")
        stored_hash = stored_hash.replace("$2b$", "$2a$", 1)

        class FakeCursor:
            def __init__(self) -> None:
                self.query_count = 0

            def execute(self, *_args: object) -> None:
                self.query_count += 1

            def fetchone(self):  # type: ignore[no-untyped-def]
                return (418, "jani.konicoti", stored_hash, None, 1, 0, 136)

            def fetchall(self):  # type: ignore[no-untyped-def]
                return [(140,), (136,)]

            def close(self) -> None:
                return None

        class FakeConnection:
            def cursor(self) -> FakeCursor:
                return FakeCursor()

        identity = OsTicketAuthenticator._authenticate_connection(FakeConnection(), "jani.konicoti", password)

        self.assertIsNotNone(identity)
        self.assertEqual(identity.staff_id, 418)  # type: ignore[union-attr]
        self.assertEqual(identity.department_ids, (136, 140))  # type: ignore[union-attr]

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
                        ticket_id,
                        ticket_id,
                        ticket_id,
                        f"Reparto {ticket_id}",
                        "Email",
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
