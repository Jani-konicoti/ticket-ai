from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

import faiss
import numpy as np

from backend.index_builder import VectorIndexBuilder
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

            def capture_path(*_args: object, **_kwargs: object) -> None:
                selected_paths.append(builder.index_path)

            builder._run_query_to_index = capture_path  # type: ignore[method-assign]
            builder.append_until_today(DatabaseConfig(), type("Job", (), {})())  # type: ignore[arg-type]

            self.assertEqual(len(selected_paths), 1)
            self.assertRegex(selected_paths[0].name, r"^ticket_index\.append-\d{6}\.faiss$")
            self.assertEqual(builder.index_path, faiss_dir / "ticket_index.faiss")

    @staticmethod
    def _write_index(path: Path, vectors: list[list[float]]) -> None:
        index = faiss.IndexFlatL2(2)
        index.add(np.asarray(vectors, dtype="float32"))
        faiss.write_index(index, str(path))

    @staticmethod
    def _write_metadata(path: Path, ticket_ids: tuple[int, ...] = (1, 2, 3)) -> None:
        columns = ["id", "thread_id", "staff_id", "user_id", "poster", "created", "title", "clean_body", "chunk_index", "chunk_count"]
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle, delimiter=";")
            writer.writerow(columns)
            for ticket_id in ticket_ids:
                writer.writerow([ticket_id, ticket_id, "", "", "test", "2026-10-01 00:00:00", f"Ticket {ticket_id}", "body", 1, 1])


if __name__ == "__main__":
    unittest.main()
