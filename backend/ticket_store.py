from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import faiss
import numpy as np
import pandas as pd


TEXT_COLUMNS = ("title", "clean_body")


def _clean_text(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    return re.sub(r"\s+", " ", str(value)).strip()


def _excerpt(text: str, limit: int = 700) -> str:
    text = _clean_text(text)
    if len(text) <= limit:
        return text
    return text[: limit - 1].rstrip() + "..."


def _normalize_group_key(title: str) -> str:
    value = title.lower()
    value = re.sub(r"\b(rif|re|fw|fwd|ticket|richiesta|anomalia|problema)\b[:\-\s]*", " ", value)
    value = re.sub(r"[^a-z0-9àèéìòù]+", " ", value)
    words = [w for w in value.split() if len(w) > 2]
    return " ".join(words[:8]) or "senza titolo"


@dataclass(frozen=True)
class StoreStats:
    vectors: int
    dimension: int
    tickets: int
    ids: int
    warnings: list[str]
    index_path: Path
    csv_path: Path
    ids_path: Path


class TicketStore:
    def __init__(self, faiss_dir: Path) -> None:
        self.faiss_dir = faiss_dir
        self.index_path = faiss_dir / "ticket_index.faiss"
        self.csv_path = faiss_dir / "ticket_data.csv"
        self.ids_path = faiss_dir / "ticket_ids.txt"

        missing = [path for path in (self.index_path, self.csv_path, self.ids_path) if not path.exists()]
        if missing:
            missing_names = ", ".join(str(path) for path in missing)
            raise FileNotFoundError(f"File FAISS mancanti: {missing_names}")

        self.index = faiss.read_index(str(self.index_path))
        self.ticket_ids = self._read_ticket_ids()
        self.tickets = self._read_tickets()
        self.tickets_by_id = self.tickets.set_index("id", drop=False)
        self.warnings: list[str] = []

        if len(self.ticket_ids) != self.index.ntotal:
            self.warnings.append(
                f"ticket_ids.txt contiene {len(self.ticket_ids)} righe, ma l'indice FAISS ha {self.index.ntotal} vettori. "
                "I vettori senza ID verranno ignorati nei risultati."
            )
        if len(self.tickets) != self.index.ntotal:
            self.warnings.append(
                f"ticket_data.csv contiene {len(self.tickets)} righe, ma l'indice FAISS ha {self.index.ntotal} vettori. "
                "I vettori senza riga CSV verranno ignorati nei risultati."
            )

    def _read_ticket_ids(self) -> list[str]:
        return [
            line.strip()
            for line in self.ids_path.read_text(encoding="utf-8", errors="ignore").splitlines()
            if line.strip()
        ]

    def _read_tickets(self) -> pd.DataFrame:
        df = pd.read_csv(self.csv_path, sep=";", encoding="utf-8")
        expected = {"id", "thread_id", "poster", "created", "title", "clean_body"}
        missing = expected - set(df.columns)
        if missing:
            raise ValueError(f"Colonne mancanti nel CSV: {', '.join(sorted(missing))}")
        df["created_dt"] = pd.to_datetime(df["created"], errors="coerce")
        return df

    @property
    def stats(self) -> StoreStats:
        return StoreStats(
            vectors=self.index.ntotal,
            dimension=self.index.d,
            tickets=len(self.tickets),
            ids=len(self.ticket_ids),
            warnings=self.warnings,
            index_path=self.index_path,
            csv_path=self.csv_path,
            ids_path=self.ids_path,
        )

    def search(self, query_embedding: list[float], top_k: int = 8) -> list[dict[str, Any]]:
        vector = np.array([query_embedding], dtype="float32")
        if vector.shape[1] != self.index.d:
            raise ValueError(
                f"L'embedding della domanda ha dimensione {vector.shape[1]}, ma l'indice FAISS richiede {self.index.d}. "
                "Verifica OPENAI_EMBEDDING_MODEL."
            )

        if self.index.metric_type == faiss.METRIC_INNER_PRODUCT:
            faiss.normalize_L2(vector)

        metadata_count = min(len(self.ticket_ids), len(self.tickets))
        metadata_gap = max(0, self.index.ntotal - metadata_count)
        search_k = min(self.index.ntotal, top_k + min(metadata_gap, 500))
        distances, positions = self.index.search(vector, search_k)
        hits: list[dict[str, Any]] = []
        for rank, (score, position) in enumerate(zip(distances[0], positions[0]), start=1):
            if position < 0:
                continue
            if int(position) >= metadata_count:
                continue
            ticket_id = self.ticket_ids[int(position)]
            row = self._row_for_position(ticket_id, int(position))
            if row is None:
                continue
            body = _clean_text(row.get("clean_body"))
            title = _clean_text(row.get("title")) or "Senza titolo"
            hits.append(
                {
                    "rank": rank,
                    "score": float(score),
                    "id": self._json_value(row.get("id", ticket_id)),
                    "thread_id": self._json_value(row.get("thread_id")),
                    "title": title,
                    "created": self._json_value(row.get("created")),
                    "poster": self._json_value(row.get("poster")),
                    "excerpt": _excerpt(body),
                    "body": body,
                }
            )
            if len(hits) >= top_k:
                break
        return hits

    def _row_for_position(self, ticket_id: str, position: int) -> pd.Series | None:
        if position < len(self.tickets):
            return self.tickets.iloc[position]

        try:
            numeric_id = int(ticket_id)
        except ValueError:
            numeric_id = ticket_id

        if numeric_id in self.tickets_by_id.index:
            row = self.tickets_by_id.loc[numeric_id]
            if isinstance(row, pd.DataFrame):
                return row.iloc[0]
            return row
        return None

    def recent_problem_groups(self, days: int = 30, limit: int = 12) -> tuple[pd.DataFrame, list[dict[str, Any]], str | None]:
        latest_date = self.tickets["created_dt"].max()
        if pd.isna(latest_date):
            recent = self.tickets.copy()
            since = None
        else:
            since_dt = latest_date - pd.Timedelta(days=days)
            recent = self.tickets[self.tickets["created_dt"] >= since_dt].copy()
            since = since_dt.strftime("%Y-%m-%d %H:%M:%S")

        recent["group_key"] = recent["title"].map(lambda title: _normalize_group_key(_clean_text(title)))
        grouped: list[dict[str, Any]] = []
        for key, group in recent.groupby("group_key", dropna=False):
            group = group.sort_values("created_dt", ascending=False)
            titles = [_clean_text(title) for title in group["title"].head(3).tolist()]
            grouped.append(
                {
                    "key": str(key),
                    "title": titles[0] if titles else str(key),
                    "count": int(len(group)),
                    "first_seen": self._date_value(group["created_dt"].min()),
                    "last_seen": self._date_value(group["created_dt"].max()),
                    "sample_ticket_ids": [self._json_value(value) for value in group["id"].head(5).tolist()],
                    "sample_titles": titles,
                }
            )

        grouped.sort(key=lambda item: (item["count"], item["last_seen"] or ""), reverse=True)
        return recent, grouped[:limit], since

    def recent_sample_for_prompt(self, days: int = 30, limit: int = 80) -> list[dict[str, Any]]:
        recent, _, _ = self.recent_problem_groups(days=days, limit=limit)
        recent = recent.sort_values("created_dt", ascending=False).head(limit)
        sample = []
        for _, row in recent.iterrows():
            sample.append(
                {
                    "id": self._json_value(row.get("id")),
                    "created": self._json_value(row.get("created")),
                    "title": _clean_text(row.get("title")),
                    "body": _excerpt(_clean_text(row.get("clean_body")), 350),
                }
            )
        return sample

    @staticmethod
    def _date_value(value: Any) -> str | None:
        if value is None or pd.isna(value):
            return None
        if hasattr(value, "strftime"):
            return value.strftime("%Y-%m-%d %H:%M:%S")
        return str(value)

    @staticmethod
    def _json_value(value: Any) -> Any:
        if value is None or pd.isna(value):
            return None
        if isinstance(value, np.integer):
            return int(value)
        if isinstance(value, np.floating):
            return float(value)
        return value
