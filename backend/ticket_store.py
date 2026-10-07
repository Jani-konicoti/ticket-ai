from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import faiss
import numpy as np
import pandas as pd


TEXT_COLUMNS = ("title", "clean_body")
STOPWORDS = {
    "che",
    "con",
    "del",
    "della",
    "delle",
    "dei",
    "gli",
    "per",
    "nel",
    "nella",
    "sono",
    "ticket",
    "richiesta",
    "problema",
    "errore",
    "cliente",
    "clienti",
    "utente",
    "utenti",
    "non",
    "una",
    "uno",
    "come",
    "alla",
    "alle",
    "dal",
    "dai",
    "tra",
    "piu",
}
HIGH_IMPACT_TERMS = (
    "blocc",
    "impossibile",
    "urgente",
    "massivo",
    "non funziona",
    "errore",
    "accesso",
    "login",
    "scaden",
    "fattur",
    "pagamento",
    "produzione",
    "ccnl",
)
GENERIC_GROUP_KEYS = {
    "assistenza",
    "richiesta",
    "risposta",
    "nota",
    "info",
    "subject",
    "senza titolo",
    "anomalia",
    "errore",
    "urgente",
}
IGNORE_TITLE_PATTERNS = (
    "successful mail delivery report",
    "delivery status notification",
    "undelivered mail returned",
    "read receipt",
    "ticket trasferito da",
    "ticket creato dall'inserimento",
)


def _shard_sort_key(path: Path) -> tuple[str, int]:
    match = re.match(r"^ticket_index\.append-(.+?)(?:-part(\d{3}))?\.faiss$", path.name)
    if not match:
        return path.name, 0
    group, part = match.groups()
    return group, int(part or 0)


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


def _keywords(text: str, limit: int = 6) -> list[str]:
    words = re.findall(r"[a-zA-Z0-9Ã Ã¨Ã©Ã¬Ã²Ã¹]{4,}", text.lower())
    counts: dict[str, int] = {}
    for word in words:
        normalized = word.strip()
        if normalized in STOPWORDS or normalized.isdigit():
            continue
        counts[normalized] = counts.get(normalized, 0) + 1
    ranked = sorted(counts.items(), key=lambda item: (item[1], item[0]), reverse=True)
    return [word for word, _ in ranked[:limit]]


def _category_for(text: str) -> str:
    lowered = text.lower()
    categories = [
        ("Accessi e autenticazione", ("accesso", "login", "password", "utenza", "utente", "whitenet")),
        ("Contratti e scadenze", ("ccnl", "contratt", "scaden", "agenda", "assuntiv")),
        ("Fatturazione e costi", ("fattur", "costi", "premio", "xml", "importi", "pagamento")),
        ("Import e procedure massive", ("massivo", "import", "caricamento", "procedura", "svecchia")),
        ("Comunicazioni e firme", ("sms", "otp", "firma", "mail", "pec")),
    ]
    for label, terms in categories:
        if any(term in lowered for term in terms):
            return label
    return "Operativita ticket"


def _priority_for(count: int, recent_count: int, high_impact_count: int) -> tuple[str, float]:
    score = float(count * 2 + recent_count * 1.5 + high_impact_count * 3)
    if count >= 10 or recent_count >= 5 or high_impact_count >= 3:
        return "Alta", score
    if count >= 3 or recent_count >= 2 or high_impact_count >= 1:
        return "Media", score
    return "Bassa", score


def _trend_for(count: int, recent_count: int, last_seen: Any, latest_date: Any) -> str:
    if count > 0 and recent_count >= max(2, int(count * 0.45)):
        return "In aumento"
    if latest_date is not None and not pd.isna(latest_date) and last_seen is not None and not pd.isna(last_seen):
        if latest_date - last_seen <= pd.Timedelta(days=3):
            return "Recente"
    return "Stabile"


def _is_ignored_title(title: str) -> bool:
    lowered = title.lower()
    return any(pattern in lowered for pattern in IGNORE_TITLE_PATTERNS)


def _problem_group_key(title: str, body: str) -> str:
    normalized_title = _normalize_group_key(title)
    if normalized_title not in GENERIC_GROUP_KEYS:
        return normalized_title

    combined = f"{title} {body}"
    category = _category_for(combined).lower()
    terms = _keywords(combined, limit=5)
    if terms:
        return " ".join([category, *terms[:4]])
    return normalized_title


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
    def __init__(self, faiss_dir: Path, ticket_url_base: str = "https://ticket.centropaghe.it/scp/tickets.php") -> None:
        self.faiss_dir = faiss_dir
        self.ticket_url_base = ticket_url_base.strip()
        self.index_path = faiss_dir / "ticket_index.faiss"
        self.csv_path = faiss_dir / "ticket_data.csv"
        self.ids_path = faiss_dir / "ticket_ids.txt"

        missing = [path for path in (self.index_path, self.csv_path, self.ids_path) if not path.exists()]
        if missing:
            missing_names = ", ".join(str(path) for path in missing)
            raise FileNotFoundError(f"File FAISS mancanti: {missing_names}")

        self.index_paths = [
            self.index_path,
            *sorted(faiss_dir.glob("ticket_index.append-*.faiss"), key=_shard_sort_key),
        ]
        self.indexes = [faiss.read_index(str(path)) for path in self.index_paths]
        self.index = self.indexes[0]
        self.ticket_ids = self._read_ticket_ids()
        self.tickets = self._read_tickets()
        self.tickets_by_id = self.tickets.set_index("id", drop=False)
        self.warnings: list[str] = []
        self.vector_count = sum(int(index.ntotal) for index in self.indexes)

        dimensions = {int(index.d) for index in self.indexes}
        if len(dimensions) != 1:
            raise ValueError(f"Gli indici FAISS hanno dimensioni incompatibili: {sorted(dimensions)}")
        metrics = {int(index.metric_type) for index in self.indexes}
        if len(metrics) != 1:
            raise ValueError("Gli indici FAISS usano metriche incompatibili.")
        self.dimension = dimensions.pop()
        self.metric_type = metrics.pop()

        if len(self.ticket_ids) != self.vector_count:
            self.warnings.append(
                f"ticket_ids.txt contiene {len(self.ticket_ids)} righe, ma gli indici FAISS hanno {self.vector_count} vettori. "
                "I vettori senza ID verranno ignorati nei risultati."
            )
        if len(self.tickets) != self.vector_count:
            self.warnings.append(
                f"ticket_data.csv contiene {len(self.tickets)} righe, ma gli indici FAISS hanno {self.vector_count} vettori. "
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
        for optional_column in ("ticket_id", "ticket_number", "department_id", "department_name", "ticket_source"):
            if optional_column not in df.columns:
                df[optional_column] = None
        df["created_dt"] = pd.to_datetime(df["created"], errors="coerce")
        return df

    @property
    def stats(self) -> StoreStats:
        return StoreStats(
            vectors=self.vector_count,
            dimension=self.dimension,
            tickets=len(self.tickets),
            ids=len(self.ticket_ids),
            warnings=self.warnings,
            index_path=self.index_path,
            csv_path=self.csv_path,
            ids_path=self.ids_path,
        )

    def available_filters(self, allowed_department_ids: set[int] | None = None) -> dict[str, list[dict[str, Any]] | list[str]]:
        rows = self._filtered_tickets(allowed_department_ids, None)
        departments: dict[int, str] = {}
        for _, row in rows.iterrows():
            department_id = self._int_value(row.get("department_id"))
            if department_id is None:
                continue
            name = _clean_text(row.get("department_name")) or f"Reparto {department_id}"
            departments[department_id] = name
        sources = sorted({_clean_text(value) for value in rows["ticket_source"].tolist() if _clean_text(value)}, key=str.casefold)
        return {
            "departments": [
                {"id": department_id, "name": name}
                for department_id, name in sorted(departments.items(), key=lambda item: (item[1].casefold(), item[0]))
            ],
            "sources": sources,
        }

    def search(
        self,
        query_embedding: list[float],
        top_k: int = 8,
        department_ids: set[int] | None = None,
        ticket_source: str | None = None,
    ) -> list[dict[str, Any]]:
        vector = np.array([query_embedding], dtype="float32")
        if vector.shape[1] != self.dimension:
            raise ValueError(
                f"L'embedding della domanda ha dimensione {vector.shape[1]}, ma l'indice FAISS richiede {self.dimension}. "
                "Verifica OPENAI_EMBEDDING_MODEL."
            )

        if self.metric_type == faiss.METRIC_INNER_PRODUCT:
            faiss.normalize_L2(vector)

        metadata_count = min(len(self.ticket_ids), len(self.tickets))
        candidates: list[tuple[float, int]] = []
        offset = 0
        for index in self.indexes:
            search_k = min(int(index.ntotal), max(top_k * 100, 1000))
            if search_k:
                distances, positions = index.search(vector, search_k)
                candidates.extend(
                    (float(score), offset + int(position))
                    for score, position in zip(distances[0], positions[0])
                    if position >= 0
                )
            offset += int(index.ntotal)

        candidates.sort(key=lambda item: item[0], reverse=self.metric_type == faiss.METRIC_INNER_PRODUCT)
        hits: list[dict[str, Any]] = []
        for score, position in candidates:
            if position >= metadata_count:
                continue
            ticket_id = self.ticket_ids[position]
            row = self._row_for_position(ticket_id, position)
            if row is None:
                continue
            if not self._matches_filters(row, department_ids, ticket_source):
                continue
            body = _clean_text(row.get("clean_body"))
            title = _clean_text(row.get("title")) or "Senza titolo"
            source_ticket_id = self._json_value(row.get("ticket_id"))
            ticket_number = self._json_value(row.get("ticket_number"))
            hits.append(
                {
                    "rank": len(hits) + 1,
                    "score": score,
                    "id": source_ticket_id or self._json_value(row.get("id", ticket_id)),
                    "thread_id": self._json_value(row.get("thread_id")),
                    "ticket_id": source_ticket_id,
                    "ticket_number": ticket_number,
                    "ticket_url": self._ticket_url(source_ticket_id),
                    "department_id": self._json_value(row.get("department_id")),
                    "department_name": _clean_text(row.get("department_name")) or None,
                    "ticket_source": _clean_text(row.get("ticket_source")) or None,
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

    def recent_problem_groups(
        self,
        days: int = 30,
        limit: int = 12,
        department_ids: set[int] | None = None,
        ticket_source: str | None = None,
    ) -> tuple[pd.DataFrame, list[dict[str, Any]], str | None]:
        filtered_tickets = self._filtered_tickets(department_ids, ticket_source)
        latest_date = filtered_tickets["created_dt"].max()
        if pd.isna(latest_date):
            recent_rows = filtered_tickets.copy()
            since = None
        else:
            since_dt = latest_date - pd.Timedelta(days=days)
            recent_rows = filtered_tickets[filtered_tickets["created_dt"] >= since_dt].copy()
            since = since_dt.strftime("%Y-%m-%d %H:%M:%S")

        recent = recent_rows.sort_values("created_dt", ascending=False).drop_duplicates("id", keep="first").copy()
        recent = recent[~recent["title"].map(lambda title: _is_ignored_title(_clean_text(title)))].copy()
        recent["group_key"] = recent.apply(
            lambda row: _problem_group_key(_clean_text(row.get("title")), _clean_text(row.get("clean_body"))),
            axis=1,
        )
        grouped: list[dict[str, Any]] = []
        for key, group in recent.groupby("group_key", dropna=False):
            group = group.sort_values("created_dt", ascending=False)
            count = int(group["id"].nunique())
            first_seen_dt = group["created_dt"].min()
            last_seen_dt = group["created_dt"].max()
            recent_count = 0
            if latest_date is not None and not pd.isna(latest_date):
                recent_count = int(group[group["created_dt"] >= latest_date - pd.Timedelta(days=7)]["id"].nunique())
            combined_text = " ".join(
                _clean_text(value)
                for value in pd.concat([group["title"].head(12), group["clean_body"].head(12)], ignore_index=True).tolist()
            )
            lowered = combined_text.lower()
            high_impact_count = sum(1 for term in HIGH_IMPACT_TERMS if term in lowered)
            priority, priority_score = _priority_for(count, recent_count, high_impact_count)
            latest_tickets = []
            for _, row in group.head(6).iterrows():
                source_ticket_id = self._json_value(row.get("ticket_id"))
                latest_tickets.append(
                    {
                        "id": source_ticket_id or self._json_value(row.get("id")),
                        "ticket_id": source_ticket_id,
                        "ticket_number": self._json_value(row.get("ticket_number")),
                        "ticket_url": self._ticket_url(source_ticket_id),
                        "department_id": self._json_value(row.get("department_id")),
                        "department_name": _clean_text(row.get("department_name")) or None,
                        "ticket_source": _clean_text(row.get("ticket_source")) or None,
                        "created": self._json_value(row.get("created")),
                        "title": _clean_text(row.get("title")) or "Senza titolo",
                        "poster": self._json_value(row.get("poster")),
                        "excerpt": _excerpt(_clean_text(row.get("clean_body")), 360),
                    }
                )
            titles = [_clean_text(title) for title in group["title"].head(3).tolist()]
            grouped.append(
                {
                    "key": str(key),
                    "title": titles[0] if titles else str(key),
                    "count": count,
                    "unique_ticket_count": count,
                    "priority": priority,
                    "priority_score": priority_score,
                    "category": _category_for(combined_text),
                    "trend": _trend_for(count, recent_count, last_seen_dt, latest_date),
                    "recurring": count >= 2,
                    "first_seen": self._date_value(first_seen_dt),
                    "last_seen": self._date_value(last_seen_dt),
                    "sample_ticket_ids": [self._json_value(value) for value in group["ticket_number"].head(8).tolist()],
                    "sample_titles": titles,
                    "keywords": _keywords(combined_text),
                    "latest_tickets": latest_tickets,
                }
            )

        priority_rank = {"Alta": 3, "Media": 2, "Bassa": 1}
        grouped.sort(
            key=lambda item: (
                priority_rank.get(str(item["priority"]), 0),
                item["priority_score"],
                item["count"],
                item["last_seen"] or "",
            ),
            reverse=True,
        )
        return recent, grouped[:limit], since

    def recent_sample_for_prompt(
        self,
        days: int = 30,
        limit: int = 80,
        department_ids: set[int] | None = None,
        ticket_source: str | None = None,
    ) -> list[dict[str, Any]]:
        recent, _, _ = self.recent_problem_groups(
            days=days, limit=limit, department_ids=department_ids, ticket_source=ticket_source
        )
        recent = recent.sort_values("created_dt", ascending=False).head(limit)
        sample = []
        for _, row in recent.iterrows():
            sample.append(
                {
                    "id": self._json_value(row.get("ticket_number")) or self._json_value(row.get("ticket_id")) or self._json_value(row.get("id")),
                    "ticket_number": self._json_value(row.get("ticket_number")),
                    "department_name": _clean_text(row.get("department_name")) or None,
                    "ticket_source": _clean_text(row.get("ticket_source")) or None,
                    "created": self._json_value(row.get("created")),
                    "title": _clean_text(row.get("title")),
                    "body": _excerpt(_clean_text(row.get("clean_body")), 350),
                }
            )
        return sample

    def _filtered_tickets(self, department_ids: set[int] | None, ticket_source: str | None) -> pd.DataFrame:
        rows = self.tickets
        if department_ids is not None:
            if not department_ids:
                return rows.iloc[0:0].copy()
            numeric_departments = pd.to_numeric(rows["department_id"], errors="coerce")
            rows = rows[numeric_departments.isin(department_ids)]
        normalized_source = _clean_text(ticket_source)
        if normalized_source:
            rows = rows[rows["ticket_source"].map(_clean_text).str.casefold() == normalized_source.casefold()]
        return rows

    @classmethod
    def _matches_filters(
        cls, row: pd.Series, department_ids: set[int] | None, ticket_source: str | None
    ) -> bool:
        if department_ids is not None and cls._int_value(row.get("department_id")) not in department_ids:
            return False
        normalized_source = _clean_text(ticket_source)
        return not normalized_source or _clean_text(row.get("ticket_source")).casefold() == normalized_source.casefold()

    @staticmethod
    def _int_value(value: Any) -> int | None:
        if value is None or pd.isna(value):
            return None
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    def _ticket_url(self, ticket_id: Any) -> str | None:
        value = self._json_value(ticket_id)
        if value is None or not self.ticket_url_base:
            return None
        separator = "&" if "?" in self.ticket_url_base else "?"
        return f"{self.ticket_url_base}{separator}id={value}"

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
