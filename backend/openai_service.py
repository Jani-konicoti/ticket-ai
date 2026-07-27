from __future__ import annotations

import json
from typing import Any

from openai import OpenAI
import truststore

from .models import TicketHit


truststore.inject_into_ssl()


class OpenAIService:
    def __init__(self, api_key: str, chat_model: str, embedding_model: str) -> None:
        self.api_key = api_key
        self.chat_model = chat_model
        self.embedding_model = embedding_model
        self.client = OpenAI(api_key=api_key, timeout=120.0, max_retries=1) if api_key else None

    def configured(self) -> bool:
        return self.client is not None

    def resolve_embedding_model(self, index_dimension: int) -> str:
        configured_model = self.embedding_model.strip()
        if configured_model and configured_model.lower() != "auto":
            return configured_model

        dimension_to_model = {
            1536: "text-embedding-3-small",
            3072: "text-embedding-3-large",
        }
        try:
            return dimension_to_model[index_dimension]
        except KeyError as exc:
            raise ValueError(
                f"Non so scegliere automaticamente un modello embedding per indice FAISS a {index_dimension} dimensioni. "
                "Imposta OPENAI_EMBEDDING_MODEL esplicitamente nel file .env."
            ) from exc

    def embed(self, text: str, model: str) -> list[float]:
        if not self.client:
            raise RuntimeError("OPENAI_API_KEY non configurata.")
        response = self.client.embeddings.create(model=model, input=text)
        return response.data[0].embedding

    def answer_question(self, question: str, hits: list[TicketHit]) -> str:
        if not self.client:
            raise RuntimeError("OPENAI_API_KEY non configurata.")

        context = "\n\n".join(
            [
                (
                    f"[Ticket {hit.id} | score {hit.score:.4f} | {hit.created or 'data n/d'}]\n"
                    f"Titolo: {hit.title}\n"
                    f"Testo: {hit.body[:2500]}"
                )
                for hit in hits
            ]
        )
        system = (
            "Sei un assistente per supporto tecnico interno. Rispondi in italiano. "
            "Usa solo le informazioni presenti nei ticket forniti. "
            "Se i ticket non bastano, dillo chiaramente e proponi cosa verificare. "
            "Cita sempre gli ID ticket piu' rilevanti tra parentesi."
        )
        user = (
            f"Domanda dell'operatore:\n{question}\n\n"
            f"Ticket simili trovati:\n{context}\n\n"
            "Produci una risposta pratica: possibile causa, soluzione gia' vista, passaggi consigliati, ticket citati."
        )
        return self._generate_text(system, user)

    def summarize_recent_problems(self, tickets: list[dict[str, Any]], groups: list[dict[str, Any]] | None = None) -> str | None:
        if not self.client or not tickets:
            return None

        compact_json = json.dumps({"groups": groups or [], "recent_tickets": tickets}, ensure_ascii=False)
        system = (
            "Sei un analista di ticketing. Devi individuare problemi ricorrenti o recenti. "
            "Rispondi in italiano, in modo dettagliato ma operativo. "
            "Usa Markdown semplice, senza tabelle."
        )
        user = (
            "Analizza questi ticket recenti e i gruppi gia' calcolati. "
            "Dai piu' peso ai gruppi con priority Alta/Media, ai count elevati e ai trend in aumento. "
            "Non inventare dati: cita sempre ID ticket presenti nel JSON.\n\n"
            "Formato obbligatorio:\n"
            "## Priorita operative\n"
            "- [Alta] problema; perche' e' alta; impatto; ticket citati; prima azione.\n"
            "- [Media] problema; perche' e' media; impatto; ticket citati; prima azione.\n\n"
            "## Problemi ricorrenti\n"
            "### Nome problema\n"
            "- Frequenza: numero casi, periodo, trend.\n"
            "- Evidenze: cosa sta succedendo e ticket citati.\n"
            "- Possibile causa: solo se emerge dai ticket, altrimenti scrivi da verificare.\n"
            "- Azione: cosa verificare o fare.\n\n"
            "## Da monitorare\n"
            "- Elementi da tenere sotto controllo nei prossimi giorni.\n\n"
            f"{compact_json}"
        )
        return self._generate_text(system, user)

    def _generate_text(self, system: str, user: str) -> str:
        assert self.client is not None
        response = self.client.responses.create(
            model=self.chat_model,
            input=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        text = getattr(response, "output_text", None)
        if text:
            return text.strip()

        chunks: list[str] = []
        for item in getattr(response, "output", []) or []:
            for content in getattr(item, "content", []) or []:
                value = getattr(content, "text", None)
                if value:
                    chunks.append(value)
        return "\n".join(chunks).strip()
