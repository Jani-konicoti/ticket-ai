from __future__ import annotations

import gc
from functools import lru_cache

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from .config import Settings, get_settings
from .index_builder import ConfigStore, JobManager, VectorIndexBuilder
from .models import AskRequest, AskResponse, JobResponse, RecentProblemsResponse, RebuildRequest, SaveConfigRequest, TicketHit
from .openai_service import OpenAIService
from .ticket_store import TicketStore


app = FastAPI(title="Ticket Knowledge Assistant", version="0.1.0")


@lru_cache(maxsize=1)
def get_store() -> TicketStore:
    settings = get_settings()
    return TicketStore(settings.faiss_dir)


@lru_cache(maxsize=1)
def get_config_store() -> ConfigStore:
    return ConfigStore(get_settings().faiss_dir.parent / "backend" / "app_config.sqlite")


@lru_cache(maxsize=1)
def get_openai_service() -> OpenAIService:
    settings = get_settings()
    return OpenAIService(
        api_key=settings.openai_api_key,
        chat_model=settings.openai_chat_model,
        embedding_model=settings.openai_embedding_model,
    )


job_manager = JobManager()


def get_builder() -> VectorIndexBuilder:
    settings = get_settings()
    return VectorIndexBuilder(
        faiss_dir=settings.faiss_dir,
        openai_api_key=settings.openai_api_key,
        on_complete=get_store.cache_clear,
    )


settings: Settings = get_settings()
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health() -> dict[str, object]:
    active_job = job_manager.active()
    if active_job and active_job.status in {"queued", "running"}:
        settings = get_settings()
        return {
            "ok": False,
            "building": True,
            "error": "Ricostruzione indice in corso.",
            "openai_configured": bool(settings.openai_api_key),
            "chat_model": settings.openai_chat_model,
            "embedding_model": "text-embedding-3-large",
            "embedding_model_setting": "build-running",
            "vectors": active_job.written,
            "dimension": 3072 if active_job.written else 0,
            "tickets": active_job.processed,
            "ids": active_job.written,
            "warnings": ["FAISS in ricostruzione. La ricerca sara' disponibile al completamento."],
            "faiss_dir": str(settings.faiss_dir),
        }
    try:
        store = get_store()
        stats = store.stats
        openai_service = get_openai_service()
        embedding_model = openai_service.resolve_embedding_model(stats.dimension)
        return {
            "ok": True,
            "openai_configured": openai_service.configured(),
            "chat_model": openai_service.chat_model,
            "embedding_model": embedding_model,
            "embedding_model_setting": openai_service.embedding_model,
            "vectors": stats.vectors,
            "dimension": stats.dimension,
            "tickets": stats.tickets,
            "ids": stats.ids,
            "warnings": stats.warnings,
            "faiss_dir": str(stats.index_path.parent),
        }
    except FileNotFoundError as exc:
        settings = get_settings()
        return {
            "ok": False,
            "needs_index": True,
            "error": str(exc),
            "openai_configured": bool(settings.openai_api_key),
            "chat_model": settings.openai_chat_model,
            "embedding_model": "text-embedding-3-large",
            "embedding_model_setting": "build-required",
            "vectors": 0,
            "dimension": 0,
            "tickets": 0,
            "ids": 0,
            "warnings": ["Indice FAISS non ancora creato. Usa Configurazione > Rifai FAISS da zero."],
            "faiss_dir": str(settings.faiss_dir),
        }
    except Exception as exc:
        settings = get_settings()
        return {
            "ok": False,
            "needs_index": True,
            "error": "Indice FAISS non leggibile. Esegui una ricostruzione da zero dalla pagina Configurazione.",
            "technical_error": str(exc),
            "openai_configured": bool(settings.openai_api_key),
            "chat_model": settings.openai_chat_model,
            "embedding_model": "text-embedding-3-large",
            "embedding_model_setting": "build-required",
            "vectors": 0,
            "dimension": 0,
            "tickets": 0,
            "ids": 0,
            "warnings": ["Indice FAISS corrotto o incompleto. Usa Configurazione > Rifai FAISS da zero."],
            "faiss_dir": str(settings.faiss_dir),
        }


@app.get("/api/config")
def read_config() -> dict[str, object]:
    builder = get_builder()
    return {
        "config": get_config_store().get_config(),
        "latest_local_ticket_date": builder.latest_local_ticket_date(),
        "embedding_model_for_build": "text-embedding-3-large",
    }


@app.post("/api/config")
def save_config(payload: SaveConfigRequest) -> dict[str, object]:
    config = get_config_store().save_config(payload.config)
    return {"config": config}


@app.get("/api/index/latest-date")
def latest_ticket_date() -> dict[str, object]:
    return {"latest_local_ticket_date": get_builder().latest_local_ticket_date()}


@app.post("/api/index/rebuild", response_model=JobResponse)
def rebuild_index(payload: RebuildRequest) -> JobResponse:
    try:
        config = get_config_store().get_config()
        builder = get_builder()
        get_store.cache_clear()
        gc.collect()
        job = job_manager.start("rebuild", lambda state: builder.rebuild(config, payload.from_date, state))
        return JobResponse(**job.to_dict())
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/index/rebuild/resume", response_model=JobResponse)
def resume_rebuild_index(payload: RebuildRequest) -> JobResponse:
    try:
        config = get_config_store().get_config()
        builder = get_builder()
        get_store.cache_clear()
        gc.collect()
        job = job_manager.start("resume_rebuild", lambda state: builder.resume_rebuild(config, payload.from_date, state))
        return JobResponse(**job.to_dict())
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/index/append", response_model=JobResponse)
def append_index() -> JobResponse:
    try:
        config = get_config_store().get_config()
        builder = get_builder()
        get_store.cache_clear()
        gc.collect()
        job = job_manager.start("append", lambda state: builder.append_until_today(config, state))
        return JobResponse(**job.to_dict())
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/api/index/jobs/latest", response_model=JobResponse | None)
def latest_job() -> JobResponse | None:
    job = job_manager.latest()
    return JobResponse(**job.to_dict()) if job else None


@app.get("/api/index/jobs/{job_id}", response_model=JobResponse)
def get_job(job_id: str) -> JobResponse:
    job = job_manager.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job non trovato")
    return JobResponse(**job.to_dict())


@app.post("/api/ask", response_model=AskResponse)
def ask(payload: AskRequest) -> AskResponse:
    try:
        store = get_store()
        openai_service = get_openai_service()
        embedding_model = openai_service.resolve_embedding_model(store.stats.dimension)
        embedding = openai_service.embed(payload.question, embedding_model)
        raw_hits = store.search(embedding, top_k=payload.top_k)
        if payload.min_score is not None:
            raw_hits = [hit for hit in raw_hits if hit["score"] >= payload.min_score]
        hits = [TicketHit(**hit) for hit in raw_hits]
        if not hits:
            return AskResponse(
                answer="Non ho trovato ticket abbastanza simili con la soglia impostata.",
                hits=[],
                model=openai_service.chat_model,
                embedding_model=embedding_model,
            )
        answer = openai_service.answer_question(payload.question, hits)
        return AskResponse(
            answer=answer,
            hits=hits,
            model=openai_service.chat_model,
            embedding_model=embedding_model,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=409, detail=f"Indice FAISS non ancora creato: {exc}") from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/api/analysis/recent-problems", response_model=RecentProblemsResponse)
def recent_problems(
    days: int = Query(default=30, ge=1, le=365),
    limit: int = Query(default=12, ge=1, le=50),
    include_ai: bool = Query(default=True),
) -> RecentProblemsResponse:
    try:
        store = get_store()
        recent, groups, since = store.recent_problem_groups(days=days, limit=limit)
        ai_summary = None
        ai_error = None
        openai_service = get_openai_service()
        if include_ai and openai_service.configured():
            try:
                ai_summary = openai_service.summarize_recent_problems(store.recent_sample_for_prompt(days=days), groups)
            except Exception as exc:
                ai_error = str(exc)
        priority_counts = {"Alta": 0, "Media": 0, "Bassa": 0}
        for group in groups:
            priority = str(group.get("priority", "Bassa"))
            priority_counts[priority] = priority_counts.get(priority, 0) + 1
        return RecentProblemsResponse(
            since=since,
            total_recent_tickets=len(recent),
            groups=groups,
            priority_counts=priority_counts,
            recurring_count=sum(1 for group in groups if group.get("recurring")),
            ai_summary=ai_summary,
            ai_error=ai_error,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=409, detail=f"Indice FAISS non ancora creato: {exc}") from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
