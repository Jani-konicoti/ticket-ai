from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=5000)
    top_k: int = Field(default=8, ge=1, le=20)
    min_score: float | None = None


class TicketHit(BaseModel):
    rank: int
    score: float
    id: int | str
    thread_id: int | str | None = None
    title: str
    created: str | None = None
    poster: str | None = None
    excerpt: str
    body: str


class AskResponse(BaseModel):
    answer: str
    hits: list[TicketHit]
    model: str
    embedding_model: str


class LoginRequest(BaseModel):
    username: str = Field(min_length=1, max_length=80)
    password: str = Field(min_length=1, max_length=200)


class UserResponse(BaseModel):
    id: int
    username: str
    role: str
    active: bool | int = True
    created_at: str | None = None


class AuthResponse(BaseModel):
    token: str
    user: UserResponse


class CreateUserRequest(BaseModel):
    username: str = Field(min_length=1, max_length=80)
    password: str = Field(min_length=4, max_length=200)
    role: str = Field(pattern="^(admin|user)$")


class ProblemGroup(BaseModel):
    key: str
    title: str
    count: int
    unique_ticket_count: int
    priority: str
    priority_score: float
    category: str
    trend: str
    recurring: bool
    first_seen: str | None = None
    last_seen: str | None = None
    sample_ticket_ids: list[int | str]
    sample_titles: list[str]
    keywords: list[str] = []
    latest_tickets: list[dict] = []


class RecentProblemsResponse(BaseModel):
    since: str | None
    total_recent_tickets: int
    groups: list[ProblemGroup]
    priority_counts: dict[str, int] = {}
    recurring_count: int = 0
    ai_summary: str | None = None
    ai_error: str | None = None
    generated_at: str | None = None
    vector_count: int = 0


class DatabaseConfig(BaseModel):
    ssh_host: str = ""
    ssh_port: int = 22
    ssh_user: str = ""
    ssh_password: str = ""
    db_host: str = "127.0.0.1"
    db_port: int = 3306
    db_user: str = ""
    db_password: str = ""
    db_name: str = ""
    query: str = ""
    batch_size: int = 150


class SaveConfigRequest(BaseModel):
    config: DatabaseConfig


class RebuildRequest(BaseModel):
    from_date: str


class JobResponse(BaseModel):
    id: str
    type: str
    status: str
    step: str
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
    started_at: str
    finished_at: str | None = None
