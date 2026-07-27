from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


load_dotenv()


class Settings(BaseSettings):
    openai_api_key: str = Field(default="", alias="OPENAI_API_KEY")
    openai_chat_model: str = Field(default="gpt-4.1-mini", alias="OPENAI_CHAT_MODEL")
    openai_embedding_model: str = Field(default="auto", alias="OPENAI_EMBEDDING_MODEL")
    faiss_dir: Path = Field(default=Path("FAISS"), alias="FAISS_DIR")
    app_data_dir: Path = Field(default=Path("backend"), alias="APP_DATA_DIR")
    analysis_schedule_hour: int = Field(default=2, ge=0, le=23, alias="ANALYSIS_SCHEDULE_HOUR")
    allowed_origins: str = Field(
        default="http://localhost:5173,http://127.0.0.1:5173",
        alias="ALLOWED_ORIGINS",
    )

    model_config = SettingsConfigDict(populate_by_name=True)

    @property
    def origins(self) -> list[str]:
        return [origin.strip() for origin in self.allowed_origins.split(",") if origin.strip()]


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
