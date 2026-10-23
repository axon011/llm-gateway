"""Settings. Every value comes from the environment (.env in dev, compose env in Docker)."""

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # provider
    llm_provider: str = Field(default="groq", description="groq | gemini | fake")
    gemini_api_key: str = ""
    gemini_model: str = "gemini-2.5-flash"
    groq_api_key: str = ""
    groq_model: str = "openai/gpt-oss-20b"
    provider_timeout_s: float = 30.0

    # langfuse (self-hosted by default; leave keys empty to run without tracing)
    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""
    langfuse_host: str = "http://localhost:3000"

    # usage ledger (SQLite). ":memory:" for tests.
    usage_db_path: str = "data/usage.sqlite"

    # When set, every /v1 call needs `Authorization: Bearer <key>`. Required once the gateway is
    # reachable from outside localhost (the deployed tutor calls it over the internet).
    gateway_api_key: str = ""

    @property
    def tracing_enabled(self) -> bool:
        return bool(self.langfuse_public_key and self.langfuse_secret_key)


settings = Settings()
