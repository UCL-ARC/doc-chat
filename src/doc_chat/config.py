"""Configuration module for the application."""

from functools import lru_cache
import os
from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Application settings.

    Attributes:
        PROJECT_NAME: Name of the project.
        VERSION: Version of the project.
        API_V1_STR: API version string.
        SECRET_KEY: Secret key for JWT token generation.
        ACCESS_TOKEN_EXPIRE_MINUTES: Expiration time for access tokens.
        DATABASE_URL: URL for the PostgreSQL database.
        DB_ECHO_LOG: Whether to log database operations.
        OPENAI_API_KEY: API key for OpenAI.
        GOOGLE_API_KEY: API key for Google Gemini.
        AZURE_API_BASE: Base URL for Azure API.
        AZURE_API_KEY: API key for Azure.
        AZURE_API_DEPLOYMENT_NAME: Deployment name for Azure.
        AZURE_API_VERSION: Version of Azure API.
        OLLAMA_API_BASE_URL: Base URL for Ollama API.
        RAG_ENABLED: Whether RAG is enabled.
        RAG_EMBEDDING_MODEL: Embedding model for RAG.
        RAG_CHUNK_SIZE: Size of RAG chunks.
        RAG_CHUNK_OVERLAP: Overlap between RAG chunks.
        RAG_TOP_K: Top K for RAG.
        FAISS_INDEX_PATH: Path to FAISS index.
        LITELLM_LOG_LEVEL: Log level for LiteLLM.
        DISABLE_AUTH: Whether to disable authentication (for development/testing).

    """

    PROJECT_NAME: str = "Document Analysis API"
    VERSION: str = "1.0.0"
    API_V1_STR: str = "/api/v1"

    SECRET_KEY: str = os.getenv("SECRET_KEY", "your-secret-key-here")
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30

    DATABASE_URL: str = os.getenv("DATABASE_URL", "postgresql+asyncpg://user:password@localhost:5432/doc_chat")
    DB_ECHO_LOG: bool = False

    OPENAI_API_KEY: Optional[str] = os.getenv("OPENAI_API_KEY")
    GOOGLE_API_KEY: str = ""

    AZURE_API_BASE: str | None = None
    AZURE_API_KEY: str | None = None
    AZURE_API_DEPLOYMENT_NAME: str | None = None
    AZURE_API_VERSION: str | None = None

    # Ollama configuration - defaults to localhost for local development
    # In Docker, this should be set to "http://ollama:11434" via environment variable
    OLLAMA_API_BASE_URL: str = os.getenv("OLLAMA_API_BASE_URL", "http://localhost:11434")

    # RAG Configuration
    RAG_ENABLED: bool = os.getenv("RAG_ENABLED", "false").lower() == "true"
    # Use Ollama by default to avoid Hugging Face token; use "sentence-transformers/..." for HF models
    RAG_EMBEDDING_MODEL: str = os.getenv(
        "RAG_EMBEDDING_MODEL", "ollama/nomic-embed-text"
    )
    RAG_CHUNK_SIZE: int = int(os.getenv("RAG_CHUNK_SIZE", "512"))
    RAG_CHUNK_OVERLAP: int = int(os.getenv("RAG_CHUNK_OVERLAP", "50"))
    RAG_TOP_K: int = int(os.getenv("RAG_TOP_K", "5"))
    FAISS_INDEX_PATH: str = os.getenv("FAISS_INDEX_PATH", "./data/faiss_index")

    LITELLM_LOG_LEVEL: str = os.getenv("LITELLM_LOG_LEVEL", "ERROR")

    # Authentication Configuration (default: disabled for easier local/dev use)
    DISABLE_AUTH: bool = os.getenv("DISABLE_AUTH", "true").lower() == "true"

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", case_sensitive=True, extra="allow"
    )


@lru_cache
def get_settings() -> Settings:
    """
    Get cached settings instance.

    Returns:
        Settings: Application settings instance.

    """
    return Settings()


settings = get_settings()
