"""Configuration module for the application."""

from functools import lru_cache

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

    """

    PROJECT_NAME: str = "Document Analysis API"
    VERSION: str = "1.0.0"
    API_V1_STR: str = "/api/v1"

    SECRET_KEY: str = "your-secret-key-here"  # Change in production
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 24 * 8  # 8 days

    DATABASE_URL: str = "postgresql+asyncpg://user:password@localhost/dbname"
    DB_ECHO_LOG: bool = False

    OPENAI_API_KEY: str = ""
    GOOGLE_API_KEY: str = ""

    AZURE_API_BASE: str | None = None
    AZURE_API_KEY: str | None = None
    AZURE_API_DEPLOYMENT_NAME: str | None = None
    AZURE_API_VERSION: str | None = None

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
