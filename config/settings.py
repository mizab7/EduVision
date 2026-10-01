from functools import lru_cache
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment variables or .env file."""
    DATABASE_URL: str = "sqlite:///./eduvision.db"
    SECRET_KEY: str = "default_secret_key_for_dev_only"
    GEMINI_API_KEY: str | None = None
    CAMERA_INDEX: int = 0
    FACE_RECOGNITION_THRESHOLD: float = 0.6
    ENGAGEMENT_DROP_THRESHOLD: float = 40.0
    FPS_TARGET: int = 15

    class Config:
        env_file = ".env"
        extra = "ignore"


@lru_cache()
def get_settings() -> Settings:
    """Returns a cached instance of the settings."""
    return Settings()
