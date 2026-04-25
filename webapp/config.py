import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent


class Settings:
    SECRET_KEY: str = os.environ.get("SECRET_KEY", "dev-secret-change-me-in-production")
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    REFRESH_TOKEN_EXPIRE_DAYS: int = 7

    STRIPE_SECRET_KEY: str = os.environ.get("STRIPE_SECRET_KEY", "")
    STRIPE_WEBHOOK_SECRET: str = os.environ.get("STRIPE_WEBHOOK_SECRET", "")
    STRIPE_PRICE_ID: str = os.environ.get("STRIPE_PRICE_ID", "")

    DATABASE_URL: str = os.environ.get("DATABASE_URL", f"sqlite:///{BASE_DIR / 'data' / 'noion.db'}")
    UPLOAD_DIR: str = os.environ.get("UPLOAD_DIR", str(BASE_DIR / "data" / "uploads"))

    MAX_WORKERS: int = int(os.environ.get("MAX_WORKERS", "2"))
    MAX_FREE_DURATION_S: int = int(os.environ.get("MAX_FREE_DURATION_S", "30"))
    MAX_UPLOAD_SIZE_MB: int = int(os.environ.get("MAX_UPLOAD_SIZE_MB", "50"))

    ALLOWED_EXTENSIONS: set = {".mp3", ".wav", ".flac", ".ogg", ".m4a", ".aac"}


settings = Settings()
