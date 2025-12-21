from __future__ import annotations

import os
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()


@dataclass
class Settings:
    groq_api_key: str = os.environ.get("GROQ_API_KEY", "")
    data_dir: str = os.environ.get("DATA_DIR", "data")
    persist_dir: str = os.environ.get("PERSIST_DIR", "doc_db")
    model: str = os.environ.get("MODEL", "llama-3.1-8b-instant")
    temperature: float = float(os.environ.get("TEMPERATURE", "0"))


settings = Settings()


def require_groq_key() -> None:
    if not settings.groq_api_key:
        raise EnvironmentError(
            "GROQ_API_KEY is not set. Set it in environment or .env file."
        )
