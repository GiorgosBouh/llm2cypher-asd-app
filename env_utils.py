from __future__ import annotations

from pathlib import Path

from dotenv import find_dotenv, load_dotenv


def load_project_env() -> None:
    """Load environment variables from a nearby `.env` without overriding existing values."""
    dotenv_path = find_dotenv(usecwd=True)
    if dotenv_path:
        load_dotenv(dotenv_path=dotenv_path, override=False)
        return

    project_root = Path(__file__).resolve().parent
    fallback_path = project_root / ".env"
    if fallback_path.exists():
        load_dotenv(dotenv_path=fallback_path, override=False)
