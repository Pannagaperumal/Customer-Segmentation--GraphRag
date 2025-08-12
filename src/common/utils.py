import json
import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

# PROJECT_ROOT should be the repository root (one level above `src/`)
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Load environment variables from a local .env file if present
load_dotenv(PROJECT_ROOT / ".env")


def ensure_directory(path: Path) -> None:
    if not path.exists():
        path.mkdir(parents=True, exist_ok=True)


def data_dir(*parts: str) -> Path:
    base = PROJECT_ROOT / "data"
    ensure_directory(base)
    for part in parts[:-1]:
        base = base / part
        ensure_directory(base)
    return base / parts[-1] if parts else base


def examples_dir(*parts: str) -> Path:
    base = PROJECT_ROOT / "examples"
    ensure_directory(base)
    for part in parts[:-1]:
        base = base / part
        ensure_directory(base)
    return base / parts[-1] if parts else base


def write_json(path: Path, data: Any) -> None:
    ensure_directory(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def get_env(name: str, default: str | None = None) -> str | None:
    return os.environ.get(name, default)


