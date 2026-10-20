"""Small shared utilities; artifacts are explicit, JSON, and never overwritten."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
ARCHIVE = ROOT.parent


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def file_digest(path: str | Path) -> str:
    return digest(Path(path).read_bytes())


def json_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n").encode()


def write_json(path: str | Path, value: Any) -> Path:
    path = Path(path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(json_bytes(value))
    return path


def load_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(), parse_constant=lambda v: (_ for _ in ()).throw(ValueError(f"nonfinite JSON: {v}")))


def check_hash(path: str | Path, expected: str) -> None:
    if file_digest(path) != expected:
        raise ValueError(f"Frozen artifact changed: {path}")
