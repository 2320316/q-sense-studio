import hashlib
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(os.environ.get("QSENSE_HOME", Path(__file__).resolve().parents[1] / ".local")).resolve()


def now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False, encoding="utf8") as f:
        json.dump(value, f, allow_nan=False)
    os.replace(f.name, path)


def read(path):
    return json.loads(path.read_text(encoding="utf8"))


def folder(kind, key):
    if not key.isalnum() or len(key) > 64:
        raise ValueError("Invalid identifier")
    path = ROOT / kind / key
    if not path.is_dir():
        raise ValueError("Record does not exist")
    return path


def digest(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def catalog(kind, filename):
    paths = sorted((ROOT / kind).glob("*/" + filename), key=lambda p: p.stat().st_mtime, reverse=True)
    return [read(p) for p in paths]
