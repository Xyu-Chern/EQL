"""Experiment JSON records and source fingerprints."""
import hashlib
import json
import subprocess
from functools import lru_cache
from pathlib import Path
from tempfile import NamedTemporaryFile

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def save(path, record):
    data = json.dumps(record, indent=2, allow_nan=False)
    stream = NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, suffix=".tmp", delete=False)
    temporary = Path(stream.name)
    try:
        with stream:
            stream.write(data)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def source_digest(name):
    return hashlib.sha256((ROOT / name).read_bytes()).hexdigest()


def verify_source(name, expected):
    if "/" in name:
        if source_digest(name) != expected:
            raise ValueError(f"Source checksum mismatch for {name}")
        return
    verify_archived_source(name, expected)


@lru_cache
def verify_archived_source(name, expected):
    # The pre-refactor protocols used flat filenames. The original baseline
    # predates the exact zero endpoint, so it needs the older eql.py blob.
    for revision in ("520815f", "bf47903", "4bffa56"):
        result = subprocess.run(["git", "show", f"{revision}:{name}"], cwd=ROOT, capture_output=True)
        if result.returncode == 0:
            lf = result.stdout.replace(b"\r\n", b"\n")
            for version in (lf, lf.replace(b"\n", b"\r\n")):
                if hashlib.sha256(version).hexdigest() == expected:
                    return
    raise ValueError(f"Cannot verify archived source {name} with SHA-256 {expected}")
