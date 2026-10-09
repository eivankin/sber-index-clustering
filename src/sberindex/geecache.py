"""A cache for Earth Engine responses, kept outside the repository.

The noncommercial quota is finite and one of the two keys we have is already throttled into restricted
mode, so re-asking Earth Engine for something already fetched is a cost that has to be avoided on purpose.
Reproducibility checks are also cheaper this way: re-deriving an artifact from a stored response shows that
the derivation is stable, which is a different and more useful claim than "the download worked twice".

Responses are stored under a directory taken from SBER_GEE_CACHE, defaulting outside the repository tree,
on a filesystem that survives a reboot. /tmp is not used: it is a 16 G tmpfs wiped on restart, and a cache
that disappears with it would quietly turn every re-run into a fresh download.

The key covers everything that could change the answer - the job specification, the simplification
tolerance, and the territory ids in the chunk - so a different polygon, a different scale or a different
set of districts is a different key rather than a stale hit.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import os
from pathlib import Path

DEFAULT_DIR = "gee-cache"  # resolved under the user's home, never hard-coded to one machine


def cache_dir() -> Path:
    raw = os.environ.get("SBER_GEE_CACHE", DEFAULT_DIR)
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = Path.home() / path
    if str(path).startswith("/tmp"):
        raise RuntimeError(
            f"the Earth Engine cache must survive a reboot; {path} is a tmpfs that does not"
        )
    path.mkdir(parents=True, exist_ok=True)
    return path


def key(material: object) -> str:
    encoded = json.dumps(material, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def cached(key_value: str, produce):
    """Return a stored response, or run the producer and store what it returns.

    The second value of the pair says where the payload came from, so a report can distinguish a replay
    from a fresh Earth Engine call instead of counting them as the same work.
    """
    path = cache_dir() / f"{key_value}.json.gz"
    if path.exists():
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            return json.load(handle), "cache"
    payload = produce()
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False)
    return payload, "earth_engine"


def stats() -> dict:
    """How much is stored, so a quota decision can be made from numbers."""
    directory = cache_dir()
    files = list(directory.glob("*.json.gz"))
    return {
        "responses": len(files),
        "bytes": sum(f.stat().st_size for f in files),
        "directory": str(directory),
        "inside_the_repository": bool(
            Path.cwd() in [p.parent for p in files] or str(directory).startswith(str(Path.cwd()))
        ),
    }
