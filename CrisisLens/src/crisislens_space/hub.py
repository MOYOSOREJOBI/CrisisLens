from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from huggingface_hub import snapshot_download


def get_cache_dir() -> Path:
    data_dir = Path("/data")
    if data_dir.exists():
        cache_dir = data_dir / "hf_cache"
    else:
        cache_dir = Path("./cache")
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


@lru_cache(maxsize=1)
def fetch_model_snapshot() -> Path | None:
    repo_id = os.getenv("MODEL_REPO_ID", "m0yosore/crisislens")
    try:
        path = snapshot_download(repo_id=repo_id, cache_dir=str(get_cache_dir()))
        return Path(path)
    except Exception:
        return None
