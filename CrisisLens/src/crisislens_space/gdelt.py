from __future__ import annotations

import time
from pathlib import Path

import requests


GDELT_DOC = "https://api.gdeltproject.org/api/v2/doc/doc"


def fetch_gdelt_timeline(query: str, start_date: str, end_date: str, retries: int = 3) -> dict | None:
    params = {
        "query": query,
        "mode": "timelinevolraw",
        "format": "json",
        "startdatetime": start_date.replace("-", "") + "000000",
        "enddatetime": end_date.replace("-", "") + "235959",
    }
    delay = 1.0
    for _ in range(retries):
        try:
            response = requests.get(GDELT_DOC, params=params, timeout=30)
            response.raise_for_status()
            return response.json()
        except Exception:
            time.sleep(delay)
            delay *= 2
    return None
