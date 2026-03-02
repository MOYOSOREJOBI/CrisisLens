from __future__ import annotations

import os
import time

import pandas as pd
import requests


FRED_URL = "https://api.stlouisfed.org/fred/series/observations"


def fetch_fred_series(series_id: str, start: str, end: str, retries: int = 3) -> pd.DataFrame | None:
    api_key = os.getenv("FRED_API_KEY")
    if not api_key:
        return None
    params = {
        "series_id": series_id,
        "api_key": api_key,
        "file_type": "json",
        "observation_start": start,
        "observation_end": end,
    }
    delay = 1.0
    for _ in range(retries):
        try:
            response = requests.get(FRED_URL, params=params, timeout=30)
            response.raise_for_status()
            payload = response.json()
            rows = []
            for item in payload.get("observations", []):
                try:
                    value = float(item["value"])
                except ValueError:
                    continue
                rows.append({"date": item["date"], series_id: value})
            return pd.DataFrame(rows)
        except Exception:
            time.sleep(delay)
            delay *= 2
    return None
