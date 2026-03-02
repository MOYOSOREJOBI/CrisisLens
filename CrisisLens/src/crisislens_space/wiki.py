from __future__ import annotations

import time

import pandas as pd
import requests


BASE = "https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article"


def fetch_pageviews(article: str, start: str, end: str, retries: int = 3) -> pd.DataFrame | None:
    url = f"{BASE}/en.wikipedia/all-access/user/{article}/daily/{start.replace('-', '')}/{end.replace('-', '')}"
    delay = 1.0
    for _ in range(retries):
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            payload = response.json()
            rows = [
                {
                    "date": item["timestamp"][:4] + "-" + item["timestamp"][4:6] + "-" + item["timestamp"][6:8],
                    "pageviews": int(item["views"]),
                }
                for item in payload.get("items", [])
            ]
            return pd.DataFrame(rows)
        except Exception:
            time.sleep(delay)
            delay *= 2
    return None
