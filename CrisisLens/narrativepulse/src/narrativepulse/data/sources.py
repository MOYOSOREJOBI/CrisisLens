from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import requests

from narrativepulse.utils.dates import date_range, format_yyyymmdd, parse_date
from narrativepulse.utils.io import ensure_dir, write_json


WIKI_BASE = "https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article"
FRED_BASE = "https://api.stlouisfed.org/fred/series/observations"
GDELT_DOC = "https://api.gdeltproject.org/api/v2/doc/doc"


def _safe_get(url: str, params: dict[str, str] | None = None, timeout: int = 30) -> dict:
    response = requests.get(url, params=params, timeout=timeout)
    response.raise_for_status()
    return response.json()


def fetch_wikipedia_pageviews(article: str, start_date: str, end_date: str) -> pd.DataFrame:
    start = format_yyyymmdd(parse_date(start_date))
    end = format_yyyymmdd(parse_date(end_date))
    url = f"{WIKI_BASE}/en.wikipedia/all-access/user/{article}/daily/{start}/{end}"
    payload = _safe_get(url, timeout=45)
    rows = [
        {
            "entity": article.replace("_", " "),
            "date": item["timestamp"][:4] + "-" + item["timestamp"][4:6] + "-" + item["timestamp"][6:8],
            "wikipedia_pageviews": int(item["views"]),
        }
        for item in payload.get("items", [])
    ]
    return pd.DataFrame(rows)


def fetch_fred_series(series_id: str, api_key: str, start_date: str, end_date: str) -> pd.DataFrame:
    params = {
        "series_id": series_id,
        "api_key": api_key,
        "file_type": "json",
        "observation_start": start_date,
        "observation_end": end_date,
    }
    payload = _safe_get(FRED_BASE, params=params, timeout=45)
    rows = []
    for item in payload.get("observations", []):
        try:
            value = float(item["value"])
        except ValueError:
            value = np.nan
        rows.append({"date": item["date"], f"macro_{series_id}": value})
    return pd.DataFrame(rows)


def fetch_gdelt_doc_timeline(query: str, start_date: str, end_date: str) -> dict:
    params = {
        "query": query,
        "mode": "timelinevolraw",
        "format": "json",
        "startdatetime": start_date.replace("-", "") + "000000",
        "enddatetime": end_date.replace("-", "") + "235959",
    }
    return _safe_get(GDELT_DOC, params=params, timeout=60)


def synthetic_gdelt_frame(entity: str, dates: Iterable[str]) -> pd.DataFrame:
    date_list = list(dates)
    idx = np.arange(len(date_list))
    base = 10 + 3 * np.sin(idx / 5.0) + (idx % 17 == 0) * 12
    return pd.DataFrame(
        {
            "entity": entity,
            "date": date_list,
            "gdelt_event_count_total": np.maximum(base.astype(int), 0),
            "gdelt_event_count_by_quadclass_1": np.maximum((base * 0.35).astype(int), 0),
            "gdelt_event_count_by_quadclass_2": np.maximum((base * 0.25).astype(int), 0),
            "gdelt_event_count_by_quadclass_3": np.maximum((base * 0.20).astype(int), 0),
            "gdelt_event_count_by_quadclass_4": np.maximum((base * 0.20).astype(int), 0),
            "gdelt_goldstein_mean": -1.5 + 0.5 * np.cos(idx / 7.0),
            "gdelt_goldstein_std": 0.8 + 0.1 * np.sin(idx / 11.0),
            "gdelt_goldstein_min": -4.0 - 0.5 * (idx % 9 == 0),
        }
    )


def synthetic_wiki_frame(entity: str, dates: Iterable[str]) -> pd.DataFrame:
    date_list = list(dates)
    idx = np.arange(len(date_list))
    views = 1000 + 100 * np.sin(idx / 6.0) + (idx % 23 == 0) * 600
    return pd.DataFrame({"entity": entity, "date": date_list, "wikipedia_pageviews": views.astype(int)})


def synthetic_fred_frame(series_ids: list[str], dates: Iterable[str]) -> pd.DataFrame:
    date_list = list(dates)
    idx = np.arange(len(date_list))
    frame = pd.DataFrame({"date": date_list})
    for i, series_id in enumerate(series_ids):
        frame[f"macro_{series_id}"] = (i + 1) * 0.5 + np.sin(idx / (4 + i))
    return frame


def cache_json(payload: dict, path: str | Path) -> None:
    ensure_dir(Path(path).parent)
    write_json(path, payload)


def fallback_dates(start_date: str, end_date: str) -> list[str]:
    return date_range(start_date, end_date)
