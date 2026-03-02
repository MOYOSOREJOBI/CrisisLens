from __future__ import annotations

from datetime import datetime, timedelta
from typing import Iterable


def parse_date(value: str) -> datetime:
    return datetime.strptime(value, "%Y-%m-%d")


def format_yyyymmdd(value: datetime) -> str:
    return value.strftime("%Y%m%d")


def date_range(start: str, end: str) -> list[str]:
    current = parse_date(start)
    stop = parse_date(end)
    out: list[str] = []
    while current <= stop:
        out.append(current.strftime("%Y-%m-%d"))
        current += timedelta(days=1)
    return out


def ensure_sorted_unique(values: Iterable[str]) -> list[str]:
    return sorted(set(values))
