from __future__ import annotations

import pandas as pd

from narrativepulse.data.dataset_builder import build_daily_panel


def test_alignment_and_targets_exist() -> None:
    dates = pd.date_range("2024-01-01", periods=10, freq="D").strftime("%Y-%m-%d")
    gdelt = pd.DataFrame(
        {
            "entity": ["Ukraine"] * 10,
            "date": dates,
            "gdelt_event_count_total": list(range(10, 20)),
            "gdelt_event_count_by_quadclass_1": [1] * 10,
            "gdelt_event_count_by_quadclass_2": [2] * 10,
            "gdelt_event_count_by_quadclass_3": [3] * 10,
            "gdelt_event_count_by_quadclass_4": [4] * 10,
            "gdelt_goldstein_mean": [-1.0] * 10,
            "gdelt_goldstein_std": [0.5] * 10,
            "gdelt_goldstein_min": [-2.0] * 10,
        }
    )
    wiki = pd.DataFrame({"entity": ["Ukraine"] * 10, "date": dates, "wikipedia_pageviews": list(range(100, 110))})
    fred = pd.DataFrame({"date": dates, "macro_VIXCLS": [20.0] * 10})
    result = build_daily_panel(gdelt, wiki, fred, shock_quantile=0.99, rolling_window=5)
    assert "shock_label" in result.panel.columns
    assert "severity_t_plus_1" in result.panel.columns
    assert "event_intensity_t_plus_1" in result.panel.columns
