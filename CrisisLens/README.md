---
title: CrisisLens — Shock Timeline
emoji: 🌍
colorFrom: indigo
colorTo: purple
sdk: gradio
app_file: app.py
pinned: false
---

# CrisisLens — Narrative Shock Timeline

This Space is an interactive dashboard for **CrisisLens**.
It turns open event, attention, and macro signals into a visible timeline of
potential narrative shocks.

Users can choose an entity or country, set a date range, and inspect:
- shock probability
- shock severity
- event intensity
- attention and risk indices
- the highest-risk days in the selected window

The app supports two operating modes:
- **Hub mode** loads artifacts from a model repository on Hugging Face.
- **Demo mode** falls back to a bundled sample parquet file when artifacts are
  not available.

## How it works

The dashboard tries to load cached artifacts from a Hugging Face model repo by
using `huggingface_hub.snapshot_download`.
Set `MODEL_REPO_ID` to point to the model repo.
If the repo is missing or incomplete, the app switches to demo mode and uses
`data/demo_predictions.parquet`.

If you choose **Live mode**, the app will also try to fetch fresh public data:
- GDELT for event timelines
- Wikimedia Pageviews for attention
- FRED for macro context

Live mode is best-effort.
If those downloads fail, the UI still renders from cached or demo data.

## Data sources
- GDELT (global events): https://www.gdeltproject.org/
- Wikimedia Pageviews API: https://doc.wikimedia.org/generated-data-platform/aqs/analytics-api/reference/page-views.html
- FRED API: https://fred.stlouisfed.org/docs/api/fred/

## How to run locally

```bash
cd /Users/mac/Desktop/CrisisLens
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export FRED_API_KEY="YOUR_KEY"
export MODEL_REPO_ID="m0yosore/crisislens"
python app.py
```

## Secrets and configuration

The Space reads configuration from environment variables.

- `FRED_API_KEY`
  Used for optional FRED macro downloads in live mode.
- `MODEL_REPO_ID`
  Defaults to `m0yosore/crisislens`.
  This is the model repo used by `snapshot_download`.

The cache path is chosen automatically:
- `/data/hf_cache` if the Space runtime exposes `/data`
- `./cache` otherwise

## What the dashboard shows

### Shock Timeline
- Daily shock probability
- AttentionShockIndex and RiskShockIndex
- Hawkes-style event intensity
- Top shock days table
- Downloadable JSON summary

### Compare Entities
- Overlay of shock probability across up to 5 entities
- Side-by-side summary table

### World Map
- Latest risk by country for the selected date range
- Top 10 countries ranked by latest shock probability

### About / Methods
- Plain-language description of narrative shocks
- Data provenance
- Limitations and ethical constraints

## Limitations

- This Space is a dashboard, not a trading or policy engine.
- Public attention is noisy and can overreact to headlines.
- GDELT coverage varies by source density and region.
- Demo mode uses synthetic sample outputs for reliability.
- Live mode is network-dependent and can be rate-limited.

## Roadmap

- Add richer artifact loading from the model repo
- Add better map drill-down for entity-to-country aggregation
- Add saved comparison presets
- Add stronger report exports

## Author

Moyosore Ogunjobi
