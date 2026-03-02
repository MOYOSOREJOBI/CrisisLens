# NarrativePulse — Narrative Shock Early Warning (GDELT x Wikipedia x Macro)

[![CI](https://img.shields.io/badge/CI-placeholder-lightgrey)](#)
[![HuggingFace](https://img.shields.io/badge/HF-placeholder-lightgrey)](#)

**NarrativePulse** is a research-grade machine learning system that detects
**narrative shocks** and forecasts their likely near-term impact across the
next 1 to 7 days.

It fuses:
- **GDELT Events** for event intensity and conflict structure.
- **Wikipedia Pageviews** for public attention.
- **FRED macro indicators** for regime context.

It trains an ensemble of **7 components**:
- shock probability classification
- shock severity regression
- event intensity forecasting
- calibrated uncertainty and conformal intervals

Built to be transparent, modular, and reproducible.

## What it is
This repo builds a daily panel for entities and countries.
It turns open event data, attention data, and macro context into a
leakage-safe forecasting pipeline.
The output is an early warning layer for emerging narrative stress.

## Why it matters
Narratives move before consensus catches up.
Attention spikes, event clustering, and regime shifts often appear before
institutional response.
This project is useful for:
- risk operations
- OSINT monitoring
- corporate reputation monitoring
- policy and safety analysis
- applied research on attention dynamics

## The 7-component system
1. **Text Encoder (DeBERTa)**  
   Produces dense document embeddings and weak semantic supervision signals.
2. **Weak Supervision Label Model**  
   Converts heuristic spike rules into probabilistic pseudo-labels.
3. **GraphSAGE**  
   Learns entity and country relationships from co-mention graphs.
4. **Hawkes Process**  
   Models self-exciting event arrivals and intensity anomalies.
5. **Temporal Fusion Transformer-style Forecaster**  
   Produces multi-horizon shock forecasts.
6. **Stacking Meta-Ensemble**  
   Combines model outputs into final shock probability and severity.
7. **Calibration + Conformal Prediction**  
   Improves probability reliability and interval coverage.

Each component exists because no single signal is enough.
Text captures meaning.
Graphs capture linkage.
Point processes capture clustering.
Temporal models capture trajectory.
Calibration makes the output usable.

## Data sources and ethics
- **GDELT** provides large-scale open event records.
- **Wikimedia Pageviews API** provides public attention proxies.
- **FRED** provides macro and risk context.

These are open public sources, but they still require care.
Attention is not truth.
Event counts can reflect source bias.
Macro series are slow-moving and can mask local effects.
This system is designed for monitoring and research, not for automated action.

## Install
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Quick demo (small config)
```bash
python scripts/download_gdelt.py --config configs/small.yaml
python scripts/download_wikipedia_pageviews.py --config configs/small.yaml
python scripts/download_fred.py --config configs/small.yaml
python scripts/build_dataset.py --config configs/small.yaml
python -m narrativepulse.train.train_text --config configs/small.yaml
python -m narrativepulse.train.train_graph --config configs/small.yaml
python -m narrativepulse.train.train_hawkes --config configs/small.yaml
python -m narrativepulse.train.train_tft --config configs/small.yaml
python -m narrativepulse.train.train_stacker --config configs/small.yaml
```

This config is designed to run on CPU on a short date range.

## Full pipeline (base config)
```bash
python scripts/download_gdelt.py --config configs/base.yaml
python scripts/download_wikipedia_pageviews.py --config configs/base.yaml
python scripts/download_fred.py --config configs/base.yaml
python scripts/build_dataset.py --config configs/base.yaml
python scripts/make_graph.py --config configs/base.yaml
python -m narrativepulse.train.train_text --config configs/base.yaml
python -m narrativepulse.train.train_graph --config configs/base.yaml
python -m narrativepulse.train.train_hawkes --config configs/base.yaml
python -m narrativepulse.train.train_tft --config configs/base.yaml
python -m narrativepulse.train.train_stacker --config configs/base.yaml
```

## Inference
Use the final stacker to score new entity-date rows.

```bash
python -m narrativepulse.inference.predict \
  --config configs/small.yaml \
  --entity Ukraine \
  --as_of_date 2025-01-10
```

For a lightweight live-style loop:
```bash
python -m narrativepulse.inference.realtime_demo --config configs/small.yaml
```

## Limitations and failure modes
- Weak labels are heuristic.
- Wikipedia attention is noisy and can be event-driven without true risk.
- GDELT can contain source duplication and uneven coverage.
- The Hawkes block is intentionally simple for reproducibility.
- The small config is a demo path, not a production benchmark.

## Roadmap
- Add stronger document aggregation for the text encoder.
- Add richer edge construction for the co-mention graph.
- Add better calibration diagnostics and drift monitoring.
- Add optional auxiliary pretraining tasks.

## Author
Moyosore Ogunjobi

Made by Moyosore Ogunjobi.
