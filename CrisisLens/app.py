from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
import sys

import gradio as gr
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from crisislens_space.io import load_entities
from crisislens_space.pipeline import (
    build_compare_outputs,
    build_json_report,
    build_world_map_outputs,
    run_inference_panel,
)
from crisislens_space.plotting import empty_figure

ENTITIES = load_entities(ROOT / "data" / "sample_entities.txt")


def _resolve_entity(dropdown_value: str | None, text_value: str | None) -> str:
    override = (text_value or "").strip()
    if override:
        return override
    if dropdown_value:
        return dropdown_value
    raise gr.Error("Select an entity or enter one manually.")


def run_dashboard(entity_choice: str | None, entity_override: str, start_date: str, end_date: str, horizon: int, fast_mode: bool):
    entity = _resolve_entity(entity_choice, entity_override)
    frame, metadata = run_inference_panel(
        entity=entity,
        start_date=start_date,
        end_date=end_date,
        horizon=horizon,
        fast_mode=fast_mode,
        root_dir=ROOT,
    )
    latest = frame.sort_values("date").iloc[-1]
    latest_shock = f"{latest['shock_prob']:.1%}"
    latest_severity = f"{latest['severity_pred']:.3f}"
    latest_intensity = f"{latest['intensity_pred']:.3f}"
    trend = "↑" if float(frame["shock_prob"].iloc[-1]) >= float(frame["shock_prob"].iloc[0]) else "↓"
    table = (
        frame.sort_values(["shock_prob", "severity_pred"], ascending=False)
        .loc[:, ["date", "shock_prob", "severity_pred", "attention_index", "risk_index"]]
        .head(10)
    )
    report_json = build_json_report(frame, entity, horizon, metadata)
    gr.Info(f"Mode: {metadata['mode']}")
    return (
        latest_shock,
        latest_severity,
        latest_intensity,
        trend,
        metadata["timeline_plot"],
        table,
        report_json,
    )


def compare_entities(entities: list[str], start_date: str, end_date: str, horizon: int, fast_mode: bool):
    if not entities:
        raise gr.Error("Choose at least one entity.")
    plot, table = build_compare_outputs(
        entities=entities[:5],
        start_date=start_date,
        end_date=end_date,
        horizon=horizon,
        fast_mode=fast_mode,
        root_dir=ROOT,
    )
    return plot, table


def world_map(start_date: str, end_date: str, horizon: int, fast_mode: bool):
    plot, table = build_world_map_outputs(
        start_date=start_date,
        end_date=end_date,
        horizon=horizon,
        fast_mode=fast_mode,
        root_dir=ROOT,
    )
    return plot, table


with gr.Blocks(title="CrisisLens — Narrative Shock Timeline") as demo:
    gr.Markdown("# CrisisLens — Narrative Shock Timeline")
    gr.Markdown(
        "An interactive dashboard for narrative shock monitoring. "
        "Select an entity, run the panel, and inspect probability, severity, and event intensity."
    )

    with gr.Tabs():
        with gr.Tab("Shock Timeline"):
            with gr.Row():
                entity_dropdown = gr.Dropdown(choices=ENTITIES, label="Entity / Country", value=ENTITIES[0] if ENTITIES else None)
                entity_override = gr.Textbox(label="Override entity (optional)", placeholder="Type a custom page/entity")
            with gr.Row():
                start_input = gr.Textbox(label="Start date", value="2025-01-01")
                end_input = gr.Textbox(label="End date", value="2025-02-15")
                horizon_input = gr.Dropdown(choices=[1, 3, 7], value=1, label="Horizon (days)")
                fast_mode = gr.Checkbox(label="Fast mode (cached)", value=True)
            run_button = gr.Button("Run", variant="primary")
            with gr.Row():
                latest_shock = gr.Textbox(label="Latest shock probability")
                latest_severity = gr.Textbox(label="Latest severity")
                latest_intensity = gr.Textbox(label="Latest intensity")
                trend_arrow = gr.Textbox(label="Trend")
            timeline_plot = gr.Plot(label="Shock timeline", value=empty_figure("Run a query to see results"))
            top_shock_days = gr.Dataframe(
                headers=["date", "shock_prob", "severity_pred", "attention_index", "risk_index"],
                label="Top shock days",
            )
            json_output = gr.File(label="JSON report")

            run_button.click(
                fn=run_dashboard,
                inputs=[entity_dropdown, entity_override, start_input, end_input, horizon_input, fast_mode],
                outputs=[latest_shock, latest_severity, latest_intensity, trend_arrow, timeline_plot, top_shock_days, json_output],
            )

        with gr.Tab("Compare Entities"):
            compare_select = gr.Dropdown(choices=ENTITIES, value=ENTITIES[:2], multiselect=True, label="Entities (max 5)")
            with gr.Row():
                compare_start = gr.Textbox(label="Start date", value="2025-01-01")
                compare_end = gr.Textbox(label="End date", value="2025-02-15")
                compare_horizon = gr.Dropdown(choices=[1, 3, 7], value=1, label="Horizon (days)")
                compare_fast = gr.Checkbox(label="Fast mode (cached)", value=True)
            compare_button = gr.Button("Compare")
            compare_plot = gr.Plot(label="Shock probability comparison", value=empty_figure("Select entities to compare"))
            compare_table = gr.Dataframe(
                headers=["entity", "max_prob", "mean_prob", "shock_days"],
                label="Entity comparison",
            )
            compare_button.click(
                fn=compare_entities,
                inputs=[compare_select, compare_start, compare_end, compare_horizon, compare_fast],
                outputs=[compare_plot, compare_table],
            )

        with gr.Tab("World Map"):
            with gr.Row():
                world_start = gr.Textbox(label="Start date", value="2025-01-01")
                world_end = gr.Textbox(label="End date", value="2025-02-15")
                world_horizon = gr.Dropdown(choices=[1, 3, 7], value=1, label="Horizon (days)")
                world_fast = gr.Checkbox(label="Fast mode (cached)", value=True)
            world_button = gr.Button("Render map")
            world_plot = gr.Plot(label="Latest risk by country", value=empty_figure("Render the world view"))
            world_table = gr.Dataframe(
                headers=["entity", "date", "shock_prob", "severity_pred"],
                label="Top 10 countries",
            )
            world_button.click(
                fn=world_map,
                inputs=[world_start, world_end, world_horizon, world_fast],
                outputs=[world_plot, world_table],
            )

        with gr.Tab("About / Methods"):
            gr.Markdown(
                """
### What narrative shocks are
Narrative shocks are abrupt jumps in public attention, event clustering, or risk perception.
This dashboard tracks those jumps over short horizons and turns them into a visible timeline.

### Data sources
- GDELT for event activity
- Wikimedia Pageviews for attention
- FRED for macro context

### Limitations
- This is a monitoring interface, not a final decision engine.
- Public data can be noisy, delayed, or biased.
- Demo mode uses bundled sample predictions when model artifacts are unavailable.

### Ethics
Attention does not equal truth.
Users should treat the output as a decision-support signal, not a definitive claim.
"""
            )

demo.queue(default_concurrency_limit=2)

if __name__ == "__main__":
    demo.launch()
