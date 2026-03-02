from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def empty_figure(title: str) -> go.Figure:
    fig = go.Figure()
    fig.update_layout(title=title, template="plotly_white")
    return fig


def timeline_figure(frame: pd.DataFrame, entity: str) -> go.Figure:
    melted = frame.melt(
        id_vars=["date"],
        value_vars=["shock_prob", "attention_index", "risk_index", "intensity_pred"],
        var_name="series",
        value_name="value",
    )
    fig = px.line(
        melted,
        x="date",
        y="value",
        color="series",
        title=f"{entity} — shock timeline",
    )
    fig.update_layout(template="plotly_white", legend_title_text="")
    return fig


def compare_figure(frame: pd.DataFrame) -> go.Figure:
    fig = px.line(frame, x="date", y="shock_prob", color="entity", title="Shock probability comparison")
    fig.update_layout(template="plotly_white", legend_title_text="")
    return fig


def world_map_figure(frame: pd.DataFrame) -> go.Figure:
    latest = frame.sort_values("date").groupby("entity").tail(1)
    fig = px.choropleth(
        latest,
        locations="entity",
        locationmode="country names",
        color="shock_prob",
        hover_data=["severity_pred", "date"],
        title="Latest shock probability by country",
        color_continuous_scale="Reds",
    )
    fig.update_layout(template="plotly_white")
    return fig
