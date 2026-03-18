from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go



def build_forecast_figure(
    baseline: pd.DataFrame,
    scenario: pd.DataFrame,
    show_baseline: bool = True,
    show_scenario: bool = True,
) -> go.Figure:
    fig = go.Figure()

    if show_baseline and not baseline.empty:
        fig.add_trace(
            go.Scatter(
                x=baseline["date"],
                y=baseline["p90"],
                line=dict(width=0),
                showlegend=False,
                hoverinfo="skip",
                name="Baseline p90",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=baseline["date"],
                y=baseline["p10"],
                fill="tonexty",
                fillcolor="rgba(44, 123, 229, 0.18)",
                line=dict(width=0),
                name="Baseline Band (p10-p90)",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=baseline["date"],
                y=baseline["median"],
                mode="lines+markers",
                line=dict(color="#2C7BE5", width=3),
                marker=dict(size=8),
                name="Baseline Median",
            )
        )

    if show_scenario and not scenario.empty:
        fig.add_trace(
            go.Scatter(
                x=scenario["date"],
                y=scenario["p90"],
                line=dict(width=0),
                showlegend=False,
                hoverinfo="skip",
                name="Scenario p90",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=scenario["date"],
                y=scenario["p10"],
                fill="tonexty",
                fillcolor="rgba(0, 166, 153, 0.2)",
                line=dict(width=0),
                name="Scenario Band (p10-p90)",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=scenario["date"],
                y=scenario["median"],
                mode="lines+markers",
                line=dict(color="#00A699", width=3),
                marker=dict(size=8),
                name="Scenario Median",
            )
        )

    fig.update_layout(
        title="3-Day Energy Forecast: Baseline vs Scenario",
        xaxis_title="Date",
        yaxis_title="Energy Score",
        yaxis=dict(range=[0, 100]),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        template="plotly_white",
        margin=dict(l=30, r=30, t=70, b=30),
    )
    return fig



def build_importance_figure(importance_map: dict[str, float], top_n: int = 10) -> go.Figure:
    if not importance_map:
        return go.Figure()

    sorted_items = sorted(importance_map.items(), key=lambda item: item[1], reverse=True)[:top_n]
    features = [item[0] for item in sorted_items][::-1]
    scores = [item[1] for item in sorted_items][::-1]

    fig = go.Figure(
        go.Bar(
            x=scores,
            y=features,
            orientation="h",
            marker=dict(color="#EB5E28"),
        )
    )
    fig.update_layout(
        title="Top Forecast Drivers",
        xaxis_title="Importance",
        yaxis_title="Feature",
        template="plotly_white",
        margin=dict(l=30, r=30, t=60, b=30),
    )
    return fig
