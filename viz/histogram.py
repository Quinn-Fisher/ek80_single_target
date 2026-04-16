"""Plotly target-strength histogram."""

from __future__ import annotations

import plotly.graph_objects as go


def plot_ts_histogram(detections_df, ts_min_db):
    fig = go.Figure()
    if detections_df is not None and not detections_df.empty:
        fig.add_trace(
            go.Histogram(
                x=detections_df["ts_compensated_db"].values,
                xbins={"size": 1},
                marker={"color": "#2ca02c"},
                name="TS distribution",
            )
        )

    fig.add_vline(
        x=ts_min_db,
        line_dash="dash",
        line_color="white",
        annotation_text="TSmin",
        annotation_position="top right",
    )
    fig.update_layout(
        template="plotly_dark",
        height=350,
        xaxis={"title": "Target Strength (dB re 1 m²)", "range": [ts_min_db, ts_min_db + 40]},
        yaxis={"title": "Count"},
        bargap=0.05,
    )
    return fig

