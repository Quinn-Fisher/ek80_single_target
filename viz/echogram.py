"""Plotly echogram with detection overlays."""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go


def plot_echogram(ds_Sv, detections_df, ch, title="Echogram"):
    sv_da = ds_Sv["Sv"].sel(channel=ch)
    sv_values = sv_da.values
    ping_time = sv_da["ping_time"].values
    y_title = "Range (m)"
    if "echo_range" in ds_Sv.coords:
        range_arr = ds_Sv["echo_range"].sel(channel=ch).values
    elif "echo_range" in ds_Sv.variables:
        range_arr = ds_Sv["echo_range"].sel(channel=ch).values
    else:
        range_arr = sv_da["range_sample"].values
        y_title = "Range Sample Index"

    # Plotly heatmap y-axis expects 1D. For per-ping echo_range, use the
    # representative median profile across pings.
    if np.ndim(range_arr) == 1:
        range_y = range_arr
    else:
        range_y = np.nanmedian(range_arr, axis=0)

    fig = go.Figure()
    fig.add_trace(
        go.Heatmap(
            x=ping_time,
            y=range_y,
            z=sv_values.T,
            colorscale="Viridis",
            reversescale=True,
            zmin=-80,
            zmax=-20,
            colorbar={"title": "Sv (dB re 1 m⁻¹)"},
            name="Sv",
        )
    )

    n_det = 0
    if detections_df is not None and not detections_df.empty:
        n_det = len(detections_df)
        hover = (
            "TScomp: "
            + detections_df["ts_compensated_db"].round(2).astype(str)
            + " dB<br>Along: "
            + detections_df["angle_alongship_deg"].round(2).astype(str)
            + "°<br>Athwart: "
            + detections_df["angle_athwartship_deg"].round(2).astype(str)
            + "°"
        )
        fig.add_trace(
            go.Scatter(
                x=detections_df["ping_time"].values,
                y=detections_df["range_m"].values,
                mode="markers",
                marker={
                    "size": 6,
                    "color": "white",
                    "line": {"color": "black", "width": 1},
                },
                name="Single targets",
                hovertext=hover,
                hoverinfo="text",
            )
        )

    fig.update_layout(
        template="plotly_dark",
        height=600,
        title=f"{title} | Detections: {n_det}",
        xaxis={"title": "Ping Time", "tickformat": "%H:%M:%S"},
        yaxis={"title": y_title, "autorange": "reversed"},
        margin={"l": 60, "r": 30, "t": 50, "b": 50},
    )
    return fig

