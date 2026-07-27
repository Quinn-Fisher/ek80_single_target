"""Plotly echogram with detection overlays."""

from __future__ import annotations

import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from detection.tracker import UNASSIGNED_TRACK_ID

# Qualitative, high-contrast palette for per-track coloring. Dark24 reads
# well against both the plotly_dark template and the reversed-Viridis
# heatmap background. Indexed by `track_id % len(_TRACK_COLORS)` so a given
# track_id always gets the same color across re-renders (not reassigned
# randomly), at the cost of eventual color reuse once track_id exceeds the
# palette length.
_TRACK_COLORS = px.colors.qualitative.Dark24

# With many tracks, giving every single one its own always-visible legend
# entry makes the legend unusably long. Only the first this-many tracks (in
# track_id order) get an individual legend entry; the rest are still drawn
# (colored, connected) but omitted from the legend to keep it usable.
_MAX_LEGEND_TRACKS = 20


def _track_color(track_id: int) -> str:
    return _TRACK_COLORS[int(track_id) % len(_TRACK_COLORS)]


def plot_echogram(dataset, detections_df, ch, value_var="Sv", title="Echogram"):
    da = dataset[value_var].sel(channel=ch)
    values = da.values
    ping_time = da["ping_time"].values
    y_title = "Range (m)"
    if "echo_range" in dataset.coords:
        range_arr = dataset["echo_range"].sel(channel=ch).values
    elif "echo_range" in dataset.variables:
        range_arr = dataset["echo_range"].sel(channel=ch).values
    else:
        range_arr = da["range_sample"].values
        y_title = "Range Sample Index"

    # Plotly heatmap y-axis expects 1D. For per-ping echo_range, use the
    # representative median profile across pings.
    if np.ndim(range_arr) == 1:
        range_y = range_arr
    else:
        range_y = np.nanmedian(range_arr, axis=0)

    if value_var == "TS":
        colorbar_title = "TS (dB re 1 m²)"
        trace_name = "TS"
    else:
        colorbar_title = "Sv (dB re 1 m⁻¹)"
        trace_name = "Sv"

    fig = go.Figure()
    fig.add_trace(
        go.Heatmap(
            x=ping_time,
            y=range_y,
            z=values.T,
            colorscale="Viridis",
            reversescale=True,
            zmin=-80,
            zmax=-20,
            colorbar={"title": colorbar_title},
            name=trace_name,
        )
    )

    n_det = 0
    has_tracks = detections_df is not None and "track_id" in detections_df.columns
    if detections_df is not None and not detections_df.empty:
        n_det = len(detections_df)

        if not has_tracks:
            _add_plain_detections_trace(fig, detections_df)
        else:
            _add_tracked_detections_traces(fig, detections_df)

    fig.update_layout(
        template="plotly_dark",
        height=600,
        title=f"{title} | Detections: {n_det}",
        xaxis={"title": "Ping Time", "tickformat": "%H:%M:%S"},
        yaxis={"title": y_title, "autorange": "reversed"},
        margin={"l": 60, "r": 30, "t": 50, "b": 50},
    )
    return fig


def _hover_text(df):
    return (
        "TScomp: "
        + df["ts_compensated_db"].round(2).astype(str)
        + " dB<br>Along: "
        + df["angle_alongship_deg"].round(2).astype(str)
        + "°<br>Athwart: "
        + df["angle_athwartship_deg"].round(2).astype(str)
        + "°"
    )


def _add_plain_detections_trace(fig, detections_df):
    """Original, track-agnostic rendering: all detections as white dots."""
    hover = _hover_text(detections_df)
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


def _add_tracked_detections_traces(fig, detections_df):
    """Color/connect detections by track_id; unassigned rows stay plain dots."""
    unassigned = detections_df[detections_df["track_id"] == UNASSIGNED_TRACK_ID]
    if not unassigned.empty:
        hover = "Track: unassigned<br>" + _hover_text(unassigned)
        fig.add_trace(
            go.Scatter(
                x=unassigned["ping_time"].values,
                y=unassigned["range_m"].values,
                mode="markers",
                marker={
                    "size": 6,
                    "color": "white",
                    "line": {"color": "black", "width": 1},
                },
                name="Unassigned",
                hovertext=hover,
                hoverinfo="text",
            )
        )

    track_ids = sorted(
        int(t) for t in detections_df["track_id"].unique() if t != UNASSIGNED_TRACK_ID
    )
    for legend_rank, track_id in enumerate(track_ids):
        track_rows = detections_df[detections_df["track_id"] == track_id]
        sort_col = "ping_index" if "ping_index" in track_rows.columns else "ping_time"
        track_rows = track_rows.sort_values(sort_col, kind="stable")

        hover = f"Track: {track_id}<br>" + _hover_text(track_rows)
        color = _track_color(track_id)
        fig.add_trace(
            go.Scatter(
                x=track_rows["ping_time"].values,
                y=track_rows["range_m"].values,
                mode="lines+markers",
                marker={"size": 6, "color": color, "line": {"color": "black", "width": 1}},
                line={"color": color, "width": 1.5},
                name=f"Track {track_id}",
                legendgroup=f"track-{track_id}",
                showlegend=legend_rank < _MAX_LEGEND_TRACKS,
                hovertext=hover,
                hoverinfo="text",
            )
        )
