"""Tests for viz/echogram.py: track-aware and plain-detection rendering.

Uses a small, hand-built xarray.Dataset (a few pings x range samples) so
plot_echogram can be exercised end-to-end without a real EK80 file.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest
import xarray as xr

from detection.tracker import UNASSIGNED_TRACK_ID
from viz.echogram import plot_echogram

CH = "GPT 38 kHz"
N_PINGS = 6
N_RANGE = 10


@pytest.fixture
def dataset():
    ping_time = pd.date_range("2026-01-01", periods=N_PINGS, freq="1s")
    range_sample = np.arange(N_RANGE)
    echo_range = np.tile(np.linspace(1.0, 20.0, N_RANGE), (N_PINGS, 1))
    sv = np.random.default_rng(0).uniform(-80, -20, size=(1, N_PINGS, N_RANGE))
    ds = xr.Dataset(
        {
            "Sv": (["channel", "ping_time", "range_sample"], sv),
            "echo_range": (["channel", "ping_time", "range_sample"], echo_range[np.newaxis, ...]),
        },
        coords={
            "channel": [CH],
            "ping_time": ping_time,
            "range_sample": range_sample,
        },
    )
    return ds


def _detections_df(with_track_id: bool):
    ping_time = pd.date_range("2026-01-01", periods=N_PINGS, freq="1s")
    rows = []
    for i in range(N_PINGS):
        rows.append(
            {
                "ping_time": ping_time[i],
                "ping_index": i,
                "range_m": 5.0 + i * 0.1,
                "ts_compensated_db": -45.0,
                "angle_alongship_deg": 0.5,
                "angle_athwartship_deg": -0.5,
            }
        )
    df = pd.DataFrame(rows)
    if with_track_id:
        # Two real tracks plus some unassigned rows.
        df["track_id"] = [0, 0, 0, 1, 1, UNASSIGNED_TRACK_ID]
    return df


def test_plain_detections_no_track_id_column(dataset):
    df = _detections_df(with_track_id=False)
    assert "track_id" not in df.columns
    fig = plot_echogram(dataset, df, CH, value_var="Sv")
    assert isinstance(fig, go.Figure)
    # Heatmap + a single "Single targets" scatter trace, unchanged behavior.
    trace_names = [t.name for t in fig.data]
    assert "Single targets" in trace_names
    assert not any(name and name.startswith("Track") for name in trace_names)


def test_empty_detections_df(dataset):
    fig = plot_echogram(dataset, pd.DataFrame(), CH, value_var="Sv")
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 1  # heatmap only


def test_tracked_detections_with_unassigned_rows(dataset):
    df = _detections_df(with_track_id=True)
    fig = plot_echogram(dataset, df, CH, value_var="Sv")
    assert isinstance(fig, go.Figure)
    trace_names = [t.name for t in fig.data]
    assert "Unassigned" in trace_names
    assert "Track 0" in trace_names
    assert "Track 1" in trace_names

    # Track traces should connect points in ping order via lines+markers.
    for t in fig.data:
        if t.name in ("Track 0", "Track 1"):
            assert t.mode == "lines+markers"
        if t.name == "Unassigned":
            assert t.mode == "markers"

    # Track 0 and Track 1 should get distinct colors.
    track0 = next(t for t in fig.data if t.name == "Track 0")
    track1 = next(t for t in fig.data if t.name == "Track 1")
    assert track0.marker.color != track1.marker.color


def test_ts_variant_still_works(dataset):
    ds = dataset.rename({"Sv": "TS"})
    df = _detections_df(with_track_id=False)
    fig = plot_echogram(ds, df, CH, value_var="TS")
    assert isinstance(fig, go.Figure)
