"""Tests for detection/tracker.py alpha-beta track formation.

These use small, hand-built synthetic detections_df fixtures (not real EK80
files) designed to exercise: smooth single-track formation, identity
preservation across a track crossing, gap-bridging via missed-ping search,
rejection of scattered noise, and summarize_tracks aggregation.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from detection.tracker import (
    TRACKER_PARAM_DEFAULTS,
    UNASSIGNED_TRACK_ID,
    assign_tracks,
    summarize_tracks,
)


def _angles_for(range_m: float, major_m: float, minor_m: float) -> tuple:
    """Invert the tracker's angle->Cartesian conversion to build a detection
    row's angle_alongship_deg / angle_athwartship_deg from a desired local
    Cartesian position, matching tracker.py's axis convention:
    major_axis_m <- athwartship angle, minor_axis_m <- alongship angle.
    """
    angle_alongship_deg = math.degrees(math.atan(minor_m / range_m))
    angle_athwartship_deg = math.degrees(math.atan(major_m / range_m))
    return angle_alongship_deg, angle_athwartship_deg


def _make_row(ping_index: int, range_m: float, major_m: float, minor_m: float, ts_compensated_db: float, **extra) -> dict:
    angle_along, angle_athwart = _angles_for(range_m, major_m, minor_m)
    row = {
        "ping_time": pd.Timestamp("2026-01-01") + pd.Timedelta(seconds=ping_index * 0.1),
        "ping_index": ping_index,
        "range_sample_index": 100 + ping_index,
        "range_m": range_m,
        "angle_alongship_deg": angle_along,
        "angle_athwartship_deg": angle_athwart,
        "ts_uncompensated_db": ts_compensated_db - 1.0,
        "ts_compensated_db": ts_compensated_db,
        "phase_std_alongship_deg": 0.3,
        "phase_std_athwartship_deg": 0.3,
        "normalized_pulse_width": 1.0,
        "compensation_db": 1.0,
        "threshold_level_passed": 3,
        "phase_gate_skipped": False,
    }
    row.update(extra)
    return row


def _tracked_track_ids(tracked_df: pd.DataFrame) -> set:
    return set(tracked_df.loc[tracked_df["track_id"] != UNASSIGNED_TRACK_ID, "track_id"].unique())


def test_single_smooth_track_forms_one_accepted_track():
    rows = []
    n_pings = 18
    # range_m deliberately drifts by only ~0.085m total (17 steps * 0.005m)
    # over the whole track -- comfortably inside max_range_deviation_m
    # (0.12m) -- so this fixture continues to test smooth single-track
    # *formation* across many pings, not the (separately tested) range-
    # deviation acceptance gate.
    for i in range(n_pings):
        range_m = 10.0 + i * 0.005
        major_m = 0.2 + i * 0.03
        minor_m = -0.1 + i * 0.02
        rows.append(_make_row(i, range_m, major_m, minor_m, ts_compensated_db=-35.0))
    df = pd.DataFrame(rows)

    result = assign_tracks(df, TRACKER_PARAM_DEFAULTS)

    assert len(result) == n_pings
    track_ids = _tracked_track_ids(result)
    assert len(track_ids) == 1
    assert (result["track_id"] != UNASSIGNED_TRACK_ID).all()


def test_crossing_tracks_keep_correct_identity_via_ts_weighting():
    n_pings = 15
    rows = []
    # Fish A: major axis sweeps from -5m to +5m; low TS.
    # Fish B: major axis sweeps from +5m to -5m; high TS.
    # They cross (both near major=0) around ping index 7.
    for i in range(n_pings):
        major_a = -5.0 + i * (10.0 / (n_pings - 1))
        major_b = 5.0 - i * (10.0 / (n_pings - 1))
        rows.append(_make_row(i, range_m=20.0, major_m=major_a, minor_m=0.0, ts_compensated_db=-40.0, true_fish="A"))
        rows.append(_make_row(i, range_m=20.0, major_m=major_b, minor_m=0.0, ts_compensated_db=-25.0, true_fish="B"))
    df = pd.DataFrame(rows)

    result = assign_tracks(df, TRACKER_PARAM_DEFAULTS)

    track_ids = _tracked_track_ids(result)
    assert len(track_ids) == 2, "expected exactly two accepted tracks (fish A and fish B)"

    for tid in track_ids:
        subset = result[result["track_id"] == tid]
        # Each accepted track must contain detections from only one true fish
        # across the whole span, including before and after the crossing at
        # ping index 7 -- i.e. identities must not swap at the crossing.
        assert subset["true_fish"].nunique() == 1, (
            f"track {tid} mixes fish identities across the crossing: "
            f"{subset[['ping_index', 'true_fish']].to_dict('records')}"
        )
        # Each fish contributes one detection per ping, so a fully-tracked
        # fish should have all n_pings detections in the same track.
        assert len(subset) == n_pings


def test_track_survives_a_few_consecutive_missed_pings():
    # Single fish, linear constant-velocity motion, but ping indices 5,6,7
    # produce no detection (simulating missed detections mid-track). With
    # perfectly linear motion the alpha-beta prediction should still land
    # inside the (gap-expanded) gate when the fish reappears at ping 8.
    missing = {5, 6, 7}
    rows = []
    for i in range(15):
        if i in missing:
            continue
        range_m = 15.0
        major_m = 0.0
        minor_m = 0.1 * i
        rows.append(_make_row(i, range_m, major_m, minor_m, ts_compensated_db=-35.0))
    df = pd.DataFrame(rows)

    result = assign_tracks(df, TRACKER_PARAM_DEFAULTS)

    track_ids = _tracked_track_ids(result)
    assert len(track_ids) == 1, "the gap should be bridged into a single track, not split in two"
    tid = next(iter(track_ids))
    subset = result[result["track_id"] == tid]
    assert len(subset) == 15 - len(missing)
    assert subset["ping_index"].min() == 0
    assert subset["ping_index"].max() == 14


def test_scattered_noise_detections_are_unassigned():
    # Widely scattered, uncorrelated single-ping detections: no two should
    # ever fall inside each other's gate (gate half-widths are only a few
    # meters), so every detection should end as a singleton, unaccepted
    # track (track_id == UNASSIGNED_TRACK_ID).
    scattered_positions = [
        (10.0, -40.0, 5.0),
        (25.0, 30.0, -20.0),
        (60.0, -15.0, 40.0),
        (12.0, 50.0, -45.0),
        (33.0, -60.0, 10.0),
        (80.0, 5.0, -30.0),
        (18.0, 45.0, 45.0),
        (48.0, -35.0, -10.0),
        (70.0, 20.0, 25.0),
        (22.0, -50.0, 55.0),
    ]
    rows = [
        _make_row(i, range_m=r, major_m=maj, minor_m=minr, ts_compensated_db=-38.0)
        for i, (r, maj, minr) in enumerate(scattered_positions)
    ]
    df = pd.DataFrame(rows)

    result = assign_tracks(df, TRACKER_PARAM_DEFAULTS)

    assert (result["track_id"] == UNASSIGNED_TRACK_ID).all()


def test_range_deviation_gate_rejects_wandering_track():
    # Same "smooth track" shape (major/minor axes) as the passing fixture
    # above -- enough pings/detections to satisfy min_st_track /
    # min_pings_track on their own -- but range_m now oscillates by 0.3m
    # (well under the alpha-beta range gate's excl_dist_range_m=2.0m, so it
    # still forms a single track, but well over max_range_deviation_m=0.12m),
    # so the new range-deviation acceptance gate should reject it even
    # though it previously would have passed.
    rows = []
    n_pings = 18
    for i in range(n_pings):
        range_m = 10.0 + (0.3 if i % 2 == 0 else 0.0)
        major_m = 0.2 + i * 0.03
        minor_m = -0.1 + i * 0.02
        rows.append(_make_row(i, range_m, major_m, minor_m, ts_compensated_db=-35.0))
    df = pd.DataFrame(rows)

    result = assign_tracks(df, TRACKER_PARAM_DEFAULTS)

    assert (result["track_id"] == UNASSIGNED_TRACK_ID).all(), (
        "track's range wanders by 0.3m > max_range_deviation_m=0.12m; should be rejected"
    )


def test_near_range_relaxation_accepts_short_close_track():
    # Only 3 pings/3 detections, all within near_range_track_m (range ~5m)
    # and within the range-deviation gate. min_st_track is overridden to 3
    # here to isolate the ping-span relaxation logic (the default
    # min_st_track=4 would otherwise reject any 3-detection track on
    # detection count alone, before the ping-span/range-conditional logic
    # even gets exercised). Under the *old* ESP3-derived min_pings_track=10
    # default this 3-ping track would have been rejected outright; under
    # the new near-range relaxation (min_pings_track_near=3 when
    # range < near_range_track_m=10.0) it should be accepted.
    params = {**TRACKER_PARAM_DEFAULTS, "min_st_track": 3}
    rows = []
    for i in range(3):
        range_m = 5.0 + i * 0.02
        major_m = 0.0
        minor_m = 0.1 * i
        rows.append(_make_row(i, range_m, major_m, minor_m, ts_compensated_db=-35.0))
    df = pd.DataFrame(rows)

    result = assign_tracks(df, params)

    assert len(_tracked_track_ids(result)) == 1
    assert (result["track_id"] != UNASSIGNED_TRACK_ID).all()


def test_near_range_relaxation_does_not_apply_far_from_sensor():
    # Identical shape to the test above (3 pings, min_st_track relaxed to 3,
    # within the range-deviation gate) but range is ~50m, well outside
    # near_range_track_m=10.0. The relaxed 3-ping threshold must not apply
    # here -- the full min_pings_track=4 still governs, so this 3-ping track
    # should be rejected, proving the relaxation is range-conditional and
    # not universal.
    params = {**TRACKER_PARAM_DEFAULTS, "min_st_track": 3}
    rows = []
    for i in range(3):
        range_m = 50.0 + i * 0.02
        major_m = 0.0
        minor_m = 0.1 * i
        rows.append(_make_row(i, range_m, major_m, minor_m, ts_compensated_db=-35.0))
    df = pd.DataFrame(rows)

    result = assign_tracks(df, params)

    assert (result["track_id"] == UNASSIGNED_TRACK_ID).all(), (
        "3-ping track far from sensor should still be rejected -- near-range "
        "relaxation must not apply universally"
    )


def _base_time(ping_index: int) -> pd.Timestamp:
    return pd.Timestamp("2026-01-01") + pd.Timedelta(seconds=ping_index * 0.1)


def test_summarize_tracks_aggregates_correctly():
    ping_index = [0, 1, 2, 5, 6, 7, 8]
    tracked_df = pd.DataFrame(
        {
            "ping_index": ping_index,
            "ping_time": [_base_time(i) for i in ping_index],
            "ts_compensated_db": [-30.0, -32.0, -28.0, -20.0, -22.0, -24.0, -18.0],
            "track_id": [0, 0, 0, 1, 1, 1, 1],
            "range_m": [10.0, 12.0, 11.0, 30.0, 33.0, 28.0, 31.0],
            # Cartesian columns are unused by range/TS stats -- constant here,
            # speed behavior is covered by a dedicated test below.
            "major_axis_m": [0.0] * 7,
            "minor_axis_m": [0.0] * 7,
            "range_axis_m": [10.0, 12.0, 11.0, 30.0, 33.0, 28.0, 31.0],
        }
    )

    summary = summarize_tracks(tracked_df)

    assert list(summary["track_id"]) == [0, 1]
    assert "equivalent_length_cm" not in summary.columns

    row0 = summary[summary["track_id"] == 0].iloc[0]
    assert row0["n_detections"] == 3
    assert row0["n_pings"] == 3
    assert row0["first_ping_index"] == 0
    assert row0["last_ping_index"] == 2
    assert row0["ts_compensated_db_mean"] == pytest.approx((-30.0 - 32.0 - 28.0) / 3.0)
    assert row0["ts_compensated_db_min"] == pytest.approx(-32.0)
    assert row0["ts_compensated_db_max"] == pytest.approx(-28.0)
    assert row0["range_m_mean"] == pytest.approx((10.0 + 12.0 + 11.0) / 3.0)
    assert row0["range_m_min"] == pytest.approx(10.0)
    assert row0["range_m_max"] == pytest.approx(12.0)
    assert row0["range_deviation_m"] == pytest.approx(12.0 - 10.0)

    row1 = summary[summary["track_id"] == 1].iloc[0]
    assert row1["n_detections"] == 4
    assert row1["n_pings"] == 4
    assert row1["first_ping_index"] == 5
    assert row1["last_ping_index"] == 8
    assert row1["ts_compensated_db_mean"] == pytest.approx((-20.0 - 22.0 - 24.0 - 18.0) / 4.0)
    assert row1["ts_compensated_db_min"] == pytest.approx(-24.0)
    assert row1["ts_compensated_db_max"] == pytest.approx(-18.0)
    assert row1["range_m_mean"] == pytest.approx((30.0 + 33.0 + 28.0 + 31.0) / 4.0)
    assert row1["range_m_min"] == pytest.approx(28.0)
    assert row1["range_m_max"] == pytest.approx(33.0)
    assert row1["range_deviation_m"] == pytest.approx(33.0 - 28.0)


def test_summarize_tracks_excludes_unassigned():
    ping_index = [0, 1, 2]
    tracked_df = pd.DataFrame(
        {
            "ping_index": ping_index,
            "ping_time": [_base_time(i) for i in ping_index],
            "ts_compensated_db": [-30.0, -31.0, -29.0],
            "track_id": [UNASSIGNED_TRACK_ID, UNASSIGNED_TRACK_ID, UNASSIGNED_TRACK_ID],
            "range_m": [10.0, 11.0, 12.0],
            "major_axis_m": [0.0, 0.0, 0.0],
            "minor_axis_m": [0.0, 0.0, 0.0],
            "range_axis_m": [10.0, 11.0, 12.0],
        }
    )
    summary = summarize_tracks(tracked_df)
    assert len(summary) == 0


def test_summarize_tracks_speed_with_irregular_ping_times():
    # Single track, constant velocity of 2.0 m/s purely along the minor
    # axis, but with deliberately irregular ping_time spacing (0.1s, 0.3s,
    # 0.05s steps) so the test actually exercises real-elapsed-time speed
    # computation rather than a ping-count approximation. Positions are
    # chosen to match velocity * elapsed_time exactly, so the expected mean
    # speed is exactly 2.0 m/s regardless of the irregular spacing.
    t0 = pd.Timestamp("2026-01-01T00:00:00")
    dt_steps = [0.1, 0.3, 0.05]  # seconds between consecutive detections
    times = [t0]
    for dt in dt_steps:
        times.append(times[-1] + pd.Timedelta(seconds=dt))

    speed = 2.0
    minor_positions = [0.0]
    for dt in dt_steps:
        minor_positions.append(minor_positions[-1] + speed * dt)

    n = len(times)
    tracked_df = pd.DataFrame(
        {
            "ping_index": list(range(n)),
            "ping_time": times,
            "ts_compensated_db": [-30.0] * n,
            "track_id": [0] * n,
            "range_m": [20.0] * n,
            "major_axis_m": [0.0] * n,
            "minor_axis_m": minor_positions,
            "range_axis_m": [20.0] * n,
        }
    )

    summary = summarize_tracks(tracked_df)

    assert len(summary) == 1
    assert summary.iloc[0]["speed_m_per_s_mean"] == pytest.approx(2.0, rel=1e-9)


def test_summarize_tracks_equivalent_length_present_when_coefficients_given():
    ping_index = [0, 1, 2]
    tracked_df = pd.DataFrame(
        {
            "ping_index": ping_index,
            "ping_time": [_base_time(i) for i in ping_index],
            "ts_compensated_db": [-40.0, -40.0, -40.0],
            "track_id": [0, 0, 0],
            "range_m": [15.0, 15.0, 15.0],
            "major_axis_m": [0.0, 0.0, 0.0],
            "minor_axis_m": [0.0, 0.0, 0.0],
            "range_axis_m": [15.0, 15.0, 15.0],
        }
    )

    ts_length_a, ts_length_b = 20.0, -68.0
    summary = summarize_tracks(tracked_df, ts_length_a=ts_length_a, ts_length_b=ts_length_b)

    assert "equivalent_length_cm" in summary.columns
    expected_length_cm = 10 ** ((-40.0 - ts_length_b) / ts_length_a)
    assert summary.iloc[0]["equivalent_length_cm"] == pytest.approx(expected_length_cm)

    # Default call (no coefficients) must not include the column at all.
    summary_default = summarize_tracks(tracked_df)
    assert "equivalent_length_cm" not in summary_default.columns


def test_assign_tracks_output_includes_local_cartesian_columns():
    rows = []
    for i in range(12):
        range_m = 10.0 + i * 0.05
        major_m = 0.2 + i * 0.03
        minor_m = -0.1 + i * 0.02
        rows.append(_make_row(i, range_m, major_m, minor_m, ts_compensated_db=-35.0))
    df = pd.DataFrame(rows)

    result = assign_tracks(df, TRACKER_PARAM_DEFAULTS)

    for col in ("major_axis_m", "minor_axis_m", "range_axis_m"):
        assert col in result.columns
        assert result[col].notna().all()
