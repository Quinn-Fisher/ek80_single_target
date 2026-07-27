"""Alpha-beta track formation on top of per-ping single-target detections.

This mirrors NIWA's ESP3 (`src/algos/tarck_target/track_targets_angular.m`)
angular alpha-beta tracker, adapted for a stationary shore-based deployment
with no vessel attitude data (heave/pitch/roll/yaw) -- i.e. the
`IgnoreAttitude=true` path in ESP3. Positions are tracked in a local
range/angle-derived Cartesian frame; there is no UTM/GPS position available
yet in this deployment (tracked separately, out of scope here).

Coordinate convention (matches ESP3's `angles_to_pos_single` with zero
attitude -- please read this twice, it is easy to flip):
    major_axis_m -> derived from ATHWARTSHIP angle (ESP3's "major axis" / Y)
    minor_axis_m -> derived from ALONGSHIP angle  (ESP3's "minor axis" / X)
    range_axis_m -> range_m unchanged

This module only forms tracks. It does not compute equivalent size-from-TS,
UTM position, or speed -- that is a separate follow-up step.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# ESP3-derived default tracker parameters (see track_targets_angular.m
# function-argument defaults in NIWA's ESP3 toolbox).
TRACKER_PARAM_DEFAULTS: Dict[str, float] = {
    # Alpha-beta filter gains, per axis.
    "alpha_major": 0.7,
    "alpha_minor": 0.7,
    "alpha_range": 0.7,
    "beta_major": 0.5,
    "beta_minor": 0.5,
    "beta_range": 0.5,
    # Gate half-width base terms (meters) and angular widening (degrees).
    "excl_dist_major_m": 2.0,
    "excl_dist_minor_m": 2.0,
    "excl_dist_range_m": 2.0,
    "max_std_major_deg": 2.0,
    "max_std_minor_deg": 2.0,
    # Missed-ping gate expansion, percent per ping of gap.
    "missed_ping_exp_major_pct": 5.0,
    "missed_ping_exp_minor_pct": 5.0,
    "missed_ping_exp_range_pct": 5.0,
    # Assignment cost weights.
    "weight_major": 10.0,
    "weight_minor": 10.0,
    "weight_range": 70.0,
    "weight_ts": 5.0,
    "weight_ping_gap": 5.0,
    "delta_ts_max_db": 30.0,
    # Track acceptance / gap tolerance.
    "min_st_track": 8,
    "min_pings_track": 10,
    "max_gap_track": 5,
}

# Value used in the output `track_id` column for detections that are not
# part of any accepted track (either never matched into a track, or matched
# into a track that failed the min_st_track / min_pings_track acceptance
# test). Chosen over NaN so the column can stay a plain integer dtype.
UNASSIGNED_TRACK_ID = -1


class _Track:
    """Mutable alpha-beta track state while tracking is in progress."""

    __slots__ = (
        "track_id",
        "pos",
        "vel",
        "last_ping_index",
        "last_ts_db",
        "row_indices",
    )

    def __init__(self, track_id: int, ping_index: int, major: float, minor: float, range_m: float, ts_db: float, row_index: int):
        self.track_id = track_id
        # Smoothed position per axis: (major, minor, range).
        self.pos = np.array([major, minor, range_m], dtype=float)
        # Smoothed velocity per axis (units per ping); zero at initiation
        # since a singleton track has no velocity estimate yet.
        self.vel = np.zeros(3, dtype=float)
        self.last_ping_index = ping_index
        self.last_ts_db = ts_db
        self.row_indices: List[int] = [row_index]

    def predict(self, ping_gap: int) -> np.ndarray:
        return self.pos + self.vel * ping_gap

    def update(self, predicted: np.ndarray, obs: np.ndarray, ping_gap: int, alpha: np.ndarray, beta: np.ndarray) -> None:
        residual = obs - predicted
        new_pos = predicted + alpha * residual
        observed_vel = residual / ping_gap
        new_vel = self.vel + beta * (observed_vel - self.vel)
        self.pos = new_pos
        self.vel = new_vel


def _angles_to_local_cartesian(detections_df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert (range_m, angle_alongship_deg, angle_athwartship_deg) to a local
    Cartesian frame, mirroring ESP3's `angles_to_pos_single` with zero
    attitude (IgnoreAttitude=true -- there is no heave/pitch/roll/yaw data
    for this stationary shore-based deployment).

    NOTE the axis naming: ESP3's "minor axis" (X) comes from the ALONGSHIP
    angle and its "major axis" (Y) comes from the ATHWARTSHIP angle. This is
    the opposite of what the names might suggest at a glance -- easy to flip.
    """
    out = detections_df.copy()
    range_m = out["range_m"].to_numpy(dtype=float)
    out["minor_axis_m"] = range_m * np.tan(np.radians(out["angle_alongship_deg"].to_numpy(dtype=float)))
    out["major_axis_m"] = range_m * np.tan(np.radians(out["angle_athwartship_deg"].to_numpy(dtype=float)))
    out["range_axis_m"] = range_m
    return out


def assign_tracks(detections_df: pd.DataFrame, params: dict) -> pd.DataFrame:
    """
    Form tracks from per-ping single-target detections using an ESP3-style
    per-axis alpha-beta filter with gated, cost-weighted association.

    Returns a copy of `detections_df` with an added integer `track_id`
    column. Detections not part of an accepted track are given
    `track_id = UNASSIGNED_TRACK_ID` (-1), not NaN, so the column stays a
    plain integer dtype (there is no missing-detection case to distinguish
    from "definitely not tracked").

    Deviations from the literal ESP3 source (documented per the task spec):
    - Missed-ping search is a straightforward "look back up to
      max_gap_track pings, extrapolate each candidate track's state to that
      gap" loop, not a transliteration of ESP3's MATLAB indexing.
    - The assignment cost's TS and ping-gap terms use
      `(ts_obs - ts_last_in_track)` and the ping gap actually being
      evaluated, rather than ESP3's own diff_TS/diff_pings terms (which
      reference an oddly-indexed pair of prior track points -- likely an
      implementation quirk rather than an intentional design choice).
    - Conflict resolution is a simple greedy global assignment: collect all
      valid (track, candidate, cost) triples for a ping across all active
      tracks, sort by cost ascending, and assign greedily with each track
      and each candidate used at most once. This is a clean stand-in for
      ESP3's own somewhat ad hoc resolution.
    """
    p = {**TRACKER_PARAM_DEFAULTS, **params}

    if len(detections_df) == 0:
        result = detections_df.copy()
        result["track_id"] = pd.Series(dtype="int64")
        return result

    df = _angles_to_local_cartesian(detections_df)
    # Remember each row's original position so the output can be restored to
    # the caller's input row order at the end, regardless of ping_index
    # sort order used internally for tracking.
    original_order = np.arange(len(df))
    df = df.assign(_original_row=original_order)
    df = df.sort_values(["ping_index"], kind="stable").reset_index(drop=True)

    alpha = np.array([p["alpha_major"], p["alpha_minor"], p["alpha_range"]], dtype=float)
    beta = np.array([p["beta_major"], p["beta_minor"], p["beta_range"]], dtype=float)
    max_gap_track = int(p["max_gap_track"])

    # Group row indices (into `df`) by ping_index, in ascending ping order.
    ping_groups: Dict[int, List[int]] = {}
    for row_idx, ping_idx in enumerate(df["ping_index"].to_numpy()):
        ping_groups.setdefault(int(ping_idx), []).append(row_idx)
    ordered_pings = sorted(ping_groups.keys())

    major = df["major_axis_m"].to_numpy(dtype=float)
    minor = df["minor_axis_m"].to_numpy(dtype=float)
    range_axis = df["range_axis_m"].to_numpy(dtype=float)
    ts_compensated = df["ts_compensated_db"].to_numpy(dtype=float)

    active_tracks: List[_Track] = []
    finished_tracks: List[_Track] = []
    next_track_id = 0
    # Rows already claimed by a track at the current ping, to avoid a row
    # being reused across multiple ping-gap lookbacks within one ping's
    # matching pass.
    row_track_id = np.full(len(df), UNASSIGNED_TRACK_ID, dtype=int)

    def gate_widths(track: _Track, ping_gap: int, obs_range_m: float) -> np.ndarray:
        gate_major = (
            p["excl_dist_major_m"] + obs_range_m * np.tan(np.radians(p["max_std_major_deg"]))
        ) * (1.0 + ping_gap * p["missed_ping_exp_major_pct"] / 100.0)
        gate_minor = (
            p["excl_dist_minor_m"] + obs_range_m * np.tan(np.radians(p["max_std_minor_deg"]))
        ) * (1.0 + ping_gap * p["missed_ping_exp_minor_pct"] / 100.0)
        gate_range = p["excl_dist_range_m"] * (1.0 + ping_gap * p["missed_ping_exp_range_pct"] / 100.0)
        return np.array([gate_major, gate_minor, gate_range], dtype=float)

    for ping_idx in ordered_pings:
        candidate_rows = [r for r in ping_groups[ping_idx] if row_track_id[r] == UNASSIGNED_TRACK_ID]
        if not candidate_rows:
            continue

        # Drop tracks that have exceeded the max allowed gap -- they can no
        # longer be extended and are finalized now.
        still_active: List[_Track] = []
        for tr in active_tracks:
            if ping_idx - tr.last_ping_index > max_gap_track:
                finished_tracks.append(tr)
            else:
                still_active.append(tr)
        active_tracks = still_active

        # Build all valid (track, row, cost) triples for this ping, gating
        # against each track's state extrapolated to its actual ping_gap.
        triples: List[Tuple[float, int, int]] = []  # (cost, track_list_idx, row)
        for t_idx, tr in enumerate(active_tracks):
            ping_gap = ping_idx - tr.last_ping_index
            if ping_gap <= 0 or ping_gap > max_gap_track:
                continue
            predicted = tr.predict(ping_gap)
            for row in candidate_rows:
                obs = np.array([major[row], minor[row], range_axis[row]], dtype=float)
                gates = gate_widths(tr, ping_gap, range_axis[row])
                normalized = (obs - predicted) / gates
                gate_metric = float(np.sum(normalized**2))
                if gate_metric >= 1.0:
                    continue
                cost = (
                    p["weight_major"] * normalized[0] ** 2
                    + p["weight_minor"] * normalized[1] ** 2
                    + p["weight_range"] * normalized[2] ** 2
                    + p["weight_ts"] * (ts_compensated[row] - tr.last_ts_db) ** 2 / p["delta_ts_max_db"] ** 2
                    + p["weight_ping_gap"] * ping_gap**2 / max(max_gap_track, 1) ** 2
                )
                triples.append((cost, t_idx, row))

        triples.sort(key=lambda x: x[0])
        used_tracks: set = set()
        used_rows: set = set()
        for cost, t_idx, row in triples:
            if t_idx in used_tracks or row in used_rows:
                continue
            used_tracks.add(t_idx)
            used_rows.add(row)
            tr = active_tracks[t_idx]
            ping_gap = ping_idx - tr.last_ping_index
            predicted = tr.predict(ping_gap)
            obs = np.array([major[row], minor[row], range_axis[row]], dtype=float)
            tr.update(predicted, obs, ping_gap, alpha, beta)
            tr.last_ping_index = ping_idx
            tr.last_ts_db = ts_compensated[row]
            tr.row_indices.append(row)
            row_track_id[row] = tr.track_id

        # Any candidate row not claimed this ping starts a new singleton
        # track.
        for row in candidate_rows:
            if row in used_rows:
                continue
            new_track = _Track(
                track_id=next_track_id,
                ping_index=ping_idx,
                major=major[row],
                minor=minor[row],
                range_m=range_axis[row],
                ts_db=ts_compensated[row],
                row_index=row,
            )
            row_track_id[row] = next_track_id
            active_tracks.append(new_track)
            next_track_id += 1

    finished_tracks.extend(active_tracks)

    # Track acceptance: keep only tracks with enough detections and enough
    # ping span; discard the rest (their detections become unassigned).
    min_st_track = int(p["min_st_track"])
    min_pings_track = int(p["min_pings_track"])

    accepted_row_track_id = np.full(len(df), UNASSIGNED_TRACK_ID, dtype=int)
    accepted_id_counter = 0
    for tr in finished_tracks:
        n_detections = len(tr.row_indices)
        first_ping = df["ping_index"].to_numpy()[tr.row_indices].min()
        last_ping = df["ping_index"].to_numpy()[tr.row_indices].max()
        ping_span = int(last_ping - first_ping + 1)
        if n_detections >= min_st_track and ping_span >= min_pings_track:
            for row in tr.row_indices:
                accepted_row_track_id[row] = accepted_id_counter
            accepted_id_counter += 1

    df["track_id"] = accepted_row_track_id

    # Restore the caller's original row order (tracking above works in
    # ping_index order internally, which need not match the input order).
    df = df.sort_values("_original_row", kind="stable")
    result = detections_df.copy()
    result["track_id"] = df["track_id"].to_numpy()
    return result


def summarize_tracks(tracked_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build one summary row per accepted track_id (unassigned detections,
    track_id == UNASSIGNED_TRACK_ID, are excluded).

    Does not compute equivalent size-from-TS, UTM position, or speed -- that
    is a separate follow-up step, out of scope here.
    """
    accepted = tracked_df[tracked_df["track_id"] != UNASSIGNED_TRACK_ID]
    if len(accepted) == 0:
        return pd.DataFrame(
            columns=[
                "track_id",
                "n_detections",
                "n_pings",
                "first_ping_index",
                "last_ping_index",
                "ts_compensated_db_mean",
                "ts_compensated_db_min",
                "ts_compensated_db_max",
            ]
        )

    grouped = accepted.groupby("track_id")
    summary = grouped.agg(
        n_detections=("ping_index", "size"),
        n_pings=("ping_index", "nunique"),
        first_ping_index=("ping_index", "min"),
        last_ping_index=("ping_index", "max"),
        ts_compensated_db_mean=("ts_compensated_db", "mean"),
        ts_compensated_db_min=("ts_compensated_db", "min"),
        ts_compensated_db_max=("ts_compensated_db", "max"),
    ).reset_index()

    return summary.sort_values("track_id").reset_index(drop=True)
