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

This module forms tracks (`assign_tracks`) and summarizes per-track
statistics (`summarize_tracks`), including speed through the local Cartesian
frame and an optional TS-derived equivalent length. It does not compute UTM
position -- there is no GPS in this deployment yet, that is a separate
follow-up step out of scope here.
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
    #
    # min_st_track / min_pings_track: ESP3's generic defaults were 8 / 10.
    # Both have since been overridden below with values empirically derived
    # by Scott (the client's hydroacoustics contractor) from hand-labeled
    # analysis of stationary side-looking EK80 WBT data in a heavy
    # air-bubble-noise environment: "I have tried to isolate only those
    # tracks that were greater than 4-pings long (or 3-pings < 10m range)
    # and deviated in range < 12 cm." That quote is the source of
    # min_pings_track=4 here, and of min_pings_track_near / near_range_track_m
    # / max_range_deviation_m below. If a future reader finds ESP3's 8/10
    # cited elsewhere in project history, that is not a bug -- it is the
    # prior generic default, superseded here for this noise environment by
    # someone who actually looked at labeled real data.
    #
    # min_st_track is set equal to min_pings_track (4, not ESP3's 8) because
    # detection-level within-ping dedup happens upstream of tracking, so in
    # practice a track's detection count and its ping count are normally the
    # same number (at most one detection per ping survives dedup). Requiring
    # 8 detections while only requiring a 4-ping span would make the
    # ping-span criterion moot -- a track could satisfy min_pings_track at 4
    # pings but still be rejected on detection count alone, silently
    # re-imposing something close to the old 8/10 pair through the back
    # door. Aligning the two numbers keeps the ping-span gate meaningful.
    "min_st_track": 4,
    "min_pings_track": 4,
    "max_gap_track": 5,
    # Range-deviation acceptance gate (Scott's "deviated in range < 12 cm"):
    # a track is rejected unless max(range_m) - min(range_m) across all its
    # detections is <= this value. Deliberately computed from the raw
    # range_m column, not the local Cartesian range_axis_m -- range_m is the
    # physically interpretable quantity Scott is describing (a straight
    # along-beam range reading), whereas range_axis_m is an internal
    # tracking-frame convenience copy of the same numbers (see
    # _angles_to_local_cartesian) that happens to be numerically identical
    # here but is conceptually the wrong column to say "matches Scott's
    # heuristic" against.
    "max_range_deviation_m": 0.12,
    # Range-conditional minimum ping-span relaxation (Scott's "3-pings <
    # 10m range"): tracks whose range is within near_range_track_m may pass
    # acceptance with only min_pings_track_near pings instead of the full
    # min_pings_track.
    "near_range_track_m": 10.0,
    "min_pings_track_near": 3,
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
    column, plus the `major_axis_m` / `minor_axis_m` / `range_axis_m` local
    Cartesian columns computed internally by `_angles_to_local_cartesian`
    (see its docstring for the axis convention). These are attached to every
    row (including unassigned ones), not just accepted tracks, since they
    are a straightforward per-detection geometric quantity independent of
    track acceptance; downstream code (e.g. `summarize_tracks`) uses them
    directly rather than recomputing the angle-to-Cartesian conversion.

    Detections not part of an accepted track are given
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
        result["major_axis_m"] = pd.Series(dtype="float64")
        result["minor_axis_m"] = pd.Series(dtype="float64")
        result["range_axis_m"] = pd.Series(dtype="float64")
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

    # Track acceptance: keep only tracks with enough detections, enough ping
    # span, and range coherence; discard the rest (their detections become
    # unassigned). See Scott's heuristic quoted above TRACKER_PARAM_DEFAULTS.
    min_st_track = int(p["min_st_track"])
    min_pings_track = int(p["min_pings_track"])
    min_pings_track_near = int(p["min_pings_track_near"])
    near_range_track_m = float(p["near_range_track_m"])
    max_range_deviation_m = float(p["max_range_deviation_m"])
    range_m_arr = df["range_m"].to_numpy(dtype=float)

    accepted_row_track_id = np.full(len(df), UNASSIGNED_TRACK_ID, dtype=int)
    accepted_id_counter = 0
    for tr in finished_tracks:
        n_detections = len(tr.row_indices)
        first_ping = df["ping_index"].to_numpy()[tr.row_indices].min()
        last_ping = df["ping_index"].to_numpy()[tr.row_indices].max()
        ping_span = int(last_ping - first_ping + 1)

        track_range_m = range_m_arr[tr.row_indices]
        # "How close does this track get" is judged by its minimum range_m,
        # not its mean -- Scott's relaxation ("3-pings < 10m range") reads as
        # being about whether the target ever comes within the near-range
        # threshold (better SNR at close range), not about its average
        # range over the whole track. A track that starts far and swims
        # into 8m range should get the same benefit of the doubt as one
        # that stayed at 8m throughout; using the mean would penalize a
        # track for time spent farther away even though the close approach
        # is what gives us confidence in its detections.
        track_min_range_m = float(track_range_m.min())
        applicable_min_pings = (
            min_pings_track_near if track_min_range_m < near_range_track_m else min_pings_track
        )
        range_deviation_m = float(track_range_m.max() - track_range_m.min())

        if (
            n_detections >= min_st_track
            and ping_span >= applicable_min_pings
            and range_deviation_m <= max_range_deviation_m
        ):
            for row in tr.row_indices:
                accepted_row_track_id[row] = accepted_id_counter
            accepted_id_counter += 1

    df["track_id"] = accepted_row_track_id

    # Restore the caller's original row order (tracking above works in
    # ping_index order internally, which need not match the input order).
    df = df.sort_values("_original_row", kind="stable")
    result = detections_df.copy()
    result["track_id"] = df["track_id"].to_numpy()
    # Also expose the local Cartesian frame used internally for tracking --
    # this is generally useful output (e.g. for speed-through-track
    # computations downstream in summarize_tracks), not just an
    # implementation detail, so we attach it here rather than making callers
    # recompute _angles_to_local_cartesian themselves.
    result["major_axis_m"] = df["major_axis_m"].to_numpy()
    result["minor_axis_m"] = df["minor_axis_m"].to_numpy()
    result["range_axis_m"] = df["range_axis_m"].to_numpy()
    return result


def _track_mean_speed_m_per_s(track_rows: pd.DataFrame) -> float:
    """
    Average speed magnitude of a single track through the local Cartesian
    frame (`major_axis_m`, `minor_axis_m`, `range_axis_m`), using actual
    elapsed wall-clock time between consecutive detections (`ping_time`)
    rather than ping-count, since ping cadence may not be perfectly uniform.

    `track_rows` must already be sorted by ping order. A step is skipped
    (rather than raising) if its time delta is zero -- this should not
    happen in practice (two detections of the same track at the identical
    timestamp), but we guard against a division by zero regardless. Returns
    NaN if there are fewer than two detections, or if every step was
    skipped, since no speed estimate is possible.
    """
    positions = track_rows[["major_axis_m", "minor_axis_m", "range_axis_m"]].to_numpy(dtype=float)
    times = track_rows["ping_time"].to_numpy()

    if len(positions) < 2:
        return float("nan")

    step_speeds: List[float] = []
    for i in range(len(positions) - 1):
        dt = (times[i + 1] - times[i]) / np.timedelta64(1, "s")
        if dt <= 0:
            continue
        displacement = np.linalg.norm(positions[i + 1] - positions[i])
        step_speeds.append(displacement / dt)

    if not step_speeds:
        return float("nan")
    return float(np.mean(step_speeds))


def summarize_tracks(
    tracked_df: pd.DataFrame,
    ts_length_a: Optional[float] = None,
    ts_length_b: Optional[float] = None,
) -> pd.DataFrame:
    """
    Build one summary row per accepted track_id (unassigned detections,
    track_id == UNASSIGNED_TRACK_ID, are excluded).

    `tracked_df` must be the output of `assign_tracks` (or otherwise carry
    its `major_axis_m` / `minor_axis_m` / `range_axis_m` columns), since
    `speed_m_per_s_mean` is computed from them directly rather than
    recomputing the angle-to-Cartesian conversion here.

    `ts_length_a` / `ts_length_b` are optional coefficients of the standard
    fisheries-acoustics regression `TS = a*log10(length_cm) + b`; if both
    are provided, an `equivalent_length_cm` column is added per track by
    inverting that regression against `ts_compensated_db_mean`:
    `10 ** ((ts_compensated_db_mean - b) / a)`. If either is None (the
    default), the column is omitted entirely -- this constant is
    species-specific and not yet available for this deployment's target
    species, so it is a pass-through for the caller to supply later, not
    something to compute or default here.

    Does not compute UTM position -- there is no GPS in this deployment yet,
    that is a separate follow-up step, out of scope here.
    """
    base_columns = [
        "track_id",
        "n_detections",
        "n_pings",
        "first_ping_index",
        "last_ping_index",
        "ts_compensated_db_mean",
        "ts_compensated_db_min",
        "ts_compensated_db_max",
        "range_m_mean",
        "range_m_min",
        "range_m_max",
        "range_deviation_m",
        "speed_m_per_s_mean",
    ]
    if ts_length_a is not None and ts_length_b is not None:
        base_columns.append("equivalent_length_cm")

    accepted = tracked_df[tracked_df["track_id"] != UNASSIGNED_TRACK_ID]
    if len(accepted) == 0:
        return pd.DataFrame(columns=base_columns)

    grouped = accepted.groupby("track_id")
    summary = grouped.agg(
        n_detections=("ping_index", "size"),
        n_pings=("ping_index", "nunique"),
        first_ping_index=("ping_index", "min"),
        last_ping_index=("ping_index", "max"),
        ts_compensated_db_mean=("ts_compensated_db", "mean"),
        ts_compensated_db_min=("ts_compensated_db", "min"),
        ts_compensated_db_max=("ts_compensated_db", "max"),
        range_m_mean=("range_m", "mean"),
        range_m_min=("range_m", "min"),
        range_m_max=("range_m", "max"),
    ).reset_index()
    # Same max-min quantity used for the tracker's range-deviation
    # acceptance gate (see max_range_deviation_m), surfaced per-track here
    # since it is directly relevant to interpreting result quality.
    summary["range_deviation_m"] = summary["range_m_max"] - summary["range_m_min"]

    speed_by_track = {}
    for track_id, track_rows in accepted.groupby("track_id"):
        track_rows = track_rows.sort_values("ping_index", kind="stable")
        speed_by_track[track_id] = _track_mean_speed_m_per_s(track_rows)
    summary["speed_m_per_s_mean"] = summary["track_id"].map(speed_by_track)

    if ts_length_a is not None and ts_length_b is not None:
        summary["equivalent_length_cm"] = 10 ** (
            (summary["ts_compensated_db_mean"] - ts_length_b) / ts_length_a
        )

    return summary.sort_values("track_id").reset_index(drop=True)[base_columns]
