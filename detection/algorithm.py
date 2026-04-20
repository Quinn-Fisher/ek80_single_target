"""Soule et al. (1997) single target detection algorithm for EK80 split-beam IQ data."""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.signal import find_peaks

from detection.compensation import compute_beam_compensation


def _safe_channel_value(ds: Any, var_name: str, ch: Any, default: float) -> float:
    """Get a channel value if present, otherwise return default."""
    if var_name not in ds.variables:
        return float(default)
    da = ds[var_name]
    if "channel" in da.dims:
        da = da.sel(channel=ch)
    vals = np.asarray(da.values, dtype=float).reshape(-1)
    if vals.size == 0:
        return float(default)
    v = float(vals[0])
    if np.isnan(v):
        return float(default)
    return v


def _compute_ts_db(iq: np.ndarray, data: Dict[str, Any], params: Dict[str, float]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert complex EK80 split-beam IQ samples to uncalibrated TS (dB re 1 m^2).

    EK80 CW TS equation:
    TS = Pr + 20*log10(R) + 2*alpha*R
         - 10*log10(Pt) - 2*G0 - 20*log10(lambda) + 10*log10(16*pi^2)
    """
    ch = data["ch_splitbeam"]
    ds_Sv = data["ds_Sv"]
    ds_TS = data.get("ds_TS")

    # Prefer echopype-calibrated TS for EK80-consistent metadata use
    # (transmit power, frequency, absorption, sound speed, and impedance terms).
    if ds_TS is not None and "TS" in ds_TS.variables:
        ts_da = ds_TS["TS"]
        if "channel" in ts_da.dims:
            ts_da = ts_da.sel(channel=ch)
        ts_db = np.asarray(ts_da.values, dtype=float)

        # Apply user gain offset as a gain-term change in the sonar equation.
        # TS contains -2*G, so delta_gain maps to -2*delta_gain in TS.
        gain_offset_db = float(data.get("gain_db", data.get("gain_db_from_file", 0.0))) - float(
            data.get("gain_db_from_file", 0.0)
        )
        if gain_offset_db != 0.0:
            ts_db = ts_db - 2.0 * gain_offset_db

        if "echo_range" in ds_TS.coords or "echo_range" in ds_TS.variables:
            range_da = ds_TS["echo_range"]
        elif "echo_range" in ds_Sv.coords or "echo_range" in ds_Sv.variables:
            range_da = ds_Sv["echo_range"]
        else:
            raise ValueError("Could not find `echo_range` in TS/Sv datasets.")

        if "channel" in range_da.dims:
            range_da = range_da.sel(channel=ch)
        range_m = np.asarray(range_da.values, dtype=float)
        if range_m.ndim == 1:
            range_m_2d = np.broadcast_to(range_m[np.newaxis, :], ts_db.shape)
        else:
            range_m_2d = range_m
        range_m_2d = np.where(range_m_2d <= 0, 1e-6, range_m_2d)
        return ts_db, range_m_2d

    raise ValueError("Could not find calibrated TS in loaded data. Expected `ds_TS['TS']`.")


def _window_within_minus_6db(power_row: np.ndarray, peak_idx: int, peak_val: float) -> np.ndarray:
    threshold = peak_val - 6.0
    left = peak_idx
    right = peak_idx

    while left - 1 >= 0 and power_row[left - 1] >= threshold:
        left -= 1
    while right + 1 < power_row.size and power_row[right + 1] >= threshold:
        right += 1
    return np.arange(left, right + 1, dtype=int)


def _range_at(range_m_arr: np.ndarray, ping_idx: int, peak_idx: int) -> float:
    if np.ndim(range_m_arr) == 1:
        return float(range_m_arr[peak_idx])
    return float(range_m_arr[ping_idx, peak_idx])


def detect_single_targets(
    data: Dict[str, Any],
    params: Dict[str, float],
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """
    Run Soule 1997 phase-based single target discrimination.

    Notes:
    - `phase_std_max_deg` default 0.237 is one electrical phase step converted
      to mechanical degrees for ES38-18 style split-beam geometry.
    """
    beam = data["beam"]
    ch = data["ch_splitbeam"]
    pulse_dur = float(data["pulse_duration_s"])
    sample_spacing_s = float(data["sample_spacing_s"])

    bs_r = beam["backscatter_r"].sel(channel=ch).values
    bs_i = beam["backscatter_i"].sel(channel=ch).values
    iq = bs_r + 1j * bs_i

    ts_db, range_m_arr = _compute_ts_db(iq=iq, data=data, params=params)

    ping_times = beam.ping_time.values

    mgc = float(params["max_gain_compensation_db"])
    ts_min = float(params["ts_min_db"])
    th1 = ts_min - (2 * mgc + 6)
    th2 = ts_min - ((2 * mgc + 6) / 2.0)
    th3 = ts_min

    min_npw = float(params["min_normalized_pulse_width"])
    max_npw = float(params["max_normalized_pulse_width"])
    phase_std_max = float(params["phase_std_max_deg"])
    min_range_m = params.get("min_range_m")
    max_range_m = params.get("max_range_m")
    min_range_m = float(min_range_m) if min_range_m is not None else None
    max_range_m = float(max_range_m) if max_range_m is not None else None

    n_pings = ts_db.shape[0]
    detections: List[Dict[str, Any]] = []

    n_candidates_after_amplitude = 0
    n_rejected_duration = 0
    n_rejected_phase = 0
    n_rejected_final_ts = 0
    n_rejected_depth = 0
    n_phase_gate_skipped = 0

    angle_sens_along = float(data["angle_sensitivity_alongship"])
    angle_sens_athwart = float(data["angle_sensitivity_athwartship"])
    angle_offset_along = float(data["angle_offset_alongship"])
    angle_offset_athwart = float(data["angle_offset_athwartship"])

    for ping_idx in range(n_pings):
        row = ts_db[ping_idx, :]
        peaks, _ = find_peaks(row, height=th1, distance=2)

        candidates: List[Tuple[int, int]] = []
        for p in peaks:
            v = row[p]
            # Assign the strictest threshold tier passed (3 is strictest).
            if v >= th3:
                candidates.append((int(p), 3))
            elif v >= th2:
                candidates.append((int(p), 2))
            elif v >= th1:
                candidates.append((int(p), 1))

        n_candidates_after_amplitude += len(candidates)

        for peak_idx, level in candidates:
            range_val_m = _range_at(range_m_arr, ping_idx, peak_idx)
            if min_range_m is not None and range_val_m < min_range_m:
                n_rejected_depth += 1
                continue
            if max_range_m is not None and range_val_m > max_range_m:
                n_rejected_depth += 1
                continue

            peak_power_db = float(row[peak_idx])
            window_idx = _window_within_minus_6db(row, peak_idx, peak_power_db)

            n_samples = int(window_idx.size)
            actual_duration_s = n_samples * sample_spacing_s
            normalized_width = actual_duration_s / pulse_dur if pulse_dur > 0 else np.inf

            if not (min_npw <= normalized_width <= max_npw):
                n_rejected_duration += 1
                continue

            phase_gate_skipped = False
            std_along_deg = np.nan
            std_athwart_deg = np.nan
            angle_along_deg = 0.0
            angle_athwart_deg = 0.0

            if n_samples < 3:
                phase_gate_skipped = True
                n_phase_gate_skipped += 1
            else:
                phase_along = np.angle(iq[ping_idx, window_idx, 0] * np.conj(iq[ping_idx, window_idx, 1]))
                phase_athwart = np.angle(
                    iq[ping_idx, window_idx, 2]
                    * np.conj((iq[ping_idx, window_idx, 0] + iq[ping_idx, window_idx, 1]) / 2.0)
                )

                # Convert electrical phase (rad) to mechanical angle (deg)
                # using EK80 channel calibration sensitivities and offsets.
                phase_along_deg = (phase_along * 180.0 / np.pi) / angle_sens_along + angle_offset_along
                phase_athwart_deg = (phase_athwart * 180.0 / np.pi) / angle_sens_athwart + angle_offset_athwart

                std_along_deg = float(np.std(phase_along_deg))
                std_athwart_deg = float(np.std(phase_athwart_deg))

                if std_along_deg > phase_std_max or std_athwart_deg > phase_std_max:
                    n_rejected_phase += 1
                    continue

                angle_along_deg = float(np.mean(phase_along_deg))
                angle_athwart_deg = float(np.mean(phase_athwart_deg))

            compensation_db = compute_beam_compensation(
                angle_along=angle_along_deg,
                angle_athwart=angle_athwart_deg,
                bw_along=float(data["beamwidth_alongship"]),
                bw_athwart=float(data["beamwidth_athwartship"]),
                max_gain_db=mgc,
            )
            ts_compensated = peak_power_db + compensation_db

            if ts_compensated < ts_min:
                n_rejected_final_ts += 1
                continue

            detections.append(
                {
                    "ping_time": ping_times[ping_idx],
                    "ping_index": ping_idx,
                    "range_sample_index": peak_idx,
                    "range_m": range_val_m,
                    "angle_alongship_deg": angle_along_deg,
                    "angle_athwartship_deg": angle_athwart_deg,
                    "ts_uncompensated_db": peak_power_db,
                    "ts_compensated_db": float(ts_compensated),
                    "phase_std_alongship_deg": std_along_deg,
                    "phase_std_athwartship_deg": std_athwart_deg,
                    "normalized_pulse_width": float(normalized_width),
                    "compensation_db": float(compensation_db),
                    "threshold_level_passed": level,
                    "phase_gate_skipped": bool(phase_gate_skipped),
                }
            )

        if progress_callback is not None:
            progress_callback(ping_idx + 1, n_pings)

    detections_df = pd.DataFrame(detections)
    if not detections_df.empty:
        detections_df = detections_df.sort_values(["ping_index", "range_sample_index"]).reset_index(drop=True)

    diagnostics = {
        "n_candidates_after_amplitude": int(n_candidates_after_amplitude),
        "n_rejected_duration": int(n_rejected_duration),
        "n_rejected_phase": int(n_rejected_phase),
        "n_rejected_final_ts": int(n_rejected_final_ts),
        "n_rejected_depth": int(n_rejected_depth),
        "n_accepted": int(len(detections_df)),
        "n_phase_gate_skipped": int(n_phase_gate_skipped),
        "samples_per_pulse": float(pulse_dur / sample_spacing_s) if sample_spacing_s > 0 else np.nan,
    }
    return detections_df, diagnostics

