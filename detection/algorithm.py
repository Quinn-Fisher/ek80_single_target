"""Soule et al. (1997) single target detection algorithm for EK80 split-beam IQ data."""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.signal import find_peaks

from detection.compensation import compute_beam_compensation


def _window_within_minus_6db(power_row: np.ndarray, peak_idx: int, peak_val: float) -> np.ndarray:
    threshold = peak_val - 6.0
    left = peak_idx
    right = peak_idx

    while left - 1 >= 0 and power_row[left - 1] >= threshold:
        left -= 1
    while right + 1 < power_row.size and power_row[right + 1] >= threshold:
        right += 1
    return np.arange(left, right + 1, dtype=int)


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

    power_linear = np.mean(np.abs(iq) ** 2, axis=-1)
    power_db = 10.0 * np.log10(power_linear + 1e-20)

    ping_times = beam.ping_time.values
    if "echo_range" in data["ds_Sv"].coords:
        range_m_arr = data["ds_Sv"]["echo_range"].sel(channel=ch).values.astype(float)
    elif "echo_range" in data["ds_Sv"].variables:
        range_m_arr = data["ds_Sv"]["echo_range"].sel(channel=ch).values.astype(float)
    else:
        range_m_arr = beam.range_sample.values.astype(float)

    mgc = float(params["max_gain_compensation_db"])
    ts_min = float(params["ts_min_db"])
    th1 = ts_min - (2 * mgc + 6)
    th2 = ts_min - ((2 * mgc + 6) / 2.0)
    th3 = ts_min

    min_npw = float(params["min_normalized_pulse_width"])
    max_npw = float(params["max_normalized_pulse_width"])
    phase_std_max = float(params["phase_std_max_deg"])

    n_pings = power_db.shape[0]
    detections: List[Dict[str, Any]] = []

    n_candidates_after_amplitude = 0
    n_rejected_duration = 0
    n_rejected_phase = 0
    n_rejected_final_ts = 0
    n_phase_gate_skipped = 0

    angle_factor_along = float(data["beamwidth_alongship"]) / (2.0 * np.pi)
    angle_factor_athwart = float(data["beamwidth_athwartship"]) / (2.0 * np.pi)

    for ping_idx in range(n_pings):
        row = power_db[ping_idx, :]
        peaks, _ = find_peaks(row)

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

                std_along_deg = float(np.std(phase_along) * (180.0 / np.pi))
                std_athwart_deg = float(np.std(phase_athwart) * (180.0 / np.pi))

                if std_along_deg > phase_std_max or std_athwart_deg > phase_std_max:
                    n_rejected_phase += 1
                    continue

                mean_phase_along = float(np.mean(phase_along))
                mean_phase_athwart = float(np.mean(phase_athwart))
                # Beamwidths are already in degrees, so factor yields deg/rad.
                angle_along_deg = mean_phase_along * angle_factor_along
                angle_athwart_deg = mean_phase_athwart * angle_factor_athwart

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
                    "range_m": float(range_m_arr[peak_idx] if np.ndim(range_m_arr) == 1 else range_m_arr[ping_idx, peak_idx]),
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
        "n_accepted": int(len(detections_df)),
        "n_phase_gate_skipped": int(n_phase_gate_skipped),
        "samples_per_pulse": float(pulse_dur / sample_spacing_s) if sample_spacing_s > 0 else np.nan,
    }
    return detections_df, diagnostics

