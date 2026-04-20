"""Compare single-target pipeline output against EK80 calibration XML TargetHits."""

from __future__ import annotations

import argparse
import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from statistics import mean, pstdev
from typing import Dict, List, Optional

import numpy as np

from detection.algorithm import detect_single_targets
from detection.loader import build_channel_data, load_raw_file


@dataclass
class XmlStats:
    n_hits: int
    ts_comp_mean: float
    ts_comp_std: float
    ts_uncomp_mean: float
    ts_uncomp_std: float
    comp_term_mean: float
    range_mean: float
    range_std: float
    range_min: float
    range_max: float
    target_reference_ts: float


@dataclass
class AppStats:
    n_hits: int
    ts_comp_mean: float
    ts_comp_std: float
    ts_uncomp_mean: float
    ts_uncomp_std: float
    comp_term_mean: float
    range_mean: float
    range_std: float
    range_min: float
    range_max: float


def _parse_xml_stats(xml_path: str) -> XmlStats:
    root = ET.parse(xml_path).getroot()
    hits = root.findall(".//TargetHits/HitData")

    ts_comp = [float(h.findtext("TsComp")) for h in hits]
    ts_un = [float(h.findtext("TsUncomp")) for h in hits]
    ranges = [float(h.findtext("Range")) for h in hits]
    comp_term = [c - u for c, u in zip(ts_comp, ts_un)]
    target_reference = float(root.findtext(".//TargetReference/ResponseAtCWCenterFrequency"))

    return XmlStats(
        n_hits=len(ts_comp),
        ts_comp_mean=mean(ts_comp),
        ts_comp_std=pstdev(ts_comp),
        ts_uncomp_mean=mean(ts_un),
        ts_uncomp_std=pstdev(ts_un),
        comp_term_mean=mean(comp_term),
        range_mean=mean(ranges),
        range_std=pstdev(ranges),
        range_min=min(ranges),
        range_max=max(ranges),
        target_reference_ts=target_reference,
    )


def _build_params(args: argparse.Namespace) -> Dict[str, float]:
    return {
        "ts_min_db": float(args.ts_min_db),
        "max_gain_compensation_db": float(args.max_gain_comp_db),
        "min_normalized_pulse_width": float(args.min_npw),
        "max_normalized_pulse_width": float(args.max_npw),
        "phase_std_max_deg": float(args.phase_std_max_deg),
        "min_range_m": float(args.min_range_m),
        "max_range_m": float(args.max_range_m),
    }


def _app_stats_or_none(detections) -> Optional[AppStats]:
    if detections.empty:
        return None
    comp_term = detections["ts_compensated_db"] - detections["ts_uncompensated_db"]
    return AppStats(
        n_hits=int(len(detections)),
        ts_comp_mean=float(detections["ts_compensated_db"].mean()),
        ts_comp_std=float(detections["ts_compensated_db"].std(ddof=0)),
        ts_uncomp_mean=float(detections["ts_uncompensated_db"].mean()),
        ts_uncomp_std=float(detections["ts_uncompensated_db"].std(ddof=0)),
        comp_term_mean=float(comp_term.mean()),
        range_mean=float(detections["range_m"].mean()),
        range_std=float(detections["range_m"].std(ddof=0)),
        range_min=float(detections["range_m"].min()),
        range_max=float(detections["range_m"].max()),
    )


def _compute_suggested_gain_offset_db(xml_stats: XmlStats, app_stats: AppStats) -> float:
    # TS includes -2*Gain, so to correct TsUncomp mean mismatch:
    # delta_gain = -(xml_uncomp - app_uncomp)/2
    return -(xml_stats.ts_uncomp_mean - app_stats.ts_uncomp_mean) / 2.0


def _dominant_rejection_stage(diagnostics: Dict[str, float]) -> str:
    stages = {
        "depth": float(diagnostics.get("n_rejected_depth", 0)),
        "duration": float(diagnostics.get("n_rejected_duration", 0)),
        "phase": float(diagnostics.get("n_rejected_phase", 0)),
        "final_ts": float(diagnostics.get("n_rejected_final_ts", 0)),
    }
    return max(stages, key=stages.get)


def _print_stats_block(title: str, stats) -> None:
    print(title)
    print(f"n_hits:                {stats.n_hits}")
    print(f"mean TsComp (dB):      {stats.ts_comp_mean:.3f}  std: {stats.ts_comp_std:.3f}")
    print(f"mean TsUncomp (dB):    {stats.ts_uncomp_mean:.3f}  std: {stats.ts_uncomp_std:.3f}")
    print(f"mean comp term (dB):   {stats.comp_term_mean:.3f}")
    print(
        f"range (m):             mean {stats.range_mean:.3f}  std {stats.range_std:.3f}  "
        f"min {stats.range_min:.3f}  max {stats.range_max:.3f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify calibration pipeline against EK80 XML TargetHits.")
    parser.add_argument("--raw", required=True, help="Path to EK80 .raw file")
    parser.add_argument("--xml", required=True, help="Path to EK80 CalibrationDataFile XML")
    parser.add_argument("--gain-offset-db", type=float, default=0.0, help="Gain offset added to file gain")
    parser.add_argument("--ts-min-db", type=float, default=-56.0)
    parser.add_argument("--max-gain-comp-db", type=float, default=6.0)
    parser.add_argument("--min-npw", type=float, default=0.5)
    parser.add_argument("--max-npw", type=float, default=2.0)
    parser.add_argument("--phase-std-max-deg", type=float, default=20.0)
    parser.add_argument("--min-range-m", type=float, default=9.6)
    parser.add_argument("--max-range-m", type=float, default=10.2)
    parser.add_argument("--hit-count-tol", type=int, default=60)
    parser.add_argument("--ts-mean-tol-db", type=float, default=0.75)
    parser.add_argument("--comp-mean-tol-db", type=float, default=0.5)
    args = parser.parse_args()

    xml_stats = _parse_xml_stats(args.xml)
    params = _build_params(args)

    base = load_raw_file(args.raw)
    data = build_channel_data(base, base["ch_splitbeam"])
    data["gain_db"] = float(data["gain_db_from_file"]) + float(args.gain_offset_db)

    detections, diagnostics = detect_single_targets(data, params)
    app_stats = _app_stats_or_none(detections)

    print("=== XML Reference ===")
    _print_stats_block("", xml_stats)
    print(f"target reference (dB): {xml_stats.target_reference_ts:.3f}")
    print()
    print("=== App Output ===")
    print(f"gain from file (dB):   {float(data['gain_db_from_file']):.3f}")
    print(f"gain offset (dB):      {float(args.gain_offset_db):.3f}")
    print(f"effective gain (dB):   {float(data['gain_db']):.3f}")
    print("params:")
    for k in (
        "ts_min_db",
        "max_gain_compensation_db",
        "min_normalized_pulse_width",
        "max_normalized_pulse_width",
        "phase_std_max_deg",
        "min_range_m",
        "max_range_m",
    ):
        print(f"  - {k}: {params[k]}")
    print()

    if app_stats is None:
        print("No detections with current settings.")
    else:
        _print_stats_block("", app_stats)
        print()
        print("=== Mismatch vs XML ===")
        print(f"hit count delta:       {app_stats.n_hits - xml_stats.n_hits:+d}")
        print(f"mean TsComp delta:     {app_stats.ts_comp_mean - xml_stats.ts_comp_mean:+.3f} dB")
        print(f"mean TsUncomp delta:   {app_stats.ts_uncomp_mean - xml_stats.ts_uncomp_mean:+.3f} dB")
        print(f"mean comp term delta:  {app_stats.comp_term_mean - xml_stats.comp_term_mean:+.3f} dB")
        print(f"range mean delta:      {app_stats.range_mean - xml_stats.range_mean:+.3f} m")

        suggested_gain_offset_db = _compute_suggested_gain_offset_db(xml_stats, app_stats)
        print()
        print("=== Suggested Gain Offset ===")
        print(f"suggested offset (dB): {suggested_gain_offset_db:+.3f} (CalTSGain - OrigTSGain)")

        hit_count_pass = abs(app_stats.n_hits - xml_stats.n_hits) <= int(args.hit_count_tol)
        ts_mean_pass = abs(app_stats.ts_comp_mean - xml_stats.ts_comp_mean) <= float(args.ts_mean_tol_db)
        comp_mean_pass = abs(app_stats.comp_term_mean - xml_stats.comp_term_mean) <= float(args.comp_mean_tol_db)

        print()
        print("=== Pass/Fail ===")
        print(f"hit count (±{args.hit_count_tol}):         {'PASS' if hit_count_pass else 'FAIL'}")
        print(f"mean TsComp (±{args.ts_mean_tol_db} dB):   {'PASS' if ts_mean_pass else 'FAIL'}")
        print(f"mean comp term (±{args.comp_mean_tol_db}): {'PASS' if comp_mean_pass else 'FAIL'}")

    print()
    print("=== Diagnostics ===")
    for k, v in diagnostics.items():
        print(f"{k}: {v}")
    print(f"dominant rejection stage: {_dominant_rejection_stage(diagnostics)}")


if __name__ == "__main__":
    main()

