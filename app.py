"""Streamlit app for EK80 single target detection."""

from __future__ import annotations

import hashlib
import tempfile
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st

from detection.algorithm import detect_single_targets
from detection.loader import build_channel_data, load_raw_file
from detection.tracker import TRACKER_PARAM_DEFAULTS, assign_tracks, summarize_tracks
from viz.echogram import plot_echogram
from viz.histogram import plot_ts_histogram


st.set_page_config(layout="wide", page_title="EK80 Single Target Detection")
st.title("Hydroacoustic Single Target Detection (Soule et al. 1997)")


@st.cache_data(show_spinner=False)
def _load_file_cached(raw_bytes: bytes, filename: str):
    suffix = Path(filename).suffix or ".raw"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(raw_bytes)
        tmp_path = tmp.name
    return load_raw_file(tmp_path)


def _format_detection_table(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    float_cols = out.select_dtypes(include="float").columns
    out[float_cols] = out[float_cols].round(2)
    return out


def _get_range_bounds_m(data):
    ch = data["ch_splitbeam"]
    ds_sv = data["ds_Sv"]
    if "echo_range" in ds_sv.coords:
        arr = ds_sv["echo_range"].sel(channel=ch).values
    elif "echo_range" in ds_sv.variables:
        arr = ds_sv["echo_range"].sel(channel=ch).values
    else:
        arr = data["beam"]["range_sample"].values
    arr = arr.astype(float)
    return float(arr.min()), float(arr.max())


if "loaded_data" not in st.session_state:
    st.session_state.loaded_data = None
if "loaded_filename" not in st.session_state:
    st.session_state.loaded_filename = None
if "detections_df" not in st.session_state:
    st.session_state.detections_df = None
if "diagnostics" not in st.session_state:
    st.session_state.diagnostics = None
if "last_params" not in st.session_state:
    st.session_state.last_params = None
if "selected_channel" not in st.session_state:
    st.session_state.selected_channel = None
if "results_channel" not in st.session_state:
    st.session_state.results_channel = None
if "loaded_file_signature" not in st.session_state:
    st.session_state.loaded_file_signature = None
if "tracked_df" not in st.session_state:
    st.session_state.tracked_df = None
if "track_summary_df" not in st.session_state:
    st.session_state.track_summary_df = None

with st.sidebar:
    uploaded_file = st.file_uploader("Upload EK80 raw file", type=["raw"])

if uploaded_file is not None:
    raw_bytes = uploaded_file.getvalue()
    file_hash = hashlib.sha1(raw_bytes).hexdigest()[:12]
    file_signature = f"{uploaded_file.name}:{len(raw_bytes)}:{file_hash}"
    file_changed = st.session_state.loaded_file_signature != file_signature
    if file_changed or st.session_state.loaded_data is None:
        try:
            with st.spinner("Loading EK80 file..."):
                data = _load_file_cached(raw_bytes, uploaded_file.name)
            st.session_state.loaded_data = data
            st.session_state.loaded_filename = uploaded_file.name
            st.session_state.loaded_file_signature = file_signature
            st.session_state.detections_df = None
            st.session_state.diagnostics = None
            st.session_state.last_params = None
            st.session_state.results_channel = None
            st.session_state.tracked_df = None
            st.session_state.track_summary_df = None
            split_options = data.get("ch_splitbeam_all", [data["ch_splitbeam"]])
            st.session_state.selected_channel = split_options[0] if split_options else None
            st.success(f"Loaded: {uploaded_file.name}")
        except Exception as e:
            st.session_state.loaded_data = None
            st.error(f"Failed to load file: {e}")

data = st.session_state.loaded_data

with st.sidebar:
    if data is not None:
        split_options = data.get("ch_splitbeam_all", [data["ch_splitbeam"]])
        if split_options:
            if st.session_state.selected_channel not in split_options:
                st.session_state.selected_channel = split_options[0]
            st.selectbox(
                "Split-beam channel",
                options=split_options,
                key="selected_channel",
                help="Detection runs on this selected split-beam channel.",
            )
            if (
                st.session_state.results_channel is not None
                and st.session_state.selected_channel != st.session_state.results_channel
            ):
                st.session_state.detections_df = None
                st.session_state.diagnostics = None
                st.session_state.tracked_df = None
                st.session_state.track_summary_df = None

    st.subheader("Calibration")
    gain_offset_db = st.number_input(
        "TS Gain offset (dB)  [calibrated − file gain]",
        value=0.0,
        step=0.1,
        format="%.2f",
        help=(
            "Enter (calibrated gain) − (file gain). "
            "For Lake Superior: calibrated=18.46 dB, file=22.0 dB → enter −3.54. "
            "A negative value raises TS (corrects an over-estimated file gain). "
            "Leave at 0 if the file was already calibrated."
        ),
    )

    st.markdown("── Detection Parameters ──")
    with st.form("detection_form"):
        ts_min_db = st.number_input("TSmin (dB)", min_value=-80.0, max_value=-20.0, value=-60.0, step=1.0)
        max_gain_compensation_db = st.number_input(
            "Max gain comp (dB)", min_value=0.0, max_value=12.0, value=6.0, step=0.5
        )
        min_normalized_pulse_width = st.number_input(
            "Min pulse width",
            min_value=0.05,
            max_value=2.0,
            value=0.5,
            step=0.05,
            help=(
                "Echo duration (at −6 dB) divided by the nominal pulse duration. "
                "Note: Kongsberg/EV normalise by half the pulse duration, so their value of 0.80 equals 0.40 here. "
                "For ~4 samples/pulse (survey), 0.5 is practical. For ~32 samples/pulse (calibration), use ~0.4."
            ),
        )
        max_normalized_pulse_width = st.number_input(
            "Max pulse width",
            min_value=0.05,
            max_value=4.0,
            value=2.0,
            step=0.05,
            help=(
                "Echo duration (at −6 dB) divided by the nominal pulse duration. "
                "Note: Kongsberg/EV normalise by half the pulse duration, so their value of 1.80 equals 0.90 here. "
                "For ~4 samples/pulse (survey), 2.0 is practical. For ~32 samples/pulse (calibration), use ~0.9."
            ),
        )
        phase_std_max_deg = st.number_input(
            "Phase std max (deg)",
            min_value=0.1,
            max_value=5.0,
            value=0.537,
            step=0.001,
            format="%.3f",
            help="0.537 deg corresponds to 1 electrical phase step mapped to mechanical angle for ES38-18 geometry.",
        )
        st.markdown("── Temporary Depth Gate (diagnostic) ──")
        enable_depth_gate = st.checkbox(
            "Enable depth gate",
            value=False,
            help="Diagnostic filter to limit detections to a target depth window.",
        )
        min_range_m = st.number_input("Min analysis range (m)", min_value=0.0, value=0.0, step=1.0)
        max_range_m = st.number_input("Max analysis range (m)", min_value=1.0, value=500.0, step=1.0)
        run_detection = st.form_submit_button("Run Detection", type="primary")

    st.markdown("── Tracking Parameters ──")
    with st.form("tracking_form"):
        excl_dist_major_m = st.number_input(
            "Exclusion dist, major axis (m)",
            min_value=0.0,
            max_value=50.0,
            value=float(TRACKER_PARAM_DEFAULTS["excl_dist_major_m"]),
            step=0.5,
            help="Minimum gate half-width along the athwartship-derived major axis, before angular widening.",
        )
        excl_dist_minor_m = st.number_input(
            "Exclusion dist, minor axis (m)",
            min_value=0.0,
            max_value=50.0,
            value=float(TRACKER_PARAM_DEFAULTS["excl_dist_minor_m"]),
            step=0.5,
            help="Minimum gate half-width along the alongship-derived minor axis, before angular widening.",
        )
        excl_dist_range_m = st.number_input(
            "Exclusion dist, range axis (m)",
            min_value=0.0,
            max_value=50.0,
            value=float(TRACKER_PARAM_DEFAULTS["excl_dist_range_m"]),
            step=0.5,
            help="Gate half-width along the range axis.",
        )
        min_st_track = st.number_input(
            "Min detections per track",
            min_value=1,
            max_value=1000,
            value=int(TRACKER_PARAM_DEFAULTS["min_st_track"]),
            step=1,
            help="A candidate track is discarded unless it contains at least this many detections.",
        )
        min_pings_track = st.number_input(
            "Min ping span per track",
            min_value=1,
            max_value=1000,
            value=int(TRACKER_PARAM_DEFAULTS["min_pings_track"]),
            step=1,
            help="A candidate track is discarded unless it spans at least this many pings (last_ping - first_ping + 1).",
        )
        max_gap_track = st.number_input(
            "Max ping gap",
            min_value=1,
            max_value=100,
            value=int(TRACKER_PARAM_DEFAULTS["max_gap_track"]),
            step=1,
            help="Maximum number of consecutive missed pings before a track can no longer be extended.",
        )

        with st.expander("Advanced tracker parameters"):
            st.markdown("Alpha-beta filter gains")
            alpha_major = st.number_input(
                "alpha_major", min_value=0.0, max_value=1.0, value=float(TRACKER_PARAM_DEFAULTS["alpha_major"]), step=0.05
            )
            alpha_minor = st.number_input(
                "alpha_minor", min_value=0.0, max_value=1.0, value=float(TRACKER_PARAM_DEFAULTS["alpha_minor"]), step=0.05
            )
            alpha_range = st.number_input(
                "alpha_range", min_value=0.0, max_value=1.0, value=float(TRACKER_PARAM_DEFAULTS["alpha_range"]), step=0.05
            )
            beta_major = st.number_input(
                "beta_major", min_value=0.0, max_value=1.0, value=float(TRACKER_PARAM_DEFAULTS["beta_major"]), step=0.05
            )
            beta_minor = st.number_input(
                "beta_minor", min_value=0.0, max_value=1.0, value=float(TRACKER_PARAM_DEFAULTS["beta_minor"]), step=0.05
            )
            beta_range = st.number_input(
                "beta_range", min_value=0.0, max_value=1.0, value=float(TRACKER_PARAM_DEFAULTS["beta_range"]), step=0.05
            )

            st.markdown("Angular gate widening")
            max_std_major_deg = st.number_input(
                "max_std_major_deg",
                min_value=0.0,
                max_value=30.0,
                value=float(TRACKER_PARAM_DEFAULTS["max_std_major_deg"]),
                step=0.1,
            )
            max_std_minor_deg = st.number_input(
                "max_std_minor_deg",
                min_value=0.0,
                max_value=30.0,
                value=float(TRACKER_PARAM_DEFAULTS["max_std_minor_deg"]),
                step=0.1,
            )

            st.markdown("Missed-ping gate expansion (% per ping of gap)")
            missed_ping_exp_major_pct = st.number_input(
                "missed_ping_exp_major_pct",
                min_value=0.0,
                max_value=100.0,
                value=float(TRACKER_PARAM_DEFAULTS["missed_ping_exp_major_pct"]),
                step=1.0,
            )
            missed_ping_exp_minor_pct = st.number_input(
                "missed_ping_exp_minor_pct",
                min_value=0.0,
                max_value=100.0,
                value=float(TRACKER_PARAM_DEFAULTS["missed_ping_exp_minor_pct"]),
                step=1.0,
            )
            missed_ping_exp_range_pct = st.number_input(
                "missed_ping_exp_range_pct",
                min_value=0.0,
                max_value=100.0,
                value=float(TRACKER_PARAM_DEFAULTS["missed_ping_exp_range_pct"]),
                step=1.0,
            )

            st.markdown("Assignment cost weights")
            weight_major = st.number_input(
                "weight_major", min_value=0.0, max_value=1000.0, value=float(TRACKER_PARAM_DEFAULTS["weight_major"]), step=1.0
            )
            weight_minor = st.number_input(
                "weight_minor", min_value=0.0, max_value=1000.0, value=float(TRACKER_PARAM_DEFAULTS["weight_minor"]), step=1.0
            )
            weight_range = st.number_input(
                "weight_range", min_value=0.0, max_value=1000.0, value=float(TRACKER_PARAM_DEFAULTS["weight_range"]), step=1.0
            )
            weight_ts = st.number_input(
                "weight_ts", min_value=0.0, max_value=1000.0, value=float(TRACKER_PARAM_DEFAULTS["weight_ts"]), step=1.0
            )
            weight_ping_gap = st.number_input(
                "weight_ping_gap",
                min_value=0.0,
                max_value=1000.0,
                value=float(TRACKER_PARAM_DEFAULTS["weight_ping_gap"]),
                step=1.0,
            )
            delta_ts_max_db = st.number_input(
                "delta_ts_max_db",
                min_value=0.1,
                max_value=200.0,
                value=float(TRACKER_PARAM_DEFAULTS["delta_ts_max_db"]),
                step=1.0,
                help="TS difference (dB) that normalizes the assignment cost's TS term.",
            )

        st.markdown("── Equivalent Length (optional) ──")
        enable_equivalent_length = st.checkbox(
            "I have a TS-length regression for this species",
            value=False,
            help=(
                "Adds an equivalent_length_cm column to the track summary by inverting "
                "TS = a*log10(length_cm) + b. The a/b coefficients are species-specific "
                "published/measured constants -- do not guess these."
            ),
        )
        ts_length_a = st.number_input(
            "Regression coefficient a",
            value=0.0,
            step=0.1,
            format="%.3f",
            help="Slope 'a' in TS = a*log10(length_cm) + b, from a published/measured regression for this species. No default exists.",
        )
        ts_length_b = st.number_input(
            "Regression coefficient b",
            value=0.0,
            step=0.1,
            format="%.3f",
            help="Intercept 'b' in TS = a*log10(length_cm) + b, from a published/measured regression for this species. No default exists.",
        )
        run_tracking = st.form_submit_button("Run Tracking", type="primary")

if run_detection:
    if uploaded_file is None and data is None:
        st.error("Please upload a valid EK80 .raw file first.")
    else:
        params = {
            "ts_min_db": float(ts_min_db),
            "max_gain_compensation_db": float(max_gain_compensation_db),
            "min_normalized_pulse_width": float(min_normalized_pulse_width),
            "max_normalized_pulse_width": float(max_normalized_pulse_width),
            "phase_std_max_deg": float(phase_std_max_deg),
        }
        if enable_depth_gate:
            params["min_range_m"] = float(min_range_m)
            params["max_range_m"] = float(max_range_m)
        try:
            data = st.session_state.loaded_data
            if data is None:
                st.error("Please upload a valid EK80 .raw file first.")
                st.stop()
            if not st.session_state.selected_channel:
                st.error("No split-beam channel available for detection.")
                st.stop()
            data = build_channel_data(data, st.session_state.selected_channel)
            data["gain_db"] = float(data["gain_db_from_file"]) + float(gain_offset_db)
        except Exception as e:
            st.error(f"Failed to prepare selected channel: {e}")
            data = None

        if data is None:
            st.stop()

        if enable_depth_gate:
            data_min_m, data_max_m = _get_range_bounds_m(data)
            if max_range_m < data_min_m or min_range_m > data_max_m:
                st.warning(
                    f"Depth gate [{min_range_m:.1f}, {max_range_m:.1f}] m is outside data range "
                    f"[{data_min_m:.1f}, {data_max_m:.1f}] m; all candidates will be rejected."
                )
            elif min_range_m >= max_range_m:
                st.error("Depth gate invalid: Min analysis range must be less than Max analysis range.")
                st.stop()

        progress = st.progress(0, text="Running detection...")
        try:
            def _cb(done, total):
                if total <= 0:
                    progress.progress(0)
                else:
                    progress.progress(min(int(done / total * 100), 100))

            detections_df, diagnostics = detect_single_targets(data, params, progress_callback=_cb)
            progress.progress(100, text="Detection complete.")
            st.session_state.detections_df = detections_df
            st.session_state.diagnostics = diagnostics
            st.session_state.last_params = params
            st.session_state.results_channel = st.session_state.selected_channel
            st.session_state.tracked_df = None
            st.session_state.track_summary_df = None

            if detections_df.empty:
                st.warning(
                    "No single targets detected with current parameters. Try relaxing TSmin or increasing Phase std max."
                )
        except Exception as e:
            st.error("Unexpected error during detection.")
            st.exception(e)
        finally:
            progress.empty()

if run_tracking:
    if st.session_state.detections_df is None or st.session_state.detections_df.empty:
        st.error("Run detection first -- tracking requires a non-empty set of detections.")
    else:
        tracker_params = {
            "excl_dist_major_m": float(excl_dist_major_m),
            "excl_dist_minor_m": float(excl_dist_minor_m),
            "excl_dist_range_m": float(excl_dist_range_m),
            "min_st_track": int(min_st_track),
            "min_pings_track": int(min_pings_track),
            "max_gap_track": int(max_gap_track),
            "alpha_major": float(alpha_major),
            "alpha_minor": float(alpha_minor),
            "alpha_range": float(alpha_range),
            "beta_major": float(beta_major),
            "beta_minor": float(beta_minor),
            "beta_range": float(beta_range),
            "max_std_major_deg": float(max_std_major_deg),
            "max_std_minor_deg": float(max_std_minor_deg),
            "missed_ping_exp_major_pct": float(missed_ping_exp_major_pct),
            "missed_ping_exp_minor_pct": float(missed_ping_exp_minor_pct),
            "missed_ping_exp_range_pct": float(missed_ping_exp_range_pct),
            "weight_major": float(weight_major),
            "weight_minor": float(weight_minor),
            "weight_range": float(weight_range),
            "weight_ts": float(weight_ts),
            "weight_ping_gap": float(weight_ping_gap),
            "delta_ts_max_db": float(delta_ts_max_db),
        }
        try:
            with st.spinner("Running tracking..."):
                tracked_df = assign_tracks(st.session_state.detections_df, tracker_params)
                ts_a = float(ts_length_a) if enable_equivalent_length else None
                ts_b = float(ts_length_b) if enable_equivalent_length else None
                track_summary_df = summarize_tracks(tracked_df, ts_length_a=ts_a, ts_length_b=ts_b)
            st.session_state.tracked_df = tracked_df
            st.session_state.track_summary_df = track_summary_df

            if track_summary_df.empty:
                st.warning(
                    "No tracks met the acceptance criteria (min detections / min ping span). "
                    "Try relaxing those, or the gate widths, and re-run tracking."
                )
        except Exception as e:
            st.error("Unexpected error during tracking.")
            st.exception(e)

with st.sidebar:
    if data is not None and st.session_state.diagnostics is not None:
        d = st.session_state.diagnostics
        st.markdown("── Diagnostics ──")
        st.text(f"Candidates found:     {d['n_candidates_after_amplitude']}")
        st.text(f"Rejected (duration):  {d['n_rejected_duration']}")
        st.text(f"Rejected (phase):     {d['n_rejected_phase']}")
        st.text(f"Rejected (final TS):  {d['n_rejected_final_ts']}")
        if "n_rejected_depth" in d:
            st.text(f"Rejected (depth):     {d['n_rejected_depth']}")
        st.text("──────────────────────────")
        st.text(f"Accepted targets:     {d['n_accepted']}")
        if d["n_phase_gate_skipped"] > 0:
            st.warning(
                f"{d['n_phase_gate_skipped']} detections had <3 samples in -6dB window\n"
                "(phase gate bypassed — pulse duration marginal)"
            )

    if data is not None:
        view_data = data
        try:
            if st.session_state.selected_channel:
                view_data = build_channel_data(data, st.session_state.selected_channel)
        except Exception:
            view_data = data

        beam = view_data["beam"]
        ch = view_data["ch_splitbeam"]
        freq_hz = beam["frequency_nominal"].sel(channel=ch).values.reshape(-1)[0]
        n_pings = int(beam["ping_time"].size)
        n_range = int(beam["range_sample"].size)
        st.markdown("── File Info ──")
        st.text(f"Channel:      {ch}")
        st.text(f"Waveform:     {view_data.get('waveform_mode', 'CW')}")
        st.text(f"Frequency:    {freq_hz / 1000:.0f} kHz")
        st.text(f"Pulse dur:    {view_data['pulse_duration_s'] * 1e3:.3f} ms")
        if st.session_state.diagnostics is not None:
            st.text(f"Samples/pulse: {st.session_state.diagnostics['samples_per_pulse']:.1f}")
        else:
            st.text("Samples/pulse: --")
        st.text(f"Sound speed:  {view_data['sound_speed']:.1f} m/s")
        st.text(f"Pings:        {n_pings}")
        st.text(f"Range samples: {n_range}")
        range_min_m, range_max_m = _get_range_bounds_m(view_data)
        st.text(f"Range extent: {range_min_m:.1f}–{range_max_m:.1f} m")

tab1, tab2, tab3, tab4 = st.tabs(["Echogram", "TS Distribution", "Detection Table", "Tracks"])

with tab1:
    if data is None:
        st.info("Upload a .raw file to view echogram and detections.")
    else:
        ch_plot = st.session_state.selected_channel or data["ch_splitbeam"]
        if st.session_state.tracked_df is not None and not st.session_state.tracked_df.empty:
            detections_for_plot = st.session_state.tracked_df
        elif st.session_state.detections_df is not None:
            detections_for_plot = st.session_state.detections_df
        else:
            detections_for_plot = pd.DataFrame()

        fig_sv = plot_echogram(
            data["ds_Sv"],
            detections_for_plot,
            ch_plot,
            value_var="Sv",
            title=f"{st.session_state.loaded_filename or 'Echogram'} | Sv",
        )
        st.plotly_chart(fig_sv, use_container_width=True)

        if "ds_TS" in data and data["ds_TS"] is not None:
            fig_ts = plot_echogram(
                data["ds_TS"],
                detections_for_plot,
                ch_plot,
                value_var="TS",
                title=f"{st.session_state.loaded_filename or 'Echogram'} | TS",
            )
            st.plotly_chart(fig_ts, use_container_width=True)
        else:
            st.info("TS echogram unavailable for this file load.")

        if st.session_state.tracked_df is not None and not st.session_state.tracked_df.empty:
            st.info(
                "Top panel: Sv echogram. Bottom panel: TS echogram. "
                "Colored connected paths indicate accepted tracks; white circles are "
                "unassigned/untracked detections."
            )
        else:
            st.info(
                "Top panel: Sv echogram. Bottom panel: TS echogram. "
                "White circles indicate accepted single-target detections."
            )

with tab2:
    df = st.session_state.detections_df
    ts_min_for_plot = (
        float(st.session_state.last_params["ts_min_db"])
        if st.session_state.last_params is not None
        else float(ts_min_db)
    )
    if df is None:
        st.info("Run detection to view TS distribution.")
    else:
        fig = plot_ts_histogram(df, ts_min_db=ts_min_for_plot)
        st.plotly_chart(fig, use_container_width=True)
        c1, c2, c3, c4 = st.columns(4)
        if df.empty:
            c1.metric("Mean TS", "N/A")
            c2.metric("Median TS", "N/A")
            c3.metric("Std TS", "N/A")
            c4.metric("Count", "0")
        else:
            c1.metric("Mean TS", f"{df['ts_compensated_db'].mean():.2f} dB")
            c2.metric("Median TS", f"{df['ts_compensated_db'].median():.2f} dB")
            c3.metric("Std TS", f"{df['ts_compensated_db'].std(ddof=0):.2f} dB")
            c4.metric("Count", f"{len(df)}")

with tab3:
    df = st.session_state.detections_df
    if df is None:
        st.info("Run detection to view detection table.")
    else:
        formatted_df = _format_detection_table(df)
        st.dataframe(formatted_df, use_container_width=True, hide_index=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        stem = Path(st.session_state.loaded_filename or "file").stem
        csv_name = f"detections_{stem}_{ts}.csv"
        st.download_button(
            "Download CSV",
            data=formatted_df.to_csv(index=False).encode("utf-8"),
            file_name=csv_name,
            mime="text/csv",
        )

with tab4:
    track_summary_df = st.session_state.track_summary_df
    if track_summary_df is None:
        st.info("Run tracking to view track summaries.")
    else:
        formatted_tracks_df = _format_detection_table(track_summary_df)
        st.dataframe(formatted_tracks_df, use_container_width=True, hide_index=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        stem = Path(st.session_state.loaded_filename or "file").stem
        tracks_csv_name = f"tracks_{stem}_{ts}.csv"
        st.download_button(
            "Download CSV",
            data=formatted_tracks_df.to_csv(index=False).encode("utf-8"),
            file_name=tracks_csv_name,
            mime="text/csv",
        )

