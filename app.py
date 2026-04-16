"""Streamlit app for EK80 single target detection."""

from __future__ import annotations

import tempfile
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st

from detection.algorithm import detect_single_targets
from detection.loader import load_raw_file
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

with st.sidebar:
    uploaded_file = st.file_uploader("Upload EK80 raw file", type=["raw"])
    st.markdown("── Detection Parameters ──")
    with st.form("detection_form"):
        ts_min_db = st.number_input("TSmin (dB)", min_value=-80.0, max_value=-20.0, value=-60.0, step=1.0)
        max_gain_compensation_db = st.number_input(
            "Max gain comp (dB)", min_value=0.0, max_value=12.0, value=6.0, step=0.5
        )
        min_normalized_pulse_width = st.number_input(
            "Min pulse width", min_value=0.5, max_value=1.0, value=0.8, step=0.05
        )
        max_normalized_pulse_width = st.number_input(
            "Max pulse width", min_value=1.0, max_value=3.0, value=1.5, step=0.05
        )
        phase_std_max_deg = st.number_input(
            "Phase std max (deg)",
            min_value=0.1,
            max_value=5.0,
            value=0.237,
            step=0.001,
            format="%.3f",
            help="0.237 deg corresponds to 1 electrical phase step mapped to mechanical angle for ES38-18 geometry.",
        )
        run_detection = st.form_submit_button("Run Detection", type="primary")

data = st.session_state.loaded_data

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
        try:
            if uploaded_file is not None:
                file_changed = st.session_state.loaded_filename != uploaded_file.name
                if file_changed or st.session_state.loaded_data is None:
                    with st.spinner("Loading EK80 file..."):
                        data = _load_file_cached(uploaded_file.getvalue(), uploaded_file.name)
                    st.session_state.loaded_data = data
                    st.session_state.loaded_filename = uploaded_file.name
                    st.session_state.detections_df = None
                    st.session_state.diagnostics = None
                else:
                    data = st.session_state.loaded_data
            else:
                data = st.session_state.loaded_data
        except Exception as e:
            st.session_state.loaded_data = None
            st.error(f"Failed to load file: {e}")
            data = None

        if data is None:
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

            if detections_df.empty:
                st.warning(
                    "No single targets detected with current parameters. Try relaxing TSmin or increasing Phase std max."
                )
        except Exception as e:
            st.error("Unexpected error during detection.")
            st.exception(e)
        finally:
            progress.empty()

with st.sidebar:
    if data is not None and st.session_state.diagnostics is not None:
        d = st.session_state.diagnostics
        st.markdown("── Diagnostics ──")
        st.text(f"Candidates found:     {d['n_candidates_after_amplitude']}")
        st.text(f"Rejected (duration):  {d['n_rejected_duration']}")
        st.text(f"Rejected (phase):     {d['n_rejected_phase']}")
        st.text(f"Rejected (final TS):  {d['n_rejected_final_ts']}")
        st.text("──────────────────────────")
        st.text(f"Accepted targets:     {d['n_accepted']}")
        if d["n_phase_gate_skipped"] > 0:
            st.warning(
                f"{d['n_phase_gate_skipped']} detections had <3 samples in -6dB window\n"
                "(phase gate bypassed — pulse duration marginal)"
            )

    if data is not None:
        beam = data["beam"]
        ch = data["ch_splitbeam"]
        freq_hz = beam["frequency_nominal"].sel(channel=ch).values.reshape(-1)[0]
        n_pings = int(beam["ping_time"].size)
        n_range = int(beam["range_sample"].size)
        st.markdown("── File Info ──")
        st.text(f"Channel:      {ch}")
        st.text(f"Frequency:    {freq_hz / 1000:.0f} kHz")
        st.text(f"Pulse dur:    {data['pulse_duration_s'] * 1e3:.3f} ms")
        if st.session_state.diagnostics is not None:
            st.text(f"Samples/pulse: {st.session_state.diagnostics['samples_per_pulse']:.1f}")
        else:
            st.text("Samples/pulse: --")
        st.text(f"Sound speed:  {data['sound_speed']:.1f} m/s")
        st.text(f"Pings:        {n_pings}")
        st.text(f"Range samples: {n_range}")

tab1, tab2, tab3 = st.tabs(["Echogram", "TS Distribution", "Detection Table"])

with tab1:
    if data is None:
        st.info("Upload a .raw file to view echogram and detections.")
    else:
        fig = plot_echogram(
            data["ds_Sv"],
            st.session_state.detections_df if st.session_state.detections_df is not None else pd.DataFrame(),
            data["ch_splitbeam"],
            title=st.session_state.loaded_filename or "Echogram",
        )
        st.plotly_chart(fig, use_container_width=True)
        st.info("White circles indicate accepted single target detections. Hover for TS and angle values.")

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

