"""Raw EK80 loading and metadata extraction."""

from __future__ import annotations

from typing import Any, Dict, List

import echopype as ep
import numpy as np


def _first_value(arr: Any) -> float:
    """Safely extract first scalar from xarray/numpy containers."""
    values = np.asarray(arr)
    return float(values.reshape(-1)[0])


def _extract_sound_speed(ed: Any, beam: Any, ds_Sv: Any, ch_splitbeam: str) -> float:
    """
    Resolve sound speed from multiple common EK80/echopype locations.

    Priority:
    1) Beam_group1 per-channel variable
    2) Environment group (matches notebook workflow)
    3) Calibrated Sv dataset variables
    """
    for candidate in ("sound_speed_indicative", "sound_speed"):
        if candidate in beam.variables:
            da = beam[candidate]
            if "channel" in da.dims:
                da = da.sel(channel=ch_splitbeam)
            return _first_value(da)

    env = ed["Environment"] if "Environment" in ed.group_paths else None
    if env is not None:
        for candidate in ("sound_speed_indicative", "sound_speed"):
            if candidate in env.variables:
                da = env[candidate]
                if "channel" in da.dims and ch_splitbeam in env["channel"].values:
                    da = da.sel(channel=ch_splitbeam)
                return _first_value(da)

    for candidate in ("sound_speed_indicative", "sound_speed"):
        if candidate in ds_Sv.variables:
            da = ds_Sv[candidate]
            if "channel" in da.dims:
                da = da.sel(channel=ch_splitbeam)
            return _first_value(da)

    raise ValueError(
        "Could not find sound speed in file metadata (Beam_group1, Environment, or calibrated Sv dataset)."
    )


def _compute_sample_spacing_s(beam: Any, ds_Sv: Any, ch: str, sound_speed: float) -> float:
    # Use physical range spacing (meters), not raw sample index spacing.
    # On many EK80 files `range_sample` is just integer index and would
    # drastically overestimate sample time spacing.
    if "echo_range" in ds_Sv.coords:
        ranges = ds_Sv["echo_range"].sel(channel=ch).values.astype(float)
    elif "echo_range" in ds_Sv.variables:
        ranges = ds_Sv["echo_range"].sel(channel=ch).values.astype(float)
    elif "range_sample" in beam.coords:
        ranges = beam["range_sample"].values.astype(float)
    else:
        raise ValueError("Could not find range coordinate to compute sample spacing.")

    if ranges.size < 2:
        raise ValueError("Not enough range samples to compute sample spacing.")

    if np.ndim(ranges) == 1:
        sample_spacing_m = float(np.nanmedian(np.abs(np.diff(ranges))))
    else:
        sample_spacing_m = float(np.nanmedian(np.abs(np.diff(ranges, axis=-1))))
    if sample_spacing_m <= 0 or np.isnan(sample_spacing_m):
        raise ValueError("Computed invalid sample spacing from range coordinates.")
    return 2.0 * sample_spacing_m / sound_speed


def _extract_gain_db(ds_Sv: Any, ch: str) -> float:
    """
    Extract gain correction (dB) from calibrated Sv metadata.

    This is the file-provided baseline gain used by the TS conversion.
    """
    if "gain_correction" not in ds_Sv.variables:
        raise ValueError("Could not find `gain_correction` in calibrated Sv dataset.")
    da = ds_Sv["gain_correction"]
    if "channel" in da.dims:
        da = da.sel(channel=ch)
    return _first_value(da)


def _extract_frequency_hz(beam: Any, ch: str) -> float:
    """Extract nominal channel frequency in Hz."""
    if "frequency_nominal" not in beam.variables:
        raise ValueError("Could not find `frequency_nominal` in beam dataset.")
    return _first_value(beam["frequency_nominal"].sel(channel=ch))


def _extract_impedance_transceiver_ohm(ed: Any, beam: Any, ch: str) -> float:
    """
    Extract transceiver impedance (ohm), preferring beam metadata then Vendor_specific.
    """
    if "impedance_transceiver" in beam.variables:
        da = beam["impedance_transceiver"]
        if "channel" in da.dims:
            da = da.sel(channel=ch)
        return _first_value(da)

    if "Vendor_specific" in ed.group_paths:
        vendor = ed["Vendor_specific"]
        if "impedance_transceiver" in vendor.variables:
            da = vendor["impedance_transceiver"]
            if "channel" in da.dims:
                da = da.sel(channel=ch)
            return _first_value(da)

    # EK80/WBT default used in many files and Scott's XML.
    return 10800.0


def _extract_impedance_transducer_ohm(ed: Any, beam: Any, ch: str) -> float:
    """
    Extract transducer impedance (ohm) from available metadata.

    Many EK80 raw files do not expose this in the parsed xarray groups.
    Fall back to 75 ohm (used by Scott's calibration XML).
    """
    if "impedance_transducer" in beam.variables:
        da = beam["impedance_transducer"]
        if "channel" in da.dims:
            da = da.sel(channel=ch)
        return _first_value(da)

    if "Vendor_specific" in ed.group_paths:
        vendor = ed["Vendor_specific"]
        if "impedance_transducer" in vendor.variables:
            da = vendor["impedance_transducer"]
            if "channel" in da.dims:
                da = da.sel(channel=ch)
            return _first_value(da)

    return 75.0


def build_channel_data(base_data: Dict[str, Any], ch: str) -> Dict[str, Any]:
    """Create a channel-specific data dict for detection."""
    beam = base_data["beam"]
    ds_Sv = base_data["ds_Sv"]
    ed = base_data["ed"]

    if ch not in base_data["ch_all"]:
        raise ValueError(f"Channel not found in file: {ch}")

    tx_nom = beam["transmit_duration_nominal"].sel(channel=ch)
    pulse_duration_s = _first_value(tx_nom)
    sound_speed = _extract_sound_speed(ed, beam, ds_Sv, ch)
    beamwidth_alongship = _first_value(beam["beamwidth_twoway_alongship"].sel(channel=ch))
    beamwidth_athwartship = _first_value(beam["beamwidth_twoway_athwartship"].sel(channel=ch))
    angle_sensitivity_alongship = _first_value(beam["angle_sensitivity_alongship"].sel(channel=ch))
    angle_sensitivity_athwartship = _first_value(beam["angle_sensitivity_athwartship"].sel(channel=ch))
    angle_offset_alongship = _first_value(beam["angle_offset_alongship"].sel(channel=ch))
    angle_offset_athwartship = _first_value(beam["angle_offset_athwartship"].sel(channel=ch))
    sample_spacing_s = _compute_sample_spacing_s(beam, ds_Sv, ch, sound_speed)
    gain_db = _extract_gain_db(ds_Sv, ch)
    freq_hz = _extract_frequency_hz(beam, ch)
    impedance_transceiver_ohm = _extract_impedance_transceiver_ohm(ed, beam, ch)
    impedance_transducer_ohm = _extract_impedance_transducer_ohm(ed, beam, ch)

    out = dict(base_data)
    out.update(
        {
            "ch_splitbeam": ch,
            "pulse_duration_s": pulse_duration_s,
            "sound_speed": sound_speed,
            "beamwidth_alongship": beamwidth_alongship,
            "beamwidth_athwartship": beamwidth_athwartship,
            "angle_sensitivity_alongship": angle_sensitivity_alongship,
            "angle_sensitivity_athwartship": angle_sensitivity_athwartship,
            "angle_offset_alongship": angle_offset_alongship,
            "angle_offset_athwartship": angle_offset_athwartship,
            "sample_spacing_s": sample_spacing_s,
            "gain_db": gain_db,
            "gain_db_from_file": gain_db,
            "freq_hz": freq_hz,
            "impedance_transceiver_ohm": impedance_transceiver_ohm,
            "impedance_transducer_ohm": impedance_transducer_ohm,
        }
    )
    return out


def load_raw_file(filepath: str) -> Dict[str, Any]:
    """
    Load an EK80 raw file and extract split-beam metadata required for detection.
    """
    ed = ep.open_raw(filepath, sonar_model="EK80")
    ds_Sv = ep.calibrate.compute_Sv(ed, waveform_mode="CW", encode_mode="complex")
    ds_TS = ep.calibrate.compute_TS(ed, waveform_mode="CW", encode_mode="complex")
    beam = ed["Sonar/Beam_group1"]

    beam_type = beam["beam_type"].values
    channels = beam["channel"].values
    ch_all: List[str] = [str(ch) for ch in channels]

    split_indices = np.where(np.asarray(beam_type) == 17)[0]
    if split_indices.size == 0:
        raise ValueError(
            "No split-beam channel found in this file. Single target detection requires a split-beam transducer."
        )
    ch_splitbeam_all = [ch_all[int(i)] for i in split_indices]
    ch_splitbeam = ch_splitbeam_all[0]

    tx_nom = beam["transmit_duration_nominal"].sel(channel=ch_splitbeam)
    pulse_duration_s = _first_value(tx_nom)

    sound_speed = _extract_sound_speed(ed, beam, ds_Sv, ch_splitbeam)

    beamwidth_alongship = _first_value(beam["beamwidth_twoway_alongship"].sel(channel=ch_splitbeam))
    beamwidth_athwartship = _first_value(beam["beamwidth_twoway_athwartship"].sel(channel=ch_splitbeam))
    angle_sensitivity_alongship = _first_value(beam["angle_sensitivity_alongship"].sel(channel=ch_splitbeam))
    angle_sensitivity_athwartship = _first_value(beam["angle_sensitivity_athwartship"].sel(channel=ch_splitbeam))
    angle_offset_alongship = _first_value(beam["angle_offset_alongship"].sel(channel=ch_splitbeam))
    angle_offset_athwartship = _first_value(beam["angle_offset_athwartship"].sel(channel=ch_splitbeam))

    sample_spacing_s = _compute_sample_spacing_s(beam, ds_Sv, ch_splitbeam, sound_speed)
    gain_db = _extract_gain_db(ds_Sv, ch_splitbeam)
    freq_hz = _extract_frequency_hz(beam, ch_splitbeam)
    impedance_transceiver_ohm = _extract_impedance_transceiver_ohm(ed, beam, ch_splitbeam)
    impedance_transducer_ohm = _extract_impedance_transducer_ohm(ed, beam, ch_splitbeam)

    return {
        "ed": ed,
        "ds_Sv": ds_Sv,
        "ds_TS": ds_TS,
        "beam": beam,
        "ch_splitbeam": ch_splitbeam,
        "ch_splitbeam_all": ch_splitbeam_all,
        "ch_all": ch_all,
        "pulse_duration_s": pulse_duration_s,
        "sound_speed": sound_speed,
        "beamwidth_alongship": beamwidth_alongship,
        "beamwidth_athwartship": beamwidth_athwartship,
        "angle_sensitivity_alongship": angle_sensitivity_alongship,
        "angle_sensitivity_athwartship": angle_sensitivity_athwartship,
        "angle_offset_alongship": angle_offset_alongship,
        "angle_offset_athwartship": angle_offset_athwartship,
        "sample_spacing_s": sample_spacing_s,
        "gain_db": gain_db,
        "gain_db_from_file": gain_db,
        "freq_hz": freq_hz,
        "impedance_transceiver_ohm": impedance_transceiver_ohm,
        "impedance_transducer_ohm": impedance_transducer_ohm,
    }

