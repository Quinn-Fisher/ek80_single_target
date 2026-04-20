# EK80 Single Target Detection (Streamlit)

This app performs hydroacoustic split-beam single-target detection from EK80 `.raw` data using a Soule et al. (1997)-style workflow, with calibration behavior tuned for transparent EK80 validation.

It provides:
- EK80 loading via `echopype`
- calibrated `Sv` and `TS` products
- single-target detection with stage-by-stage diagnostics
- side-by-side `Sv` and `TS` echograms with detection overlays
- target table export and calibration verification script

---

## Project Structure

```text
single_target/
├── app.py
├── detection/
│   ├── loader.py
│   ├── algorithm.py
│   └── compensation.py
├── viz/
│   ├── echogram.py
│   └── histogram.py
├── verify_calibration.py
└── README.md
```

---

## Requirements

- Python 3.10+ (3.11/3.12 works with current dependencies)
- EK80 `.raw` with at least one split-beam channel (`beam_type == 17`)

Install:

```bash
pip install -r requirements.txt
```

---

## Run the App

From `single_target/`:

```bash
streamlit run app.py
```

Workflow:
1. Upload EK80 `.raw`
2. Select split-beam channel
3. Set parameters
4. Click **Run Detection**
5. Review:
   - **Echogram** tab (Sv panel + TS panel)
   - **TS Distribution**
   - **Detection Table**

Notes:
- Upload only loads data/metadata; detection runs only on button click.
- Changing channel or parameters does not auto-run detection.

---

## Calibration and Gain Offset

Sidebar calibration input:
- `TS Gain offset (dB) [CalTSGain - OrigTSGain]`

App behavior:
- `effective_gain_db = gain_db_from_file + gain_offset_db`
- `TS_adjusted = TS_from_echopype - 2 * gain_offset_db`

Interpretation:
- positive offset -> TS decreases by `2 * offset`
- negative offset -> TS increases by `2 * abs(offset)`
- `0.0` -> use file gain as-is

This follows TS equation gain sign (`-2*G`).

---

## Detection Parameters

- `TSmin (dB)`: final compensated TS threshold
- `Max gain comp (dB)`: cap on beam compensation
- `Min pulse width`: lower bound on normalized pulse width
- `Max pulse width`: upper bound on normalized pulse width
- `Phase std max (deg)`: gate on phase-derived mechanical angle std (default `0.237`)
- Optional temporary depth gate:
  - `Min analysis range (m)`
  - `Max analysis range (m)`

---

## Exact Detection Pipeline

Implemented in `detection/algorithm.py`.

1. **Build IQ**
   - `iq = backscatter_r + 1j * backscatter_i`

2. **Uncompensated TS source**
   - Uses `ds_TS["TS"]` from `ep.calibrate.compute_TS(..., waveform_mode="CW", encode_mode="complex")`
   - Applies user gain offset in TS space (`-2 * gain_offset_db`)

3. **Amplitude candidate thresholds**
   - `th1 = ts_min - (2*mgc + 6)`
   - `th2 = ts_min - ((2*mgc + 6)/2)`
   - `th3 = ts_min`
   - Peaks from `find_peaks(row, height=th1, distance=2)`

4. **Optional depth gate**
   - reject outside `[min_range_m, max_range_m]`

5. **Pulse-width gate**
   - contiguous `-6 dB` window around peak
   - `normalized_width = (n_samples * sample_spacing_s) / pulse_duration_s`
   - keep only within `[min_npw, max_npw]`

6. **Phase stability gate**
   - if fewer than 3 samples in window: skip phase gate for that candidate
   - alongship phase: sector 0 vs 1
   - athwartship phase: sector 2 vs average(0,1)
   - convert electrical phase to mechanical angle with EK80 metadata:
     - `angle = (phase_rad * 180/pi) / angle_sensitivity + angle_offset`
   - reject if std of either axis exceeds `phase_std_max_deg`

7. **Beam compensation**
   - `comp = 6.0206 * ((along/(bw_along/2))^2 + (athwart/(bw_athwart/2))^2)`
   - `bw_*` are EK80 two-way metadata and converted to one-way in denominator
   - capped at `max_gain_compensation_db`

8. **Final TS gate**
   - `ts_compensated = ts_uncompensated + comp`
   - keep if `ts_compensated >= ts_min`

9. **Outputs**
   - detection table with range, angles, TS, compensation, pulse width, phase std, tier, skip flag
   - diagnostics counters at each rejection stage

Candidate accounting:
`Accepted = Candidates - Rejected(depth) - Rejected(duration) - Rejected(phase) - Rejected(final TS)`

---

## Loader Behavior (`detection/loader.py`)

`load_raw_file(filepath)`:
1. `ep.open_raw(..., sonar_model="EK80")`
2. `ep.calibrate.compute_Sv(..., waveform_mode="CW", encode_mode="complex")`
3. `ep.calibrate.compute_TS(..., waveform_mode="CW", encode_mode="complex")`
4. picks split-beam channel(s)
5. extracts channel metadata:
   - pulse duration
   - sound speed (Beam -> Environment -> Sv fallback)
   - two-way beamwidths
   - angle sensitivities and offsets
   - gain correction
   - frequency
   - impedances (with defaults when missing)
6. computes sample spacing from physical range spacing (`echo_range` preferred)

`build_channel_data(base_data, ch)` applies this for selected channel.

---

## Visualization

### Echogram (`viz/echogram.py`)

Echogram tab now shows both:
- **top panel:** `Sv` heatmap (`ds_Sv["Sv"]`)
- **bottom panel:** `TS` heatmap (`ds_TS["TS"]`)

Both support:
- overlay of accepted detections
- physical range axis from `echo_range` (1D or median profile for 2D range grids)

### Histogram (`viz/histogram.py`)

- histogram of `ts_compensated_db`
- 1 dB bins
- TSmin reference line

---

## Calibration Verification Script

Use `verify_calibration.py` to compare app output with EK80 calibration XML `<TargetHits>`.

Example:

```bash
python verify_calibration.py \
  --raw /path/to/calibration.raw \
  --xml /path/to/CalibrationDataFile.xml
```

The script prints:
- XML baseline stats (`TsComp`, `TsUncomp`, compensation term, range, hit count)
- app output stats for current settings
- mismatch deltas
- pass/fail against configurable tolerances
- suggested gain offset (`CalTSGain - OrigTSGain`)
- dominant rejection stage from diagnostics

Defaults are intentionally calibration-oriented. Adjust args as needed for survey/fish runs.

---

## Changes vs Public Repo

Relative to public `main` at [Quinn-Fisher/ek80_single_target](https://github.com/Quinn-Fisher/ek80_single_target):

- uncompensated TS now sourced from `echopype.compute_TS` rather than ad hoc power-only scaling
- explicit gain-offset semantics applied in TS domain (`-2*offset`)
- phase-angle conversion now uses EK80 angle sensitivity/offset metadata
- beam compensation uses one-way beamwidth denominator derived from EK80 two-way fields
- Echogram tab now shows both `Sv` and `TS` panels
- added/expanded `verify_calibration.py` for XML-vs-app reproducible calibration checks

---

## Troubleshooting

- **No split-beam channel found**
  - File lacks split-beam data.

- **No detections**
  - Check diagnostics to identify dominant rejection stage.

- **All rejected by depth**
  - Depth gate outside data range.

- **All rejected by duration**
  - Relax pulse-width bounds.

- **All rejected by phase**
  - Increase `Phase std max (deg)` carefully from default as needed.

- **All rejected by final TS**
  - Lower `TSmin` or inspect whether detected echoes are weak for selected window.

---

## Notes

- This implementation is designed to be inspectable and calibration-auditable.
- Depth gate remains a diagnostic tool, not bottom-track logic.
- For inter-software parity, compare distributions and diagnostics, not only hit counts.
