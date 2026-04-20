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

## Exact Detection Pipeline (Hydroacoustics-Focused)

Implemented in `detection/algorithm.py`.

This section is intentionally detailed for non-programmers. Think of the detector as a sequence of "keep/reject" tests applied to every potential echo peak.

### Symbols and terms used below

- `TS`: target strength in dB re 1 m^2
- `TSmin`: final compensated TS threshold
- `mgc`: max gain compensation in dB
- `R`: range in meters
- "alongship / athwartship": split-beam angle axes
- "window": contiguous samples around a peak above `peak - 6 dB`

### Step 1: Build complex split-beam signal

For each ping and sample:
- `iq = backscatter_r + 1j * backscatter_i`

This preserves both:
- amplitude (echo strength)
- phase (used for split-beam angle stability tests)

### Step 2: Get uncompensated TS baseline

The detector does **not** re-derive TS from scratch using custom constants.
It uses EK80-calibrated TS from `echopype`:

- `ds_TS = ep.calibrate.compute_TS(ed, waveform_mode="CW", encode_mode="complex")`
- per-sample baseline is `ds_TS["TS"]`

Then user gain offset is applied in TS space:
- `TS_adjusted = TS_from_echopype - 2 * gain_offset_db`

Why `-2`? Because TS equation gain term is `-2G`.

### Step 3: Build three candidate thresholds

Using `TSmin` and `mgc`, the code defines:

- `th1 = TSmin - (2*mgc + 6)` (most permissive)
- `th2 = TSmin - ((2*mgc + 6)/2)` (intermediate)
- `th3 = TSmin` (strictest)

Then local peaks are found on the uncompensated TS trace:
- peak height must be at least `th1`
- peaks must be separated by at least 2 samples

Each peak is labeled by strongest threshold passed (tier 1/2/3).

### Step 4: Optional analysis range gate

If enabled in UI, candidate is rejected outside:
- `[min_range_m, max_range_m]`

This is a temporary diagnostic range window, not a bottom detector.

### Step 5: Pulse-width shape gate

For each surviving peak:
1. Find contiguous samples above `peak - 6 dB`.
2. Count samples in that window (`n_samples`).
3. Compute:
   - `actual_duration_s = n_samples * sample_spacing_s`
   - `normalized_pulse_width = actual_duration_s / pulse_duration_s`

Candidate passes only if:
- `min_normalized_pulse_width <= normalized_pulse_width <= max_normalized_pulse_width`

Interpretation:
- too narrow -> often impulsive/noise spike
- too wide -> often multiple/extended/overlapping returns

### Step 6: Phase stability gate (single-target discriminator)

If `n_samples < 3`, phase gate is skipped for that candidate (tracked in diagnostics).

Otherwise:
1. Compute electrical phase-difference series:
   - alongship: sector `0` vs `1`
   - athwartship: sector `2` vs average of `0` and `1`
2. Convert phase to mechanical angle with EK80 channel calibration:
   - `angle_deg = (phase_rad * 180/pi) / angle_sensitivity + angle_offset`
3. Compute std dev on each axis.
4. Reject candidate if either std exceeds `phase_std_max_deg`.

Conceptually:
- low phase std -> stable single target
- high phase std -> mixed or unstable return

### Step 7: Beam compensation

For accepted angle estimates, compensation is:

- `comp_db = 6.0206 * ((along/(bw_along/2))^2 + (athwart/(bw_athwart/2))^2)`
- clipped at `max_gain_compensation_db`

Important convention:
- EK80 metadata provides two-way beamwidth
- equation denominator uses one-way width, so code uses `bw/2`

### Step 8: Final TS acceptance

Compensated TS is:
- `TScomp = TSuncomp + comp_db`

Final pass condition:
- `TScomp >= TSmin`

### Step 9: Output row and diagnostics

Each accepted target stores:
- ping/time/range
- along/athwart angle
- uncompensated and compensated TS
- phase std values
- normalized pulse width
- compensation
- threshold tier
- phase-gate-skipped flag

Diagnostics store counts for:
- candidates found
- rejected by depth/range gate
- rejected by duration gate
- rejected by phase gate
- rejected by final TS
- accepted targets

Accounting identity:
- `Accepted = Candidates - Rejected(depth) - Rejected(duration) - Rejected(phase) - Rejected(final TS)`

### Practical interpretation of diagnostics

- large `Rejected(depth)` -> range gate is likely too narrow or misplaced
- large `Rejected(duration)` -> pulse-width bounds likely too strict
- large `Rejected(phase)` -> phase std gate likely too strict for current SNR/echo shape
- large `Rejected(final TS)` -> `TSmin` likely too high for target population

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
