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
- EK80 `.raw` with at least one split-beam channel (`beam_type` in `{1, 17, 49, 65, 81}`)

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

### Step 1: Pre-compute split-beam angles

Once per file, before the detection loop:
- `get_splitbeam_angles(ed, ds_Sv, ch_splitbeam)` is called
- Internally calls `ep.consolidate.add_splitbeam_angle()` (EchoPype's validated decoder)
- Returns `angle_along`, `angle_athwart` as `(ping, range)` arrays in mechanical degrees
- Angle offsets are baked in when calibration overrides are passed to `compute_Sv`

This replaces per-ping IQ cross-product computation. The EchoPype decoder handles all transducer geometries (`beam_type` 1, 17, 49, 65, 81) including the 3-sector ES38-18 combi transducer.

### Step 2: Get uncompensated TS baseline

The detector does **not** re-derive TS from scratch using custom constants.
It uses EK80-calibrated TS from `echopype`:

- `ds_TS = ep.calibrate.compute_TS(ed, waveform_mode="CW", encode_mode="complex")`
- per-sample baseline is `ds_TS["TS"]`

Then user gain offset is applied in TS space:
- `TS_adjusted = TS_from_echopype - 2 * gain_offset_db`

Why `-2`? Because TS equation gain term is `-2G`.

### Step 3: Pre-smooth the signal, then find candidate echo peaks

#### 3a — Signal smoothing before peak detection (Stage 1)

Before searching for peaks, the TS trace for each ping is gently smoothed by
averaging each sample with the one immediately before it. This is done in the
linear power domain (not in dB) to avoid distortion, then converted back to dB.
The result is a 2-sample causal running average — "causal" means it only uses
the current and immediately preceding sample, so it cannot introduce echoes from
the future.

**Why this matters.** With only ~4 samples per echo (for a 0.256 ms pulse at
Lake Superior sampling rates), a single noisy sample can create a spurious peak
or shift the true peak by one sample. The smoothed trace is less sensitive to
these single-sample fluctuations and matches the pre-smoothing step in the ESP3
MATLAB reference implementation.

**Important:** the smoothed trace is used only for finding peak locations. All
subsequent measurements — TS value, pulse width, phase angles — are read from
the original unsmoothed data. Smoothing is purely a peak-locator aid.

#### 3b — Amplitude thresholds

Three acceptance tiers are defined from `TSmin` and `mgc` (max gain compensation):

- `th1 = TSmin − (2·mgc + 6)` — most permissive floor; screens out obvious noise
- `th2 = TSmin − (2·mgc + 6)/2` — intermediate
- `th3 = TSmin` — strictest; candidate already passes the final TS test

Each peak is labelled with the strictest tier it clears. All peaks go through
the remaining gates regardless of tier — tier is recorded for diagnostics only.

#### 3c — Minimum separation between peaks (Stage 2)

Two peaks must be separated by at least one full pulse length in samples —
computed as `round(pulse_duration / sample_spacing)` rather than the previous
fixed value of 2 samples.

**Why this matters.** A single fish echo spans roughly one pulse length. If
two peaks can sit only 2 samples apart but a pulse is ~4 samples wide, both
peaks may originate from the same fish. With the minimum separation set to one
pulse length, the detector is prevented from splitting a single fish echo into
two candidates. This is the primary safeguard against duplicate detections and
matches the `MinSeparation = Np` logic in the ESP3 reference.

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

Angles for the echo window are looked up from the pre-computed arrays:
- `window_along   = angle_along[ping_idx, window_idx]`
- `window_athwart = angle_athwart[ping_idx, window_idx]`

NaN values (can occur at range edges in EchoPype output) are removed:
- `valid = np.isfinite(window_along) & np.isfinite(window_athwart)`

If `n_valid < 3`, phase gate is skipped for that candidate (tracked in diagnostics) and `np.nanmean` is used for the angle estimate.

Otherwise:
1. Compute std dev on each axis using only the valid samples, with the
   **N−1 denominator** (Bessel's correction, also called `ddof=1`).
2. Reject candidate if either std exceeds `phase_std_max_deg`.
3. Use `np.mean` of valid samples as the angle estimate.

Conceptually:
- low phase std → stable single target (the echo came from one direction throughout)
- high phase std → mixed or unstable return (possibly two fish close together, or noise)

**Why N−1 rather than N?** When the standard deviation is computed over a small
number of samples — typically 3–4 for a short pulse — the N denominator
systematically underestimates the true variability. The N−1 correction (Bessel's
correction) gives a better estimate and matches the behaviour of MATLAB's
`nanstd`, which the ESP3 reference uses. In practice, for a 4-sample window, the
N−1 std is about 15% larger than the N std, making the phase gate marginally
stricter. For longer windows the difference becomes negligible.

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

### Step 9: Within-ping overlap deduplication (Stage 6)

After all gates have passed, one final check runs across all detections collected
from the same ping before they are saved.

**What it does.** For every pair of detections within the same ping, the code
checks whether their echo windows (the contiguous samples above −6 dB of each
peak) overlap in range. If they do, only the detection with the higher
compensated TS is kept; the other is discarded.

**Why this is necessary.** Even with the minimum peak separation set to one
pulse length (Step 3c), two peaks can occasionally survive from the same fish
echo — for example when a noise spike sits just outside the separation gate, or
when the smoothed and unsmoothed peaks are slightly offset. Without this step,
both detections enter the output CSV and appear to the downstream tracker as two
separate fish at almost identical depth in the same ping. The tracker then either
creates a phantom short track for the weaker duplicate, or fragments a genuine
track. Keeping only the stronger of any overlapping pair removes this artefact
before it propagates.

**Order of deduplication.** Detections within a ping are ranked by compensated
TS from highest to lowest. The strongest detection is always kept. Each
subsequent detection is only kept if its window does not overlap with any already
accepted detection from that ping.

### Step 10: Output row and diagnostics

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
4. picks split-beam channel via `get_splitbeam_channel(ed)` — selects by `beam_type` from `{1, 17, 49, 65, 81}`, prefers lowest frequency if multiple are present
5. extracts channel metadata:
   - pulse duration
   - sound speed (Beam -> Environment -> Sv fallback)
   - two-way beamwidths
   - angle sensitivities and offsets
   - gain correction
   - frequency
   - impedances (with defaults when missing)
6. computes sample spacing from physical range spacing (`echo_range` preferred)

`build_channel_data(base_data, ch)` applies this for a selected channel.

`get_splitbeam_angles(ed, ds_Sv, ch_splitbeam)` computes decoded split-beam angles once per file using `ep.consolidate.add_splitbeam_angle()`. Called at the start of `detect_single_targets()` before the ping loop. Handles both complex (EK80 CW/BB) and power/angle (EK60, EK80 power mode) data paths.

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
