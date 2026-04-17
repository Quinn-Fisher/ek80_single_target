# EK80 Single Target Detection (Streamlit)

This app performs hydroacoustic single-target detection from Kongsberg EK80 `.raw` data using a Soule et al. (1997)-style phase stability workflow for split-beam transducers.

It is implemented as a Streamlit UI with:

- EK80 file loading and Sv calibration via `echopype`
- Candidate screening and phase/discrimination logic
- Interactive Plotly echogram + TS histogram
- Detection table export to CSV

---

## Project Structure

```text
single_target/
├── app.py
├── detection/
│   ├── __init__.py
│   ├── loader.py
│   ├── algorithm.py
│   └── compensation.py
├── viz/
│   ├── __init__.py
│   ├── echogram.py
│   └── histogram.py
└── requirements.txt
```

---

## Requirements

- Python 3.10+ recommended (3.11 typically easiest for dependency wheels)
- EK80 `.raw` files with at least one split-beam channel (`beam_type == 17`)

Install deps:

```bash
pip install -r requirements.txt
```

---

## Run the App

From the `single_target` directory:

```bash
streamlit run app.py
```

Then in the browser:

1. Upload an EK80 `.raw` file in the sidebar (this loads metadata/calibration and discovers channels).
2. Select the split-beam channel to analyze.
3. Set detection parameters.
4. Click **Run Detection**.
4. Review:
   - **Echogram** tab
   - **TS Distribution** tab
   - **Detection Table** tab (with CSV export)

Important behavior:

- Upload loads file metadata and channel options only (not target detection).
- Detection runs only when you click **Run Detection**.
- Parameter/channel edits alone do not trigger detection.

---

## Sidebar Controls

### Detection parameters

- `TSmin (dB)`: minimum compensated target strength threshold
- `Max gain comp (dB)`: cap for beam compensation
- `Min pulse width`: minimum normalized pulse width
- `Max pulse width`: maximum normalized pulse width
- `Phase std max (deg)`: maximum allowed phase std (mechanical-angle units)

### Temporary depth gate (diagnostic)

- `Enable depth gate`
- `Min analysis range (m)`
- `Max analysis range (m)`

This is a debugging/analysis filter to isolate depth bands.

---

## Outputs

### Diagnostics panel

Displays:

- `Candidates found`
- `Rejected (duration)`
- `Rejected (phase)`
- `Rejected (final TS)`
- `Rejected (depth)` (when depth gate is enabled)
- `Accepted targets`
- warning count for detections with `<3` samples in `-6 dB` window (phase gate bypassed)

### File info panel

Displays:

- Channel
- Frequency
- Pulse duration
- Samples per pulse
- Sound speed
- Ping count
- Range sample count
- Range extent (m)

### Detection table columns

- `ping_time`
- `ping_index`
- `range_sample_index`
- `range_m`
- `angle_alongship_deg`
- `angle_athwartship_deg`
- `ts_uncompensated_db`
- `ts_compensated_db`
- `phase_std_alongship_deg`
- `phase_std_athwartship_deg`
- `normalized_pulse_width`
- `compensation_db`
- `threshold_level_passed`
- `phase_gate_skipped`

---

## Exact Algorithm Implemented (Step-by-Step)

This section describes the current implementation in `detection/algorithm.py` exactly as coded.

### 1) Load required data arrays

From the split-beam channel:

- `backscatter_r`, `backscatter_i` from `Beam_group1`
- Form complex IQ:
  - `iq = backscatter_r + 1j * backscatter_i`

Auxiliary metadata:

- pulse duration (`pulse_duration_s`)
- sample spacing in seconds (`sample_spacing_s`) from loader
- beamwidths alongship/athwartship (degrees)
- ping times
- physical range coordinate (`echo_range`) when available

### 2) Power proxy for amplitude screening

The app uses **incoherent total sector power**:

- `power_linear = sum(|iq|^2 over beam sector dimension)`
- `power_db = 10*log10(power_linear + 1e-20)`

This avoids phase-cancellation artifacts that would occur with coherent sector summation.

### 3) Multi-threshold amplitude candidate screening

Given:

- `ts_min = TSmin`
- `mgc = max_gain_compensation_db`

Thresholds:

- `th1 = ts_min - (2*mgc + 6)` (most permissive)
- `th2 = ts_min - (2*mgc + 6)/2`
- `th3 = ts_min` (strictest)

Peak extraction per ping:

- `find_peaks(row, height=th1, distance=2)`
  - `height=th1` removes sub-th1 local maxima
  - `distance=2` reduces duplicate nearby picks

Tier assignment:

- if `v >= th3` => level `3`
- elif `v >= th2` => level `2`
- elif `v >= th1` => level `1`

The stored `threshold_level_passed` is the strictest tier passed.

### 4) Optional depth gate (temporary diagnostic)

If enabled:

- reject candidate when `range_m < min_range_m`
- reject candidate when `range_m > max_range_m`

Counter: `n_rejected_depth`

### 5) Duration gate from -6 dB window

For each remaining candidate:

1. Find contiguous sample window around peak where power stays within `peak_db - 6 dB`.
2. Compute:
   - `n_samples_in_window`
   - `actual_duration_s = n_samples_in_window * sample_spacing_s`
   - `normalized_width = actual_duration_s / pulse_duration_s`
3. Keep only if:
   - `min_normalized_pulse_width <= normalized_width <= max_normalized_pulse_width`

Counter: `n_rejected_duration`

### 6) Phase stability gate (Soule-style discriminator)

If `n_samples_in_window < 3`:

- bypass phase gate (accepted through this stage)
- mark `phase_gate_skipped = True`
- increment `n_phase_gate_skipped`

Else compute electrical phase-difference series:

- `phase_along_rad = angle(iq0 * conj(iq1))`
- `phase_athwart_rad = angle(iq2 * conj((iq0+iq1)/2))`

Convert electrical phase to mechanical-angle units:

- `angle_factor_along = beamwidth_alongship_deg / (2*pi)`
- `angle_factor_athwart = beamwidth_athwartship_deg / (2*pi)`
- `phase_along_deg = phase_along_rad * angle_factor_along`
- `phase_athwart_deg = phase_athwart_rad * angle_factor_athwart`

Phase gate:

- `std_along_deg = std(phase_along_deg)`
- `std_athwart_deg = std(phase_athwart_deg)`
- reject if either std exceeds `phase_std_max_deg`

Counter: `n_rejected_phase`

### 7) Mean angle per accepted candidate

For phase-gated candidates (or skipped phase gate):

- `angle_alongship_deg = mean(phase_along_deg)` (or `0.0` if phase skipped)
- `angle_athwartship_deg = mean(phase_athwart_deg)` (or `0.0` if phase skipped)

### 8) Beam compensation

Implemented in `detection/compensation.py`:

- `compensation_db = 6.0206 * ((angle_along/bw_along)^2 + (angle_athwart/bw_athwart)^2)`
- cap at `max_gain_compensation_db`

### 9) Final TS gate

- `ts_compensated_db = peak_power_db + compensation_db`
- keep candidate only if `ts_compensated_db >= ts_min`

Counter: `n_rejected_final_ts`

### 10) Output assembly

Accepted candidates become rows in the output DataFrame with all fields listed above.

Diagnostics dictionary includes:

- `n_candidates_after_amplitude`
- `n_rejected_duration`
- `n_rejected_phase`
- `n_rejected_final_ts`
- `n_rejected_depth` (if depth gate used)
- `n_accepted`
- `n_phase_gate_skipped`
- `samples_per_pulse = pulse_duration_s / sample_spacing_s`

---

## Loader Details (`detection/loader.py`)

`load_raw_file(filepath)` does:

1. `ep.open_raw(..., sonar_model="EK80")`
2. `ep.calibrate.compute_Sv(..., waveform_mode="CW", encode_mode="complex")`
3. Finds split-beam channel (`beam_type == 17`) or raises error
4. Extracts:
   - pulse duration
   - sound speed (searched in Beam, Environment, then Sv dataset)
   - beamwidths
   - list of split-beam channels (`ch_splitbeam_all`)
5. Computes `sample_spacing_s` from **physical range spacing** (`echo_range` preferred), not index spacing

`build_channel_data(base_data, ch)` then prepares channel-specific metadata
for whichever split-beam channel the user selected in the UI.

---

## Visualization Details

### Echogram (`viz/echogram.py`)

- Plotly `Heatmap` of Sv
- detection overlay as white markers with black outline
- dark theme
- y-axis uses physical `echo_range` when available (including 2D handling)

### Histogram (`viz/histogram.py`)

- Plotly histogram of `ts_compensated_db`
- 1 dB bins
- TSmin dashed reference line

---

## Troubleshooting

- **No split-beam channel found**
  - File may not contain split-beam data; algorithm requires split-beam transducer channel.

- **All rejected by depth**
  - Depth gate likely outside actual file range. Check `Range extent` in sidebar.

- **All rejected by duration**
  - Relax pulse width bounds (especially for short pulses / low samples-per-pulse).

- **All rejected by phase**
  - Increase `Phase std max (deg)` gradually (e.g., `0.237` -> `0.5` -> `1.0` -> `2.0`).

- **All rejected by final TS**
  - Lower `TSmin` or inspect whether echoes in selected depth band are weak.

---

## Notes

- This implementation is intended as a transparent, inspectable workflow for analysis and iteration.
- The temporary depth gate is diagnostic and may be replaced later with per-ping bottom-aware exclusion.
