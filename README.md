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

## Exact Algorithm Implemented (Detailed Walkthrough)

This section is written for non-programmers who want to verify the exact sequence of decisions the app makes in `detection/algorithm.py`.

### Big picture

For every ping, the app:

1. finds local echo peaks (possible targets),
2. removes peaks that do not look like a single-echo pulse in width,
3. removes peaks whose phase is unstable (likely overlapping targets),
4. applies beam compensation,
5. keeps only peaks above final TS threshold.

Every rejection stage is counted in Diagnostics so you can see where candidates are being filtered out.

---

### Step 1: Read channel data and build complex signal

The algorithm runs on one selected split-beam channel.

For that channel, EK80 provides:

- real part: `backscatter_r`
- imaginary part: `backscatter_i`

These are combined into a complex signal per sample:

- `iq = backscatter_r + i * backscatter_i`

Think of this as each sample having:

- **amplitude** (signal strength),
- **phase** (direction-related information used by split-beam).

The code also reads:

- pulse duration (`pulse_duration_s`),
- sample time spacing (`sample_spacing_s`),
- beamwidth alongship/athwartship,
- ping times,
- range coordinate in meters (`echo_range`, when available).

---

### Step 2: Convert signal to power for candidate screening

At each ping and range sample, the app computes a single power value from the split-beam sectors:

- sector power is `|iq|^2` per sector,
- then it sums those sector powers (incoherent sum),
- then converts to dB:
  - `power_db = 10*log10(power_linear + tiny_number)`

Why this matters:

- This produces the amplitude trace used to find candidate peaks.
- It does not yet decide if a peak is a single fish target.

---

### Step 3: Multi-threshold amplitude screening (Soule-style candidate pass)

Using user parameters:

- `TSmin` (`ts_min`)
- max compensation (`mgc`)

the algorithm computes 3 thresholds:

- `th1 = ts_min - (2*mgc + 6)` -> most permissive (lowest)
- `th2 = ts_min - (2*mgc + 6)/2` -> medium
- `th3 = ts_min` -> strictest (highest)

For each ping, local peaks are found with:

- minimum peak height = `th1`,
- minimum spacing between peaks = 2 samples.

Each peak is assigned the strictest tier it passes:

- pass `th3` -> tier 3
- else pass `th2` -> tier 2
- else pass `th1` -> tier 1

Diagnostic effect:

- all such peaks are counted in `Candidates found` (`n_candidates_after_amplitude`).

---

### Step 4: Optional temporary depth gate

If the depth gate is enabled in the UI:

- reject candidate if it is shallower than `Min analysis range`,
- reject candidate if it is deeper than `Max analysis range`.

Diagnostic effect:

- `Rejected (depth)` (`n_rejected_depth`) increases.

This gate is currently a diagnostic helper, not a full bottom-tracking model.

---

### Step 5: Duration gate using the -6 dB window

For each remaining candidate:

1. Identify the peak value (`peak_power_db`).
2. Move left and right from the peak while samples stay above `peak - 6 dB`.
3. This contiguous set is the candidate echo window.

Then compute:

- `n_samples_in_window`
- `actual_duration_s = n_samples_in_window * sample_spacing_s`
- `normalized_pulse_width = actual_duration_s / pulse_duration_s`

Pass rule:

- keep only if
  - `min_normalized_pulse_width <= normalized_pulse_width <= max_normalized_pulse_width`

Diagnostic effect:

- failures increase `Rejected (duration)`.

Interpretation:

- too narrow: likely noise spike,
- too wide: likely extended/overlapping/non-single return.

---

### Step 6: Phase stability gate (core single-vs-overlap discriminator)

If window has fewer than 3 samples:

- phase gate is skipped (firmware-like behavior),
- candidate is allowed to continue,
- `phase_gate_skipped=True`.

If window has 3+ samples:

1. Compute electrical phase-difference series:
   - alongship from sectors 0 and 1
   - athwartship from sector 2 vs average of sectors 0 and 1
2. Convert those phase series to mechanical-angle units using beamwidth factors.
3. Compute standard deviation of each converted series.
4. Reject candidate if either std exceeds `phase_std_max_deg`.

Diagnostic effect:

- failures increase `Rejected (phase)`,
- skipped short-window cases increase `n_phase_gate_skipped`.

Interpretation:

- stable phase across samples -> more like single target,
- unstable phase -> more like mixed/overlapping echoes.

---

### Step 7: Compute mean target angle

For candidates that pass phase gate:

- mean alongship angle = mean of converted alongship phase series,
- mean athwartship angle = mean of converted athwartship phase series.

These angles are used for beam compensation.

---

### Step 8: Beam compensation

The app uses a Simrad-style two-way compensation:

- `compensation_db = 6.0206 * ((angle_along/bw_along)^2 + (angle_athwart/bw_athwart)^2)`
- capped at `Max gain comp`.

Then:

- `ts_compensated_db = ts_uncompensated_db + compensation_db`

---

### Step 9: Final TS acceptance

Final keep/reject rule:

- keep only if `ts_compensated_db >= TSmin`

Diagnostic effect:

- failures increase `Rejected (final TS)`.

---

### Step 10: Save accepted detections

Each accepted detection is written as one row with:

- ping/time/range position,
- alongship/athwartship angle,
- uncompensated and compensated TS,
- phase std values,
- normalized pulse width,
- compensation amount,
- threshold tier passed,
- whether phase gate was skipped.

Diagnostics summarize how many candidates were removed at each stage:

- `Candidates found`
- `Rejected (depth)` (if enabled)
- `Rejected (duration)`
- `Rejected (phase)`
- `Rejected (final TS)`
- `Accepted targets`
- `n_phase_gate_skipped`
- `samples_per_pulse`

---

### Candidate flow equation (quick check)

You can verify counts with:

`Accepted = Candidates - Rejected(depth) - Rejected(duration) - Rejected(phase) - Rejected(final TS)`

Because filters run in that order, this should hold (allowing only for formatting/rounding display differences).

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
