# Technical Pipeline Snapshot - 2026-05-23

This document describes the current ECG QRS + HRV pipeline as of 2026-05-23
(Australia/Sydney), anchored to branch `main` at HEAD `fa8bd7d` when this file
was written. It is a dated snapshot, not a rolling description of future
versions.

## Scope

The pipeline is rule-based and does not use a QRS detection toolbox, an external
ECG detector, record-specific rules, or a trained black-box model. It processes
MATLAB `.mat` ECG records sampled at 100 Hz, detects QRS sample indices, derives
windowed HRV metrics, and supports submission `.mat` filling and visualization.

Main implementation files:

- `qrs_pipeline.py`: filtering, QRS candidate detection, noise gating, final peak cleanup, debug layers.
- `test.py`: training-set evaluation, QRS matching, HRV calculation, CSV/plot export.
- `qrs_debug_viewer.py`: interactive inspection of detector layers.
- `processor.py`: fills `ProjectTestDataAnalysis.mat` with QRS and HRV outputs.
- `visualize_submission.py`: validates and plots the filled test submission.

Historical and experimental scripts that are not part of the active pipeline
are kept under `archive/`.

## Data And Indexing

Input data is read from MATLAB `.mat` files using `scipy.io.loadmat`.

- Training data: `dataSet/ProjectTrainData.mat`.
- Test ECG data: `dataSet/ProjectTestData.mat`.
- Test submission template: `dataSet/ProjectTestDataAnalysis.mat`.
- Sampling rate: `DEFAULT_FS = 100`, so 1 sample is 10 ms.

Python internally uses zero-based sample indices. MATLAB submission arrays use
one-based indices, so `processor.py` adds 1 when writing QRS peaks back into the
submission `.mat`.

## QRS Detection

The detector has a main ECG-amplitude branch, an auxiliary QRS-energy branch,
and two conservative cleanup stages.

### 1. Broad ECG Preprocessing

`preprocess_ecg(raw)` converts the raw ECG to a broad filtered signal used for
main detection, plotting, and final peak placement:

1. Convert the input to a flat floating-point array.
2. Apply a 50 Hz notch filter with `iirnotch(50, 8, fs=100)`.
3. Apply a second-order 1 Hz high-pass Butterworth filter with zero-phase SOS filtering.
4. Apply a second-order 40 Hz low-pass Butterworth filter with zero-phase SOS filtering.

This path preserves ECG morphology better than the narrower QRS-energy branch.

### 2. QRS-Band Energy Signal

`qrs_bandpass(raw)` creates a 5-20 Hz filtered signal:

1. Apply the same 50 Hz notch filter.
2. Apply a second-order 5-20 Hz Butterworth band-pass filter.

This signal emphasizes steep QRS complexes and suppresses slower P/T-wave
components.

### 3. Main Candidate Detection

The default main detector uses `abs(filtered ECG)`, so it can detect both
upright and inverted QRS complexes.

Parameters:

- `MAIN_PERCENTILE = 97`.
- `MAIN_REFRACTORY_SEC = 0.30`.

Implementation:

1. Compute `strength = abs(filtered)`.
2. Set the peak threshold to the 97th percentile of `strength`.
3. Use `scipy.signal.find_peaks` with a 0.30 s minimum distance.

This creates the main candidate set.

### 4. Energy Candidate Detection

The energy branch follows the Pan-Tompkins/XQRS-style idea of derivative energy
and moving-window integration. It is not the main detector; it is used to rescue
missed beats in unusually long RR gaps.

Parameters:

- `ENERGY_REFRACTORY_SEC = 0.25`.
- `REFINE_SEC = 0.08`.
- Moving integration window: 0.15 s.
- Threshold: median energy + `3.5 * MAD`.

Implementation:

1. Differentiate the 5-20 Hz QRS-band signal.
2. Square the derivative.
3. Smooth it with a 0.15 s moving average.
4. Detect peaks above the adaptive median/MAD threshold.
5. Refine each energy peak to the strongest nearby absolute QRS-band sample
   within +/-0.08 s.

### 5. Long-Gap Filling

`_fill_long_gaps` uses energy candidates only when the main detector leaves an
unusually long gap.

Parameters:

- `GAP_FACTOR = 1.55`.
- `MIN_GAP_SEC = 0.45`.
- Energy candidates must stay at least 0.25 s away from the neighboring main peaks.

Implementation:

1. Compute RR intervals from main peaks.
2. Estimate typical RR from intervals between 0.25 s and 2.0 s.
3. Define the gap limit as `max(0.45 s, 1.55 * typical RR)`.
4. Add energy candidates inside gaps longer than this limit.
5. Merge close peaks after adding rescued candidates.

### 6. Signal-Quality Mask

`_quality_mask` removes windows that look like artifact rather than ECG. The
rules are deliberately simple and inspectable in the debug viewer.

Windowing:

- Window length: 5.0 s.
- Step: 2.5 s.
- Padding for hard artifact windows: 0.50 s.

Signals/features used per window:

- Raw ECG range.
- Filtered ECG standard deviation.
- Median and maximum `abs(filtered ECG)`.
- Main/preliminary candidate density.
- Energy candidate density.
- Maximum integrated energy.
- `qrs_snr`, a local contrast score: median candidate height divided by median
  background height. This is not a physical SNR.

Hard artifact rules remove windows with extreme raw range, filtered standard
deviation, dense candidate counts, or saturated energy. A softer low-confidence
rule removes a window only when QRS contrast is low, energy candidates are
dense, and a neighboring window is also low confidence.

### 7. Shape And Refractory Cleanup

After quality filtering, `_shape_filter` removes very close duplicate
detections. When two peaks are closer than `SHAPE_REFRACTORY_SEC = 0.32`, the
one with the stronger local shape score is kept.

The shape score combines:

- Local filtered ECG amplitude relative to a local median baseline.
- Local 5-20 Hz QRS-band amplitude, weighted at 0.25.

The final `_merge_close_peaks` pass enforces the 0.25 s energy refractory
distance and keeps the stronger nearby peak.

### 8. Removed Experiment

An earlier polarity/refinement step was removed from the main path. It improved
some visual examples but reduced robustness on biphasic or changing QRS
morphologies by sometimes choosing the wrong side of the complex.

## QRS Evaluation

Training evaluation is run through `test.py`.

Command:

```bash
python3 -B test.py --eval-train
```

Matching uses `match_qrs(predicted, expert, tolerance_samples=5)`.

- Tolerance: 50 ms = 5 samples at 100 Hz.
- Matching is one-to-one.
- Each predicted peak can match at most one expert peak.
- Sensitivity, PPV, and F1 are computed from aggregate TP/FP/FN counts.

The evaluator also records:

- Per-record TP, FP, FN, Sensitivity, PPV, F1.
- Prediction count and expert count.
- First unmatched expert or predicted sample.
- Common timing offsets between matched predicted and expert peaks.
- Clusters of false positives that occur close together.

Current training snapshot:

```text
Sensitivity = 0.996127
PPV         = 0.995692
F1          = 0.995909
TP          = 1116710
FP          = 4832
FN          = 4342
```

Current test-set snapshot recorded on 2026-05-23:

```text
Sensitivity = 0.995743
PPV         = 0.995819
F1          = 0.995781
```

## HRV Calculation

HRV metrics are calculated from QRS peak indices in `test.py`.

### 1. RR Intervals

`_compute_rr_intervals(peaks)` converts QRS sample differences to seconds:

```text
RR seconds = diff(peaks) / 100
```

`clean_rr_intervals` keeps only plausible intervals:

```text
0.25 s <= RR <= 2.0 s
```

### 2. Windowing

`compute_windowed_hrv` splits each record into 5-minute windows:

```text
WINDOW_SAMPLES = 5 * 60 * 100 = 30000
```

A window is skipped unless it has:

- At least 20 raw RR intervals.
- At least 20 cleaned RR intervals.
- At least 240 seconds of cleaned RR duration.

Record-level HRV values are the mean of valid window-level values.

### 3. Time-Domain HRV

For each valid window, RR intervals are converted to milliseconds.

- `avgRR`: mean RR in ms.
- `sdRR`: sample standard deviation of RR in ms, using `ddof=1`.
- `RMSSD`: root mean square of successive RR differences.
- `pNN50`: percentage of successive RR differences greater than 50 ms.

### 4. Frequency-Domain HRV

`_estimate_lf_hf_power` estimates frequency power from cleaned RR intervals.

Implementation:

1. Convert RR intervals to milliseconds.
2. Build cumulative beat timestamps.
3. Interpolate RR values onto a 4 Hz uniform time grid.
4. Remove the mean.
5. Estimate PSD with Welch using a boxcar window, full segment length, no overlap.
6. Integrate power with the trapezoid rule.

Bands:

- `LF`: 0.04 Hz <= f < 0.15 Hz.
- `HF`: 0.15 Hz <= f <= 0.40 Hz.
- `LF_HFratio`: `LF / HF` when `HF > 0`, otherwise 0.

For the final record-level LF/HF ratio, only windows with positive HF are
averaged.

Current test-set HRV snapshot recorded on 2026-05-23:

```text
MAPE_avgRR        = 0.081861
MAPE_sdRR         = 3.229323
MAPE_RMSSD        = 9.419067
MAPE_pNN50        = 11.655063
MAPE_LF           = 20.650554
MAPE_HF           = 23.786912
MAPE_LF_HFratio   = 12.962603
averageMAPE       = 11.683626
```

## Output Artifacts

Training evaluation may write:

- `metrics.csv`: per-record QRS metrics and optional HRV metrics.
- `predictions.npy`: dictionary of record number to predicted QRS sample indices.
- `outputs/qrs_eval/training_metrics.csv`: saved evaluation table.
- `outputs/qrs_eval/training_summary.txt`: aggregate QRS metrics.
- `outputs/qrs_eval/hrv_metrics.csv`: calculated HRV metrics.
- `outputs/qrs_eval/hrv_predicted_vs_reference.csv`: training HRV comparison detail.
- `outputs/qrs_eval/hrv_predicted_vs_reference_summary.csv`: training HRV comparison summary.
- `outputs/qrs_eval/*.png`: F1 plots, Sensitivity/PPV plots, and overlay plots.

Test submission filling uses:

1. `predictions.npy` for QRS peak sample indices.
2. `outputs/qrs_eval/hrv_metrics.csv` for HRV values.
3. `dataSet/ProjectTestDataAnalysis.mat` as the submission template.
4. `processor.py` to write `dataSet/ProjectTestDataAnalysis_filled.mat`.

`visualize_submission.py` can then check record counts, QRS index bounds,
monotonicity, HRV fields, and overlay plots against the test ECG.

## Debugging Workflow

For visual inspection, run:

```bash
python3 test.py --viz --patient 12 --start 0 --length 15000
```

The viewer exposes the layers that matter for diagnosis:

- Raw ECG and filtered ECG.
- `abs(filtered)` and main threshold.
- QRS-band energy and energy threshold.
- Predicted QRS peaks and expert QRS peaks when labels are available.
- Main candidates and energy candidates.
- Quality mask, noise score, removed noise peaks, and removed shape peaks.

This is the preferred way to inspect whether errors come from thresholding,
long-gap filling, artifact masking, or duplicate cleanup.
