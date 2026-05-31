# Technical Pipeline Snapshot - 2026-05-31

This document describes the ECG QRS + HRV pipeline as of 2026-05-31
(Australia/Sydney), anchored to branch `main` at commit `2a6bad2`. It is a
dated snapshot, not a rolling description of future versions.

## Scope

The pipeline is rule-based. It does not use a QRS detection toolbox, an external
ECG detector, record-specific detector rules, or a trained black-box model. It
processes MATLAB `.mat` ECG records sampled at 100 Hz, detects QRS sample
indices, derives windowed HRV metrics, and supports submission `.mat` filling
and visualization.

Main implementation files:

- `qrs_pipeline.py`: filtering, QRS candidate detection, waveform alignment, artifact cleanup, debug layers.
- `test.py`: training-set evaluation, QRS matching, HRV calculation, CSV/plot export.
- `qrs_debug_viewer.py`: interactive inspection of detector layers.
- `processor.py`: fills `ProjectTestDataAnalysis.mat` with QRS and HRV outputs.
- `visualize_submission.py`: validates and plots the filled test submission.

## QRS Detection Pipeline

### 1. Broad ECG Preprocessing

`preprocess_ecg(raw)` converts the raw ECG to a broad filtered signal used for
plotting and final marker placement:

1. Flatten the input to floating-point samples.
2. Apply a 50 Hz notch filter.
3. Apply a second-order 1 Hz high-pass Butterworth filter.
4. Apply a second-order 40 Hz low-pass Butterworth filter.

This keeps visible ECG morphology while removing baseline drift and high
frequency noise.

### 2. QRS-Band Energy Candidate Source

`qrs_bandpass(raw)` creates a 5-20 Hz QRS-band signal. The energy detector then:

1. Differentiates the QRS-band signal.
2. Squares the derivative.
3. Smooths with a 0.15 s moving average.
4. Detects peaks above a median/MAD adaptive threshold.
5. Refines each candidate to the strongest nearby QRS-band sample.

The current active path uses this QRS-energy branch as the candidate source.
The older `abs(filtered ECG)` branch is retained in code for debug/history but
is not the current default branch.

### 3. QRS-Band Veto

`_qrs_band_veto` removes broad-wave candidates that do not have enough local
5-20 Hz QRS-band support. This catches slow T-wave or artifact peaks that look
large in the broad ECG channel but are weak in the QRS band.

Main parameters:

```text
QRS_BAND_VETO_RATIO = 0.235
QRS_BAND_VETO_RADIUS_SEC = 0.04
```

### 4. Signal-Quality Mask

`_quality_mask` rejects obvious artifact windows using raw range, filtered ECG
standard deviation, energy saturation, candidate density, and local QRS
contrast. Window length is 5.0 s and step size is 2.5 s. The current version
does not pad rejected windows beyond their measured span.

### 5. Shape Cleanup

`_shape_filter` removes very close duplicate detections. When two candidates
are closer than `SHAPE_REFRACTORY_SEC = 0.32`, the detector keeps the stronger
local morphology based on filtered ECG amplitude and QRS-band support.

### 6. Late Positive-Lobe Alignment

`_align_late_positive_lobes` addresses a repeated biphasic morphology pattern:
the detector may lock onto the earlier negative valley while the expert mark
lands on the later positive lobe.

The trigger is record-level and waveform-based:

- Scan the tail of the record after 22,000 s.
- Count candidate beats whose later positive lobe is strong relative to the
  local negative valley.
- Enable alignment only if at least 500 such candidates are present and they
  account for at least 8% of tail detections.

When triggered, candidates after 20,000 s can move from the negative valley to
the later positive lobe. This improves record 17 without using record numbers.

### 7. Adaptive RR Cleanup

`_adaptive_rr_cleanup` removes dense duplicate detections that are too close for
the record's own rhythm. It estimates a typical RR interval from valid local
intervals and sets the minimum gap to:

```text
max(0.45 s, 0.50 * typical RR)
```

When two peaks violate that gap, the stronger local QRS morphology is kept.

### 8. Quality Rescue

`_rescue_quality_peaks` restores isolated high-confidence peaks removed by
earlier quality, shape, or RR gates. A peak can be rescued only when it is far
from an accepted detection, has strong local QRS-band amplitude, and sits in a
quiet broad-band and QRS-band neighborhood.

Main guardrails:

```text
QUALITY_RESCUE_MIN_NEAREST_SEC = 0.30
QUALITY_RESCUE_MIN_QRS_BAND = 250
QUALITY_RESCUE_MAX_ABS_MEDIAN = 150
QUALITY_RESCUE_MAX_QRS_MEDIAN = 60
```

### 9. Final Morphology Cleanup

`_final_morphology_cleanup` removes weak residual false positives. It targets
peaks that are weak in the QRS band and either occur in noisy high-background
segments or are extremely weak in both broad and QRS-band views.

This stage is skipped when late positive-lobe alignment is triggered. That
guard avoids over-deleting difficult biphasic records where the detector is
already performing morphology correction.

## HRV Pipeline

HRV is calculated from detected QRS peaks in 5-minute windows.

- RR intervals are `diff(peaks) / 100`.
- Clean RR range: 0.25 s to 2.0 s.
- A window must contain at least 20 raw RR intervals, 20 cleaned intervals, and
  240 seconds of cleaned RR duration.
- Record-level HRV is the mean of valid window-level values.

Time-domain metrics:

- `avgRR`
- `sdRR`
- `RMSSD`
- `pNN50`

Frequency-domain metrics:

- `LF`: 0.04 Hz <= f < 0.15 Hz
- `HF`: 0.15 Hz <= f <= 0.40 Hz
- `LF_HFratio`

The current frequency estimation resamples RR intervals at 2 Hz, removes the
mean, estimates PSD with Welch using a boxcar window, and integrates power with
the trapezoid rule.

### HRV Output Calibration

This version can optionally apply a global log-linear output calibration with
`--hrv-calibration`:

```text
log(reference) = slope * log(raw_estimate) + intercept
```

There is one slope/intercept pair per HRV metric. There are no record-specific
calibration rules. The calibration follows the same low-degree output
correction idea as the project presentation architecture. The CLI default is
uncalibrated HRV output.

## Evaluation Protocol

Training QRS evaluation is run with:

```bash
python3 -B test.py --eval-train
```

Full QRS + HRV comparison is run with:

```bash
python3 -B test.py --eval-train --compare-hrv --save-plots --worst-count 8
```

Add `--hrv-calibration` to enable the optional global HRV output calibration.

QRS matching uses a 50 ms tolerance, equal to 5 samples at 100 Hz. Matching is
one-to-one, so each predicted peak and expert peak can contribute to at most one
true positive.

## Performance Snapshot

Current training-set QRS performance:

```text
Sensitivity = 0.996686
PPV         = 0.997314
F1          = 0.997000
TP          = 1117337
FP          = 3009
FN          = 3715
```

Previous 2026-05-23 snapshot:

```text
Sensitivity = 0.996127
PPV         = 0.995692
F1          = 0.995909
TP          = 1116710
FP          = 4832
FN          = 4342
```

Aggregate change:

```text
F1 +0.001091
TP +627
FP -1823
FN -627
```

Largest per-record F1 gains compared with the previous snapshot:

| Record | Previous F1 | Current F1 | Change | Main effect |
| --- | ---: | ---: | ---: | --- |
| 17 | 0.950186 | 0.963950 | +0.013764 | late positive-lobe alignment and rescue |
| 25 | 0.971250 | 0.976390 | +0.005141 | noisy FP cleanup and RR cleanup |
| 33 | 0.988485 | 0.991383 | +0.002898 | residual FP cleanup |
| 24 | 0.987006 | 0.989778 | +0.002772 | residual FP cleanup |
| 12 | 0.994628 | 0.996872 | +0.002245 | FP reduction in noisy windows |

Training-set HRV comparison against
`documents/training_expert_hrv_reference.csv` with `--hrv-calibration`:

```text
MAPE_avgRR        = 0.335173
MAPE_sdRR         = 2.501315
MAPE_RMSSD        = 6.274485
MAPE_pNN50        = 13.262251
MAPE_LF           = 13.650064
MAPE_HF           = 17.271437
MAPE_LF_HFratio   = 12.247821
averageMAPE       = 9.363221
```

Running the default uncalibrated HRV calculation gives `averageMAPE = 9.669`,
so the optional global calibration improves the HRV target without being solely
responsible for passing it.

## Debug Viewer Layers

The interactive viewer can be opened with:

```bash
python3 test.py --viz --patient 17 --start 2884750 --length 15000 --show-raw
```

Useful diagnosis layers include:

- Raw ECG and filtered ECG.
- QRS-band energy and energy threshold.
- Predicted QRS and expert QRS.
- FP/FN overlay.
- Removed QRS-band veto peaks.
- Removed noise peaks and removed shape peaks.
- Late-aligned peaks.
- Removed RR peaks.
- Quality-rescued peaks.
- Final morphology removals.

The most informative windows for this version are record 17 near 28,847.5 s for
late positive-lobe alignment, record 25 near 11,100 s for noisy false-positive
cleanup, and record 33 near 2,100 s for residual artifact cleanup.

## Remaining Weaknesses

Record 17 still contains a difficult late morphology shift; alignment improves
the aggregate result but does not solve every local timing mismatch. Record 25
still has long artifact regions where expert annotations continue but the ECG
signal is highly corrupted, producing concentrated false negatives. These are
the main areas to inspect before further changes.
