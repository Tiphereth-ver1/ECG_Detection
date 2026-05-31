# ECG QRS Detection

This repository is for the BMET3997 major project ECG work. The current code
contains the QRS detector, windowed HRV calculation, training-set evaluation,
and helper scripts for filling and checking the test-set submission `.mat`.

## Assignment Requirements Covered Here

- Input data is MATLAB `.mat` format.
- `ProjectTrainData.mat` contains `ECG` and `QRSexpert` cell arrays.
- Each ECG record is sampled at `100 Hz`.
- QRS detections are evaluated against expert annotations with a `50 ms` tolerance, equal to `5 samples`.
- Training evaluation reports Sensitivity, Positive Predictivity, F1-score, TP, FP, and FN.

## Current QRS Method

The detector is inspired by Pan-Tompkins style processing, WFDB/XQRS style
energy detection, and PhysioNet post-processing ideas. It does not use any QRS
detection toolbox or external QRS detector.

1. Apply a 50 Hz notch filter.
2. Apply a broad ECG bandpass filter for final peak placement.
3. Run a `5-20 Hz` QRS-energy branch using derivative, squaring, moving-window integration, and local QRS-band peak refinement.
4. Use the QRS-energy branch as the current candidate source, then reject broad-wave candidates that do not have enough nearby QRS-band support.
5. Mark noisy windows with a signal-quality check using raw range, filtered standard deviation, energy saturation, candidate density, and local QRS contrast.
6. Remove close duplicate detections with shape-strength cleanup.
7. Apply waveform-based late positive-lobe alignment for records that contain many biphasic complexes whose expert marks land on the later positive lobe.
8. Apply adaptive RR cleanup to remove dense half-RR artifact duplicates.
9. Rescue isolated high-confidence QRS peaks that were removed by earlier quality/shape/RR gates.
10. Apply a conservative final morphology cleanup for weak residual false positives, unless the late-alignment trigger is active.

The earlier polarity/refinement experiment remains out of the main path. It
helped a few screenshots but often chose the wrong side of biphasic QRS
complexes. The current alignment branch is triggered by record-level waveform
statistics rather than record ID.

References used for the approach:

- Pan and Tompkins, "A Real-Time QRS Detection Algorithm", 1985.
- PhysioNet `wqrs` and `gqrs` detector documentation.
- WFDB `XQRS` processing notes.
- WFDB `correct_peaks` style post-processing idea.
- ECG signal-quality/artifact assessment ideas.

## QRS Cleanup Snapshot - 2026-05-23

- Removed the polarity detector and final peak refinement from the main QRS path.
- Kept the main `abs(filtered ECG)` detector because it works for both upright and inverted beats.
- Kept the QRS-energy branch only as a missed-beat rescue inside long RR gaps.
- Kept the signal-quality mask for extreme artifact windows, because it reduces noisy false positives.
- Added the `qrs_snr + energy density` soft gate to catch long noisy sections that are not extreme enough for the raw-range artifact rules.
- Kept simple close-peak shape cleanup, but only for duplicate/very-close detections.
- Added debug layers for noisy windows, noise score, removed-noise peaks, removed-shape peaks, main candidates, and energy candidates.
- Added training output columns for timing offset summary and false-positive clusters.

No record-specific rule is used. The rules are based on general ECG ideas:
refractory period, plausible RR interval, QRS energy, artifact amplitude, and
candidate density.

## Setup

Install the Python dependencies:

```bash
pip install numpy scipy matplotlib
```

Expected local data layout:

```text
dataSet/
  ProjectTrainData.mat
  ProjectTestData.mat
  ProjectTestDataAnalysis.mat
```

`ProjectTrainData.mat` is used for local training-set evaluation. The test-set
submission helpers read `ProjectTestData.mat` and `ProjectTestDataAnalysis.mat`.

## Run Training Evaluation

Run the full 35-record training-set QRS evaluation:

```bash
python3 -B test.py --eval-train
```

This prints one line per record plus the total training summary:

```text
Record 01: Sens=... PPV=... F1=... TP=... FP=... FN=...
...
Training summary: Sens=... PPV=... F1=...
```

## Save Result Visualizations

Run evaluation and save CSV/plots:

```bash
python3 -B test.py --eval-train --save-plots
```

Default output directory:

```text
outputs/qrs_eval/
```

Generated files:

- `training_metrics.csv`: per-record metrics, offset summary, and false-positive cluster summary.
- `training_summary.txt`: overall Sensitivity, PPV, F1, TP, FP, FN.
- `f1_by_record.png`: F1-score by record.
- `sensitivity_vs_ppv.png`: Sensitivity vs PPV scatter plot.
- `record_XX_overlay.png`: overlay plots for the worst records.

Training-set performance snapshot for this version:

```text
Sensitivity: 0.996686
PPV: 0.997314
F1: 0.997000
TP: 1117337
FP: 3009
FN: 3715
```

### Test-Set Performance Snapshot - 2026-05-23

This is a dated snapshot for the current project version and should not be
treated as a rolling "latest" result after future detector updates.

- Date recorded: 2026-05-23, Australia/Sydney.
- Version anchor: branch `main`, HEAD `fa8bd7d` when this section was written.
- Source: test-set prediction result reported after running this version's
  pipeline. The hidden test labels are not stored in this repository.

QRS detection:

```text
Sensitivity = 0.995743
PPV         = 0.995819
F1          = 0.995781
```

HRV mean absolute percentage error:

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

Comparison with the training-set snapshot above:

| Metric | Trainset | Testset | Test - Train |
| --- | ---: | ---: | ---: |
| Sensitivity | 0.996127 | 0.995743 | -0.000384 |
| PPV | 0.995692 | 0.995819 | +0.000127 |
| F1 | 0.995909 | 0.995781 | -0.000128 |
| MAPE_avgRR | 0.313610 | 0.081861 | -0.231749 |
| MAPE_sdRR | 4.905944 | 3.229323 | -1.676621 |
| MAPE_RMSSD | 10.834260 | 9.419067 | -1.415193 |
| MAPE_pNN50 | 15.161347 | 11.655063 | -3.506284 |
| MAPE_LF | 14.579588 | 20.650554 | +6.070966 |
| MAPE_HF | 18.893936 | 23.786912 | +4.892976 |
| MAPE_LF_HFratio | 13.549265 | 12.962603 | -0.586662 |
| averageMAPE | 11.176850 | 11.683626 | +0.506776 |

Interpretation: QRS detection generalizes almost unchanged from train to test.
The F1 drop is only `0.000128`, while PPV is slightly higher on test. HRV is
also stable overall: average MAPE increases by about `0.51` percentage points.
The main test-set weakness is frequency-domain power (`LF`, `HF`), while the
time-domain HRV metrics are better than the training snapshot.

### Training-Set Interpretation - 2026-05-23

Across all **35** training records, aggregated metrics against expert annotations
(with **50 ms** tolerance) are: **Sensitivity 0.9961**, **PPV 0.9957**, **F1 0.9959**
(total **1,116,710** TP, **4,832** FP, **4,342** FN). Most records score **F1 ≥ 0.997**;
the weakest are **records 17, 24, 25, and 33** (roughly **0.95–0.99** F1), where errors
show up as **dense false-positive clusters** in localized time intervals and, on some
tracks, noticeable **timing offsets** versus the expert peak indices (see per-record `offsets pred-expert` and `FP clusters` lines when running `--eval-train`). Elsewhere,
residual mismatch is mostly small index shifts rather than wholesale missed beats.

Compared with the previous quality-mask-only version, the `qrs_snr + energy
density` gate keeps sensitivity almost the same while reducing false positives
by about one thousand. Compared with the earlier `~0.9933` QRS version, it is
clearly better. Compared with the original baseline around `0.973`, it is much
better.

The main remaining weak points are noisy sections where the expert labels stop,
and morphology changes such as record 17 later in the signal.

## Pipeline And Performance Snapshot - 2026-05-31

Version anchor: branch `main`, commit `2a6bad2`.

This version adds a post-processing pipeline around the QRS-energy candidate
source:

- QRS-band veto for broad T-wave/slow-wave candidates.
- Late positive-lobe alignment for repeated biphasic morphology shifts.
- Adaptive RR cleanup for dense half-RR duplicate detections.
- Conservative quality rescue for isolated strong QRS peaks removed by earlier gates.
- Final morphology cleanup for weak residual false positives.
- Debug viewer layers for `Removed QRS-band`, `Aligned late`, `Removed RR`, `Rescued quality`, and `Removed final`.
- Global log-linear HRV output calibration enabled by default, with `--no-hrv-calibration` available for audit, using one coefficient pair per HRV metric and no record-specific rules.

Training-set QRS performance against expert annotations, 50 ms tolerance:

```text
Sensitivity: 0.996686
PPV: 0.997314
F1: 0.997000
TP: 1117337
FP: 3009
FN: 3715
```

Compared with the 2026-05-23 snapshot:

```text
F1: 0.995909 -> 0.997000
TP: 1116710 -> 1117337
FP: 4832 -> 3009
FN: 4342 -> 3715
```

Training-set HRV comparison against `documents/training_expert_hrv_reference.csv`:

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

The largest QRS gains are concentrated in the known difficult records:
record 17 improves from `0.950186` to `0.963950` F1, record 25 from `0.971250`
to `0.976390`, record 33 from `0.988485` to `0.991383`, and record 24 from
`0.987006` to `0.989778`. The main remaining weak areas are long artifact
sections with expert labels and record 17's late morphology change.

### Hidden Test-Set Result - 2026-05-31

The hidden test-set result reported by Shilei for this version is:

```text
Sensitivity = 0.994124
PPV         = 0.994533
F1          = 0.994328

MAPE_avgRR        = 0.048350
MAPE_sdRR         = 1.995051
MAPE_RMSSD        = 10.446275
MAPE_pNN50        = 10.022548
MAPE_LF           = 18.538174
MAPE_HF           = 23.922575
MAPE_LF_HFratio   = 13.104062
averageMAPE       = 11.153862
```

Compared with the training snapshot above:

| Metric | Trainset | Hidden testset | Test - Train |
| --- | ---: | ---: | ---: |
| Sensitivity | 0.996686 | 0.994124 | -0.002562 |
| PPV | 0.997314 | 0.994533 | -0.002781 |
| F1 | 0.997000 | 0.994328 | -0.002672 |
| averageMAPE | 9.363221 | 11.153862 | +1.790641 |

Interpretation: this version overfits the training set. The post-processing
pipeline improves the labeled training records, especially records 17, 24, 25,
and 33, but the hidden test F1 drops below both the target and the older
2026-05-23 test snapshot. The HRV averageMAPE also misses the `< 10` target on
hidden test. Future work should back off the late-stage cleanup/alignment stack
or validate it with stricter leave-record-out style checks before submission.

## Interactive Overlay Viewer

Open a record in an interactive matplotlib viewer:

```bash
python3 test.py --viz --patient 12 --start 0 --length 15000
```

`--patient` is a 1-based record number, so `--patient 12` means training record 12.

Useful flags:


| Flag          | Meaning                                                          |
| ------------- | ---------------------------------------------------------------- |
| `--mat PATH`  | `.mat` file to load. Defaults to `dataSet/ProjectTrainData.mat`. |
| `--patient N` | 1-based record number.                                           |
| `--start N`   | First sample shown in the viewer.                                |
| `--length N`  | Number of samples shown.                                         |
| `--show-raw`  | Also draw raw ECG behind the filtered ECG.                       |
| `--max-len N` | Crop each record for quick debugging.                            |


The viewer can draw raw ECG, filtered ECG, `abs(filtered)`, main threshold, QRS
energy, energy threshold, predicted QRS, expert QRS, main candidates, energy
candidates, noisy-window mask, noise score, removed noise peaks, removed shape
peaks, removed QRS-band veto peaks, late-aligned peaks, removed RR peaks,
quality-rescued peaks, and final morphology removals. Layers can be hidden from
the checkbox panel.

Navigation:

- Use the `Start` and `Window` sliders to move and zoom by sample range.
- Use the mouse wheel over the plot to zoom horizontally.
- Use `n`/right arrow for next record and `p`/left arrow for previous record.


## Files

- `qrs_pipeline.py`: ECG filtering, QRS detection, missed-beat filling, signal-quality filtering, and overlay plotting.
- `qrs_debug_viewer.py`: interactive debug viewer, layer toggles, sliders, record navigation, and scroll zoom.
- `test.py`: command-line evaluation, CSV output, offset summaries, FP cluster summaries, and visualization outputs.
- `processor.py`: fills the submission analysis `.mat` from saved QRS predictions and HRV metrics.
- `visualize_submission.py`: validates and visualizes the filled test-set submission `.mat`.
- `TECHNICAL_PIPELINE_2026-05-23.md`: dated technical description of the 2026-05-23 pipeline.
- `TECHNICAL_PIPELINE_2026-05-31.md`: dated technical description of the current post-processing pipeline and performance snapshot.
- `archive/`: historical and experimental scripts that are not part of the active pipeline.
- `README.md`: project summary and run instructions.

## Current Scope

This stage does not:

- use any QRS detection toolbox/library,
- use record-specific detector rules,
- train a black-box model.
