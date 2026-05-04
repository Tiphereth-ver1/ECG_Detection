# ECG QRS Detection

This repository is for the BMET3997 major project ECG work. The current code focuses only on QRS peak detection on the training set. HRV calculation is intentionally not changed in this stage.

## Assignment Requirements Covered Here

- Input data is MATLAB `.mat` format.
- `ProjectTrainData.mat` contains `ECG` and `QRSexpert` cell arrays.
- Each ECG record is sampled at `100 Hz`.
- QRS detections are evaluated against expert annotations with a `50 ms` tolerance, equal to `5 samples`.
- Training evaluation reports Sensitivity, Positive Predictivity, F1-score, TP, FP, and FN.

## Current QRS Method

The detector is a simple student-style implementation inspired by Pan-Tompkins style processing:

1. Apply a 50 Hz notch filter.
2. Apply a broad ECG bandpass filter for final peak placement.
3. Detect main candidates from local peaks in `abs(filtered ECG)`, so both upright and inverted QRS complexes can be detected.
4. Run a second `5-20 Hz` QRS-energy branch using derivative, squaring, and moving-window integration.
5. Use the energy branch only to fill suspiciously long RR gaps.
6. Merge close detections with a refractory period and keep the stronger peak.

No QRS detection toolbox or QRS detection library is used.

References used for the approach:

- Pan and Tompkins, "A Real-Time QRS Detection Algorithm", 1985.
- PhysioNet `wqrs` and `gqrs` detector documentation.
- WFDB `XQRS` processing notes.

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

Only `ProjectTrainData.mat` is used by the current QRS training evaluation.

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

- `training_metrics.csv`: per-record metrics.
- `training_summary.txt`: overall Sensitivity, PPV, F1, TP, FP, FN.
- `f1_by_record.png`: F1-score by record.
- `sensitivity_vs_ppv.png`: Sensitivity vs PPV scatter plot.
- `record_XX_overlay.png`: overlay plots for the worst records.

Current full-training result from this detector:

```text
Sensitivity: 0.997169
PPV: 0.989518
F1: 0.993329
TP: 1117878
FP: 11842
FN: 3174
```

The weakest record is currently record 25, mainly because it still has many false positives.

## Interactive Overlay Viewer

Open a record in an interactive matplotlib debug viewer:

```bash
python3 test.py --viz --patient 12 --start 0 --length 15000
```

`--patient` is a 1-based record number, so `--patient 12` means training record 12.

The viewer is implemented in `qrs_debug_viewer.py`. It draws one time-aligned figure with:

- raw ECG,
- filtered ECG,
- `abs(filtered ECG)`,
- main amplitude threshold,
- integrated QRS energy,
- energy threshold,
- predicted QRS peaks,
- expert QRS peaks,
- optional main and energy candidate peaks.

The right-side layer panel can hide/show each item independently. The `Hide controls` / `Show controls` button collapses or expands that panel. The `Start` and `Window` sliders control the displayed sample range, the mouse wheel zooms horizontally around the cursor, and `Prev record` / `Next record` plus `n/right` and `p/left` switch records.

Useful flags:

| Flag | Meaning |
| --- | --- |
| `--mat PATH` | `.mat` file to load. Defaults to `dataSet/ProjectTrainData.mat`. |
| `--patient N` | 1-based record number. |
| `--start N` | First sample shown in the viewer. |
| `--length N` | Number of samples shown. |
| `--show-raw` | Also draw raw ECG behind the filtered ECG. |
| `--max-len N` | Crop each record for quick debugging. |

## Files

- `qrs_pipeline.py`: ECG filtering, QRS detection, debug signal generation, missed-beat filling, and batch overlay plotting.
- `qrs_debug_viewer.py`: interactive QRS debug viewer with layer toggles, sliders, record navigation, and scroll zoom.
- `test.py`: command-line evaluation, saved analysis outputs, and `--viz` entry point.
- `README.md`: project summary and run instructions.

## Current Scope

This stage does not:

- calculate HRV parameters,
- generate the final test-set submission `.mat`,
- modify `hrvcalc.py`,
- use any QRS detection toolbox/library.
