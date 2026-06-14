import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy.signal import welch

import qrs_pipeline
from qrs_pipeline import DEFAULT_FS, default_test_mat, detect_qrs
from qrs_debug_viewer import run_qrs_debug_viewer


BASE_DIR = Path(__file__).resolve().parent
PROJECT_TEST_DATA = default_test_mat()
PREDICTIONS_PATH = BASE_DIR / "predictions.npy"
PREDICTIONS_MANIFEST_PATH = BASE_DIR / "predictions_manifest.json"
DEFAULT_OUTPUT_DIR = BASE_DIR / "outputs" / "qrs_eval"

HRV_KEYS = ("avgRR", "sdRR", "RMSSD", "pNN50", "LF", "HF", "LF_HFratio")
WINDOW_SAMPLES = int(5 * 60 * DEFAULT_FS)
MIN_RR_PER_WINDOW = 20
MIN_UNFLAGGED_RR_SECONDS_PER_WINDOW = 4 * 60


def write_predictions_manifest(all_predictions, mat_path, source_label, max_len=None):
    counts = {str(int(record)): int(len(peaks)) for record, peaks in sorted(all_predictions.items())}
    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "mat_path": str(mat_path),
        "source": source_label,
        "qrs_mode": "stable",
        "max_len": max_len,
        "postprocess": {},
        "prediction_count_by_record": counts,
        "qrs_settings": qrs_pipeline.qrs_settings(),
    }
    PREDICTIONS_MANIFEST_PATH.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return PREDICTIONS_MANIFEST_PATH


# Global log-space HRV calibration used by the final submission.
HRV_LOG_CALIBRATION_ALPHA = 0.60
HRV_LOG_CALIBRATION = {
    "avgRR": (1.0125512982, -0.0829013068614),
    "sdRR": (1.00358113518, -0.0498438721726),
    "RMSSD": (1.047177097, -0.250379323494),
    "pNN50": (1.05531184822, -0.212339760171),
    "LF": (1.02001922793, 0.0150197712032),
    "HF": (1.09380280754, -0.459400982043),
    "LF_HFratio": (1.06543305709, -0.0332337413407),
}


def _compute_rr_intervals(peaks, fs=DEFAULT_FS):
    peaks = np.asarray(peaks, dtype=float)
    if len(peaks) < 2:
        return np.asarray([], dtype=float)
    return np.diff(peaks) / fs


def clean_rr_intervals(rr, min_rr=0.25, max_rr=2.0):
    rr = np.asarray(rr, dtype=float)
    return rr[(rr >= min_rr) & (rr <= max_rr)]


def _estimate_lf_hf_power(
    rr_intervals,
    min_f_lf=0.04,
    max_f_lf=0.15,
    max_f_hf=0.40,
    fs_resample=2.0,
):
    rr_intervals = np.asarray(rr_intervals, dtype=float)
    if len(rr_intervals) < 10:
        return 0.0, 0.0

    rr_ms = rr_intervals * 1000.0
    time_stamps = np.concatenate([[0.0], np.cumsum(rr_ms[:-1])]) / 1000.0
    t_uniform = np.arange(time_stamps[0], time_stamps[-1], 1.0 / fs_resample)
    if len(t_uniform) < 8:
        return 0.0, 0.0

    rr_resampled = np.interp(t_uniform, time_stamps, rr_ms)
    rr_resampled -= rr_resampled.mean()

    freqs, psd = welch(
        rr_resampled,
        fs=fs_resample,
        window="boxcar",
        nperseg=len(rr_resampled),
        noverlap=0,
        scaling="density",
    )

    lf_mask = (freqs >= min_f_lf) & (freqs < max_f_lf)
    hf_mask = (freqs >= max_f_lf) & (freqs <= max_f_hf)

    lf_power = float(np.trapezoid(psd[lf_mask], freqs[lf_mask])) if lf_mask.any() else 0.0
    hf_power = float(np.trapezoid(psd[hf_mask], freqs[hf_mask])) if hf_mask.any() else 0.0
    return lf_power, hf_power


def _hrv_for_window(window_peaks, fs=DEFAULT_FS):
    rr_raw = _compute_rr_intervals(window_peaks, fs)
    if len(rr_raw) < MIN_RR_PER_WINDOW:
        return None

    rr = clean_rr_intervals(rr_raw)
    if len(rr) < MIN_RR_PER_WINDOW:
        return None

    if np.sum(rr) < MIN_UNFLAGGED_RR_SECONDS_PER_WINDOW:
        return None

    rr_ms = rr * 1000.0
    avg_rr = float(np.mean(rr_ms))
    sd_rr = float(np.std(rr_ms, ddof=1))
    rmssd = float(np.sqrt(np.mean(np.diff(rr_ms) ** 2))) if len(rr_ms) > 1 else 0.0
    pnn50 = float(np.sum(np.abs(np.diff(rr_ms)) > 50.0) / max(1, len(rr_ms) - 1) * 100.0)
    lf_power, hf_power = _estimate_lf_hf_power(rr)
    lf_hf_ratio = float(lf_power / hf_power) if hf_power > 0 else 0.0

    return {
        "avgRR": avg_rr,
        "sdRR": sd_rr,
        "RMSSD": rmssd,
        "pNN50": pnn50,
        "LF": lf_power,
        "HF": hf_power,
        "LF_HFratio": lf_hf_ratio,
    }


def compute_windowed_hrv(peaks, raw_len, fs=DEFAULT_FS, window_samples=WINDOW_SAMPLES):
    peaks = np.asarray(peaks, dtype=int)
    nan_result = {key: np.nan for key in HRV_KEYS}
    window_results = []

    for window_start in range(0, raw_len, window_samples):
        window_end = window_start + window_samples
        window_peaks = peaks[(peaks >= window_start) & (peaks < window_end)]
        result = _hrv_for_window(window_peaks, fs)
        if result is not None:
            window_results.append(result)

    if not window_results:
        return nan_result

    averaged = {}
    for key in HRV_KEYS:
        values = [window[key] for window in window_results]
        averaged[key] = float(np.mean(values))

    valid_ratios = [window["LF_HFratio"] for window in window_results if window["HF"] > 0]
    averaged["LF_HFratio"] = float(np.mean(valid_ratios)) if valid_ratios else 0.0
    return averaged


def calibrate_hrv_metrics(metrics):
    calibrated = dict(metrics)
    for key, (slope, intercept) in HRV_LOG_CALIBRATION.items():
        value = calibrated.get(key, np.nan)
        if not np.isfinite(value) or value <= 0:
            continue
        raw_log = np.log(value)
        corrected_log = slope * raw_log + intercept
        calibrated[key] = float(
            np.exp((1.0 - HRV_LOG_CALIBRATION_ALPHA) * raw_log + HRV_LOG_CALIBRATION_ALPHA * corrected_log)
        )
    return calibrated


def generate_record_outputs(mat_path=PROJECT_TEST_DATA, include_hrv=False, max_len=None, verbose=True):
    data = loadmat(mat_path)
    ecg_records = data["ECG"].ravel()

    rows = []
    all_predictions = {}

    for record_number, raw_cell in enumerate(ecg_records, start=1):
        raw = raw_cell.ravel().astype(float)
        if max_len is not None:
            raw = raw[:max_len]

        _filtered, predicted = detect_qrs(raw, DEFAULT_FS)
        row = {
            "record": record_number,
            "pred_count": len(predicted),
        }

        if include_hrv:
            row.update(compute_windowed_hrv(predicted, len(raw)))

        rows.append(row)
        all_predictions[record_number] = predicted

    if include_hrv:
        rows = [dict(row, **calibrate_hrv_metrics(row)) for row in rows]

    if verbose:
        for row in rows:
            message = f"Record {int(row['record']):02d}: pred={int(row['pred_count'])}"
            if include_hrv:
                message += (
                    f" HRV: avgRR={row['avgRR']:.1f} "
                    f"RMSSD={row['RMSSD']:.1f} "
                    f"pNN50={row['pNN50']:.1f} "
                    f"LF/HF={row['LF_HFratio']:.2f}"
                )
            print(message)

    np.save(PREDICTIONS_PATH, all_predictions, allow_pickle=True)
    write_predictions_manifest(all_predictions, mat_path, "test", max_len=max_len)
    pd.DataFrame(rows).to_csv("metrics.csv", index=False)
    print(f"\nGenerated outputs for {len(rows)} records")
    return rows


def save_hrv_metrics_csv(rows, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "hrv_metrics.csv"

    fieldnames = ["record", "avgRR", "sdRR", "RMSSD", "pNN50", "LF", "HF", "LF_HF"]
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "record": row["record"],
                    "avgRR": row["avgRR"],
                    "sdRR": row["sdRR"],
                    "RMSSD": row["RMSSD"],
                    "pNN50": row["pNN50"],
                    "LF": row["LF"],
                    "HF": row["HF"],
                    "LF_HF": row["LF_HFratio"],
                }
            )

    manifest_path = csv_path.with_name(f"{csv_path.stem}_manifest.json")
    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source": "predicted",
        "calibrated": True,
        "calibration_level": "global",
        "qrs_mode": "stable",
        "row_count": len(rows),
        "csv_path": str(csv_path),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return csv_path


def main():
    parser = argparse.ArgumentParser(description="Generate final ECG QRS and HRV outputs")
    parser.add_argument("--mat", type=Path, default=PROJECT_TEST_DATA)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--hrv", action="store_true", help="calculate calibrated HRV and save hrv_metrics.csv")
    parser.add_argument("--viz", action="store_true", help="open interactive QRS viewer")
    parser.add_argument("--patient", default="1", help="1-based record number for --viz")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--length", type=int, default=15_000, help="number of samples shown by --viz")
    parser.add_argument("--show-raw", action="store_true")
    args = parser.parse_args()

    if args.viz:
        run_qrs_debug_viewer(
            mat_path=args.mat,
            patient=args.patient,
            start=args.start,
            length=args.length,
            show_raw=args.show_raw,
        )
        return

    rows = generate_record_outputs(args.mat, include_hrv=args.hrv)
    if args.hrv:
        hrv_path = save_hrv_metrics_csv(rows, args.out_dir)
        print(f"\nSaved HRV metrics: {hrv_path}")


if __name__ == "__main__":
    main()
