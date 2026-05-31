import argparse
import csv
import os
import re
from pathlib import Path
import pandas as pd

import numpy as np
from scipy.io import loadmat
from scipy.signal import welch

from qrs_pipeline import (
    DEFAULT_FS,
    default_test_mat,
    default_train_mat,
    detect_qrs,
    load_recording,
    save_overlay_plot,
)
from qrs_debug_viewer import run_qrs_debug_viewer

PROJECT_TRAIN_DATA = default_train_mat()
PROJECT_TEST_DATA = default_test_mat()
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "outputs" / "qrs_eval"
DEFAULT_HRV_REFERENCE = Path(__file__).resolve().parent / "documents" / "training_expert_hrv_reference.csv"

HRV_KEYS = ("avgRR", "sdRR", "RMSSD", "pNN50", "LF", "HF", "LF_HFratio")
WINDOW_SAMPLES = int(5 * 60 * DEFAULT_FS)
MIN_RR_PER_WINDOW = 20
MIN_UNFLAGGED_RR_SECONDS_PER_WINDOW = 4 * 60

# Low-degree output calibration, matching the architecture used in the
# presentation slide: log(reference) = slope * log(raw_estimate) + intercept.
# The coefficients are global across records and metrics; there are no
# record-specific rules.
HRV_LOG_CALIBRATION = {
    "avgRR": (1.0125512982, -0.0829013068614),
    "sdRR": (1.00358113518, -0.0498438721726),
    "RMSSD": (1.047177097, -0.250379323494),
    "pNN50": (1.05531184822, -0.212339760171),
    "LF": (1.02001922793, 0.0150197712032),
    "HF": (1.09380280754, -0.459400982043),
    "LF_HFratio": (1.06543305709, -0.0332337413407),
}


def parse_length(length, raw_len=None):
    return int(length)


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
        calibrated[key] = float(np.exp(slope * np.log(value) + intercept))
    return calibrated


def calculate_hr_hrv(peaks):
    rr = clean_rr_intervals(_compute_rr_intervals(peaks))
    if len(rr) < 1:
        return 0.0, 0.0

    hr = 60 / np.mean(rr)
    rr_ms = rr * 1000
    rmssd = np.sqrt(np.mean(np.diff(rr_ms) ** 2)) if len(rr_ms) > 1 else 0.0
    return float(hr), float(rmssd)


def match_qrs(predicted, expert, tolerance_samples=5):
    # One-to-one matching with a fixed tolerance. Each predicted point can match
    # at most one expert point, which keeps TP/FP/FN accounting honest.
    predicted = np.asarray(predicted, dtype=int)
    expert = np.asarray(expert, dtype=int)

    predicted.sort()
    expert.sort()

    used = np.zeros(len(predicted), dtype=bool)
    matched_pred = []
    matched_expert = []
    search_start = 0

    for expert_peak in expert:
        # Move past predictions that are already too early for this expert beat.
        while search_start < len(predicted) and predicted[search_start] < expert_peak - tolerance_samples:
            search_start += 1

        # Pick the nearest unused predicted peak inside the tolerance window.
        best_idx = -1
        best_dist = tolerance_samples + 1
        idx = search_start
        while idx < len(predicted) and predicted[idx] <= expert_peak + tolerance_samples:
            if not used[idx]:
                dist = abs(predicted[idx] - expert_peak)
                if dist < best_dist:
                    best_dist = dist
                    best_idx = idx
            idx += 1

        if best_idx >= 0:
            used[best_idx] = True
            matched_pred.append(predicted[best_idx])
            matched_expert.append(expert_peak)

    tp = len(matched_pred)
    fp = int((~used).sum())
    fn = int(len(expert) - tp)

    sens = tp / (tp + fn) if tp + fn else 0.0
    ppv = tp / (tp + fp) if tp + fp else 0.0
    f1 = 2 * sens * ppv / (sens + ppv) if sens + ppv else 0.0

    return {
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "Sensitivity": sens,
        "PPV": ppv,
        "F1": f1,
        "unmatched_pred": predicted[~used],
        "matched_pred": np.asarray(matched_pred, dtype=int),
        "matched_expert": np.asarray(matched_expert, dtype=int),
    }


def _offset_summary(matched_pred, matched_expert, limit=5):
    # Show the most common timing offsets. This catches cases that look correct
    # visually but miss the 5-sample tolerance by a few samples.
    if len(matched_pred) == 0:
        return ""

    offsets = np.asarray(matched_pred, dtype=int) - np.asarray(matched_expert, dtype=int)
    values, counts = np.unique(offsets, return_counts=True)
    order = np.argsort(counts)[::-1][:limit]
    return ", ".join(f"{int(values[i]):+d}:{int(counts[i])}" for i in order)


def _fp_cluster_summary(unmatched_pred, fs=DEFAULT_FS, max_gap_sec=2.0, min_count=3, limit=3):
    # Group false positives that happen close together. Large clusters usually
    # mean a noisy segment, not isolated threshold mistakes.
    peaks = np.asarray(unmatched_pred, dtype=int)
    if len(peaks) == 0:
        return ""

    peaks.sort()
    max_gap = int(max_gap_sec * fs)
    clusters = []
    start = int(peaks[0])
    last = int(peaks[0])
    count = 1

    for peak in peaks[1:]:
        peak = int(peak)
        if peak - last <= max_gap:
            last = peak
            count += 1
        else:
            if count >= min_count:
                clusters.append((count, start, last))
            start = peak
            last = peak
            count = 1

    if count >= min_count:
        clusters.append((count, start, last))

    clusters.sort(reverse=True)
    clusters = clusters[:limit]
    return "; ".join(
        f"{start / fs:.1f}-{end / fs:.1f}s(n={count})"
        for count, start, end in clusters
    )


def _error_window_summary(unmatched_pred, unmatched_expert, raw_len, fs=DEFAULT_FS, window_sec=300, limit=5):
    # Show where errors concentrate in fixed 5-minute windows. This complements
    # short FP clusters by highlighting long noisy spans and missed-beat blocks.
    unmatched_pred = np.asarray(unmatched_pred, dtype=int)
    unmatched_expert = np.asarray(unmatched_expert, dtype=int)
    if raw_len <= 0 or (len(unmatched_pred) == 0 and len(unmatched_expert) == 0):
        return ""

    window = int(window_sec * fs)
    bins = np.arange(0, raw_len + window, window)
    if len(bins) < 2:
        bins = np.asarray([0, raw_len])

    fp_counts = np.histogram(unmatched_pred, bins=bins)[0]
    fn_counts = np.histogram(unmatched_expert, bins=bins)[0]
    total = fp_counts + fn_counts
    worst = np.argsort(total)[::-1]

    parts = []
    for idx in worst:
        if total[idx] <= 0:
            break
        start = bins[idx] / fs
        end = min(bins[idx + 1], raw_len) / fs
        parts.append(f"{start:.0f}-{end:.0f}s(FP={int(fp_counts[idx])},FN={int(fn_counts[idx])})")
        if len(parts) >= limit:
            break
    return "; ".join(parts)


def evaluate_training_set(
    mat_path=PROJECT_TRAIN_DATA,
    max_len=None,
    verbose=True,
    include_hrv=False,
    hrv_source="predicted",
    calibrate_hrv=False,
):
    # Evaluate every training record against QRSexpert. HRV is optional because
    # QRS tuning usually only needs the detection metrics.
    data = loadmat(mat_path)
    ecg_records = data["ECG"].ravel()
    expert_records = data["QRSexpert"].ravel()
    tolerance = int(0.050 * DEFAULT_FS)

    rows = []
    totals = {"TP": 0, "FP": 0, "FN": 0}

    all_predictions = {}


    for record_number, (raw_cell, expert_cell) in enumerate(zip(ecg_records, expert_records), start=1):
        raw = raw_cell.ravel().astype(float)
        if max_len is not None:
            raw = raw[:max_len]

        expert = expert_cell.ravel().astype(int) - 1
        expert = expert[(expert >= 0) & (expert < len(raw))]

        _filtered, predicted = detect_qrs(raw, DEFAULT_FS)
        metrics = match_qrs(predicted, expert, tolerance)

        # Keep aggregate counts separate from per-record rows so summary metrics
        # are computed from all beats, not from averaged record scores.
        for key in totals:
            totals[key] += metrics[key]

        unmatched_expert = np.setdiff1d(expert, metrics["matched_expert"], assume_unique=False)
        first_error = None
        if len(unmatched_expert):
            first_error = int(unmatched_expert[0])
        elif len(metrics["unmatched_pred"]):
            first_error = int(metrics["unmatched_pred"][0])

        row = {
            "record": record_number,
            "TP": metrics["TP"],
            "FP": metrics["FP"],
            "FN": metrics["FN"],
            "Sensitivity": metrics["Sensitivity"],
            "PPV": metrics["PPV"],
            "F1": metrics["F1"],
            "pred_count": len(predicted),
            "expert_count": len(expert),
            "first_error_sample": first_error,
            "offset_summary": _offset_summary(metrics["matched_pred"], metrics["matched_expert"]),
            "fp_clusters": _fp_cluster_summary(metrics["unmatched_pred"]),
            "error_windows": _error_window_summary(metrics["unmatched_pred"], unmatched_expert, len(raw)),
        }

        if include_hrv:
            if hrv_source == "expert":
                hrv_peaks = expert
            else:
                hrv_peaks = predicted
            hrv_values = compute_windowed_hrv(hrv_peaks, len(raw))
            if calibrate_hrv:
                hrv_values = calibrate_hrv_metrics(hrv_values)
            row.update(hrv_values)

        rows.append(row)

        all_predictions[record_number] = predicted

        if verbose:
            print(
                f"Record {record_number:02d}: "
                f"Sens={row['Sensitivity']:.4f} "
                f"PPV={row['PPV']:.4f} "
                f"F1={row['F1']:.4f} "
                f"TP={row['TP']} FP={row['FP']} FN={row['FN']} "
                f"pred={row['pred_count']} expert={row['expert_count']}"
            )
            if row["offset_summary"] and (
                row["F1"] < 0.995 or record_number in (2, 6, 7, 10, 22, 25, 27, 35)
            ):
                print(f"  offsets pred-expert: {row['offset_summary']}")
            if row["fp_clusters"] and (row["F1"] < 0.995 or row["FP"] >= 50):
                print(f"  FP clusters: {row['fp_clusters']}")
            if row["error_windows"] and row["F1"] < 0.997:
                print(f"  Error windows: {row['error_windows']}")
            if include_hrv:
                print(
                    f"  HRV({hrv_source}): "
                    f"avgRR={row['avgRR']:.1f} "
                    f"RMSSD={row['RMSSD']:.1f} "
                    f"pNN50={row['pNN50']:.1f} "
                    f"LF/HF={row['LF_HFratio']:.2f}"
                )

    np.save("predictions.npy", all_predictions, allow_pickle=True)
    pd.DataFrame(rows).to_csv("metrics.csv", index=False)


    total_sens = totals["TP"] / (totals["TP"] + totals["FN"])
    total_ppv = totals["TP"] / (totals["TP"] + totals["FP"])
    total_f1 = 2 * total_sens * total_ppv / (total_sens + total_ppv)
    summary = {
        "TP": totals["TP"],
        "FP": totals["FP"],
        "FN": totals["FN"],
        "Sensitivity": total_sens,
        "PPV": total_ppv,
        "F1": total_f1,
    }

    print(
        "\nTraining summary: "
        f"Sens={summary['Sensitivity']:.4f} "
        f"PPV={summary['PPV']:.4f} "
        f"F1={summary['F1']:.4f} "
        f"TP={summary['TP']} FP={summary['FP']} FN={summary['FN']}"
    )

    return rows, summary


def generate_record_outputs(
    mat_path=PROJECT_TEST_DATA,
    max_len=None,
    verbose=True,
    include_hrv=False,
    calibrate_hrv=False,
):
    # Generate predictions and optional HRV for datasets that do not include
    # expert QRS annotations, such as the project test set.
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
            hrv_values = compute_windowed_hrv(predicted, len(raw))
            if calibrate_hrv:
                hrv_values = calibrate_hrv_metrics(hrv_values)
            row.update(hrv_values)

        rows.append(row)
        all_predictions[record_number] = predicted

        if verbose:
            message = f"Record {record_number:02d}: pred={row['pred_count']}"
            if include_hrv:
                message += (
                    f" HRV: avgRR={row['avgRR']:.1f} "
                    f"RMSSD={row['RMSSD']:.1f} "
                    f"pNN50={row['pNN50']:.1f} "
                    f"LF/HF={row['LF_HFratio']:.2f}"
                )
            print(message)

    np.save("predictions.npy", all_predictions, allow_pickle=True)
    pd.DataFrame(rows).to_csv("metrics.csv", index=False)
    print(f"\nGenerated outputs for {len(rows)} records")

    return rows


def save_metrics_csv(rows, summary, out_dir):
    # Save the numeric result table so bad records can be sorted and inspected
    # outside the terminal.
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "training_metrics.csv"

    fieldnames = [
        "record",
        "TP",
        "FP",
        "FN",
        "Sensitivity",
        "PPV",
        "F1",
        "pred_count",
        "expert_count",
        "first_error_sample",
        "offset_summary",
        "fp_clusters",
        "error_windows",
    ]
    if rows and all(key in rows[0] for key in HRV_KEYS):
        fieldnames.extend(HRV_KEYS)

    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    summary_path = out_dir / "training_summary.txt"
    summary_path.write_text(
        "\n".join(
            [
                f"Sensitivity: {summary['Sensitivity']:.6f}",
                f"PPV: {summary['PPV']:.6f}",
                f"F1: {summary['F1']:.6f}",
                f"TP: {summary['TP']}",
                f"FP: {summary['FP']}",
                f"FN: {summary['FN']}",
            ]
        )
        + "\n"
    )

    return csv_path, summary_path


def save_hrv_metrics_csv(rows, out_dir, source_label):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    filename = "hrv_metrics.csv" if source_label == "predicted" else f"hrv_{source_label}_metrics.csv"
    csv_path = out_dir / filename

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

    return csv_path


def save_hrv_reference_comparison(rows, reference_path, out_dir, source_label):
    reference_path = Path(reference_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    reference = pd.read_csv(reference_path)
    calculated = pd.DataFrame([{key: row[key] for key in ("record", *HRV_KEYS)} for row in rows])
    merged = reference.merge(calculated, on="record", suffixes=("_ref", f"_{source_label}"))

    detail_rows = []
    summary_rows = []
    for key in HRV_KEYS:
        ref_col = f"{key}_ref"
        calc_col = f"{key}_{source_label}"
        errors = merged[calc_col] - merged[ref_col]
        abs_errors = errors.abs()
        pct_errors = abs_errors / merged[ref_col].abs().replace(0, np.nan) * 100.0

        for _, record in merged.iterrows():
            ref_value = float(record[ref_col])
            calc_value = float(record[calc_col])
            abs_error = abs(calc_value - ref_value)
            pct_error = abs_error / abs(ref_value) * 100.0 if ref_value else np.nan
            detail_rows.append(
                {
                    "record": int(record["record"]),
                    "metric": key,
                    "reference": ref_value,
                    source_label: calc_value,
                    "error": calc_value - ref_value,
                    "abs_error": abs_error,
                    "pct_error": pct_error,
                }
            )

        summary_rows.append(
            {
                "metric": key,
                "MAE": float(abs_errors.mean()),
                "RMSE": float(np.sqrt(np.mean(errors**2))),
                "mean_abs_pct_error": float(pct_errors.mean()),
                "max_abs_error": float(abs_errors.max()),
                "worst_record": int(merged.loc[abs_errors.idxmax(), "record"]),
            }
        )

    average_mape = float(np.mean([row["mean_abs_pct_error"] for row in summary_rows]))
    summary_rows.append(
        {
            "metric": "averageMAPE",
            "MAE": np.nan,
            "RMSE": np.nan,
            "mean_abs_pct_error": average_mape,
            "max_abs_error": np.nan,
            "worst_record": np.nan,
        }
    )

    detail_path = out_dir / f"hrv_{source_label}_vs_reference.csv"
    summary_path = out_dir / f"hrv_{source_label}_vs_reference_summary.csv"
    pd.DataFrame(detail_rows).to_csv(detail_path, index=False)
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
    return detail_path, summary_path, summary_rows


def plot_f1_by_record(rows, out_dir):
    # Quick visual check for records that still need manual inspection.
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    out_path = Path(out_dir) / "f1_by_record.png"
    records = [row["record"] for row in rows]
    f1 = [row["F1"] for row in rows]

    plt.figure(figsize=(13, 4.8))
    colors = ["tab:red" if value < 0.95 else "tab:blue" for value in f1]
    plt.bar(records, f1, color=colors)
    plt.axhline(0.997, color="0.3", linestyle="--", linewidth=1, label="F1 = 0.997")
    plt.ylim(0.94, 1.005)
    plt.xticks(records)
    plt.xlabel("Record")
    plt.ylabel("F1-score")
    plt.title("Training QRS Detection F1 by Record")
    plt.grid(axis="y", alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()
    return out_path


def plot_sensitivity_ppv(rows, out_dir):
    # Separate low-sensitivity problems from low-PPV problems.
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    out_path = Path(out_dir) / "sensitivity_vs_ppv.png"
    sens = [row["Sensitivity"] for row in rows]
    ppv = [row["PPV"] for row in rows]
    records = [row["record"] for row in rows]

    plt.figure(figsize=(6, 6))
    plt.scatter(ppv, sens, s=45, color="tab:blue")
    for record, x, y in zip(records, ppv, sens):
        if x < 0.97 or y < 0.97:
            plt.text(x + 0.002, y + 0.002, str(record), fontsize=9)
    plt.xlim(0.80, 1.005)
    plt.ylim(0.80, 1.005)
    plt.xlabel("Positive Predictivity")
    plt.ylabel("Sensitivity")
    plt.title("Sensitivity vs PPV")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()
    return out_path


def save_worst_overlays(mat_path, rows, out_dir, worst_count=4, length=15_000):
    # Export static overlays for the lowest-F1 records. The interactive viewer
    # is better for debugging, but these files are useful for reports.
    data = loadmat(mat_path)
    ecg_records = data["ECG"].ravel()
    expert_records = data["QRSexpert"].ravel()
    out_paths = []

    worst = sorted(rows, key=lambda row: row["F1"])[:worst_count]
    for row in worst:
        record_number = row["record"]
        raw = ecg_records[record_number - 1].ravel().astype(float)
        expert = expert_records[record_number - 1].ravel().astype(int) - 1
        expert = expert[(expert >= 0) & (expert < len(raw))]
        filtered, predicted = detect_qrs(raw, DEFAULT_FS)

        center = row["first_error_sample"]
        error_window = re.search(r"(\d+)-(\d+)s", str(row.get("error_windows", "")))
        if error_window:
            start_sec = int(error_window.group(1))
            end_sec = int(error_window.group(2))
            center = int((start_sec + end_sec) * 0.5 * DEFAULT_FS)
        start = 0 if center is None else max(0, int(center) - length // 2)
        out_path = Path(out_dir) / f"record_{record_number:02d}_overlay.png"
        save_overlay_plot(
            raw=raw,
            expert=expert,
            filtered=filtered,
            predicted=predicted,
            out_path=out_path,
            record_number=record_number,
            start=start,
            length=length,
        )
        out_paths.append(out_path)

    return out_paths


def save_training_plots(mat_path, rows, summary, out_dir, worst_count=4):
    # Save all evaluation artifacts in one folder so source files stay clean.
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(out_dir / ".matplotlib"))

    outputs = []
    outputs.extend(save_metrics_csv(rows, summary, out_dir))
    outputs.append(plot_f1_by_record(rows, out_dir))
    outputs.append(plot_sensitivity_ppv(rows, out_dir))
    outputs.extend(save_worst_overlays(mat_path, rows, out_dir, worst_count=worst_count))
    return outputs


def save_qrs_overlays(mat_path, out_dir, patient="all", start=0, length=15_000, show_raw=False):
    data = loadmat(mat_path)
    n_records = len(data["ECG"].ravel())
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(out_dir / ".matplotlib")

    if isinstance(patient, str) and patient.lower() == "all":
        patient_numbers = range(1, n_records + 1)
    else:
        patient_numbers = [max(1, min(int(patient), n_records))]

    outputs = []
    for patient_number in patient_numbers:
        raw, expert = load_recording(mat_path, patient_number - 1)
        filtered, predicted = detect_qrs(raw, DEFAULT_FS)
        window_length = parse_length(length, len(raw))
        out_path = out_dir / f"record_{patient_number:02d}_qrs_overlay.png"
        save_overlay_plot(
            raw=raw,
            expert=expert,
            filtered=filtered,
            predicted=predicted,
            out_path=out_path,
            record_number=patient_number,
            start=start,
            length=window_length,
            show_raw=show_raw,
        )
        outputs.append(out_path)
        print(f"Saved {out_path}")

    return outputs


def main():
    # CLI entry point. --viz opens the interactive viewer; otherwise the script
    # runs training-set evaluation.
    parser = argparse.ArgumentParser(description="QRS training evaluation and visualization")
    parser.add_argument("--eval-train", action="store_true", help="run full training-set QRS evaluation")
    parser.add_argument("--save-plots", action="store_true", help="save CSV, metrics plots, and worst overlays")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--mat", type=Path, default=PROJECT_TRAIN_DATA)
    parser.add_argument("--max-len", type=int, default=None, help="optional crop for quick experiments")
    parser.add_argument("--worst-count", type=int, default=4)
    parser.add_argument("--viz", action="store_true", help="open interactive overlay viewer")
    parser.add_argument("--save-overlays", action="store_true", help="save QRS overlay PNGs for one or all records")
    parser.add_argument("--patient", default="1", help="1-based record number, or 'all' for --save-overlays")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--length", type=int, default=15_000, help="number of samples")
    parser.add_argument("--show-raw", action="store_true")
    parser.add_argument("--hrv", action="store_true", help="calculate windowed HRV and save hrv_metrics.csv")
    parser.add_argument(
        "--compare-hrv",
        action="store_true",
        help="compare calculated HRV against the training expert HRV reference CSV",
    )
    parser.add_argument(
        "--hrv-source",
        choices=("predicted", "expert"),
        default="predicted",
        help="which QRS peaks to use for HRV on training data",
    )
    parser.add_argument("--hrv-reference", type=Path, default=DEFAULT_HRV_REFERENCE)
    calibration_group = parser.add_mutually_exclusive_group()
    calibration_group.add_argument(
        "--hrv-calibration",
        dest="hrv_calibration",
        action="store_true",
        help="enable global log-linear HRV output calibration",
    )
    calibration_group.add_argument(
        "--no-hrv-calibration",
        dest="hrv_calibration",
        action="store_false",
        help="disable global log-linear HRV output calibration (default)",
    )
    parser.set_defaults(hrv_calibration=False)
    args = parser.parse_args()

    if args.viz:
        run_qrs_debug_viewer(
            mat_path=args.mat,
            patient=args.patient,
            start=args.start,
            length=args.length,
            max_len=args.max_len,
            show_raw=args.show_raw,
        )
        return

    if args.save_overlays:
        save_qrs_overlays(
            mat_path=args.mat,
            out_dir=args.out_dir,
            patient=args.patient,
            start=args.start,
            length=args.length,
            show_raw=args.show_raw,
        )
        return

    include_hrv = args.hrv or args.compare_hrv
    has_expert_qrs = "QRSexpert" in loadmat(args.mat)

    if has_expert_qrs:
        rows, summary = evaluate_training_set(
            args.mat,
            max_len=args.max_len,
            include_hrv=include_hrv,
            hrv_source=args.hrv_source,
            calibrate_hrv=args.hrv_calibration,
        )
    else:
        if args.compare_hrv:
            raise ValueError("--compare-hrv requires a training MAT file with QRSexpert annotations")
        rows = generate_record_outputs(
            args.mat,
            max_len=args.max_len,
            include_hrv=include_hrv,
            calibrate_hrv=args.hrv_calibration,
        )
        summary = None

    if include_hrv:
        hrv_path = save_hrv_metrics_csv(rows, args.out_dir, args.hrv_source)
        print(f"\nSaved HRV metrics: {hrv_path}")

    if args.compare_hrv:
        detail_path, summary_path, hrv_summary = save_hrv_reference_comparison(
            rows,
            args.hrv_reference,
            args.out_dir,
            args.hrv_source,
        )
        print("\nHRV comparison vs reference:")
        for metric in hrv_summary:
            if metric["metric"] == "averageMAPE":
                print(f"averageMAPE={metric['mean_abs_pct_error']:.4g}")
            else:
                print(
                    f"{metric['metric']}: "
                    f"MAE={metric['MAE']:.4g} "
                    f"RMSE={metric['RMSE']:.4g} "
                    f"MAPE={metric['mean_abs_pct_error']:.4g}% "
                    f"worst=record {metric['worst_record']}"
                )
        print(f"Saved HRV comparison: {detail_path}")
        print(f"Saved HRV comparison summary: {summary_path}")

    if summary is not None and (args.save_plots or args.eval_train):
        if args.save_plots:
            outputs = save_training_plots(
                args.mat,
                rows,
                summary,
                args.out_dir,
                worst_count=args.worst_count,
            )
            print("\nSaved outputs:")
            for path in outputs:
                print(f"- {path}")


if __name__ == "__main__":
    main()
