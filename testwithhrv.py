import argparse
import csv
from pathlib import Path

import numpy as np
from scipy.io import loadmat
from scipy.interpolate import interp1d
from scipy.signal import periodogram
from scipy.integrate import trapezoid

from qrs_pipeline import (
    DEFAULT_FS,
    default_train_mat,
    detect_qrs,
    save_overlay_plot,
)
from qrs_debug_viewer import run_qrs_debug_viewer

PROJECT_TRAIN_DATA = default_train_mat()
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "outputs" / "qrs_eval"

def clean_rr_intervals(rr_intervals,
                       min_rr=0.3,
                       max_rr=2.0,
                       max_deviation=0.20):
    """
    Remove implausible RR intervals caused by missed or false QRS detections.

    Parameters
    ----------
    rr_intervals : ndarray
        RR intervals in seconds

    min_rr : float
        Minimum physiologic RR interval (seconds)

    max_rr : float
        Maximum physiologic RR interval (seconds)

    max_deviation : float
        Maximum allowed deviation from median RR
        (0.20 = ±20%)
    """

    rr = np.asarray(rr_intervals, dtype=float)

    if len(rr) < 3:
        return rr

    # --- physiologic range filter ---
    valid = (rr >= min_rr) & (rr <= max_rr)
    rr = rr[valid]

    if len(rr) < 3:
        return rr

    # --- median-based artifact rejection ---
    median_rr = np.median(rr)

    valid = (
        (rr >= median_rr * (1.0 - max_deviation)) &
        (rr <= median_rr * (1.0 + max_deviation))
    )

    rr = rr[valid]

    return rr


def _estimate_lf_hf_power(
        rr_intervals,
        min_f_LF=0.04,
        max_f_LF=0.15,
        min_f_HF=0.15,
        max_f_HF=0.4,
        fs_resample=4.0):
    """
    Standard HRV LF/HF estimation with RR cleaning.

    Steps:
      1. Remove RR artifacts
      2. Create RR timestamps
      3. Cubic interpolation to uniform sampling
      4. Mean detrend
      5. Periodogram PSD
      6. Integrate LF/HF bands
    """

    # ------------------------------------------------------------------
    # CLEAN RR INTERVALS
    # ------------------------------------------------------------------
    rr_intervals = clean_rr_intervals(rr_intervals)

    if len(rr_intervals) < 4:
        return 0.0, 0.0

    # ------------------------------------------------------------------
    # RR timestamps
    # ------------------------------------------------------------------
    time_stamps = np.concatenate([[0.0], np.cumsum(rr_intervals[:-1])])

    rr_ms = rr_intervals * 1000.0

    # ------------------------------------------------------------------
    # Uniform interpolation (4 Hz standard HRV)
    # ------------------------------------------------------------------
    t_uniform = np.arange(
        time_stamps[0],
        time_stamps[-1],
        1.0 / fs_resample
    )

    if len(t_uniform) < 8:
        return 0.0, 0.0

    interpolator = interp1d(
        time_stamps,
        rr_ms,
        kind="cubic",
        bounds_error=False,
        fill_value="extrapolate"
    )

    rr_resampled = interpolator(t_uniform)


    rr_resampled = rr_resampled - np.mean(rr_resampled)

    freqs, psd = periodogram(
        rr_resampled,
        fs=fs_resample,
        scaling="density"
    )

    # ------------------------------------------------------------------
    # LF/HF bands
    # ------------------------------------------------------------------
    lf_mask = (
        (freqs >= min_f_LF) &
        (freqs < max_f_LF)
    )

    hf_mask = (
        (freqs >= min_f_HF) &
        (freqs <= max_f_HF)
    )

    LF_power = (
        float(trapezoid(psd[lf_mask], freqs[lf_mask]))
        if np.any(lf_mask) else 0.0
    )

    HF_power = (
        float(trapezoid(psd[hf_mask], freqs[hf_mask]))
        if np.any(hf_mask) else 0.0
    )

    return LF_power, HF_power


def _compute_rr_intervals(peaks, fs=DEFAULT_FS):
    peaks = np.asarray(peaks, dtype=float)
    if len(peaks) < 2:
        return np.asarray([], dtype=float)
    return np.diff(peaks) / fs


def _estimate_lf_hf_power(rr_intervals, min_f_LF=0.04, max_f_LF=0.15, max_f_HF=0.4, fs_resample=100.0):
    """
    Standard HRV frequency-domain method (matches pyHRV / hrvanalysis):
      1. Place RR timestamps starting at t=0
      2. Cubic-spline interpolate onto a uniform 100 Hz grid
      3. Detrend by subtracting the mean
      4. Compute PSD via periodogram (single FFT, no windowing/segmenting)
         so frequency resolution = fs_resample / N_resampled — fine enough
         to resolve the narrow HF band even for short recordings
      5. Integrate LF and HF bands with the trapezoid rule
    """
    if len(rr_intervals) < 2:
        return 0.0, 0.0

    # --- 1. timestamps starting at 0 ---
    time_stamps = np.concatenate([[0.0], np.cumsum(rr_intervals[:-1])])
    rr_ms = rr_intervals * 1000.0

    # --- 2. cubic-spline interpolation onto uniform 100 Hz grid ---
    t_uniform = np.arange(time_stamps[0], time_stamps[-1], 1.0 / fs_resample)
    if len(t_uniform) < 8:
        return 0.0, 0.0

    interpolator = interp1d(time_stamps, rr_ms, kind="cubic",
                            bounds_error=False,
                            fill_value=(rr_ms[0], rr_ms[-1]))
    rr_resampled = interpolator(t_uniform)

    # --- 3. detrend ---
    rr_resampled -= np.mean(rr_resampled)

    # --- 4. periodogram with full-length FFT (no Welch segmenting) ---
    freqs, psd = periodogram(rr_resampled, fs=fs_resample, scaling="density")

    # --- 5. band integration ---
    lf_mask = (freqs >= min_f_LF) & (freqs <= max_f_LF)
    hf_mask = (freqs >= max_f_LF) & (freqs <= max_f_HF)

    LF_power = float(trapezoid(psd[lf_mask], freqs[lf_mask])) if lf_mask.any() else 0.0
    HF_power = float(trapezoid(psd[hf_mask], freqs[hf_mask])) if hf_mask.any() else 0.0

    return LF_power, HF_power


def calculate_hr_hrv(peaks, detailed=False):
    rr_raw = _compute_rr_intervals(peaks)
    rr = clean_rr_intervals(rr_raw)
    if len(rr) < 1:
        return (0.0, 0.0) if not detailed else {
            "HR": 0.0,
            "RMSSD": 0.0,
            "SDNN": 0.0,
            "pNN50": 0.0,
            "LF_power": 0.0,
            "HF_power": 0.0,
        }

    hr = 60.0 / np.mean(rr)
    rr_ms = rr * 1000
    rmssd = float(np.sqrt(np.mean(np.diff(rr_ms) ** 2))) if len(rr_ms) > 1 else 0.0

    if not detailed:
        return float(hr), float(rmssd)

    sdnn = float(np.std(rr_ms, ddof=1)) if len(rr) > 1 else 0.0
    pnn50 = float(np.sum(np.abs(np.diff(rr)) > 0.05) / max(1, len(rr) - 1) * 100.0)
    lf_power, hf_power = _estimate_lf_hf_power(rr)
    lf_hf_ratio = float(lf_power / hf_power) if hf_power else 0.0

    return {
        "HR": float(hr),
        "RMSSD": float(rmssd),
        "SDNN": sdnn,
        "pNN50": pnn50,
        "LF_power": lf_power,
        "HF_power": hf_power,
        "LF_HF_ratio": lf_hf_ratio,
    }


def match_qrs(predicted, expert, tolerance_samples=5):
    predicted = np.asarray(predicted, dtype=int)
    expert = np.asarray(expert, dtype=int)

    predicted.sort()
    expert.sort()

    used = np.zeros(len(predicted), dtype=bool)
    matched_pred = []
    matched_expert = []
    search_start = 0

    for expert_peak in expert:
        while search_start < len(predicted) and predicted[search_start] < expert_peak - tolerance_samples:
            search_start += 1

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


def evaluate_training_set(mat_path=PROJECT_TRAIN_DATA, max_len=None, verbose=True):
    try:
        data = loadmat(mat_path)
    except OSError:
        print(f"Error: The file '{mat_path}' was not found.")
        print("Please ensure the dataSet folder with ProjectTrainData.mat is present in the project directory.")
        print("Refer to the README.md for the expected data layout.")
        return [], {}
    ecg_records = data["ECG"].ravel()
    expert_records = data["QRSexpert"].ravel()
    tolerance = int(0.050 * DEFAULT_FS)

    rows = []
    totals = {"TP": 0, "FP": 0, "FN": 0}

    for record_number, (raw_cell, expert_cell) in enumerate(zip(ecg_records, expert_records), start=1):
        raw = raw_cell.ravel().astype(float)
        if max_len is not None:
            raw = raw[:max_len]

        expert = expert_cell.ravel().astype(int) - 1
        expert = expert[(expert >= 0) & (expert < len(raw))]

        _filtered, predicted = detect_qrs(raw, DEFAULT_FS)
        metrics = match_qrs(predicted, expert, tolerance)

        rr_raw = _compute_rr_intervals(predicted)
        rr = clean_rr_intervals(rr_raw)
        rr_ms = rr * 1000.0

        avr_rr_int = float(np.mean(rr_ms)) if len(rr) else 0.0
        stdev_rr_int = float(np.std(rr_ms, ddof=1)) if len(rr) > 1 else 0.0
        rmssd = float(np.sqrt(np.mean(np.diff(rr_ms) ** 2))) if len(rr) > 1 else 0.0
        pnn50 = float(np.sum(np.abs(np.diff(rr)) > 0.05) / max(1, len(rr) - 1) * 100.0) if len(rr) > 1 else 0.0
        LF_power, HF_power = _estimate_lf_hf_power(rr)
        LF_HF_ratio = float(LF_power / HF_power) if HF_power else 0.0

        print(f"avgRR={avr_rr_int:.6f} ms")
        print(f"sdRR={stdev_rr_int:.6f} ms")
        print(f"RMSSD={rmssd:.6f} ms")
        print(f"pNN50={pnn50:.6f} %")
        print(f"LF={LF_power:.6f} ms2")
        print(f"HF={HF_power:.6f} ms2")
        print(f"LF_HF_ratio={LF_HF_ratio:.6f}")

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
        }
        rows.append(row)

        if verbose:
            print(
                f"Record {record_number:02d}: "
                f"Sens={row['Sensitivity']:.4f} "
                f"PPV={row['PPV']:.4f} "
                f"F1={row['F1']:.4f} "
                f"TP={row['TP']} FP={row['FP']} FN={row['FN']} "
                f"pred={row['pred_count']} expert={row['expert_count']}"
            )

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


def save_metrics_csv(rows, summary, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "training_metrics.csv"

    fieldnames = [
        "record",
        "TP", "FP", "FN",
        "Sensitivity", "PPV", "F1",
        "pred_count", "expert_count",
        "first_error_sample",
    ]
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    summary_path = out_dir / "training_summary.txt"
    summary_path.write_text(
        "\n".join([
            f"Sensitivity: {summary['Sensitivity']:.6f}",
            f"PPV: {summary['PPV']:.6f}",
            f"F1: {summary['F1']:.6f}",
            f"TP: {summary['TP']}",
            f"FP: {summary['FP']}",
            f"FN: {summary['FN']}",
        ]) + "\n"
    )

    return csv_path, summary_path


def plot_f1_by_record(rows, out_dir):
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    out_path = Path(out_dir) / "f1_by_record.png"
    records = [row["record"] for row in rows]
    f1 = [row["F1"] for row in rows]

    plt.figure(figsize=(13, 4.8))
    colors = ["tab:red" if v < 0.95 else "tab:blue" for v in f1]
    plt.bar(records, f1, color=colors)
    plt.axhline(0.99, color="0.3", linestyle="--", linewidth=1, label="F1 = 0.99")
    plt.ylim(0.80, 1.005)
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
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    outputs = []
    outputs.extend(save_metrics_csv(rows, summary, out_dir))
    outputs.append(plot_f1_by_record(rows, out_dir))
    outputs.append(plot_sensitivity_ppv(rows, out_dir))
    outputs.extend(save_worst_overlays(mat_path, rows, out_dir, worst_count=worst_count))
    return outputs


def main():
    parser = argparse.ArgumentParser(description="QRS training evaluation and visualization")
    parser.add_argument("--eval-train", action="store_true", help="run full training-set QRS evaluation")
    parser.add_argument("--save-plots", action="store_true", help="save CSV, metrics plots, and worst overlays")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--mat", type=Path, default=PROJECT_TRAIN_DATA)
    parser.add_argument("--max-len", type=int, default=None, help="optional crop for quick experiments")
    parser.add_argument("--worst-count", type=int, default=4)
    parser.add_argument("--viz", action="store_true", help="open interactive overlay viewer")
    parser.add_argument("--patient", type=int, default=1, help="1-based record number for --viz")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--length", type=int, default=15_000)
    parser.add_argument("--show-raw", action="store_true")
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

    rows, summary = evaluate_training_set(args.mat, max_len=args.max_len)
    if args.save_plots or args.eval_train:
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