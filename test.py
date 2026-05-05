import argparse
import csv
import os
from pathlib import Path

import numpy as np
from scipy.io import loadmat

from qrs_pipeline import (
    DEFAULT_FS,
    default_train_mat,
    detect_qrs,
    save_overlay_plot,
)
from qrs_debug_viewer import run_qrs_debug_viewer

PROJECT_TRAIN_DATA = default_train_mat()
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "outputs" / "qrs_eval"


def calculate_hr_hrv(peaks):
    rr = np.diff(peaks) / DEFAULT_FS
    hr = 60 / np.mean(rr)

    rr_ms = rr * 1000
    rmssd = np.sqrt(np.mean(np.diff(rr_ms) ** 2))

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


def evaluate_training_set(mat_path=PROJECT_TRAIN_DATA, max_len=None, verbose=True):
    # Evaluate every training record against QRSexpert. This does not generate
    # a test submission and does not call the HRV code.
    data = loadmat(mat_path)
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
            if row["offset_summary"] and (
                row["F1"] < 0.995 or record_number in (2, 6, 7, 10, 22, 25, 27, 35)
            ):
                print(f"  offsets pred-expert: {row['offset_summary']}")
            if row["fp_clusters"] and (row["F1"] < 0.995 or row["FP"] >= 50):
                print(f"  FP clusters: {row['fp_clusters']}")

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
    ]
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
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
