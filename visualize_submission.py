import argparse
import csv
import os
from pathlib import Path

import numpy as np
from scipy.io import loadmat

from qrs_pipeline import DEFAULT_FS, preprocess_ecg


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_TEST_MAT = BASE_DIR / "dataSet" / "ProjectTestData.mat"
DEFAULT_SUBMISSION_MAT = BASE_DIR / "dataSet" / "ProjectTestDataAnalysis_filled.mat"
DEFAULT_OUT_DIR = BASE_DIR / "outputs" / "submission_visualization"
HRV_KEYS = ("avgRR", "sdRR", "RMSSD", "pNN50", "LF", "HF", "LF_HFratio")


def parse_length(value, raw_len=None):
    return int(value)


def load_submission(test_mat, submission_mat):
    test_data = loadmat(test_mat)
    submission = loadmat(submission_mat)

    if "ECG" not in test_data:
        raise KeyError(f"{test_mat} does not contain ECG")
    if "QRS" not in submission:
        raise KeyError(f"{submission_mat} does not contain QRS")

    ecg_records = test_data["ECG"].ravel()
    qrs_records = submission["QRS"].ravel()
    if len(ecg_records) != len(qrs_records):
        raise ValueError(
            f"Record count mismatch: ECG has {len(ecg_records)}, QRS has {len(qrs_records)}"
        )

    return ecg_records, qrs_records, submission


def submitted_qrs_zero_based(qrs_cell, raw_len):
    qrs = np.asarray(qrs_cell).ravel().astype(np.int64)
    # Submission MAT uses MATLAB-style 1-based sample indices.
    qrs = qrs - 1
    return qrs[(qrs >= 0) & (qrs < raw_len)]


def hrv_for_record(submission, record_index):
    values = {}
    for key in HRV_KEYS:
        if key in submission:
            values[key] = float(np.asarray(submission[key]).reshape(-1)[record_index])
    return values


def decimated_slice(values, start, end, max_points):
    values = np.asarray(values)
    step = max(1, int(np.ceil((end - start) / max_points)))
    idx = np.arange(start, end, step, dtype=int)
    return idx, values[idx]


def validate_records(ecg_records, qrs_records, submission):
    rows = []
    for record_index, (raw_cell, qrs_cell) in enumerate(zip(ecg_records, qrs_records), start=1):
        raw_len = len(np.asarray(raw_cell).ravel())
        qrs_one_based = np.asarray(qrs_cell).ravel().astype(np.int64)
        qrs_zero_based = qrs_one_based - 1
        in_bounds = (qrs_zero_based >= 0) & (qrs_zero_based < raw_len)
        decreases = int(np.sum(np.diff(qrs_one_based) < 0)) if len(qrs_one_based) else 0

        row = {
            "record": record_index,
            "raw_len": raw_len,
            "qrs_count": len(qrs_one_based),
            "first_qrs_1based": int(qrs_one_based[0]) if len(qrs_one_based) else "",
            "last_qrs_1based": int(qrs_one_based[-1]) if len(qrs_one_based) else "",
            "out_of_bounds_qrs": int((~in_bounds).sum()),
            "decreasing_steps": decreases,
        }
        row.update(hrv_for_record(submission, record_index - 1))
        rows.append(row)
    return rows


def save_summary_csv(rows, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "submission_summary.csv"
    fieldnames = [
        "record",
        "raw_len",
        "qrs_count",
        "first_qrs_1based",
        "last_qrs_1based",
        "out_of_bounds_qrs",
        "decreasing_steps",
        *HRV_KEYS,
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    return path


def plot_submission_record(
    raw,
    qrs,
    hrv,
    out_path,
    record_number,
    start=0,
    length=15_000,
    show_raw=False,
    max_line_points=200_000,
):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(out_path.parent / ".matplotlib")
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    raw = np.asarray(raw, dtype=float).ravel()
    qrs = np.asarray(qrs, dtype=int)
    filtered = preprocess_ecg(raw, DEFAULT_FS)

    start = max(0, min(int(start), len(raw) - 1))
    length = parse_length(length, len(raw))
    end = min(start + int(length), len(raw))
    line_idx, filtered_plot = decimated_slice(filtered, start, end, max_line_points)
    t_line = line_idx / DEFAULT_FS

    qrs_vis = qrs[(qrs >= start) & (qrs < end)]

    fig, ax = plt.subplots(figsize=(18, 6))
    if show_raw:
        _raw_idx, raw_plot = decimated_slice(raw, start, end, max_line_points)
        ax.plot(_raw_idx / DEFAULT_FS, raw_plot, color="#9ecae1", linewidth=0.55, alpha=0.55, label="Raw ECG")

    ax.plot(t_line, filtered_plot, color="0.15", linewidth=0.7, label="Filtered ECG")
    if len(qrs_vis):
        ax.scatter(
            qrs_vis / DEFAULT_FS,
            filtered[qrs_vis],
            s=12 if len(qrs_vis) > 5000 else 28,
            c="tab:orange",
            marker="v",
            label=f"Submitted QRS ({len(qrs_vis)})",
            zorder=5,
        )

    hrv_text = "\n".join(
        f"{key}: {value:.4g}"
        for key, value in hrv.items()
        if np.isfinite(value)
    )
    if hrv_text:
        ax.text(
            0.012,
            0.98,
            hrv_text,
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=9,
            bbox={"facecolor": "white", "alpha": 0.78, "edgecolor": "0.8"},
        )

    ax.set_title(f"Submission MAT QRS overlay | record {record_number} | samples [{start}:{end})")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude (filtered ECG)")
    ax.grid(alpha=0.22)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return out_path


def save_overlays(test_mat, submission_mat, out_dir, patient="all", start=0, length=15_000, show_raw=False):
    ecg_records, qrs_records, submission = load_submission(test_mat, submission_mat)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if isinstance(patient, str) and patient.lower() == "all":
        patient_numbers = range(1, len(ecg_records) + 1)
    else:
        patient_numbers = [max(1, min(int(patient), len(ecg_records)))]

    outputs = []
    for patient_number in patient_numbers:
        record_index = patient_number - 1
        raw = np.asarray(ecg_records[record_index]).ravel().astype(float)
        qrs = submitted_qrs_zero_based(qrs_records[record_index], len(raw))
        hrv = hrv_for_record(submission, record_index)
        out_path = out_dir / f"submission_record_{patient_number:02d}.png"
        outputs.append(
            plot_submission_record(
                raw=raw,
                qrs=qrs,
                hrv=hrv,
                out_path=out_path,
                record_number=patient_number,
                start=start,
                length=length,
                show_raw=show_raw,
            )
        )
        print(f"Saved {out_path}")

    return outputs


def show_interactive(test_mat, submission_mat, patient=1, start=0, length=15_000, show_raw=False):
    import matplotlib.pyplot as plt
    from matplotlib.widgets import Button, Slider

    ecg_records, qrs_records, submission = load_submission(test_mat, submission_mat)
    n_records = len(ecg_records)
    patient = max(1, min(int(patient), n_records))
    current = [patient - 1]
    cache = {}

    def get_record(record_index):
        if record_index not in cache:
            raw = np.asarray(ecg_records[record_index]).ravel().astype(float)
            filtered = preprocess_ecg(raw, DEFAULT_FS)
            qrs = submitted_qrs_zero_based(qrs_records[record_index], len(raw))
            hrv = hrv_for_record(submission, record_index)
            cache[record_index] = raw, filtered, qrs, hrv
        return cache[record_index]

    def clamp(win_start, win_len, n_samples):
        win_len = max(500, min(int(win_len), n_samples))
        win_start = max(0, min(int(win_start), max(0, n_samples - win_len)))
        return win_start, win_len

    fig, ax = plt.subplots(figsize=(15, 7))
    plt.subplots_adjust(bottom=0.20)
    slider_start_ax = fig.add_axes((0.13, 0.09, 0.58, 0.03))
    slider_len_ax = fig.add_axes((0.13, 0.045, 0.58, 0.03))
    prev_ax = fig.add_axes((0.75, 0.075, 0.10, 0.05))
    next_ax = fig.add_axes((0.86, 0.075, 0.10, 0.05))

    raw0, _filtered0, _qrs0, _hrv0 = get_record(current[0])
    start0, length0 = clamp(start, parse_length(length, len(raw0)), len(raw0))
    slider_start = Slider(slider_start_ax, "Start", 0, max(len(raw0) - 1, 1), valinit=start0, valstep=1)
    slider_len = Slider(slider_len_ax, "Window", 500, max(len(raw0), 500), valinit=length0, valstep=100)
    btn_prev = Button(prev_ax, "Prev")
    btn_next = Button(next_ax, "Next")

    def draw():
        raw, filtered, qrs, hrv = get_record(current[0])
        start_i, length_i = clamp(slider_start.val, slider_len.val, len(raw))
        end = min(start_i + length_i, len(raw))
        idx, filtered_plot = decimated_slice(filtered, start_i, end, 120_000)
        qrs_vis = qrs[(qrs >= start_i) & (qrs < end)]

        ax.clear()
        if show_raw:
            raw_idx, raw_plot = decimated_slice(raw, start_i, end, 120_000)
            ax.plot(raw_idx / DEFAULT_FS, raw_plot, color="#9ecae1", linewidth=0.55, alpha=0.55, label="Raw ECG")
        ax.plot(idx / DEFAULT_FS, filtered_plot, color="0.15", linewidth=0.75, label="Filtered ECG")
        if len(qrs_vis):
            ax.scatter(qrs_vis / DEFAULT_FS, filtered[qrs_vis], s=18, c="tab:orange", marker="v", label=f"Submitted QRS ({len(qrs_vis)})")
        hrv_text = " | ".join(f"{key}={value:.4g}" for key, value in hrv.items() if np.isfinite(value))
        ax.set_title(
            f"Submission MAT | record {current[0] + 1}/{n_records} | samples [{start_i}:{end})\n{hrv_text}"
        )
        ax.set_xlabel("Time (s)")
        ax.grid(alpha=0.22)
        ax.legend(loc="upper right")
        fig.canvas.draw_idle()

    def set_record(delta):
        current[0] = (current[0] + delta) % n_records
        raw, _filtered, _qrs, _hrv = get_record(current[0])
        slider_start.valmax = max(len(raw) - 1, 1)
        slider_start.ax.set_xlim(slider_start.valmin, slider_start.valmax)
        slider_len.valmax = max(len(raw), 500)
        slider_len.ax.set_xlim(slider_len.valmin, slider_len.valmax)
        draw()

    slider_start.on_changed(lambda _value: draw())
    slider_len.on_changed(lambda _value: draw())
    btn_prev.on_clicked(lambda _event: set_record(-1))
    btn_next.on_clicked(lambda _event: set_record(1))
    draw()
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Visualize the final submission MAT file.")
    parser.add_argument("--test-mat", type=Path, default=DEFAULT_TEST_MAT)
    parser.add_argument("--submission-mat", type=Path, default=DEFAULT_SUBMISSION_MAT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--patient", default="1", help="1-based record number, or 'all' for --save-overlays")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--length", type=int, default=15_000, help="number of samples")
    parser.add_argument("--show-raw", action="store_true")
    parser.add_argument("--viz", action="store_true", help="open interactive viewer for one record")
    parser.add_argument("--save-overlays", action="store_true", help="save PNG overlays for one or all records")
    parser.add_argument("--summary", action="store_true", help="write a CSV summary of submitted QRS/HRV values")
    args = parser.parse_args()

    if args.summary:
        ecg_records, qrs_records, submission = load_submission(args.test_mat, args.submission_mat)
        path = save_summary_csv(validate_records(ecg_records, qrs_records, submission), args.out_dir)
        print(f"Saved {path}")

    if args.save_overlays:
        save_overlays(
            test_mat=args.test_mat,
            submission_mat=args.submission_mat,
            out_dir=args.out_dir,
            patient=args.patient,
            start=args.start,
            length=args.length,
            show_raw=args.show_raw,
        )

    if args.viz:
        if isinstance(args.patient, str) and args.patient.lower() == "all":
            raise ValueError("--viz supports one patient at a time; use --save-overlays for --patient all")
        show_interactive(
            test_mat=args.test_mat,
            submission_mat=args.submission_mat,
            patient=args.patient,
            start=args.start,
            length=args.length,
            show_raw=args.show_raw,
        )

    if not (args.summary or args.save_overlays or args.viz):
        parser.print_help()


if __name__ == "__main__":
    main()
