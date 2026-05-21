import argparse
import csv
from pathlib import Path

import numpy as np
from scipy.io import loadmat, savemat
from scipy.signal import welch


from qrs_pipeline import (
    DEFAULT_FS,
    default_test_mat,
    detect_qrs,
    save_overlay_plot,
)
from qrs_debug_viewer import run_qrs_debug_viewer

PROJECT_TRAIN_DATA = default_test_mat()
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "outputs" / "qrs_eval"

# 5-minute window in samples
WINDOW_SAMPLES = int(5 * 60 * DEFAULT_FS)

# Minimum number of clean RR intervals required to trust a window
MIN_RR_PER_WINDOW = 20

# Minimum accumulated clean RR interval duration required by the project rubric
# for a 5-minute window to count toward HRV.
MIN_UNFLAGGED_RR_SECONDS_PER_WINDOW = 4 * 60


# =============================================================================
# RR INTERVAL UTILITIES
# =============================================================================

def _compute_rr_intervals(peaks, fs=DEFAULT_FS):
    """Convert R-peak sample indices → RR intervals in seconds."""
    peaks = np.asarray(peaks, dtype=float)
    if len(peaks) < 2:
        return np.asarray([], dtype=float)
    return np.diff(peaks) / fs


def clean_rr_intervals(rr, min_rr=0.25, max_rr=2.0):
    rr = np.asarray(rr, dtype=float)
    return rr[(rr >= min_rr) & (rr <= max_rr)]

# =============================================================================
# LF / HF POWER  (pure scipy, no HRV libraries)
# =============================================================================

def _estimate_lf_hf_power(rr_intervals,
                           min_f_LF=0.04,
                           max_f_LF=0.15,
                           max_f_HF=0.40,
                           fs_resample=4.0):
    """
    Estimate LF and HF spectral power from cleaned RR intervals (seconds).

    Pipeline (Task Force standard):
      1. Convert RR to milliseconds
      2. Build timestamps starting at t=0 (seconds)
      3. Linearly interpolate onto uniform 4 Hz grid
      4. Subtract mean (remove DC)
      5. Welch PSD with nperseg=full signal (= periodogram, max resolution)
      6. Trapezoid-integrate LF (0.04-0.15 Hz) and HF (0.15-0.40 Hz) → ms²

    Parameters
    ----------
    rr_intervals : array of cleaned RR intervals in seconds

    Returns
    -------
    LF_power, HF_power : floats in ms²
    """
    rr_intervals = np.asarray(rr_intervals, dtype=float)
    if len(rr_intervals) < 10:
        return 0.0, 0.0

    rr_ms = rr_intervals * 1000.0

    # timestamps in seconds, t[0]=0
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
        scaling="density"
    )   

    lf_mask = (freqs >= min_f_LF) & (freqs < max_f_LF)
    hf_mask = (freqs >= max_f_LF) & (freqs <= max_f_HF)

    LF_power = float(np.trapezoid(psd[lf_mask], freqs[lf_mask])) if lf_mask.any() else 0.0
    HF_power = float(np.trapezoid(psd[hf_mask], freqs[hf_mask])) if hf_mask.any() else 0.0

    return LF_power, HF_power


# =============================================================================
# SINGLE-WINDOW HRV
# =============================================================================

def _hrv_for_window(window_peaks, fs=DEFAULT_FS):
    """
    Compute all HRV parameters for one 5-minute window of peaks.

    Returns
    -------
    dict with avgRR, sdRR, RMSSD, pNN50, LF, HF, LF_HF  — or None if the
    window is too noisy / too short to be reliable.
    """
    rr_raw = _compute_rr_intervals(window_peaks, fs)

    if len(rr_raw) < MIN_RR_PER_WINDOW:
        return None   # not enough beats

    rr = clean_rr_intervals(rr_raw)
    if len(rr) < MIN_RR_PER_WINDOW:
        return None

    if np.sum(rr) < MIN_UNFLAGGED_RR_SECONDS_PER_WINDOW:
        return None

    rr_ms = rr * 1000.0

    avg_rr = float(np.mean(rr_ms))
    sd_rr  = float(np.std(rr_ms, ddof=1))
    rmssd  = float(np.sqrt(np.mean(np.diff(rr_ms) ** 2))) if len(rr_ms) > 1 else 0.0
    pnn50  = float(np.sum(np.abs(np.diff(rr_ms)) > 50.0)
                   / max(1, len(rr_ms) - 1) * 100.0)

    lf, hf = _estimate_lf_hf_power(rr)
    lf_hf  = float(lf / hf) if hf > 0 else 0.0

    return {
        "avgRR": avg_rr,
        "sdRR":  sd_rr,
        "RMSSD": rmssd,
        "pNN50": pnn50,
        "LF":    lf,
        "HF":    hf,
        "LF_HF": lf_hf,
    }


# =============================================================================
# WINDOWED HRV FOR ONE RECORD
# =============================================================================

def compute_windowed_hrv(predicted, raw_len, fs=DEFAULT_FS,
                         window_samples=WINDOW_SAMPLES):
    """
    Split predicted peaks into non-overlapping 5-minute windows, compute HRV
    per window, then average across valid windows.

    Parameters
    ----------
    predicted     : array of R-peak sample indices for the full record
    raw_len       : total number of samples in the record
    fs            : sampling rate (Hz)
    window_samples: samples per window (default 5 min × fs)

    Returns
    -------
    dict of averaged HRV parameters, or dict of NaNs if no valid window found.
    keys: avgRR, sdRR, RMSSD, pNN50, LF, HF, LF_HF
    """
    predicted = np.asarray(predicted, dtype=int)
    nan_result = {k: np.nan for k in
                  ("avgRR", "sdRR", "RMSSD", "pNN50", "LF", "HF", "LF_HF")}

    window_results = []
    window_start = 0

    while window_start < raw_len:
        window_end = window_start + window_samples

        # peaks that fall inside this window
        mask = (predicted >= window_start) & (predicted < window_end)
        window_peaks = predicted[mask]

        result = _hrv_for_window(window_peaks, fs)
        if result is not None:
            window_results.append(result)

        window_start = window_end

    if not window_results:
        return nan_result

    averaged = {}

    for key in window_results[0]:
        values = [w[key] for w in window_results]

        averaged[key] = float(np.mean(values))

    lf_hf_values = [
        w["LF_HF"]
        for w in window_results
        if w["HF"] > 0
    ]

    averaged["LF_HF"] = (
        float(np.mean(lf_hf_values))
        if lf_hf_values else 0.0
    )

    return averaged

# =============================================================================
# HR + HRV  (used externally, e.g. in the debug viewer)
# =============================================================================

def calculate_hr_hrv(peaks, detailed=False):
    rr = clean_rr_intervals(_compute_rr_intervals(peaks))

    if len(rr) < 1:
        return (0.0, 0.0) if not detailed else {
            "HR": 0.0, "RMSSD": 0.0, "SDNN": 0.0,
            "pNN50": 0.0, "LF_power": 0.0, "HF_power": 0.0,
        }

    hr    = 60.0 / np.mean(rr)
    rr_ms = rr * 1000.0
    rmssd = float(np.sqrt(np.mean(np.diff(rr_ms) ** 2))) if len(rr_ms) > 1 else 0.0

    if not detailed:
        return float(hr), float(rmssd)

    sdnn  = float(np.std(rr_ms, ddof=1)) if len(rr) > 1 else 0.0
    pnn50 = float(np.sum(np.abs(np.diff(rr_ms)) > 50.0)
                  / max(1, len(rr_ms) - 1) * 100.0)
    lf_power, hf_power = _estimate_lf_hf_power(rr)
    lf_hf_ratio = float(lf_power / hf_power) if hf_power else 0.0

    return {
        "HR":          float(hr),
        "RMSSD":       float(rmssd),
        "SDNN":        sdnn,
        "pNN50":       pnn50,
        "LF_power":    lf_power,
        "HF_power":    hf_power,
        "LF_HF_ratio": lf_hf_ratio,
    }


# =============================================================================
# QRS MATCHING
# =============================================================================

def match_qrs(predicted, expert, tolerance_samples=5):
    predicted = np.asarray(predicted, dtype=int)
    expert    = np.asarray(expert,    dtype=int)
    predicted.sort()
    expert.sort()

    used           = np.zeros(len(predicted), dtype=bool)
    matched_pred   = []
    matched_expert = []
    search_start   = 0

    for expert_peak in expert:
        while (search_start < len(predicted)
               and predicted[search_start] < expert_peak - tolerance_samples):
            search_start += 1

        best_idx, best_dist = -1, tolerance_samples + 1
        idx = search_start
        while idx < len(predicted) and predicted[idx] <= expert_peak + tolerance_samples:
            if not used[idx]:
                dist = abs(predicted[idx] - expert_peak)
                if dist < best_dist:
                    best_dist, best_idx = dist, idx
            idx += 1

        if best_idx >= 0:
            used[best_idx] = True
            matched_pred.append(predicted[best_idx])
            matched_expert.append(expert_peak)

    tp = len(matched_pred)
    fp = int((~used).sum())
    fn = int(len(expert) - tp)

    sens = tp / (tp + fn) if tp + fn else 0.0
    ppv  = tp / (tp + fp) if tp + fp else 0.0
    f1   = 2 * sens * ppv / (sens + ppv) if sens + ppv else 0.0

    return {
        "TP": tp, "FP": fp, "FN": fn,
        "Sensitivity": sens, "PPV": ppv, "F1": f1,
        "unmatched_pred":  predicted[~used],
        "matched_pred":    np.asarray(matched_pred,   dtype=int),
        "matched_expert":  np.asarray(matched_expert, dtype=int),
    }


# =============================================================================
# TRAINING SET EVALUATION
# =============================================================================

def evaluate_training_set(mat_path=PROJECT_TRAIN_DATA,
                          max_len=None,
                          verbose=True):

    try:
        data = loadmat(mat_path)
    except OSError:
        print(f"Error: '{mat_path}' not found.")
        return []

    ecg_records = data["ECG"].ravel()

    rows = []
    all_predictions = {}

    for record_number, raw_cell in enumerate(ecg_records, start=1):

        raw = raw_cell.ravel().astype(float)

        if max_len is not None:
            raw = raw[:max_len]

        # detect peaks
        _filtered, predicted = detect_qrs(raw, DEFAULT_FS)

        # compute HRV from predicted peaks
        hrv = compute_windowed_hrv(predicted, len(raw))

        print(f"avgRR={hrv['avgRR']:.6f} ms")
        print(f"sdRR={hrv['sdRR']:.6f} ms")
        print(f"RMSSD={hrv['RMSSD']:.6f} ms")
        print(f"pNN50={hrv['pNN50']:.6f} %")
        print(f"LF={hrv['LF']:.6f} ms2")
        print(f"HF={hrv['HF']:.6f} ms2")
        print(f"LF_HF_ratio={hrv['LF_HF']:.6f}")

        row = {
            "record": record_number,
            **hrv,
        }

        rows.append(row)

        all_predictions[record_number] = predicted

        if verbose:
            print(f"Record {record_number:02d} complete\n")

    np.save("predictions.npy", all_predictions, allow_pickle=True)

    return rows

# =============================================================================
# WRITE HRV RESULTS INTO .mat FILE
# =============================================================================

def write_hrv_to_mat(rows, mat_path):
    """
    Load the test/analysis .mat file, replace NaN HRV arrays with computed
    averaged-windowed values, and save back to the same file.

    Expected .mat variables (1-based indexing handled by MATLAB convention):
        avgRR, sdRR, RMSSD, pNN50, LF, HF, LF_HFratio
    """
    mat_path = Path(mat_path)
    data = loadmat(mat_path)

    n = len(rows)

    # build arrays (NaN for any record that had no valid window)
    avg_rr   = np.full(n, np.nan)
    sd_rr    = np.full(n, np.nan)
    rmssd    = np.full(n, np.nan)
    pnn50    = np.full(n, np.nan)
    lf       = np.full(n, np.nan)
    hf       = np.full(n, np.nan)
    lf_hf    = np.full(n, np.nan)

    for row in rows:
        i = row["record"] - 1   # 0-based index
        avg_rr[i] = row["avgRR"]
        sd_rr[i]  = row["sdRR"]
        rmssd[i]  = row["RMSSD"]
        pnn50[i]  = row["pNN50"]
        lf[i]     = row["LF"]
        hf[i]     = row["HF"]
        lf_hf[i]  = row["LF_HF"]

    data["avgRR"]      = avg_rr
    data["sdRR"]       = sd_rr
    data["RMSSD"]      = rmssd
    data["pNN50"]      = pnn50
    data["LF"]         = lf
    data["HF"]         = hf
    data["LF_HFratio"] = lf_hf

    savemat(mat_path, data)
    print(f"\nHRV results written to: {mat_path}")


# =============================================================================
# SAVE OUTPUTS
# =============================================================================

def save_metrics_csv(rows, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "hrv_metrics.csv"

    fieldnames = [
        "record",
        "avgRR", "sdRR", "RMSSD", "pNN50",
        "LF", "HF", "LF_HF",
    ]

    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    return csv_path

def plot_f1_by_record(rows, out_dir):
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    out_path = Path(out_dir) / "f1_by_record.png"
    records  = [row["record"] for row in rows]
    f1       = [row["F1"]     for row in rows]

    plt.figure(figsize=(13, 4.8))
    colors = ["tab:red" if v < 0.95 else "tab:blue" for v in f1]
    plt.bar(records, f1, color=colors)
    plt.axhline(0.99, color="0.3", linestyle="--", linewidth=1, label="F1 = 0.99")
    plt.ylim(0.80, 1.005)
    plt.xticks(records)
    plt.xlabel("Record"); plt.ylabel("F1-score")
    plt.title("Training QRS Detection F1 by Record")
    plt.grid(axis="y", alpha=0.25); plt.legend(); plt.tight_layout()
    plt.savefig(out_path, dpi=160); plt.close()
    return out_path


def plot_sensitivity_ppv(rows, out_dir):
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    out_path = Path(out_dir) / "sensitivity_vs_ppv.png"
    sens    = [row["Sensitivity"] for row in rows]
    ppv     = [row["PPV"]         for row in rows]
    records = [row["record"]      for row in rows]

    plt.figure(figsize=(6, 6))
    plt.scatter(ppv, sens, s=45, color="tab:blue")
    for record, x, y in zip(records, ppv, sens):
        if x < 0.97 or y < 0.97:
            plt.text(x + 0.002, y + 0.002, str(record), fontsize=9)
    plt.xlim(0.80, 1.005); plt.ylim(0.80, 1.005)
    plt.xlabel("Positive Predictivity"); plt.ylabel("Sensitivity")
    plt.title("Sensitivity vs PPV")
    plt.grid(alpha=0.25); plt.tight_layout()
    plt.savefig(out_path, dpi=160); plt.close()
    return out_path


def save_worst_overlays(mat_path, rows, out_dir, worst_count=4, length=15_000):
    data           = loadmat(mat_path)
    ecg_records    = data["ECG"].ravel()
    expert_records = data["QRSexpert"].ravel()
    out_paths      = []

    for row in sorted(rows, key=lambda r: r["F1"])[:worst_count]:
        record_number = row["record"]
        raw    = ecg_records[record_number - 1].ravel().astype(float)
        expert = expert_records[record_number - 1].ravel().astype(int) - 1
        expert = expert[(expert >= 0) & (expert < len(raw))]
        filtered, predicted = detect_qrs(raw, DEFAULT_FS)

        center   = row["first_error_sample"]
        start    = 0 if center is None else max(0, int(center) - length // 2)
        out_path = Path(out_dir) / f"record_{record_number:02d}_overlay.png"
        save_overlay_plot(
            raw=raw, expert=expert, filtered=filtered, predicted=predicted,
            out_path=out_path, record_number=record_number,
            start=start, length=length,
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
    outputs.extend(save_worst_overlays(mat_path, rows, out_dir, worst_count))
    return outputs


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="QRS training evaluation and visualization")
    parser.add_argument("--eval-train",   action="store_true")
    parser.add_argument("--save-plots",   action="store_true")
    parser.add_argument("--write-mat",    type=Path, default=None,
                        help="path to ProjectTestDataAnalysis.mat to write HRV results into")
    parser.add_argument("--out-dir",      type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--mat",          type=Path, default=PROJECT_TRAIN_DATA)
    parser.add_argument("--max-len",      type=int,  default=None)
    parser.add_argument("--worst-count",  type=int,  default=4)
    parser.add_argument("--viz",          action="store_true")
    parser.add_argument("--patient",      type=int,  default=1)
    parser.add_argument("--start",        type=int,  default=0)
    parser.add_argument("--length",       type=int,  default=15_000)
    parser.add_argument("--show-raw",     action="store_true")
    args = parser.parse_args()

    if args.viz:
        run_qrs_debug_viewer(
            mat_path=args.mat, patient=args.patient,
            start=args.start, length=args.length,
            max_len=args.max_len, show_raw=args.show_raw,
        )
        return

    
    rows = evaluate_training_set(args.mat, max_len=args.max_len)

    csv_path = save_metrics_csv(rows, args.out_dir)
    print(f"\nSaved HRV CSV: {csv_path}")

    if args.write_mat:
        write_hrv_to_mat(rows, args.write_mat)

if __name__ == "__main__":
    main()
