from pathlib import Path

import numpy as np
from scipy.io import loadmat
from scipy.signal import butter, filtfilt, find_peaks, iirnotch, sosfiltfilt

DEFAULT_FS = 100
MAIN_PERCENTILE = 97
MAIN_REFRACTORY_SEC = 0.30
ENERGY_REFRACTORY_SEC = 0.25
REFINE_SEC = 0.08
GAP_FACTOR = 1.55
MIN_GAP_SEC = 0.45


def default_train_mat():
    return Path(__file__).resolve().parent / "dataSet" / "ProjectTrainData.mat"


def _notch_filter(raw, fs=DEFAULT_FS):
    b_notch, a_notch = iirnotch(50, 8, fs=fs)
    return filtfilt(b_notch, a_notch, raw)


def preprocess_ecg(raw, fs=DEFAULT_FS):
    raw = np.asarray(raw, dtype=float).ravel()
    x = _notch_filter(raw, fs)

    sos_low = butter(2, 1, btype="highpass", fs=fs, output="sos")
    sos_high = butter(2, 40, btype="lowpass", fs=fs, output="sos")
    x = sosfiltfilt(sos_low, x)
    x = sosfiltfilt(sos_high, x)
    return x


def qrs_bandpass(raw, fs=DEFAULT_FS):
    raw = np.asarray(raw, dtype=float).ravel()
    x = _notch_filter(raw, fs)
    sos_qrs = butter(2, [5, 20], btype="bandpass", fs=fs, output="sos")
    return sosfiltfilt(sos_qrs, x)


def _main_abs_peak_candidates(filtered, fs=DEFAULT_FS):
    strength = np.abs(filtered)
    threshold = np.percentile(strength, MAIN_PERCENTILE)
    distance = int(MAIN_REFRACTORY_SEC * fs)
    peaks, _ = find_peaks(strength, height=threshold, distance=distance)
    return peaks.astype(int)


def _energy_peak_candidates(qrs_band, fs=DEFAULT_FS):
    derivative = np.diff(qrs_band, prepend=qrs_band[0])
    window = max(1, int(0.15 * fs))
    energy = derivative * derivative
    integrated = np.convolve(energy, np.ones(window) / window, mode="same")

    mid = np.median(integrated)
    mad = np.median(np.abs(integrated - mid))
    threshold = mid + 3.5 * mad
    distance = int(ENERGY_REFRACTORY_SEC * fs)
    candidates, _ = find_peaks(integrated, height=threshold, distance=distance)

    radius = int(REFINE_SEC * fs)
    peaks = []
    for candidate in candidates:
        lo = max(0, candidate - radius)
        hi = min(len(qrs_band), candidate + radius + 1)
        peaks.append(lo + int(np.argmax(np.abs(qrs_band[lo:hi]))))

    return np.asarray(sorted(set(peaks)), dtype=int), integrated


def _merge_close_peaks(peaks, strength_signal, fs=DEFAULT_FS):
    if len(peaks) == 0:
        return peaks

    min_gap = int(ENERGY_REFRACTORY_SEC * fs)
    peaks = np.asarray(sorted(set(peaks)), dtype=int)
    keep = []

    for peak in peaks:
        if not keep or peak - keep[-1] > min_gap:
            keep.append(int(peak))
        elif abs(strength_signal[peak]) > abs(strength_signal[keep[-1]]):
            keep[-1] = int(peak)

    return np.asarray(keep, dtype=int)


def _fill_long_gaps(main_peaks, energy_peaks, filtered, fs=DEFAULT_FS):
    if len(main_peaks) < 2 or len(energy_peaks) == 0:
        return main_peaks

    rr = np.diff(main_peaks)
    valid_rr = rr[(rr > int(0.25 * fs)) & (rr < int(2.0 * fs))]
    typical_rr = np.median(valid_rr) if len(valid_rr) else np.median(rr)
    gap_limit = max(int(MIN_GAP_SEC * fs), int(GAP_FACTOR * typical_rr))

    extra = []
    for left, right in zip(main_peaks[:-1], main_peaks[1:]):
        if right - left <= gap_limit:
            continue

        in_gap = energy_peaks[
            (energy_peaks > left + int(ENERGY_REFRACTORY_SEC * fs))
            & (energy_peaks < right - int(ENERGY_REFRACTORY_SEC * fs))
        ]
        extra.extend(int(x) for x in in_gap)

    if not extra:
        return main_peaks

    merged = np.concatenate([main_peaks, np.asarray(extra, dtype=int)])
    return _merge_close_peaks(merged, filtered, fs)


def detect_qrs(raw, fs=DEFAULT_FS):
    filtered = preprocess_ecg(raw, fs)
    qrs_band = qrs_bandpass(raw, fs)

    main_peaks = _main_abs_peak_candidates(filtered, fs)
    energy_peaks, _energy = _energy_peak_candidates(qrs_band, fs)
    peaks = _fill_long_gaps(main_peaks, energy_peaks, filtered, fs)

    return filtered, peaks.astype(int)


def predict_peaks(raw):
    return detect_qrs(raw, DEFAULT_FS)


def load_recording(mat_path, index, max_len=None):
    data = loadmat(mat_path)
    raw = data["ECG"].ravel()[index].ravel().astype(float)

    if max_len is not None:
        raw = raw[:max_len]

    expert = None
    if "QRSexpert" in data:
        expert = data["QRSexpert"].ravel()[index].ravel().astype(int) - 1
        expert = expert[(expert >= 0) & (expert < len(raw))]
    return raw, expert


def save_overlay_plot(
    raw,
    expert,
    filtered,
    predicted,
    out_path,
    record_number,
    start=0,
    length=15_000,
    show_raw=False,
    fs=DEFAULT_FS,
):
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    start = max(0, min(int(start), len(filtered) - 1))
    end = min(start + int(length), len(filtered))
    sl = slice(start, end)
    t = np.arange(start, end) / fs

    predicted = np.asarray(predicted, dtype=int)
    pred_vis = predicted[(predicted >= start) & (predicted < end)]

    expert_vis = np.asarray([], dtype=int)
    if expert is not None:
        expert = np.asarray(expert, dtype=int)
        expert_vis = expert[(expert >= start) & (expert < end)]

    plt.figure(figsize=(16, 5))
    if show_raw:
        plt.plot(t, raw[sl], color="#9ecae1", linewidth=0.7, alpha=0.7, label="Raw ECG")
    plt.plot(t, filtered[sl], color="0.15", linewidth=0.9, label="Filtered ECG")

    if len(pred_vis):
        plt.scatter(
            pred_vis / fs,
            filtered[pred_vis],
            s=28,
            c="tab:orange",
            marker="v",
            label=f"Predicted ({len(pred_vis)})",
            zorder=5,
        )

    if len(expert_vis):
        plt.scatter(
            expert_vis / fs,
            filtered[expert_vis],
            s=45,
            facecolors="none",
            edgecolors="tab:green",
            linewidths=1.6,
            label=f"Expert ({len(expert_vis)})",
            zorder=6,
        )

    plt.title(f"Record {record_number} QRS overlay, samples [{start}:{end})")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude (uV, filtered)")
    plt.grid(alpha=0.25)
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def run_qrs_overlay_interactive(
    mat_path=None,
    patient=1,
    start=0,
    length=15_000,
    max_len=None,
    show_raw=False,
):
    import matplotlib.pyplot as plt
    from matplotlib.widgets import Button, Slider

    if mat_path is None:
        mat_path = default_train_mat()

    mat_path = Path(mat_path)
    n_subjects = len(loadmat(mat_path)["ECG"].ravel())
    patient = max(1, min(int(patient), n_subjects))
    current_patient = [patient - 1]
    cache = {"key": None, "data": None}

    def get_full(patient_idx):
        key = (patient_idx, max_len, str(mat_path))
        if cache["key"] != key:
            raw_full, expert_full = load_recording(mat_path, patient_idx, max_len)
            filtered_full, pred_full = predict_peaks(raw_full)
            cache["key"] = key
            cache["data"] = raw_full, expert_full, filtered_full, pred_full
        return cache["data"]

    def draw(patient_idx, win_start, win_len):
        raw_full, expert_full, filtered_full, pred_full = get_full(patient_idx)

        start_i = max(0, min(int(win_start), len(filtered_full) - 1))
        end = min(start_i + int(win_len), len(filtered_full))

        sl = slice(start_i, end)
        t = np.arange(start_i, end) / DEFAULT_FS

        pred = pred_full[(pred_full >= start_i) & (pred_full < end)]
        expert_vis = (
            expert_full[(expert_full >= start_i) & (expert_full < end)]
            if expert_full is not None
            else []
        )

        ax.clear()
        if show_raw:
            ax.plot(t, raw_full[sl], color="#aec7e8", linewidth=0.8, alpha=0.85, label="ECG raw")
        ax.plot(t, filtered_full[sl], color="0.15", linewidth=1.0, label="ECG filtered")

        if len(pred):
            ax.scatter(
                pred / DEFAULT_FS,
                filtered_full[pred],
                s=40,
                c="tab:orange",
                marker="v",
                zorder=5,
                label=f"Predicted QRS ({len(pred)})",
            )

        if expert_full is not None and len(expert_vis):
            ax.scatter(
                expert_vis / DEFAULT_FS,
                filtered_full[expert_vis],
                s=60,
                facecolors="none",
                edgecolors="tab:green",
                linewidths=2,
                zorder=6,
                label=f"Expert QRS ({len(expert_vis)})",
            )

        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Amplitude (uV, filtered)")
        ax.set_title(
            f"{mat_path.name} | record {patient_idx + 1}/{n_subjects} "
            f"| samples [{start_i}:{end}) | keys: n/p or arrows"
        )
        ax.legend(loc="upper right", fontsize=9)
        ax.grid(True, alpha=0.25)
        fig.canvas.draw_idle()

    fig, ax = plt.subplots(figsize=(14, 5))
    plt.subplots_adjust(bottom=0.26)

    ax_slider_start = fig.add_axes((0.12, 0.14, 0.55, 0.03))
    ax_slider_len = fig.add_axes((0.12, 0.10, 0.55, 0.03))
    ax_btn_prev = fig.add_axes((0.12, 0.03, 0.12, 0.045))
    ax_btn_next = fig.add_axes((0.26, 0.03, 0.12, 0.045))

    raw0, _ = load_recording(mat_path, current_patient[0], max_len)
    n0 = len(raw0)

    slider_start = Slider(
        ax_slider_start,
        "Start",
        0,
        max(n0 - 1, 1),
        valinit=min(start, max(n0 - 1, 0)),
        valstep=1,
    )
    slider_len = Slider(
        ax_slider_len,
        "Window",
        500,
        min(500_000, max(n0, 500)),
        valinit=min(length, n0),
        valstep=100,
    )
    btn_prev = Button(ax_btn_prev, "Prev record")
    btn_next = Button(ax_btn_next, "Next record")

    def on_change(_val=None):
        draw(current_patient[0], int(slider_start.val), int(slider_len.val))

    def bump_subject(delta):
        current_patient[0] = (current_patient[0] + delta) % n_subjects
        raw_i, _ = load_recording(mat_path, current_patient[0], max_len)
        ni = len(raw_i)
        slider_start.valmax = max(ni - 1, 1)
        slider_start.ax.set_xlim(slider_start.valmin, slider_start.valmax)
        slider_len.valmax = min(500_000, max(ni, 500))
        slider_len.ax.set_xlim(slider_len.valmin, slider_len.valmax)
        slider_start.eventson = False
        slider_len.eventson = False
        slider_start.set_val(0)
        if float(slider_len.val) > slider_len.valmax:
            slider_len.set_val(slider_len.valmax)
        slider_start.eventson = True
        slider_len.eventson = True
        on_change()

    slider_start.on_changed(on_change)
    slider_len.on_changed(on_change)

    btn_prev.on_clicked(lambda _e: bump_subject(-1))
    btn_next.on_clicked(lambda _e: bump_subject(1))

    def on_key(event):
        if event.key is None:
            return
        key = event.key.lower()
        if key in ("n", "right"):
            bump_subject(1)
        elif key in ("p", "left"):
            bump_subject(-1)

    fig.canvas.mpl_connect("key_press_event", on_key)

    draw(current_patient[0], start, length)
    plt.show()
