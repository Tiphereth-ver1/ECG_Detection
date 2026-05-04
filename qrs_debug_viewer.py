from pathlib import Path

import numpy as np
from scipy.io import loadmat

from qrs_pipeline import DEFAULT_FS, default_train_mat, detect_qrs_debug, load_recording

MIN_WINDOW = 500
MAX_WINDOW = 500_000


def _peaks_in_window(peaks, start, end):
    peaks = np.asarray(peaks, dtype=int)
    return peaks[(peaks >= start) & (peaks < end)]


def _window_limits(n_samples):
    min_window = min(MIN_WINDOW, max(1, n_samples))
    max_window = min(MAX_WINDOW, max(min_window, n_samples))
    return min_window, max_window


def run_qrs_debug_viewer(
    mat_path=None,
    patient=1,
    start=0,
    length=15_000,
    max_len=None,
    show_raw=False,
):
    import matplotlib.pyplot as plt
    from matplotlib.widgets import Button, CheckButtons, Slider

    if mat_path is None:
        mat_path = default_train_mat()

    mat_path = Path(mat_path)
    n_subjects = len(loadmat(mat_path)["ECG"].ravel())
    patient = max(1, min(int(patient), n_subjects))
    current_patient = [patient - 1]
    cache = {"key": None, "data": None}

    layer_visible = {
        "Raw ECG": bool(show_raw),
        "Filtered ECG": True,
        "Abs filtered": True,
        "Main threshold": True,
        "QRS energy": True,
        "Energy threshold": True,
        "Predicted QRS": True,
        "Expert QRS": True,
        "Main candidates": False,
        "Energy candidates": False,
    }

    def get_full(patient_idx):
        key = (patient_idx, max_len, str(mat_path))
        if cache["key"] != key:
            raw_full, expert_full = load_recording(mat_path, patient_idx, max_len)
            debug_full = detect_qrs_debug(raw_full, DEFAULT_FS)
            cache["key"] = key
            cache["data"] = raw_full, expert_full, debug_full
        return cache["data"]

    def clamp_window(win_start, win_len, n_samples):
        min_window, max_window = _window_limits(n_samples)
        win_len = int(max(min_window, min(int(win_len), max_window)))
        max_start = max(0, n_samples - win_len)
        win_start = int(max(0, min(int(win_start), max_start)))
        return win_start, win_len

    def draw(patient_idx, win_start, win_len):
        raw_full, expert_full, debug = get_full(patient_idx)
        n_samples = len(raw_full)
        start_i, win_len = clamp_window(win_start, win_len, n_samples)
        end = min(start_i + win_len, n_samples)
        sl = slice(start_i, end)
        t = np.arange(start_i, end) / DEFAULT_FS

        filtered = debug["filtered"]
        abs_filtered = debug["abs_filtered"]
        energy = debug["energy_integrated"]

        pred_vis = _peaks_in_window(debug["predicted_peaks"], start_i, end)
        main_vis = _peaks_in_window(debug["main_peaks"], start_i, end)
        energy_vis = _peaks_in_window(debug["energy_peaks"], start_i, end)

        if expert_full is None:
            expert_vis = np.asarray([], dtype=int)
        else:
            expert_vis = _peaks_in_window(expert_full, start_i, end)

        ax.clear()
        ax_energy.clear()

        if layer_visible["Raw ECG"]:
            ax.plot(t, raw_full[sl], color="#9ecae1", linewidth=0.7, alpha=0.75, label="Raw ECG")
        if layer_visible["Filtered ECG"]:
            ax.plot(t, filtered[sl], color="0.15", linewidth=0.95, label="Filtered ECG")
        if layer_visible["Abs filtered"]:
            ax.plot(t, abs_filtered[sl], color="tab:purple", linewidth=0.8, alpha=0.85, label="Abs filtered")
        if layer_visible["Main threshold"]:
            ax.axhline(
                debug["main_threshold"],
                color="tab:purple",
                linestyle="--",
                linewidth=1.0,
                label="Main threshold",
            )

        if layer_visible["Main candidates"] and len(main_vis):
            ax.scatter(
                main_vis / DEFAULT_FS,
                abs_filtered[main_vis],
                s=22,
                color="tab:purple",
                marker="x",
                linewidths=1.1,
                label=f"Main candidates ({len(main_vis)})",
                zorder=7,
            )

        if layer_visible["Predicted QRS"] and len(pred_vis):
            ax.scatter(
                pred_vis / DEFAULT_FS,
                filtered[pred_vis],
                s=38,
                c="tab:orange",
                marker="v",
                label=f"Predicted QRS ({len(pred_vis)})",
                zorder=8,
            )

        if layer_visible["Expert QRS"] and len(expert_vis):
            ax.scatter(
                expert_vis / DEFAULT_FS,
                filtered[expert_vis],
                s=55,
                facecolors="none",
                edgecolors="tab:green",
                linewidths=1.7,
                label=f"Expert QRS ({len(expert_vis)})",
                zorder=9,
            )

        energy_axis_on = (
            layer_visible["QRS energy"]
            or layer_visible["Energy threshold"]
            or layer_visible["Energy candidates"]
        )
        ax_energy.set_visible(energy_axis_on)
        if energy_axis_on:
            if layer_visible["QRS energy"]:
                ax_energy.plot(t, energy[sl], color="tab:cyan", linewidth=0.8, alpha=0.85, label="QRS energy")
            if layer_visible["Energy threshold"]:
                ax_energy.axhline(
                    debug["energy_threshold"],
                    color="tab:cyan",
                    linestyle="--",
                    linewidth=1.0,
                    label="Energy threshold",
                )
            if layer_visible["Energy candidates"] and len(energy_vis):
                ax_energy.scatter(
                    energy_vis / DEFAULT_FS,
                    energy[energy_vis],
                    s=22,
                    color="tab:cyan",
                    marker="o",
                    label=f"Energy candidates ({len(energy_vis)})",
                    zorder=6,
                )

        ax.set_xlim(start_i / DEFAULT_FS, end / DEFAULT_FS)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("ECG amplitude")
        ax_energy.set_ylabel("Integrated QRS energy")
        ax.set_title(
            f"{mat_path.name} | record {patient_idx + 1}/{n_subjects} "
            f"| samples [{start_i}:{end}) | keys: n/p or arrows | scroll: zoom"
        )
        ax.grid(True, alpha=0.25)

        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(handles, labels, loc="upper left", fontsize=8)

        energy_handles, energy_labels = ax_energy.get_legend_handles_labels()
        if energy_axis_on and energy_handles:
            ax_energy.legend(energy_handles, energy_labels, loc="upper right", fontsize=8)

        fig.canvas.draw_idle()

    fig, ax = plt.subplots(figsize=(15, 6))
    ax_energy = ax.twinx()
    ax_energy.patch.set_visible(False)
    plt.subplots_adjust(left=0.08, right=0.77, bottom=0.24)

    ax_check = fig.add_axes((0.80, 0.29, 0.18, 0.54))
    ax_slider_start = fig.add_axes((0.12, 0.15, 0.55, 0.03))
    ax_slider_len = fig.add_axes((0.12, 0.10, 0.55, 0.03))
    ax_btn_prev = fig.add_axes((0.12, 0.035, 0.12, 0.045))
    ax_btn_next = fig.add_axes((0.26, 0.035, 0.12, 0.045))
    ax_btn_controls = fig.add_axes((0.80, 0.86, 0.18, 0.045))

    labels = list(layer_visible.keys())
    checks = CheckButtons(ax_check, labels, [layer_visible[label] for label in labels])
    ax_check.set_title("Layers", fontsize=10)

    raw0, _expert0, _debug0 = get_full(current_patient[0])
    n0 = len(raw0)
    start0, length0 = clamp_window(start, length, n0)
    min_len0, max_len0 = _window_limits(n0)

    slider_start = Slider(
        ax_slider_start,
        "Start",
        0,
        max(n0 - 1, 1),
        valinit=start0,
        valstep=1,
    )
    slider_len = Slider(
        ax_slider_len,
        "Window",
        min_len0,
        max_len0,
        valinit=length0,
        valstep=100,
    )
    btn_prev = Button(ax_btn_prev, "Prev record")
    btn_next = Button(ax_btn_next, "Next record")
    btn_controls = Button(ax_btn_controls, "Hide controls")

    def set_slider_values(new_start, new_length):
        raw_full, _expert_full, _debug_full = get_full(current_patient[0])
        n_samples = len(raw_full)
        new_start, new_length = clamp_window(new_start, new_length, n_samples)

        slider_start.eventson = False
        slider_len.eventson = False
        slider_start.set_val(new_start)
        slider_len.set_val(new_length)
        slider_start.eventson = True
        slider_len.eventson = True
        return new_start, new_length

    def update_slider_limits(n_samples):
        min_len, max_len = _window_limits(n_samples)
        slider_start.valmax = max(n_samples - 1, 1)
        slider_start.ax.set_xlim(slider_start.valmin, slider_start.valmax)
        slider_len.valmin = min_len
        slider_len.valmax = max_len
        slider_len.ax.set_xlim(slider_len.valmin, slider_len.valmax)

    def on_change(_val=None):
        draw(current_patient[0], int(slider_start.val), int(slider_len.val))

    def bump_subject(delta):
        current_patient[0] = (current_patient[0] + delta) % n_subjects
        raw_i, _expert_i, _debug_i = get_full(current_patient[0])
        update_slider_limits(len(raw_i))
        new_len = min(int(slider_len.val), _window_limits(len(raw_i))[1])
        set_slider_values(0, new_len)
        on_change()

    def toggle_layer(label):
        layer_visible[label] = not layer_visible[label]
        on_change()

    controls_open = [True]

    def toggle_controls(_event):
        controls_open[0] = not controls_open[0]
        ax_check.set_visible(controls_open[0])
        btn_controls.label.set_text("Hide controls" if controls_open[0] else "Show controls")
        fig.canvas.draw_idle()

    def on_scroll(event):
        if event.inaxes not in (ax, ax_energy):
            return

        raw_full, _expert_full, _debug_full = get_full(current_patient[0])
        n_samples = len(raw_full)
        old_start, old_len = clamp_window(slider_start.val, slider_len.val, n_samples)
        old_end = min(old_start + old_len, n_samples)

        if event.xdata is None:
            center = old_start + old_len // 2
        else:
            center = int(event.xdata * DEFAULT_FS)
            center = max(old_start, min(center, old_end))

        if event.button == "up":
            new_len = int(old_len * 0.80)
        elif event.button == "down":
            new_len = int(old_len * 1.25)
        else:
            return

        min_len, max_len = _window_limits(n_samples)
        new_len = int(max(min_len, min(new_len, max_len)))
        if old_len <= 0:
            focus = 0.5
        else:
            focus = (center - old_start) / old_len
        new_start = int(center - focus * new_len)
        new_start, new_len = set_slider_values(new_start, new_len)
        draw(current_patient[0], new_start, new_len)

    def on_key(event):
        if event.key is None:
            return
        key = event.key.lower()
        if key in ("n", "right"):
            bump_subject(1)
        elif key in ("p", "left"):
            bump_subject(-1)

    slider_start.on_changed(on_change)
    slider_len.on_changed(on_change)
    checks.on_clicked(toggle_layer)
    btn_prev.on_clicked(lambda _event: bump_subject(-1))
    btn_next.on_clicked(lambda _event: bump_subject(1))
    btn_controls.on_clicked(toggle_controls)
    fig.canvas.mpl_connect("scroll_event", on_scroll)
    fig.canvas.mpl_connect("key_press_event", on_key)

    draw(current_patient[0], start0, length0)
    plt.show()
