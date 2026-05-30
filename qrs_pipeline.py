from pathlib import Path

import numpy as np
from scipy.io import loadmat
from scipy.signal import butter, filtfilt, find_peaks, iirnotch, sosfiltfilt

# The dataset is sampled at 100 Hz, so 1 sample = 10 ms.
DEFAULT_FS = 100

# Main branch settings. The default detector uses abs(filtered ECG), which works
# for both upright and inverted QRS complexes.
USE_ABS_MAIN_BRANCH = True
MAIN_PERCENTILE = 97
MAIN_REFRACTORY_SEC = 0.30

# Energy branch settings. This branch is not the main detector; it only helps
# rescue missed beats when the main branch leaves an unusually long RR gap.
ENERGY_REFRACTORY_SEC = 0.25
REFINE_SEC = 0.08
GAP_FACTOR = 1.55
MIN_GAP_SEC = 0.45

# Simple signal-quality gate. These are intentionally rule-based and easy to
# inspect from the debug viewer instead of using a black-box model or fitting
# record-specific rules.
QUALITY_WINDOW_SEC = 5.0
QUALITY_STEP_SEC = 2.5
QUALITY_PAD_SEC = 0.0
RAW_RANGE_BIG = 12000
FILTERED_STD_BIG = 1300
ABS_BIG = 5000
PEAK_COUNT_HUGE = 11
PEAK_COUNT_DENSE = 10
ABS_MED_DENSE = 250
STD_DENSE = 450
ENERGY_BIG = 2_000_000
RAW_RANGE_ENERGY = 9000
QRS_SNR_LOW = 4.25
QRS_SNR_MIN_PEAKS = 4
QRS_SNR_MIN_ENERGY_PEAKS = 10
SHAPE_REFRACTORY_SEC = 0.32

# T-wave suppression. Some records have T waves whose broad ECG amplitude is
# larger than the QRS peak. The main abs(filtered) branch can then lock onto a
# slow T wave. Keep only candidates that have enough nearby 5-20 Hz QRS-band
# strength relative to the current candidate set.
QRS_BAND_VETO_RATIO = 0.235
QRS_BAND_VETO_RADIUS_SEC = 0.04

# Expert annotations in one morphology-change pattern land on the later
# positive lobe of a biphasic complex rather than on the earlier negative
# valley. The trigger is record-level and waveform-based: only records with many
# late positive-lobe candidates enable this alignment branch.
LATE_ALIGN_SCAN_START_SEC = 20_000
LATE_ALIGN_TRIGGER_START_SEC = 22_000
LATE_ALIGN_MIN_COUNT = 500
LATE_ALIGN_MIN_FRACTION = 0.08
LATE_ALIGN_POS_START_SEC = 0.15
LATE_ALIGN_POS_END_SEC = 0.40
LATE_ALIGN_CENTER_RADIUS_SEC = 0.08
LATE_ALIGN_POS_EARLY_RATIO = 1.2
LATE_ALIGN_POS_VALLEY_RATIO = 0.25

# Final sequence cleanup. This is a record-adaptive refractory layer: records
# with dense artifact clusters often have extra peaks at about half the local RR
# interval. Keep the stronger local QRS shape when two detections are too close
# for that record's rhythm.
ADAPTIVE_RR_MIN_SEC = 0.45
ADAPTIVE_RR_FACTOR = 0.50
ADAPTIVE_RR_QRS_WEIGHT = 1.00

# Conservative recovery from the quality gate. Some low-background windows are
# rejected because they border obvious artifact, but still contain isolated QRS
# complexes. Only restore peaks that are far from an accepted detection and
# have strong local QRS-band support over a quiet broad-band baseline.
QUALITY_RESCUE_MIN_NEAREST_SEC = 0.30
QUALITY_RESCUE_MIN_QRS_BAND = 250
QUALITY_RESCUE_MAX_ABS_MEDIAN = 150
QUALITY_RESCUE_MAX_QRS_MEDIAN = 60

# Final morphology cleanup for residual false positives. These rules are
# intentionally conservative: either the peak sits in a noisy high-background
# segment with weak QRS support, or the peak itself is extremely weak in both
# broad ECG and QRS-band views.
FINAL_NOISY_QRS_MAX = 800
FINAL_NOISY_FILTERED_MAX = 2000
FINAL_NOISY_ABS_MEDIAN_MIN = 200
FINAL_WEAK_QRS_MAX = 100
FINAL_WEAK_FILTERED_MAX = 150


def default_train_mat():
    """Return the default training .mat path used by the CLI and viewer."""
    return Path(__file__).resolve().parent / "dataSet" / "ProjectTrainData.mat"

def default_test_mat():
    """Return the default training .mat path used by the CLI and viewer."""
    return Path(__file__).resolve().parent / "dataSet" / "ProjectTestData.mat"


def _notch_filter(raw, fs=DEFAULT_FS):
    """Remove 50 Hz power-line noise before the broader ECG filters."""
    b_notch, a_notch = iirnotch(50, 8, fs=fs)
    return filtfilt(b_notch, a_notch, raw)


def preprocess_ecg(raw, fs=DEFAULT_FS):
    """Create the broad filtered ECG used for plotting and final peak output."""
    raw = np.asarray(raw, dtype=float).ravel()
    x = _notch_filter(raw, fs)

    # This path keeps ECG morphology visible. It is wider than the QRS energy
    # branch because the final marker is drawn on this signal.
    sos_low = butter(2, 1, btype="highpass", fs=fs, output="sos")
    sos_high = butter(2, 40, btype="lowpass", fs=fs, output="sos")
    x = sosfiltfilt(sos_low, x)
    x = sosfiltfilt(sos_high, x)
    return x


def qrs_bandpass(raw, fs=DEFAULT_FS):
    """Create a narrow QRS-band signal for the energy detector."""
    raw = np.asarray(raw, dtype=float).ravel()
    x = _notch_filter(raw, fs)

    # A 5-20 Hz band highlights the steep QRS complex and suppresses slower
    # P/T waves. This follows the same idea as Pan-Tompkins/XQRS style energy.
    sos_qrs = butter(2, [5, 20], btype="bandpass", fs=fs, output="sos")
    return sosfiltfilt(sos_qrs, x)


def _main_abs_peak_candidates(filtered, fs=DEFAULT_FS):
    """Find the main QRS event candidates from abs(filtered ECG)."""
    strength = np.abs(filtered)
    threshold = np.percentile(strength, MAIN_PERCENTILE)
    distance = int(MAIN_REFRACTORY_SEC * fs)
    peaks, _ = find_peaks(strength, height=threshold, distance=distance)
    return peaks.astype(int)


def _energy_peak_candidates(qrs_band, fs=DEFAULT_FS):
    """Find QRS-like events from derivative-squared moving-window energy."""
    derivative = np.diff(qrs_band, prepend=qrs_band[0])
    window = max(1, int(0.15 * fs))
    energy = derivative * derivative
    integrated = np.convolve(energy, np.ones(window) / window, mode="same")

    # Median + MAD gives a simple adaptive threshold that is less affected by
    # a few very large artifacts than mean + standard deviation.
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
        # The energy peak marks the QRS region, not always the exact sample.
        # Move it to the strongest nearby QRS-band point.
        peaks.append(lo + int(np.argmax(np.abs(qrs_band[lo:hi]))))

    return np.asarray(sorted(set(peaks)), dtype=int), integrated


def _main_abs_peak_candidates_debug(filtered, fs=DEFAULT_FS):
    """Debug version of the main branch that also returns signal and threshold."""
    strength = np.abs(filtered)
    threshold = np.percentile(strength, MAIN_PERCENTILE)
    distance = int(MAIN_REFRACTORY_SEC * fs)
    peaks, _ = find_peaks(strength, height=threshold, distance=distance)
    return peaks.astype(int), strength, float(threshold)


def _energy_peak_candidates_debug(qrs_band, fs=DEFAULT_FS):
    """Debug version of the energy branch that also returns energy and threshold."""
    derivative = np.diff(qrs_band, prepend=qrs_band[0])
    window = max(1, int(0.15 * fs))
    energy = derivative * derivative
    integrated = np.convolve(energy, np.ones(window) / window, mode="same")

    mid = np.median(integrated)
    mad = np.median(np.abs(integrated - mid))
    threshold = float(mid + 3.5 * mad)
    distance = int(ENERGY_REFRACTORY_SEC * fs)
    candidates, _ = find_peaks(integrated, height=threshold, distance=distance)

    radius = int(REFINE_SEC * fs)
    peaks = []
    for candidate in candidates:
        lo = max(0, candidate - radius)
        hi = min(len(qrs_band), candidate + radius + 1)
        peaks.append(lo + int(np.argmax(np.abs(qrs_band[lo:hi]))))

    return np.asarray(sorted(set(peaks)), dtype=int), integrated, threshold


def _merge_close_peaks(peaks, strength_signal, fs=DEFAULT_FS):
    """Merge detections that are closer than the refractory period."""
    if len(peaks) == 0:
        return peaks

    # Refractory merge: two QRS detections too close together are usually the
    # same beat, so keep the one with stronger local signal.
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
    """Use energy candidates to fill long gaps left by the main detector."""
    if len(main_peaks) < 2 or len(energy_peaks) == 0:
        return main_peaks

    # The main detector is conservative. If a gap is much longer than the usual
    # RR interval, use the energy branch to search for missed beats inside it.
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


def _quality_mask(raw, filtered, qrs_band, energy, fs=DEFAULT_FS, peaks=None, energy_peaks=None):
    """Return a per-sample mask for segments that look usable for QRS detection."""
    # Mark windows as noisy when they look like artifact instead of ECG:
    # huge raw range, saturated energy, too many QRS-like candidates, or
    # sustained low-confidence QRS detections inside dense energy activity.
    n = len(filtered)
    good = np.ones(n, dtype=bool)
    noise_score = np.zeros(n, dtype=float)
    if n == 0:
        return good, noise_score

    if peaks is None:
        peaks = _main_abs_peak_candidates(filtered, fs)
    peaks = np.asarray(peaks, dtype=int)
    if energy_peaks is None:
        energy_peaks = np.asarray([], dtype=int)
    else:
        energy_peaks = np.asarray(energy_peaks, dtype=int)

    win = max(1, int(QUALITY_WINDOW_SEC * fs))
    step = max(1, int(QUALITY_STEP_SEC * fs))
    pad = int(QUALITY_PAD_SEC * fs)
    abs_filtered = np.abs(filtered)
    windows = []
    low_snr_flags = []
    low_snr_scores = []

    for start in range(0, n, step):
        end = min(n, start + win)
        if end <= start:
            continue

        # Each window gets a few simple artifact features. The goal is only to
        # catch extreme noise blocks where the expert usually has no QRS labels.
        local_peaks = peaks[(peaks >= start) & (peaks < end)]
        peak_count = len(local_peaks)
        energy_peak_count = int(np.sum((energy_peaks >= start) & (energy_peaks < end)))
        raw_range = float(np.ptp(raw[start:end]))
        filtered_std = float(np.std(filtered[start:end]))
        abs_median = float(np.median(abs_filtered[start:end]))
        abs_max = float(np.max(abs_filtered[start:end]))
        energy_max = float(np.max(energy[start:end])) if len(energy) else 0.0

        # This is not a physical SNR. It is only a simple contrast score:
        # median detected-QRS height divided by the median background height in
        # the same window. Real QRS windows usually have a large ratio; artifact
        # sections often produce many weak "peaks" that do not stand out.
        if peak_count:
            qrs_snr = float(np.median(abs_filtered[local_peaks]) / (abs_median + 1e-9))
        else:
            qrs_snr = float("inf")

        low_snr = (
            peak_count >= QRS_SNR_MIN_PEAKS
            and energy_peak_count >= QRS_SNR_MIN_ENERGY_PEAKS
            and qrs_snr < QRS_SNR_LOW
        )
        windows.append((start, end))
        low_snr_flags.append(low_snr)
        low_snr_scores.append(max(1.0, QRS_SNR_LOW - min(qrs_snr, QRS_SNR_LOW)))

        bad = False
        score = 0.0
        if raw_range > RAW_RANGE_BIG and abs_max > ABS_BIG:
            bad = True
            score = max(score, 3.0)
        if filtered_std > FILTERED_STD_BIG and abs_max > ABS_BIG:
            bad = True
            score = max(score, 3.0)
        if peak_count >= PEAK_COUNT_HUGE:
            bad = True
            score = max(score, 2.0)
        if peak_count >= PEAK_COUNT_DENSE and abs_median > ABS_MED_DENSE and filtered_std > STD_DENSE:
            bad = True
            score = max(score, 2.0)
        if energy_max > ENERGY_BIG and raw_range > RAW_RANGE_ENERGY:
            bad = True
            score = max(score, 2.5)

        if bad:
            # Pad a little around noisy windows because artifacts often start
            # before the exact window boundary.
            lo = max(0, start - pad)
            hi = min(n, end + pad)
            good[lo:hi] = False
            noise_score[lo:hi] = np.maximum(noise_score[lo:hi], score)

    # Low QRS contrast alone is too aggressive because some real ECG segments
    # are low amplitude. Require another low-SNR neighbor and dense energy
    # candidates before deleting the window, and do not pad this rule.
    for i, (start, end) in enumerate(windows):
        if not low_snr_flags[i]:
            continue
        has_low_neighbor = (
            (i > 0 and low_snr_flags[i - 1])
            or (i + 1 < len(low_snr_flags) and low_snr_flags[i + 1])
        )
        if has_low_neighbor:
            good[start:end] = False
            noise_score[start:end] = np.maximum(noise_score[start:end], low_snr_scores[i])

    return good, noise_score


def _shape_score(filtered, qrs_band, peak, fs=DEFAULT_FS):
    """Score a close duplicate candidate by local ECG and QRS-band strength."""
    # A small helper for resolving two very close detections. Stronger local
    # amplitude and stronger QRS-band response should win.
    radius = max(2, int(REFINE_SEC * fs))
    lo = max(0, int(peak) - radius)
    hi = min(len(filtered), int(peak) + radius + 1)
    if hi <= lo:
        return 0.0

    baseline = np.median(filtered[lo:hi])
    amp = abs(filtered[int(peak)] - baseline)
    qrs_amp = np.max(np.abs(qrs_band[lo:hi])) if len(qrs_band[lo:hi]) else 0.0
    return float(amp + 0.25 * qrs_amp)


def _shape_filter(filtered, qrs_band, energy, peaks, fs=DEFAULT_FS):
    """Remove very close duplicate detections after quality filtering."""
    # Final close-peak cleanup. This catches some P/T-wave or duplicate QRS
    # detections that survive the first refractory merge.
    peaks = np.asarray(sorted(set(np.asarray(peaks, dtype=int))), dtype=int)
    if len(peaks) == 0:
        return peaks, np.asarray([], dtype=int)

    min_gap = int(SHAPE_REFRACTORY_SEC * fs)
    keep = []
    removed = []
    for peak in peaks:
        if not keep or peak - keep[-1] > min_gap:
            keep.append(int(peak))
            continue

        old = keep[-1]
        if _shape_score(filtered, qrs_band, peak, fs) > _shape_score(filtered, qrs_band, old, fs):
            removed.append(old)
            keep[-1] = int(peak)
        else:
            removed.append(int(peak))

    return np.asarray(keep, dtype=int), np.asarray(removed, dtype=int)


def _qrs_band_veto(qrs_band, peaks, fs=DEFAULT_FS):
    """Remove broad-wave candidates that do not have local QRS-band support."""
    peaks = np.asarray(sorted(set(np.asarray(peaks, dtype=int))), dtype=int)
    if len(peaks) == 0:
        return peaks, np.asarray([], dtype=int)

    qrs_strength = np.abs(qrs_band)
    baseline = float(np.median(qrs_strength[peaks]))
    if baseline <= 0:
        return peaks, np.asarray([], dtype=int)

    radius = max(1, int(QRS_BAND_VETO_RADIUS_SEC * fs))
    keep = []
    removed = []
    for peak in peaks:
        lo = max(0, int(peak) - radius)
        hi = min(len(qrs_strength), int(peak) + radius + 1)
        local_qrs = float(np.max(qrs_strength[lo:hi])) if hi > lo else 0.0
        if local_qrs >= QRS_BAND_VETO_RATIO * baseline:
            keep.append(int(peak))
        else:
            removed.append(int(peak))

    return np.asarray(keep, dtype=int), np.asarray(removed, dtype=int)


def _late_positive_alignment_candidate(raw, qrs_band, peak, fs=DEFAULT_FS):
    """Return a later positive-lobe candidate for a biphasic QRS, if present."""
    peak = int(peak)
    context = max(1, int(0.35 * fs))
    center_radius = max(1, int(LATE_ALIGN_CENTER_RADIUS_SEC * fs))
    late_start = peak + int(LATE_ALIGN_POS_START_SEC * fs)
    late_end = peak + int(LATE_ALIGN_POS_END_SEC * fs)
    early_start = peak - center_radius
    early_end = peak + center_radius

    if early_start < 0 or late_end > len(raw) or late_end <= late_start:
        return None

    context_lo = max(0, peak - context)
    context_hi = min(len(raw), peak + int(0.45 * fs))
    baseline = float(np.median(raw[context_lo:context_hi]))
    late = raw[late_start:late_end] - baseline
    early = raw[early_start:early_end] - baseline
    if len(late) == 0 or len(early) == 0:
        return None

    late_rel = int(np.argmax(late))
    late_peak = late_start + late_rel
    late_amp = float(late[late_rel])
    early_max = float(np.max(early))
    early_min = float(np.min(early))

    is_negative_center = qrs_band[peak] < 0 and (raw[peak] - baseline) < 0
    strong_late_lobe = (
        late_amp >= LATE_ALIGN_POS_EARLY_RATIO * abs(early_max)
        and late_amp >= LATE_ALIGN_POS_VALLEY_RATIO * abs(early_min)
    )
    if is_negative_center and strong_late_lobe:
        return int(late_peak)
    return None


def _late_positive_alignment_stats(raw, qrs_band, peaks, fs=DEFAULT_FS):
    """Count late-positive morphology candidates in the tail of a record."""
    peaks = np.asarray(peaks, dtype=int)
    tail_start = int(LATE_ALIGN_TRIGGER_START_SEC * fs)
    tail_peaks = peaks[peaks >= tail_start]
    if len(tail_peaks) == 0:
        return 0, 0, 0.0

    count = 0
    for peak in tail_peaks:
        if _late_positive_alignment_candidate(raw, qrs_band, peak, fs) is not None:
            count += 1
    fraction = count / len(tail_peaks)
    return int(count), int(len(tail_peaks)), float(fraction)


def _align_late_positive_lobes(raw, qrs_band, peaks, fs=DEFAULT_FS):
    """Move selected biphasic detections to the later positive lobe."""
    peaks = np.asarray(sorted(set(np.asarray(peaks, dtype=int))), dtype=int)
    if len(peaks) == 0:
        return peaks, np.asarray([], dtype=int), np.asarray([], dtype=int), {
            "triggered": False,
            "count": 0,
            "total": 0,
            "fraction": 0.0,
        }

    count, total, fraction = _late_positive_alignment_stats(raw, qrs_band, peaks, fs)
    triggered = count >= LATE_ALIGN_MIN_COUNT and fraction >= LATE_ALIGN_MIN_FRACTION
    stats = {
        "triggered": bool(triggered),
        "count": int(count),
        "total": int(total),
        "fraction": float(fraction),
    }
    if not triggered:
        return peaks, np.asarray([], dtype=int), np.asarray([], dtype=int), stats

    align_start = int(LATE_ALIGN_SCAN_START_SEC * fs)
    aligned = []
    moved_from = []
    moved_to = []
    for peak in peaks:
        replacement = None
        if peak >= align_start:
            replacement = _late_positive_alignment_candidate(raw, qrs_band, peak, fs)

        if replacement is None:
            aligned.append(int(peak))
        else:
            aligned.append(int(replacement))
            moved_from.append(int(peak))
            moved_to.append(int(replacement))

    return (
        np.asarray(sorted(set(aligned)), dtype=int),
        np.asarray(moved_from, dtype=int),
        np.asarray(moved_to, dtype=int),
        stats,
    )


def _adaptive_rr_cleanup(filtered, qrs_band, peaks, fs=DEFAULT_FS):
    """Remove extra detections that are too dense for the record's rhythm."""
    peaks = np.asarray(sorted(set(np.asarray(peaks, dtype=int))), dtype=int)
    if len(peaks) < 3:
        return peaks, np.asarray([], dtype=int)

    rr = np.diff(peaks)
    valid_rr = rr[(rr > int(0.30 * fs)) & (rr < int(2.0 * fs))]
    typical_rr = np.median(valid_rr) if len(valid_rr) else np.median(rr)
    if not np.isfinite(typical_rr) or typical_rr <= 0:
        return peaks, np.asarray([], dtype=int)

    min_gap = int(max(ADAPTIVE_RR_MIN_SEC * fs, ADAPTIVE_RR_FACTOR * typical_rr))
    keep = []
    removed = []

    def rr_score(peak):
        peak = int(peak)
        radius = max(2, int(REFINE_SEC * fs))
        lo = max(0, peak - radius)
        hi = min(len(filtered), peak + radius + 1)
        if hi <= lo:
            return 0.0
        filtered_amp = abs(filtered[peak] - np.median(filtered[lo:hi]))
        qrs_amp = np.max(np.abs(qrs_band[lo:hi])) if len(qrs_band[lo:hi]) else 0.0
        return float(filtered_amp + ADAPTIVE_RR_QRS_WEIGHT * qrs_amp)

    for peak in peaks:
        if not keep or peak - keep[-1] > min_gap:
            keep.append(int(peak))
            continue

        old = keep[-1]
        if rr_score(peak) > rr_score(old):
            removed.append(old)
            keep[-1] = int(peak)
        else:
            removed.append(int(peak))

    return np.asarray(keep, dtype=int), np.asarray(removed, dtype=int)


def _rescue_quality_peaks(filtered, qrs_band, predicted_peaks, removed_peaks, fs=DEFAULT_FS):
    """Restore isolated high-confidence QRS peaks removed by the quality gate."""
    predicted_peaks = np.asarray(sorted(set(np.asarray(predicted_peaks, dtype=int))), dtype=int)
    removed_peaks = np.asarray(sorted(set(np.asarray(removed_peaks, dtype=int))), dtype=int)
    if len(removed_peaks) == 0:
        return predicted_peaks, np.asarray([], dtype=int)

    abs_filtered = np.abs(filtered)
    qrs_strength = np.abs(qrs_band)
    min_nearest = int(QUALITY_RESCUE_MIN_NEAREST_SEC * fs)
    keep = []
    for peak in removed_peaks:
        peak = int(peak)
        nearest = np.min(np.abs(predicted_peaks - peak)) if len(predicted_peaks) else np.inf
        if nearest <= min_nearest:
            continue

        lo = max(0, peak - int(0.50 * fs))
        hi = min(len(filtered), peak + int(0.50 * fs) + 1)
        if hi <= lo:
            continue

        if abs(qrs_band[peak]) < QUALITY_RESCUE_MIN_QRS_BAND:
            continue
        if np.median(abs_filtered[lo:hi]) >= QUALITY_RESCUE_MAX_ABS_MEDIAN:
            continue
        if np.median(qrs_strength[lo:hi]) >= QUALITY_RESCUE_MAX_QRS_MEDIAN:
            continue

        keep.append(peak)

    if not keep:
        return predicted_peaks, np.asarray([], dtype=int)

    rescued = np.asarray(keep, dtype=int)
    merged = np.asarray(sorted(set(np.concatenate([predicted_peaks, rescued]))), dtype=int)
    return _merge_close_peaks(merged, filtered, fs), rescued


def _final_morphology_cleanup(filtered, qrs_band, peaks, fs=DEFAULT_FS):
    """Remove residual low-confidence detections from the final sequence."""
    peaks = np.asarray(sorted(set(np.asarray(peaks, dtype=int))), dtype=int)
    if len(peaks) == 0:
        return peaks, np.asarray([], dtype=int)

    abs_filtered = np.abs(filtered)
    keep = []
    removed = []
    radius = int(0.50 * fs)
    for peak in peaks:
        peak = int(peak)
        lo = max(0, peak - radius)
        hi = min(len(filtered), peak + radius + 1)
        if hi <= lo:
            keep.append(peak)
            continue

        baseline = np.median(filtered[lo:hi])
        filtered_amp = abs(filtered[peak] - baseline)
        qrs_amp = abs(qrs_band[peak])
        local_abs_median = np.median(abs_filtered[lo:hi])

        noisy_weak_peak = (
            qrs_amp < FINAL_NOISY_QRS_MAX
            and filtered_amp < FINAL_NOISY_FILTERED_MAX
            and local_abs_median >= FINAL_NOISY_ABS_MEDIAN_MIN
        )
        extremely_weak_peak = qrs_amp < FINAL_WEAK_QRS_MAX and filtered_amp < FINAL_WEAK_FILTERED_MAX
        if noisy_weak_peak or extremely_weak_peak:
            removed.append(peak)
        else:
            keep.append(peak)

    return np.asarray(keep, dtype=int), np.asarray(removed, dtype=int)


def _detect_qrs_all(raw, fs=DEFAULT_FS):
    # Shared pipeline for normal detection and debug visualization. Keeping one
    # path avoids evaluation and viewer using slightly different logic.
    raw = np.asarray(raw, dtype=float).ravel()
    filtered = preprocess_ecg(raw, fs)
    qrs_band = qrs_bandpass(raw, fs)

    energy_peaks, energy_integrated, energy_threshold = _energy_peak_candidates_debug(qrs_band, fs)

    if USE_ABS_MAIN_BRANCH:
        # Legacy path: abs(filtered) is the main detector and energy fills gaps.
        main_peaks, abs_filtered, main_threshold = _main_abs_peak_candidates_debug(filtered, fs)
        preliminary_peaks = _fill_long_gaps(main_peaks, energy_peaks, filtered, fs)
    else:
        # Current path: avoid the abs(filtered) detector and use the QRS energy
        # branch directly as the candidate source.
        main_peaks = np.asarray([], dtype=int)
        abs_filtered = np.abs(filtered)
        main_threshold = float("nan")
        preliminary_peaks = _merge_close_peaks(energy_peaks, filtered, fs)

    preliminary_peaks, removed_qrs_band_peaks = _qrs_band_veto(qrs_band, preliminary_peaks, fs)

    quality_mask, noise_score = _quality_mask(
        raw,
        filtered,
        qrs_band,
        energy_integrated,
        fs,
        peaks=preliminary_peaks,
        energy_peaks=energy_peaks,
    )

    # This version intentionally does not use the earlier polarity/refinement
    # detector. It improved a few screenshots but reduced general robustness by
    # choosing the wrong side of biphasic QRS complexes in records like 6/17/25.
    preliminary_peaks = preliminary_peaks[(preliminary_peaks >= 0) & (preliminary_peaks < len(filtered))]
    in_quality = quality_mask[preliminary_peaks] if len(preliminary_peaks) else np.asarray([], dtype=bool)
    quality_peaks = preliminary_peaks[in_quality]
    removed_noise_peaks = preliminary_peaks[~in_quality]
    shaped_peaks, removed_shape_peaks = _shape_filter(filtered, qrs_band, energy_integrated, quality_peaks, fs)
    predicted_peaks = _merge_close_peaks(shaped_peaks, filtered, fs)
    aligned_peaks, aligned_from_peaks, aligned_to_peaks, alignment_stats = _align_late_positive_lobes(
        raw,
        qrs_band,
        predicted_peaks,
        fs,
    )
    predicted_peaks, removed_rr_peaks = _adaptive_rr_cleanup(filtered, qrs_band, aligned_peaks, fs)
    rescue_input_peaks = np.concatenate([removed_noise_peaks, removed_shape_peaks, removed_rr_peaks])
    predicted_peaks, rescued_quality_peaks = _rescue_quality_peaks(
        filtered,
        qrs_band,
        predicted_peaks,
        rescue_input_peaks,
        fs,
    )
    predicted_peaks, removed_rr_peaks_after_rescue = _adaptive_rr_cleanup(filtered, qrs_band, predicted_peaks, fs)
    if alignment_stats["triggered"]:
        removed_final_morphology_peaks = np.asarray([], dtype=int)
    else:
        predicted_peaks, removed_final_morphology_peaks = _final_morphology_cleanup(
            filtered,
            qrs_band,
            predicted_peaks,
            fs,
        )

    return {
        "filtered": filtered,
        "abs_filtered": abs_filtered,
        "main_threshold": main_threshold,
        "qrs_band": qrs_band,
        "energy_integrated": energy_integrated,
        "energy_threshold": energy_threshold,
        "main_peaks": main_peaks.astype(int),
        "energy_peaks": energy_peaks.astype(int),
        "quality_input_peaks": preliminary_peaks.astype(int),
        "quality_mask": quality_mask,
        "noise_score": noise_score,
        "removed_qrs_band_peaks": removed_qrs_band_peaks.astype(int),
        "removed_noise_peaks": removed_noise_peaks.astype(int),
        "removed_shape_peaks": removed_shape_peaks.astype(int),
        "aligned_from_peaks": aligned_from_peaks.astype(int),
        "aligned_to_peaks": aligned_to_peaks.astype(int),
        "alignment_triggered": alignment_stats["triggered"],
        "alignment_candidate_count": alignment_stats["count"],
        "alignment_candidate_total": alignment_stats["total"],
        "alignment_candidate_fraction": alignment_stats["fraction"],
        "removed_rr_peaks": removed_rr_peaks.astype(int),
        "rescued_quality_peaks": rescued_quality_peaks.astype(int),
        "removed_rr_peaks_after_rescue": removed_rr_peaks_after_rescue.astype(int),
        "removed_final_morphology_peaks": removed_final_morphology_peaks.astype(int),
        "predicted_peaks": predicted_peaks.astype(int),
    }


def detect_qrs(raw, fs=DEFAULT_FS):
    """Public detector used by evaluation code. Returns filtered ECG and peaks."""
    debug = _detect_qrs_all(raw, fs)
    return debug["filtered"], debug["predicted_peaks"].astype(int)


def detect_qrs_debug(raw, fs=DEFAULT_FS):
    """Return all intermediate arrays needed by the interactive debug viewer."""
    return _detect_qrs_all(raw, fs)


def predict_peaks(raw):
    """Compatibility wrapper for callers that only need peak sample indices."""
    return detect_qrs(raw, DEFAULT_FS)


def load_recording(mat_path, index, max_len=None):
    """Load one ECG record and optional expert QRS labels from a MATLAB file."""
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
    """Save a static predicted-vs-expert overlay for one record segment."""
    import os
    import matplotlib

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(out_path.parent / ".matplotlib"))

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

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
