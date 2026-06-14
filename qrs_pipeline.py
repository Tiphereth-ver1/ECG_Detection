from pathlib import Path

import numpy as np
from scipy.io import loadmat
from scipy.signal import butter, filtfilt, find_peaks, iirnotch, sosfiltfilt

# The dataset is sampled at 100 Hz, so 1 sample = 10 ms.
DEFAULT_FS = 100

# Main branch settings. The detector uses abs(filtered ECG), which works for
# both upright and inverted QRS complexes.
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
QUALITY_PAD_SEC = 0.50
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
OSC_NOISE_MIN_PEAKS = 7
OSC_NOISE_MIN_ENERGY_PEAKS = 14
OSC_NOISE_MIN_ZERO_CROSSINGS = 95
OSC_NOISE_QRS_SNR_LOW = 6.0
OSC_NOISE_MIN_FILTERED_STD = 450
OSC_NOISE_MAX_FILTERED_STD = 1000
OSC_NOISE_MAX_RAW_RANGE = 9000
OSC_NOISE_MAX_QRS_P95 = 1800
OSCILLATORY_NOISE_SCORE = 4.0
DENSE_LOW_CONTRAST_MIN_PEAKS = 7
DENSE_LOW_CONTRAST_MIN_ENERGY_PEAKS = 14
DENSE_LOW_CONTRAST_MIN_ZERO_CROSSINGS = 80
DENSE_LOW_CONTRAST_MAX_ZERO_CROSSINGS = 98
DENSE_LOW_CONTRAST_MIN_ABS_MEDIAN = 380
DENSE_LOW_CONTRAST_MIN_FILTERED_STD = 700
DENSE_LOW_CONTRAST_MAX_FILTERED_STD = 1350
DENSE_LOW_CONTRAST_MAX_RAW_RANGE = 9500
DENSE_LOW_CONTRAST_MAX_QRS_P95 = 1400
DENSE_LOW_CONTRAST_MAX_QRS_SNR = 5.3
DENSE_LOW_CONTRAST_MIN_QRS_MEDIAN = 150
SHAPE_REFRACTORY_SEC = 0.32
SHAPE_SCORE_QRS_WEIGHT = 1.50
SHAPE_SCORE_SLOPE_WEIGHT = 0.75

# Early-negative alignment is active but guarded at record level. It only fires
# for records with a sustained late-positive/T-like morphology pattern.
EARLY_ALIGN_LOOKBACK_MIN_SEC = 0.18
EARLY_ALIGN_LOOKBACK_MAX_SEC = 0.34
EARLY_ALIGN_MIN_CURRENT_QRS = 70
EARLY_ALIGN_MIN_EARLY_QRS = 70
EARLY_ALIGN_MIN_QRS_RATIO = 0.65
EARLY_ALIGN_MIN_FILTERED_DELTA = 80
EARLY_ALIGN_MIN_COUNT = 250
EARLY_ALIGN_MIN_FRACTION = 0.03

# Final sequence cleanup. This is a record-adaptive refractory layer: records
# with dense artifact clusters often have extra peaks at about half the local RR
# interval. Keep the stronger local QRS shape when two detections are too close
# for that record's rhythm. This also protects HRV metrics from short-RR false
# positives in noisy windows.
ADAPTIVE_RR_MIN_SEC = 0.45
ADAPTIVE_RR_FACTOR = 0.50
ADAPTIVE_RR_QRS_WEIGHT = 1.00

# Very narrow final duplicate cleanup. Do not raise the global adaptive RR
# threshold because training and HRV include real 0.45-0.50 s RR intervals.
# Instead, only remove the weaker peak from a close pair when it is much weaker
# than its neighbor and sits in a locally oscillatory QRS-band segment.
WEAK_SHORT_RR_MAX_GAP_SEC = 0.56
WEAK_SHORT_RR_MAX_SCORE_RATIO = 0.35
WEAK_SHORT_RR_MAX_WEAK_SCORE = 1500
WEAK_SHORT_RR_MIN_ZERO_CROSSINGS = 13
WEAK_SHORT_RR_SCORE_RADIUS_SEC = 0.08
WEAK_SHORT_RR_PAIR_PAD_SEC = 0.15

# Local recovery for QRS complexes sitting next to artifact. The broad quality
# mask can reject a whole 5-second block when an artifact crosses it, but clear,
# narrow QRS peaks near the artifact edge should still be restored.
ARTIFACT_RESCUE_MIN_NEAREST_SEC = 0.30
ARTIFACT_RESCUE_MIN_NOISE_SCORE = 2.5
ARTIFACT_RESCUE_MIN_QRS_BAND = 1200
ARTIFACT_RESCUE_MIN_FILTERED = 2000
ARTIFACT_RESCUE_MAX_ABS_MEDIAN = 520
ARTIFACT_RESCUE_MAX_RAW_RANGE = 12000
ARTIFACT_RESCUE_MAX_FILTERED_STD = 1600
ARTIFACT_RESCUE_MIN_CONTRAST = 4.5
ARTIFACT_RESCUE_STRONG_QRS_BAND = 2200
ARTIFACT_RESCUE_STRONG_FILTERED = 3200
ARTIFACT_RESCUE_MAX_STRONG_FILTERED = 7000
ARTIFACT_RESCUE_MAX_STRONG_QRS_BAND = 4500
ARTIFACT_RESCUE_MAX_HALF_WIDTH = 4
LOW_AMP_RESCUE_MIN_QRS_BAND = 150
LOW_AMP_RESCUE_MIN_FILTERED = 275
LOW_AMP_RESCUE_MAX_ABS_MEDIAN = 120
LOW_AMP_RESCUE_MAX_RAW_RANGE = 1200
LOW_AMP_RESCUE_MAX_FILTERED_STD = 180
LOW_AMP_RESCUE_MIN_CONTRAST = 3.2
MEDIUM_AMP_RESCUE_MIN_QRS_BAND = 500
MEDIUM_AMP_RESCUE_MIN_FILTERED = 850
MEDIUM_AMP_RESCUE_MAX_ABS_MEDIAN = 280
MEDIUM_AMP_RESCUE_MAX_RAW_RANGE = 3300
MEDIUM_AMP_RESCUE_MAX_FILTERED_STD = 520
MEDIUM_AMP_RESCUE_MIN_CONTRAST = 5.0
RHYTHM_RESCUE_MIN_GAP_SEC = 0.43
RHYTHM_RESCUE_MAX_GAP_SEC = 1.50
RHYTHM_RESCUE_MAX_GAP_RATIO = 1.85
RHYTHM_RESCUE_MIN_QRS_BAND = 1000
RHYTHM_RESCUE_MAX_QRS_BAND = 5000
RHYTHM_RESCUE_MIN_FILTERED = 1200
RHYTHM_RESCUE_MAX_FILTERED = 9000
RHYTHM_RESCUE_MAX_ABS_MEDIAN = 650
RHYTHM_RESCUE_MAX_RAW_RANGE = 13000
RHYTHM_RESCUE_MAX_FILTERED_STD = 2800
RHYTHM_RESCUE_MIN_CONTRAST = 5.0
RHYTHM_RESCUE_MAX_HALF_WIDTH = 4
MILD_MASK_RHYTHM_RESCUE_MAX_NOISE_SCORE = 1.5
MILD_MASK_RHYTHM_RESCUE_MIN_QRS_BAND = 250
MILD_MASK_RHYTHM_RESCUE_MAX_QRS_BAND = 900
MILD_MASK_RHYTHM_RESCUE_MIN_FILTERED = 400
MILD_MASK_RHYTHM_RESCUE_MAX_FILTERED = 900
MILD_MASK_RHYTHM_RESCUE_MAX_ABS_MEDIAN = 160
MILD_MASK_RHYTHM_RESCUE_MAX_RAW_RANGE = 1600
MILD_MASK_RHYTHM_RESCUE_MAX_FILTERED_STD = 230
MILD_MASK_RHYTHM_RESCUE_MIN_CONTRAST = 3.4
MILD_MASK_RHYTHM_RESCUE_MAX_HALF_WIDTH = 6


def default_train_mat():
    """Return the default training .mat path used by the CLI and viewer."""
    return Path(__file__).resolve().parent / "dataSet" / "ProjectTrainData.mat"

def default_test_mat():
    """Return the default test .mat path used by the final generator."""
    return Path(__file__).resolve().parent / "dataSet" / "ProjectTestData.mat"




def qrs_settings():
    return {
        "MAIN_PERCENTILE": MAIN_PERCENTILE,
        "QUALITY_PAD_SEC": QUALITY_PAD_SEC,
        "ARTIFACT_QRS_RESCUE": True,
        "EARLY_NEGATIVE_ALIGNMENT": True,
        "ADAPTIVE_RR_CLEANUP": True,
        "WEAK_SHORT_RR_CLEANUP": True,
        "RHYTHM_RESCUE_MIN_GAP_SEC": RHYTHM_RESCUE_MIN_GAP_SEC,
        "OSCILLATORY_NOISE_SCORE": OSCILLATORY_NOISE_SCORE,
        "DENSE_LOW_CONTRAST_MIN_PEAKS": DENSE_LOW_CONTRAST_MIN_PEAKS,
        "DENSE_LOW_CONTRAST_MAX_QRS_SNR": DENSE_LOW_CONTRAST_MAX_QRS_SNR,
        "tolerance_ms": 50,
    }

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

    peaks = np.asarray(peaks if peaks is not None else [], dtype=int)
    density_peaks, _density_removed = _shape_filter(filtered, qrs_band, peaks, fs)
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
        local_density_peaks = density_peaks[(density_peaks >= start) & (density_peaks < end)]
        peak_count = len(local_density_peaks)
        energy_peak_count = int(np.sum((energy_peaks >= start) & (energy_peaks < end)))
        raw_range = float(np.ptp(raw[start:end]))
        filtered_std = float(np.std(filtered[start:end]))
        abs_median = float(np.median(abs_filtered[start:end]))
        abs_max = float(np.max(abs_filtered[start:end]))
        energy_max = float(np.max(energy[start:end])) if len(energy) else 0.0
        local_qrs_band = qrs_band[start:end]
        qrs_zero_crossings = int(np.sum(np.diff(np.signbit(local_qrs_band)) != 0))
        qrs_p95 = float(np.percentile(np.abs(local_qrs_band), 95)) if end > start else 0.0

        # This is not a physical SNR. It is only a simple contrast score:
        # median detected-QRS height divided by the median background height in
        # the same window. Real QRS windows usually have a large ratio; artifact
        # sections often produce many weak "peaks" that do not stand out.
        if peak_count:
            qrs_snr = float(np.median(abs_filtered[local_density_peaks]) / (abs_median + 1e-9))
        else:
            qrs_snr = float("inf")

        low_snr = (
            peak_count >= QRS_SNR_MIN_PEAKS
            and energy_peak_count >= QRS_SNR_MIN_ENERGY_PEAKS
            and qrs_snr < QRS_SNR_LOW
        )
        oscillatory_noise = (
            peak_count >= OSC_NOISE_MIN_PEAKS
            and energy_peak_count >= OSC_NOISE_MIN_ENERGY_PEAKS
            and qrs_zero_crossings >= OSC_NOISE_MIN_ZERO_CROSSINGS
            and qrs_snr < OSC_NOISE_QRS_SNR_LOW
            and filtered_std >= OSC_NOISE_MIN_FILTERED_STD
            and filtered_std <= OSC_NOISE_MAX_FILTERED_STD
            and raw_range <= OSC_NOISE_MAX_RAW_RANGE
            and qrs_p95 <= OSC_NOISE_MAX_QRS_P95
        )
        dense_low_contrast_noise = (
            peak_count >= DENSE_LOW_CONTRAST_MIN_PEAKS
            and energy_peak_count >= DENSE_LOW_CONTRAST_MIN_ENERGY_PEAKS
            and qrs_zero_crossings >= DENSE_LOW_CONTRAST_MIN_ZERO_CROSSINGS
            and qrs_zero_crossings <= DENSE_LOW_CONTRAST_MAX_ZERO_CROSSINGS
            and abs_median >= DENSE_LOW_CONTRAST_MIN_ABS_MEDIAN
            and filtered_std >= DENSE_LOW_CONTRAST_MIN_FILTERED_STD
            and filtered_std <= DENSE_LOW_CONTRAST_MAX_FILTERED_STD
            and raw_range <= DENSE_LOW_CONTRAST_MAX_RAW_RANGE
            and qrs_p95 <= DENSE_LOW_CONTRAST_MAX_QRS_P95
            and qrs_snr <= DENSE_LOW_CONTRAST_MAX_QRS_SNR
            and float(np.median(np.abs(local_qrs_band))) >= DENSE_LOW_CONTRAST_MIN_QRS_MEDIAN
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
        if oscillatory_noise:
            bad = True
            score = max(score, OSCILLATORY_NOISE_SCORE)
        if dense_low_contrast_noise:
            bad = True
            score = max(score, OSCILLATORY_NOISE_SCORE)

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
    slope_lo = max(0, int(peak) - int(0.05 * fs))
    slope_hi = min(len(filtered), int(peak) + int(0.05 * fs) + 1)
    slope = np.max(np.abs(np.diff(filtered[slope_lo:slope_hi]))) if slope_hi - slope_lo > 1 else 0.0
    return float(amp + SHAPE_SCORE_QRS_WEIGHT * qrs_amp + SHAPE_SCORE_SLOPE_WEIGHT * slope)


def _shape_filter(filtered, qrs_band, peaks, fs=DEFAULT_FS):
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


def _early_negative_alignment_candidate(filtered, qrs_band, peak, fs=DEFAULT_FS):
    """Return an earlier negative QRS candidate when a later positive lobe was selected."""
    peak = int(peak)
    lookback_start = peak - int(EARLY_ALIGN_LOOKBACK_MAX_SEC * fs)
    lookback_end = peak - int(EARLY_ALIGN_LOOKBACK_MIN_SEC * fs) + 1
    if lookback_end <= lookback_start or lookback_end <= 0:
        return None

    lookback_start = max(0, lookback_start)
    lookback = qrs_band[lookback_start:lookback_end]
    if len(lookback) == 0:
        return None

    early = lookback_start + int(np.argmin(lookback))
    current_qrs = float(qrs_band[peak])
    early_qrs = float(qrs_band[early])
    if current_qrs <= EARLY_ALIGN_MIN_CURRENT_QRS:
        return None
    if early_qrs >= -EARLY_ALIGN_MIN_EARLY_QRS:
        return None
    if abs(early_qrs) < EARLY_ALIGN_MIN_QRS_RATIO * abs(current_qrs):
        return None

    context_lo = max(0, peak - int(0.12 * fs))
    context_hi = min(len(filtered), peak + int(0.12 * fs) + 1)
    if context_hi <= context_lo:
        return None

    baseline = float(np.median(filtered[context_lo:context_hi]))
    if filtered[peak] - baseline <= EARLY_ALIGN_MIN_FILTERED_DELTA:
        return None
    if filtered[early] - baseline >= -EARLY_ALIGN_MIN_FILTERED_DELTA:
        return None

    return int(early)


def _early_negative_alignment_stats(filtered, qrs_band, peaks, fs=DEFAULT_FS):
    peaks = np.asarray(peaks, dtype=int)
    if len(peaks) == 0:
        return 0, 0, 0.0

    count = 0
    for peak in peaks:
        if _early_negative_alignment_candidate(filtered, qrs_band, peak, fs) is not None:
            count += 1
    return int(count), int(len(peaks)), float(count / len(peaks))


def _apply_early_negative_alignment_guard(filtered, qrs_band, peaks, fs=DEFAULT_FS):
    """Move a record-level late-positive/T-like morphology back to the QRS valley."""
    peaks = np.asarray(sorted(set(np.asarray(peaks, dtype=int))), dtype=int)
    if len(peaks) == 0:
        return peaks, np.asarray([], dtype=int), np.asarray([], dtype=int), {
            "triggered": False,
            "count": 0,
            "total": 0,
            "fraction": 0.0,
        }

    count, total, fraction = _early_negative_alignment_stats(filtered, qrs_band, peaks, fs)
    triggered = count >= EARLY_ALIGN_MIN_COUNT and fraction >= EARLY_ALIGN_MIN_FRACTION
    stats = {
        "triggered": bool(triggered),
        "count": int(count),
        "total": int(total),
        "fraction": float(fraction),
    }
    if not triggered:
        return peaks, np.asarray([], dtype=int), np.asarray([], dtype=int), stats

    aligned = []
    moved_from = []
    moved_to = []
    for peak in peaks:
        replacement = _early_negative_alignment_candidate(filtered, qrs_band, peak, fs)
        if replacement is None:
            aligned.append(int(peak))
        else:
            aligned.append(int(replacement))
            moved_from.append(int(peak))
            moved_to.append(int(replacement))

    return (
        _merge_close_peaks(np.asarray(aligned, dtype=int), filtered, fs),
        np.asarray(moved_from, dtype=int),
        np.asarray(moved_to, dtype=int),
        stats,
    )


def _adaptive_rr_cleanup(filtered, qrs_band, peaks, fs=DEFAULT_FS, protected_peaks=None):
    """Remove extra detections that are too dense for the record's rhythm."""
    peaks = np.asarray(sorted(set(np.asarray(peaks, dtype=int))), dtype=int)
    if len(peaks) < 3:
        return peaks, np.asarray([], dtype=int)
    protected = set(np.asarray(protected_peaks if protected_peaks is not None else [], dtype=int).tolist())

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
        old_protected = int(old) in protected
        peak_protected = int(peak) in protected
        if old_protected and peak_protected:
            keep.append(int(peak))
        elif old_protected and not peak_protected:
            removed.append(int(peak))
        elif peak_protected and not old_protected:
            removed.append(old)
            keep[-1] = int(peak)
        elif rr_score(peak) > rr_score(old):
            removed.append(old)
            keep[-1] = int(peak)
        else:
            removed.append(int(peak))

    return np.asarray(keep, dtype=int), np.asarray(removed, dtype=int)


def _weak_short_rr_cleanup(filtered, qrs_band, peaks, fs=DEFAULT_FS):
    """Remove a clearly weaker duplicate from a locally noisy short-RR pair."""
    peaks = np.asarray(sorted(set(np.asarray(peaks, dtype=int))), dtype=int)
    if len(peaks) < 2:
        return peaks, np.asarray([], dtype=int)

    max_gap = int(WEAK_SHORT_RR_MAX_GAP_SEC * fs)
    score_radius = max(1, int(WEAK_SHORT_RR_SCORE_RADIUS_SEC * fs))
    pair_pad = max(1, int(WEAK_SHORT_RR_PAIR_PAD_SEC * fs))
    keep = []
    removed = []
    i = 0

    def local_score(peak):
        peak = int(peak)
        lo = max(0, peak - score_radius)
        hi = min(len(filtered), peak + score_radius + 1)
        if hi <= lo:
            return 0.0
        return float(np.max(np.abs(filtered[lo:hi])) + np.max(np.abs(qrs_band[lo:hi])))

    while i < len(peaks):
        if i + 1 < len(peaks) and peaks[i + 1] - peaks[i] <= max_gap:
            left = int(peaks[i])
            right = int(peaks[i + 1])
            left_score = local_score(left)
            right_score = local_score(right)
            if left_score <= right_score:
                weak_peak, strong_peak = left, right
                weak_score, strong_score = left_score, right_score
            else:
                weak_peak, strong_peak = right, left
                weak_score, strong_score = right_score, left_score

            lo = max(0, min(left, right) - pair_pad)
            hi = min(len(qrs_band), max(left, right) + pair_pad + 1)
            pair_qrs = qrs_band[lo:hi]
            zero_crossings = int(np.sum(np.diff(np.signbit(pair_qrs)) != 0)) if len(pair_qrs) > 1 else 0
            weak_ratio = weak_score / (strong_score + 1e-9)
            weak_duplicate = (
                zero_crossings >= WEAK_SHORT_RR_MIN_ZERO_CROSSINGS
                and weak_score <= WEAK_SHORT_RR_MAX_WEAK_SCORE
                and weak_ratio <= WEAK_SHORT_RR_MAX_SCORE_RATIO
            )
            if weak_duplicate:
                keep.append(strong_peak)
                removed.append(weak_peak)
                i += 2
                continue

        keep.append(int(peaks[i]))
        i += 1

    return np.asarray(sorted(set(keep)), dtype=int), np.asarray(removed, dtype=int)


def _rescue_artifact_neighbor_qrs(raw, filtered, qrs_band, noise_score, predicted_peaks, removed_noise_peaks, fs=DEFAULT_FS):
    """Restore high-confidence QRS peaks removed only because their window was noisy."""
    predicted_peaks = np.asarray(sorted(set(np.asarray(predicted_peaks, dtype=int))), dtype=int)
    removed_noise_peaks = np.asarray(sorted(set(np.asarray(removed_noise_peaks, dtype=int))), dtype=int)
    if len(removed_noise_peaks) == 0:
        return predicted_peaks, np.asarray([], dtype=int)

    abs_filtered = np.abs(filtered)
    qrs_strength = np.abs(qrs_band)
    min_nearest = int(ARTIFACT_RESCUE_MIN_NEAREST_SEC * fs)
    short_radius = max(2, int(0.08 * fs))
    context_radius = max(1, int(0.60 * fs))
    rhythm_context_radius = max(1, int(0.25 * fs))
    rhythm_min_gap = int(RHYTHM_RESCUE_MIN_GAP_SEC * fs)
    rhythm_max_gap = int(RHYTHM_RESCUE_MAX_GAP_SEC * fs)
    rhythm_candidates = np.asarray(
        sorted(set(np.concatenate([predicted_peaks, removed_noise_peaks]))),
        dtype=int,
    )
    keep = []

    for peak in removed_noise_peaks:
        peak = int(peak)
        nearest = np.min(np.abs(predicted_peaks - peak)) if len(predicted_peaks) else np.inf
        if nearest <= min_nearest:
            continue
        if peak >= len(noise_score):
            continue
        if noise_score[peak] >= OSCILLATORY_NOISE_SCORE:
            continue

        slo = max(0, peak - short_radius)
        shi = min(len(filtered), peak + short_radius + 1)
        clo = max(0, peak - context_radius)
        chi = min(len(filtered), peak + context_radius + 1)
        if shi <= slo or chi <= clo:
            continue

        local_filtered = float(np.max(abs_filtered[slo:shi]))
        local_qrs = float(np.max(qrs_strength[slo:shi]))
        context_median = float(np.median(abs_filtered[clo:chi]))
        context_raw_range = float(np.ptp(raw[clo:chi]))
        context_filtered_std = float(np.std(filtered[clo:chi]))
        contrast = local_filtered / (context_median + 1e-9)
        half_level = context_median + 0.5 * max(0.0, local_filtered - context_median)
        half_width = int(np.sum(abs_filtered[slo:shi] >= half_level))

        rlo = max(0, peak - rhythm_context_radius)
        rhi = min(len(filtered), peak + rhythm_context_radius + 1)
        rhythm_median = float(np.median(abs_filtered[rlo:rhi])) if rhi > rlo else context_median
        rhythm_raw_range = float(np.ptp(raw[rlo:rhi])) if rhi > rlo else context_raw_range
        rhythm_filtered_std = float(np.std(filtered[rlo:rhi])) if rhi > rlo else context_filtered_std
        rhythm_contrast = local_filtered / (rhythm_median + 1e-9)
        rhythm_half_level = rhythm_median + 0.5 * max(0.0, local_filtered - rhythm_median)
        rhythm_half_width = int(np.sum(abs_filtered[slo:shi] >= rhythm_half_level))

        order_idx = int(np.searchsorted(rhythm_candidates, peak))
        left_gap = None
        right_gap = None
        if order_idx > 0:
            left_gap = int(peak - rhythm_candidates[order_idx - 1])
        if order_idx + 1 < len(rhythm_candidates):
            right_gap = int(rhythm_candidates[order_idx + 1] - peak)
        has_rhythm_neighbors = (
            left_gap is not None
            and right_gap is not None
            and rhythm_min_gap <= left_gap <= rhythm_max_gap
            and rhythm_min_gap <= right_gap <= rhythm_max_gap
            and min(left_gap, right_gap) / max(left_gap, right_gap) >= 1.0 / RHYTHM_RESCUE_MAX_GAP_RATIO
        )

        low_amplitude_clean_qrs = (
            local_qrs >= LOW_AMP_RESCUE_MIN_QRS_BAND
            and local_filtered >= LOW_AMP_RESCUE_MIN_FILTERED
            and context_median <= LOW_AMP_RESCUE_MAX_ABS_MEDIAN
            and context_raw_range <= LOW_AMP_RESCUE_MAX_RAW_RANGE
            and context_filtered_std <= LOW_AMP_RESCUE_MAX_FILTERED_STD
            and contrast >= LOW_AMP_RESCUE_MIN_CONTRAST
        )

        medium_amplitude_clean_qrs = (
            local_qrs >= MEDIUM_AMP_RESCUE_MIN_QRS_BAND
            and local_filtered >= MEDIUM_AMP_RESCUE_MIN_FILTERED
            and context_median <= MEDIUM_AMP_RESCUE_MAX_ABS_MEDIAN
            and context_raw_range <= MEDIUM_AMP_RESCUE_MAX_RAW_RANGE
            and context_filtered_std <= MEDIUM_AMP_RESCUE_MAX_FILTERED_STD
            and contrast >= MEDIUM_AMP_RESCUE_MIN_CONTRAST
        )

        artifact_edge_qrs = (
            local_qrs >= ARTIFACT_RESCUE_MIN_QRS_BAND
            and local_filtered >= ARTIFACT_RESCUE_MIN_FILTERED
            and context_median <= ARTIFACT_RESCUE_MAX_ABS_MEDIAN
            and context_raw_range <= ARTIFACT_RESCUE_MAX_RAW_RANGE
            and context_filtered_std <= ARTIFACT_RESCUE_MAX_FILTERED_STD
            and contrast >= ARTIFACT_RESCUE_MIN_CONTRAST
        )

        strong_artifact_edge_qrs = (
            noise_score[peak] >= ARTIFACT_RESCUE_MIN_NOISE_SCORE
            and local_qrs >= ARTIFACT_RESCUE_STRONG_QRS_BAND
            and local_qrs <= ARTIFACT_RESCUE_MAX_STRONG_QRS_BAND
            and local_filtered >= ARTIFACT_RESCUE_STRONG_FILTERED
            and local_filtered <= ARTIFACT_RESCUE_MAX_STRONG_FILTERED
            and context_median <= ARTIFACT_RESCUE_MAX_ABS_MEDIAN
            and contrast >= 3.2
            and half_width <= ARTIFACT_RESCUE_MAX_HALF_WIDTH
        )

        regular_rhythm_masked_qrs = (
            has_rhythm_neighbors
            and local_qrs >= RHYTHM_RESCUE_MIN_QRS_BAND
            and local_qrs <= RHYTHM_RESCUE_MAX_QRS_BAND
            and local_filtered >= RHYTHM_RESCUE_MIN_FILTERED
            and local_filtered <= RHYTHM_RESCUE_MAX_FILTERED
            and rhythm_median <= RHYTHM_RESCUE_MAX_ABS_MEDIAN
            and rhythm_raw_range <= RHYTHM_RESCUE_MAX_RAW_RANGE
            and rhythm_filtered_std <= RHYTHM_RESCUE_MAX_FILTERED_STD
            and rhythm_contrast >= RHYTHM_RESCUE_MIN_CONTRAST
            and rhythm_half_width <= RHYTHM_RESCUE_MAX_HALF_WIDTH
        )

        mild_mask_regular_qrs = (
            has_rhythm_neighbors
            and noise_score[peak] <= MILD_MASK_RHYTHM_RESCUE_MAX_NOISE_SCORE
            and local_qrs >= MILD_MASK_RHYTHM_RESCUE_MIN_QRS_BAND
            and local_qrs <= MILD_MASK_RHYTHM_RESCUE_MAX_QRS_BAND
            and local_filtered >= MILD_MASK_RHYTHM_RESCUE_MIN_FILTERED
            and local_filtered <= MILD_MASK_RHYTHM_RESCUE_MAX_FILTERED
            and context_median <= MILD_MASK_RHYTHM_RESCUE_MAX_ABS_MEDIAN
            and context_raw_range <= MILD_MASK_RHYTHM_RESCUE_MAX_RAW_RANGE
            and context_filtered_std <= MILD_MASK_RHYTHM_RESCUE_MAX_FILTERED_STD
            and contrast >= MILD_MASK_RHYTHM_RESCUE_MIN_CONTRAST
            and half_width <= MILD_MASK_RHYTHM_RESCUE_MAX_HALF_WIDTH
        )

        if (
            low_amplitude_clean_qrs
            or medium_amplitude_clean_qrs
            or artifact_edge_qrs
            or strong_artifact_edge_qrs
            or regular_rhythm_masked_qrs
            or mild_mask_regular_qrs
        ):
            keep.append(peak)

    if not keep:
        return predicted_peaks, np.asarray([], dtype=int)

    rescued = np.asarray(keep, dtype=int)
    merged = np.asarray(sorted(set(np.concatenate([predicted_peaks, rescued]))), dtype=int)
    return _merge_close_peaks(merged, filtered, fs), rescued


def _detect_qrs_all(raw, fs=DEFAULT_FS):
    # Shared pipeline for normal detection and debug visualization. Keeping one
    # path avoids evaluation and viewer using slightly different logic.
    raw = np.asarray(raw, dtype=float).ravel()
    filtered = preprocess_ecg(raw, fs)
    qrs_band = qrs_bandpass(raw, fs)

    energy_peaks, energy_integrated, energy_threshold = _energy_peak_candidates_debug(qrs_band, fs)
    main_peaks, abs_filtered, main_threshold = _main_abs_peak_candidates_debug(filtered, fs)
    preliminary_peaks = _fill_long_gaps(main_peaks, energy_peaks, filtered, fs)

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
    shaped_peaks, removed_shape_peaks = _shape_filter(filtered, qrs_band, quality_peaks, fs)
    predicted_peaks = _merge_close_peaks(shaped_peaks, filtered, fs)

    predicted_peaks, rescued_artifact_peaks = _rescue_artifact_neighbor_qrs(
        raw,
        filtered,
        qrs_band,
        noise_score,
        predicted_peaks,
        removed_noise_peaks,
        fs,
    )

    aligned_peaks, early_aligned_from_peaks, early_aligned_to_peaks, early_alignment_stats = (
        _apply_early_negative_alignment_guard(
            filtered,
            qrs_band,
            predicted_peaks,
            fs,
        )
    )

    predicted_peaks, removed_rr_peaks = _adaptive_rr_cleanup(
        filtered,
        qrs_band,
        aligned_peaks,
        fs,
        protected_peaks=rescued_artifact_peaks,
    )
    predicted_peaks, removed_weak_short_rr_peaks = _weak_short_rr_cleanup(
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
        "removed_noise_peaks": removed_noise_peaks.astype(int),
        "removed_shape_peaks": removed_shape_peaks.astype(int),
        "early_aligned_from_peaks": early_aligned_from_peaks.astype(int),
        "early_aligned_to_peaks": early_aligned_to_peaks.astype(int),
        "early_alignment_triggered": early_alignment_stats["triggered"],
        "early_alignment_candidate_count": early_alignment_stats["count"],
        "early_alignment_candidate_total": early_alignment_stats["total"],
        "early_alignment_candidate_fraction": early_alignment_stats["fraction"],
        "removed_rr_peaks": removed_rr_peaks.astype(int),
        "rescued_artifact_peaks": rescued_artifact_peaks.astype(int),
        "removed_weak_short_rr_peaks": removed_weak_short_rr_peaks.astype(int),
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
    _filtered, peaks = detect_qrs(raw, DEFAULT_FS)
    return peaks


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
