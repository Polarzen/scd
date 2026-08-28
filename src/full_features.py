"""Bounded 5-minute ECG extraction for the full MUSIC Holter cohort.

The Phase 4 extractor deliberately keeps the Phase 3 feature names and
ordering.  It differs from the legacy extractor only in its unbounded,
non-overlapping 5-minute schedule and in the explicit NN-RR mask used by the
HRV features.  A window reader never receives outcome or label information and
issues one exact ``wfdb.rdrecord`` segment read; no signal is padded or cached.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import wfdb
from scipy.signal import butter, filtfilt, find_peaks
from scipy.spatial.distance import pdist, squareform


# The public order is frozen by Phase 3 and must not be changed in Phase 4.
FEATURE_NAMES: tuple[str, ...] = (
    "sig_mean",
    "sig_std",
    "sig_p2p",
    "sig_skew",
    "sig_kurt",
    "beats",
    "beats_per_min",
    "mean_rr",
    "sdnn",
    "rmssd",
    "pnn50",
    "mean_hr",
    "rr_cv",
    "rr_sampen",
    "rr_apen",
    "rr_dfa_alpha",
    "pow_lf",
    "pow_mf",
    "pow_hf",
    "pow_hf_ratio",
)

FEATURE_IDS: tuple[str, ...] = tuple(f"feature_{index:02d}" for index in range(1, 21))
FEATURE_VALIDITY_COLUMNS: tuple[str, ...] = tuple(f"{name}_valid" for name in FEATURE_NAMES)
FEATURE_VALIDITY_IDS: tuple[str, ...] = tuple(f"{feature_id}_valid" for feature_id in FEATURE_IDS)
REQUIRES_NN_FEATURE_IDS: tuple[str, ...] = tuple(f"feature_{index:02d}" for index in range(8, 17))
REQUIRES_NN_FEATURE_NAMES: tuple[str, ...] = FEATURE_NAMES[7:16]
SIGNAL_QUALITY_FEATURE_NAMES: tuple[str, ...] = ("sig_skew", "sig_kurt")
FFT_FEATURE_NAMES: tuple[str, ...] = ("pow_lf", "pow_mf", "pow_hf", "pow_hf_ratio")

WINDOW_SEC = 300
FIRST_START_SEC = 60
INTERVAL_SEC = 300
CHANNEL_INDEX = 0
MIN_VALID_NN = 10
MAX_INVALID_RR_RATIO = 0.20
RR_MIN_SEC = 0.30
RR_MAX_SEC = 2.00
LOCAL_MEDIAN_WINDOW = 5
LOCAL_MEDIAN_TOLERANCE = 0.20

SUCCESS = "SUCCESS"
OUTSIDE_RECORD = "OUTSIDE_RECORD"
READ_ERROR = "READ_ERROR"
EMPTY_SIGNAL = "EMPTY_SIGNAL"
SHORT_READ = "SHORT_READ"
FEATURE_ERROR = "FEATURE_ERROR"
AF_INCOMPATIBLE_HRV = "AF_INCOMPATIBLE_HRV"
INSUFFICIENT_VALID_NN = "INSUFFICIENT_VALID_NN"
INVALID_RR_RATIO = "INVALID_RR_RATIO"


def _to_float(value: Any) -> float:
    if pd.isna(value):
        return np.nan
    text = str(value).strip().strip('"').strip("'")
    if text in {"", "NA", "na", "N/A", "None", "<NA>"}:
        return np.nan
    try:
        return float(text)
    except ValueError:
        pass
    if "," in text and "." not in text:
        parts = text.split(",")
        if len(parts) == 2 and parts[1].isdigit() and len(parts[1]) <= 3:
            try:
                return float(text.replace(",", "."))
            except ValueError:
                pass
    try:
        return float(text.replace(",", ""))
    except ValueError:
        return np.nan


def _safe_float_scalar(value: Any) -> float:
    parsed = _to_float(value)
    return float(parsed) if np.isfinite(parsed) else np.nan


@lru_cache(maxsize=16)
def _filter_coefficients(fs: float, low: float, high: float, order: int) -> tuple[np.ndarray, np.ndarray] | None:
    nyq = 0.5 * float(fs)
    hi = min(float(high), nyq * 0.95)
    lo = max(float(low), 0.05)
    if lo >= hi:
        return None
    coeffs = butter(int(order), [lo / nyq, hi / nyq], btype="band", output="ba")
    return np.asarray(coeffs[0], dtype=np.float64), np.asarray(coeffs[1], dtype=np.float64)


def bandpass_filter(
    x: np.ndarray,
    fs: float,
    low: float = 0.5,
    high: float = 40.0,
    order: int = 3,
) -> np.ndarray:
    """Apply the legacy third-order zero-phase Butterworth filter."""

    values = np.asarray(x, dtype=np.float64)
    coeffs = _filter_coefficients(float(fs), float(low), float(high), int(order))
    if coeffs is None:
        return values
    try:
        return filtfilt(coeffs[0], coeffs[1], values)
    except ValueError:
        # A very short synthetic test signal cannot satisfy filtfilt's padlen.
        # The full 5-minute records are much longer; retaining the input here
        # lets the caller report a feature result rather than mutate/pad data.
        return values


def preprocess_ecg(sig_1d: np.ndarray, fs: float) -> np.ndarray:
    """Legacy median-center, bandpass, recenter, population-scale pipeline."""

    x = np.asarray(sig_1d, dtype=np.float64)
    x = x - np.nanmedian(x)
    x = bandpass_filter(x, fs=fs)
    x = x - np.nanmedian(x)
    std = np.nanstd(x)
    if std > 1e-8:
        x = x / std
    return x.astype(np.float64, copy=False)


def detect_r_peaks(ecg: np.ndarray, fs: float) -> np.ndarray:
    """Detect squared-signal peaks with the frozen percentile rule."""

    values = np.asarray(ecg, dtype=np.float64)
    if values.size == 0 or not np.isfinite(values).any():
        return np.array([], dtype=np.int64)
    enhanced = values * values
    distance = int(0.25 * float(fs))
    threshold = np.percentile(enhanced, 93)
    peaks, _ = find_peaks(enhanced, distance=max(distance, 1), height=threshold)
    return np.asarray(peaks, dtype=np.int64)


def clean_rr_intervals(
    rr_sec: Sequence[float] | np.ndarray,
    *,
    minimum_sec: float = RR_MIN_SEC,
    maximum_sec: float = RR_MAX_SEC,
    local_median_window: int = LOCAL_MEDIAN_WINDOW,
    relative_tolerance: float = LOCAL_MEDIAN_TOLERANCE,
) -> tuple[np.ndarray, np.ndarray]:
    """Return valid NN intervals and their boolean validity mask.

    Validity is the conjunction of the physiological range and a centered
    local-median deviation no greater than 20 percent.  The mask remains
    aligned with the raw ``diff(peaks) / fs`` sequence for auditable counts.
    """

    values = np.asarray(rr_sec, dtype=np.float64).reshape(-1)
    if values.size == 0:
        return values.copy(), np.zeros(0, dtype=bool)
    width = int(local_median_window)
    if width < 1 or width % 2 == 0:
        raise ValueError("local_median_window must be a positive odd integer")
    # pandas rolling median is deterministic and handles edges with the
    # explicit min_periods=1 policy used in the config.
    local = (
        pd.Series(values)
        .rolling(width, center=True, min_periods=1)
        .median()
        .to_numpy(dtype=np.float64)
    )
    physiological = np.isfinite(values) & (values >= float(minimum_sec)) & (values <= float(maximum_sec))
    usable_median = np.isfinite(local) & (local > 0)
    relative_error = np.full(values.shape, np.inf, dtype=np.float64)
    relative_error[usable_median] = np.abs(values[usable_median] - local[usable_median]) / local[usable_median]
    valid = physiological & usable_median & (relative_error <= float(relative_tolerance) + 1e-12)
    return values[valid], valid


def hrv_features(rr_sec: Sequence[float] | np.ndarray) -> Dict[str, float]:
    """Compute the six frozen HRV features from valid NN intervals only."""

    rr = np.asarray(rr_sec, dtype=np.float64).reshape(-1)
    if rr.size < 3:
        return {name: np.nan for name in REQUIRES_NN_FEATURE_NAMES[:6]}
    diff_rr = np.diff(rr)
    mean_rr = float(np.mean(rr))
    return {
        "mean_rr": mean_rr,
        "sdnn": float(np.std(rr, ddof=1)) if rr.size > 1 else np.nan,
        "rmssd": float(np.sqrt(np.mean(diff_rr**2))) if diff_rr.size else np.nan,
        "pnn50": float(np.mean(np.abs(diff_rr) > 0.05) * 100.0) if diff_rr.size else np.nan,
        "mean_hr": float(60.0 / mean_rr) if mean_rr > 1e-8 else np.nan,
        "rr_cv": float(np.std(rr) / (mean_rr + 1e-8)),
    }


def _template_view(signal: np.ndarray, dimension: int) -> np.ndarray:
    values = np.asarray(signal, dtype=np.float64)
    count = len(values) - int(dimension)
    if count <= 0:
        return np.empty((0, int(dimension)), dtype=np.float64)
    # sliding_window_view gives exactly the n-m templates used by the legacy
    # loops (the final possible start index is intentionally excluded).
    return np.lib.stride_tricks.sliding_window_view(values, int(dimension))[:count]


def sample_entropy(signal: Sequence[float] | np.ndarray, m: int = 2, r_ratio: float = 0.2) -> float:
    """Exact legacy sample entropy, evaluated with a C-level distance kernel."""

    values = np.asarray(signal, dtype=np.float64).reshape(-1)
    n = len(values)
    if n < 10:
        return np.nan
    std = np.std(values)
    if not np.isfinite(std) or std < 1e-12:
        return np.nan
    radius = float(r_ratio) * std
    counts: list[int] = []
    for dimension in (int(m), int(m) + 1):
        templates = _template_view(values, dimension)
        if len(templates) < 2:
            counts.append(0)
            continue
        distances = pdist(templates, metric="chebyshev")
        counts.append(int(np.count_nonzero(distances <= radius)))
    b, a = counts
    if b == 0 or a == 0:
        return np.nan
    return float(-np.log(a / b))


def approximate_entropy(signal: Sequence[float] | np.ndarray, m: int = 2, r_ratio: float = 0.2) -> float:
    """Exact legacy approximate entropy, evaluated with a C-level kernel."""

    values = np.asarray(signal, dtype=np.float64).reshape(-1)
    n = len(values)
    if n < 10:
        return np.nan
    std = np.std(values)
    if not np.isfinite(std) or std < 1e-12:
        return np.nan
    radius = float(r_ratio) * std

    def phi(dimension: int) -> float:
        templates = _template_view(values, dimension)
        count = len(templates)
        if count <= 0:
            return np.nan
        if count == 1:
            return 0.0
        distances = squareform(pdist(templates, metric="chebyshev"))
        c_values = np.count_nonzero(distances <= radius, axis=1) / count
        return float(np.sum(np.log(c_values + 1e-8)) / count)

    phi_m = phi(int(m))
    phi_m1 = phi(int(m) + 1)
    if not np.isfinite(phi_m) or not np.isfinite(phi_m1):
        return np.nan
    return float(phi_m - phi_m1)


def dfa(signal: Sequence[float] | np.ndarray) -> float:
    """Frozen detrended fluctuation analysis implementation."""

    values = np.asarray(signal, dtype=np.float64).reshape(-1)
    n = len(values)
    if n < 16:
        return np.nan
    values = values - np.mean(values)
    cumulative = np.cumsum(values)
    max_scale = n // 4
    if max_scale < 4:
        return np.nan
    scales = np.floor(np.logspace(np.log10(4), np.log10(max_scale), num=10)).astype(int)
    scales = np.unique(scales)
    fluctuations: list[float] = []
    used_scales: list[float] = []
    for scale in scales:
        if scale < 4:
            continue
        segments = n // int(scale)
        if segments < 2:
            continue
        t = np.arange(int(scale), dtype=np.float64)
        values_segment: list[float] = []
        for index in range(segments):
            segment = cumulative[index * scale : (index + 1) * scale]
            coeffs = np.polyfit(t, segment, 1)
            trend = np.polyval(coeffs, t)
            values_segment.append(float(np.mean((segment - trend) ** 2)))
        fluctuation = float(np.sqrt(np.mean(values_segment)))
        if np.isfinite(fluctuation) and fluctuation > 0:
            fluctuations.append(fluctuation)
            used_scales.append(float(scale))
    if len(fluctuations) < 2:
        return np.nan
    coeffs = np.polyfit(np.log(np.asarray(used_scales)), np.log(np.asarray(fluctuations)), 1)
    return float(coeffs[0])


def nonlinear_hrv_features(rr_sec: Sequence[float] | np.ndarray) -> Dict[str, float]:
    rr = np.asarray(rr_sec, dtype=np.float64).reshape(-1)
    if rr.size < 10:
        return {name: np.nan for name in REQUIRES_NN_FEATURE_NAMES[6:]}
    return {
        "rr_sampen": sample_entropy(rr, m=2, r_ratio=0.2),
        "rr_apen": approximate_entropy(rr, m=2, r_ratio=0.2),
        "rr_dfa_alpha": dfa(rr),
    }


def spectral_features(ecg: Sequence[float] | np.ndarray, fs: float) -> Dict[str, float]:
    """Return the frozen ECG FFT signal-power bands (not NN-HRV bands)."""

    values = np.asarray(ecg, dtype=np.float64).reshape(-1)
    n = values.size
    if n < 8:
        return {name: np.nan for name in FFT_FEATURE_NAMES}
    fft = np.fft.rfft(values)
    psd = (np.abs(fft) ** 2) / n
    freqs = np.fft.rfftfreq(n, d=1.0 / float(fs))
    total = float(np.sum(psd) + 1e-12)

    def band_power(low: float, high: float) -> float:
        mask = (freqs >= low) & (freqs < high)
        return float(np.sum(psd[mask]) / total)

    low = band_power(0.5, 4.0)
    mid = band_power(4.0, 15.0)
    high = band_power(15.0, 40.0)
    return {"pow_lf": low, "pow_mf": mid, "pow_hf": high, "pow_hf_ratio": float(high / (low + mid + 1e-12))}


def _signal_moments(ecg: np.ndarray) -> tuple[float, float]:
    """Pandas-compatible unbiased skew and excess kurtosis for finite input."""

    values = np.asarray(ecg, dtype=np.float64)
    n = values.size
    if n < 3:
        return np.nan, np.nan
    mean = float(np.mean(values))
    centered = values - mean
    m2 = float(np.mean(centered**2))
    if not np.isfinite(m2) or m2 <= 0:
        return np.nan, np.nan
    m3 = float(np.mean(centered**3))
    m4 = float(np.mean(centered**4))
    # These are the unbiased Fisher-Pearson conventions used by pandas
    # Series.skew() and Series.kurt() for a complete numeric series.
    # pandas uses the unbiased Fisher-Pearson coefficient based on population
    # central moments: sqrt(n*(n-1))/(n-2) * m3/m2**1.5.
    skew = (np.sqrt(n * (n - 1)) / (n - 2)) * (m3 / (m2 ** 1.5))
    if n <= 3:
        kurt = np.nan
    else:
        kurt = ((n - 1) / ((n - 2) * (n - 3))) * ((n + 1) * (m4 / (m2**2) - 3.0) + 6.0)
    return float(skew), float(kurt)


def _empty_feature_values() -> dict[str, float]:
    return {name: np.nan for name in FEATURE_NAMES}


def window_features(ecg: np.ndarray, fs: float) -> Dict[str, float]:
    """Compute the frozen 20-feature dictionary for one preprocessed window."""

    values = np.asarray(ecg, dtype=np.float64).reshape(-1)
    peaks = detect_r_peaks(values, float(fs))
    raw_rr = np.diff(peaks).astype(np.float64) / float(fs) if peaks.size >= 2 else np.array([], dtype=np.float64)
    valid_rr, _ = clean_rr_intervals(raw_rr)
    skew, kurt = _signal_moments(values)
    feats: dict[str, float] = {
        "sig_mean": float(np.mean(values)) if values.size else np.nan,
        "sig_std": float(np.std(values)) if values.size else np.nan,
        "sig_p2p": float(np.max(values) - np.min(values)) if values.size else np.nan,
        "sig_skew": skew,
        "sig_kurt": kurt,
        "beats": float(peaks.size),
        "beats_per_min": float(peaks.size * 60.0 / (values.size / float(fs) + 1e-8)) if values.size else np.nan,
    }
    feats.update(hrv_features(valid_rr))
    feats.update(nonlinear_hrv_features(valid_rr))
    feats.update(spectral_features(values, float(fs)))
    if tuple(feats) != FEATURE_NAMES:
        raise AssertionError(f"full feature order changed: {tuple(feats)}")
    return {name: float(feats[name]) for name in FEATURE_NAMES}


def complete_window_starts(
    duration_sec: float,
    *,
    first_start_sec: int = FIRST_START_SEC,
    stride_sec: int = INTERVAL_SEC,
    window_sec: int = WINDOW_SEC,
) -> tuple[int, ...]:
    """Return all complete starts ``60 + 300*k`` for a record duration."""

    duration = float(duration_sec)
    first = int(first_start_sec)
    stride = int(stride_sec)
    length = int(window_sec)
    if not np.isfinite(duration) or duration < 0:
        raise ValueError("duration_sec must be finite and non-negative")
    if first < 0 or stride <= 0 or length <= 0:
        raise ValueError("invalid full-window schedule")
    if duration + 1e-9 < first + length:
        return ()
    count = int(np.floor((duration - first - length) / stride + 1e-9)) + 1
    return tuple(first + stride * index for index in range(max(count, 0)))


def fixed_window_starts(duration_sec: float, **kwargs: Any) -> tuple[int, ...]:
    """Descriptive alias for callers shared with the legacy extractor."""

    return complete_window_starts(duration_sec, **kwargs)


def full_window_count(duration_sec: float, **kwargs: Any) -> int:
    return len(complete_window_starts(duration_sec, **kwargs))


def tail_seconds(
    duration_sec: float,
    starts: Sequence[int] | None = None,
    *,
    first_start_sec: int = FIRST_START_SEC,
    stride_sec: int = INTERVAL_SEC,
    window_sec: int = WINDOW_SEC,
) -> float:
    """Return the unwindowed record tail after the final complete segment."""

    duration = float(duration_sec)
    selected = tuple(starts) if starts is not None else complete_window_starts(
        duration,
        first_start_sec=first_start_sec,
        stride_sec=stride_sec,
        window_sec=window_sec,
    )
    if not selected:
        return float(max(duration, 0.0))
    return float(max(duration - (max(selected) + int(window_sec)), 0.0))


def _base_window_row(
    *,
    patient_id: str | None,
    window_idx: int,
    window_start_sec: int,
    fs: float,
    window_sec: int,
    channel_name: str | None,
) -> dict[str, Any]:
    start_sample = int(round(int(window_start_sec) * float(fs)))
    requested_samples = int(round(int(window_sec) * float(fs)))
    row: dict[str, Any] = {
        "patient_id": patient_id,
        "window_idx": int(window_idx),
        "window_start_sec": int(window_start_sec),
        "window_end_sec": int(window_start_sec + window_sec),
        "start_sample": start_sample,
        "requested_samples": requested_samples,
        "actual_samples": 0,
        "sampling_frequency": float(fs),
        "channel_selected": CHANNEL_INDEX,
        "channel_name": channel_name,
        "window_expected": True,
        "window_within_record": True,
        "waveform_read_success": False,
        "feature_extraction_success": False,
        "qc_valid": False,
        "window_status": READ_ERROR,
        "failure_reason": READ_ERROR,
        "qc_status": "FAIL",
        "qc_reason": READ_ERROR,
        "raw_rpeak_count": 0,
        "raw_rr_count": 0,
        "valid_rr_count": 0,
        "removed_rr_count": 0,
        "removed_rr_ratio": 0.0,
    }
    row.update(_empty_feature_values())
    row.update({name: False for name in FEATURE_VALIDITY_COLUMNS})
    return row


def _set_qc_fields(row: dict[str, Any], *, af_flag: bool = False) -> None:
    if not bool(row.get("feature_extraction_success", False)):
        row["qc_valid"] = False
        row["qc_status"] = "FAIL"
        row["qc_reason"] = str(row.get("failure_reason") or READ_ERROR)
        return
    reasons: list[str] = []
    if bool(af_flag):
        reasons.append(AF_INCOMPATIBLE_HRV)
    if int(row.get("valid_rr_count", 0)) < MIN_VALID_NN:
        reasons.append(INSUFFICIENT_VALID_NN)
    if float(row.get("removed_rr_ratio", 0.0)) > MAX_INVALID_RR_RATIO:
        reasons.append(INVALID_RR_RATIO)
    row["qc_valid"] = not reasons
    row["qc_status"] = "PASS" if not reasons else "FAIL"
    row["qc_reason"] = "" if not reasons else ";".join(reasons)


def extract_one_window(
    record_stem: str | Path,
    *,
    fs: float,
    sample_count: int | None,
    window_idx: int,
    window_start_sec: int,
    patient_id: str | None = None,
    window_sec: int = WINDOW_SEC,
    channel: int = CHANNEL_INDEX,
    channel_name: str | None = None,
) -> dict[str, Any]:
    """Read and compute one exact complete segment.

    ``patient_id`` is provenance only.  Label, endpoint, and outcome values are
    intentionally absent from this API and from all extraction calculations.
    """

    if not np.isfinite(fs) or float(fs) <= 0:
        raise ValueError("fs must be positive")
    if int(channel) != CHANNEL_INDEX:
        raise ValueError("full extraction is fixed to channel 0")
    if int(window_sec) != WINDOW_SEC:
        raise ValueError("full extraction is fixed to 300-second windows")
    row = _base_window_row(
        patient_id=patient_id,
        window_idx=window_idx,
        window_start_sec=window_start_sec,
        fs=float(fs),
        window_sec=window_sec,
        channel_name=channel_name,
    )
    start_sample = int(row["start_sample"])
    requested_samples = int(row["requested_samples"])
    stop_sample = start_sample + requested_samples
    if sample_count is not None:
        try:
            available = int(sample_count)
        except (TypeError, ValueError):
            available = -1
        if available >= 0 and (start_sample < 0 or stop_sample > available):
            row["window_within_record"] = False
            row["window_status"] = OUTSIDE_RECORD
            row["failure_reason"] = OUTSIDE_RECORD
            row["qc_reason"] = OUTSIDE_RECORD
            return row
    try:
        # The exact sampfrom/sampto and channels=[0] are part of the read
        # contract.  Never replace this with a full-record read or padding.
        record = wfdb.rdrecord(
            str(record_stem),
            sampfrom=start_sample,
            sampto=stop_sample,
            channels=[CHANNEL_INDEX],
        )
    except Exception as exc:  # retain failed theoretical rows
        row["window_status"] = READ_ERROR
        row["failure_reason"] = f"{READ_ERROR}:{type(exc).__name__}"
        row["qc_reason"] = row["failure_reason"]
        return row
    signal = getattr(record, "p_signal", None)
    if signal is None or np.asarray(signal).size == 0:
        row["window_status"] = EMPTY_SIGNAL
        row["failure_reason"] = EMPTY_SIGNAL
        row["qc_reason"] = EMPTY_SIGNAL
        return row
    array = np.asarray(signal)
    if array.ndim != 2 or array.shape[1] < 1:
        row["window_status"] = EMPTY_SIGNAL
        row["failure_reason"] = EMPTY_SIGNAL
        row["qc_reason"] = EMPTY_SIGNAL
        return row
    lead = np.asarray(array[:, 0], dtype=np.float64)
    row["actual_samples"] = int(lead.size)
    if lead.size != requested_samples:
        row["window_status"] = SHORT_READ
        row["failure_reason"] = SHORT_READ
        row["qc_reason"] = SHORT_READ
        return row
    row["waveform_read_success"] = True
    try:
        ecg = preprocess_ecg(lead, float(fs))
        values = window_features(ecg, float(fs))
        peaks = detect_r_peaks(ecg, float(fs))
        raw_rr = np.diff(peaks).astype(np.float64) / float(fs) if peaks.size >= 2 else np.array([], dtype=np.float64)
        _, valid_mask = clean_rr_intervals(raw_rr)
    except Exception as exc:
        row["window_status"] = FEATURE_ERROR
        row["failure_reason"] = f"{FEATURE_ERROR}:{type(exc).__name__}"
        row["qc_reason"] = row["failure_reason"]
        return row
    row.update(values)
    row["window_status"] = SUCCESS
    row["feature_extraction_success"] = True
    row["failure_reason"] = ""
    row["raw_rpeak_count"] = int(peaks.size)
    row["raw_rr_count"] = int(raw_rr.size)
    row["valid_rr_count"] = int(np.count_nonzero(valid_mask))
    row["removed_rr_count"] = int(raw_rr.size - np.count_nonzero(valid_mask))
    row["removed_rr_ratio"] = float(row["removed_rr_count"] / row["raw_rr_count"]) if row["raw_rr_count"] else 0.0
    for name in FEATURE_NAMES:
        row[f"{name}_valid"] = bool(np.isfinite(float(row[name])))
    _set_qc_fields(row)
    return row


def apply_af_policy(frame: pd.DataFrame, af_flag: bool) -> pd.DataFrame:
    """Apply the subject-level AF rule without changing non-HRV features."""

    result = frame.copy()
    if not bool(af_flag):
        return result
    for name in REQUIRES_NN_FEATURE_NAMES:
        result[name] = np.nan
        result[f"{name}_valid"] = False
    if "feature_extraction_success" in result.columns:
        result["qc_valid"] = False
        result["qc_status"] = "FAIL"
        result["qc_reason"] = np.where(
            result["feature_extraction_success"].fillna(False).astype(bool),
            AF_INCOMPATIBLE_HRV,
            result.get("failure_reason", AF_INCOMPATIBLE_HRV),
        )
    return result


def extract_full_windows(
    record_stem: str | Path,
    *,
    fs: float,
    sample_count: int,
    patient_id: str | None = None,
    channel_name: str | None = None,
    first_start_sec: int = FIRST_START_SEC,
    interval_sec: int = INTERVAL_SEC,
    window_sec: int = WINDOW_SEC,
) -> pd.DataFrame:
    """Extract all complete 5-minute windows for one record."""

    if sample_count is None or int(sample_count) < 0:
        raise ValueError("sample_count is required for complete-window generation")
    duration = float(sample_count) / float(fs)
    starts = complete_window_starts(
        duration,
        first_start_sec=first_start_sec,
        stride_sec=interval_sec,
        window_sec=window_sec,
    )
    rows = [
        extract_one_window(
            record_stem,
            fs=float(fs),
            sample_count=int(sample_count),
            window_idx=index,
            window_start_sec=start,
            patient_id=patient_id,
            window_sec=window_sec,
            channel_name=channel_name,
        )
        for index, start in enumerate(starts)
    ]
    columns = [
        "patient_id", "window_idx", "window_start_sec", "window_end_sec", "start_sample", "requested_samples",
        "actual_samples", "sampling_frequency", "channel_selected", "channel_name", "window_expected",
        "window_within_record", "waveform_read_success", "feature_extraction_success", "qc_valid", "window_status",
        "failure_reason", "qc_status", "qc_reason", "raw_rpeak_count", "raw_rr_count", "valid_rr_count",
        "removed_rr_count", "removed_rr_ratio", *FEATURE_NAMES, *FEATURE_VALIDITY_COLUMNS,
    ]
    result = pd.DataFrame(rows, columns=columns)
    if result.empty:
        return result
    for name in FEATURE_NAMES:
        result[name] = pd.to_numeric(result[name], errors="coerce").astype("float64")
    for name in FEATURE_VALIDITY_COLUMNS:
        result[name] = result[name].fillna(False).astype(bool)
    for name in ("window_expected", "window_within_record", "waveform_read_success", "feature_extraction_success", "qc_valid"):
        result[name] = result[name].fillna(False).astype(bool)
    return result


def feature_names() -> list[str]:
    return list(FEATURE_NAMES)


__all__ = [
    "FEATURE_NAMES", "FEATURE_IDS", "FEATURE_VALIDITY_COLUMNS", "FEATURE_VALIDITY_IDS",
    "REQUIRES_NN_FEATURE_IDS", "REQUIRES_NN_FEATURE_NAMES", "SIGNAL_QUALITY_FEATURE_NAMES",
    "FFT_FEATURE_NAMES", "WINDOW_SEC", "FIRST_START_SEC", "INTERVAL_SEC", "CHANNEL_INDEX",
    "MIN_VALID_NN", "MAX_INVALID_RR_RATIO", "RR_MIN_SEC", "RR_MAX_SEC", "LOCAL_MEDIAN_WINDOW",
    "LOCAL_MEDIAN_TOLERANCE", "SUCCESS", "OUTSIDE_RECORD", "READ_ERROR", "EMPTY_SIGNAL",
    "SHORT_READ", "FEATURE_ERROR", "AF_INCOMPATIBLE_HRV", "INSUFFICIENT_VALID_NN", "INVALID_RR_RATIO",
    "bandpass_filter", "preprocess_ecg", "detect_r_peaks", "clean_rr_intervals", "hrv_features",
    "sample_entropy", "approximate_entropy", "dfa", "nonlinear_hrv_features", "spectral_features",
    "window_features", "complete_window_starts", "fixed_window_starts", "full_window_count", "tail_seconds",
    "extract_one_window", "extract_full_windows", "apply_af_policy", "feature_names",
]
