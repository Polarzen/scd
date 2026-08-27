"""Phase 3 fixed-window implementation of the legacy ECG features.

The numerical feature functions in this module intentionally mirror the
implementation in :mod:`心电图分析.py`.  Phase 3 changes only the extraction
schedule: each Holter record is queried at the fixed starts
``60 + 3600 * k`` seconds for ``k=0..23``.  A theoretical window is retained
even when it cannot be read; failed windows contain null feature values and a
machine-readable status.

This module does not cache waveform data.  The only waveform operation is the
bounded ``wfdb.rdrecord`` call made by :func:`extract_fixed_windows`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import wfdb
from scipy.signal import butter, filtfilt, find_peaks


# The order is the frozen order in config/legacy_features.yaml and is also the
# order used by the old ``window_features`` dictionary.
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

WINDOW_SEC = 120
FIRST_START_SEC = 60
INTERVAL_SEC = 3600
MAX_WINDOWS = 24
SUCCESS = "SUCCESS"
OUTSIDE_RECORD = "OUTSIDE_RECORD"
READ_ERROR = "READ_ERROR"
EMPTY_SIGNAL = "EMPTY_SIGNAL"
SHORT_READ = "SHORT_READ"
FEATURE_ERROR = "FEATURE_ERROR"


def _to_float(value: Any) -> float:
    """Convert the scalar formats used by the old script to a float."""

    if pd.isna(value):
        return np.nan
    text = str(value).strip().strip('"').strip("'")
    if text in {"", "NA", "na", "N/A", "None"}:
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
    value_float = _to_float(value)
    return float(value_float) if np.isfinite(value_float) else np.nan


def bandpass_filter(
    x: np.ndarray,
    fs: float,
    low: float = 0.5,
    high: float = 40.0,
    order: int = 3,
) -> np.ndarray:
    """Apply the exact old third-order zero-phase Butterworth filter."""

    nyq = 0.5 * fs
    hi = min(high, nyq * 0.95)
    lo = max(low, 0.05)
    if lo >= hi:
        return x
    coeffs = butter(order, [lo / nyq, hi / nyq], btype="band", output="ba")
    if coeffs is None or len(coeffs) < 2:
        return x
    b = np.asarray(coeffs[0], dtype=np.float64)
    a = np.asarray(coeffs[1], dtype=np.float64)
    return filtfilt(b, a, x)


def preprocess_ecg(sig_1d: np.ndarray, fs: float) -> np.ndarray:
    """Legacy median-center, bandpass, recenter, population-scale pipeline."""

    x = sig_1d.astype(np.float64)
    x = x - np.nanmedian(x)
    x = bandpass_filter(x, fs=fs)
    x = x - np.nanmedian(x)
    std = np.nanstd(x)
    if std > 1e-8:
        x = x / std
    return x


def detect_r_peaks(ecg: np.ndarray, fs: float) -> np.ndarray:
    """Detect squared-signal peaks with the old fixed percentile rule."""

    enh = ecg * ecg
    distance = int(0.25 * fs)
    threshold = np.percentile(enh, 93)
    peaks, _ = find_peaks(enh, distance=max(distance, 1), height=threshold)
    return peaks


def hrv_features(rr_sec: np.ndarray) -> Dict[str, float]:
    """Return the old six raw-RR HRV features without RR cleaning."""

    if rr_sec.size < 3:
        return {
            "mean_rr": np.nan,
            "sdnn": np.nan,
            "rmssd": np.nan,
            "pnn50": np.nan,
            "mean_hr": np.nan,
            "rr_cv": np.nan,
        }

    diff_rr = np.diff(rr_sec)
    mean_rr = float(np.mean(rr_sec))
    sdnn = float(np.std(rr_sec, ddof=1)) if rr_sec.size > 1 else np.nan
    rmssd = float(np.sqrt(np.mean(diff_rr**2))) if diff_rr.size > 0 else np.nan
    pnn50 = float(np.mean(np.abs(diff_rr) > 0.05) * 100.0) if diff_rr.size > 0 else np.nan
    mean_hr = float(60.0 / mean_rr) if mean_rr > 1e-8 else np.nan
    rr_cv = float(np.std(rr_sec) / (mean_rr + 1e-8))

    return {
        "mean_rr": mean_rr,
        "sdnn": sdnn,
        "rmssd": rmssd,
        "pnn50": pnn50,
        "mean_hr": mean_hr,
        "rr_cv": rr_cv,
    }


def sample_entropy(signal: np.ndarray, m: int = 2, r_ratio: float = 0.2) -> float:
    """Exact legacy sample entropy implementation."""

    x = np.asarray(signal, dtype=np.float64)
    n = len(x)
    if n < 10:
        return np.nan

    std = np.std(x)
    if not np.isfinite(std) or std < 1e-12:
        return np.nan

    r = r_ratio * std

    def _phi(mm: int) -> int:
        count = 0
        end = n - mm
        for i in range(end):
            xi = x[i : i + mm]
            for j in range(i + 1, end):
                if np.all(np.abs(xi - x[j : j + mm]) <= r):
                    count += 1
        return count

    b = _phi(m)
    a = _phi(m + 1)

    if b == 0 or a == 0:
        return np.nan
    return float(-np.log(a / b))


def approximate_entropy(signal: np.ndarray, m: int = 2, r_ratio: float = 0.2) -> float:
    """Exact legacy approximate entropy implementation."""

    x = np.asarray(signal, dtype=np.float64)
    n = len(x)
    if n < 10:
        return np.nan

    std = np.std(x)
    if not np.isfinite(std) or std < 1e-12:
        return np.nan

    r = r_ratio * std

    def _phi(mm: int) -> float:
        c_vals = []
        end = n - mm
        if end <= 0:
            return np.nan
        for i in range(end):
            xi = x[i : i + mm]
            count = 0
            for j in range(end):
                if np.all(np.abs(xi - x[j : j + mm]) <= r):
                    count += 1
            c_vals.append(count / end)

        c_arr = np.asarray(c_vals, dtype=np.float64)
        return float(np.sum(np.log(c_arr + 1e-8)) / end)

    phi_m = _phi(m)
    phi_m1 = _phi(m + 1)
    if not np.isfinite(phi_m) or not np.isfinite(phi_m1):
        return np.nan
    return float(phi_m - phi_m1)


def dfa(signal: np.ndarray) -> float:
    """Exact legacy detrended fluctuation analysis implementation."""

    x = np.asarray(signal, dtype=np.float64)
    n = len(x)
    if n < 16:
        return np.nan

    x = x - np.mean(x)
    y = np.cumsum(x)

    max_scale = n // 4
    if max_scale < 4:
        return np.nan

    scales = np.floor(np.logspace(np.log10(4), np.log10(max_scale), num=10)).astype(int)
    scales = np.unique(scales)
    flucts = []
    used_scales = []

    for s in scales:
        if s < 4:
            continue

        n_segments = n // s
        if n_segments < 2:
            continue

        f_nu = []
        t = np.arange(s, dtype=np.float64)
        for i in range(n_segments):
            seg = y[i * s : (i + 1) * s]
            coeffs = np.polyfit(t, seg, 1)
            trend = np.polyval(coeffs, t)
            f_nu.append(np.mean((seg - trend) ** 2))

        fn = np.sqrt(np.mean(f_nu))
        if np.isfinite(fn) and fn > 0:
            flucts.append(float(fn))
            used_scales.append(float(s))

    if len(flucts) < 2:
        return np.nan

    coeffs = np.polyfit(np.log(np.asarray(used_scales)), np.log(np.asarray(flucts)), 1)
    return float(coeffs[0])


def nonlinear_hrv_features(rr_sec: np.ndarray) -> Dict[str, float]:
    """Return the old nonlinear raw-RR features."""

    if rr_sec.size < 10:
        return {
            "rr_sampen": np.nan,
            "rr_apen": np.nan,
            "rr_dfa_alpha": np.nan,
        }

    return {
        "rr_sampen": sample_entropy(rr_sec, m=2, r_ratio=0.2),
        "rr_apen": approximate_entropy(rr_sec, m=2, r_ratio=0.2),
        "rr_dfa_alpha": dfa(rr_sec),
    }


def spectral_features(ecg: np.ndarray, fs: float) -> Dict[str, float]:
    """Return raw FFT signal-power bands used by the old implementation."""

    n = ecg.size
    if n < 8:
        return {"pow_lf": np.nan, "pow_mf": np.nan, "pow_hf": np.nan, "pow_hf_ratio": np.nan}

    fft = np.fft.rfft(ecg)
    psd = (np.abs(fft) ** 2) / n
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)

    total = float(np.sum(psd) + 1e-12)

    def band_power(lo: float, hi: float) -> float:
        mask = (freqs >= lo) & (freqs < hi)
        return float(np.sum(psd[mask]) / total)

    pow_lf = band_power(0.5, 4.0)
    pow_mf = band_power(4.0, 15.0)
    pow_hf = band_power(15.0, 40.0)
    return {
        "pow_lf": pow_lf,
        "pow_mf": pow_mf,
        "pow_hf": pow_hf,
        "pow_hf_ratio": float(pow_hf / (pow_lf + pow_mf + 1e-12)),
    }


def window_features(ecg: np.ndarray, fs: float) -> Dict[str, float]:
    """Compute the frozen 20-feature dictionary for one preprocessed window."""

    peaks = detect_r_peaks(ecg, fs)
    rr = np.diff(peaks) / fs if peaks.size >= 2 else np.array([])

    feats = {
        "sig_mean": float(np.mean(ecg)),
        "sig_std": float(np.std(ecg)),
        "sig_p2p": float(np.max(ecg) - np.min(ecg)),
        "sig_skew": _safe_float_scalar(pd.Series(ecg).skew()),
        "sig_kurt": _safe_float_scalar(pd.Series(ecg).kurt()),
        "beats": float(peaks.size),
        "beats_per_min": float(peaks.size * 60.0 / (ecg.size / fs + 1e-8)),
    }
    feats.update(hrv_features(rr))
    feats.update(nonlinear_hrv_features(rr))
    feats.update(spectral_features(ecg, fs))
    # Fail loudly if a future edit changes the public order or count.
    if tuple(feats) != FEATURE_NAMES:
        raise AssertionError(f"legacy feature order changed: {tuple(feats)}")
    return {name: float(feats[name]) for name in FEATURE_NAMES}


def fixed_window_starts(
    *,
    first_start_sec: int = FIRST_START_SEC,
    interval_sec: int = INTERVAL_SEC,
    max_windows: int = MAX_WINDOWS,
) -> tuple[int, ...]:
    """Return the theoretical fixed-window starts in seconds."""

    if first_start_sec < 0 or interval_sec <= 0 or max_windows < 0:
        raise ValueError("invalid fixed-window schedule")
    return tuple(int(first_start_sec + interval_sec * k) for k in range(int(max_windows)))


def _base_window_row(
    *,
    patient_id: str | None,
    window_idx: int,
    window_start_sec: int,
    fs: float,
    window_sec: int,
) -> dict[str, Any]:
    start_sample = int(window_start_sec * fs)
    requested_samples = int(window_sec * fs)
    row: dict[str, Any] = {
        "patient_id": patient_id,
        "window_idx": int(window_idx),
        "window_start_sec": int(window_start_sec),
        "window_end_sec": int(window_start_sec + window_sec),
        "start_sample": start_sample,
        "requested_samples": requested_samples,
        "actual_samples": 0,
        "window_status": READ_ERROR,
        "failure_reason": READ_ERROR,
        "raw_rpeak_count": 0,
        "raw_rr_count": 0,
        "valid_rr_count": 0,
        "removed_rr_count": 0,
        "removed_rr_ratio": 0.0,
    }
    row.update({name: np.nan for name in FEATURE_NAMES})
    return row


def extract_one_window(
    record_stem: str | Path,
    *,
    fs: float,
    sample_count: int | None,
    window_idx: int,
    window_start_sec: int,
    patient_id: str | None = None,
    window_sec: int = WINDOW_SEC,
    channel: int = 0,
) -> dict[str, Any]:
    """Read and compute one bounded theoretical window.

    ``sample_count`` is taken from the frozen Phase 2 header metadata.  An
    out-of-range theoretical window is represented without issuing a partial
    read, which ensures that no padding or >120-second read can occur.
    """

    if not np.isfinite(fs) or float(fs) <= 0:
        raise ValueError("fs must be positive")
    if int(channel) != 0:
        raise ValueError("Phase 3 legacy extraction is fixed to channel 0")
    if int(window_sec) <= 0 or int(window_sec) > WINDOW_SEC:
        raise ValueError("window_sec must be in (0, 120]")

    row = _base_window_row(
        patient_id=patient_id,
        window_idx=window_idx,
        window_start_sec=window_start_sec,
        fs=float(fs),
        window_sec=int(window_sec),
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
            row["window_status"] = OUTSIDE_RECORD
            row["failure_reason"] = OUTSIDE_RECORD
            return row

    try:
        # The explicit channels=[0] and exact 120-second sampto are part of
        # the Phase 3 read contract.  Do not replace with rdheader/all leads.
        rec = wfdb.rdrecord(
            str(record_stem),
            sampfrom=start_sample,
            sampto=stop_sample,
            channels=[0],
        )
    except Exception as exc:  # waveform read failures are retained as rows
        row["window_status"] = READ_ERROR
        row["failure_reason"] = f"{READ_ERROR}:{type(exc).__name__}"
        return row

    p_signal = getattr(rec, "p_signal", None)
    if p_signal is None or np.asarray(p_signal).size == 0:
        row["window_status"] = EMPTY_SIGNAL
        row["failure_reason"] = EMPTY_SIGNAL
        return row
    signal = np.asarray(p_signal)
    if signal.ndim != 2 or signal.shape[1] < 1:
        row["window_status"] = EMPTY_SIGNAL
        row["failure_reason"] = EMPTY_SIGNAL
        return row
    lead = np.asarray(signal[:, 0], dtype=np.float64)
    row["actual_samples"] = int(lead.size)
    if lead.size != requested_samples:
        row["window_status"] = SHORT_READ
        row["failure_reason"] = SHORT_READ
        return row

    try:
        ecg = preprocess_ecg(lead, float(fs))
        values = window_features(ecg, float(fs))
    except Exception as exc:
        row["window_status"] = FEATURE_ERROR
        row["failure_reason"] = f"{FEATURE_ERROR}:{type(exc).__name__}"
        return row

    row.update(values)
    beat_count = values.get("beats", np.nan)
    row["raw_rpeak_count"] = int(float(beat_count)) if np.isfinite(beat_count) else 0
    rr_count = int(max(float(beat_count) - 1.0, 0.0)) if np.isfinite(beat_count) else 0
    # Legacy RR QC has no rejection/cleaning: raw RR == valid RR.
    row["raw_rr_count"] = rr_count
    row["valid_rr_count"] = rr_count
    row["removed_rr_count"] = 0
    row["removed_rr_ratio"] = 0.0
    row["window_status"] = SUCCESS
    row["failure_reason"] = ""
    return row


def extract_fixed_windows(
    record_stem: str | Path,
    *,
    fs: float,
    sample_count: int | None,
    patient_id: str | None = None,
    first_start_sec: int = FIRST_START_SEC,
    interval_sec: int = INTERVAL_SEC,
    max_windows: int = MAX_WINDOWS,
    window_sec: int = WINDOW_SEC,
) -> pd.DataFrame:
    """Extract all theoretical fixed windows, retaining failures."""

    rows = [
        extract_one_window(
            record_stem,
            fs=float(fs),
            sample_count=sample_count,
            window_idx=k,
            window_start_sec=start,
            patient_id=patient_id,
            window_sec=window_sec,
        )
        for k, start in enumerate(
            fixed_window_starts(
                first_start_sec=first_start_sec,
                interval_sec=interval_sec,
                max_windows=max_windows,
            )
        )
    ]
    frame = pd.DataFrame(rows)
    for name in FEATURE_NAMES:
        frame[name] = pd.to_numeric(frame[name], errors="coerce").astype("float64")
    return frame


def feature_names() -> list[str]:
    """Return a mutable list of the frozen feature names."""

    return list(FEATURE_NAMES)


__all__ = [
    "FEATURE_NAMES",
    "WINDOW_SEC",
    "FIRST_START_SEC",
    "INTERVAL_SEC",
    "MAX_WINDOWS",
    "SUCCESS",
    "OUTSIDE_RECORD",
    "READ_ERROR",
    "EMPTY_SIGNAL",
    "SHORT_READ",
    "FEATURE_ERROR",
    "bandpass_filter",
    "preprocess_ecg",
    "detect_r_peaks",
    "hrv_features",
    "sample_entropy",
    "approximate_entropy",
    "dfa",
    "nonlinear_hrv_features",
    "spectral_features",
    "window_features",
    "fixed_window_starts",
    "extract_one_window",
    "extract_fixed_windows",
    "feature_names",
]
