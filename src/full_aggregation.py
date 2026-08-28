"""Deterministic patient aggregation and PVC burden derivation for Phase 4."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from .full_features import (
    FEATURE_NAMES,
    FEATURE_VALIDITY_COLUMNS,
    SIGNAL_QUALITY_FEATURE_NAMES,
    SUCCESS,
)


AGGREGATION_SUFFIXES: tuple[str, ...] = ("mean", "std", "p10", "p50", "p90")
AGGREGATED_FEATURE_NAMES: tuple[str, ...] = tuple(
    f"{feature}_{suffix}" for feature in FEATURE_NAMES for suffix in AGGREGATION_SUFFIXES
)
VALID_COUNT_NAMES: tuple[str, ...] = tuple(f"{feature}_valid_count" for feature in FEATURE_NAMES)
PHYSIOLOGY_AGGREGATED_FEATURE_NAMES: tuple[str, ...] = tuple(
    f"{feature}_{suffix}"
    for feature in FEATURE_NAMES
    if feature not in SIGNAL_QUALITY_FEATURE_NAMES
    for suffix in AGGREGATION_SUFFIXES
)
PHYSIOLOGY_VALID_COUNT_NAMES: tuple[str, ...] = tuple(
    f"{feature}_valid_count" for feature in FEATURE_NAMES if feature not in SIGNAL_QUALITY_FEATURE_NAMES
)

METADATA_COLUMNS: tuple[str, ...] = (
    "patient_id",
    "label",
    "followup_days",
    "cause_of_death",
    "fs",
    "record_id",
    "n_windows_theoretical",
    "n_windows_successful",
    "n_windows_qc_valid",
    "n_windows_used",
    "window_success_rate",
    "raw_rpeak_count_total",
    "raw_rr_count_total",
    "valid_rr_count_total",
    "removed_rr_count_total",
    "tail_seconds",
    "pvc_count_24h",
    "pvc_denominator_beats",
    "pvc_burden",
    "high_pvc_burden",
)


def aggregated_feature_names(*, physiology_only: bool = False) -> list[str]:
    return list(PHYSIOLOGY_AGGREGATED_FEATURE_NAMES if physiology_only else AGGREGATED_FEATURE_NAMES)


def valid_count_names(*, physiology_only: bool = False) -> list[str]:
    return list(PHYSIOLOGY_VALID_COUNT_NAMES if physiology_only else VALID_COUNT_NAMES)


def _stats(values: Sequence[Any] | pd.Series | np.ndarray) -> dict[str, float]:
    x = pd.to_numeric(pd.Series(values), errors="coerce").to_numpy(dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return {suffix: np.nan for suffix in AGGREGATION_SUFFIXES}
    return {
        "mean": float(np.mean(x)),
        "std": float(np.std(x, ddof=1)) if x.size > 1 else 0.0,
        "p10": float(np.percentile(x, 10)),
        "p50": float(np.percentile(x, 50)),
        "p90": float(np.percentile(x, 90)),
    }


def _feature_valid_mask(frame: pd.DataFrame, feature: str) -> pd.Series:
    finite = pd.to_numeric(frame.get(feature, pd.Series(np.nan, index=frame.index)), errors="coerce").notna()
    finite &= np.isfinite(pd.to_numeric(frame.get(feature, pd.Series(np.nan, index=frame.index)), errors="coerce"))
    bit = f"{feature}_valid"
    if bit in frame.columns:
        finite &= frame[bit].fillna(False).astype(bool)
    return finite


def _successful_mask(frame: pd.DataFrame) -> pd.Series:
    if "feature_extraction_success" in frame.columns:
        return frame["feature_extraction_success"].fillna(False).astype(bool)
    if "window_status" in frame.columns:
        return frame["window_status"].astype("string").str.upper().eq(SUCCESS)
    if "waveform_read_success" in frame.columns:
        return frame["waveform_read_success"].fillna(False).astype(bool)
    return pd.Series(True, index=frame.index)


def _qc_mask(frame: pd.DataFrame) -> pd.Series:
    if "qc_valid" in frame.columns:
        return frame["qc_valid"].fillna(False).astype(bool)
    return _successful_mask(frame)


def _first_value(group: pd.DataFrame, column: str, default: Any = np.nan) -> Any:
    if column not in group.columns:
        return default
    values = group[column].dropna()
    return values.iloc[0] if not values.empty else default


def derive_pvc_burden(
    pvc_count_24h: Any,
    windows: pd.DataFrame,
    *,
    threshold: float = 0.20,
) -> dict[str, Any]:
    """Derive the configured high-PVC flag after extraction.

    The denominator is the sum of detected R peaks in all successfully read
    covered windows.  It is intentionally not a window-generation input and is
    reported alongside the burden so an unavailable denominator cannot be
    mistaken for a measured zero.
    """

    if not np.isfinite(float(threshold)) or float(threshold) < 0:
        raise ValueError("threshold must be finite and non-negative")
    try:
        pvc = float(pvc_count_24h)
    except (TypeError, ValueError):
        pvc = np.nan
    if not np.isfinite(pvc):
        return {
            "pvc_count_24h": np.nan,
            "pvc_denominator_beats": np.nan,
            "pvc_burden": np.nan,
            "high_pvc_burden": False,
            "pvc_burden_status": "UNAVAILABLE_OFFICIAL_COUNT",
        }
    if windows is None or windows.empty:
        return {
            "pvc_count_24h": pvc,
            "pvc_denominator_beats": np.nan,
            "pvc_burden": np.nan,
            "high_pvc_burden": False,
            "pvc_burden_status": "UNAVAILABLE_NO_COVERAGE",
        }
    if "waveform_read_success" in windows.columns:
        covered = windows["waveform_read_success"].fillna(False).astype(bool)
    else:
        covered = _successful_mask(windows)
    beats = pd.to_numeric(windows.loc[covered, "raw_rpeak_count"], errors="coerce") if "raw_rpeak_count" in windows.columns else pd.Series(dtype=float)
    denominator = float(beats[ np.isfinite(beats.to_numpy(dtype=np.float64)) ].sum()) if len(beats) else 0.0
    if denominator <= 0:
        return {
            "pvc_count_24h": pvc,
            "pvc_denominator_beats": denominator,
            "pvc_burden": np.nan,
            "high_pvc_burden": False,
            "pvc_burden_status": "UNAVAILABLE_NO_DETECTED_BEATS",
        }
    burden = float(pvc / denominator)
    return {
        "pvc_count_24h": pvc,
        "pvc_denominator_beats": denominator,
        "pvc_burden": burden,
        "high_pvc_burden": bool(burden > float(threshold)),
        "pvc_burden_status": "AVAILABLE",
    }


def aggregate_dynamic_features(
    df_win: pd.DataFrame,
    *,
    feature_names: Sequence[str] = FEATURE_NAMES,
    use_qc_valid: bool = False,
) -> dict[str, float]:
    """Aggregate a window frame using finite, per-feature-valid values."""

    base_mask = _qc_mask(df_win) if use_qc_valid else _successful_mask(df_win)
    out: dict[str, float] = {}
    for feature in feature_names:
        values = df_win.loc[base_mask & _feature_valid_mask(df_win, feature), feature] if feature in df_win else pd.Series(dtype=float)
        stats = _stats(values)
        for suffix in AGGREGATION_SUFFIXES:
            out[f"{feature}_{suffix}"] = stats[suffix]
        out[f"{feature}_valid_count"] = int(np.isfinite(pd.to_numeric(values, errors="coerce")).sum())
    out["n_windows_used"] = int(base_mask.sum())
    return out


def _subject_ids(windows: pd.DataFrame, subjects: pd.DataFrame | None) -> list[str]:
    ids: set[str] = set()
    if windows is not None and "patient_id" in windows.columns:
        ids.update(str(value) for value in windows["patient_id"].dropna())
    if subjects is not None and "patient_id" in subjects.columns:
        ids.update(str(value) for value in subjects["patient_id"].dropna())
    return sorted(ids)


def aggregate_patient_features(
    windows: pd.DataFrame,
    *,
    subjects: pd.DataFrame | None = None,
    subject_metadata: pd.DataFrame | None = None,
    include_ineligible: bool = True,
    min_qc_valid_windows: int = 1,
    physiology_only: bool = False,
    use_qc_valid: bool = False,
    pvc_threshold: float = 0.20,
) -> pd.DataFrame:
    """Return one deterministic row per patient with 100 feature statistics.

    ``subjects`` is optional for small unit fixtures.  The full build supplies
    all 992 official subjects so patients without Holter data remain explicit.
    Aggregated statistics use sample SD and finite values only; validity bits
    are applied independently for every feature.
    """

    if windows is None:
        windows = pd.DataFrame(columns=["patient_id", *FEATURE_NAMES])
    if "patient_id" not in windows.columns and (subjects is None or "patient_id" not in subjects.columns):
        raise ValueError("windows or subjects must contain patient_id")
    if int(min_qc_valid_windows) < 0:
        raise ValueError("min_qc_valid_windows must be non-negative")
    if subjects is None and subject_metadata is not None:
        subjects = subject_metadata
    names = [feature for feature in FEATURE_NAMES if not physiology_only or feature not in SIGNAL_QUALITY_FEATURE_NAMES]
    aggregate_names = [f"{feature}_{suffix}" for feature in names for suffix in AGGREGATION_SUFFIXES]
    count_names = [f"{feature}_valid_count" for feature in names]
    work = windows.copy()
    if "patient_id" in work.columns:
        work["patient_id"] = work["patient_id"].astype("string")
    subject_work = subjects.copy() if subjects is not None else None
    if subject_work is not None:
        subject_work["patient_id"] = subject_work["patient_id"].astype("string")
    rows: list[dict[str, Any]] = []
    for patient_id in _subject_ids(work, subject_work):
        group = work.loc[work["patient_id"].astype("string").eq(patient_id)].copy() if "patient_id" in work.columns else work.iloc[0:0].copy()
        source = subject_work.loc[subject_work["patient_id"].eq(patient_id)] if subject_work is not None else pd.DataFrame()
        if len(source) > 1:
            raise ValueError(f"duplicate subject metadata for {patient_id}")
        source_row = source.iloc[0] if len(source) else None
        n_theoretical = int(len(group))
        successful = _successful_mask(group)
        qc = _qc_mask(group)
        n_successful = int(successful.sum())
        n_qc = int(qc.sum())
        if n_qc < int(min_qc_valid_windows) and not include_ineligible:
            continue
        row: dict[str, Any] = {"patient_id": patient_id}
        for column in ("label", "followup_days", "cause_of_death", "fs", "record_id", "tail_seconds"):
            value = source_row[column] if source_row is not None and column in source_row.index else _first_value(group, column)
            row[column] = value
        stats_mask = qc if use_qc_valid else successful
        for feature in names:
            mask = stats_mask & _feature_valid_mask(group, feature)
            values = group.loc[mask, feature] if feature in group.columns else pd.Series(dtype=float)
            stats = _stats(values)
            for suffix in AGGREGATION_SUFFIXES:
                row[f"{feature}_{suffix}"] = stats[suffix]
            row[f"{feature}_valid_count"] = int(len(values))
        # If physiology_only was requested, dropped categories are intentionally
        # absent.  The default output always contains all 20*5 values.
        row["n_windows_theoretical"] = n_theoretical
        row["n_windows_successful"] = n_successful
        row["n_windows_qc_valid"] = n_qc
        row["n_windows_used"] = int(stats_mask.sum())
        row["window_success_rate"] = float(n_successful / n_theoretical) if n_theoretical else 0.0
        for source_name, target_name in (
            ("raw_rpeak_count", "raw_rpeak_count_total"),
            ("raw_rr_count", "raw_rr_count_total"),
            ("valid_rr_count", "valid_rr_count_total"),
            ("removed_rr_count", "removed_rr_count_total"),
        ):
            if source_name in group.columns:
                values = pd.to_numeric(group.loc[successful, source_name], errors="coerce")
                row[target_name] = float(values.fillna(0).sum())
            else:
                row[target_name] = 0.0
        if pd.isna(row.get("tail_seconds")):
            row["tail_seconds"] = _first_value(group, "tail_seconds", 0.0)
        pvc_value = source_row["pvc_count_24h"] if source_row is not None and "pvc_count_24h" in source_row.index else _first_value(group, "pvc_count_24h")
        pvc = derive_pvc_burden(pvc_value, group, threshold=pvc_threshold)
        row.update(pvc)
        rows.append(row)
    columns = ["patient_id", "label", "followup_days", "cause_of_death", "fs", "record_id"]
    columns += aggregate_names + count_names
    columns += [c for c in METADATA_COLUMNS if c not in columns]
    result = pd.DataFrame(rows)
    if result.empty:
        return pd.DataFrame(columns=columns)
    for name in aggregate_names:
        result[name] = pd.to_numeric(result[name], errors="coerce").astype("float64")
    for name in count_names:
        result[name] = pd.to_numeric(result[name], errors="coerce").fillna(0).astype("Int64")
    result["patient_id"] = result["patient_id"].astype("string")
    result = result.reindex(columns=columns)
    return result.sort_values("patient_id", kind="stable").reset_index(drop=True)


aggregate_full_windows = aggregate_patient_features
aggregate_patient = aggregate_patient_features


__all__ = [
    "AGGREGATION_SUFFIXES", "AGGREGATED_FEATURE_NAMES", "VALID_COUNT_NAMES",
    "PHYSIOLOGY_AGGREGATED_FEATURE_NAMES", "PHYSIOLOGY_VALID_COUNT_NAMES", "METADATA_COLUMNS",
    "aggregated_feature_names", "valid_count_names", "derive_pvc_burden", "aggregate_dynamic_features",
    "aggregate_patient_features", "aggregate_full_windows", "aggregate_patient",
]
