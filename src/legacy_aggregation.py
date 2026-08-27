"""Patient-level aggregation for the Phase 3 legacy fixed-window table."""

from __future__ import annotations

from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from .legacy_features import FEATURE_NAMES, SUCCESS


AGGREGATION_SUFFIXES: tuple[str, ...] = ("mean", "std", "p10", "p50", "p90")
AGGREGATED_FEATURE_NAMES: tuple[str, ...] = tuple(
    f"{feature}_{suffix}" for feature in FEATURE_NAMES for suffix in AGGREGATION_SUFFIXES
)
METADATA_COLUMNS: tuple[str, ...] = (
    "patient_id",
    "label",
    "followup_days",
    "cause_of_death",
    "fs",
    "n_windows_theoretical",
    "n_windows_successful",
    "n_windows_used",
    "window_success_rate",
    "raw_rr_count_total",
    "valid_rr_count_total",
    "removed_rr_count_total",
)


def aggregated_feature_names() -> list[str]:
    """Return the exact 100 model feature columns in frozen order."""

    return list(AGGREGATED_FEATURE_NAMES)


def _stats(values: pd.Series | np.ndarray | Sequence[Any]) -> dict[str, float]:
    """Compute the old mean/sample-SD/percentile semantics."""

    x = pd.to_numeric(pd.Series(values), errors="coerce").dropna().to_numpy(dtype=np.float64)
    if x.size == 0:
        return {suffix: np.nan for suffix in AGGREGATION_SUFFIXES}
    return {
        "mean": float(np.mean(x)),
        "std": float(np.std(x, ddof=1)) if x.size > 1 else 0.0,
        "p10": float(np.percentile(x, 10)),
        "p50": float(np.percentile(x, 50)),
        "p90": float(np.percentile(x, 90)),
    }


def aggregate_dynamic_features(
    df_win: pd.DataFrame,
    *,
    feature_names: Sequence[str] = FEATURE_NAMES,
) -> dict[str, float]:
    """Aggregate one patient's successful window rows with legacy semantics.

    This is intentionally compatible with the old ``aggregate_dynamic_features``
    helper: NaNs are omitted independently per feature, a one-value sample SD is
    zero, and ``n_windows_used`` counts rows supplied by the caller.
    """

    names = list(feature_names)
    out: dict[str, float] = {}
    for name in names:
        stats = _stats(df_win[name] if name in df_win.columns else pd.Series(dtype=float))
        for suffix in AGGREGATION_SUFFIXES:
            out[f"{name}_{suffix}"] = stats[suffix]
    out["n_windows_used"] = float(len(df_win))
    return out


def _successful_mask(df: pd.DataFrame) -> pd.Series:
    if "window_status" not in df.columns:
        return pd.Series(True, index=df.index)
    status = df["window_status"].astype("string").str.upper()
    return status.eq(SUCCESS)


def _first_value(group: pd.DataFrame, column: str, default: Any = np.nan) -> Any:
    if column not in group.columns:
        return default
    values = group[column].dropna()
    return values.iloc[0] if not values.empty else default


def aggregate_patient_features(
    windows: pd.DataFrame,
    *,
    min_successful_windows: int = 4,
    include_ineligible: bool = False,
    feature_names: Sequence[str] = FEATURE_NAMES,
) -> pd.DataFrame:
    """Aggregate fixed-window rows into one row per eligible patient.

    All theoretical windows remain in ``windows``.  The minimum-window gate is
    applied only to the patient-level modeling table, matching the old behavior
    that discarded patients with fewer than four successfully read windows.
    Set ``include_ineligible=True`` when an audit table must preserve such
    patient rows; their 100 feature columns are null when no usable rows exist.
    """

    if "patient_id" not in windows.columns:
        raise ValueError("windows must contain patient_id")
    if int(min_successful_windows) < 1:
        raise ValueError("min_successful_windows must be positive")
    names = list(feature_names)
    missing = [name for name in names if name not in windows.columns]
    if missing:
        raise ValueError(f"window feature columns missing: {missing}")

    rows: list[dict[str, Any]] = []
    work = windows.copy()
    # Stable sort makes generated parquet and reports deterministic without
    # relying on the input row order.
    work["patient_id"] = work["patient_id"].astype("string")
    for patient_id, group in work.groupby("patient_id", sort=True, dropna=False):
        success = group.loc[_successful_mask(group)].copy()
        n_theoretical = int(len(group))
        n_successful = int(len(success))
        if n_successful < int(min_successful_windows) and not include_ineligible:
            continue

        row: dict[str, Any] = {"patient_id": patient_id}
        # Labels and provenance are metadata, never model features.
        row["label"] = _first_value(group, "label")
        if pd.isna(row["label"]):
            row["label"] = _first_value(group, "scd_high_risk")
        row["followup_days"] = _first_value(group, "followup_days")
        row["cause_of_death"] = _first_value(group, "cause_of_death")
        row["fs"] = _first_value(group, "fs")
        for name in names:
            stats = _stats(success[name] if n_successful else pd.Series(dtype=float))
            for suffix in AGGREGATION_SUFFIXES:
                row[f"{name}_{suffix}"] = stats[suffix]
        row["n_windows_theoretical"] = n_theoretical
        row["n_windows_successful"] = n_successful
        # Keep the old output name as an audit alias; it is not one of the 100
        # model columns and always means successful windows here.
        row["n_windows_used"] = n_successful
        row["window_success_rate"] = float(n_successful / n_theoretical) if n_theoretical else np.nan
        for source, target in (
            ("raw_rr_count", "raw_rr_count_total"),
            ("valid_rr_count", "valid_rr_count_total"),
            ("removed_rr_count", "removed_rr_count_total"),
        ):
            if source in success.columns:
                row[target] = float(pd.to_numeric(success[source], errors="coerce").fillna(0).sum())
            else:
                row[target] = np.nan
        rows.append(row)

    columns = ["patient_id", "label", "followup_days", "cause_of_death", "fs"]
    columns += list(AGGREGATED_FEATURE_NAMES)
    columns += [c for c in METADATA_COLUMNS if c not in columns]
    result = pd.DataFrame(rows)
    if result.empty:
        return pd.DataFrame(columns=columns)
    for name in AGGREGATED_FEATURE_NAMES:
        result[name] = pd.to_numeric(result[name], errors="coerce").astype("float64")
    result = result.reindex(columns=columns)
    return result.sort_values("patient_id", kind="stable").reset_index(drop=True)


# Descriptive aliases used by scripts/tests and by downstream callers.
aggregate_legacy_windows = aggregate_patient_features
aggregate_patient = aggregate_patient_features


__all__ = [
    "AGGREGATION_SUFFIXES",
    "AGGREGATED_FEATURE_NAMES",
    "METADATA_COLUMNS",
    "aggregated_feature_names",
    "aggregate_dynamic_features",
    "aggregate_patient_features",
    "aggregate_legacy_windows",
    "aggregate_patient",
]
