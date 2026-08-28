"""Analysis profiles, eligibility status, and endpoint joins for Phase 4."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import yaml

from .endpoints import SUPPORTED_HORIZONS, build_endpoint
from .full_aggregation import AGGREGATED_FEATURE_NAMES, VALID_COUNT_NAMES
from .full_features import FEATURE_NAMES, SIGNAL_QUALITY_FEATURE_NAMES


DEFAULT_PROFILE_PATH = Path(__file__).resolve().parents[1] / "config" / "analysis_profiles.yaml"
PROFILE_PRIMARY = "primary_sinus_hrv"
PROFILE_PHYSIOLOGY_ONLY = "physiology_only"
PROFILE_FULL = "full_20_feature"
REASON_NO_HOLTER = "NO_HOLTER"
REASON_PROCESSING_FAILED = "PROCESSING_FAILED"
REASON_NO_QC_VALID_WINDOWS = "NO_QC_VALID_WINDOWS"
REASON_AF = "AF"
REASON_HIGH_PVC = "HIGH_PVC_BURDEN"
REASON_ELIGIBLE = "ELIGIBLE"


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_hash(value: Any) -> str:
    payload = yaml.safe_dump(value, sort_keys=True, allow_unicode=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def load_analysis_profiles(path: str | Path = DEFAULT_PROFILE_PATH) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle) or {}
    if not isinstance(value, dict) or not isinstance(value.get("profiles"), dict):
        raise ValueError("analysis profile configuration must contain profiles")
    return value


def get_profile(name: str = PROFILE_PRIMARY, path: str | Path = DEFAULT_PROFILE_PATH) -> dict[str, Any]:
    profiles = load_analysis_profiles(path).get("profiles", {})
    if name not in profiles:
        raise KeyError(f"unknown analysis profile: {name}")
    profile = profiles[name]
    if not isinstance(profile, dict):
        raise ValueError(f"analysis profile {name} must be a mapping")
    return dict(profile)


def profile_feature_names(
    name: str = PROFILE_FULL,
    *,
    path: str | Path = DEFAULT_PROFILE_PATH,
) -> list[str]:
    profile = get_profile(name, path)
    drops = set(profile.get("drop_categories", []))
    if name == PROFILE_PHYSIOLOGY_ONLY or drops:
        # Categories are frozen by features_v2.yaml; the only configured drop
        # is SIGNAL_QUALITY (sig_skew and sig_kurt).
        return [feature for feature in FEATURE_NAMES if feature not in SIGNAL_QUALITY_FEATURE_NAMES]
    return list(FEATURE_NAMES)


def _as_bool(value: Any) -> bool:
    if value is None or pd.isna(value):
        return False
    return bool(value)


def _numeric(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if np.isfinite(result) else default


def _window_summary(windows: pd.DataFrame, patient_id: str) -> dict[str, Any]:
    if windows is None or "patient_id" not in windows.columns:
        group = pd.DataFrame()
    else:
        group = windows.loc[windows["patient_id"].astype("string").eq(str(patient_id))]
    n_theoretical = int(len(group))
    if "feature_extraction_success" in group.columns:
        successful = group["feature_extraction_success"].fillna(False).astype(bool)
    elif "window_status" in group.columns:
        successful = group["window_status"].astype("string").str.upper().eq("SUCCESS")
    else:
        successful = pd.Series(True, index=group.index)
    qc = group["qc_valid"].fillna(False).astype(bool) if "qc_valid" in group.columns else successful
    row: dict[str, Any] = {
        "n_windows_theoretical": n_theoretical,
        "n_windows_successful": int(successful.sum()),
        "n_windows_qc_valid": int(qc.sum()),
        "raw_rpeak_count_total": 0.0,
        "raw_rr_count_total": 0.0,
        "valid_rr_count_total": 0.0,
        "removed_rr_count_total": 0.0,
        "tail_seconds": _numeric(group["tail_seconds"].dropna().iloc[0], 0.0) if "tail_seconds" in group and group["tail_seconds"].notna().any() else 0.0,
    }
    for source, target in (
        ("raw_rpeak_count", "raw_rpeak_count_total"),
        ("raw_rr_count", "raw_rr_count_total"),
        ("valid_rr_count", "valid_rr_count_total"),
        ("removed_rr_count", "removed_rr_count_total"),
    ):
        if source in group.columns:
            values = pd.to_numeric(group.loc[successful, source], errors="coerce")
            row[target] = float(values.fillna(0).sum())
    return row


def _aggregate_row(patient_features: pd.DataFrame | None, patient_id: str) -> pd.Series | None:
    if patient_features is None or "patient_id" not in patient_features.columns:
        return None
    matches = patient_features.loc[patient_features["patient_id"].astype("string").eq(str(patient_id))]
    if matches.empty:
        return None
    return matches.iloc[0]


def _processed_value(aggregate: pd.Series | None, summary: Mapping[str, Any], has_holter: bool) -> bool:
    if aggregate is not None and "processed_holter" in aggregate.index:
        value = aggregate["processed_holter"]
        if not pd.isna(value):
            return bool(value)
    # A non-empty theoretical window set means the builder attempted the
    # record, including the case where every exact read failed.
    return bool(has_holter and int(summary.get("n_windows_theoretical", 0)) > 0)


def build_patient_analysis_status(
    subjects: pd.DataFrame,
    windows: pd.DataFrame | None = None,
    patient_features: pd.DataFrame | None = None,
    *,
    endpoints: Mapping[int, pd.DataFrame] | None = None,
    pvc_threshold: float = 0.20,
) -> pd.DataFrame:
    """Build one status row for every official subject and dynamic endpoint."""

    if "patient_id" not in subjects.columns:
        raise KeyError("subjects must contain patient_id")
    if subjects["patient_id"].duplicated().any():
        raise ValueError("subjects patient_id must be unique")
    subject = subjects.copy()
    subject["patient_id"] = subject["patient_id"].astype("string")
    endpoint_tables: dict[int, pd.DataFrame] = {}
    for horizon in sorted(SUPPORTED_HORIZONS):
        if endpoints is not None and int(horizon) in endpoints:
            endpoint = endpoints[int(horizon)].copy()
        else:
            endpoint = build_endpoint(subject, int(horizon))
        if endpoint["patient_id"].duplicated().any():
            raise ValueError(f"endpoint {horizon} contains duplicate patient_id")
        endpoint_tables[int(horizon)] = endpoint.set_index("patient_id")

    rows: list[dict[str, Any]] = []
    for source in subject.sort_values("patient_id", kind="stable").to_dict("records"):
        patient_id = str(source["patient_id"])
        summary = _window_summary(windows if windows is not None else pd.DataFrame(), patient_id)
        aggregate = _aggregate_row(patient_features, patient_id)
        # Compact builds normally pass only the patient aggregation here.  In
        # that mode recover status counts from the aggregate rather than
        # silently reporting zero windows for every processed Holter.
        if aggregate is not None:
            for aggregate_name, summary_name in (
                ("n_windows_theoretical", "n_windows_theoretical"),
                ("n_windows_successful", "n_windows_successful"),
                ("n_windows_qc_valid", "n_windows_qc_valid"),
                ("raw_rpeak_count_total", "raw_rpeak_count_total"),
                ("raw_rr_count_total", "raw_rr_count_total"),
                ("valid_rr_count_total", "valid_rr_count_total"),
                ("removed_rr_count_total", "removed_rr_count_total"),
                ("tail_seconds", "tail_seconds"),
            ):
                if aggregate_name in aggregate.index and not pd.isna(aggregate[aggregate_name]):
                    summary[summary_name] = aggregate[aggregate_name]
        # The full builder aggregates one patient at a time to keep memory
        # bounded.  In that path the patient-level counters are the authoritative
        # window summary and avoid materialising every waveform-derived row just
        # to construct the 992-row status table.
        if aggregate is not None and int(summary.get("n_windows_theoretical", 0)) == 0:
            for key in (
                "n_windows_theoretical", "n_windows_successful", "n_windows_qc_valid",
                "raw_rpeak_count_total", "raw_rr_count_total", "valid_rr_count_total",
                "removed_rr_count_total", "tail_seconds",
            ):
                if key in aggregate.index and not pd.isna(aggregate[key]):
                    summary[key] = aggregate[key]
        has_holter = _as_bool(source.get("has_holter", False))
        processed = _processed_value(aggregate, summary, has_holter)
        af = _as_bool(source.get("af_flag", False))
        if aggregate is not None and "high_pvc_burden" in aggregate.index and not pd.isna(aggregate["high_pvc_burden"]):
            high_pvc = bool(aggregate["high_pvc_burden"])
        else:
            high_pvc = False
        if not has_holter:
            reason = REASON_NO_HOLTER
        elif not processed:
            reason = REASON_PROCESSING_FAILED
        elif af:
            reason = REASON_AF
        elif high_pvc:
            reason = REASON_HIGH_PVC
        elif int(summary["n_windows_qc_valid"]) < 1:
            reason = REASON_NO_QC_VALID_WINDOWS
        else:
            reason = REASON_ELIGIBLE
        eligible = reason == REASON_ELIGIBLE
        row: dict[str, Any] = {
            "patient_id": patient_id,
            "has_holter": has_holter,
            "processed_holter": processed,
            "holter_processed": processed,
            "af_flag": af,
            "n_windows_theoretical": int(summary["n_windows_theoretical"]),
            "n_windows_successful": int(summary["n_windows_successful"]),
            "n_windows_qc_valid": int(summary["n_windows_qc_valid"]),
            "valid_window_count": int(summary["n_windows_qc_valid"]),
            "tail_seconds": float(summary["tail_seconds"]),
            "raw_rpeak_count_total": float(summary["raw_rpeak_count_total"]),
            "raw_rr_count_total": float(summary["raw_rr_count_total"]),
            "valid_rr_count_total": float(summary["valid_rr_count_total"]),
            "removed_rr_count_total": float(summary["removed_rr_count_total"]),
            "pvc_count_24h": aggregate.get("pvc_count_24h", np.nan) if aggregate is not None else source.get("pvc_count_24h", np.nan),
            "pvc_denominator_beats": aggregate.get("pvc_denominator_beats", np.nan) if aggregate is not None else np.nan,
            "pvc_burden": aggregate.get("pvc_burden", np.nan) if aggregate is not None else np.nan,
            "pvc_burden_threshold": float(pvc_threshold),
            "high_pvc_burden": high_pvc,
            "high_pvc_flag": high_pvc,
            "primary_sinus_hrv_eligible": eligible,
            "primary_sinus_hrv_reason": reason,
        }
        for horizon, endpoint in endpoint_tables.items():
            if patient_id not in endpoint.index:
                raise ValueError(f"endpoint {horizon} missing subject {patient_id}")
            value = endpoint.loc[patient_id]
            row[f"endpoint_{horizon}_state"] = value["endpoint_state"]
            row[f"endpoint_{horizon}_binary_label"] = value["binary_label_if_evaluable"]
            row[f"endpoint_{horizon}_time_to_event"] = value["time_to_event"]
            row[f"endpoint_{horizon}_event_type"] = value["event_type"]
        endpoint_365_state = row["endpoint_365_state"]
        row["model_365_included"] = bool(eligible and endpoint_365_state in {"POSITIVE", "NEGATIVE"})
        if not eligible:
            row["model_365_exclusion_reason"] = reason
        elif endpoint_365_state not in {"POSITIVE", "NEGATIVE"}:
            row["model_365_exclusion_reason"] = str(endpoint_365_state)
        else:
            row["model_365_exclusion_reason"] = "INCLUDED"
        rows.append(row)
    result = pd.DataFrame(rows)
    result["patient_id"] = result["patient_id"].astype("string")
    for column in ("has_holter", "processed_holter", "holter_processed", "af_flag", "high_pvc_burden", "high_pvc_flag", "primary_sinus_hrv_eligible", "model_365_included"):
        result[column] = result[column].astype("boolean")
    for column in ("pvc_burden", "pvc_denominator_beats", "pvc_count_24h", "tail_seconds", "pvc_burden_threshold"):
        result[column] = pd.to_numeric(result[column], errors="coerce").astype("float64")
    for horizon in sorted(SUPPORTED_HORIZONS):
        result[f"endpoint_{horizon}_state"] = result[f"endpoint_{horizon}_state"].astype("string")
        result[f"endpoint_{horizon}_binary_label"] = pd.to_numeric(result[f"endpoint_{horizon}_binary_label"], errors="coerce").astype("Int64")
        result[f"endpoint_{horizon}_time_to_event"] = pd.to_numeric(result[f"endpoint_{horizon}_time_to_event"], errors="coerce").astype("float64")
        result[f"endpoint_{horizon}_event_type"] = result[f"endpoint_{horizon}_event_type"].astype("string")
    return result.sort_values("patient_id", kind="stable").reset_index(drop=True)


def build_survival_ready(
    subjects: pd.DataFrame,
    patient_features: pd.DataFrame,
    status: pd.DataFrame,
    *,
    horizon_days: int = 365,
) -> pd.DataFrame:
    """Join full patient aggregation and status while retaining all subjects."""

    if horizon_days not in SUPPORTED_HORIZONS:
        raise ValueError(f"unsupported endpoint horizon: {horizon_days}")
    base = subjects.loc[:, ["patient_id"]].copy()
    base["patient_id"] = base["patient_id"].astype("string")
    if base["patient_id"].duplicated().any():
        raise ValueError("subjects patient_id must be unique")
    features = patient_features.copy()
    features["patient_id"] = features["patient_id"].astype("string")
    if features["patient_id"].duplicated().any():
        raise ValueError("patient_features patient_id must be unique")
    status_copy = status.copy()
    status_copy["patient_id"] = status_copy["patient_id"].astype("string")
    if status_copy["patient_id"].duplicated().any():
        raise ValueError("status patient_id must be unique")
    result = base.merge(features, on="patient_id", how="left", validate="one_to_one", suffixes=("", "_features"))
    keep_status = [
        "patient_id", "processed_holter", "af_flag", "n_windows_theoretical", "n_windows_successful",
        "n_windows_qc_valid", "tail_seconds", "pvc_burden", "high_pvc_burden", "primary_sinus_hrv_eligible",
        "primary_sinus_hrv_reason",
    ]
    keep_status += [column for column in status_copy.columns if column.startswith("endpoint_")]
    result = result.merge(status_copy.loc[:, list(dict.fromkeys(keep_status))], on="patient_id", how="left", validate="one_to_one", suffixes=("", "_status"))
    result["time_to_event"] = result[f"endpoint_{horizon_days}_time_to_event"]
    result["event_type"] = result[f"endpoint_{horizon_days}_event_type"]
    state = result[f"endpoint_{horizon_days}_state"].astype("string")
    result["censored"] = state.eq("CENSORED").astype("boolean")
    result["competing_event"] = state.eq("COMPETING_EVENT").astype("boolean")
    # Do not silently drop official rows if a malformed build omitted one.
    if len(result) != len(base) or result["patient_id"].duplicated().any():
        raise RuntimeError("survival-ready identity cardinality failed")
    return result.sort_values("patient_id", kind="stable").reset_index(drop=True)


def build_analysis_population_365(
    survival_ready: pd.DataFrame,
    *,
    feature_names: Sequence[str] = AGGREGATED_FEATURE_NAMES,
) -> pd.DataFrame:
    """Select the 365-day positive/negative primary eligible population."""

    required = {"patient_id", "endpoint_365_state", "primary_sinus_hrv_eligible", *feature_names}
    missing = sorted(required - set(survival_ready.columns))
    if missing:
        raise KeyError(f"survival_ready missing columns: {missing}")
    state = survival_ready["endpoint_365_state"].astype("string").isin(["POSITIVE", "NEGATIVE"])
    eligible = survival_ready["primary_sinus_hrv_eligible"].fillna(False).astype(bool)
    finite = np.isfinite(survival_ready.loc[:, list(feature_names)].apply(pd.to_numeric, errors="coerce")).all(axis=1)
    result = survival_ready.loc[state & eligible & finite].copy()
    result["label_365d"] = result["endpoint_365_state"].map({"POSITIVE": 1, "NEGATIVE": 0}).astype("Int64")
    return result.sort_values("patient_id", kind="stable").reset_index(drop=True)


__all__ = [
    "DEFAULT_PROFILE_PATH", "PROFILE_PRIMARY", "PROFILE_PHYSIOLOGY_ONLY", "PROFILE_FULL",
    "REASON_NO_HOLTER", "REASON_PROCESSING_FAILED", "REASON_NO_QC_VALID_WINDOWS", "REASON_AF",
    "REASON_HIGH_PVC", "REASON_ELIGIBLE", "sha256_file", "canonical_hash", "load_analysis_profiles",
    "get_profile", "profile_feature_names", "build_patient_analysis_status", "build_survival_ready",
    "build_analysis_population_365",
]
