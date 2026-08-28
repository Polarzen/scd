"""Preparation contracts for MODEL OPTIMIZATION V1.

This module intentionally stops at a deterministic, auditable modeling frame.
It does not run cross-validation or fit a selector.  Outcomes are rebuilt from
the official subject source through :func:`src.endpoints.build_endpoint`, and
the feature allowlist is explicit so metadata, identifiers, and labels cannot
silently become model inputs.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from .endpoints import build_endpoint
from .full_aggregation import AGGREGATED_FEATURE_NAMES
from .full_features import FEATURE_NAMES


RHYTHM_SAFE_BASES: tuple[str, ...] = (
    "sig_mean",
    "sig_std",
    "sig_p2p",
    "sig_skew",
    "sig_kurt",
    "beats",
    "beats_per_min",
    "pow_lf",
    "pow_mf",
    "pow_hf",
    "pow_hf_ratio",
)
RHYTHM_SAFE_FEATURES = RHYTHM_SAFE_BASES
P50_SUFFIX = "p50"
ROBUST_SUFFIX = "p90_minus_p10"
ALL20_100_PROFILE = "all20_100"
P50_20_PROFILE = "median20"
ROBUST40_PROFILE = "robust40"
RHYTHM_SAFE_PROFILE = "rhythm_safe"
FROZEN_100_PROFILE = "compact_selected"
PVC_SOURCE_FEATURE = "pvc_count_24h"
# Verbose aliases keep the public schema readable to downstream callers.
PROFILE_ALL20_100 = ALL20_100_PROFILE
PROFILE_P50_20 = P50_20_PROFILE
PROFILE_ROBUST40 = ROBUST40_PROFILE
PROFILE_RHYTHM_SAFE = RHYTHM_SAFE_PROFILE
PROFILE_FROZEN_100 = FROZEN_100_PROFILE

PVC_CONTINUOUS_UNAVAILABLE = True
PVC_UNAVAILABLE_REASON = (
    "derived pvc_burden is unavailable: its numerator is a 24-hour source "
    "count while the extracted-window denominator has a different time base"
)

# OOF columns are metadata only.  They may be retained in a prepared frame,
# but they are never accepted by the feature allowlist.
OOF_METADATA_COLUMNS: tuple[str, ...] = (
    "outer_fold",
    "fold",
    "fold_threshold",
    "threshold",
    "prediction_probability",
    "probability",
    "prediction_label",
    "prediction",
    "true_label",
    "y_true",
    "model",
    "profile",
    "seed",
)
ENDPOINT_METADATA_COLUMNS: tuple[str, ...] = (
    "endpoint_state",
    "endpoint_horizon_days",
    "time_to_event",
    "event_type",
)
STATUS_COLUMNS: tuple[str, ...] = (
    "has_holter",
    "processed_holter",
    "holter_processed",
    "processing_status",
    "n_windows_theoretical",
    "n_windows_successful",
    "n_windows_qc_valid",
    "valid_window_count",
    "af_flag",
    "pvc_count_24h",
    "pvc_information_available",
    "pvc_burden",
    "pvc_denominator_beats",
    "pvc_burden_status",
    "high_pvc_burden",
    "high_pvc_flag",
    "primary_sinus_hrv_eligible",
    "primary_sinus_hrv_reason",
)


def _profile_key(value: str) -> str:
    key = str(value).strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "all20": ALL20_100_PROFILE,
        "all_20": ALL20_100_PROFILE,
        "all20_100": ALL20_100_PROFILE,
        "all_100": ALL20_100_PROFILE,
        "full": ALL20_100_PROFILE,
        "p50": P50_20_PROFILE,
        "p50_20": P50_20_PROFILE,
        "median20": P50_20_PROFILE,
        "20_p50": P50_20_PROFILE,
        "40": ROBUST40_PROFILE,
        "robust": ROBUST40_PROFILE,
        "robust40": ROBUST40_PROFILE,
        "40_robust": ROBUST40_PROFILE,
        "rhythm": RHYTHM_SAFE_PROFILE,
        "rhythm_safe": RHYTHM_SAFE_PROFILE,
        "frozen": FROZEN_100_PROFILE,
        "frozen100": FROZEN_100_PROFILE,
        "frozen_100": FROZEN_100_PROFILE,
        "compact_selected": FROZEN_100_PROFILE,
    }
    if key not in aliases:
        raise ValueError(f"unknown optimization feature profile: {value}")
    return aliases[key]


def _profile_feature_columns(profile: str) -> list[str]:
    key = _profile_key(profile)
    if key in {ALL20_100_PROFILE, FROZEN_100_PROFILE}:
        return list(AGGREGATED_FEATURE_NAMES)
    if key == P50_20_PROFILE:
        return [f"{base}_{P50_SUFFIX}" for base in FEATURE_NAMES]
    if key == ROBUST40_PROFILE:
        return [
            *[f"{base}_{P50_SUFFIX}" for base in FEATURE_NAMES],
            *[f"{base}_{ROBUST_SUFFIX}" for base in FEATURE_NAMES],
        ]
    if key == RHYTHM_SAFE_PROFILE:
        return [
            *[f"{base}_{P50_SUFFIX}" for base in RHYTHM_SAFE_BASES],
            *[f"{base}_{ROBUST_SUFFIX}" for base in RHYTHM_SAFE_BASES],
            "af_flag",
        ]
    raise AssertionError(f"unhandled profile {key}")


def profile_feature_columns(profile: str = ALL20_100_PROFILE) -> list[str]:
    """Return the frozen, ordered feature columns for a profile."""

    return _profile_feature_columns(profile)


@dataclass(frozen=True)
class CandidateSpec:
    """Fixed MODEL OPTIMIZATION V1 candidate contract."""

    name: str
    profile: str
    model: str
    population: str = "primary"
    include_pvc: bool = False
    selector_pipeline_owned: bool = False

    @property
    def feature_cols(self) -> tuple[str, ...]:
        columns = _profile_feature_columns(self.profile)
        if self.include_pvc:
            columns.append(PVC_SOURCE_FEATURE)
        return tuple(columns)

    @property
    def feature_count(self) -> int:
        return len(self.feature_cols)


_CANDIDATES: dict[str, CandidateSpec] = {
    "B0": CandidateSpec("B0", ALL20_100_PROFILE, "extratrees", "primary"),
    "M1": CandidateSpec("M1", P50_20_PROFILE, "elasticnet", "primary"),
    "M2": CandidateSpec("M2", ROBUST40_PROFILE, "elasticnet", "primary"),
    "M3": CandidateSpec("M3", ROBUST40_PROFILE, "extratrees_regularized", "primary"),
    "M4": CandidateSpec(
        "M4",
        FROZEN_100_PROFILE,
        "elasticnet_selected",
        "primary",
        selector_pipeline_owned=True,
    ),
    "A1": CandidateSpec("A1", RHYTHM_SAFE_PROFILE, "elasticnet", "rhythm_safe"),
    "A2": CandidateSpec("A2", RHYTHM_SAFE_PROFILE, "extratrees_regularized", "rhythm_safe"),
    # PVC candidates use the same AF-compatible population/profile.  P2 differs
    # from P1 by one source feature; derived pvc_burden is never a model input.
    "P1": CandidateSpec("P1", RHYTHM_SAFE_PROFILE, "elasticnet", "rhythm_safe"),
    "P2": CandidateSpec("P2", RHYTHM_SAFE_PROFILE, "elasticnet", "rhythm_safe", include_pvc=True),
}
CANDIDATES = dict(_CANDIDATES)
CANDIDATE_SPECS = CANDIDATES
CANDIDATE_REGISTRY = CANDIDATES
FEATURE_PROFILES = {
    name: tuple(_profile_feature_columns(name))
    for name in (
        ALL20_100_PROFILE,
        P50_20_PROFILE,
        ROBUST40_PROFILE,
        RHYTHM_SAFE_PROFILE,
        FROZEN_100_PROFILE,
    )
}
PROFILE_FEATURES = FEATURE_PROFILES


def candidate_spec(candidate: str = "B0") -> CandidateSpec:
    """Resolve a candidate name case-insensitively."""

    key = str(candidate).strip().upper()
    if key not in _CANDIDATES:
        raise ValueError(f"unknown MODEL OPTIMIZATION V1 candidate: {candidate}")
    return _CANDIDATES[key]


def candidate_registry() -> dict[str, CandidateSpec]:
    """Return a copy of the fixed candidate registry."""

    return dict(_CANDIDATES)


def candidate_feature_columns(candidate: str = "B0") -> list[str]:
    return list(candidate_spec(candidate).feature_cols)


_SAFE_FEATURES = frozenset(
    set(AGGREGATED_FEATURE_NAMES)
    | {f"{base}_{P50_SUFFIX}" for base in FEATURE_NAMES}
    | {f"{base}_{ROBUST_SUFFIX}" for base in FEATURE_NAMES}
    | set(RHYTHM_SAFE_BASES)
    | {"af_flag", PVC_SOURCE_FEATURE}
)
_FORBIDDEN_FEATURE_TOKENS = (
    "label",
    "outcome",
    "endpoint",
    "cause",
    "death",
    "event",
    "followup",
    "patient_id",
    "record_id",
    "prediction",
    "probability",
    "fold",
    "threshold",
)


def validate_feature_allowlist(
    feature_cols: Sequence[str],
    *,
    profile: str | None = None,
    frame: pd.DataFrame | None = None,
) -> list[str]:
    """Validate an explicit feature allowlist and reject injection columns."""

    columns = [str(column) for column in feature_cols]
    if not columns:
        raise ValueError("feature allowlist must not be empty")
    if len(columns) != len(set(columns)):
        raise ValueError("feature allowlist contains duplicate columns")
    if profile is not None:
        expected = set(_profile_feature_columns(profile)) | {PVC_SOURCE_FEATURE}
        unknown_to_profile = sorted(set(columns) - expected)
        if unknown_to_profile:
            raise ValueError(f"feature allowlist is outside profile {profile}: {unknown_to_profile}")
    unknown = sorted(set(columns) - _SAFE_FEATURES)
    if unknown:
        raise ValueError(f"feature allowlist contains non-approved columns: {unknown}")
    injected = [
        column
        for column in columns
        if any(token in column.lower() for token in _FORBIDDEN_FEATURE_TOKENS)
    ]
    if injected:
        raise ValueError(f"feature allowlist contains outcome/ID metadata: {injected}")
    if frame is not None:
        missing = [column for column in columns if column not in frame.columns]
        if missing:
            raise KeyError(f"feature frame is missing allowlisted columns: {missing}")
    return columns


def _as_frame(value: pd.DataFrame | Path | str, table_name: str) -> pd.DataFrame:
    if isinstance(value, pd.DataFrame):
        return value.copy(deep=True)
    path = Path(value)
    if not path.is_file():
        raise FileNotFoundError(f"{table_name} does not exist: {path}")
    return pd.read_parquet(path)


def _endpoint_table(subjects: pd.DataFrame, horizon_days: int) -> pd.DataFrame:
    source_required = {"patient_id", "followup_days", "cause_of_death_raw", "event_source_valid"}
    if source_required <= set(subjects.columns):
        # This is the only label construction path for source subjects.
        return build_endpoint(subjects, int(horizon_days))
    endpoint_required = {"patient_id", "endpoint_state", "binary_label_if_evaluable"}
    if endpoint_required <= set(subjects.columns):
        endpoint = subjects.loc[:, list(subjects.columns)].copy()
        if endpoint["patient_id"].isna().any() or endpoint["patient_id"].duplicated().any():
            raise ValueError("endpoint table must contain unique, non-null patient_id")
        if "endpoint_horizon_days" not in endpoint.columns:
            endpoint["endpoint_horizon_days"] = int(horizon_days)
        return endpoint
    missing = sorted(source_required - set(subjects.columns))
    raise ValueError(f"subjects must contain endpoint source fields; missing {missing}")


def _normalise_ids(frame: pd.DataFrame, name: str) -> pd.DataFrame:
    if "patient_id" not in frame.columns:
        raise KeyError(f"{name} must contain patient_id")
    if frame["patient_id"].isna().any() or frame["patient_id"].duplicated().any():
        raise ValueError(f"{name} must contain one unique, non-null row per patient")
    result = frame.copy(deep=True)
    result["patient_id"] = result["patient_id"].astype("string")
    return result


def _numeric_series(frame: pd.DataFrame, column: str, default: float = np.nan) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(default, index=frame.index, dtype="float64")
    return pd.to_numeric(frame[column], errors="coerce")


def _bool_series(frame: pd.DataFrame, column: str, default: bool = False) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(default, index=frame.index, dtype=bool)
    values = frame[column]
    if pd.api.types.is_bool_dtype(values):
        return values.fillna(default).astype(bool)
    text = values.astype("string").str.strip().str.lower()
    mapped = text.map({"true": True, "1": True, "yes": True, "y": True, "false": False, "0": False, "no": False, "n": False})
    numeric = pd.to_numeric(values, errors="coerce").eq(1)
    return mapped.fillna(numeric).fillna(default).astype(bool)


def _coalesce_status(
    endpoint: pd.DataFrame,
    features: pd.DataFrame,
    subjects: pd.DataFrame,
) -> pd.DataFrame:
    """Join status metadata, preferring true source fields where available."""

    result = endpoint.copy(deep=True)
    source = subjects.set_index("patient_id", drop=False)
    feature = features.set_index("patient_id", drop=False)
    ids = result["patient_id"].astype("string")
    for column in STATUS_COLUMNS:
        source_value = source[column].reindex(ids) if column in source.columns else pd.Series(np.nan, index=ids)
        feature_value = feature[column].reindex(ids) if column in feature.columns else pd.Series(np.nan, index=ids)
        # Source metadata is authoritative for AF and the official PVC count;
        # processing/window status is normally only present in features.
        if column in {"af_flag", PVC_SOURCE_FEATURE, "pvc_information_available"}:
            # When the source column exists, preserve its missingness.  In
            # particular, a missing official PVC count must not be replaced by
            # a copied/derived feature-table value.
            value = source_value if column in source.columns else feature_value
        else:
            value = feature_value.where(feature_value.notna(), source_value)
        result[column] = value.to_numpy()

    # A source cohort without an explicit derived status is still safe to
    # inspect, but rhythm-safe eligibility remains fail-closed below.
    result["has_holter"] = _bool_series(result, "has_holter")
    processed = _bool_series(result, "processed_holter")
    if "processing_status" in result.columns:
        processed = processed | result["processing_status"].astype("string").str.upper().eq("COMPLETE")
    result["processed_holter"] = processed
    result["holter_processed"] = processed
    result["af_flag"] = _bool_series(result, "af_flag")
    result["n_windows_successful"] = _numeric_series(result, "n_windows_successful", 0.0).fillna(0.0)
    result["pvc_count_24h"] = _numeric_series(result, PVC_SOURCE_FEATURE)
    result["pvc_information_available"] = _bool_series(result, "pvc_information_available", False)
    # Derived burden is intentionally retained for audit visibility but is not
    # considered available for continuous PVC optimization.
    if "pvc_burden_status" not in result.columns:
        result["pvc_burden_status"] = "UNAVAILABLE_TIME_BASE_MISMATCH"
    else:
        result["pvc_burden_status"] = result["pvc_burden_status"].astype("string").fillna(
            "UNAVAILABLE_TIME_BASE_MISMATCH"
        )
    primary_present = "primary_sinus_hrv_eligible" in source.columns or "primary_sinus_hrv_eligible" in feature.columns
    if primary_present:
        # Preserve the existing analysis decision exactly when supplied by the
        # phase-4 status/feature build.
        result["primary_sinus_hrv_eligible"] = _bool_series(result, "primary_sinus_hrv_eligible", False)
    else:
        # Small fixtures may omit the persisted status.  Reconstruct only the
        # conservative core eligibility gates; AF and high PVC remain excluded
        # from this fallback, while an explicit persisted value always wins.
        result["primary_sinus_hrv_eligible"] = (
            _bool_series(result, "has_holter")
            & _bool_series(result, "processed_holter")
            & _numeric_series(result, "n_windows_successful", 0.0).fillna(0).ge(1)
            & ~_bool_series(result, "af_flag")
            & ~_bool_series(result, "high_pvc_burden")
        )
    return result


def _add_derived_feature_columns(frame: pd.DataFrame, needed: Iterable[str]) -> pd.DataFrame:
    result = frame.copy(deep=True)
    requested = set(needed)
    for column in requested:
        marker = f"_{ROBUST_SUFFIX}"
        if column.endswith(marker):
            base = column[: -len(marker)]
            p90 = f"{base}_p90"
            p10 = f"{base}_p10"
            if p90 in result.columns and p10 in result.columns:
                # Always recompute when the deterministic source pair exists;
                # a pre-existing robust column may have been produced by a
                # different aggregation convention.
                result[column] = _numeric_series(result, p90) - _numeric_series(result, p10)
            elif column not in result.columns:
                continue
        elif column in result.columns:
            continue
    return result


def _source_feature_frame(features: pd.DataFrame, needed: Sequence[str]) -> pd.DataFrame:
    result = _add_derived_feature_columns(features, needed)
    # P50 and all frozen aggregate names are supplied by the existing feature
    # build.  Robust columns are derived above from its deterministic p90/p10.
    missing = [column for column in needed if column not in result.columns]
    if missing:
        raise KeyError(f"feature frame is missing optimization columns: {missing}")
    return result


def _rhythm_safe_mask(frame: pd.DataFrame, feature_cols: Sequence[str]) -> pd.Series:
    rhythm_columns = [
        column
        for column in feature_cols
        if column != "af_flag" and column != PVC_SOURCE_FEATURE
    ]
    if not rhythm_columns:
        return pd.Series(False, index=frame.index)
    numeric = frame.loc[:, rhythm_columns].apply(pd.to_numeric, errors="coerce")
    finite_value = np.isfinite(numeric.to_numpy(dtype=np.float64)).any(axis=1)
    return (
        _bool_series(frame, "has_holter")
        & _bool_series(frame, "processed_holter")
        & _numeric_series(frame, "n_windows_successful", 0.0).fillna(0).ge(1)
        & pd.Series(finite_value, index=frame.index)
    )


def _population_mask(frame: pd.DataFrame, population: str, feature_cols: Sequence[str]) -> pd.Series:
    binary = frame["endpoint_state"].astype("string").isin(["POSITIVE", "NEGATIVE"])
    key = str(population).strip().lower().replace("-", "_")
    if key in {"primary", "baseline", "model", "m"}:
        return binary & _bool_series(frame, "primary_sinus_hrv_eligible", False)
    if key in {"rhythm_safe", "rhythm", "af_safe", "af_inclusive"}:
        return binary & _rhythm_safe_mask(frame, feature_cols)
    if key in {"binary", "all"}:
        return binary
    raise ValueError(f"unknown optimization population: {population}")


def _manifest_and_audit(
    frame: pd.DataFrame,
    *,
    horizon_days: int,
    official_strict: bool = True,
) -> tuple[dict[str, Any], dict[str, Any]]:
    state = frame["endpoint_state"].astype("string")
    binary = state.isin(["POSITIVE", "NEGATIVE"])
    positive = state.eq("POSITIVE")
    primary = binary & _bool_series(frame, "primary_sinus_hrv_eligible", False)
    rhythm = binary & _rhythm_safe_mask(frame, [column for column in frame.columns if column in _SAFE_FEATURES])
    baseline_positive_excluded = positive & ~primary
    rhythm_positive_excluded = positive & ~rhythm

    counts: dict[str, Any] = {
        "source_subject_count": int(len(frame)),
        "official_rows": int(len(frame)),
        "endpoint_positive": int(positive.sum()),
        "endpoint_negative": int(state.eq("NEGATIVE").sum()),
        "endpoint_censored": int(state.eq("CENSORED").sum()),
        "endpoint_competing_event": int(state.eq("COMPETING_EVENT").sum()),
        "endpoint_unknown": int(state.eq("UNKNOWN").sum()),
        "full_positive": int(positive.sum()),
        "full_positives": int(positive.sum()),
        "full_negative": int(state.eq("NEGATIVE").sum()),
        "full_binary": int(binary.sum()),
        "baseline_modeled": int(primary.sum()),
        "baseline_positive": int((primary & positive).sum()),
        "modeled_positive": int((primary & positive).sum()),
        "baseline_negative": int((primary & state.eq("NEGATIVE")).sum()),
        "baseline_excluded": int((binary & ~primary).sum()),
        "baseline_excluded_positive": int(baseline_positive_excluded.sum()),
        "excluded_positive": int(baseline_positive_excluded.sum()),
        "baseline_excluded_positive_af": int((baseline_positive_excluded & _bool_series(frame, "af_flag")).sum()),
        "af_positive": int((baseline_positive_excluded & _bool_series(frame, "af_flag")).sum()),
        "baseline_excluded_positive_no_holter": int((baseline_positive_excluded & ~_bool_series(frame, "has_holter")).sum()),
        "no_holter_positive": int((baseline_positive_excluded & ~_bool_series(frame, "has_holter")).sum()),
        "rhythm_safe_modeled": int(rhythm.sum()),
        "rhythm_safe_positive": int((rhythm & positive).sum()),
        "rhythm_safe_negative": int((rhythm & state.eq("NEGATIVE")).sum()),
        "rhythm_safe_excluded_positive": int(rhythm_positive_excluded.sum()),
        "rhythm_safe_recovered_positive_af": int((rhythm & positive & _bool_series(frame, "af_flag")).sum()),
        "rhythm_safe_recovered_positive_no_holter": int((rhythm & positive & ~_bool_series(frame, "has_holter")).sum()),
    }
    expected = {
        "full_positive": 38,
        "baseline_positive": 27,
        "baseline_excluded_positive": 11,
        "baseline_excluded_positive_af": 10,
        "baseline_excluded_positive_no_holter": 1,
    }
    invariant = all(counts[key] == value for key, value in expected.items())
    # The official compact cohort has 992 rows.  Synthetic unit fixtures are
    # deliberately allowed to exercise schema/mask behavior without having to
    # reproduce the official counts; production-sized cohorts fail closed.
    if official_strict and int(horizon_days) == 365 and len(frame) >= 900 and not invariant:
        observed = {key: counts[key] for key in expected}
        raise AssertionError(f"365-day baseline invariant failed: expected {expected}, observed {observed}")

    pvc_audit: dict[str, Any] = {
        "PVC_CONTINUOUS_UNAVAILABLE": True,
        "pvc_continuous_unavailable": True,
        "pvc_continuous_unavailable": True,
        "reason": PVC_UNAVAILABLE_REASON,
        "derived_pvc_burden_used": False,
        "source_feature_allowed": PVC_SOURCE_FEATURE,
        "fold_local_imputation_required": True,
    }
    audit: dict[str, Any] = {
        "horizon_days": int(horizon_days),
        "baseline_365_invariant_pass": bool(invariant) if int(horizon_days) == 365 else None,
        "baseline_365_expected": expected,
        "baseline_365": counts,
        "rhythm_safe_recovery": {
            "positive_af_included": counts["rhythm_safe_recovered_positive_af"],
            "positive_no_holter_included": counts["rhythm_safe_recovered_positive_no_holter"],
            "positive_excluded": counts["rhythm_safe_excluded_positive"],
        },
        "pvc": pvc_audit,
        # Keep the explicit top-level spelling easy to assert in reports.
        "PVC_CONTINUOUS_UNAVAILABLE": True,
    }
    manifest: dict[str, Any] = {
        "counts": counts,
        "baseline": {
            "mask": primary,
            "positive_ids": frame.loc[primary & positive, "patient_id"].astype(str).tolist(),
            "excluded_positive_ids": frame.loc[baseline_positive_excluded, "patient_id"].astype(str).tolist(),
        },
        "rhythm_safe": {
            "mask": rhythm,
            "positive_ids": frame.loc[rhythm & positive, "patient_id"].astype(str).tolist(),
            "recovered_af_positive_ids": frame.loc[rhythm & positive & _bool_series(frame, "af_flag"), "patient_id"].astype(str).tolist(),
        },
        "table": frame.loc[:, [
            column
            for column in (
                "patient_id",
                "endpoint_state",
                "label",
                "primary_sinus_hrv_eligible",
                "af_flag",
                "has_holter",
                "processed_holter",
                "n_windows_successful",
                "pvc_count_24h",
                "pvc_denominator_beats",
                "pvc_burden",
                "pvc_burden_status",
                "high_pvc_burden",
                "high_pvc_flag",
            )
            if column in frame.columns
        ]].copy(),
    }
    return manifest, audit


@dataclass
class OptimizationBundle:
    """Prepared frame and fixed contract metadata for one candidate."""

    frame: pd.DataFrame
    feature_cols: list[str]
    model: str
    profile: str
    candidate: str
    population_manifest: dict[str, Any]
    audit: dict[str, Any]

    @property
    def population(self) -> dict[str, Any]:
        return self.population_manifest

    @property
    def population_counts(self) -> dict[str, Any]:
        return dict(self.population_manifest.get("counts", {}))

    @property
    def population_manifest_table(self) -> pd.DataFrame:
        return self.population_manifest["table"].copy(deep=True)


ModelOptimizationBundle = OptimizationBundle


def prepare_optimization_bundle(
    features: pd.DataFrame | Path | str,
    subjects: pd.DataFrame | Path | str,
    *,
    candidate: str = "B0",
    horizon_days: int = 365,
    feature_allowlist: Sequence[str] | None = None,
    custom_feature_allowlist: Sequence[str] | None = None,
    oof_metadata: pd.DataFrame | None = None,
    strict_official_audit: bool = True,
) -> OptimizationBundle:
    """Build one deterministic optimization frame and its audit bundle."""

    if int(horizon_days) not in {90, 180, 365, 730}:
        raise ValueError("optimization endpoint must be one of 90, 180, 365, 730 days")
    spec = candidate_spec(candidate)
    selected_allowlist = feature_allowlist if feature_allowlist is not None else custom_feature_allowlist
    requested_cols = list(spec.feature_cols) if selected_allowlist is None else list(selected_allowlist)
    if PVC_SOURCE_FEATURE in requested_cols and not spec.include_pvc:
        raise ValueError("pvc_count_24h is reserved for candidate P2")
    validate_feature_allowlist(requested_cols, profile=spec.profile)

    feature_frame = _normalise_ids(_as_frame(features, "features"), "features")
    subject_frame = _normalise_ids(_as_frame(subjects, "subjects"), "subjects")
    if spec.include_pvc and PVC_SOURCE_FEATURE not in subject_frame.columns:
        raise ValueError("candidate P2 requires source subjects.pvc_count_24h")
    endpoint = _endpoint_table(subject_frame, int(horizon_days))
    endpoint = _normalise_ids(endpoint, "endpoint")
    enriched = _coalesce_status(endpoint, feature_frame, subject_frame)
    enriched = _add_derived_feature_columns(enriched, requested_cols)

    # Merge only selected source feature columns.  In particular, do not use
    # feature-frame label/endpoint columns: endpoint labels come from above.
    source = _source_feature_frame(feature_frame, [column for column in requested_cols if column != "af_flag"])
    source = source.set_index("patient_id", drop=False)
    feature_values: dict[str, Any] = {}
    for column in requested_cols:
        if column == "af_flag":
            feature_values[column] = _bool_series(enriched, "af_flag").to_numpy()
        elif column == PVC_SOURCE_FEATURE:
            # Reassert source priority after feature joins; P2 must use the
            # official count and preserve missingness for fold-local imputation.
            feature_values[column] = enriched[PVC_SOURCE_FEATURE].to_numpy()
        else:
            feature_values[column] = source[column].reindex(enriched["patient_id"]).to_numpy()
    enriched = enriched.drop(columns=[column for column in requested_cols if column in enriched.columns])
    enriched = pd.concat([enriched, pd.DataFrame(feature_values, index=enriched.index)], axis=1)
    enriched = _add_derived_feature_columns(enriched, requested_cols)
    # Keep the audit manifest's label synchronized with endpoint state while
    # never consulting a feature-table label.
    enriched["label"] = enriched["endpoint_state"].map({"POSITIVE": 1, "NEGATIVE": 0}).astype("Int64")

    # Recompute the candidate masks with the candidate's exact safe columns.
    baseline_mask = _population_mask(enriched, "primary", requested_cols)
    rhythm_mask = _population_mask(enriched, "rhythm_safe", requested_cols)
    population_key = spec.population
    selected_mask = baseline_mask if population_key == "primary" else rhythm_mask
    selected = enriched.loc[selected_mask].copy()
    if selected.empty:
        raise ValueError(f"candidate {spec.name} has no evaluable population rows")
    for column in requested_cols:
        if column != "af_flag":
            selected[column] = pd.to_numeric(selected[column], errors="coerce").astype("float64")
    generated_label = selected["endpoint_state"].map({"POSITIVE": 1, "NEGATIVE": 0}).astype(int).rename("label")
    selected = pd.concat([selected.drop(columns=["label"], errors="ignore"), generated_label], axis=1)

    # Preserve endpoint and OOF metadata in a stable, inspectable order.
    metadata = [
        "patient_id",
        "label",
        *ENDPOINT_METADATA_COLUMNS,
        *[column for column in STATUS_COLUMNS if column in selected.columns],
    ]
    if oof_metadata is not None:
        oof = _normalise_ids(oof_metadata, "oof_metadata")
        keep = [column for column in OOF_METADATA_COLUMNS if column in oof.columns]
        if keep:
            selected = selected.merge(oof.loc[:, ["patient_id", *keep]], on="patient_id", how="left", validate="one_to_one", suffixes=("", "_oof"))
            metadata.extend([column for column in keep if column not in metadata and column in selected.columns])
    metadata = [column for column in metadata if column in selected.columns]
    result = selected.loc[:, [*dict.fromkeys([*metadata, *requested_cols])]].sort_values("patient_id", kind="stable").reset_index(drop=True)
    if result["patient_id"].duplicated().any() or result["label"].nunique() != 2:
        raise ValueError("optimization frame must contain unique patients and both binary labels")

    # Audit against the full endpoint/status frame before candidate filtering.
    manifest, audit = _manifest_and_audit(
        enriched,
        horizon_days=int(horizon_days),
        official_strict=strict_official_audit,
    )
    audit["candidate_population"] = {
        "candidate": spec.name,
        "profile": spec.profile,
        "model": spec.model,
        "population": spec.population,
        "rows": int(len(result)),
        "positive": int(result["label"].sum()),
        "negative": int((result["label"] == 0).sum()),
        "excluded": int(len(enriched) - len(result)),
    }
    table = manifest["table"].copy(deep=True)
    included_ids = set(result["patient_id"].astype(str))
    table["included"] = table["patient_id"].astype(str).isin(included_ids)
    reason = pd.Series("ELIGIBLE", index=enriched.index, dtype="string")
    nonbinary = ~enriched["endpoint_state"].astype("string").isin(["POSITIVE", "NEGATIVE"])
    reason.loc[nonbinary] = enriched.loc[nonbinary, "endpoint_state"].astype("string")
    if spec.population == "primary":
        excluded_binary = ~nonbinary & ~baseline_mask
        stored_reason = enriched.get("primary_sinus_hrv_reason", pd.Series("INELIGIBLE", index=enriched.index)).astype("string")
        reason.loc[excluded_binary] = stored_reason.loc[excluded_binary].fillna("INELIGIBLE")
    else:
        excluded_binary = ~nonbinary & ~rhythm_mask
        reason.loc[excluded_binary & ~_bool_series(enriched, "has_holter")] = "NO_HOLTER"
        reason.loc[excluded_binary & _bool_series(enriched, "has_holter") & ~_bool_series(enriched, "processed_holter")] = "PROCESSING_FAILED"
        reason.loc[excluded_binary & _bool_series(enriched, "processed_holter") & _numeric_series(enriched, "n_windows_successful", 0).lt(1)] = "NO_SUCCESSFUL_WINDOWS"
        reason.loc[excluded_binary & reason.eq("ELIGIBLE")] = "NO_RHYTHM_SAFE_FEATURES"
    reason_by_id = pd.Series(reason.to_numpy(), index=enriched["patient_id"].astype(str)).to_dict()
    table["exclusion_reason"] = table["patient_id"].astype(str).map(reason_by_id)
    manifest["table"] = table
    audit["candidate_population"]["exclusion_reasons"] = {
        str(key): int(value)
        for key, value in table.loc[~table["included"], "exclusion_reason"].value_counts(dropna=False).items()
    }
    manifest["candidate"] = spec.name
    manifest["profile"] = spec.profile
    manifest["model"] = spec.model
    manifest["selected_population"] = spec.population
    manifest["selected_ids"] = result["patient_id"].astype(str).tolist()
    return OptimizationBundle(
        frame=result,
        feature_cols=requested_cols,
        model=spec.model,
        profile=spec.profile,
        candidate=spec.name,
        population_manifest=manifest,
        audit=audit,
    )


# Friendly names used by notebooks and tests.
prepare_model_optimization = prepare_optimization_bundle
prepare_candidate = prepare_optimization_bundle
prepare_candidate_bundle = prepare_optimization_bundle
prepare_candidate_frame = prepare_optimization_bundle
build_optimization_frame = prepare_optimization_bundle
build_candidate_frame = prepare_optimization_bundle
get_candidate_spec = candidate_spec


__all__ = [
    "RHYTHM_SAFE_BASES",
    "RHYTHM_SAFE_FEATURES",
    "P50_SUFFIX",
    "ROBUST_SUFFIX",
    "ALL20_100_PROFILE",
    "P50_20_PROFILE",
    "ROBUST40_PROFILE",
    "RHYTHM_SAFE_PROFILE",
    "FROZEN_100_PROFILE",
    "PROFILE_ALL20_100",
    "PROFILE_P50_20",
    "PROFILE_ROBUST40",
    "PROFILE_RHYTHM_SAFE",
    "PROFILE_FROZEN_100",
    "PVC_SOURCE_FEATURE",
    "PVC_CONTINUOUS_UNAVAILABLE",
    "PVC_UNAVAILABLE_REASON",
    "CandidateSpec",
    "OptimizationBundle",
    "ModelOptimizationBundle",
    "CANDIDATES",
    "CANDIDATE_SPECS",
    "CANDIDATE_REGISTRY",
    "FEATURE_PROFILES",
    "PROFILE_FEATURES",
    "candidate_spec",
    "candidate_registry",
    "candidate_feature_columns",
    "profile_feature_columns",
    "validate_feature_allowlist",
    "prepare_optimization_bundle",
    "prepare_model_optimization",
    "prepare_candidate",
    "prepare_candidate_bundle",
    "prepare_candidate_frame",
    "build_optimization_frame",
    "build_candidate_frame",
    "get_candidate_spec",
]
