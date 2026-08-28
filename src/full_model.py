"""Shared data, feature-profile, and estimator definitions for P4-C.

This module intentionally consumes already-built parquet tables.  It never
opens MUSIC waveform files and never derives labels from the feature table.
Labels are rebuilt from the Phase 2 endpoint source through
``src.endpoints.build_endpoint`` before the binary modeling frame is formed.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import yaml
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import StandardScaler

from .endpoints import ALLOWED_STATES, build_endpoint
from .legacy_aggregation import AGGREGATED_FEATURE_NAMES, AGGREGATION_SUFFIXES
from .legacy_features import FEATURE_NAMES


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FEATURE_CONFIG = REPO_ROOT / "config" / "features_v2.yaml"
DEFAULT_FEATURE_PATH = REPO_ROOT / "data" / "features" / "full_5min" / "patient_features.parquet"
DEFAULT_SUBJECTS_PATH = REPO_ROOT / "data" / "cohort" / "subjects.parquet"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "data" / "validation" / "full_model"

DEFAULT_RANDOM_STATE = 42
DEFAULT_OUTER_FOLDS = 5
DEFAULT_INNER_FOLDS = 3
DEFAULT_EXTRA_TREES_N_ITER = 24
DEFAULT_TARGET_SPECIFICITY = 0.70
DEFAULT_BOOTSTRAP_RESAMPLES = 2000
DEFAULT_N_JOBS = 1

MODEL_ALIASES: dict[str, str] = {
    "extra_trees": "extratrees",
    "extra-trees": "extratrees",
    "extra trees": "extratrees",
    "et": "extratrees",
    "l2_logistic": "logistic",
    "l2-logistic": "logistic",
    "logistic_l2": "logistic",
    "logit": "logistic",
    "prevalence": "dummy",
    "dummy_prevalence": "dummy",
    "elasticnet-selected": "elasticnet_selected",
    "elasticnet selected": "elasticnet_selected",
    "elastic_net": "elasticnet",
    "elastic-net": "elasticnet",
    "elastic_net_selected": "elasticnet_selected",
    "elastic-net-selected": "elasticnet_selected",
    "extra_trees_regularized": "extratrees_regularized",
    "extra-trees-regularized": "extratrees_regularized",
    "extra trees regularized": "extratrees_regularized",
    "et_regularized": "extratrees_regularized",
}

# Public frozen column constants are convenient for notebooks and make the
# exact 100-column contract explicit to callers without requiring a config
# parser round trip.  ``model_feature_names`` remains the source of truth for
# profile-specific selection.
ALL20_FEATURE_NAMES: tuple[str, ...] = tuple(AGGREGATED_FEATURE_NAMES)


def canonical_model_name(model: str) -> str:
    value = str(model).strip().lower()
    value = MODEL_ALIASES.get(value, value)
    if value not in {"extratrees", "logistic", "dummy", "elasticnet", "elasticnet_selected", "extratrees_regularized"}:
        raise ValueError(
            "model must be one of extratrees, logistic, dummy, elasticnet, "
            "elasticnet_selected, extratrees_regularized"
        )
    return value


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    return value if isinstance(value, dict) else {}


def _manifest_entries(config_path: Path | None = None) -> list[dict[str, Any]]:
    """Read flexible features_v2 manifest spellings used by project builds."""

    path = Path(config_path) if config_path is not None else DEFAULT_FEATURE_CONFIG
    raw = _load_yaml(path)
    entries: Any = raw.get("features", raw.get("base_features", []))
    if isinstance(entries, dict):
        entries = [dict(value, name=key) if isinstance(value, dict) else {"name": key, "category": value} for key, value in entries.items()]
    if not isinstance(entries, list):
        return []
    output: list[dict[str, Any]] = []
    for item in entries:
        if not isinstance(item, dict):
            continue
        # Feature manifests in earlier phases used feature_name; newer ones
        # commonly use name or base_name.  Keep all keys for category lookup.
        name = item.get("base_name", item.get("feature_name", item.get("name")))
        if name is None and item.get("feature_id") in {f"feature_{i:02d}" for i in range(1, 21)}:
            name = item.get("feature_id")
        if name is None:
            continue
        row = dict(item)
        row["_name"] = str(name)
        row["_category"] = str(item.get("category", item.get("feature_category", ""))).upper()
        output.append(row)
    return output


def _normalise_base_name(name: str, position: int | None = None) -> str | None:
    text = str(name).strip()
    if text in FEATURE_NAMES:
        return text
    # A manifest may identify the frozen feature by feature_01 rather than by
    # the generated column prefix.  The order is frozen in legacy_features.py.
    if text.startswith("feature_"):
        try:
            index = int(text.rsplit("_", 1)[1]) - 1
        except ValueError:
            index = -1
        if 0 <= index < len(FEATURE_NAMES):
            return FEATURE_NAMES[index]
    # Some manifests accidentally include an aggregate suffix in the name;
    # stripping it is safe because profiles always regenerate the five fixed
    # aggregate columns from the base name.
    for suffix in AGGREGATION_SUFFIXES:
        marker = f"_{suffix}"
        if text.endswith(marker) and text[: -len(marker)] in FEATURE_NAMES:
            return text[: -len(marker)]
    if position is not None and 0 <= position < len(FEATURE_NAMES):
        return FEATURE_NAMES[position]
    return None


def feature_manifest(config_path: Path | str | None = None) -> list[dict[str, Any]]:
    """Return the twenty frozen base features with their manifest categories."""

    entries = _manifest_entries(Path(config_path) if config_path is not None else None)
    by_name: dict[str, dict[str, Any]] = {}
    for position, item in enumerate(entries):
        name = _normalise_base_name(item["_name"], position)
        if name and name not in by_name:
            by_name[name] = {"name": name, "category": item.get("_category", "")}
    # Legacy config is a useful fallback until features_v2 is built.  It also
    # keeps this module usable in a clean checkout with only Phase 3 artifacts.
    if not by_name:
        legacy_path = REPO_ROOT / "config" / "legacy_features.yaml"
        for position, item in enumerate(_manifest_entries(legacy_path)):
            name = _normalise_base_name(item["_name"], position)
            if name and name not in by_name:
                by_name[name] = {"name": name, "category": item.get("_category", "")}
    result: list[dict[str, Any]] = []
    for position, name in enumerate(FEATURE_NAMES):
        row = by_name.get(name, {"name": name, "category": ""})
        result.append({"name": name, "category": str(row.get("category", "")).upper()})
    return result


def base_feature_names(
    profile: str = "all20",
    *,
    feature_config_path: Path | str | None = None,
) -> list[str]:
    """Return frozen base names for ``all20`` or ``physiology_only``."""

    profile_key = str(profile).strip().lower().replace("-", "_")
    if profile_key in {"all", "all_20", "full", "all20", "full20", "full_20_feature"}:
        profile_key = "all20"
    elif profile_key in {"physiology", "physiology_only", "physiology_only_feature", "physio"}:
        profile_key = "physiology_only"
    else:
        raise ValueError("profile must be all20 or physiology_only")
    names = [item["name"] for item in feature_manifest(feature_config_path)]
    if len(names) != 20 or len(set(names)) != 20:
        raise ValueError("features_v2 must describe exactly twenty unique base features")
    if profile_key == "physiology_only":
        signal_quality = {
            item["name"] for item in feature_manifest(feature_config_path) if item["category"] == "SIGNAL_QUALITY"
        }
        names = [name for name in names if name not in signal_quality]
    return names


def model_feature_names(
    profile: str = "all20",
    *,
    feature_config_path: Path | str | None = None,
) -> list[str]:
    """Return the exact generated aggregate columns for a feature profile."""

    bases = base_feature_names(profile, feature_config_path=feature_config_path)
    return [f"{base}_{suffix}" for base in bases for suffix in AGGREGATION_SUFFIXES]


ALL_FEATURE_COLUMNS = list(ALL20_FEATURE_NAMES)


def get_model_feature_columns(
    frame: pd.DataFrame | None = None,
    profile: str = "all20",
    *,
    feature_cols: Sequence[str] | None = None,
    feature_config_path: Path | str | None = None,
) -> list[str]:
    """Return and, when given a frame, validate the profile's feature columns."""

    cols = list(feature_cols) if feature_cols is not None else model_feature_names(profile, feature_config_path=feature_config_path)
    # Explicit feature lists are valid custom profiles; the frame still
    # validates every requested column below.  The generated all20 profile
    # retains its frozen 100-column contract.
    if feature_cols is None:
        expected_count = 100 if str(profile).strip().lower().replace("-", "_") in {"all20", "all_20", "all", "full"} else len(cols)
    else:
        expected_count = len(cols)
    if not cols or len(cols) != expected_count or len(set(cols)) != len(cols):
        raise ValueError(f"model feature profile requires {expected_count} unique columns, got {len(cols)}")
    unsafe = {
        "patient_id",
        "label",
        "true_label",
        "binary_label_if_evaluable",
        "endpoint_state",
        "endpoint_horizon_days",
        "time_to_event",
        "event_type",
    }
    unsafe_requested = [column for column in cols if str(column) in unsafe]
    forbidden_tokens = ("patient_id", "label", "outcome", "endpoint", "cause", "death", "event", "followup", "prediction", "probability", "threshold")
    unsafe_requested.extend(
        column
        for column in cols
        if column not in unsafe_requested and any(token in str(column).lower() for token in forbidden_tokens)
    )
    if unsafe_requested:
        raise ValueError(f"model feature columns cannot include identifiers or outcome fields: {unsafe_requested}")
    if frame is not None:
        missing = [column for column in cols if column not in frame.columns]
        if missing:
            raise ValueError(f"model feature columns missing: {missing}")
    return cols


def profile_feature_columns(
    profile: str = "all20",
    *,
    feature_config_path: Path | str | None = None,
) -> list[str]:
    """Alias returning profile columns without requiring a frame."""

    return model_feature_names(profile, feature_config_path=feature_config_path)


def _as_frame(value: pd.DataFrame | Path | str, *, table_name: str) -> pd.DataFrame:
    if isinstance(value, pd.DataFrame):
        return value.copy(deep=True)
    path = Path(value)
    if not path.is_file():
        raise FileNotFoundError(f"{table_name} parquet does not exist: {path}")
    return pd.read_parquet(path)


def _endpoint_table(subjects_or_endpoints: pd.DataFrame, horizon_days: int) -> pd.DataFrame:
    required = {"patient_id", "followup_days", "cause_of_death_raw", "event_source_valid"}
    if required <= set(subjects_or_endpoints.columns):
        return build_endpoint(subjects_or_endpoints, int(horizon_days))
    # Supporting a pre-built endpoint table is useful for pure unit tests and
    # for downstream consumers that persist the dynamic endpoint separately.
    endpoint_columns = {"patient_id", "endpoint_state", "binary_label_if_evaluable"}
    if endpoint_columns <= set(subjects_or_endpoints.columns):
        endpoint = subjects_or_endpoints.copy(deep=True)
        if endpoint["patient_id"].duplicated().any():
            raise ValueError("endpoint table must contain one row per patient")
        if "endpoint_horizon_days" not in endpoint.columns:
            endpoint["endpoint_horizon_days"] = int(horizon_days)
        return endpoint
    missing = sorted(required - set(subjects_or_endpoints.columns))
    raise ValueError(f"subjects must contain endpoint source columns or a built endpoint table; missing {missing}")


def prepare_model_frame(
    features: pd.DataFrame | Path | str,
    subjects_or_endpoints: pd.DataFrame | Path | str,
    *,
    horizon_days: int = 365,
    profile: str = "all20",
    feature_config_path: Path | str | None = None,
) -> pd.DataFrame:
    """Merge patient features with a dynamic endpoint and retain binary rows."""

    feature_frame = _as_frame(features, table_name="features")
    subject_frame = _as_frame(subjects_or_endpoints, table_name="subjects")
    if "patient_id" not in feature_frame.columns:
        raise ValueError("features must contain patient_id")
    if feature_frame["patient_id"].isna().any() or feature_frame["patient_id"].duplicated().any():
        raise ValueError("features must contain one non-null row per patient")
    cols = get_model_feature_columns(feature_frame, profile, feature_config_path=feature_config_path)
    endpoint = _endpoint_table(subject_frame, int(horizon_days))
    if "patient_id" not in endpoint.columns or endpoint["patient_id"].duplicated().any():
        raise ValueError("endpoint must contain one row per patient")
    endpoint = endpoint.loc[endpoint["endpoint_state"].isin(["POSITIVE", "NEGATIVE"])].copy()
    endpoint["label"] = pd.to_numeric(endpoint["binary_label_if_evaluable"], errors="coerce")
    endpoint = endpoint.loc[endpoint["label"].isin([0, 1])].copy()
    feature_frame = feature_frame.copy()
    feature_frame["patient_id"] = feature_frame["patient_id"].astype("string")
    # The primary binary analysis is an explicit runtime view.  Full cohort
    # facts remain in subjects/patient_features; AF, high-PVC, failed and
    # no-QC patients are excluded here only when the status column is present.
    if "primary_sinus_hrv_eligible" in feature_frame.columns:
        feature_frame = feature_frame.loc[
            feature_frame["primary_sinus_hrv_eligible"].fillna(False).astype(bool)
        ].copy()
    endpoint["patient_id"] = endpoint["patient_id"].astype("string")
    merged = endpoint.merge(
        feature_frame[["patient_id", *cols]],
        on="patient_id",
        how="inner",
        validate="one_to_one",
        sort=False,
    )
    if merged.empty:
        raise ValueError("no evaluable endpoint patients have feature rows")
    for column in cols:
        merged[column] = pd.to_numeric(merged[column], errors="coerce").astype("float64")
    merged["label"] = merged["label"].astype(int)
    preferred = ["patient_id", "label", "endpoint_state", "endpoint_horizon_days", "time_to_event", "event_type"]
    metadata = [column for column in preferred if column in merged.columns]
    result = merged[metadata + cols].sort_values("patient_id", kind="stable").reset_index(drop=True)
    if result["patient_id"].duplicated().any():
        raise AssertionError("prepared model frame contains duplicate patient rows")
    if result["label"].nunique() != 2:
        raise ValueError("binary model requires both POSITIVE and NEGATIVE patients")
    return result


load_validation_frame = prepare_model_frame
load_model_frame = prepare_model_frame
build_validation_frame = prepare_model_frame
prepare_dataset = prepare_model_frame
load_data = prepare_model_frame


def build_model(
    model: str,
    feature_cols: Sequence[str],
    *,
    seed: int = DEFAULT_RANDOM_STATE,
    n_jobs: int = DEFAULT_N_JOBS,
    estimator_params: Mapping[str, Any] | None = None,
) -> Pipeline:
    """Build one estimator with all preprocessing represented in a pipeline."""

    kind = canonical_model_name(model)
    cols = list(feature_cols)
    if not cols:
        raise ValueError("at least one feature column is required")
    steps: list[tuple[str, Any]] = []
    if kind in {"logistic", "elasticnet", "elasticnet_selected"}:
        preprocessor = Pipeline(
            steps=[
                (
                    "imputer",
                    SimpleImputer(
                        strategy="median",
                        keep_empty_features=True,
                        add_indicator=kind in {"elasticnet", "elasticnet_selected"},
                    ),
                ),
                ("scale", StandardScaler()),
            ]
        )
        if kind == "elasticnet_selected":
            preprocessor.steps.append(("select", SelectKBest(score_func=f_classif, k="all")))
            estimator = LogisticRegression(
                solver="saga",
                penalty="elasticnet",
                max_iter=10000,
                random_state=int(seed),
            )
        elif kind == "elasticnet":
            estimator = LogisticRegression(
                solver="saga",
                penalty="elasticnet",
                max_iter=10000,
                random_state=int(seed),
            )
        else:
            estimator = None
        # sklearn 1.8 represents L2 with l1_ratio=0 and warns when the
        # legacy explicit ``penalty='l2'`` spelling is fitted.  Keep the
        # compatibility branch for older versions while preserving the same
        # L2 objective on both APIs.
        if kind == "logistic" and LogisticRegression().get_params().get("penalty") == "deprecated":
            estimator = LogisticRegression(
                l1_ratio=0.0,
                solver="lbfgs",
                max_iter=2000,
                random_state=int(seed),
            )
        elif kind == "logistic":
            estimator = LogisticRegression(
                penalty="l2",
                solver="liblinear",
                max_iter=2000,
                random_state=int(seed),
            )
    elif kind == "dummy":
        preprocessor = SimpleImputer(strategy="median", keep_empty_features=True)
        estimator = DummyClassifier(strategy="prior")
    elif kind == "extratrees_regularized":
        preprocessor = SimpleImputer(strategy="median", keep_empty_features=True)
        estimator = ExtraTreesClassifier(
            n_estimators=500,
            max_depth=None,
            min_samples_leaf=2,
            min_samples_split=4,
            max_features="sqrt",
            class_weight="balanced",
            criterion="gini",
            bootstrap=False,
            random_state=int(seed),
            n_jobs=int(n_jobs),
        )
    else:
        preprocessor = SimpleImputer(strategy="median", keep_empty_features=True)
        estimator = ExtraTreesClassifier(
            n_estimators=800,
            max_depth=None,
            min_samples_leaf=2,
            min_samples_split=6,
            max_features="sqrt",
            class_weight="balanced",
            random_state=int(seed),
            n_jobs=int(n_jobs),
        )
    pipeline = Pipeline(steps=[("pre", preprocessor), ("clf", estimator)])
    if estimator_params:
        params = dict(estimator_params)
        # Accept both the public clf__ names and the older classifier__ spelling.
        params = {("clf" + key[len("classifier") :]) if key.startswith("classifier__") else key: value for key, value in params.items()}
        pipeline_params = {key: value for key, value in params.items() if key.startswith(("clf__", "pre__"))}
        direct = {key[len("clf__") :]: value for key, value in params.items() if key.startswith("clf__")}
        nested = {
            key: value
            for key, value in params.items()
            if not key.startswith(("clf__", "pre__"))
        }
        estimator.set_params(**direct, **nested)
        if pipeline_params:
            pipeline.set_params(**pipeline_params)
    return pipeline


def get_param_distributions(model: str = "extratrees") -> dict[str, list[Any]]:
    """Return the fixed bounded search space for a supported model family."""

    kind = canonical_model_name(model)
    if kind == "extratrees":
        # Keep the legacy bounded search exactly, while allowing n_iter=1 for
        # CI smoke runs through the nested-CV argument.
        return {
            "clf__n_estimators": [400, 800, 1200, 1600],
            "clf__max_depth": [None, 4, 6, 8, 10, 14],
            "clf__min_samples_split": [2, 4, 6, 8, 10],
            "clf__min_samples_leaf": [1, 2, 3, 4],
            "clf__max_features": ["sqrt", "log2", 0.5, 0.7],
            "clf__class_weight": ["balanced", {0: 1, 1: 2}, {0: 1, 1: 3}],
        }
    if kind == "logistic":
        return {"clf__C": [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]}
    if kind == "elasticnet":
        return {
            "clf__C": [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3],
            "clf__l1_ratio": [0, 0.1, 0.25, 0.5, 0.75, 1],
            "clf__class_weight": [None, "balanced"],
        }
    if kind == "elasticnet_selected":
        return {
            "clf__C": [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3],
            "clf__l1_ratio": [0, 0.1, 0.25, 0.5, 0.75, 1],
            "clf__class_weight": [None, "balanced"],
            "pre__select__k": [8, 12, 20, 30],
        }
    if kind == "extratrees_regularized":
        return {
            "clf__n_estimators": [300, 500, 800],
            "clf__max_depth": [3, 5, 8, 12, None],
            "clf__min_samples_leaf": [2, 4, 6, 10, 15],
            "clf__min_samples_split": [4, 8, 12, 20],
            "clf__max_features": ["sqrt", 0.2, 0.3, 0.5],
            "clf__class_weight": ["balanced", "balanced_subsample"],
            "clf__criterion": ["gini", "log_loss"],
            "clf__bootstrap": [False, True],
        }
    return {}


def estimator_feature_columns(model_frame: pd.DataFrame, profile: str = "all20", *, feature_config_path: Path | str | None = None) -> list[str]:
    """Convenience wrapper used by CLI and tests."""

    return get_model_feature_columns(model_frame, profile, feature_config_path=feature_config_path)


__all__ = [
    "REPO_ROOT",
    "DEFAULT_FEATURE_CONFIG",
    "DEFAULT_FEATURE_PATH",
    "DEFAULT_SUBJECTS_PATH",
    "DEFAULT_OUTPUT_DIR",
    "DEFAULT_RANDOM_STATE",
    "DEFAULT_OUTER_FOLDS",
    "DEFAULT_INNER_FOLDS",
    "DEFAULT_EXTRA_TREES_N_ITER",
    "DEFAULT_TARGET_SPECIFICITY",
    "DEFAULT_BOOTSTRAP_RESAMPLES",
    "DEFAULT_N_JOBS",
    "MODEL_ALIASES",
    "ALL20_FEATURE_NAMES",
    "ALL_FEATURE_COLUMNS",
    "canonical_model_name",
    "feature_manifest",
    "base_feature_names",
    "model_feature_names",
    "get_model_feature_columns",
    "profile_feature_columns",
    "prepare_model_frame",
    "build_validation_frame",
    "prepare_dataset",
    "load_data",
    "load_validation_frame",
    "load_model_frame",
    "build_model",
    "get_param_distributions",
    "estimator_feature_columns",
]
