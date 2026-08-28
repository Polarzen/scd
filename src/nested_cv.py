"""Strict patient-level nested cross-validation for P4-C.

The outer test partition is touched only once per patient.  Search and
threshold calibration are confined to the corresponding outer training
partition; the threshold is selected from inner out-of-fold probabilities.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, StratifiedKFold, cross_val_predict

from .full_model import (
    DEFAULT_BOOTSTRAP_RESAMPLES,
    DEFAULT_EXTRA_TREES_N_ITER,
    DEFAULT_INNER_FOLDS,
    DEFAULT_N_JOBS,
    DEFAULT_OUTER_FOLDS,
    DEFAULT_RANDOM_STATE,
    DEFAULT_TARGET_SPECIFICITY,
    build_model,
    canonical_model_name,
    get_model_feature_columns,
    get_param_distributions,
)
from .metrics import (
    METRIC_NAMES,
    bootstrap_metrics,
    compute_metrics,
    metric_aliases,
    select_threshold_target_specificity,
)


def _jsonable(value: Any) -> Any:
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


@dataclass
class NestedCVResult:
    """Machine-readable outputs from one model/seed nested evaluation."""

    oof: pd.DataFrame
    folds: pd.DataFrame
    summary: dict[str, Any]
    model: str = "extratrees"
    seed: int = DEFAULT_RANDOM_STATE

    @property
    def oof_predictions(self) -> pd.DataFrame:
        return self.oof

    @property
    def fold_metrics(self) -> pd.DataFrame:
        return self.folds


def make_outer_splits(
    y: Sequence[int] | np.ndarray,
    *,
    outer_folds: int = DEFAULT_OUTER_FOLDS,
    seed: int = DEFAULT_RANDOM_STATE,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Create deterministic shuffled patient indices for one seed."""

    y_arr = np.asarray(y, dtype=int).reshape(-1)
    if int(outer_folds) < 2:
        raise ValueError("outer_folds must be at least two")
    if y_arr.size == 0 or not np.isin(y_arr, [0, 1]).all():
        raise ValueError("y must be a non-empty binary vector")
    if np.bincount(y_arr, minlength=2).min() < int(outer_folds):
        raise ValueError("each class needs at least outer_folds patients")
    splitter = StratifiedKFold(n_splits=int(outer_folds), shuffle=True, random_state=int(seed))
    return [(train.astype(int), test.astype(int)) for train, test in splitter.split(np.zeros(len(y_arr)), y_arr)]


def _validate_outer_splits(
    splits: Sequence[tuple[Sequence[int], Sequence[int]]],
    n_rows: int,
    y: np.ndarray,
) -> list[tuple[np.ndarray, np.ndarray]]:
    if len(splits) < 2:
        raise ValueError("outer_splits must contain at least two folds")
    seen = np.zeros(n_rows, dtype=int)
    normalized: list[tuple[np.ndarray, np.ndarray]] = []
    all_indices = set(range(n_rows))
    for train_raw, test_raw in splits:
        train = np.asarray(train_raw, dtype=int).reshape(-1)
        test = np.asarray(test_raw, dtype=int).reshape(-1)
        if train.size == 0 or test.size == 0:
            raise ValueError("outer folds must have non-empty train and test partitions")
        if len(set(train.tolist())) != train.size or len(set(test.tolist())) != test.size:
            raise ValueError("outer fold indices must be unique")
        if not set(train.tolist()) <= all_indices or not set(test.tolist()) <= all_indices:
            raise ValueError("outer fold index is out of range")
        if set(train.tolist()) & set(test.tolist()):
            raise ValueError("outer train and test partitions overlap")
        if np.bincount(y[train], minlength=2).min() < 1 or np.bincount(y[test], minlength=2).min() < 1:
            raise ValueError("each outer train/test partition must contain both classes")
        seen[test] += 1
        normalized.append((train, test))
    if not np.all(seen == 1):
        raise ValueError("outer test partitions must cover each patient exactly once")
    return normalized


def _coerce_frame(frame: pd.DataFrame, profile: str, feature_cols: Sequence[str] | None) -> tuple[pd.DataFrame, list[str], np.ndarray]:
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("frame must be a pandas DataFrame")
    if "patient_id" not in frame.columns:
        raise ValueError("frame must contain patient_id")
    work = frame.copy(deep=True)
    if "label" not in work.columns:
        if "binary_label_if_evaluable" in work.columns:
            work["label"] = pd.to_numeric(work["binary_label_if_evaluable"], errors="coerce")
        elif "true_label" in work.columns:
            work["label"] = pd.to_numeric(work["true_label"], errors="coerce")
        else:
            raise ValueError("frame must contain label or binary_label_if_evaluable")
    work["patient_id"] = work["patient_id"].astype("string")
    if work["patient_id"].isna().any() or work["patient_id"].duplicated().any():
        raise ValueError("nested model requires one non-null row per patient")
    work["label"] = pd.to_numeric(work["label"], errors="coerce")
    if work["label"].isna().any() or not work["label"].isin([0, 1]).all():
        raise ValueError("frame label must contain only binary non-null values")
    work = work.sort_values("patient_id", kind="stable").reset_index(drop=True)
    cols = get_model_feature_columns(work, profile, feature_cols=feature_cols)
    for column in cols:
        work[column] = pd.to_numeric(work[column], errors="coerce").astype("float64")
    y = work["label"].to_numpy(dtype=int)
    if np.unique(y).size != 2:
        raise ValueError("nested model requires both labels")
    return work, cols, y


def _fit_search(
    kind: str,
    x_train: pd.DataFrame,
    y_train: np.ndarray,
    feature_cols: Sequence[str],
    *,
    inner_folds: int,
    n_iter: int,
    seed: int,
    n_jobs: int,
    tune_n_jobs: int | None,
    param_distributions: Mapping[str, Sequence[Any]] | None,
    estimator_params: Mapping[str, Any] | None,
) -> tuple[Any, dict[str, Any], StratifiedKFold]:
    n_pos = int(np.sum(y_train == 1))
    n_neg = int(np.sum(y_train == 0))
    if min(n_pos, n_neg) < int(inner_folds):
        raise ValueError("inner stratified CV requires at least inner_folds patients per class")
    inner_cv = StratifiedKFold(n_splits=int(inner_folds), shuffle=True, random_state=int(seed))
    base = build_model(kind, feature_cols, seed=int(seed), n_jobs=int(n_jobs), estimator_params=estimator_params)
    search_space = dict(param_distributions) if param_distributions is not None else get_param_distributions(kind)
    search_jobs = int(n_jobs if tune_n_jobs is None else tune_n_jobs)
    if kind == "dummy" or not search_space:
        base.fit(x_train, y_train)
        return base, {}, inner_cv
    if kind == "logistic":
        search: Any = GridSearchCV(
            estimator=base,
            param_grid=search_space,
            scoring="average_precision",
            n_jobs=search_jobs,
            cv=inner_cv,
            refit=True,
            error_score="raise",
        )
    else:
        # RandomizedSearchCV is intentionally bounded by n_iter.  The default
        # 24 candidates preserves the old ExtraTrees reproduction; CI can pass
        # n_iter=1 without changing the formal default.
        search = RandomizedSearchCV(
            estimator=base,
            param_distributions=search_space,
            n_iter=max(1, int(n_iter)),
            scoring="average_precision",
            n_jobs=search_jobs,
            cv=inner_cv,
            refit=True,
            random_state=int(seed),
            error_score="raise",
        )
    search.fit(x_train, y_train)
    return search.best_estimator_, _jsonable(search.best_params_), inner_cv


def run_nested_cv(
    frame: pd.DataFrame,
    model: str = "extratrees",
    *,
    model_name: str | None = None,
    profile: str = "all20",
    feature_cols: Sequence[str] | None = None,
    seed: int = DEFAULT_RANDOM_STATE,
    random_state: int | None = None,
    outer_folds: int = DEFAULT_OUTER_FOLDS,
    inner_folds: int = DEFAULT_INNER_FOLDS,
    n_iter: int = DEFAULT_EXTRA_TREES_N_ITER,
    n_jobs: int = DEFAULT_N_JOBS,
    tune_n_jobs: int | None = None,
    target_specificity: float = DEFAULT_TARGET_SPECIFICITY,
    bootstrap_resamples: int = DEFAULT_BOOTSTRAP_RESAMPLES,
    bootstrap_seed: int | None = None,
    param_distributions: Mapping[str, Sequence[Any]] | None = None,
    estimator_params: Mapping[str, Any] | None = None,
    outer_splits: Sequence[tuple[Sequence[int], Sequence[int]]] | None = None,
) -> NestedCVResult:
    """Run one strict nested patient-level evaluation."""

    if random_state is not None:
        seed = int(random_state)
    if model_name is not None:
        model = model_name
    kind = canonical_model_name(model)
    work, cols, y = _coerce_frame(frame, profile, feature_cols)
    if int(outer_folds) < 2 or int(inner_folds) < 2:
        raise ValueError("outer_folds and inner_folds must be at least two")
    if int(n_iter) < 1:
        raise ValueError("n_iter must be positive")
    if outer_splits is None:
        splits = make_outer_splits(y, outer_folds=int(outer_folds), seed=int(seed))
    else:
        splits = _validate_outer_splits(outer_splits, len(work), y)
        if len(splits) != int(outer_folds):
            # Explicit reusable folds are authoritative, but metadata remains
            # honest about the actual number of partitions used.
            outer_folds = len(splits)

    oof_rows: list[dict[str, Any]] = []
    fold_rows: list[dict[str, Any]] = []
    x_all = work[cols]
    for fold_number, (train_idx, test_idx) in enumerate(splits, start=1):
        x_train = x_all.iloc[train_idx]
        y_train = y[train_idx]
        x_test = x_all.iloc[test_idx]
        y_test = y[test_idx]
        best, best_params, inner_cv = _fit_search(
            kind,
            x_train,
            y_train,
            cols,
            inner_folds=int(inner_folds),
            n_iter=int(n_iter),
            seed=int(seed),
            n_jobs=int(n_jobs),
            tune_n_jobs=tune_n_jobs,
            param_distributions=param_distributions,
            estimator_params=estimator_params,
        )
        # The fitted search result is not reused to calibrate its own training
        # predictions.  Cloned best parameters generate genuine inner OOF
        # predictions, and therefore cannot see an outer test observation.
        inner_prob = cross_val_predict(
            clone(best),
            x_train,
            y_train,
            cv=inner_cv,
            method="predict_proba",
            n_jobs=int(n_jobs if tune_n_jobs is None else tune_n_jobs),
        )[:, 1]
        threshold = select_threshold_target_specificity(
            y_train,
            inner_prob,
            target_specificity=float(target_specificity),
        )
        test_prob = np.asarray(best.predict_proba(x_test)[:, 1], dtype=float)
        test_prob = np.clip(test_prob, 0.0, 1.0)
        test_pred = (test_prob >= float(threshold)).astype(int)
        fold_metrics = metric_aliases(compute_metrics(y_test, test_prob, test_pred))
        fold_row: dict[str, Any] = {
            "model": kind,
            "profile": profile,
            "seed": int(seed),
            "outer_fold": int(fold_number),
            "fold": int(fold_number),
            "outer_train_n": int(train_idx.size),
            "outer_test_n": int(test_idx.size),
            "outer_train_positive": int(np.sum(y_train == 1)),
            "outer_train_negative": int(np.sum(y_train == 0)),
            "outer_test_positive": int(np.sum(y_test == 1)),
            "outer_test_negative": int(np.sum(y_test == 0)),
            "threshold": float(threshold),
            "inner_oof_AP": float(compute_metrics(y_train, inner_prob, inner_prob >= threshold)["AP"]),
            "best_params": json.dumps(best_params, sort_keys=True, ensure_ascii=False),
            **fold_metrics,
        }
        fold_rows.append(fold_row)
        for index, probability, prediction in zip(test_idx, test_prob, test_pred):
            patient = str(work.iloc[int(index)]["patient_id"])
            row: dict[str, Any] = {
                "patient_id": patient,
                "true_label": int(y[index]),
                "y_true": int(y[index]),
                "prediction_probability": float(probability),
                "probability": float(probability),
                "prediction_label": int(prediction),
                "prediction": int(prediction),
                "outer_fold": int(fold_number),
                "fold": int(fold_number),
                "fold_threshold": float(threshold),
                "threshold": float(threshold),
                "model": kind,
                "profile": profile,
                "seed": int(seed),
            }
            for metadata in ("endpoint_state", "endpoint_horizon_days", "time_to_event", "event_type"):
                if metadata in work.columns:
                    value = work.iloc[int(index)][metadata]
                    row[metadata] = None if pd.isna(value) else value
            oof_rows.append(row)

    oof = pd.DataFrame(oof_rows).sort_values("patient_id", kind="stable").reset_index(drop=True)
    folds = pd.DataFrame(fold_rows).sort_values("outer_fold", kind="stable").reset_index(drop=True)
    if len(oof) != len(work) or oof["patient_id"].duplicated().any():
        raise AssertionError("nested evaluation did not produce exactly one OOF row per patient")
    if set(oof["patient_id"]) != set(work["patient_id"].astype(str)):
        raise AssertionError("nested evaluation OOF patients do not match input patients")
    y_oof = oof["true_label"].to_numpy(dtype=int)
    p_oof = oof["prediction_probability"].to_numpy(dtype=float)
    pred_oof = oof["prediction_label"].to_numpy(dtype=int)
    metrics = compute_metrics(y_oof, p_oof, pred_oof)
    summary: dict[str, Any] = {
        "model": kind,
        "profile": profile,
        "seed": int(seed),
        "random_state": int(seed),
        "patient_count": int(len(oof)),
        "positive_count": int(y_oof.sum()),
        "negative_count": int(len(y_oof) - y_oof.sum()),
        "outer_folds": int(len(splits)),
        "inner_folds": int(inner_folds),
        "n_iter": int(n_iter),
        "n_jobs": int(n_jobs),
        "scoring": "average_precision",
        "threshold_method": "target-specificity",
        "target_specificity": float(target_specificity),
        "feature_count": int(len(cols)),
        "feature_columns": list(cols),
        "fold_thresholds": [float(value) for value in folds["threshold"].to_numpy(dtype=float)],
        "metrics": metrics,
    }
    summary.update(metrics)
    summary.update(metric_aliases(metrics))
    if int(bootstrap_resamples) > 0:
        summary["bootstrap"] = bootstrap_metrics(
            y_oof,
            p_oof,
            pred_oof,
            n_resamples=int(bootstrap_resamples),
            seed=int(seed if bootstrap_seed is None else bootstrap_seed),
        )
    else:
        summary["bootstrap"] = {
            "point": metrics,
            "ci": {name: {"lower": float("nan"), "upper": float("nan"), "n_valid": 0} for name in METRIC_NAMES},
            "n_resamples": 0,
            "seed": int(seed if bootstrap_seed is None else bootstrap_seed),
        }
    return NestedCVResult(oof=oof, folds=folds, summary=summary, model=kind, seed=int(seed))


# Friendly aliases for callers using the older terminology.
nested_cv = run_nested_cv
run_nested_patient_cv = run_nested_cv
nested_patient_oof = run_nested_cv
run_nested_patient_oof = run_nested_cv


__all__ = [
    "NestedCVResult",
    "make_outer_splits",
    "run_nested_cv",
    "nested_cv",
    "run_nested_patient_cv",
    "nested_patient_oof",
    "run_nested_patient_oof",
]
