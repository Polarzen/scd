"""Nested patient-level ExtraTrees reproduction for Phase 3 legacy features."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline

from .legacy_aggregation import AGGREGATED_FEATURE_NAMES


RANDOM_STATE = 42
DEFAULT_OUTER_FOLDS = 5
DEFAULT_INNER_FOLDS = 3
DEFAULT_N_ITER = 24
DEFAULT_TARGET_SPECIFICITY = 0.70
DEFAULT_N_JOBS = 1


def model_feature_names() -> list[str]:
    """Return exactly the 100 aggregated feature columns used by the model."""

    return list(AGGREGATED_FEATURE_NAMES)


def get_model_feature_columns(
    frame: pd.DataFrame,
    feature_cols: Sequence[str] | None = None,
) -> list[str]:
    """Validate and return the 100 feature columns, excluding all metadata."""

    cols = list(feature_cols) if feature_cols is not None else model_feature_names()
    if len(cols) != 100 or len(set(cols)) != 100:
        raise ValueError(f"legacy model requires exactly 100 unique feature columns, got {len(cols)}")
    missing = [column for column in cols if column not in frame.columns]
    if missing:
        raise ValueError(f"model feature columns missing: {missing}")
    return cols


def build_model(
    feature_cols: Sequence[str],
    *,
    model_params: Mapping[str, Any] | None = None,
    n_jobs: int = DEFAULT_N_JOBS,
) -> Pipeline:
    """Build the old ExtraTrees estimator with imputation inside the pipeline."""

    cols = list(feature_cols)
    if len(cols) != 100 or len(set(cols)) != 100:
        raise ValueError("ExtraTrees reproduction requires exactly 100 feature columns")
    pre = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(steps=[("imp", SimpleImputer(strategy="median"))]),
                cols,
            )
        ],
        remainder="drop",
    )
    estimator = ExtraTreesClassifier(
        n_estimators=800,
        max_depth=None,
        min_samples_leaf=2,
        min_samples_split=6,
        max_features="sqrt",
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=n_jobs,
    )
    if model_params:
        estimator.set_params(**dict(model_params))
    return Pipeline(steps=[("pre", pre), ("clf", estimator)])


def get_param_distributions() -> dict[str, list[Any]]:
    """Return the exact old ExtraTrees randomized-search distributions."""

    return {
        "clf__n_estimators": [400, 800, 1200, 1600],
        "clf__max_depth": [None, 4, 6, 8, 10, 14],
        "clf__min_samples_split": [2, 4, 6, 8, 10],
        "clf__min_samples_leaf": [1, 2, 3, 4],
        "clf__max_features": ["sqrt", "log2", 0.5, 0.7],
        "clf__class_weight": ["balanced", {0: 1, 1: 2}, {0: 1, 1: 3}],
    }


def select_best_threshold(
    y_true: Sequence[int] | np.ndarray,
    prob: Sequence[float] | np.ndarray,
    *,
    method: str = "target-specificity",
    target_specificity: float = DEFAULT_TARGET_SPECIFICITY,
) -> float:
    """Select a threshold using the old target-specificity rule."""

    from sklearn.metrics import precision_recall_curve, roc_curve

    y_arr = np.asarray(y_true, dtype=int)
    p_arr = np.asarray(prob, dtype=np.float64)
    if method == "target-specificity":
        target_specificity = float(np.clip(target_specificity, 0.0, 1.0))
        neg_prob = np.sort(p_arr[y_arr == 0])
        if neg_prob.size == 0:
            return 0.5
        if target_specificity >= 1.0:
            return float(np.nextafter(neg_prob[-1], np.inf))
        k = int(np.floor(target_specificity * neg_prob.size))
        k = min(max(k, 0), neg_prob.size - 1)
        return float(np.nextafter(float(neg_prob[k]), np.inf))
    if method == "f1":
        precision, recall, thresholds = precision_recall_curve(y_arr, p_arr)
        if thresholds.size == 0:
            return 0.5
        f1_vals = 2.0 * precision[:-1] * recall[:-1] / (precision[:-1] + recall[:-1] + 1e-12)
        if f1_vals.size == 0 or np.all(~np.isfinite(f1_vals)):
            return 0.5
        return float(thresholds[int(np.nanargmax(f1_vals))])
    fpr, tpr, thresholds = roc_curve(y_arr, p_arr)
    return float(thresholds[int(np.argmax(tpr - fpr))])


def compute_classification_stats(y_true: Sequence[int], pred: Sequence[int]) -> dict[str, Any]:
    y_arr = np.asarray(y_true, dtype=int)
    p_arr = np.asarray(pred, dtype=int)
    cm = confusion_matrix(y_arr, p_arr, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    return {
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "sensitivity": float(tp / (tp + fn + 1e-12)),
        "specificity": float(tn / (tn + fp + 1e-12)),
        "f1": float(f1_score(y_arr, p_arr, zero_division=0)),
    }


def _safe_ap(y_true: np.ndarray, prob: np.ndarray) -> float:
    return float(average_precision_score(y_true, prob)) if np.unique(y_true).size > 1 else np.nan


def _safe_auc(y_true: np.ndarray, prob: np.ndarray) -> float:
    return float(roc_auc_score(y_true, prob)) if np.unique(y_true).size > 1 else np.nan


def _jsonable(value: Any) -> Any:
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


@dataclass
class NestedOOFResult:
    """Outputs from the strict 5x3 nested patient-level evaluation."""

    oof: pd.DataFrame
    folds: pd.DataFrame
    summary: dict[str, Any]


def nested_patient_oof(
    frame: pd.DataFrame,
    *,
    feature_cols: Sequence[str] | None = None,
    outer_folds: int = DEFAULT_OUTER_FOLDS,
    inner_folds: int = DEFAULT_INNER_FOLDS,
    n_iter: int = DEFAULT_N_ITER,
    n_jobs: int = DEFAULT_N_JOBS,
    tune_n_jobs: int | None = None,
    target_specificity: float = DEFAULT_TARGET_SPECIFICITY,
    param_distributions: Mapping[str, Sequence[Any]] | None = None,
    estimator_params: Mapping[str, Any] | None = None,
) -> NestedOOFResult:
    """Run strict patient-level nested CV and return one OOF row per patient.

    Hyperparameter search and inner OOF threshold calibration are both
    performed using only each outer training partition.  Numeric imputation is
    a pipeline step, so it is refit inside every training partition.
    """

    if "patient_id" not in frame.columns or "label" not in frame.columns:
        raise ValueError("frame must contain patient_id and label")
    if frame["patient_id"].duplicated().any():
        raise ValueError("nested model requires one aggregate row per patient")
    if int(outer_folds) < 2 or int(inner_folds) < 2:
        raise ValueError("outer_folds and inner_folds must be at least two")

    work = frame.sort_values("patient_id", kind="stable").reset_index(drop=True).copy()
    cols = get_model_feature_columns(work, feature_cols)
    y = pd.to_numeric(work["label"], errors="coerce").astype(int).to_numpy()
    if np.unique(y).size != 2:
        raise ValueError("nested model requires both labels")
    if np.bincount(y, minlength=2).min() < int(outer_folds):
        raise ValueError("each class needs at least outer_folds patients")

    outer_cv = StratifiedKFold(n_splits=int(outer_folds), shuffle=True, random_state=RANDOM_STATE)
    oof_rows: list[dict[str, Any]] = []
    fold_rows: list[dict[str, Any]] = []
    search_space = dict(param_distributions or get_param_distributions())
    search_jobs = n_jobs if tune_n_jobs is None else int(tune_n_jobs)

    for fold_idx, (train_idx, test_idx) in enumerate(outer_cv.split(work, y), start=1):
        x_train = work.iloc[train_idx][cols]
        y_train = y[train_idx]
        x_test = work.iloc[test_idx][cols]
        y_test = y[test_idx]
        n_pos = int(y_train.sum())
        n_neg = int(len(y_train) - n_pos)
        n_inner = min(int(inner_folds), n_pos, n_neg)
        if n_inner < 2:
            raise ValueError("inner stratified CV requires at least two patients per class")

        base = build_model(cols, model_params=estimator_params, n_jobs=n_jobs)
        search = RandomizedSearchCV(
            estimator=base,
            param_distributions=search_space,
            n_iter=max(1, int(n_iter)),
            scoring="average_precision",
            n_jobs=search_jobs,
            cv=StratifiedKFold(n_splits=n_inner, shuffle=True, random_state=RANDOM_STATE),
            refit=True,
            random_state=RANDOM_STATE,
        )
        search.fit(x_train, y_train)
        best = search.best_estimator_

        # Calibrate the threshold from training-side inner OOF predictions,
        # not from outer test observations or fitted training probabilities.
        inner_cv = StratifiedKFold(n_splits=n_inner, shuffle=True, random_state=RANDOM_STATE)
        inner_prob = cross_val_predict(
            clone(best),
            x_train,
            y_train,
            cv=inner_cv,
            method="predict_proba",
            n_jobs=search_jobs,
        )[:, 1]
        threshold = select_best_threshold(
            y_train,
            inner_prob,
            method="target-specificity",
            target_specificity=target_specificity,
        )
        test_prob = best.predict_proba(x_test)[:, 1]
        test_pred = (test_prob >= threshold).astype(int)
        fold_stats = compute_classification_stats(y_test, test_pred)
        fold_row: dict[str, Any] = {
            "fold": fold_idx,
            "outer_train_n": int(len(train_idx)),
            "outer_test_n": int(len(test_idx)),
            "outer_train_positive": n_pos,
            "outer_train_negative": n_neg,
            "outer_test_positive": int(y_test.sum()),
            "outer_test_negative": int(len(y_test) - y_test.sum()),
            "threshold": float(threshold),
            "inner_oof_ap": _safe_ap(y_train, inner_prob),
            "outer_test_ap": _safe_ap(y_test, test_prob),
            "outer_test_roc_auc": _safe_auc(y_test, test_prob),
            "best_params": json.dumps(_jsonable(search.best_params_), sort_keys=True),
            **fold_stats,
        }
        fold_rows.append(fold_row)
        for idx, probability, prediction in zip(test_idx, test_prob, test_pred):
            oof_rows.append(
                {
                    "patient_id": str(work.iloc[idx]["patient_id"]),
                    "true_label": int(y[idx]),
                    "prediction_probability": float(probability),
                    "prediction_label": int(prediction),
                    "outer_fold": fold_idx,
                    "fold_threshold": float(threshold),
                    "seed": RANDOM_STATE,
                }
            )

    oof = pd.DataFrame(oof_rows).sort_values("patient_id", kind="stable").reset_index(drop=True)
    folds = pd.DataFrame(fold_rows).sort_values("fold", kind="stable").reset_index(drop=True)
    if len(oof) != len(work) or oof["patient_id"].duplicated().any():
        raise AssertionError("nested evaluation did not produce exactly one OOF row per patient")
    y_oof = oof["true_label"].to_numpy(dtype=int)
    p_oof = oof["prediction_probability"].to_numpy(dtype=float)
    pred_oof = oof["prediction_label"].to_numpy(dtype=int)
    summary: dict[str, Any] = {
        "patient_count": int(len(oof)),
        "positive_count": int(y_oof.sum()),
        "negative_count": int(len(y_oof) - y_oof.sum()),
        "outer_folds": int(outer_folds),
        "inner_folds": int(inner_folds),
        "random_state": RANDOM_STATE,
        "n_iter": int(n_iter),
        "scoring": "average_precision",
        "threshold_method": "target-specificity",
        "target_specificity": float(target_specificity),
        "feature_count": len(cols),
        "feature_columns": cols,
        "oof_average_precision": _safe_ap(y_oof, p_oof),
        "oof_roc_auc": _safe_auc(y_oof, p_oof),
        "oof_brier": float(brier_score_loss(y_oof, p_oof)),
        **compute_classification_stats(y_oof, pred_oof),
        "fold_thresholds": [float(x) for x in folds["threshold"].to_numpy()],
    }
    return NestedOOFResult(oof=oof, folds=folds, summary=summary)


# Descriptive aliases for callers that use "run" terminology.
run_nested_oof = nested_patient_oof
fit_nested_model = nested_patient_oof


__all__ = [
    "RANDOM_STATE",
    "DEFAULT_OUTER_FOLDS",
    "DEFAULT_INNER_FOLDS",
    "DEFAULT_N_ITER",
    "DEFAULT_TARGET_SPECIFICITY",
    "AGGREGATED_FEATURE_NAMES",
    "NestedOOFResult",
    "model_feature_names",
    "get_model_feature_columns",
    "build_model",
    "get_param_distributions",
    "select_best_threshold",
    "compute_classification_stats",
    "nested_patient_oof",
    "run_nested_oof",
    "fit_nested_model",
]
