"""Repeated nested-CV orchestration with shared patient folds."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from .full_model import (
    DEFAULT_BOOTSTRAP_RESAMPLES,
    DEFAULT_EXTRA_TREES_N_ITER,
    DEFAULT_INNER_FOLDS,
    DEFAULT_N_JOBS,
    DEFAULT_OUTER_FOLDS,
    DEFAULT_RANDOM_STATE,
    DEFAULT_TARGET_SPECIFICITY,
    canonical_model_name,
)
from .metrics import METRIC_NAMES
from .nested_cv import NestedCVResult, make_outer_splits, run_nested_cv


SUMMARY_STATISTICS: tuple[str, ...] = (
    "mean",
    "std",
    "min",
    "p2.5",
    "p25",
    "median",
    "p75",
    "p97.5",
    "max",
)


@dataclass
class RepeatedCVResult:
    """Results for all requested model/seed combinations."""

    per_seed: pd.DataFrame
    summary: pd.DataFrame
    runs: dict[tuple[str, int], NestedCVResult]
    oof: pd.DataFrame
    summary_wide: pd.DataFrame | None = None

    def __getitem__(self, key: str) -> Any:
        if key in {"per_seed", "seed_metrics", "metrics"}:
            return self.per_seed
        if key in {"summary", "summary_metrics"}:
            return self.summary
        if key == "oof":
            return self.oof
        if key == "runs":
            return self.runs
        raise KeyError(key)


def normalize_seeds(
    seeds: Iterable[int] | int | None = None,
    *,
    seed_start: int | None = None,
    seed_stop: int | None = None,
    seed_step: int = 1,
    seed_range: Sequence[int] | None = None,
) -> list[int]:
    """Normalize explicit seeds or an inclusive integer range.

    ``seed_stop`` is inclusive, which is less surprising for command-line
    ranges and makes ``--seed-start 1 --seed-stop 1`` run one evaluation.
    ``seed_range`` follows the same inclusive convention as a compact tuple.
    """

    if seed_range is not None:
        values = list(seed_range)
        if isinstance(seed_range, range) or len(values) > 3:
            # A range object (and a longer explicit sequence) is already a
            # concrete seed collection rather than a compact start/stop tuple.
            if values:
                seeds = values
            seed_start = seed_stop = None
        else:
            if len(values) not in {2, 3}:
                raise ValueError("seed_range must contain start, stop[, step]")
            seed_start = int(values[0])
            seed_stop = int(values[1])
            seed_step = int(values[2]) if len(values) == 3 else 1
    if seed_start is not None or seed_stop is not None:
        if seed_start is None or seed_stop is None:
            raise ValueError("seed_start and seed_stop must be supplied together")
        if int(seed_step) == 0:
            raise ValueError("seed_step must not be zero")
        stop = int(seed_stop)
        step = int(seed_step)
        if (stop - int(seed_start)) * step < 0:
            raise ValueError("seed range direction conflicts with seed_step")
        stop_exclusive = stop + (1 if step > 0 else -1)
        values = list(range(int(seed_start), stop_exclusive, step))
    elif seeds is None:
        values = [DEFAULT_RANDOM_STATE]
    elif isinstance(seeds, (int, np.integer)):
        values = [int(seeds)]
    else:
        values = [int(value) for value in seeds]
    if not values:
        raise ValueError("at least one seed is required")
    if len(set(values)) != len(values):
        raise ValueError("seeds must be unique")
    return values


def _labels_for_folds(frame: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    if "patient_id" not in frame.columns:
        raise ValueError("frame must contain patient_id")
    ordered = frame.copy(deep=True)
    ordered["patient_id"] = ordered["patient_id"].astype("string")
    if ordered["patient_id"].isna().any() or ordered["patient_id"].duplicated().any():
        raise ValueError("repeated nested CV requires one row per patient")
    ordered = ordered.sort_values("patient_id", kind="stable").reset_index(drop=True)
    label_column = "label" if "label" in ordered.columns else "binary_label_if_evaluable"
    if label_column not in ordered.columns:
        raise ValueError("frame must contain label or binary_label_if_evaluable")
    labels = pd.to_numeric(ordered[label_column], errors="coerce")
    if labels.isna().any() or not labels.isin([0, 1]).all():
        raise ValueError("frame labels must be binary and non-null")
    return ordered, labels.to_numpy(dtype=int)


def _models_argument(model: str | Sequence[str] | None, models: Sequence[str] | None) -> list[str]:
    value: Any = models if models is not None else model
    if value is None:
        value = ("extratrees", "logistic", "dummy")
    if isinstance(value, str):
        values = [item.strip() for item in value.split(",") if item.strip()]
    else:
        values = list(value)
    if not values:
        raise ValueError("at least one model is required")
    canonical = [canonical_model_name(item) for item in values]
    if len(set(canonical)) != len(canonical):
        raise ValueError("models must be unique")
    return canonical


def _summary_frames(per_seed: pd.DataFrame, models: Sequence[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    long_rows: list[dict[str, Any]] = []
    for model in models:
        subset = per_seed.loc[per_seed["model"] == model]
        for statistic in SUMMARY_STATISTICS:
            row: dict[str, Any] = {"model": model, "statistic": statistic, "seed_count": int(len(subset))}
            for name in METRIC_NAMES:
                values = pd.to_numeric(subset[name], errors="coerce").to_numpy(dtype=float)
                values = values[np.isfinite(values)]
                if values.size == 0:
                    value = float("nan")
                elif statistic == "mean":
                    value = np.mean(values)
                elif statistic == "std":
                    value = np.std(values, ddof=0)
                elif statistic == "min":
                    value = np.min(values)
                elif statistic == "max":
                    value = np.max(values)
                elif statistic == "median":
                    value = np.percentile(values, 50)
                elif statistic == "p2.5":
                    value = np.percentile(values, 2.5)
                elif statistic == "p25":
                    value = np.percentile(values, 25)
                elif statistic == "p75":
                    value = np.percentile(values, 75)
                elif statistic == "p97.5":
                    value = np.percentile(values, 97.5)
                else:  # pragma: no cover - SUMMARY_STATISTICS is closed above
                    value = float("nan")
                row[name] = float(value)
            long_rows.append(row)
    summary = pd.DataFrame(long_rows)
    wide_rows: list[dict[str, Any]] = []
    for model in models:
        row: dict[str, Any] = {"model": model, "seed_count": int((per_seed["model"] == model).sum())}
        for statistic in SUMMARY_STATISTICS:
            selected = summary.loc[(summary["model"] == model) & (summary["statistic"] == statistic)]
            if selected.empty:
                continue
            for name in METRIC_NAMES:
                row[f"{name}_{statistic}"] = float(selected.iloc[0][name])
        wide_rows.append(row)
    return summary, pd.DataFrame(wide_rows)


def run_repeated_cv(
    frame: pd.DataFrame,
    model: str | Sequence[str] | None = None,
    *,
    seeds: Iterable[int] | int | None = None,
    seed_start: int | None = None,
    seed_stop: int | None = None,
    seed_step: int = 1,
    seed_range: Sequence[int] | None = None,
    models: Sequence[str] | None = None,
    profile: str = "all20",
    feature_cols: Sequence[str] | None = None,
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
) -> RepeatedCVResult:
    """Run one or more models for explicit seeds with shared outer folds."""

    seed_values = normalize_seeds(
        seeds,
        seed_start=seed_start,
        seed_stop=seed_stop,
        seed_step=seed_step,
        seed_range=seed_range,
    )
    model_values = _models_argument(model, models)
    ordered, labels = _labels_for_folds(frame)
    runs: dict[tuple[str, int], NestedCVResult] = {}
    per_seed_rows: list[dict[str, Any]] = []
    oof_frames: list[pd.DataFrame] = []
    for seed in seed_values:
        shared_splits = make_outer_splits(labels, outer_folds=int(outer_folds), seed=int(seed))
        for kind in model_values:
            result = run_nested_cv(
                ordered,
                model=kind,
                profile=profile,
                feature_cols=feature_cols,
                seed=int(seed),
                outer_folds=int(outer_folds),
                inner_folds=int(inner_folds),
                n_iter=int(n_iter),
                n_jobs=int(n_jobs),
                tune_n_jobs=tune_n_jobs,
                target_specificity=float(target_specificity),
                bootstrap_resamples=int(bootstrap_resamples),
                bootstrap_seed=(int(seed) if bootstrap_seed is None else int(bootstrap_seed) + int(seed)),
                param_distributions=param_distributions,
                estimator_params=estimator_params,
                outer_splits=shared_splits,
            )
            runs[(kind, int(seed))] = result
            row: dict[str, Any] = {
                "model": kind,
                "profile": profile,
                "seed": int(seed),
                "patient_count": int(result.summary["patient_count"]),
                "outer_folds": int(result.summary["outer_folds"]),
                "inner_folds": int(result.summary["inner_folds"]),
            }
            row.update({name: result.summary.get(name, np.nan) for name in METRIC_NAMES})
            row.update(
                {
                    "auc": row["AUC"],
                    "average_precision": row["AP"],
                    "brier": row["Brier"],
                    "sensitivity": row["Sens"],
                    "specificity": row["Spec"],
                    "f1": row["F1"],
                    "ppv": row["PPV"],
                    "npv": row["NPV"],
                }
            )
            per_seed_rows.append(row)
            oof_frames.append(result.oof.copy())
    per_seed = pd.DataFrame(per_seed_rows)
    summary, summary_wide = _summary_frames(per_seed, model_values)
    oof = pd.concat(oof_frames, ignore_index=True) if oof_frames else pd.DataFrame()
    return RepeatedCVResult(
        per_seed=per_seed,
        summary=summary,
        runs=runs,
        oof=oof,
        summary_wide=summary_wide,
    )


repeated_nested_cv = run_repeated_cv
run_repeated_nested_cv = run_repeated_cv


__all__ = [
    "SUMMARY_STATISTICS",
    "RepeatedCVResult",
    "normalize_seeds",
    "run_repeated_cv",
    "repeated_nested_cv",
    "run_repeated_nested_cv",
]
