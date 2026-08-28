"""Metrics and patient-bootstrap helpers for P4-C model validation.

The validation code deliberately keeps metric calculation outside the model
implementation.  This makes it possible to use exactly the same definitions
for ExtraTrees, logistic regression, and the prevalence baseline, and keeps
the bootstrap at the patient level (one input row is one patient).
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


METRIC_NAMES: tuple[str, ...] = (
    "AUC",
    "AP",
    "Brier",
    "Sens",
    "Spec",
    "F1",
    "PPV",
    "NPV",
    "BrierSkill",
    "calibration_intercept",
    "calibration_slope",
)


def _arrays(
    y_true: Sequence[int] | np.ndarray,
    probability: Sequence[float] | np.ndarray,
    prediction: Sequence[int] | np.ndarray | None = None,
    *,
    threshold: float = 0.5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Normalize metric inputs and enforce the binary patient contract."""

    y = np.asarray(y_true, dtype=int).reshape(-1)
    p = np.asarray(probability, dtype=float).reshape(-1)
    if y.size != p.size:
        raise ValueError("y_true and probability must have the same length")
    if y.size == 0:
        raise ValueError("at least one patient is required")
    if not np.isin(y, [0, 1]).all():
        raise ValueError("y_true must contain only 0 and 1")
    # A fitted sklearn classifier should already produce probabilities in this
    # range.  Clipping here makes metric output stable for custom estimators
    # and prevents one malformed value from poisoning a bootstrap replicate.
    p = np.clip(p, 0.0, 1.0)
    if not np.isfinite(p).all():
        raise ValueError("probability contains non-finite values")
    if prediction is None:
        pred = (p >= float(threshold)).astype(int)
    else:
        pred = np.asarray(prediction, dtype=int).reshape(-1)
        if pred.size != y.size:
            raise ValueError("prediction must have the same length as y_true")
        if not np.isin(pred, [0, 1]).all():
            raise ValueError("prediction must contain only 0 and 1")
    return y, p, pred


def _safe_auc(y: np.ndarray, p: np.ndarray) -> float:
    return float(roc_auc_score(y, p)) if np.unique(y).size == 2 else float("nan")


def _safe_ap(y: np.ndarray, p: np.ndarray) -> float:
    return float(average_precision_score(y, p)) if np.unique(y).size == 2 else float("nan")


def _safe_brier(y: np.ndarray, p: np.ndarray) -> float:
    return float(brier_score_loss(y, p))


def _safe_brier_skill(y: np.ndarray, p: np.ndarray, brier: float) -> float:
    """Compare Brier loss with the constant observed-prevalence forecast."""

    prevalence = float(np.mean(y))
    reference = prevalence * (1.0 - prevalence)
    # There is no informative constant-prevalence baseline for a degenerate
    # label vector; returning NaN is preferable to an arbitrary division by
    # zero or a manufactured score.
    if not np.isfinite(reference) or reference <= 0.0:
        return float("nan")
    return float(1.0 - (float(brier) / reference))


def _logit(probability: np.ndarray) -> np.ndarray:
    # Clipping is only for the descriptive calibration regression.  The
    # original probabilities remain unchanged for all other metrics.
    eps = np.finfo(float).eps
    clipped = np.clip(probability, eps, 1.0 - eps)
    return np.log(clipped) - np.log1p(-clipped)


def _expit(value: np.ndarray | float) -> np.ndarray:
    """Numerically stable logistic transform without overflow warnings."""

    array = np.asarray(value, dtype=float)
    result = np.empty_like(array)
    positive = array >= 0.0
    result[positive] = 1.0 / (1.0 + np.exp(-array[positive]))
    exp_value = np.exp(array[~positive])
    result[~positive] = exp_value / (1.0 + exp_value)
    return result


def _safe_calibration_intercept(y: np.ndarray, p: np.ndarray) -> float:
    """Fit the calibration intercept with the model logit as an offset."""

    if np.unique(y).size < 2:
        return float("nan")
    eta = _logit(p)
    target = float(np.sum(y))
    # The score is monotone, so a bracketed bisection is deterministic and
    # remains stable for probabilities at the clipping boundaries.
    low, high = -100.0, 100.0
    for _ in range(120):
        mid = (low + high) / 2.0
        fitted = _expit(np.clip(eta + mid, -745.0, 745.0))
        if float(np.sum(fitted)) < target:
            low = mid
        else:
            high = mid
    value = (low + high) / 2.0
    return float(value) if np.isfinite(value) and abs(value) < 100.0 else float("nan")


def _safe_calibration_slope(y: np.ndarray, p: np.ndarray) -> float:
    """Fit logistic calibration ``y ~ intercept + slope * logit(p)``."""

    if np.unique(y).size < 2:
        return float("nan")
    x = _logit(p)
    if np.ptp(x) <= np.finfo(float).eps:
        return float("nan")
    design = np.column_stack((np.ones(x.size, dtype=float), x))
    beta = np.array([_safe_calibration_intercept(y, p), 1.0], dtype=float)
    if not np.isfinite(beta).all():
        return float("nan")
    for _ in range(100):
        eta = np.clip(design @ beta, -745.0, 745.0)
        fitted = _expit(eta)
        weights = fitted * (1.0 - fitted)
        hessian = design.T @ (weights[:, None] * design)
        gradient = design.T @ (y - fitted)
        if not np.isfinite(hessian).all() or not np.isfinite(gradient).all():
            return float("nan")
        try:
            step = np.linalg.solve(hessian, gradient)
        except np.linalg.LinAlgError:
            return float("nan")
        if not np.isfinite(step).all():
            return float("nan")
        beta += step
        if not np.isfinite(beta).all() or np.max(np.abs(beta)) > 1e6:
            # Divergence indicates separation, for which no finite MLE exists.
            return float("nan")
        if np.max(np.abs(step)) <= 1e-10:
            return float(beta[1]) if np.isfinite(beta[1]) else float("nan")
    return float("nan")


def compute_metrics(
    y_true: Sequence[int] | np.ndarray,
    probability: Sequence[float] | np.ndarray,
    prediction: Sequence[int] | np.ndarray | None = None,
    *,
    threshold: float = 0.5,
) -> dict[str, float]:
    """Return the frozen P4-C metric set.

    ``AUC`` and ``AP`` are ``nan`` when a sample contains only one class;
    callers can retain that fact in machine-readable output instead of
    silently manufacturing a score.  ``prediction`` is optional and is
    derived from ``threshold`` when omitted.
    """

    y, p, pred = _arrays(y_true, probability, prediction, threshold=threshold)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    sensitivity = float(tp / (tp + fn)) if (tp + fn) else float("nan")
    specificity = float(tn / (tn + fp)) if (tn + fp) else float("nan")
    ppv = float(tp / (tp + fp)) if (tp + fp) else float("nan")
    npv = float(tn / (tn + fn)) if (tn + fn) else float("nan")
    # sklearn's zero_division=0 gives a defined F1 for an all-negative
    # prevalence prediction while Sens/Spec/PPV/NPV retain their statistical
    # undefined state where a denominator is absent.
    brier = _safe_brier(y, p)
    values = {
        "AUC": _safe_auc(y, p),
        "AP": _safe_ap(y, p),
        "Brier": brier,
        "Sens": sensitivity,
        "Spec": specificity,
        "F1": float(f1_score(y, pred, zero_division=0)),
        "PPV": ppv,
        "NPV": npv,
        "BrierSkill": _safe_brier_skill(y, p, brier),
        "calibration_intercept": _safe_calibration_intercept(y, p),
        "calibration_slope": _safe_calibration_slope(y, p),
    }
    return values


def classification_metrics(
    y_true: Sequence[int] | np.ndarray,
    probability: Sequence[float] | np.ndarray,
    prediction: Sequence[int] | np.ndarray | None = None,
    *,
    threshold: float = 0.5,
) -> dict[str, float]:
    """Descriptive alias for :func:`compute_metrics`."""

    return compute_metrics(y_true, probability, prediction, threshold=threshold)


def metric_aliases(values: Mapping[str, Any]) -> dict[str, Any]:
    """Add readable lower-case aliases without changing canonical metrics."""

    aliases = {
        "auc": "AUC",
        "average_precision": "AP",
        "ap": "AP",
        "brier": "Brier",
        "sensitivity": "Sens",
        "specificity": "Spec",
        "f1": "F1",
        "ppv": "PPV",
        "npv": "NPV",
        "brier_skill": "BrierSkill",
        "calibration_intercept": "calibration_intercept",
        "calibration_slope": "calibration_slope",
    }
    result = dict(values)
    for alias, source in aliases.items():
        if source in values:
            result[alias] = values[source]
    return result


def bootstrap_metrics(
    y_true: Sequence[int] | np.ndarray,
    probability: Sequence[float] | np.ndarray,
    prediction: Sequence[int] | np.ndarray | None = None,
    *,
    threshold: float = 0.5,
    n_resamples: int = 2000,
    seed: int = 42,
    alpha: float = 0.05,
) -> dict[str, Any]:
    """Patient-level percentile bootstrap confidence intervals.

    Rows are resampled with replacement, so a patient can occur multiple
    times in a replicate but no window-level resampling is possible.  AUC/AP
    replicates containing one class are ignored for their percentile; the
    point estimate remains the full-sample value.
    """

    if int(n_resamples) < 1:
        raise ValueError("n_resamples must be positive")
    if not 0 <= float(alpha) < 1:
        raise ValueError("alpha must be in [0, 1)")
    y, p, pred = _arrays(y_true, probability, prediction, threshold=threshold)
    point = compute_metrics(y, p, pred, threshold=threshold)
    rng = np.random.default_rng(int(seed))
    samples: dict[str, list[float]] = {name: [] for name in METRIC_NAMES}
    n = len(y)
    for _ in range(int(n_resamples)):
        indices = rng.integers(0, n, size=n)
        values = compute_metrics(y[indices], p[indices], pred[indices], threshold=threshold)
        for name in METRIC_NAMES:
            value = values[name]
            if np.isfinite(value):
                samples[name].append(float(value))

    lower_q = 100.0 * (float(alpha) / 2.0)
    upper_q = 100.0 * (1.0 - float(alpha) / 2.0)
    intervals: dict[str, dict[str, float]] = {}
    for name in METRIC_NAMES:
        values = np.asarray(samples[name], dtype=float)
        if values.size == 0:
            lower = upper = float("nan")
        else:
            lower, upper = np.percentile(values, [lower_q, upper_q])
        intervals[name] = {
            "lower": float(lower),
            "upper": float(upper),
            "n_valid": int(values.size),
        }
    return {
        "point": point,
        "ci": intervals,
        "n_resamples": int(n_resamples),
        "seed": int(seed),
        "alpha": float(alpha),
        "level": float(1.0 - float(alpha)),
    }


def bootstrap_ci(*args: Any, **kwargs: Any) -> dict[str, Any]:
    """Alias retained for callers that request confidence intervals directly."""

    # ``n_bootstrap``/``random_state`` are common spellings in analysis code;
    # accept them while keeping the implementation's explicit names.
    if "n_bootstrap" in kwargs and "n_resamples" not in kwargs:
        kwargs["n_resamples"] = kwargs.pop("n_bootstrap")
    if "random_state" in kwargs and "seed" not in kwargs:
        kwargs["seed"] = kwargs.pop("random_state")
    return bootstrap_metrics(*args, **kwargs)


def select_threshold_target_specificity(
    y_true: Sequence[int] | np.ndarray,
    probability: Sequence[float] | np.ndarray,
    *,
    target_specificity: float = 0.70,
) -> float:
    """Select a training-only threshold targeting a minimum specificity.

    The threshold is selected from negative-class inner-OOF probabilities and
    nudged above the selected negative probability.  This ensures the rule is
    deterministic, does not inspect an outer test row, and remains conservative
    when there are tied probabilities.
    """

    y = np.asarray(y_true, dtype=int).reshape(-1)
    p = np.asarray(probability, dtype=float).reshape(-1)
    if y.size != p.size or y.size == 0:
        raise ValueError("y_true and probability must be non-empty and equally sized")
    if not np.isin(y, [0, 1]).all():
        raise ValueError("y_true must contain only 0 and 1")
    if not np.isfinite(p).all():
        raise ValueError("probability contains non-finite values")
    target = float(target_specificity)
    if not 0.0 <= target <= 1.0:
        raise ValueError("target_specificity must be in [0, 1]")
    negatives = np.sort(np.clip(p[y == 0], 0.0, 1.0))
    if negatives.size == 0:
        return 0.5
    # Need at least ceil(target * n_neg) negatives below threshold.  The
    # floor formulation below matches the frozen legacy rule at exact values
    # while guaranteeing a valid array index for target=0 and target=1.
    if target >= 1.0:
        return float(np.nextafter(negatives[-1], np.inf))
    index = int(np.floor(target * negatives.size))
    index = min(max(index, 0), negatives.size - 1)
    return float(np.nextafter(float(negatives[index]), np.inf))


# Compatibility names used by older validation notebooks.
compute_classification_metrics = compute_metrics
calculate_metrics = compute_metrics
select_threshold = select_threshold_target_specificity
patient_bootstrap = bootstrap_metrics


__all__ = [
    "METRIC_NAMES",
    "compute_metrics",
    "classification_metrics",
    "compute_classification_metrics",
    "calculate_metrics",
    "metric_aliases",
    "bootstrap_metrics",
    "bootstrap_ci",
    "patient_bootstrap",
    "select_threshold_target_specificity",
    "select_threshold",
]
