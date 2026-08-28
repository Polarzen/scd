"""Aggregation, integrity checking, and reporting for repeated model searches.

This module intentionally treats a ``candidate``/``seed`` pair as the atomic
unit of a repeated optimization run.  The validation performed here is
separate from model fitting: it checks that artifacts can be paired without
silently dropping a seed or a patient, then produces flat, portable artifacts
for downstream review.

The public functions accept either pandas objects, record dictionaries, or a
directory containing CSV/JSON/Parquet artifacts.  No confidence intervals are
computed in this module.  The percentile columns in the summary describe the
observed split-to-split seed distribution only.
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score


FORMAL_SEEDS: frozenset[int] = frozenset(range(100))
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
PAIR_METRICS: tuple[str, ...] = ("AUC", "AP", "Brier")
REQUIRED_REPORT_HEADINGS: tuple[str, ...] = (
    "Baseline",
    "Small-N/high-D experiment",
    "AF-compatible experiment",
    "PVC incremental experiment",
    "Calibration",
    "100-seed robustness",
    "Endpoint sensitivity",
    "Final selected model",
    "Limitations",
    "Advisor-facing conclusion",
)


class IntegrityError(ValueError):
    """Raised when an optimization artifact violates its run contract."""


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    try:
        result = pd.isna(value)
        return bool(result) if not isinstance(result, (list, tuple, np.ndarray, pd.Series)) else False
    except (TypeError, ValueError):
        return False


def _jsonable(value: Any) -> Any:
    """Convert numpy/pandas values and non-finite floats to strict JSON values."""

    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, np.ndarray, pd.Series)):
        return [_jsonable(item) for item in list(value)]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        result = float(value)
        return result if math.isfinite(result) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if _is_missing(value):
        return None
    if isinstance(value, (str, int, bool)):
        return value
    return str(value)


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(_jsonable(value), indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _first(row: Mapping[str, Any], names: Sequence[str], default: Any = None) -> Any:
    for name in names:
        if name in row and not _is_missing(row[name]):
            return row[name]
    return default


def _parse_seed(value: Any, *, label: str = "seed") -> int:
    if _is_missing(value) or isinstance(value, bool):
        raise IntegrityError(f"{label} must be a finite integer")
    if isinstance(value, str):
        text = value.strip()
        if not re.fullmatch(r"[+-]?\d+", text):
            raise IntegrityError(f"{label} must be an integer, got {value!r}")
        result = int(text)
    else:
        try:
            number = float(value)
        except (TypeError, ValueError, OverflowError) as exc:
            raise IntegrityError(f"{label} must be a finite integer") from exc
        if not math.isfinite(number) or not number.is_integer():
            raise IntegrityError(f"{label} must be a finite integer")
        result = int(number)
    return result


def _finite_number(value: Any, *, label: str, lower: float | None = None, upper: float | None = None) -> float:
    if _is_missing(value):
        raise IntegrityError(f"{label} is missing")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise IntegrityError(f"{label} must be numeric") from exc
    if not math.isfinite(result):
        raise IntegrityError(f"{label} must be finite")
    if lower is not None and result < lower:
        raise IntegrityError(f"{label} must be >= {lower}")
    if upper is not None and result > upper:
        raise IntegrityError(f"{label} must be <= {upper}")
    return result


def _binary_number(value: Any, *, label: str) -> int:
    number = _finite_number(value, label=label)
    if number not in (0.0, 1.0):
        raise IntegrityError(f"{label} must be 0 or 1")
    return int(number)


def _endpoint_number(value: Any, *, label: str) -> int | float:
    result = _finite_number(value, label=label)
    return int(result) if result.is_integer() else result


def _metadata(row: Mapping[str, Any], name: str) -> Any:
    aliases = {
        "candidate": ("candidate", "candidate_id", "name"),
        "profile": ("profile", "analysis_profile"),
        "model": ("model", "model_name", "estimator"),
        "seed": ("seed", "random_seed", "random_state"),
        "endpoint": ("endpoint_horizon_days", "endpoint_horizon", "horizon_days"),
    }
    value = _first(row, aliases[name])
    if name == "seed":
        return _parse_seed(value)
    if name == "endpoint":
        if value is None:
            return None
        result = _finite_number(value, label="endpoint_horizon")
        return int(result) if result.is_integer() else result
    if value is None:
        raise IntegrityError(f"{name} is required")
    result = str(value).strip()
    if not result:
        raise IntegrityError(f"{name} must not be empty")
    return result


def _records_from_value(value: Any, *, kind: str) -> list[Any]:
    """Flatten common record/container forms while retaining DataFrames."""

    if value is None:
        return []
    if isinstance(value, pd.DataFrame):
        return [value.copy(deep=True)]
    if isinstance(value, Mapping):
        for key in (kind, f"{kind}_records", "records", "rows", "data"):
            if key in value and isinstance(value[key], (list, tuple, pd.DataFrame, Mapping)):
                nested = value[key]
                if isinstance(nested, Mapping):
                    return [nested]
                return _records_from_value(nested, kind=kind)
        # A dictionary with a required metadata key is one record.  A mapping
        # keyed by run identifiers is handled as a collection of records.
        if any(key in value for key in ("candidate", "candidate_id", "seed", "random_seed")):
            return [dict(value)]
        return [dict(item) if isinstance(item, Mapping) else item for item in value.values()]
    if isinstance(value, (list, tuple)):
        result: list[Any] = []
        for item in value:
            if isinstance(item, pd.DataFrame):
                result.append(item.copy(deep=True))
            elif isinstance(item, Mapping) and kind in item and isinstance(item[kind], (Mapping, pd.DataFrame)):
                nested = dict(item[kind]) if isinstance(item[kind], Mapping) else item[kind]
                if isinstance(nested, Mapping):
                    nested = {**dict(item), **nested}
                    nested.pop(kind, None)
                result.append(nested)
            else:
                result.append(item)
        return result
    raise TypeError(f"unsupported {kind} input type: {type(value).__name__}")


def _read_file(path: Path) -> Any:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    if suffix == ".json":
        return json.loads(path.read_text(encoding="utf-8"))
    raise ValueError(f"unsupported artifact file: {path}")


def load_optimization_artifacts(directory: str | Path) -> tuple[list[Any], list[Any]]:
    """Recursively load summary and OOF files from ``directory``.

    File names containing ``oof`` or ``prediction`` are treated as OOF tables;
    all other supported files are treated as summary records.  This explicit
    name rule avoids accidentally ingesting generated reports as data.
    """

    root = Path(directory)
    if not root.exists():
        raise FileNotFoundError(root)
    files = [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in {".csv", ".json", ".parquet", ".pq"}]
    summary: list[Any] = []
    oof: list[Any] = []
    # Nested-run writers deliberately emit CSV and Parquet copies of OOF data.
    # Prefer Parquet once per logical stem so integrity validation sees each
    # patient once rather than treating portable duplicates as repeated rows.
    selected: list[Path] = []
    by_parent_stem: dict[tuple[Path, str], list[Path]] = {}
    for path in files:
        lowered = path.name.lower()
        if "oof" in lowered or "prediction" in lowered:
            by_parent_stem.setdefault((path.parent, path.stem.lower()), []).append(path)
        elif lowered.endswith("_summary.json") or lowered == "summary.json":
            selected.append(path)
    for choices in by_parent_stem.values():
        selected.append(sorted(choices, key=lambda p: (p.suffix.lower() not in {".parquet", ".pq"}, p.name.lower()))[0])
    for path in sorted(selected, key=lambda item: item.as_posix().lower()):
        lowered = path.name.lower()
        value = _read_file(path)
        if "oof" in lowered or "prediction" in lowered:
            oof.extend(_records_from_value(value, kind="oof"))
        else:
            # A generated structured summary has no one-row-per-seed records;
            # plain per-run summary files (including a file literally named
            # ``summary.json``) remain valid inputs.
            if isinstance(value, Mapping) and "candidates" in value and "seed" not in value:
                continue
            if isinstance(value, pd.DataFrame) and "seed" not in value.columns:
                continue
            summary.extend(_records_from_value(value, kind="summary"))
    return summary, oof


def _flatten_summary(record: Mapping[str, Any]) -> dict[str, Any]:
    row = dict(record)
    nested = row.get("metrics")
    if isinstance(nested, Mapping):
        for key, value in nested.items():
            row.setdefault(str(key), value)
    bootstrap = row.get("bootstrap")
    if isinstance(bootstrap, Mapping) and isinstance(bootstrap.get("point"), Mapping):
        for key, value in bootstrap["point"].items():
            row.setdefault(str(key), value)
    return row


def _normalise_summary_records(records: Any) -> pd.DataFrame:
    items = _records_from_value(records, kind="summary")
    rows: list[dict[str, Any]] = []
    for item in items:
        if isinstance(item, pd.DataFrame):
            rows.extend(item.to_dict(orient="records"))
        elif isinstance(item, Mapping):
            rows.append(_flatten_summary(item))
        else:
            raise TypeError("summary records must be mappings or DataFrames")
    if not rows:
        raise IntegrityError("at least one summary record is required")
    canonical: list[dict[str, Any]] = []
    metric_aliases = {
        "AUC": ("AUC", "auc", "roc_auc"),
        "AP": ("AP", "ap", "average_precision"),
        "Brier": ("Brier", "brier"),
        "threshold": ("threshold", "fold_threshold", "threshold_mean"),
    }
    for raw in rows:
        row = dict(raw)
        normalized: dict[str, Any] = {
            "candidate": _metadata(row, "candidate"),
            "profile": _metadata(row, "profile"),
            "model": _metadata(row, "model"),
            "seed": _metadata(row, "seed"),
            "endpoint_horizon": _metadata(row, "endpoint"),
        }
        for output_name, names in metric_aliases.items():
            value = _first(row, names)
            # ``_first`` treats null as unavailable so that an OOF-derived
            # value can fill it.  A literal non-finite numeric metric, on the
            # other hand, is a malformed summary and fails closed.
            present_name = next((name for name in names if name in row), None)
            if present_name is not None and isinstance(row[present_name], (float, np.floating)) and not math.isfinite(float(row[present_name])):
                raise IntegrityError(f"{output_name} summary must be finite when supplied")
            if value is not None:
                # A missing source value is allowed for metrics that can be
                # derived from OOF, but an explicitly supplied non-finite
                # primary metric is a malformed artifact.
                if output_name in {"AUC", "AP", "Brier"}:
                    normalized[output_name] = _finite_number(value, label=f"{output_name} summary")
                elif output_name == "threshold":
                    normalized[output_name] = _finite_number(value, label="threshold", lower=0.0, upper=1.0)
        for key, value in row.items():
            if key not in normalized and key not in {"metrics", "bootstrap"}:
                normalized[key] = value
        canonical.append(normalized)
    result = pd.DataFrame(canonical)
    if result.duplicated(["candidate", "seed"]).any():
        duplicated = result.loc[result.duplicated(["candidate", "seed"], keep=False), ["candidate", "seed"]]
        raise IntegrityError(f"duplicate summary for candidate-seed: {duplicated.to_dict('records')}")
    for candidate, subset in result.groupby("candidate", sort=False):
        for field in ("profile", "model", "endpoint_horizon"):
            values = {_jsonable(value) for value in subset[field].tolist()}
            if len(values) != 1:
                raise IntegrityError(f"mixed {field} metadata within candidate {candidate!r}")
    return result.sort_values(["candidate", "seed"], kind="stable").reset_index(drop=True)


def _normalise_oof_tables(tables: Any) -> pd.DataFrame:
    items = _records_from_value(tables, kind="oof")
    frames: list[pd.DataFrame] = []
    for item in items:
        if isinstance(item, pd.DataFrame):
            frames.append(item.copy(deep=True))
        elif isinstance(item, Mapping):
            # A single row mapping is also accepted for small synthetic runs.
            frames.append(pd.DataFrame([dict(item)]))
        elif isinstance(item, (list, tuple)):
            frames.append(pd.DataFrame(item))
        else:
            raise TypeError("OOF tables must be DataFrames or row mappings")
    if not frames:
        return pd.DataFrame()
    frame = pd.concat(frames, ignore_index=True, sort=False)
    aliases = {
        "candidate": ("candidate", "candidate_id"),
        "profile": ("profile", "analysis_profile"),
        "model": ("model", "model_name"),
        "seed": ("seed", "random_seed", "random_state"),
        "patient_id": ("patient_id", "patient", "subject_id"),
        "y_true": ("y_true", "true_label", "label"),
        "probability": ("prediction_probability", "y_prob", "probability", "prediction_prob"),
        "prediction": ("prediction_label", "y_pred", "prediction"),
        "outer_fold": ("outer_fold", "fold"),
        "threshold": ("threshold", "fold_threshold"),
        "endpoint_horizon": ("endpoint_horizon_days", "endpoint_horizon", "horizon_days"),
    }
    output = pd.DataFrame(index=frame.index)
    for name, names in aliases.items():
        source = next((candidate for candidate in names if candidate in frame.columns), None)
        if source is not None:
            output[name] = frame[source]
    for optional in ("af_flag", "pvc_count_24h", "high_pvc", "high_pvc_burden", "high_pvc_flag"):
        if optional in frame.columns:
            output[optional] = frame[optional]
    for name in ("candidate", "profile", "model", "patient_id"):
        if name not in output:
            raise IntegrityError(f"OOF table must contain {name}")
        output[name] = output[name].map(lambda value: str(value).strip() if not _is_missing(value) else "")
        if (output[name] == "").any():
            raise IntegrityError(f"OOF {name} must not be empty")
    if "seed" not in output:
        raise IntegrityError("OOF table must contain seed")
    output["seed"] = output["seed"].map(lambda value: _parse_seed(value, label="OOF seed"))
    if "y_true" not in output:
        raise IntegrityError("OOF table must contain y_true")
    output["y_true"] = output["y_true"].map(lambda value: _binary_number(value, label="OOF y_true"))
    if "probability" not in output:
        raise IntegrityError("OOF table must contain prediction_probability/y_prob")
    output["probability"] = output["probability"].map(lambda value: _finite_number(value, label="OOF probability", lower=0.0, upper=1.0))
    if "prediction" in output:
        output["prediction"] = output["prediction"].map(lambda value: _binary_number(value, label="OOF prediction"))
    if "outer_fold" in output:
        output["outer_fold"] = output["outer_fold"].map(lambda value: _parse_seed(value, label="outer_fold"))
        if (output["outer_fold"] < 1).any():
            raise IntegrityError("OOF outer_fold must be positive")
    if "threshold" in output:
        output["threshold"] = output["threshold"].map(lambda value: _finite_number(value, label="OOF threshold", lower=0.0, upper=1.0))
    if "endpoint_horizon" in output:
        output["endpoint_horizon"] = output["endpoint_horizon"].map(
            lambda value: None if _is_missing(value) else _endpoint_number(value, label="OOF endpoint_horizon")
        )
    if "af_flag" in output:
        output["af_flag"] = output["af_flag"].map(
            lambda value: None if _is_missing(value) else _binary_number(value, label="OOF af_flag")
        )
    for column in ("high_pvc", "high_pvc_burden", "high_pvc_flag"):
        if column in output:
            output[column] = output[column].map(
                lambda value, name=column: None if _is_missing(value) else _binary_number(value, label=f"OOF {name}")
            )
    if "pvc_count_24h" in output:
        output["pvc_count_24h"] = output["pvc_count_24h"].map(
            lambda value: None if _is_missing(value) else _finite_number(value, label="OOF pvc_count_24h", lower=0.0)
        )
    return output


def _expected_seed_set(formal: bool, expected_seeds: Iterable[int] | None) -> set[int]:
    if formal:
        supplied = None if expected_seeds is None else {_parse_seed(value, label="expected seed") for value in expected_seeds}
        if supplied is not None and supplied != set(FORMAL_SEEDS):
            raise IntegrityError("formal mode requires expected seed set {0..99}")
        return set(FORMAL_SEEDS)
    if expected_seeds is None:
        raise IntegrityError("non-formal mode requires an explicit expected_seeds set")
    result = {_parse_seed(value, label="expected seed") for value in expected_seeds}
    if not result:
        raise IntegrityError("expected_seeds must not be empty")
    return result


def validate_repeated_seed_integrity(
    summary_records: Any,
    oof_tables: Any = None,
    *,
    formal: bool = False,
    expected_seeds: Iterable[int] | None = None,
    expected_candidates: Iterable[str] | None = None,
) -> dict[str, Any]:
    """Validate summaries, metadata homogeneity, and one-row-per-patient OOF."""

    summaries = _normalise_summary_records(summary_records)
    expected = _expected_seed_set(bool(formal), expected_seeds)
    actual_by_candidate = {str(candidate): set(group["seed"].tolist()) for candidate, group in summaries.groupby("candidate", sort=False)}
    if expected_candidates is not None:
        expected_candidates_set = {str(candidate).strip() for candidate in expected_candidates}
        actual_candidates_set = set(actual_by_candidate)
        if actual_candidates_set != expected_candidates_set:
            raise IntegrityError(
                f"candidate set mismatch; missing={sorted(expected_candidates_set - actual_candidates_set)}, "
                f"unknown={sorted(actual_candidates_set - expected_candidates_set)}"
            )
    for candidate, actual in actual_by_candidate.items():
        if actual != expected:
            missing = sorted(expected - actual)
            unknown = sorted(actual - expected)
            raise IntegrityError(f"candidate {candidate!r} seed set mismatch; missing={missing[:8]}, unknown={unknown[:8]}")
    oof = _normalise_oof_tables(oof_tables) if oof_tables is not None else pd.DataFrame()
    if formal and oof.empty:
        raise IntegrityError("formal mode requires OOF tables")
    keys = set(zip(summaries["candidate"], summaries["seed"]))
    if not oof.empty:
        oof_keys = set(zip(oof["candidate"], oof["seed"]))
        unknown = sorted(oof_keys - keys)
        missing = sorted(keys - oof_keys)
        if unknown:
            raise IntegrityError(f"OOF contains unknown candidate-seed pairs: {unknown[:8]}")
        if missing:
            raise IntegrityError(f"OOF missing candidate-seed pairs: {missing[:8]}")
        duplicate_mask = oof.duplicated(["candidate", "seed", "patient_id"], keep=False)
        if duplicate_mask.any():
            rows = oof.loc[duplicate_mask, ["candidate", "seed", "patient_id"]].head(8).to_dict("records")
            raise IntegrityError(f"OOF patient appears more than once in candidate-seed: {rows}")
        summary_lookup = summaries.set_index(["candidate", "seed"])
        for key, subset in oof.groupby(["candidate", "seed"], sort=False):
            meta = summary_lookup.loc[key]
            for field in ("profile", "model", "endpoint_horizon"):
                if field in subset and subset[field].notna().any():
                    values = {str(value) for value in subset[field].dropna().tolist()}
                    if len(values) > 1:
                        raise IntegrityError(f"mixed OOF {field} metadata within {key}")
                    if field != "endpoint_horizon" and values and str(meta[field]) not in values:
                        raise IntegrityError(f"OOF {field} disagrees with summary for {key}")
                    if field == "endpoint_horizon" and values and str(meta[field]) not in values:
                        raise IntegrityError(f"OOF endpoint_horizon disagrees with summary for {key}")
        # A missing patient in one run is not intrinsically invalid, but every
        # run must have at least one row and each row is already unique.
        if (oof.groupby(["candidate", "seed"], sort=False).size() < 1).any():  # pragma: no cover
            raise IntegrityError("empty OOF candidate-seed group")
    return {
        "formal": bool(formal),
        "expected_seeds": sorted(expected),
        "expected_candidates": sorted(actual_by_candidate if expected_candidates is None else expected_candidates_set),
        "candidates": sorted(actual_by_candidate),
        "summary_count": int(len(summaries)),
        "oof_rows": int(len(oof)),
        "oof_patient_counts": {
            f"{candidate}:{seed}": int(len(group))
            for (candidate, seed), group in oof.groupby(["candidate", "seed"], sort=True)
        },
        "summaries": summaries,
        "oof": oof,
    }


def _safe_auc(y: np.ndarray, p: np.ndarray) -> float:
    return float(roc_auc_score(y, p)) if np.unique(y).size == 2 else float("nan")


def _safe_ap(y: np.ndarray, p: np.ndarray) -> float:
    return float(average_precision_score(y, p)) if np.unique(y).size == 2 else float("nan")


def _calibration_values(y: np.ndarray, p: np.ndarray, n_bins: int = 10) -> dict[str, float]:
    edges = np.linspace(0.0, 1.0, int(n_bins) + 1)
    index = np.searchsorted(edges, p, side="right") - 1
    index[p == 1.0] = int(n_bins) - 1
    ece = 0.0
    mce = 0.0
    for bin_number in range(int(n_bins)):
        selected = index == bin_number
        if not selected.any():
            continue
        error = abs(float(np.mean(p[selected])) - float(np.mean(y[selected])))
        ece += float(selected.mean()) * error
        mce = max(mce, error)
    if np.unique(y).size == 2 and np.ptp(p) > 0:
        clipped = np.clip(p, 1e-6, 1 - 1e-6)
        x = np.log(clipped / (1 - clipped))
        slope, intercept = np.polyfit(x, y, 1)
    else:
        slope = intercept = float("nan")
    return {
        "calibration": float(ece),
        "calibration_ece": float(ece),
        "calibration_mce": float(mce),
        "calibration_slope": float(slope),
        "calibration_intercept": float(intercept),
    }


def _run_metric_row(group: pd.DataFrame, summary: Mapping[str, Any], n_bins: int) -> dict[str, Any]:
    y = group["y_true"].to_numpy(dtype=int)
    p = group["probability"].to_numpy(dtype=float)
    metrics = {
        "AUC": _safe_auc(y, p),
        "AP": _safe_ap(y, p),
        "Brier": float(brier_score_loss(y, p)),
    }
    # Explicit run summaries are authoritative, while OOF recomputation fills
    # summaries that only carry metadata.
    for name in PAIR_METRICS:
        if name in summary and not _is_missing(summary[name]):
            metrics[name] = _finite_number(summary[name], label=f"{name} summary")
    prevalence = float(np.mean(y))
    null_brier = prevalence * (1.0 - prevalence)
    metrics["BrierSkill"] = float(1.0 - metrics["Brier"] / null_brier) if null_brier > 0 else float("nan")
    metrics.update(_calibration_values(y, p, n_bins=n_bins))
    threshold_values = group["threshold"].to_numpy(dtype=float) if "threshold" in group else np.asarray([], dtype=float)
    if threshold_values.size:
        threshold = float(np.median(threshold_values))
    elif "threshold" in summary and not _is_missing(summary["threshold"]):
        threshold = _finite_number(summary["threshold"], label="threshold", lower=0.0, upper=1.0)
    else:
        threshold = float("nan")
    endpoint = summary.get("endpoint_horizon")
    return {
        "candidate": str(summary["candidate"]),
        "profile": summary["profile"],
        "model": summary["model"],
        "seed": int(summary["seed"]),
        "endpoint_horizon": endpoint,
        "endpoint_horizon_days": endpoint,
        "patient_count": int(len(group)),
        "positive_count": int(y.sum()),
        "negative_count": int(len(y) - y.sum()),
        "threshold": threshold,
        **metrics,
    }


def _distribution(values: Sequence[Any]) -> dict[str, float | None]:
    array = np.asarray([float(value) for value in values if value is not None and np.isfinite(float(value))], dtype=float)
    if array.size == 0:
        return {name: None for name in SUMMARY_STATISTICS}
    quantiles = np.percentile(array, [2.5, 25, 50, 75, 97.5])
    return {
        "mean": float(np.mean(array)),
        "std": float(np.std(array, ddof=0)),
        "min": float(np.min(array)),
        "p2.5": float(quantiles[0]),
        "p25": float(quantiles[1]),
        "median": float(quantiles[2]),
        "p75": float(quantiles[3]),
        "p97.5": float(quantiles[4]),
        "max": float(np.max(array)),
    }


def _build_summary(run_rows: pd.DataFrame, integrity: Mapping[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
    distribution_metrics = (*PAIR_METRICS, "BrierSkill", "calibration", "calibration_ece", "calibration_mce", "calibration_slope", "calibration_intercept", "threshold")
    output_rows: list[dict[str, Any]] = []
    structured: dict[str, Any] = {
        "formal": bool(integrity["formal"]),
        "expected_seeds": list(integrity["expected_seeds"]),
        "percentile_semantics": "seed-split sensitivity distributions, not confidence intervals",
        "candidates": {},
    }
    for candidate, group in run_rows.groupby("candidate", sort=True):
        first = group.iloc[0]
        row: dict[str, Any] = {
            "candidate": candidate,
            "profile": first["profile"],
            "model": first["model"],
            "endpoint_horizon": first.get("endpoint_horizon"),
            "seed_count": int(len(group)),
        }
        candidate_json: dict[str, Any] = {
            "candidate": candidate,
            "profile": first["profile"],
            "model": first["model"],
            "endpoint_horizon": first.get("endpoint_horizon"),
            "seed_count": int(len(group)),
            "metrics": {},
        }
        for metric in distribution_metrics:
            distribution = _distribution(group[metric].tolist())
            candidate_json["metrics"][metric] = distribution
            for statistic, value in distribution.items():
                row[f"{metric}_{statistic}"] = value
        for count_name in ("patient_count", "positive_count", "negative_count"):
            row[f"{count_name}_mean"] = float(np.mean(group[count_name]))
            candidate_json[count_name] = int(round(float(np.mean(group[count_name]))))
        candidate_json["thresholds"] = candidate_json["metrics"].get("threshold", {})
        structured["candidates"][candidate] = candidate_json
        output_rows.append(row)
    return pd.DataFrame(output_rows), structured


def compute_paired_comparison(
    runs: pd.DataFrame,
    *,
    baseline_candidate: str | None = None,
    baseline: str | None = None,
) -> dict[str, Any]:
    """Compare candidates with the baseline on identical seeds.

    Deltas are always candidate minus baseline.  AUC/AP favor positive values;
    Brier favors negative values.  Ties are not counted as a candidate win.
    """

    if baseline_candidate is not None and baseline is not None and baseline_candidate != baseline:
        raise ValueError("baseline and baseline_candidate disagree")
    if baseline_candidate is None:
        baseline_candidate = baseline
    if runs.empty:
        return {"baseline": baseline_candidate, "baseline_candidate": baseline_candidate, "comparisons": {}}
    runs = runs.copy(deep=True)
    for canonical, aliases in {
        "candidate": ("candidate", "candidate_id"),
        "seed": ("seed", "random_seed", "random_state"),
        "AUC": ("AUC", "auc", "roc_auc"),
        "AP": ("AP", "ap", "average_precision"),
        "Brier": ("Brier", "brier"),
    }.items():
        if canonical not in runs.columns:
            source = next((name for name in aliases if name in runs.columns), None)
            if source is None:
                raise ValueError(f"runs table must contain {canonical}")
            runs[canonical] = runs[source]
    runs["seed"] = runs["seed"].map(lambda value: _parse_seed(value, label="run seed"))
    for metric in PAIR_METRICS:
        runs[metric] = runs[metric].map(lambda value: _finite_number(value, label=f"run {metric}"))
    candidates = sorted(str(item) for item in runs["candidate"].dropna().unique())
    if baseline_candidate is None:
        inferred = next((item for item in candidates if item.lower() in {"baseline", "reference", "dummy"}), None)
        baseline_candidate = inferred
    if baseline_candidate is None or baseline_candidate not in candidates:
        return {
            "baseline": baseline_candidate,
            "baseline_candidate": baseline_candidate,
            "comparisons": {},
            "status": "NO_BASELINE",
        }
    baseline = runs.loc[runs["candidate"] == baseline_candidate].set_index("seed")
    result: dict[str, Any] = {
        "baseline": baseline_candidate,
        "baseline_candidate": baseline_candidate,
        "comparisons": {},
    }
    for candidate in candidates:
        if candidate == baseline_candidate:
            continue
        current = runs.loc[runs["candidate"] == candidate].set_index("seed")
        common = sorted(set(current.index) & set(baseline.index))
        per_seed: list[dict[str, Any]] = []
        for seed in common:
            row: dict[str, Any] = {"seed": int(seed)}
            for metric in PAIR_METRICS:
                left = float(current.loc[seed, metric])
                right = float(baseline.loc[seed, metric])
                delta = left - right
                row[f"{metric}_candidate_minus_baseline"] = delta
                row[f"{metric}_delta"] = delta
            per_seed.append(row)
        comparison: dict[str, Any] = {
            "candidate": candidate,
            "baseline": baseline_candidate,
            "matched_seed_count": len(per_seed),
            "per_seed": per_seed,
            "deltas": {metric: [row[f"{metric}_delta"] for row in per_seed] for metric in PAIR_METRICS},
        }
        fractions: dict[str, float | None] = {}
        for metric in PAIR_METRICS:
            values = np.asarray(comparison["deltas"][metric], dtype=float)
            orientation = -1.0 if metric == "Brier" else 1.0
            favorable = orientation * values > 0
            fractions[metric] = float(np.mean(favorable)) if values.size else None
            comparison[f"{metric}_mean_delta"] = float(np.mean(values)) if values.size else None
            comparison[f"{metric}_std_delta"] = float(np.std(values, ddof=1)) if values.size > 1 else None
            comparison[f"{metric}_median_delta"] = float(np.median(values)) if values.size else None
            for label, quantile in (("p2.5", 0.025), ("p25", 0.25), ("p75", 0.75), ("p97.5", 0.975)):
                comparison[f"{metric}_{label}_delta"] = float(np.quantile(values, quantile)) if values.size else None
            comparison[f"candidate_better_fraction_{metric}"] = fractions[metric]
        comparison["candidate_better_fraction"] = fractions
        result["comparisons"][candidate] = comparison
        # Keep a direct candidate-key view for simple consumers while the
        # namespaced ``comparisons`` view remains the canonical structure.
        result[candidate] = comparison
    return result


def _calibration_tables(oof: pd.DataFrame, *, n_bins: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    if oof.empty:
        return pd.DataFrame(), pd.DataFrame()
    bin_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    edges = np.linspace(0.0, 1.0, int(n_bins) + 1)
    for candidate, group in oof.groupby("candidate", sort=True):
        y = group["y_true"].to_numpy(dtype=int)
        p = group["probability"].to_numpy(dtype=float)
        bin_index = np.searchsorted(edges, p, side="right") - 1
        bin_index[p == 1.0] = int(n_bins) - 1
        for number in range(int(n_bins)):
            selected = bin_index == number
            count = int(selected.sum())
            observed = float(np.mean(y[selected])) if count else float("nan")
            predicted = float(np.mean(p[selected])) if count else float("nan")
            bin_rows.append(
                {
                    "candidate": candidate,
                    "bin": number,
                    "bin_lower": float(edges[number]),
                    "bin_upper": float(edges[number + 1]),
                    "count": count,
                    "positive_count": int(y[selected].sum()) if count else 0,
                    "mean_predicted": predicted,
                    "observed_fraction": observed,
                    "absolute_error": abs(predicted - observed) if count else float("nan"),
                }
            )
        values = _calibration_values(y, p, n_bins=n_bins)
        summary_rows.append(
            {
                "candidate": candidate,
                "pooled_oof_count": len(group),
                "AUC": _safe_auc(y, p),
                "AP": _safe_ap(y, p),
                "Brier": float(brier_score_loss(y, p)),
                **values,
            }
        )
    return pd.DataFrame(summary_rows), pd.DataFrame(bin_rows)


def compute_calibration_bins(oof: Any, *, n_bins: int = 10) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return pooled-OOF calibration summary and fixed-width bin tables."""

    frame = _normalise_oof_tables(oof)
    return _calibration_tables(frame, n_bins=int(n_bins))


def _subgroup_table(oof: pd.DataFrame, *, column: str, output_name: str) -> pd.DataFrame:
    if column not in oof.columns:
        return pd.DataFrame()
    work = oof.copy()
    work[column] = work[column].map(lambda value: None if _is_missing(value) else value)
    work = work.loc[work[column].notna()].copy()
    if work.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for (candidate, level), group in work.groupby(["candidate", column], sort=True):
        y = group["y_true"].to_numpy(dtype=int)
        p = group["probability"].to_numpy(dtype=float)
        rows.append(
            {
                "candidate": candidate,
                "subgroup": output_name,
                "level": str(level),
                "n": len(group),
                "positive_count": int(y.sum()),
                "AUC": _safe_auc(y, p),
                "AP": _safe_ap(y, p),
                "Brier": float(brier_score_loss(y, p)),
                **_calibration_values(y, p),
                "exploratory": True,
            }
        )
    return pd.DataFrame(rows)


def _promotion_status(paired: Mapping[str, Any]) -> tuple[str, str | None]:
    comparisons = paired.get("comparisons", {}) if isinstance(paired, Mapping) else {}
    if not comparisons or not paired.get("baseline"):
        return "NO_CLEAR_WINNER", None
    eligible: list[tuple[str, float]] = []
    for candidate, comparison in comparisons.items():
        fractions = comparison.get("candidate_better_fraction", {})
        means = {metric: comparison.get(f"{metric}_mean_delta") for metric in PAIR_METRICS}
        if all(value is not None and math.isfinite(float(value)) for value in means.values()) and all(
            fractions.get(metric) is not None and float(fractions[metric]) >= 0.5 for metric in PAIR_METRICS
        ) and means["AUC"] > 0 and means["AP"] > 0 and means["Brier"] < 0:
            score = float(means["AUC"] + means["AP"] - means["Brier"])
            eligible.append((candidate, score))
    if eligible:
        return "PROMOTED_CANDIDATE", sorted(eligible, key=lambda item: (-item[1], item[0]))[0][0]
    return "BASELINE_RETAINED", None


def _legacy_generate_final_report(
    target_dir: str | Path,
    *,
    integrity: Mapping[str, Any],
    summary: Mapping[str, Any],
    paired: Mapping[str, Any],
    calibration_summary: pd.DataFrame | None = None,
    report_inputs: Mapping[str, Any] | None = None,
) -> tuple[Path, Path]:
    """Write the fixed ten-section Markdown report and machine JSON report."""

    root = Path(target_dir) / "reports"
    root.mkdir(parents=True, exist_ok=True)
    status, promoted = _promotion_status(paired)
    inputs = dict(report_inputs or {})
    objective = str(inputs.get("objective", "Repeated-seed optimization artifact review.")).replace("clinical utility", "clinical-use claims")
    baseline = paired.get("baseline")
    lines = [
        "# Model Optimization Report",
        "",
        f"Promotion status: **{status}**",
        "",
        f"## {REQUIRED_REPORT_HEADINGS[0]}",
        "",
        objective,
        "This report describes predictive performance, calibration, and split sensitivity; it does not establish clinical use.",
        "",
        f"## {REQUIRED_REPORT_HEADINGS[1]}",
        "",
        f"Baseline: `{baseline or 'not specified'}`. Promoted candidate: `{promoted or 'none'}`.",
        "",
        f"## {REQUIRED_REPORT_HEADINGS[2]}",
        "",
        f"Formal mode: `{bool(integrity.get('formal'))}`; candidates: `{len(integrity.get('candidates', []))}`; summary rows: `{integrity.get('summary_count', 0)}`; OOF rows: `{integrity.get('oof_rows', 0)}`.",
        "",
        f"## {REQUIRED_REPORT_HEADINGS[3]}",
        "",
        "Each seed is a repeated patient-level split. Percentiles are split-sensitivity summaries and are not confidence intervals.",
        "",
        f"## {REQUIRED_REPORT_HEADINGS[4]}",
        "",
        "| Candidate | Seeds | AUC median | AP median | Brier median | BrierSkill median | Calibration median |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for candidate, details in summary.get("candidates", {}).items():
        metrics = details.get("metrics", {})
        lines.append(
            f"| {candidate} | {details.get('seed_count', '')} | {_fmt(metrics.get('AUC', {}).get('median'))} | {_fmt(metrics.get('AP', {}).get('median'))} | {_fmt(metrics.get('Brier', {}).get('median'))} | {_fmt(metrics.get('BrierSkill', {}).get('median'))} | {_fmt(metrics.get('calibration', {}).get('median'))} |"
        )
    lines.extend(
        [
            "",
            f"## {REQUIRED_REPORT_HEADINGS[5]}",
            "",
            "Deltas are candidate minus baseline. Positive AUC/AP deltas and negative Brier deltas favor the candidate.",
            "",
        ]
    )
    for candidate, comparison in paired.get("comparisons", {}).items():
        lines.append(
            f"- `{candidate}`: matched seeds={comparison.get('matched_seed_count', 0)}, AUC Δ={_fmt(comparison.get('AUC_mean_delta'))}, AP Δ={_fmt(comparison.get('AP_mean_delta'))}, Brier Δ={_fmt(comparison.get('Brier_mean_delta'))}."
        )
    lines.extend(
        [
            "",
            f"## {REQUIRED_REPORT_HEADINGS[6]}",
            "",
            "Calibration summaries and deterministic pooled-OOF bins are emitted as CSV artifacts.",
            "",
            f"## {REQUIRED_REPORT_HEADINGS[7]}",
            "",
            "AF and PVC subgroup results, when present, are exploratory and are not used as standalone promotion evidence.",
            "",
            f"## {REQUIRED_REPORT_HEADINGS[8]}",
            "",
            f"Decision: **{status}**. The decision is based on paired repeated-seed discrimination and Brier orientation rules; no claim beyond this evaluation is made.",
            "",
            f"## {REQUIRED_REPORT_HEADINGS[9]}",
            "",
            "Review the exact seed set, metadata, patient uniqueness, and generated CSV/JSON artifacts before using the result in another analysis.",
            "",
        ]
    )
    markdown = root / "MODEL_OPTIMIZATION.md"
    markdown.write_text("\n".join(lines), encoding="utf-8")
    machine = {
        "promotion_status": status,
        "promoted_candidate": promoted,
        "baseline": baseline,
        "integrity": {key: value for key, value in integrity.items() if key not in {"summaries", "oof"}},
        "summary": summary,
        "paired_comparison": paired,
    }
    machine_path = root / "model_optimization.json"
    _write_json(machine_path, machine)
    return markdown, machine_path


def generate_final_report(
    target_dir: str | Path,
    *,
    integrity: Mapping[str, Any],
    summary: Mapping[str, Any],
    paired: Mapping[str, Any],
    calibration_summary: pd.DataFrame | None = None,
    report_inputs: Mapping[str, Any] | None = None,
) -> tuple[Path, Path]:
    """Write the required ten-section report without auto-promoting a model."""

    root = Path(target_dir) / "reports"
    root.mkdir(parents=True, exist_ok=True)
    inputs = dict(report_inputs or {})
    status = str(inputs.get("promotion_status", "NO_CLEAR_WINNER"))
    if status not in {"PROMOTED_CANDIDATE", "BASELINE_RETAINED", "NO_CLEAR_WINNER"}:
        raise ValueError(f"invalid promotion_status: {status}")
    promoted = inputs.get("promoted_candidate") if status == "PROMOTED_CANDIDATE" else None
    if status == "PROMOTED_CANDIDATE" and not promoted:
        raise ValueError("PROMOTED_CANDIDATE requires promoted_candidate")
    baseline = paired.get("baseline")
    lines = [
        "# MODEL OPTIMIZATION V1",
        "",
        f"Promotion status: **{status}**",
        "",
        f"## {REQUIRED_REPORT_HEADINGS[0]}",
        "",
        f"Frozen comparison candidate: `{baseline or 'not specified'}`. This evaluation does not establish clinical use.",
        "",
        f"## {REQUIRED_REPORT_HEADINGS[1]}",
        "",
        "The 100-, 20-, 40-, and inner-fold-selected representations are compared below using patient-level nested CV.",
        "",
        f"## {REQUIRED_REPORT_HEADINGS[2]}",
        "",
        "AF-compatible candidates use only audited rhythm-safe features; low-event AF subgroup estimates are exploratory.",
        "",
        f"## {REQUIRED_REPORT_HEADINGS[3]}",
        "",
        "P1/P2 are seed-paired and differ only by source `pvc_count_24h`; the time-base-mismatched derived burden is excluded.",
        "",
        f"## {REQUIRED_REPORT_HEADINGS[4]}",
        "",
        "Brier, Brier Skill Score, calibration intercept/slope, and deterministic calibration bins are emitted as artifacts.",
        "",
        f"## {REQUIRED_REPORT_HEADINGS[5]}",
        "",
        f"Formal mode: `{bool(integrity.get('formal'))}`; summary rows: `{integrity.get('summary_count', 0)}`; OOF rows: `{integrity.get('oof_rows', 0)}`.",
        "The 2.5–97.5 percentiles are split-sensitivity distributions, not confidence intervals.",
        "",
        "| Candidate | Seeds | AUC median | AP median | Brier median | BrierSkill median |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for candidate, details in summary.get("candidates", {}).items():
        metrics = details.get("metrics", {})
        lines.append(
            f"| {candidate} | {details.get('seed_count', '')} | {_fmt(metrics.get('AUC', {}).get('median'))} | {_fmt(metrics.get('AP', {}).get('median'))} | {_fmt(metrics.get('Brier', {}).get('median'))} | {_fmt(metrics.get('BrierSkill', {}).get('median'))} |"
        )
    lines.extend(["", "Paired deltas are candidate minus baseline; positive AUC/AP and negative Brier favor the candidate.", ""])
    for candidate, comparison in paired.get("comparisons", {}).items():
        lines.append(
            f"- `{candidate}`: matched seeds={comparison.get('matched_seed_count', 0)}, median AUC delta={_fmt(comparison.get('AUC_median_delta'))}, median AP delta={_fmt(comparison.get('AP_median_delta'))}, median Brier delta={_fmt(comparison.get('Brier_median_delta'))}."
        )
    lines.extend(
        [
            "",
            f"## {REQUIRED_REPORT_HEADINGS[6]}",
            "",
            "The prespecified primary endpoint is 365 d; 90/180/730 d results are sensitivity analyses only.",
            "",
            f"## {REQUIRED_REPORT_HEADINGS[7]}",
            "",
            f"Decision: **{status}**. Promoted candidate: `{promoted or 'none'}`. Promotion requires composite Root review.",
            "",
            f"## {REQUIRED_REPORT_HEADINGS[8]}",
            "",
            "The event count is small; subgroup estimates are exploratory; split percentiles are not population confidence intervals.",
            "",
            f"## {REQUIRED_REPORT_HEADINGS[9]}",
            "",
            "The advisor-facing conclusion must follow exact seed, patient, paired metric, AF recovery, PVC, calibration, and endpoint artifact review.",
            "",
        ]
    )
    markdown = root / "MODEL_OPTIMIZATION.md"
    markdown.write_text("\n".join(lines), encoding="utf-8")
    machine = {
        "promotion_status": status,
        "promoted_candidate": promoted,
        "baseline": baseline,
        "integrity": {key: value for key, value in integrity.items() if key not in {"summaries", "oof"}},
        "summary": summary,
        "paired_comparison": paired,
        "report_inputs": inputs,
    }
    machine_path = root / "model_optimization.json"
    _write_json(machine_path, machine)
    return markdown, machine_path


def _fmt(value: Any) -> str:
    if value is None or _is_missing(value):
        return "n/a"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    return "n/a" if not math.isfinite(number) else f"{number:.4f}"


def aggregate_optimization_artifacts(
    summary_records: Any = None,
    oof_tables: Any = None,
    *,
    artifact_dir: str | Path | None = None,
    target_dir: str | Path | None = None,
    formal: bool = False,
    expected_seeds: Iterable[int] | None = None,
    expected_seed_set: Iterable[int] | None = None,
    expected_candidates: Iterable[str] | None = None,
    baseline_candidate: str | None = None,
    calibration_bins: int = 10,
    report_inputs: Mapping[str, Any] | None = None,
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Validate and aggregate optimization artifacts, optionally writing files."""

    if int(calibration_bins) < 2:
        raise ValueError("calibration_bins must be at least two")
    if expected_seeds is not None:
        expected_seeds = list(expected_seeds)
    if expected_seed_set is not None:
        expected_seed_set = list(expected_seed_set)
    if expected_seeds is not None and expected_seed_set is not None:
        if {_parse_seed(value) for value in expected_seeds} != {_parse_seed(value) for value in expected_seed_set}:
            raise ValueError("expected_seeds and expected_seed_set disagree")
    if expected_seeds is None:
        expected_seeds = expected_seed_set
    if target_dir is not None and output_dir is not None and Path(target_dir) != Path(output_dir):
        raise ValueError("target_dir and output_dir disagree")
    if target_dir is None:
        target_dir = output_dir
    if artifact_dir is not None:
        if summary_records is not None or oof_tables is not None:
            raise ValueError("artifact_dir cannot be combined with explicit records")
        summary_records, oof_tables = load_optimization_artifacts(artifact_dir)
    if summary_records is None:
        raise ValueError("summary_records or artifact_dir is required")
    integrity = validate_repeated_seed_integrity(
        summary_records,
        oof_tables,
        formal=bool(formal),
        expected_seeds=expected_seeds,
        expected_candidates=expected_candidates,
    )
    summaries = integrity["summaries"]
    oof = integrity["oof"]
    summary_lookup = summaries.set_index(["candidate", "seed"]).to_dict(orient="index")
    # ``to_dict(orient='index')`` omits index values; retain them because the
    # per-run output is deliberately self-describing.
    for key, value in list(summary_lookup.items()):
        value.setdefault("candidate", key[0])
        value.setdefault("seed", key[1])
    run_rows: list[dict[str, Any]] = []
    if oof.empty:
        # Summary-only mode remains useful for a precomputed run table, but it
        # cannot fabricate patient-level calibration or subgroup artifacts.
        for _, row in summaries.iterrows():
            run = dict(row)
            for metric in PAIR_METRICS:
                run[metric] = run.get(metric, np.nan)
            run.setdefault("BrierSkill", np.nan)
            for key in ("calibration", "calibration_ece", "calibration_mce", "calibration_slope", "calibration_intercept", "threshold"):
                run.setdefault(key, np.nan)
            run_rows.append(run)
    else:
        for key, group in oof.groupby(["candidate", "seed"], sort=True):
            run_rows.append(_run_metric_row(group, summary_lookup[key], int(calibration_bins)))
    runs = pd.DataFrame(run_rows).sort_values(["candidate", "seed"], kind="stable").reset_index(drop=True)
    summary_frame, summary_json = _build_summary(runs, integrity)
    paired = compute_paired_comparison(runs, baseline_candidate=baseline_candidate)
    calibration_summary, calibration_bins_frame = _calibration_tables(oof, n_bins=int(calibration_bins))
    af_subgroup = _subgroup_table(oof, column="af_flag", output_name="af")
    pvc_column = next((column for column in ("high_pvc", "high_pvc_burden", "high_pvc_flag") if column in oof.columns), None)
    if pvc_column is None and "pvc_count_24h" in oof.columns:
        pvc_values = pd.to_numeric(oof["pvc_count_24h"], errors="coerce")
        if pvc_values.notna().any():
            oof = oof.copy()
            oof["pvc_present"] = np.where(pvc_values.notna(), pvc_values.gt(0).astype(int), np.nan)
            pvc_column = "pvc_present"
    pvc_subgroup = _subgroup_table(oof, column=pvc_column, output_name="pvc") if pvc_column else pd.DataFrame()
    output_paths: dict[str, str] = {}
    if target_dir is not None:
        target = Path(target_dir)
        target.mkdir(parents=True, exist_ok=True)
        runs_csv = target / "runs.csv"
        runs.to_csv(runs_csv, index=False, na_rep="")
        runs_parquet = target / "runs.parquet"
        runs.to_parquet(runs_parquet, index=False)
        summary_csv = target / "summary.csv"
        summary_frame.to_csv(summary_csv, index=False, na_rep="")
        summary_json_path = target / "summary.json"
        _write_json(summary_json_path, summary_json)
        paired_path = target / "paired_comparison.json"
        _write_json(paired_path, paired)
        calibration_summary_path = target / "calibration_summary.csv"
        calibration_summary.to_csv(calibration_summary_path, index=False, na_rep="")
        calibration_bins_path = target / "calibration_bins.csv"
        calibration_bins_frame.to_csv(calibration_bins_path, index=False, na_rep="")
        output_paths.update(
            {
                "runs_csv": str(runs_csv),
                "runs_parquet": str(runs_parquet),
                "summary_csv": str(summary_csv),
                "summary_json": str(summary_json_path),
                "paired_comparison": str(paired_path),
                "calibration_summary": str(calibration_summary_path),
                "calibration_bins": str(calibration_bins_path),
            }
        )
        if not af_subgroup.empty:
            path = target / "af_subgroup.csv"
            af_subgroup.to_csv(path, index=False, na_rep="")
            output_paths["af_subgroup"] = str(path)
        if not pvc_subgroup.empty:
            path = target / "pvc_subgroup.csv"
            pvc_subgroup.to_csv(path, index=False, na_rep="")
            output_paths["pvc_subgroup"] = str(path)
        if report_inputs is not None:
            markdown, machine = generate_final_report(
                target,
                integrity=integrity,
                summary=summary_json,
                paired=paired,
                calibration_summary=calibration_summary,
                report_inputs=report_inputs,
            )
            output_paths["report_markdown"] = str(markdown)
            output_paths["report_json"] = str(machine)
    return {
        "integrity": {key: value for key, value in integrity.items() if key not in {"summaries", "oof"}},
        "runs": runs,
        "summary": summary_frame,
        "summary_json": summary_json,
        "paired_comparison": paired,
        "calibration_summary": calibration_summary,
        "calibration_bins": calibration_bins_frame,
        "af_subgroup": af_subgroup,
        "pvc_subgroup": pvc_subgroup,
        "output_paths": output_paths,
    }


# Friendly names used by scripts and notebooks.
aggregate_artifacts = aggregate_optimization_artifacts
aggregate_optimization_runs = aggregate_optimization_artifacts
aggregate_optimization_results = aggregate_optimization_artifacts
validate_artifact_integrity = validate_repeated_seed_integrity
paired_candidate_comparison = compute_paired_comparison
compare_candidates = compute_paired_comparison
compute_paired_comparisons = compute_paired_comparison
aggregate_reports = aggregate_optimization_artifacts
write_final_report = generate_final_report
build_optimization_report = generate_final_report
summarize_runs = aggregate_optimization_artifacts


__all__ = [
    "FORMAL_SEEDS",
    "SUMMARY_STATISTICS",
    "PAIR_METRICS",
    "REQUIRED_REPORT_HEADINGS",
    "IntegrityError",
    "load_optimization_artifacts",
    "validate_repeated_seed_integrity",
    "validate_artifact_integrity",
    "compute_paired_comparison",
    "compute_paired_comparisons",
    "paired_candidate_comparison",
    "compute_calibration_bins",
    "generate_final_report",
    "write_final_report",
    "aggregate_optimization_artifacts",
    "aggregate_optimization_runs",
    "aggregate_optimization_results",
    "aggregate_artifacts",
    "compare_candidates",
    "build_optimization_report",
    "summarize_runs",
    "aggregate_reports",
]
