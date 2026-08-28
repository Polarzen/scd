"""Import the immutable formal B0 baseline into the current run layout.

The formal B0 repeated-CV artifacts were produced by a successful historical
Actions run.  This adapter makes their provenance explicit, validates the
patient-level contract, recomputes metrics from OOF predictions, and writes
the same per-run files consumed by :mod:`src.optimization_reporting`.

This module deliberately does not fit a model and does not derive endpoint
features.  AF/PVC columns are copied only from the current audited B0
population manifest, joined by ``patient_id``.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd

from src.metrics import compute_metrics


LEGACY_RUN_ID = "33141323899"
LEGACY_HEAD_SHA = "d4ddd52b759e60b03db0b41307ba57c331605882"
EXPECTED_SEEDS = tuple(range(100))
EXPECTED_ENDPOINT = 365
EXPECTED_PATIENT_COUNT = 703
EXPECTED_POSITIVE_COUNT = 27
EXPECTED_NEGATIVE_COUNT = 676
EXPECTED_OUTER_FOLDS = 5
EXPECTED_INNER_FOLDS = 3
EXPECTED_PROFILE = "all20"
OUTPUT_PROFILE = "all20_100"
EXPECTED_MODEL = "extratrees"
OUTPUT_CANDIDATE = "B0"


class BaselineImportError(ValueError):
    """Raised when a legacy artifact violates the formal B0 contract."""


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    try:
        result = pd.isna(value)
    except (TypeError, ValueError):
        return False
    return bool(result) if not isinstance(result, (list, tuple, np.ndarray, pd.Series)) else False


def _as_int(value: Any, *, label: str) -> int:
    if _is_missing(value) or isinstance(value, bool):
        raise BaselineImportError(f"{label} must be an integer")
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise BaselineImportError(f"{label} must be an integer") from exc
    if not math.isfinite(number) or not number.is_integer():
        raise BaselineImportError(f"{label} must be an integer")
    return int(number)


def _finite(value: Any, *, label: str, lower: float | None = None, upper: float | None = None) -> float:
    if _is_missing(value):
        raise BaselineImportError(f"{label} is missing")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise BaselineImportError(f"{label} must be numeric") from exc
    if not math.isfinite(result):
        raise BaselineImportError(f"{label} must be finite")
    if lower is not None and result < lower:
        raise BaselineImportError(f"{label} must be >= {lower}")
    if upper is not None and result > upper:
        raise BaselineImportError(f"{label} must be <= {upper}")
    return result


def _binary(value: Any, *, label: str, allow_missing: bool = False) -> int | None:
    if _is_missing(value):
        if allow_missing:
            return None
        raise BaselineImportError(f"{label} is missing")
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"true", "yes", "y", "1"}:
            return 1
        if text in {"false", "no", "n", "0"}:
            return 0
    number = _finite(value, label=label)
    if number not in {0.0, 1.0}:
        raise BaselineImportError(f"{label} must be 0 or 1")
    return int(number)


def _first(row: Mapping[str, Any], names: Iterable[str], *, label: str, required: bool = True) -> Any:
    for name in names:
        if name in row and not _is_missing(row[name]):
            return row[name]
    if required:
        raise BaselineImportError(f"{label} is required")
    return None


def _normal_model(value: Any, *, label: str) -> str:
    text = str(value).strip().lower().replace("-", "").replace("_", "").replace(" ", "")
    if text not in {"extratrees", "extratreesclassifier"}:
        raise BaselineImportError(f"{label} must identify ExtraTrees, got {value!r}")
    return EXPECTED_MODEL


def _normal_profile(value: Any, *, label: str) -> str:
    text = str(value).strip().lower().replace("-", "_").replace(" ", "_")
    if text != EXPECTED_PROFILE:
        raise BaselineImportError(f"{label} must be the legacy all20 profile, got {value!r}")
    return OUTPUT_PROFILE


def _read_table(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise BaselineImportError(f"manifest does not exist: {path}")
    suffix = path.suffix.lower()
    try:
        if suffix == ".csv":
            return pd.read_csv(path)
        if suffix in {".parquet", ".pq"}:
            return pd.read_parquet(path)
        if suffix == ".json":
            value = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(value, Mapping) and isinstance(value.get("table"), list):
                value = value["table"]
            if isinstance(value, Mapping) and isinstance(value.get("records"), list):
                value = value["records"]
            if not isinstance(value, list):
                raise BaselineImportError("JSON manifest must contain a list of patient records")
            return pd.DataFrame(value)
    except BaselineImportError:
        raise
    except Exception as exc:  # pragma: no cover - pandas/json gives details
        raise BaselineImportError(f"unable to read manifest: {path}") from exc
    raise BaselineImportError(f"unsupported manifest format: {path}")


def _load_manifest(path: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    frame = _read_table(path)
    if "patient_id" not in frame.columns:
        raise BaselineImportError("current B0 manifest must contain patient_id")
    work = frame.copy(deep=True)
    work["patient_id"] = work["patient_id"].astype("string").str.strip()
    if work["patient_id"].isna().any() or (work["patient_id"] == "").any() or work["patient_id"].duplicated().any():
        raise BaselineImportError("current B0 manifest must contain unique, non-empty patient_id")
    label_column = next((name for name in ("label", "y_true", "true_label", "binary_label_if_evaluable") if name in work.columns), None)
    if label_column is None:
        raise BaselineImportError("current B0 manifest must contain label")
    included_column = next((name for name in ("included", "selected", "selected_population") if name in work.columns), None)
    if included_column is not None:
        if included_column == "selected_population":
            included = work[included_column].astype("string").str.strip().str.lower().eq("primary")
        else:
            included = work[included_column].map(lambda value: bool(_binary(value, label="manifest included")))
        selected = work.loc[included].copy()
    else:
        if len(work) != EXPECTED_PATIENT_COUNT:
            raise BaselineImportError("manifest without included column must contain exactly 703 patients")
        selected = work.copy()
    if len(selected) != EXPECTED_PATIENT_COUNT:
        raise BaselineImportError(f"current B0 manifest selected population must contain {EXPECTED_PATIENT_COUNT} patients")
    # Endpoint labels are required only for the selected B0 population.  The
    # audited full manifest intentionally retains censored/competing subjects
    # with missing binary labels, and those rows must not be coerced before
    # the fold-independent inclusion filter is applied.
    selected["_label"] = selected[label_column].map(lambda value: _binary(value, label="manifest label"))
    labels = selected["_label"].to_numpy(dtype=int)
    if int(labels.sum()) != EXPECTED_POSITIVE_COUNT or int((labels == 0).sum()) != EXPECTED_NEGATIVE_COUNT:
        raise BaselineImportError("current B0 manifest population counts must be 27 positive and 676 negative")
    selected = selected.set_index("patient_id", drop=False)
    optional = [name for name in ("af_flag", "pvc_count_24h", "high_pvc_burden", "high_pvc_flag") if name in selected.columns]
    for name in optional:
        if name == "pvc_count_24h":
            values = pd.to_numeric(selected[name], errors="coerce")
            bad = values.notna() & (~np.isfinite(values.to_numpy(dtype=float)) | values.lt(0))
            if bool(bad.any()):
                raise BaselineImportError("manifest pvc_count_24h must be finite and non-negative")
            selected[name] = values
        else:
            selected[name] = selected[name].map(lambda value, column=name: _binary(value, label=f"manifest {column}", allow_missing=True))
    manifest = {
        "candidate": OUTPUT_CANDIDATE,
        "profile": OUTPUT_PROFILE,
        "model": EXPECTED_MODEL,
        "selected_population": "primary",
        "selected_ids": selected["patient_id"].astype(str).tolist(),
        "counts": {
            "patient_count": EXPECTED_PATIENT_COUNT,
            "positive_count": EXPECTED_POSITIVE_COUNT,
            "negative_count": EXPECTED_NEGATIVE_COUNT,
        },
        "legacy_run_id": LEGACY_RUN_ID,
        "legacy_head_sha": LEGACY_HEAD_SHA,
    }
    return selected, manifest


def _artifact_dirs(root: Path) -> dict[int, Path]:
    if not root.is_dir():
        raise BaselineImportError(f"legacy artifact root does not exist: {root}")
    pattern = re.compile(r"^formal-repeated-all20-seed-(\d+)$")
    found: list[tuple[int, Path]] = []
    for path in root.rglob("*"):
        if not path.is_dir() or not path.name.startswith("formal-repeated-all20-seed-"):
            continue
        match = pattern.fullmatch(path.name)
        if match is None:
            raise BaselineImportError(f"malformed legacy artifact directory: {path.name}")
        found.append((int(match.group(1)), path))
    by_seed: dict[int, list[Path]] = {}
    for seed, path in found:
        by_seed.setdefault(seed, []).append(path)
    duplicate = {seed: paths for seed, paths in by_seed.items() if len(paths) != 1}
    if duplicate:
        raise BaselineImportError(f"duplicate legacy artifact seed(s): {sorted(duplicate)}")
    actual = set(by_seed)
    expected = set(EXPECTED_SEEDS)
    if actual != expected:
        raise BaselineImportError(f"legacy artifact seed set mismatch; missing={sorted(expected - actual)}, unknown={sorted(actual - expected)}")
    return {seed: paths[0] for seed, paths in by_seed.items()}


def _single_summary(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise BaselineImportError(f"unable to read legacy summary: {path}") from exc
    if isinstance(payload, Mapping):
        for key in ("per_seed", "records", "data", "rows"):
            if key in payload:
                payload = payload[key]
                break
    if isinstance(payload, Mapping):
        rows = [dict(payload)]
    elif isinstance(payload, list) and all(isinstance(item, Mapping) for item in payload):
        rows = [dict(item) for item in payload]
    else:
        raise BaselineImportError(f"legacy per_seed JSON must contain summary records: {path}")
    if len(rows) != 1:
        raise BaselineImportError(f"legacy per_seed JSON must contain exactly one model/seed record: {path}")
    return rows[0]


def _find_one(directory: Path, name: str) -> Path:
    matches = [path for path in directory.rglob(name) if path.is_file()]
    if len(matches) != 1:
        raise BaselineImportError(f"legacy artifact must contain exactly one {name}: {directory}")
    return matches[0]


def _legacy_metadata(row: Mapping[str, Any], artifact_seed: int) -> dict[str, Any]:
    candidate = _first(row, ("candidate", "candidate_id"), label="legacy candidate", required=False)
    if candidate is not None and str(candidate).strip().upper() != OUTPUT_CANDIDATE:
        raise BaselineImportError(f"legacy summary candidate must be B0, got {candidate!r}")
    seed = _as_int(_first(row, ("seed", "random_seed", "random_state"), label="legacy summary seed"), label="legacy summary seed")
    if seed != artifact_seed:
        raise BaselineImportError(f"legacy summary seed {seed} disagrees with artifact seed {artifact_seed}")
    model = _normal_model(_first(row, ("model", "model_name", "estimator"), label="legacy summary model"), label="legacy summary model")
    profile_value = _first(row, ("profile", "analysis_profile"), label="legacy summary profile")
    profile = _normal_profile(profile_value, label="legacy summary profile")
    endpoint = _first(row, ("endpoint_horizon_days", "endpoint_horizon", "horizon_days"), label="legacy summary endpoint", required=False)
    if endpoint is not None and _as_int(endpoint, label="legacy summary endpoint") != EXPECTED_ENDPOINT:
        raise BaselineImportError("legacy summary endpoint must be 365 days")
    population = _first(row, ("population", "selected_population"), label="legacy summary population", required=False)
    if population is not None and str(population).strip().lower() not in {"primary", "baseline", "model"}:
        raise BaselineImportError(f"legacy summary population must be primary, got {population!r}")
    for name, expected in (
        ("patient_count", EXPECTED_PATIENT_COUNT),
        ("positive_count", EXPECTED_POSITIVE_COUNT),
        ("negative_count", EXPECTED_NEGATIVE_COUNT),
        ("outer_folds", EXPECTED_OUTER_FOLDS),
        ("inner_folds", EXPECTED_INNER_FOLDS),
    ):
        value = _first(row, (name,), label=name, required=False)
        if value is not None and _as_int(value, label=name) != expected:
            raise BaselineImportError(f"legacy summary {name} must be {expected}")
    feature_count = _first(row, ("feature_count",), label="feature_count", required=False)
    if feature_count is not None and _as_int(feature_count, label="feature_count") != 100:
        raise BaselineImportError("legacy summary feature_count must be 100")
    return {
        "seed": seed,
        "model": model,
        "profile": profile,
        "endpoint_horizon_days": EXPECTED_ENDPOINT,
        "population": "primary",
        "patient_count": EXPECTED_PATIENT_COUNT,
        "positive_count": EXPECTED_POSITIVE_COUNT,
        "negative_count": EXPECTED_NEGATIVE_COUNT,
        "outer_folds": EXPECTED_OUTER_FOLDS,
        "inner_folds": EXPECTED_INNER_FOLDS,
        "feature_count": 100,
    }


def _source_column(frame: pd.DataFrame, names: Iterable[str], *, label: str) -> str:
    source = next((name for name in names if name in frame.columns), None)
    if source is None:
        raise BaselineImportError(f"legacy OOF must contain {label}")
    return source


def _adapt_oof(path: Path, *, metadata: Mapping[str, Any], manifest: pd.DataFrame, seed: int) -> tuple[pd.DataFrame, dict[str, float]]:
    try:
        source = pd.read_parquet(path)
    except Exception as exc:
        raise BaselineImportError(f"unable to read legacy OOF: {path}") from exc
    patient_column = _source_column(source, ("patient_id", "patient", "subject_id"), label="patient_id")
    label_column = _source_column(source, ("y_true", "true_label", "label"), label="y_true")
    probability_column = _source_column(source, ("probability", "prediction_probability", "y_prob", "prediction_prob"), label="probability")
    model_column = _source_column(source, ("model", "model_name", "estimator"), label="model")
    profile_column = _source_column(source, ("profile", "analysis_profile"), label="profile")
    seed_column = _source_column(source, ("seed", "random_seed", "random_state"), label="seed")
    work = source.copy(deep=True)
    work["patient_id"] = work[patient_column].astype("string").str.strip()
    if work["patient_id"].isna().any() or (work["patient_id"] == "").any() or work["patient_id"].duplicated().any():
        raise BaselineImportError(f"legacy OOF seed {seed} must contain one row per patient")
    if len(work) != EXPECTED_PATIENT_COUNT:
        raise BaselineImportError(f"legacy OOF seed {seed} must contain {EXPECTED_PATIENT_COUNT} rows")
    if set(work["patient_id"].astype(str)) != set(manifest.index.astype(str)):
        raise BaselineImportError(f"legacy OOF seed {seed} patient set disagrees with current B0 manifest")
    labels = work[label_column].map(lambda value: _binary(value, label=f"legacy OOF seed {seed} y_true"))
    if int(labels.sum()) != EXPECTED_POSITIVE_COUNT or int((labels == 0).sum()) != EXPECTED_NEGATIVE_COUNT:
        raise BaselineImportError(f"legacy OOF seed {seed} must contain 27 positive and 676 negative labels")
    expected_labels = manifest.loc[work["patient_id"].astype(str), "_label"].to_numpy(dtype=int)
    if not np.array_equal(labels.to_numpy(dtype=int), expected_labels):
        raise BaselineImportError(f"legacy OOF seed {seed} labels disagree with current B0 manifest")
    models = {_normal_model(value, label=f"legacy OOF seed {seed} model") for value in work[model_column].tolist()}
    profiles = {_normal_profile(value, label=f"legacy OOF seed {seed} profile") for value in work[profile_column].tolist()}
    if models != {EXPECTED_MODEL} or profiles != {OUTPUT_PROFILE}:
        raise BaselineImportError(f"legacy OOF seed {seed} metadata is not ExtraTrees/all20")
    seeds = {_as_int(value, label=f"legacy OOF seed {seed} seed") for value in work[seed_column].tolist()}
    if seeds != {seed}:
        raise BaselineImportError(f"legacy OOF seed {seed} contains mismatched seed metadata")
    endpoint_column = next((name for name in ("endpoint_horizon_days", "endpoint_horizon", "horizon_days") if name in work.columns), None)
    if endpoint_column is not None:
        endpoints = {_as_int(value, label=f"legacy OOF seed {seed} endpoint") for value in work[endpoint_column].tolist()}
        if endpoints != {EXPECTED_ENDPOINT}:
            raise BaselineImportError(f"legacy OOF seed {seed} endpoint must be 365 days")
    outer_column = _source_column(work, ("outer_fold", "fold"), label="outer_fold")
    outer_fold = work[outer_column].map(lambda value: _as_int(value, label=f"legacy OOF seed {seed} outer_fold"))
    if set(outer_fold) != set(range(1, EXPECTED_OUTER_FOLDS + 1)):
        raise BaselineImportError(f"legacy OOF seed {seed} must contain outer folds 1..5")
    probabilities = work[probability_column].map(lambda value: _finite(value, label=f"legacy OOF seed {seed} probability", lower=0.0, upper=1.0))
    prediction_column = next((name for name in ("prediction", "prediction_label", "y_pred") if name in work.columns), None)
    threshold_column = next((name for name in ("threshold", "fold_threshold") if name in work.columns), None)
    if threshold_column is None:
        threshold = pd.Series(0.5, index=work.index, dtype=float)
    else:
        threshold = work[threshold_column].map(lambda value: _finite(value, label=f"legacy OOF seed {seed} threshold", lower=0.0, upper=1.0))
    if prediction_column is None:
        prediction = pd.Series((probabilities.to_numpy(dtype=float) >= threshold.to_numpy(dtype=float)).astype(int), index=work.index)
    else:
        prediction = work[prediction_column].map(lambda value: _binary(value, label=f"legacy OOF seed {seed} prediction"))
    metrics = compute_metrics(labels.to_numpy(dtype=int), probabilities.to_numpy(dtype=float), prediction.to_numpy(dtype=int))
    for name in ("AUC", "AP", "Brier", "BrierSkill"):
        if not math.isfinite(float(metrics[name])):
            raise BaselineImportError(f"legacy OOF seed {seed} produced non-finite core metric {name}")
    output = pd.DataFrame(
        {
            "candidate": OUTPUT_CANDIDATE,
            "patient_id": work["patient_id"].astype(str),
            "true_label": labels.astype(int),
            "y_true": labels.astype(int),
            "prediction_probability": probabilities.astype(float),
            "probability": probabilities.astype(float),
            "prediction_label": prediction.astype(int),
            "prediction": prediction.astype(int),
            "outer_fold": outer_fold.astype(int),
            "fold": outer_fold.astype(int),
            "fold_threshold": threshold.astype(float),
            "threshold": threshold.astype(float),
            "model": EXPECTED_MODEL,
            "profile": OUTPUT_PROFILE,
            "seed": seed,
            "endpoint_horizon_days": EXPECTED_ENDPOINT,
        }
    )
    output = output.sort_values("patient_id", kind="stable").reset_index(drop=True)
    joined = manifest.loc[output["patient_id"].tolist()]
    for column in ("af_flag", "pvc_count_24h", "high_pvc_burden", "high_pvc_flag"):
        if column in joined.columns:
            output[column] = joined[column].to_numpy()
    return output, metrics


def _summary(metadata: Mapping[str, Any], metrics: Mapping[str, float]) -> dict[str, Any]:
    # Calibration MLEs and denominator-dependent classification metrics can
    # be legitimately undefined (for example, calibration separation in the
    # immutable B0 seed 90).  Preserve that fact as JSON null; discrimination,
    # probability loss, probabilities and labels remain fail-closed above.
    serialized_metrics = {
        key: float(value) if math.isfinite(float(value)) else None
        for key, value in metrics.items()
    }
    values = dict(serialized_metrics)
    values.update(
        {
            "candidate": OUTPUT_CANDIDATE,
            "model": EXPECTED_MODEL,
            "profile": OUTPUT_PROFILE,
            "seed": int(metadata["seed"]),
            "random_state": int(metadata["seed"]),
            "endpoint_horizon_days": EXPECTED_ENDPOINT,
            "endpoint_horizon": EXPECTED_ENDPOINT,
            "population": "primary",
            "patient_count": EXPECTED_PATIENT_COUNT,
            "positive_count": EXPECTED_POSITIVE_COUNT,
            "negative_count": EXPECTED_NEGATIVE_COUNT,
            "outer_folds": EXPECTED_OUTER_FOLDS,
            "inner_folds": EXPECTED_INNER_FOLDS,
            "feature_count": 100,
            "legacy_run_id": LEGACY_RUN_ID,
            "legacy_head_sha": LEGACY_HEAD_SHA,
        }
    )
    values.update(
        {
            "auc": serialized_metrics["AUC"],
            "average_precision": serialized_metrics["AP"],
            "ap": serialized_metrics["AP"],
            "brier": serialized_metrics["Brier"],
            "sensitivity": serialized_metrics["Sens"],
            "specificity": serialized_metrics["Spec"],
            "f1": serialized_metrics["F1"],
            "ppv": serialized_metrics["PPV"],
            "npv": serialized_metrics["NPV"],
            "metrics": serialized_metrics,
        }
    )
    return values


def import_formal_baseline(
    legacy_root: str | Path,
    output_dir: str | Path,
    manifest_path: str | Path,
    *,
    endpoint: int = EXPECTED_ENDPOINT,
) -> dict[str, Any]:
    """Validate and import the 100 historical B0 seed artifacts."""

    if int(endpoint) != EXPECTED_ENDPOINT:
        raise BaselineImportError("formal baseline import requires endpoint 365")
    artifact_dirs = _artifact_dirs(Path(legacy_root))
    manifest, manifest_payload = _load_manifest(Path(manifest_path))
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    imported: list[dict[str, Any]] = []
    expected_ids = set(manifest.index.astype(str))
    for seed in EXPECTED_SEEDS:
        directory = artifact_dirs[seed]
        summary_path = _find_one(directory, "repeated_per_seed.json")
        oof_path = _find_one(directory, "repeated_oof.parquet")
        metadata = _legacy_metadata(_single_summary(summary_path), seed)
        oof, metrics = _adapt_oof(oof_path, metadata=metadata, manifest=manifest, seed=seed)
        if set(oof["patient_id"]) != expected_ids:  # defensive check after sorting/join
            raise BaselineImportError(f"legacy OOF seed {seed} patient set changed during import")
        seed_dir = target / OUTPUT_CANDIDATE / f"seed-{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)
        prefix = f"optimization_{OUTPUT_CANDIDATE}_{EXPECTED_ENDPOINT}d_seed{seed:03d}"
        oof.to_parquet(seed_dir / f"{prefix}_oof.parquet", engine="pyarrow", compression="zstd", index=False)
        (seed_dir / f"{prefix}_summary.json").write_text(
            json.dumps(_summary(metadata, metrics), ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        per_seed_manifest = dict(manifest_payload)
        per_seed_manifest["seed"] = seed
        per_seed_manifest["legacy_artifact"] = directory.name
        (seed_dir / "population_manifest.json").write_text(
            json.dumps(per_seed_manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        imported.append(
            {
                "seed": seed,
                "patient_count": len(oof),
                **{key: float(value) if math.isfinite(float(value)) else None for key, value in metrics.items()},
            }
        )
    root_manifest = dict(manifest_payload)
    root_manifest["seed_count"] = len(imported)
    root_manifest["imported_seeds"] = list(EXPECTED_SEEDS)
    (target / "B0" / "population_manifest.json").write_text(
        json.dumps(root_manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return {"candidate": OUTPUT_CANDIDATE, "profile": OUTPUT_PROFILE, "model": EXPECTED_MODEL, "seeds": imported, "manifest": root_manifest}


# Descriptive aliases for callers that use adapter/import terminology.
adapt_formal_baseline = import_formal_baseline
import_legacy_formal_baseline = import_formal_baseline


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--legacy-root", type=Path, required=True, help="downloaded formal-repeated-all20-seed-* artifact tree")
    parser.add_argument("--output-dir", type=Path, required=True, help="recursive optimization aggregate input tree")
    parser.add_argument("--manifest", "--current-b0-manifest", dest="manifest_path", type=Path, required=True, help="current audited B0 population manifest")
    parser.add_argument("--endpoint", type=int, default=EXPECTED_ENDPOINT)
    args = parser.parse_args(argv)
    result = import_formal_baseline(args.legacy_root, args.output_dir, args.manifest_path, endpoint=args.endpoint)
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
