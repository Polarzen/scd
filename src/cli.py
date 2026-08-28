"""Command-line entry points for the P4-C validation framework."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd

from .full_model import (
    DEFAULT_BOOTSTRAP_RESAMPLES,
    DEFAULT_EXTRA_TREES_N_ITER,
    DEFAULT_FEATURE_CONFIG,
    DEFAULT_FEATURE_PATH,
    DEFAULT_INNER_FOLDS,
    DEFAULT_N_JOBS,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_OUTER_FOLDS,
    DEFAULT_RANDOM_STATE,
    DEFAULT_SUBJECTS_PATH,
    DEFAULT_TARGET_SPECIFICITY,
    canonical_model_name,
    get_model_feature_columns,
    get_param_distributions,
    prepare_model_frame,
)
from .nested_cv import NestedCVResult, run_nested_cv
from .model_optimization import CANDIDATES, prepare_optimization_bundle
from .optimization_reporting import aggregate_optimization_artifacts
from .repeated_cv import RepeatedCVResult, normalize_seeds, run_repeated_cv


def _jsonable(value: Any) -> Any:
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [_jsonable(item) for item in value]
    if isinstance(value, pd.DataFrame):
        return [_jsonable(item) for item in value.to_dict("records")]
    if isinstance(value, pd.Series):
        return [_jsonable(item) for item in value.tolist()]
    return value


def _csv_values(value: str | None) -> list[int]:
    if value is None:
        return []
    values: list[int] = []
    for token in str(value).replace(";", ",").split(","):
        token = token.strip()
        if token:
            values.append(int(token))
    return values


def _range_value(value: str | None) -> tuple[int, ...] | None:
    if value is None:
        return None
    tokens = [token.strip() for token in str(value).replace(",", ":").split(":") if token.strip()]
    if len(tokens) not in {2, 3}:
        raise ValueError("--seed-range must be START:STOP[:STEP]")
    return tuple(int(token) for token in tokens)


def _path(value: Path | None, default: Path) -> Path:
    return Path(default if value is None else value)


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(value), ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _write_frame(path: Path, frame: pd.DataFrame, *, parquet: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if parquet:
        frame.to_parquet(path, engine="pyarrow", compression="zstd", index=False)
    else:
        frame.to_csv(path, index=False)


def _add_common_data_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--feature-path", "--features", dest="feature_path", type=Path, default=None, help="patient feature parquet")
    parser.add_argument("--subjects-path", "--subjects", dest="subjects_path", type=Path, default=None, help="Phase 2 subjects parquet")
    parser.add_argument("--feature-config", "--config", dest="feature_config", type=Path, default=None, help="features_v2.yaml")
    parser.add_argument("--output-dir", type=Path, default=None, help="machine-readable output directory")
    parser.add_argument("--n-jobs", type=int, default=DEFAULT_N_JOBS)


def _add_cv_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--outer-folds", "--outer", dest="outer_folds", type=int, default=None)
    parser.add_argument("--inner-folds", "--inner", dest="inner_folds", type=int, default=DEFAULT_INNER_FOLDS)
    parser.add_argument("--n-iter", type=int, default=DEFAULT_EXTRA_TREES_N_ITER)
    parser.add_argument("--target-specificity", type=float, default=DEFAULT_TARGET_SPECIFICITY)
    parser.add_argument("--bootstrap-resamples", type=int, default=DEFAULT_BOOTSTRAP_RESAMPLES)
    parser.add_argument("--smoke", action="store_true", help="use one search candidate and a small outer CV")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="model-validation",
        description="P4-C strict patient-level nested model validation",
    )
    subparsers = parser.add_subparsers(dest="command", metavar="COMMAND")

    verify = subparsers.add_parser("verify", help="verify input tables and feature schema")
    _add_common_data_args(verify)

    validate = subparsers.add_parser("validate", help="run one strict nested-CV model")
    _add_common_data_args(validate)
    _add_cv_args(validate)
    validate.add_argument("--endpoint", "--horizon-days", dest="horizon_days", type=int, default=365)
    validate.add_argument("--profile", default="all20", choices=["all20", "physiology_only"])
    validate.add_argument("--model", default="extratrees", choices=["extratrees", "logistic", "dummy"])
    validate.add_argument("--seed", type=int, default=DEFAULT_RANDOM_STATE)

    repeated = subparsers.add_parser("repeated-cv", help="run nested CV over explicit seeds")
    _add_common_data_args(repeated)
    _add_cv_args(repeated)
    repeated.add_argument("--endpoint", "--horizon-days", dest="horizon_days", type=int, default=365)
    repeated.add_argument("--profile", default="all20", choices=["all20", "physiology_only"])
    repeated.add_argument("--models", "--model", dest="models", default="extratrees,logistic,dummy")
    repeated.add_argument("--seeds", default=None, help="comma-separated explicit seeds")
    repeated.add_argument("--seed-range", default=None, help="inclusive START:STOP[:STEP] seed range")
    repeated.add_argument("--seed-start", type=int, default=None)
    repeated.add_argument("--seed-stop", type=int, default=None)
    repeated.add_argument("--seed-step", type=int, default=1)

    sensitivity = subparsers.add_parser("endpoint-sensitivity", help="run the selected model at multiple horizons")
    _add_common_data_args(sensitivity)
    _add_cv_args(sensitivity)
    sensitivity.add_argument("--horizons", default="90,180,365,730")
    sensitivity.add_argument("--profile", default="all20", choices=["all20", "physiology_only"])
    sensitivity.add_argument("--model", default="extratrees", choices=["extratrees", "logistic", "dummy"])
    sensitivity.add_argument("--seed", type=int, default=DEFAULT_RANDOM_STATE)

    optimize = subparsers.add_parser("optimize", help="run one MODEL OPTIMIZATION V1 candidate")
    _add_common_data_args(optimize)
    _add_cv_args(optimize)
    optimize.add_argument("--endpoint", "--horizon-days", dest="horizon_days", type=int, default=365)
    optimize.add_argument("--candidate", required=True, choices=sorted(CANDIDATES))
    optimize.add_argument("--seed", type=int, default=DEFAULT_RANDOM_STATE)
    optimize.add_argument(
        "--model-override",
        choices=["elasticnet", "elasticnet_selected", "extratrees_regularized"],
        default=None,
        help="use only to keep P1/P2 on the same screened model family",
    )

    audit = subparsers.add_parser("optimization-audit", help="write population, AF, PVC, and profile audits")
    _add_common_data_args(audit)
    audit.add_argument("--endpoint", "--horizon-days", dest="horizon_days", type=int, default=365)

    aggregate = subparsers.add_parser("optimization-aggregate", help="strictly aggregate optimization seed artifacts")
    aggregate.add_argument("--artifact-dir", type=Path, required=True)
    aggregate.add_argument("--output-dir", type=Path, required=True)
    aggregate.add_argument("--expected-candidates", required=True, help="comma-separated candidate IDs")
    aggregate.add_argument("--expected-seeds", required=True, help="comma-separated seeds or START:STOP")
    aggregate.add_argument("--baseline-candidate", default="B0")
    aggregate.add_argument("--formal", action="store_true")
    aggregate.add_argument("--write-report", action="store_true")
    return parser


def _data_frame(args: argparse.Namespace) -> pd.DataFrame:
    feature_path = _path(args.feature_path, DEFAULT_FEATURE_PATH)
    subjects_path = _path(args.subjects_path, DEFAULT_SUBJECTS_PATH)
    feature_config = _path(args.feature_config, DEFAULT_FEATURE_CONFIG)
    return prepare_model_frame(
        feature_path,
        subjects_path,
        horizon_days=int(args.horizon_days),
        profile=args.profile,
        feature_config_path=feature_config,
    )


def _cv_values(args: argparse.Namespace) -> dict[str, Any]:
    outer = DEFAULT_OUTER_FOLDS if args.outer_folds is None else int(args.outer_folds)
    n_iter = int(args.n_iter)
    bootstrap = int(args.bootstrap_resamples)
    if args.smoke:
        n_iter = 1
        if args.outer_folds is None:
            outer = min(3, DEFAULT_OUTER_FOLDS)
        if bootstrap == DEFAULT_BOOTSTRAP_RESAMPLES:
            bootstrap = 50
    return {
        "outer_folds": outer,
        "inner_folds": int(args.inner_folds),
        "n_iter": n_iter,
        "target_specificity": float(args.target_specificity),
        "bootstrap_resamples": bootstrap,
        "n_jobs": int(args.n_jobs),
    }


def verify_inputs(
    *,
    feature_path: Path = DEFAULT_FEATURE_PATH,
    subjects_path: Path = DEFAULT_SUBJECTS_PATH,
    feature_config_path: Path = DEFAULT_FEATURE_CONFIG,
) -> dict[str, Any]:
    """Read only metadata/schema checks; no model fitting is performed."""

    errors: list[str] = []
    checks: dict[str, Any] = {
        "feature_path": str(feature_path),
        "subjects_path": str(subjects_path),
        "feature_config_path": str(feature_config_path),
        "music_raw_dir_used": False,
    }
    if not feature_path.is_file():
        errors.append(f"missing feature parquet: {feature_path}")
    if not subjects_path.is_file():
        errors.append(f"missing subjects parquet: {subjects_path}")
    try:
        if feature_path.is_file():
            features = pd.read_parquet(feature_path)
            cols = get_model_feature_columns(features, "all20", feature_config_path=feature_config_path)
            checks["feature_rows"] = int(len(features))
            checks["feature_count"] = int(len(cols))
            checks["feature_patient_unique"] = bool(features["patient_id"].is_unique) if "patient_id" in features.columns else False
            if not checks["feature_patient_unique"]:
                errors.append("feature table does not have a unique patient_id")
    except Exception as exc:
        errors.append(f"feature schema error: {exc}")
    try:
        if subjects_path.is_file():
            subjects = pd.read_parquet(subjects_path)
            required = {"patient_id", "followup_days", "cause_of_death_raw", "event_source_valid"}
            missing = sorted(required - set(subjects.columns))
            if missing:
                errors.append(f"subjects endpoint columns missing: {missing}")
            checks["subject_rows"] = int(len(subjects))
            checks["subject_patient_unique"] = bool(subjects["patient_id"].is_unique) if "patient_id" in subjects.columns else False
    except Exception as exc:
        errors.append(f"subjects schema error: {exc}")
    checks["ok"] = not errors
    checks["errors"] = errors
    return checks


def _output_dir(args: argparse.Namespace) -> Path:
    return _path(args.output_dir, DEFAULT_OUTPUT_DIR)


def _write_nested_outputs(result: NestedCVResult, output_dir: Path, *, prefix: str = "validation") -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "summary_json": output_dir / f"{prefix}_summary.json",
        "oof_parquet": output_dir / f"{prefix}_oof.parquet",
        "oof_csv": output_dir / f"{prefix}_oof.csv",
        "folds_csv": output_dir / f"{prefix}_folds.csv",
        "folds_parquet": output_dir / f"{prefix}_folds.parquet",
    }
    _write_json(paths["summary_json"], result.summary)
    _write_frame(paths["oof_parquet"], result.oof, parquet=True)
    _write_frame(paths["oof_csv"], result.oof)
    _write_frame(paths["folds_csv"], result.folds)
    _write_frame(paths["folds_parquet"], result.folds, parquet=True)
    return {key: str(value) for key, value in paths.items()}


def _write_repeated_outputs(result: RepeatedCVResult, output_dir: Path) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "per_seed_json": output_dir / "repeated_per_seed.json",
        "per_seed_csv": output_dir / "repeated_per_seed.csv",
        "summary_json": output_dir / "repeated_summary.json",
        "summary_csv": output_dir / "repeated_summary.csv",
        "summary_wide_csv": output_dir / "repeated_summary_wide.csv",
        "oof_parquet": output_dir / "repeated_oof.parquet",
    }
    _write_json(paths["per_seed_json"], result.per_seed)
    _write_json(paths["summary_json"], result.summary)
    _write_frame(paths["per_seed_csv"], result.per_seed)
    _write_frame(paths["summary_csv"], result.summary)
    if result.summary_wide is not None:
        _write_frame(paths["summary_wide_csv"], result.summary_wide)
    _write_frame(paths["oof_parquet"], result.oof, parquet=True)
    return {key: str(value) for key, value in paths.items()}


def _cmd_verify(args: argparse.Namespace) -> int:
    result = verify_inputs(
        feature_path=_path(args.feature_path, DEFAULT_FEATURE_PATH),
        subjects_path=_path(args.subjects_path, DEFAULT_SUBJECTS_PATH),
        feature_config_path=_path(args.feature_config, DEFAULT_FEATURE_CONFIG),
    )
    print(json.dumps(_jsonable(result), ensure_ascii=False, indent=2, allow_nan=False))
    return 0 if result["ok"] else 1


def _cmd_validate(args: argparse.Namespace) -> int:
    frame = _data_frame(args)
    result = run_nested_cv(
        frame,
        model=canonical_model_name(args.model),
        profile=args.profile,
        seed=int(args.seed),
        **_cv_values(args),
    )
    output_dir = _output_dir(args)
    paths = _write_nested_outputs(result, output_dir, prefix=f"{args.model}_{args.profile}_{args.horizon_days}d_seed{args.seed}")
    population_columns = [
        column for column in
        ("patient_id", "endpoint_state", "binary_label_if_evaluable", "time_to_event", "event_type")
        if column in frame.columns
    ]
    population = frame.loc[:, population_columns].copy()
    population.insert(1, "included", True)
    population.insert(2, "reason", "ELIGIBLE")
    population_path = output_dir / f"{args.model}_{args.profile}_{args.horizon_days}d_seed{args.seed}_analysis_population.csv"
    _write_frame(population_path, population)
    paths["analysis_population_csv"] = str(population_path)
    print(json.dumps(_jsonable({"summary": result.summary, "outputs": paths}), ensure_ascii=False, indent=2, allow_nan=False))
    return 0


def _cmd_repeated(args: argparse.Namespace) -> int:
    frame = _data_frame(args)
    seeds_text = str(args.seeds) if args.seeds is not None else None
    compact_range = _range_value(args.seed_range)
    if seeds_text and ":" in seeds_text:
        if compact_range is not None:
            raise ValueError("use only one of --seeds START:STOP or --seed-range")
        compact_range = _range_value(seeds_text)
        explicit = []
    else:
        explicit = _csv_values(args.seeds)
    result = run_repeated_cv(
        frame,
        seeds=explicit or None,
        seed_start=args.seed_start,
        seed_stop=args.seed_stop,
        seed_step=int(args.seed_step),
        seed_range=compact_range,
        models=[item.strip() for item in args.models.split(",") if item.strip()],
        profile=args.profile,
        **_cv_values(args),
    )
    paths = _write_repeated_outputs(result, _output_dir(args))
    print(json.dumps(_jsonable({"per_seed": result.per_seed, "summary": result.summary, "outputs": paths}), ensure_ascii=False, indent=2, allow_nan=False))
    return 0


def _cmd_sensitivity(args: argparse.Namespace) -> int:
    horizons = _csv_values(args.horizons)
    if not horizons:
        raise ValueError("at least one endpoint horizon is required")
    values = _cv_values(args)
    rows: list[dict[str, Any]] = []
    output_dir = _output_dir(args)
    for horizon in horizons:
        args.horizon_days = int(horizon)
        frame = _data_frame(args)
        result = run_nested_cv(
            frame,
            model=canonical_model_name(args.model),
            profile=args.profile,
            seed=int(args.seed),
            **values,
        )
        row = {"endpoint_horizon_days": int(horizon), **{key: result.summary.get(key) for key in ("patient_count", "positive_count", "negative_count", "AUC", "AP", "Brier", "Sens", "Spec", "F1", "PPV", "NPV")}}
        rows.append(row)
        _write_nested_outputs(result, output_dir, prefix=f"{args.model}_{args.profile}_{horizon}d_seed{args.seed}")
    table = pd.DataFrame(rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(output_dir / "endpoint_sensitivity.json", table)
    _write_frame(output_dir / "endpoint_sensitivity.csv", table)
    _write_frame(output_dir / "endpoint_sensitivity.parquet", table, parquet=True)
    print(json.dumps(_jsonable({"results": table, "output_dir": str(output_dir)}), ensure_ascii=False, indent=2, allow_nan=False))
    return 0


def _optimization_bundle(args: argparse.Namespace, candidate: str):
    return prepare_optimization_bundle(
        _path(args.feature_path, DEFAULT_FEATURE_PATH),
        _path(args.subjects_path, DEFAULT_SUBJECTS_PATH),
        candidate=candidate,
        horizon_days=int(args.horizon_days),
    )


def _manifest_json(bundle: Any) -> dict[str, Any]:
    manifest = bundle.population_manifest
    return {
        "candidate": manifest.get("candidate"),
        "profile": manifest.get("profile"),
        "model": manifest.get("model"),
        "selected_population": manifest.get("selected_population"),
        "selected_ids": manifest.get("selected_ids", []),
        "counts": manifest.get("counts", {}),
        "baseline_positive_ids": manifest.get("baseline", {}).get("positive_ids", []),
        "baseline_excluded_positive_ids": manifest.get("baseline", {}).get("excluded_positive_ids", []),
        "rhythm_safe_positive_ids": manifest.get("rhythm_safe", {}).get("positive_ids", []),
        "recovered_af_positive_ids": manifest.get("rhythm_safe", {}).get("recovered_af_positive_ids", []),
    }


def _cmd_optimize(args: argparse.Namespace) -> int:
    bundle = _optimization_bundle(args, args.candidate)
    model = bundle.model if args.model_override is None else canonical_model_name(args.model_override)
    if args.model_override is not None and bundle.candidate not in {"P1", "P2"}:
        raise ValueError("--model-override is reserved for the paired P1/P2 experiment")
    search_space = None
    if args.smoke:
        full_space = get_param_distributions(model)
        search_space = {key: [values[0]] for key, values in full_space.items()}
        if "pre__select__k" in search_space:
            search_space["pre__select__k"] = [min(8, len(bundle.feature_cols))]
    result = run_nested_cv(
        bundle.frame,
        model=model,
        profile=bundle.profile,
        feature_cols=bundle.feature_cols,
        seed=int(args.seed),
        param_distributions=search_space,
        **_cv_values(args),
    )
    result.summary.update(
        {
            "candidate": bundle.candidate,
            "population": bundle.population_manifest.get("selected_population"),
            "af_included_count": int(bundle.frame.get("af_flag", pd.Series(False, index=bundle.frame.index)).fillna(False).astype(bool).sum()),
            "af_positive_included_count": int(
                (bundle.frame.get("af_flag", pd.Series(False, index=bundle.frame.index)).fillna(False).astype(bool)
                 & bundle.frame["label"].eq(1)).sum()
            ),
            "pvc_fields": [column for column in bundle.feature_cols if "pvc" in column.lower()],
        }
    )
    result.oof.insert(0, "candidate", bundle.candidate)
    result.folds.insert(0, "candidate", bundle.candidate)
    output_dir = _output_dir(args)
    prefix = f"optimization_{bundle.candidate}_{args.horizon_days}d_seed{int(args.seed):03d}"
    paths = _write_nested_outputs(result, output_dir, prefix=prefix)
    audit_path = output_dir / "population_audit.json"
    manifest_path = output_dir / "population_manifest.json"
    manifest_csv = output_dir / "population_manifest.csv"
    _write_json(audit_path, bundle.audit)
    _write_json(manifest_path, _manifest_json(bundle))
    _write_frame(manifest_csv, bundle.population_manifest_table)
    paths.update(
        {
            "population_audit_json": str(audit_path),
            "population_manifest_json": str(manifest_path),
            "population_manifest_csv": str(manifest_csv),
        }
    )
    print(json.dumps(_jsonable({"summary": result.summary, "outputs": paths}), ensure_ascii=False, indent=2, allow_nan=False))
    return 0


def _cmd_optimization_audit(args: argparse.Namespace) -> int:
    bundles = {candidate: _optimization_bundle(args, candidate) for candidate in sorted(CANDIDATES)}
    reference = bundles["B0"]
    audit_table = reference.population_manifest_table
    positive = audit_table["endpoint_state"].astype("string").eq("POSITIVE")
    pvc_fields: list[dict[str, Any]] = []
    for column in [name for name in audit_table.columns if "pvc" in name.lower()]:
        series = audit_table[column]
        numeric = pd.to_numeric(series, errors="coerce").astype("float64")
        record: dict[str, Any] = {
            "field_name": column,
            "dtype": str(series.dtype),
            "non_null_count": int(series.notna().sum()),
            "patient_coverage": float(series.notna().mean()),
            "positive_patient_coverage_count": int((series.notna() & positive).sum()),
        }
        if numeric.notna().any():
            record["distribution"] = {
                "min": float(numeric.min()),
                "p25": float(numeric.quantile(0.25)),
                "median": float(numeric.median()),
                "p75": float(numeric.quantile(0.75)),
                "max": float(numeric.max()),
            }
        else:
            record["distribution"] = series.astype("string").value_counts(dropna=False).head(20).to_dict()
        pvc_fields.append(record)
    payload = {
        "endpoint_horizon_days": int(args.horizon_days),
        "baseline": reference.audit.get("baseline_365"),
        "baseline_365_invariant_pass": reference.audit.get("baseline_365_invariant_pass"),
        "af_audit": reference.audit.get("rhythm_safe_recovery"),
        "pvc_availability_audit": reference.audit.get("pvc"),
        "pvc_fields": pvc_fields,
        "PVC_CONTINUOUS_UNAVAILABLE": True,
        "candidates": {
            candidate: {
                "profile": bundle.profile,
                "model": bundle.model,
                "population": bundle.population_manifest.get("selected_population"),
                "patient_count": int(len(bundle.frame)),
                "positive_count": int(bundle.frame["label"].sum()),
                "negative_count": int(bundle.frame["label"].eq(0).sum()),
                "feature_count": int(len(bundle.feature_cols)),
                "af_included_count": int(bundle.frame.get("af_flag", pd.Series(False, index=bundle.frame.index)).fillna(False).astype(bool).sum()),
                "af_positive_included_count": int((bundle.frame.get("af_flag", pd.Series(False, index=bundle.frame.index)).fillna(False).astype(bool) & bundle.frame["label"].eq(1)).sum()),
                "pvc_fields": [column for column in bundle.feature_cols if "pvc" in column.lower()],
            }
            for candidate, bundle in bundles.items()
        },
    }
    output_dir = _output_dir(args)
    path = output_dir / "population_audit.json"
    _write_json(path, payload)
    print(json.dumps(_jsonable({"audit": payload, "output": str(path)}), ensure_ascii=False, indent=2, allow_nan=False))
    return 0


def _expected_seeds(value: str) -> list[int]:
    if ":" in str(value):
        parsed = _range_value(value)
        if parsed is None:
            return []
        start, stop = parsed[0], parsed[1]
        step = parsed[2] if len(parsed) == 3 else 1
        return list(range(start, stop + (1 if step > 0 else -1), step))
    return _csv_values(value)


def _cmd_optimization_aggregate(args: argparse.Namespace) -> int:
    candidates = [value.strip().upper() for value in str(args.expected_candidates).split(",") if value.strip()]
    seeds = _expected_seeds(args.expected_seeds)
    report_inputs = None
    if args.write_report:
        report_inputs = {
            "objective": (
                "Evaluate dimensionality reduction, AF-compatible coverage, PVC increment, calibration, "
                "and patient-split sensitivity under repeated patient-level nested CV."
            )
        }
    result = aggregate_optimization_artifacts(
        artifact_dir=args.artifact_dir,
        target_dir=args.output_dir,
        formal=bool(args.formal),
        expected_seeds=seeds,
        expected_candidates=candidates,
        baseline_candidate=str(args.baseline_candidate).upper(),
        report_inputs=report_inputs,
    )
    print(json.dumps(_jsonable({"integrity": result["integrity"], "outputs": result["output_paths"]}), ensure_ascii=False, indent=2, allow_nan=False))
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    values = list(sys.argv[1:] if argv is None else argv)
    # Explicitly short-circuit no-argument invocation so help never triggers
    # parquet reads, endpoint construction, or model fitting.
    if not values:
        parser.print_help()
        return 0
    args = parser.parse_args(values)
    if args.command is None:
        parser.print_help()
        return 0
    if args.command == "verify":
        return _cmd_verify(args)
    if args.command == "validate":
        return _cmd_validate(args)
    if args.command == "repeated-cv":
        return _cmd_repeated(args)
    if args.command == "endpoint-sensitivity":
        return _cmd_sensitivity(args)
    if args.command == "optimize":
        return _cmd_optimize(args)
    if args.command == "optimization-audit":
        return _cmd_optimization_audit(args)
    if args.command == "optimization-aggregate":
        return _cmd_optimization_aggregate(args)
    parser.error(f"unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["build_parser", "verify_inputs", "main"]
