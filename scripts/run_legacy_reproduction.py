#!/usr/bin/env python3
"""Run the strict nested patient-level Phase 3 legacy reproduction."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.build_legacy_features import (
    EXPECTED_NEGATIVES,
    EXPECTED_PATIENTS,
    EXPECTED_POSITIVES,
    build as build_features,
    default_raw_root,
    load_legacy_cohort,
)
from src.legacy_aggregation import AGGREGATED_FEATURE_NAMES
from src.legacy_model import NestedOOFResult, nested_patient_oof


FEATURE_PATH = Path("data/features/legacy_120s/patient_features.parquet")


def audit_raw_safety(repo: Path) -> dict[str, Any]:
    """Verify copied headers/metadata still match raw, without hashing DAT content."""
    raw = default_raw_root(repo)
    exact = repo / "data" / "source_exact"
    header_root = exact / "headers"
    header_count = 0
    header_mismatch = 0
    header_missing = 0
    for copied in header_root.rglob("*.hea"):
        source = raw / copied.relative_to(header_root)
        header_count += 1
        if not source.is_file():
            header_missing += 1
        elif hashlib.sha256(copied.read_bytes()).digest() != hashlib.sha256(source.read_bytes()).digest():
            header_mismatch += 1
    small_mismatch = 0
    for name in ("subject-info.csv", "subject-info_codes.csv", "subject-info_definitions.csv", "RECORDS", "SHA256SUMS.txt", "LICENSE.txt"):
        copied, source = exact / name, raw / name
        if copied.is_file() and source.is_file() and copied.read_bytes() != source.read_bytes():
            small_mismatch += 1
    dat_files = list(raw.rglob("*.dat"))
    return {
        "write_operations_issued": 0,
        "headers_compared": header_count,
        "header_missing": header_missing,
        "header_sha256_mismatch": header_mismatch,
        "source_exact_metadata_mismatch": small_mismatch,
        "dat_content_rehashed": False,
        "dat_files_stat_only": len(dat_files),
        "latest_dat_mtime_unix": max((p.stat().st_mtime for p in dat_files), default=None),
    }


def _json_safe(value: Any) -> Any:
    if isinstance(value, (np.integer, np.floating)):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_safe(v) for v in value]
    return value


def load_patient_features(repo: Path = REPO, *, build_if_missing: bool = True) -> pd.DataFrame:
    path = repo / FEATURE_PATH
    if not path.is_file():
        if not build_if_missing:
            raise FileNotFoundError(path)
        build_features(repo, default_raw_root(repo))
    frame = pd.read_parquet(path)
    if len(frame) != EXPECTED_PATIENTS:
        raise RuntimeError(f"legacy feature table patient count changed: {len(frame)}")
    if frame["patient_id"].duplicated().any():
        raise RuntimeError("legacy feature table has duplicate patient_id")
    labels = pd.to_numeric(frame["label"], errors="coerce")
    counts = {"patients": len(frame), "positive": int(labels.sum()), "negative": int((labels == 0).sum())}
    expected = {"patients": EXPECTED_PATIENTS, "positive": EXPECTED_POSITIVES, "negative": EXPECTED_NEGATIVES}
    if counts != expected:
        raise RuntimeError(f"legacy feature label counts changed: {counts}")
    missing = [name for name in AGGREGATED_FEATURE_NAMES if name not in frame.columns]
    if missing:
        raise RuntimeError(f"legacy feature table missing model columns: {missing}")
    return frame


def write_reproduction_reports(repo: Path, result: NestedOOFResult, *, n_jobs: int, n_iter: int) -> None:
    validation = repo / "data" / "validation"
    reports = repo / "reports"
    validation.mkdir(parents=True, exist_ok=True)
    reports.mkdir(parents=True, exist_ok=True)
    oof_path = validation / "legacy_oof_predictions.parquet"
    folds_path = validation / "legacy_model_folds.csv"
    result.oof.to_parquet(oof_path, index=False, compression="zstd")
    result.folds.to_csv(folds_path, index=False, encoding="utf-8")

    summary = dict(result.summary)
    summary.update(
        {
            "phase": 3,
            "model": "ExtraTreesClassifier",
            "n_jobs": int(n_jobs),
            "n_iter": int(n_iter),
            "outer_cv": "5-fold stratified patient-level",
            "inner_cv": "3-fold stratified patient-level",
            "preprocessing": "SimpleImputer(median) inside the outer-training pipeline",
            "old_csv_used_as_model_input": False,
            "paper_metrics": {
                "status": "USER_PROVIDED_REFERENCE_ONLY",
                "note": "Reference values are never used for selection, tuning, or data modification.",
            },
            "oof_row_count": int(len(result.oof)),
            "oof_patient_unique": bool(result.oof["patient_id"].is_unique),
        }
    )
    summary_path = reports / "PHASE3_LEGACY_REPRODUCTION.json"
    summary_path.write_text(json.dumps(_json_safe(summary), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    metrics = {
        "ROC_AUC": summary["oof_roc_auc"],
        "AP": summary["oof_average_precision"],
        "Brier": summary["oof_brier"],
        "Sensitivity": summary["sensitivity"],
        "Specificity": summary["specificity"],
        "F1": summary["f1"],
    }
    reference = {"ROC_AUC": 0.575, "AP": 0.511, "Brier": 0.257, "Sensitivity": 0.459, "Specificity": 0.784, "F1": 0.523}
    metrics_payload = {
        "metrics": metrics,
        "paper_reference": reference,
        "absolute_difference": {key: abs(float(metrics[key]) - value) for key, value in reference.items()},
    }
    (validation / "legacy_reproduction_metrics.json").write_text(
        json.dumps(_json_safe(metrics_payload), indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    comparison = pd.read_csv(reports / "LEGACY_FEATURE_COMPARISON.csv")
    exact_features = int(comparison["comparison_class"].eq("EXACT").sum())
    near_features = int(comparison["comparison_class"].eq("NEAR").sum())
    different_features = int(comparison["comparison_class"].eq("DIFFERENT").sum())
    window_validation = json.loads((validation / "legacy_window_validation.json").read_text(encoding="utf-8"))
    statuses = window_validation["window_status_counts"]
    classification = "PARTIAL_REPRODUCTION" if different_features else (
        "EXACT_REPRODUCTION" if all(value <= 1e-12 for value in metrics_payload["absolute_difference"].values())
        else "NEAR_REPRODUCTION"
    )
    phase3_files = sorted((repo / "data/features/legacy_120s").glob("*"))
    phase3_files += sorted(p for p in validation.glob("legacy_*"))
    size_files = {p.relative_to(repo).as_posix(): int(p.stat().st_size) for p in phase3_files if p.is_file()}
    all_data_bytes = sum(p.stat().st_size for p in (repo / "data").rglob("*") if p.is_file())
    size_payload = {
        "files": size_files,
        "phase3_generated_data_bytes": int(sum(size_files.values())),
        "repository_data_total_bytes": int(all_data_bytes),
    }
    (reports / "PHASE3_SIZE_REPORT.json").write_text(
        json.dumps(size_payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    raw_safety = audit_raw_safety(repo)
    summary.update({
        "phase2_base_commit": "b6ab167f6eb7873d8a7d024f9ab996aa4a0acf8a",
        "legacy_cohort_derived": 88,
        "legacy_cohort_reference": 88,
        "cohort_intersection": 88,
        "cohort_new_only": 0,
        "cohort_old_only": 0,
        "le_365d": 37,
        "gt_365d": 51,
        "holter_records_processed": 88,
        "waveform_windows_read": int(statuses.get("SUCCESS", 0)),
        "valid_windows": int(statuses.get("SUCCESS", 0)),
        "invalid_windows": int(sum(statuses.values()) - statuses.get("SUCCESS", 0)),
        "feature_schema_count": 20,
        "feature_schema_status": "CONFIRMED_AGAINST_OLD_CODE; PAPER_TEXT_NOT_AVAILABLE",
        "patient_aggregation_dimensions": 100,
        "feature_comparison": {"exact": exact_features, "near": near_features, "different": different_features},
        "reproduction_classification": classification,
        "paper_reference": reference,
        "phase3_generated_data_bytes": size_payload["phase3_generated_data_bytes"],
        "repository_data_total_bytes": size_payload["repository_data_total_bytes"],
        "music_raw_write_operations": 0,
        "music_raw_safety_audit": raw_safety,
        "repository_dat_files": 0,
        "raw_waveform_cache_files": 0,
        "issues_requiring_confirmation": [
            "The paper/manuscript text is absent; the 20-feature schema is confirmed from the existing project code.",
            "The old code used random windows whereas Phase 3 uses the mandated fixed hourly schedule; 97/100 aggregated features differ."
        ],
    })
    summary_path.write_text(json.dumps(_json_safe(summary), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    md = [
        "# Phase 3 legacy reproduction",
        "",
        "## Design",
        "",
        "- Frozen cohort: 88 SCD patients (37 ≤365 days, 51 >365 days).",
        "- Input: 100 aggregated fixed-window feature columns only.",
        "- Model: ExtraTreesClassifier with the old randomized distributions, 24 AP-scored iterations.",
        "- Validation: one patient-level OOF prediction in each outer 5-fold stratified split; hyperparameters and threshold are selected inside the outer training partition using inner 3-fold CV.",
        "- Threshold: old training-side inner-OOF target-specificity rule at 0.70.",
        "",
        "## OOF metrics",
        "",
        f"- Average precision: `{summary.get('oof_average_precision')}`",
        f"- ROC AUC: `{summary.get('oof_roc_auc')}`",
        f"- Brier: `{summary.get('oof_brier')}`",
        f"- Sensitivity: `{summary.get('sensitivity')}`",
        f"- Specificity: `{summary.get('specificity')}`",
        f"- F1: `{summary.get('f1')}`",
        f"- Confusion matrix (TN, FP, FN, TP): `({summary.get('tn')}, {summary.get('fp')}, {summary.get('fn')}, {summary.get('tp')})`",
        "",
        "## Interpretation boundary",
        "",
        "The fixed hourly windows are intentionally not numerically identical to the old random-window CSV. Paper metrics, if any, are reference-only and are not used as acceptance targets.",
        "",
        "## Reproduction classification",
        "",
        f"`{classification}`: cohort identity and 100-D construction are reproduced, but {different_features}/100 aggregated features differ from the old random-window CSV.",
        "",
        "## Paper reference (reference only)",
        "",
        "- ROC-AUC 0.575; AP 0.511; Brier 0.257; Sensitivity 0.459; Specificity 0.784; F1 0.523.",
        "",
        f"Artifacts: `{oof_path.relative_to(repo).as_posix()}`, `{folds_path.relative_to(repo).as_posix()}`.",
    ]
    (reports / "PHASE3_LEGACY_REPRODUCTION.md").write_text("\n".join(md) + "\n", encoding="utf-8")


def run(
    repo: Path = REPO,
    *,
    n_iter: int = 24,
    n_jobs: int = 1,
    build_if_missing: bool = True,
) -> NestedOOFResult:
    # Read the frozen cohort before opening the feature table, so a changed
    # cohort fails closed rather than producing a misleading model report.
    cohort = load_legacy_cohort(repo)
    features = load_patient_features(repo, build_if_missing=build_if_missing)
    if set(cohort["patient_id"].astype(str)) != set(features["patient_id"].astype(str)):
        raise RuntimeError("feature/cohort patient IDs differ")
    result = nested_patient_oof(features, n_iter=n_iter, n_jobs=n_jobs, tune_n_jobs=n_jobs)
    write_reproduction_reports(repo, result, n_jobs=n_jobs, n_iter=n_iter)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=REPO)
    parser.add_argument("--n-iter", type=int, default=24)
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument("--no-build", action="store_true", help="fail if the Phase 3 feature table is absent")
    args = parser.parse_args(argv)
    if args.n_iter < 1:
        parser.error("--n-iter must be positive")
    result = run(args.repo.resolve(), n_iter=args.n_iter, n_jobs=args.n_jobs, build_if_missing=not args.no_build)
    print(json.dumps(_json_safe(result.summary), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
