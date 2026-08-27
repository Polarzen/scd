#!/usr/bin/env python3
"""Build the Phase 3 fixed-window legacy feature tables.

The script reads only the frozen Phase 2 cohort metadata and bounded 120-second
WFDB windows from the supplied MUSIC raw root.  It never writes waveform or
intermediate signal/cache files.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.legacy_aggregation import AGGREGATED_FEATURE_NAMES, aggregate_patient_features
from src.legacy_features import (
    FEATURE_NAMES,
    FIRST_START_SEC,
    INTERVAL_SEC,
    MAX_WINDOWS,
    SUCCESS,
    WINDOW_SEC,
    extract_fixed_windows,
)


EXPECTED_PATIENTS = 88
EXPECTED_POSITIVES = 37
EXPECTED_NEGATIVES = 51
OLD_DATASET_NAME = "scd_dataset.csv"


def load_legacy_cohort(repo: Path = REPO) -> pd.DataFrame:
    """Load the source-derived Phase 2 legacy timing cohort."""

    subjects = pd.read_parquet(repo / "data" / "cohort" / "subjects.parquet")
    selected = subjects.loc[
        subjects["cause_of_death_raw"].astype("string").eq("3")
        & subjects["event_source_valid"].fillna(False).astype(bool)
        & subjects["has_holter"].fillna(False).astype(bool),
        [
            "patient_id",
            "followup_days",
            "cause_of_death_raw",
            "holter_record_id",
            "holter_sampling_frequency",
            "holter_sample_count",
            "holter_duration_sec",
            "holter_lead_names",
        ],
    ].copy()
    selected["label"] = pd.to_numeric(selected["followup_days"], errors="coerce").le(365).astype(int)
    selected = selected.sort_values("patient_id", kind="stable").reset_index(drop=True)
    counts = {
        "patients": int(len(selected)),
        "positive": int(selected["label"].sum()),
        "negative": int((selected["label"] == 0).sum()),
    }
    if counts != {
        "patients": EXPECTED_PATIENTS,
        "positive": EXPECTED_POSITIVES,
        "negative": EXPECTED_NEGATIVES,
    }:
        raise RuntimeError(f"frozen legacy cohort count changed: {counts}")
    if selected["patient_id"].duplicated().any():
        raise RuntimeError("legacy cohort contains duplicate patient_id")
    return selected


def default_raw_root(repo: Path = REPO) -> Path:
    return repo / "music-sudden-cardiac-death-in-chronic-heart-failure-1.0.1"


def build_window_table(cohort: pd.DataFrame, raw_root: Path) -> pd.DataFrame:
    """Extract all 24 theoretical windows for every cohort patient."""

    rows: list[pd.DataFrame] = []
    for record in cohort.to_dict(orient="records"):
        patient_id = str(record["patient_id"])
        record_id = str(record["holter_record_id"])
        fs = float(record["holter_sampling_frequency"])
        sample_count = int(record["holter_sample_count"])
        stem = raw_root / "Holter_ECG" / record_id
        if not Path(str(stem) + ".hea").is_file():
            raise FileNotFoundError(f"missing Holter header for {patient_id}: {stem}.hea")
        if not Path(str(stem) + ".dat").is_file():
            raise FileNotFoundError(f"missing Holter signal for {patient_id}: {stem}.dat")

        frame = extract_fixed_windows(
            stem,
            fs=fs,
            sample_count=sample_count,
            patient_id=patient_id,
            first_start_sec=FIRST_START_SEC,
            interval_sec=INTERVAL_SEC,
            max_windows=MAX_WINDOWS,
            window_sec=WINDOW_SEC,
        )
        frame.insert(1, "record_id", record_id)
        leads = record.get("holter_lead_names")
        channel_name = str(leads[0]) if leads is not None and len(leads) else None
        frame["channel_selected"] = 0
        frame["channel_name"] = channel_name
        frame["label"] = int(record["label"])
        frame["followup_days"] = float(record["followup_days"])
        frame["cause_of_death"] = float(record["cause_of_death_raw"])
        frame["fs"] = fs
        rows.append(frame)

    output = pd.concat(rows, ignore_index=True)
    expected_rows = EXPECTED_PATIENTS * MAX_WINDOWS
    if len(output) != expected_rows:
        raise RuntimeError(f"expected {expected_rows} theoretical windows, got {len(output)}")
    if output.groupby("patient_id", sort=False).size().ne(MAX_WINDOWS).any():
        raise RuntimeError("every patient must have exactly 24 theoretical windows")
    return output


def _finite_pair(left: Any, right: Any) -> bool:
    return bool(np.isfinite(float(left)) and np.isfinite(float(right)))


def compare_with_old_csv(
    patient_features: pd.DataFrame,
    old_csv: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create an audit-only feature comparison against the old CSV.

    The old table is never used as a feature source.  A per-patient wide audit
    and a per-feature summary are returned separately.
    """

    old = pd.read_csv(old_csv, dtype={"patient_id": "string"})
    old["patient_id"] = old["patient_id"].astype("string")
    new = patient_features.copy()
    new["patient_id"] = new["patient_id"].astype("string")
    merged = new.merge(old, on="patient_id", how="outer", suffixes=("_new", "_old"), indicator=True)
    wide_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    for feature in AGGREGATED_FEATURE_NAMES:
        old_column = feature
        if old_column not in old.columns or feature not in new.columns:
            summary_rows.append(
                {
                    "feature": feature,
                    "old_column_present": bool(old_column in old.columns),
                    "new_column_present": bool(feature in new.columns),
                    "matched_patients": 0,
                    "missing_patients": int(len(merged)),
                    "mean_abs_diff": np.nan,
                    "median_abs_diff": np.nan,
                    "max_abs_diff": np.nan,
                    "correlation": np.nan,
                    "exact_match_count": 0,
                    "comparison_class": "DIFFERENT",
                }
            )
            continue
        left = pd.to_numeric(merged[f"{feature}_new"], errors="coerce")
        right = pd.to_numeric(merged[f"{feature}_old"], errors="coerce")
        paired = left.notna() & right.notna()
        diffs = (left[paired] - right[paired]).abs()
        correlation = float(left[paired].corr(right[paired])) if int(paired.sum()) >= 2 else np.nan
        max_error = float(diffs.max()) if len(diffs) else np.nan
        if len(diffs) and bool((diffs <= 1e-12).all()):
            comparison_class = "EXACT"
        elif len(diffs) and bool(np.isclose(left[paired], right[paired], rtol=1e-6, atol=1e-8).all()):
            comparison_class = "NEAR"
        else:
            comparison_class = "DIFFERENT"
        summary_rows.append(
            {
                "feature": feature,
                "old_column_present": True,
                "new_column_present": True,
                "matched_patients": int(paired.sum()),
                "missing_patients": int(len(merged) - paired.sum()),
                "mean_abs_diff": float(diffs.mean()) if len(diffs) else np.nan,
                "median_abs_diff": float(diffs.median()) if len(diffs) else np.nan,
                "max_abs_diff": max_error,
                "correlation": correlation,
                "exact_match_count": int(np.isclose(left[paired], right[paired], rtol=0.0, atol=1e-12).sum()),
                "comparison_class": comparison_class,
            }
        )
        for pid, new_value, old_value in zip(merged["patient_id"], left, right):
            if pd.isna(pid):
                continue
            wide_rows.append(
                {
                    "patient_id": str(pid),
                    "feature": feature,
                    "new_value": float(new_value) if pd.notna(new_value) else np.nan,
                    "old_value": float(old_value) if pd.notna(old_value) else np.nan,
                    "absolute_difference": abs(float(new_value) - float(old_value))
                    if pd.notna(new_value) and pd.notna(old_value)
                    else np.nan,
                    "comparison_status": (
                        "MATCH"
                        if pd.notna(new_value)
                        and pd.notna(old_value)
                        and np.isclose(float(new_value), float(old_value), rtol=0.0, atol=1e-12)
                        else "DIFFERENT"
                        if pd.notna(new_value) and pd.notna(old_value)
                        else "MISSING_NEW"
                        if pd.isna(new_value)
                        else "MISSING_OLD"
                    ),
                }
            )
    wide = pd.DataFrame(wide_rows).sort_values(["patient_id", "feature"], kind="stable").reset_index(drop=True)
    summary = pd.DataFrame(summary_rows)
    return wide, summary


def write_comparison_reports(
    *,
    repo: Path,
    patient_features: pd.DataFrame,
    old_csv: Path,
) -> dict[str, Any]:
    wide, summary = compare_with_old_csv(patient_features, old_csv)
    validation_dir = repo / "data" / "validation"
    reports_dir = repo / "reports"
    validation_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    wide_path = validation_dir / "legacy_feature_comparison.csv"
    summary_path = reports_dir / "LEGACY_FEATURE_COMPARISON.csv"
    wide.to_csv(wide_path, index=False, encoding="utf-8")
    summary.to_csv(summary_path, index=False, encoding="utf-8")
    compared = summary[summary["matched_patients"] > 0]
    md = [
        "# Legacy fixed-window feature comparison",
        "",
        "This is an audit-only comparison with `scd_dataset.csv`; the old CSV is never used as a replacement feature table.",
        "",
        f"- New patient rows: {len(patient_features)}",
        f"- Old CSV patient rows: {pd.read_csv(old_csv).shape[0]}",
        f"- Aggregated feature columns compared: {len(compared)} / {len(AGGREGATED_FEATURE_NAMES)}",
        f"- Exact features: {int((compared['comparison_class'] == 'EXACT').sum())}",
        f"- Near features: {int((compared['comparison_class'] == 'NEAR').sum())}",
        f"- Different features: {int((compared['comparison_class'] == 'DIFFERENT').sum())}",
        "- Expected differences: Phase 3 uses fixed hourly windows; the old CSV records random legacy windows.",
        "",
        "| Feature | Matched | Missing | Mean abs | Median abs | Max abs | Correlation | Class |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in summary.itertuples(index=False):
        md.append(
            f"| {row.feature} | {int(row.matched_patients)} | {int(row.missing_patients)} | {row.mean_abs_diff:.6g} | {row.median_abs_diff:.6g} | {row.max_abs_diff:.6g} | {row.correlation:.6g} | {row.comparison_class} |"
            if int(row.matched_patients) > 0
            else f"| {row.feature} | 0 | {int(row.missing_patients)} | n/a | n/a | n/a | n/a | DIFFERENT |"
        )
    (reports_dir / "LEGACY_FEATURE_COMPARISON.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    return {
        "new_patient_count": int(len(patient_features)),
        "old_patient_count": int(pd.read_csv(old_csv).shape[0]),
        "feature_count": len(AGGREGATED_FEATURE_NAMES),
        "features_compared": int(len(compared)),
        "exact_match_values": int(compared["exact_match_count"].sum()) if len(compared) else 0,
        "exact_features": int((compared["comparison_class"] == "EXACT").sum()),
        "near_features": int((compared["comparison_class"] == "NEAR").sum()),
        "different_features": int((compared["comparison_class"] == "DIFFERENT").sum()),
        "comparison_is_audit_only": True,
        "fixed_windows_differ_from_old_random_windows": True,
    }


def write_size_report(repo: Path, paths: list[Path], *, window_rows: int, patient_rows: int) -> None:
    sizes = {str(path.relative_to(repo)).replace("\\", "/"): int(path.stat().st_size) for path in paths if path.exists()}
    report = {
        "phase": 3,
        "window_length_sec": WINDOW_SEC,
        "theoretical_windows_per_patient": MAX_WINDOWS,
        "patient_count": int(patient_rows),
        "window_row_count": int(window_rows),
        "aggregated_feature_count": len(AGGREGATED_FEATURE_NAMES),
        "feature_dtype": "float64",
        "parquet_compression": "zstd",
        "files": sizes,
        "total_bytes": int(sum(sizes.values())),
        "raw_waveform_or_cache_written": False,
    }
    (repo / "reports" / "PHASE3_SIZE_REPORT.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def _contract_tables(windows: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split the internal calculation table into the frozen manifest and feature table."""
    status = windows["window_status"].astype("string")
    manifest = pd.DataFrame({
        "patient_id": windows["patient_id"].astype("string"),
        "window_id": windows["window_idx"].astype("int64"),
        "record_id": windows["record_id"].astype("string"),
        "start_sec": windows["window_start_sec"].astype("int64"),
        "end_sec": windows["window_end_sec"].astype("int64"),
        "start_sample": windows["start_sample"].astype("int64"),
        "end_sample": (windows["start_sample"] + windows["requested_samples"]).astype("int64"),
        "sampling_frequency": windows["fs"].astype("float64"),
        "channel_selected": windows["channel_selected"].astype("int64"),
        "channel_name": windows["channel_name"].astype("string"),
        "window_expected": True,
        "window_within_record": status.ne("OUTSIDE_RECORD"),
        "waveform_read_success": status.isin(["SUCCESS", "FEATURE_ERROR"]),
        "feature_extraction_success": status.eq("SUCCESS"),
        "valid": status.eq("SUCCESS"),
        "qc_status": np.where(status.eq("SUCCESS"), "PASS", "FAIL"),
        "qc_reason": windows["failure_reason"].astype("string"),
        "raw_rpeak_count": windows.get("raw_rpeak_count", windows["beats"].fillna(0)).astype("int64"),
        "raw_rr_count": windows["raw_rr_count"].astype("int64"),
        "valid_rr_count": windows["valid_rr_count"].astype("int64"),
        "removed_rr_count": windows["removed_rr_count"].astype("int64"),
        "removed_rr_ratio": windows.get("removed_rr_ratio", pd.Series(0.0, index=windows.index)).astype("float64"),
    })
    features = manifest[["patient_id", "window_id", "valid", "qc_status", "qc_reason"]].copy()
    for index, name in enumerate(FEATURE_NAMES, start=1):
        features[f"feature_{index:02d}"] = pd.to_numeric(windows[name], errors="coerce").astype("float64")
    return manifest, features


def _write_derived(repo: Path, windows: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    cohort = load_legacy_cohort(repo)
    if "channel_selected" not in windows.columns:
        windows = windows.copy()
        windows["channel_selected"] = 0
    if "channel_name" not in windows.columns:
        lead_map = {
            str(row.patient_id): (str(row.holter_lead_names[0]) if row.holter_lead_names is not None and len(row.holter_lead_names) else None)
            for row in cohort.itertuples(index=False)
        }
        windows["channel_name"] = windows["patient_id"].astype("string").map(lead_map)
    if "raw_rpeak_count" not in windows.columns:
        windows["raw_rpeak_count"] = pd.to_numeric(windows["beats"], errors="coerce").fillna(0).astype("int64")
    if "removed_rr_ratio" not in windows.columns:
        windows["removed_rr_ratio"] = 0.0
    patient_features = aggregate_patient_features(windows, min_successful_windows=4)
    if len(patient_features) != EXPECTED_PATIENTS:
        raise RuntimeError(f"minimum successful-window gate dropped patients: {len(patient_features)}")
    features_dir = repo / "data" / "features" / "legacy_120s"
    features_dir.mkdir(parents=True, exist_ok=True)
    windows_path = features_dir / "windows.parquet"
    feature_path = features_dir / "features.parquet"
    patients_path = features_dir / "patient_features.parquet"
    manifest, feature_table = _contract_tables(windows)
    manifest.to_parquet(windows_path, index=False, compression="zstd")
    feature_table.to_parquet(feature_path, index=False, compression="zstd")
    patient_features.to_parquet(patients_path, index=False, compression="zstd")

    validation_dir = repo / "data" / "validation"
    validation_dir.mkdir(parents=True, exist_ok=True)
    status_counts = windows["window_status"].value_counts(dropna=False).to_dict()
    rr_mismatch = int(
        (
            pd.to_numeric(windows["raw_rr_count"], errors="coerce")
            != pd.to_numeric(windows["valid_rr_count"], errors="coerce")
        ).sum()
    )
    validation = {
        "patient_count": int(len(cohort)),
        "positive_count": int(cohort["label"].sum()),
        "negative_count": int((cohort["label"] == 0).sum()),
        "theoretical_window_count": int(len(windows)),
        "expected_theoretical_window_count": EXPECTED_PATIENTS * MAX_WINDOWS,
        "window_status_counts": {str(k): int(v) for k, v in status_counts.items()},
        "patient_successful_window_min": int(patient_features["n_windows_successful"].min()),
        "patient_successful_window_max": int(patient_features["n_windows_successful"].max()),
        "raw_rr_equals_valid_rr_rows": int(len(windows) - rr_mismatch),
        "removed_rr_total": int(pd.to_numeric(windows["removed_rr_count"], errors="coerce").sum()),
        "read_contract": {
            "reader": "wfdb.rdrecord",
            "channels": [0],
            "maximum_requested_window_sec": WINDOW_SEC,
            "padding": False,
        },
    }
    validation_path = validation_dir / "legacy_window_validation.json"
    validation_path.write_text(json.dumps(validation, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    comparison = write_comparison_reports(repo=repo, patient_features=patient_features, old_csv=repo / OLD_DATASET_NAME)
    write_size_report(repo, [windows_path, feature_path, patients_path, validation_path], window_rows=len(windows), patient_rows=len(patient_features))
    return patient_features, {
        "cohort": {"patients": len(cohort), "positive": int(cohort["label"].sum()), "negative": int((cohort["label"] == 0).sum())},
        "windows": {"rows": len(windows), "status_counts": {str(k): int(v) for k, v in status_counts.items()}},
        "patient_features": {"rows": len(patient_features), "columns": len(patient_features.columns)},
        "comparison": comparison,
    }


def build(repo: Path = REPO, raw_root: Path | None = None, *, reuse_existing: bool = False) -> dict[str, Any]:
    legacy_internal = repo / "data" / "features" / "legacy_120s" / "legacy_windows.parquet"
    if reuse_existing:
        if not legacy_internal.is_file():
            raise FileNotFoundError(legacy_internal)
        windows = pd.read_parquet(legacy_internal)
    else:
        cohort = load_legacy_cohort(repo)
        raw = raw_root or default_raw_root(repo)
        windows = build_window_table(cohort, raw)
    _, result = _write_derived(repo, windows)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=REPO)
    parser.add_argument("--raw-root", type=Path, default=None)
    parser.add_argument("--reuse-existing-windows", action="store_true", help="rebuild contract tables without reading waveform")
    args = parser.parse_args(argv)
    result = build(args.repo.resolve(), args.raw_root.resolve() if args.raw_root else None, reuse_existing=args.reuse_existing_windows)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
