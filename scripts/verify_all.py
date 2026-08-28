#!/usr/bin/env python3
"""Verify the self-contained MUSIC compact cohort and finalized Phase 4 state."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import yaml


REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
ALLOWED_STATES = {"POSITIVE", "NEGATIVE", "CENSORED", "COMPETING_EVENT", "UNKNOWN"}
ALLOWED_RECORD_TYPES = {"HOLTER", "HIGH_RESOLUTION"}
SUPPORTED_HORIZONS = (90, 180, 365, 730)
RAW_CANDIDATE_SUFFIXES = {".dat", ".mat", ".wav", ".wfdb", ".npy", ".npz", ".zip", ".tar", ".gz"}
MAX_DATA_BYTES = 100 * 1024 * 1024
MAX_DATA_FILE_BYTES = 25 * 1024 * 1024
FULL_FEATURE_NAMES = (
    "sig_mean", "sig_std", "sig_p2p", "sig_skew", "sig_kurt", "beats", "beats_per_min",
    "mean_rr", "sdnn", "rmssd", "pnn50", "mean_hr", "rr_cv", "rr_sampen", "rr_apen",
    "rr_dfa_alpha", "pow_lf", "pow_mf", "pow_hf", "pow_hf_ratio",
)
FULL_VALIDITY_NAMES = tuple(f"{name}_valid" for name in FULL_FEATURE_NAMES)
FULL_AGGREGATE_NAMES = tuple(
    f"{name}_{suffix}" for name in FULL_FEATURE_NAMES for suffix in ("mean", "std", "p10", "p50", "p90")
)
FULL_QC_REASONS = {
    "", "AF_INCOMPATIBLE_HRV", "INSUFFICIENT_VALID_NN", "INVALID_RR_RATIO", "READ_ERROR",
    "EMPTY_SIGNAL", "SHORT_READ", "FEATURE_ERROR", "OUTSIDE_RECORD",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _expected_compact_paths(repo: Path) -> set[str]:
    """Return the source/data surface a finalized compact hash must cover."""

    paths: set[str] = set()
    roots = [repo / "data", repo / "config", repo / "src", repo / "scripts", repo / "tests", repo / ".github" / "workflows"]
    for root in roots:
        if not root.is_dir():
            continue
        for path in root.rglob("*"):
            if not path.is_file():
                continue
            relative = path.relative_to(repo).as_posix()
            parts = {part.lower() for part in path.relative_to(repo).parts}
            if relative in {"data/integrity/compact_sha256.txt", "data/integrity/build_manifest.json"}:
                continue
            if parts.intersection({"raw", "cache", "tmp", "temp", "build_cache", "__pycache__"}) or path.suffix.lower() in RAW_CANDIDATE_SUFFIXES or path.suffix.lower() in {".tmp", ".pyc"}:
                continue
            paths.add(relative)
    reports = repo / "reports"
    finalization_reports = {
        "reports/PHASE4_SIZE_REPORT.json",
        "reports/PHASE4_SIZE_REPORT.md",
        "reports/github_handoff.json",
    }
    if reports.is_dir():
        for path in reports.rglob("*"):
            if not path.is_file():
                continue
            relative = path.relative_to(repo).as_posix()
            if relative not in finalization_reports and path.suffix.lower() not in {".tmp", ".pyc"}:
                paths.add(relative)
    for name in (
        ".gitattributes", ".gitignore", "requirements.txt", "requirements-lock.txt",
        "README.md", "DATASET.md", "REPRODUCIBILITY.md",
    ):
        path = repo / name
        if path.is_file():
            paths.add(name)
    return paths


def source_rows(path: Path) -> list[dict[str, str]]:
    raw = path.read_bytes()
    for encoding in ("utf-8-sig", "cp1252", "latin-1"):
        try:
            text = raw.decode(encoding)
            break
        except UnicodeDecodeError:
            continue
    return list(csv.DictReader(io.StringIO(text, newline=""), delimiter=";"))


def verify_hashes(repo: Path = REPO) -> list[str]:
    errors: list[str] = []
    manifest = repo / "data" / "integrity" / "compact_sha256.txt"
    if not manifest.is_file():
        return ["compact_sha256.txt is missing"]
    seen: set[str] = set()
    try:
        lines = manifest.read_text(encoding="ascii").splitlines()
    except (OSError, UnicodeError) as exc:
        return [f"compact_sha256.txt cannot be read: {exc}"]
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            expected, relative = line.split(maxsplit=1)
        except ValueError:
            errors.append(f"malformed compact hash line {line_number}")
            continue
        relative = relative.strip().replace("\\", "/")
        path = (repo / relative).resolve()
        try:
            relative_check = path.relative_to(repo.resolve()).as_posix()
        except ValueError:
            errors.append(f"compact hash path escapes repository: {relative}")
            continue
        if relative_check in seen:
            errors.append(f"duplicate compact hash path: {relative_check}")
            continue
        seen.add(relative_check)
        if not re.fullmatch(r"[0-9a-f]{64}", expected):
            errors.append(f"malformed compact SHA256: {relative_check}")
            continue
        if relative_check == "data/integrity/compact_sha256.txt":
            errors.append("compact hash manifest must not hash itself")
            continue
        if path.suffix.lower() in RAW_CANDIDATE_SUFFIXES or bool({part.lower() for part in Path(relative_check).parts}.intersection({"raw", "cache", "tmp", "temp", "build_cache"})):
            errors.append(f"compact hash manifest includes prohibited raw/cache/temp path: {relative_check}")
            continue
        if not path.is_file():
            errors.append(f"compact file missing: {relative_check}")
        elif sha256_file(path) != expected:
            errors.append(f"compact SHA256 mismatch: {relative_check}")

    build_manifest_path = repo / "data" / "integrity" / "build_manifest.json"
    if build_manifest_path.is_file():
        try:
            build_manifest = json.loads(build_manifest_path.read_text(encoding="utf-8"))
            declared = build_manifest.get("hashes")
            if not isinstance(declared, dict):
                errors.append("build_manifest.json hashes must be a mapping")
            else:
                declared_paths = set()
                for relative, digest in declared.items():
                    relative = str(relative).replace("\\", "/")
                    declared_paths.add(relative)
                    if relative in {"data/integrity/compact_sha256.txt", "data/integrity/build_manifest.json"}:
                        errors.append(f"build manifest must not self-hash integrity output: {relative}")
                        continue
                    if Path(relative).suffix.lower() in RAW_CANDIDATE_SUFFIXES or bool({part.lower() for part in Path(relative).parts}.intersection({"raw", "cache", "tmp", "temp", "build_cache"})):
                        errors.append(f"build manifest includes prohibited raw/cache/temp path: {relative}")
                        continue
                    path = (repo / relative).resolve()
                    try:
                        path.relative_to(repo.resolve())
                    except ValueError:
                        errors.append(f"build manifest path escapes repository: {relative}")
                        continue
                    if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
                        errors.append(f"malformed build manifest SHA256: {relative}")
                    elif not path.is_file() or sha256_file(path) != digest:
                        errors.append(f"build manifest SHA256 mismatch: {relative}")
                missing_declared = sorted(declared_paths - seen)
                if missing_declared:
                    errors.append(f"compact hash manifest omits build manifest files: {missing_declared[:5]}")
                unexpected = sorted(seen - declared_paths - {"data/integrity/build_manifest.json"})
                if unexpected:
                    errors.append(f"compact hash manifest has files absent from build manifest: {unexpected[:5]}")
                omitted_compact = sorted(_expected_compact_paths(repo) - declared_paths)
                if omitted_compact:
                    errors.append(f"build manifest omits compact files: {omitted_compact[:5]}")
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            errors.append(f"invalid build_manifest.json: {exc}")
    return errors


def verify_source_exact(repo: Path = REPO) -> list[str]:
    errors: list[str] = []
    source = repo / "data" / "source_exact"
    manifest_path = source / "SHA256SUMS.txt"
    if not manifest_path.is_file():
        return ["source_exact/SHA256SUMS.txt is missing"]
    official: dict[str, str] = {}
    for line in manifest_path.read_text(encoding="ascii").splitlines():
        digest, relative = line.split(maxsplit=1)
        official[relative.lstrip("*").replace("\\", "/")] = digest
    exact_files = [path for path in source.rglob("*") if path.is_file()]
    if any(path.suffix.lower() == ".dat" for path in exact_files):
        errors.append("source_exact contains prohibited .dat content")
    for path in exact_files:
        relative = path.relative_to(source).as_posix()
        if relative.startswith("headers/"):
            official_relative = relative.removeprefix("headers/")
        else:
            official_relative = relative
        if official_relative == "SHA256SUMS.txt":
            continue
        expected = official.get(official_relative)
        if expected is None:
            errors.append(f"source_exact file absent from official manifest: {relative}")
        elif sha256_file(path) != expected:
            errors.append(f"source_exact differs from official SHA256: {relative}")
    return errors


def verify_subjects(repo: Path = REPO) -> list[str]:
    errors: list[str] = []
    subjects = pd.read_parquet(repo / "data" / "cohort" / "subjects.parquet")
    rows = source_rows(repo / "data" / "source_exact" / "subject-info.csv")
    source_ids = [row["Patient ID"].strip() for row in rows]
    if len(subjects) != len(rows):
        errors.append("subjects row count differs from source")
    if subjects["patient_id"].duplicated().any():
        errors.append("subjects patient_id is not unique")
    if set(subjects["patient_id"]) != set(source_ids):
        errors.append("subjects patient_id set differs from source")
    official_columns = list(rows[0])
    missing_columns = [column for column in official_columns if column not in subjects]
    if missing_columns:
        errors.append(f"official columns missing: {missing_columns}")
    else:
        indexed = subjects.set_index("patient_id", drop=False)
        for row in rows:
            patient_id = row["Patient ID"].strip()
            for column in official_columns:
                expected = row[column].strip() or None
                actual_value = indexed.at[patient_id, column]
                actual = None if pd.isna(actual_value) else str(actual_value)
                if actual != expected:
                    errors.append(f"official value mismatch: {patient_id}/{column}")
                    return errors
    if not subjects["record_mapping_consistent"].fillna(False).all():
        errors.append("record mapping inconsistency present")
    if (subjects.loc[~subjects["has_holter"], "holter_record_id"].notna()).any():
        errors.append("patient without Holter has a Holter record_id")
    schema = yaml.safe_load((repo / "config" / "cohort_schema.yaml").read_text(encoding="utf-8"))
    schema_columns = [item["name"] for item in schema["columns"]]
    if schema_columns != list(subjects.columns):
        errors.append("cohort schema columns differ from subjects parquet")
    return errors


def verify_records(repo: Path = REPO) -> list[str]:
    errors: list[str] = []
    subjects = pd.read_parquet(repo / "data" / "cohort" / "subjects.parquet")
    records = pd.read_parquet(repo / "data" / "cohort" / "records.parquet")
    if records.duplicated(["record_type", "record_id"]).any():
        errors.append("duplicate record key")
    if set(records["patient_id"]) - set(subjects["patient_id"]):
        errors.append("orphan record patient_id")
    if not set(records["record_type"]).issubset(ALLOWED_RECORD_TYPES):
        errors.append("invalid record_type")
    valid = records["integrity_status"] == "OK"
    if (records.loc[valid, "sampling_frequency"] <= 0).any():
        errors.append("non-positive sampling frequency")
    if (records.loc[valid, "sample_count"] <= 0).any():
        errors.append("non-positive sample count")
    expected = records.loc[valid, "sample_count"].astype(float) / records.loc[valid, "sampling_frequency"]
    if not ((expected - records.loc[valid, "duration_sec"]).abs() <= 1e-9).all():
        errors.append("duration inconsistent with sample_count/fs")
    if set(records["music_version"].dropna()) != {"1.0.1"}:
        errors.append("record source version mismatch")
    if not records.loc[valid, "dat_sha256"].str.fullmatch(r"[0-9a-f]{64}").all():
        errors.append("official dat SHA256 missing or malformed")
    for name in ("subjects.parquet", "records.parquet", "provenance.parquet"):
        parquet = pq.ParquetFile(repo / "data" / "cohort" / name)
        compressions = {parquet.metadata.row_group(rg).column(col).compression for rg in range(parquet.metadata.num_row_groups) for col in range(parquet.metadata.num_columns)}
        if compressions != {"ZSTD"}:
            errors.append(f"{name} compression is not uniformly ZSTD: {compressions}")
    return errors


def verify_provenance(repo: Path = REPO) -> list[str]:
    errors: list[str] = []
    records = pd.read_parquet(repo / "data" / "cohort" / "records.parquet").sort_values(["record_type", "record_id"]).reset_index(drop=True)
    provenance = pd.read_parquet(repo / "data" / "cohort" / "provenance.parquet").sort_values(["record_type", "record_id"]).reset_index(drop=True)
    if provenance.duplicated(["record_type", "record_id"]).any():
        errors.append("duplicate provenance key")
    if set(map(tuple, records[["record_type", "record_id"]].itertuples(index=False, name=None))) != set(map(tuple, provenance[["record_type", "record_id"]].itertuples(index=False, name=None))):
        errors.append("record/provenance key sets differ")
    comparisons = {
        "patient_id": "patient_id",
        "dat_sha256": "source_sha256",
        "hea_sha256": "header_sha256",
        "dat_file_size": "source_file_size",
        "hea_file_size": "header_file_size",
    }
    for left, right in comparisons.items():
        if not records[left].reset_index(drop=True).equals(provenance[right].reset_index(drop=True)):
            errors.append(f"record/provenance mismatch: {left}/{right}")
    return errors


def verify_endpoints(repo: Path = REPO) -> list[str]:
    from src.endpoints import build_endpoint

    errors: list[str] = []
    subjects = pd.read_parquet(repo / "data" / "cohort" / "subjects.parquet")
    for horizon in (90, 180, 365, 730):
        endpoint = build_endpoint(subjects, horizon)
        if len(endpoint) != len(subjects):
            errors.append(f"endpoint row count mismatch at {horizon}")
        if set(endpoint["endpoint_state"]) - ALLOWED_STATES:
            errors.append(f"invalid endpoint state at {horizon}")
        if endpoint["patient_id"].duplicated().any():
            errors.append(f"duplicate endpoint patient at {horizon}")
        censored = endpoint["endpoint_state"] == "CENSORED"
        if endpoint.loc[censored, "binary_label_if_evaluable"].notna().any():
            errors.append(f"censored endpoint has binary label at {horizon}")
    return errors


def verify_data_contract(repo: Path = REPO) -> list[str]:
    errors: list[str] = []
    contract_path = repo / "data" / "integrity" / "data_contract.yaml"
    if not contract_path.is_file():
        return ["data_contract.yaml is missing"]
    contract = yaml.safe_load(contract_path.read_text(encoding="utf-8"))
    subjects = pd.read_parquet(repo / "data" / "cohort" / "subjects.parquet")
    required = {
        "subject_count_from_source", "subjects_primary_key", "records_foreign_key",
        "required_source_files", "allowed_record_types", "allowed_endpoint_states",
        "music_version", "source_doi",
    }
    missing = sorted(required - set(contract))
    if missing:
        errors.append(f"data contract fields missing: {missing}")
        return errors
    if contract["subject_count_from_source"] != len(subjects):
        errors.append("data contract subject count mismatch")
    if contract["subjects_primary_key"] != "patient_id" or contract["records_foreign_key"] != "patient_id":
        errors.append("data contract key mismatch")
    if set(contract["allowed_record_types"]) != ALLOWED_RECORD_TYPES:
        errors.append("data contract record types mismatch")
    if set(contract["allowed_endpoint_states"]) != ALLOWED_STATES:
        errors.append("data contract endpoint states mismatch")
    if contract["music_version"] != "1.0.1" or contract["source_doi"] != "10.13026/z3m7-rf58":
        errors.append("data contract source identity mismatch")
    phase4 = contract.get("phase4")
    if not isinstance(phase4, dict):
        errors.append("data contract phase4 section is missing")
        return errors
    source = phase4.get("source", {})
    if not isinstance(source, dict):
        errors.append("data contract phase4 source section is malformed")
    else:
        expected_source = {
            "official_subject_count": 992,
            "record_count": 1623,
            "holter_record_count": 936,
            "high_resolution_record_count": 687,
            "no_holter_subject_count": 56,
        }
        for key, expected in expected_source.items():
            if source.get(key) != expected:
                errors.append(f"data contract phase4 source count mismatch: {key}")
    expected_keys = {
        "subjects": "patient_id",
        "records": ["record_type", "record_id"],
        "provenance": ["record_type", "record_id"],
        "windows": ["patient_id", "window_id"],
        "features": ["patient_id", "window_id"],
        "patient_aggregation": "patient_id",
        "patient_status": "patient_id",
        "survival_ready": "patient_id",
    }
    if phase4.get("keys") != expected_keys:
        errors.append("data contract phase4 key definitions mismatch")
    if set(phase4.get("endpoint_states", [])) != ALLOWED_STATES:
        errors.append("data contract phase4 endpoint states mismatch")
    windows = phase4.get("windows", {})
    if not isinstance(windows, dict) or windows.get("length_sec") != 300 or windows.get("stride_sec") != 300 or windows.get("first_start_sec") != 60:
        errors.append("data contract phase4 window schedule mismatch")
    if isinstance(windows, dict) and windows.get("start_formula") != "60 + 300*k":
        errors.append("data contract phase4 start formula mismatch")
    features = phase4.get("features", {})
    if not isinstance(features, dict) or features.get("base_feature_count") != 20 or features.get("aggregate_feature_count") != 100:
        errors.append("data contract phase4 feature count mismatch")
    sharding = phase4.get("sharding", {})
    if not isinstance(sharding, dict) or not all(sharding.get(key) is True for key in ("deterministic", "patient_sorted", "contiguous_patient_ranges")):
        errors.append("data contract phase4 sharding contract mismatch")
    raw_excluded = phase4.get("raw_excluded", {})
    if not isinstance(raw_excluded, dict) or raw_excluded.get("waveform_content") is not True or raw_excluded.get("raw_waveform_cache") is not True:
        errors.append("data contract phase4 raw exclusion mismatch")
    return errors


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _full_shard_paths(repo: Path, payload: dict[str, Any], key: str, pattern_key: str) -> list[Path]:
    full_dir = repo / "data" / "features" / "full_5min"
    values = payload.get(key)
    paths: list[Path] = []
    if isinstance(values, list):
        for item in values:
            value = item.get("path") if isinstance(item, dict) else item
            if not isinstance(value, str) or not value.strip():
                continue
            candidate = Path(value)
            path = (repo / candidate) if candidate.is_absolute() is False and (repo / candidate).is_file() else full_dir / candidate
            paths.append(path)
    else:
        pattern = payload.get(pattern_key)
        if isinstance(pattern, str) and pattern:
            paths = sorted(full_dir.glob(pattern))
    return paths


def _sort_key(patient_id: Any, window_id: Any) -> tuple[str, int]:
    return str(patient_id), int(window_id)


def _key_set(frame: pd.DataFrame, table_name: str, errors: list[str]) -> set[tuple[str, int]]:
    required = {"patient_id", "window_id"}
    missing = sorted(required - set(frame.columns))
    if missing:
        errors.append(f"{table_name} missing key columns: {missing}")
        return set()
    if frame["patient_id"].isna().any() or frame["window_id"].isna().any():
        errors.append(f"{table_name} contains null key values")
    try:
        keys = {_sort_key(patient, window) for patient, window in frame[["patient_id", "window_id"]].itertuples(index=False, name=None)}
    except (TypeError, ValueError, OverflowError):
        errors.append(f"{table_name} has malformed key values")
        return set()
    if len(keys) != len(frame):
        errors.append(f"{table_name} contains duplicate (patient_id, window_id) keys")
    return keys


def _verify_sorted_keys(frame: pd.DataFrame, table_name: str, errors: list[str]) -> None:
    if not {"patient_id", "window_id"}.issubset(frame.columns) or frame.empty:
        return
    try:
        actual = [_sort_key(patient, window) for patient, window in frame[["patient_id", "window_id"]].itertuples(index=False, name=None)]
        if actual != sorted(actual):
            errors.append(f"{table_name} is not sorted by patient_id, window_id")
    except (TypeError, ValueError, OverflowError):
        # _key_set reports malformed keys with a more specific finding.
        return


def _verify_declared_shards(
    paths: list[Path],
    declarations: Any,
    repo: Path,
    table_name: str,
    errors: list[str],
) -> None:
    if not isinstance(declarations, list):
        return
    if len(declarations) != len(paths):
        errors.append(f"{table_name} manifest shard count differs from files")
    for path, item in zip(paths, declarations):
        if not isinstance(item, dict) or not path.is_file():
            continue
        if isinstance(item.get("rows"), int):
            try:
                actual_rows = len(pd.read_parquet(path, columns=["patient_id"]))
                if actual_rows != item["rows"]:
                    errors.append(f"{table_name} declared row count differs: {path.relative_to(repo)}")
            except Exception:
                pass
        if isinstance(item.get("bytes"), int) and path.stat().st_size != item["bytes"]:
            errors.append(f"{table_name} declared byte count differs: {path.relative_to(repo)}")
        try:
            frame = pd.read_parquet(path, columns=["patient_id", "window_id"])
            if len(frame):
                if str(item.get("patient_first")) != str(frame["patient_id"].iloc[0]) or str(item.get("patient_last")) != str(frame["patient_id"].iloc[-1]):
                    errors.append(f"{table_name} declared patient range differs: {path.relative_to(repo)}")
        except Exception:
            pass


def _expected_window_count(sample_count: Any, sampling_frequency: Any) -> int | None:
    try:
        samples = int(sample_count)
        fs = float(sampling_frequency)
    except (TypeError, ValueError, OverflowError):
        return None
    if samples <= 0 or not np.isfinite(fs) or fs <= 0:
        return None
    duration = samples / fs
    # This is intentionally derived only from the frozen sample_count/fs pair
    # and the 60 + 300*k, 300-second complete-window rule.
    return max(0, int(math.floor((duration - 60.0 - 300.0) / 300.0 + 1e-12)) + 1)


def _verify_parquet_compression(path: Path, errors: list[str]) -> None:
    try:
        parquet = pq.ParquetFile(path)
        compressions = {
            parquet.metadata.row_group(row_group).column(column).compression
            for row_group in range(parquet.metadata.num_row_groups)
            for column in range(parquet.metadata.num_columns)
        }
    except Exception as exc:  # pyarrow errors are useful as a verifier finding
        errors.append(f"cannot inspect parquet compression: {path}: {exc}")
        return
    if compressions != {"ZSTD"}:
        errors.append(f"{path} compression is not uniformly ZSTD: {compressions}")


def _verify_full_table_schema(
    windows: pd.DataFrame,
    features: pd.DataFrame,
    errors: list[str],
) -> None:
    required_windows = {
        "patient_id", "record_id", "window_id", "start_sec", "end_sec", "start_sample",
        "requested_samples", "actual_samples", "sampling_frequency", "window_expected",
        "window_within_record", "waveform_read_success", "feature_extraction_success", "qc_valid",
        "window_status", "failure_reason", "qc_status", "qc_reason", "raw_rpeak_count",
        "raw_rr_count", "valid_rr_count", "removed_rr_count", "removed_rr_ratio", "tail_seconds",
    }
    missing_windows = sorted(required_windows - set(windows.columns))
    if missing_windows:
        errors.append(f"full_5min windows missing schema columns: {missing_windows}")
    required_features = {"patient_id", "window_id", "valid", "qc_valid", "qc_status", "qc_reason", *FULL_FEATURE_NAMES, *FULL_VALIDITY_NAMES}
    missing_features = sorted(required_features - set(features.columns))
    if missing_features:
        errors.append(f"full_5min features missing schema columns: {missing_features}")
    for name in FULL_FEATURE_NAMES:
        if name in features:
            if str(features[name].dtype) != "float64":
                errors.append(f"full_5min feature {name} is not float64")
            numeric = pd.to_numeric(features[name], errors="coerce")
            if name + "_valid" in features:
                valid = features[name + "_valid"].fillna(False)
                if not pd.api.types.is_bool_dtype(features[name + "_valid"]):
                    errors.append(f"full_5min validity column {name}_valid is not boolean")
                if (~valid.astype(bool) & numeric.notna()).any():
                    errors.append(f"full_5min invalid feature {name} contains a numeric value")
                if (valid.astype(bool) & ~np.isfinite(numeric.fillna(np.nan))).any():
                    errors.append(f"full_5min valid feature {name} contains null/non-finite values")
    for name, frame in (("windows", windows), ("features", features)):
        for column in ("qc_valid", "valid"):
            if column in frame and not pd.api.types.is_bool_dtype(frame[column]):
                errors.append(f"full_5min {name}.{column} is not boolean")
        if "qc_status" in frame:
            unknown_status = set(frame["qc_status"].fillna("").astype(str)) - {"PASS", "FAIL"}
            if unknown_status:
                errors.append(f"full_5min {name}.qc_status has unknown codes: {sorted(unknown_status)[:5]}")
        if "qc_reason" in frame:
            unknown_reasons: set[str] = set()
            for value in frame["qc_reason"].fillna("").astype(str):
                unknown_reasons.update(token for token in value.split(";") if token not in FULL_QC_REASONS)
            if unknown_reasons:
                errors.append(f"full_5min {name}.qc_reason has unknown codes: {sorted(unknown_reasons)[:5]}")
    if "qc_valid" in features and "qc_status" in features:
        pass_rows = features["qc_valid"].fillna(False).astype(bool)
        if features.loc[pass_rows, "qc_status"].astype(str).isin({"FAIL", "UNKNOWN"}).any():
            errors.append("full_5min feature qc_valid rows have a failing QC status")
        if features.loc[~pass_rows, "qc_status"].astype(str).isin({"PASS"}).any():
            errors.append("full_5min feature qc-invalid rows have PASS QC status")


def _verify_full_window_values(
    windows: pd.DataFrame,
    records: pd.DataFrame,
    errors: list[str],
) -> None:
    holter = records.loc[records["record_type"].astype("string").eq("HOLTER")].copy()
    if holter.duplicated("patient_id").any():
        errors.append("HOLTER records are not one-to-one by patient_id")
    record_map = {str(row["patient_id"]): row for row in holter.to_dict("records")}
    grouped = windows.groupby(windows["patient_id"].astype("string"), sort=False)
    expected_patients: set[str] = set()
    for patient_id, record in record_map.items():
        expected = _expected_window_count(record.get("sample_count"), record.get("sampling_frequency"))
        if expected is None:
            errors.append(f"cannot derive theoretical window count for {patient_id}")
            continue
        if expected:
            expected_patients.add(patient_id)
        actual = int(len(grouped.get_group(patient_id))) if patient_id in grouped.groups else 0
        if actual != expected:
            errors.append(f"theoretical window count mismatch for {patient_id}: expected {expected}, got {actual}")
    extra_patients = sorted(set(map(str, windows["patient_id"].dropna())) - set(record_map))
    if extra_patients:
        errors.append(f"full_5min windows contain patients without HOLTER records: {extra_patients[:5]}")
    actual_patients = set(map(str, windows["patient_id"].dropna()))
    if actual_patients != expected_patients:
        errors.append("full_5min window patient set differs from theoretical complete-window patient set")
    for patient_id, group in grouped:
        key = str(patient_id)
        record = record_map.get(key)
        if record is None:
            continue
        ordered = group.sort_values("window_id", kind="stable")
        if "record_id" in ordered and set(ordered["record_id"].dropna().astype(str)) != {str(record.get("record_id"))}:
            errors.append(f"full_5min record_id mapping mismatch for {key}")
        ids = pd.to_numeric(ordered["window_id"], errors="coerce")
        if ids.isna().any() or ids.astype(int).tolist() != list(range(len(ordered))):
            errors.append(f"full_5min window_id sequence is not contiguous for {key}")
        fs = float(record["sampling_frequency"])
        starts = pd.to_numeric(ordered["start_sec"], errors="coerce")
        expected_starts = 60 + 300 * pd.to_numeric(ordered["window_id"], errors="coerce")
        if not np.allclose(starts.to_numpy(dtype=float), expected_starts.to_numpy(dtype=float), rtol=0.0, atol=1e-9):
            errors.append(f"full_5min start formula mismatch for {key}")
        ends = pd.to_numeric(ordered["end_sec"], errors="coerce")
        if not np.allclose(ends.to_numpy(dtype=float), starts.to_numpy(dtype=float) + 300.0, rtol=0.0, atol=1e-9):
            errors.append(f"full_5min end formula mismatch for {key}")
        if "sampling_frequency" in ordered and not np.allclose(pd.to_numeric(ordered["sampling_frequency"], errors="coerce"), fs, rtol=0.0, atol=1e-9):
            errors.append(f"full_5min sampling frequency mismatch for {key}")
        for column in ("window_expected", "window_within_record"):
            if column in ordered and not ordered[column].fillna(False).astype(bool).all():
                errors.append(f"full_5min {column} is false for a complete window in {key}")


def _verify_patient_table(path: Path, subjects: pd.DataFrame, errors: list[str], label: str) -> pd.DataFrame | None:
    if not path.is_file():
        errors.append(f"{label} is missing: {path.relative_to(REPO)}")
        return None
    try:
        frame = pd.read_parquet(path)
    except Exception as exc:
        errors.append(f"cannot read {label}: {exc}")
        return None
    if "patient_id" not in frame:
        errors.append(f"{label} missing patient_id")
        return frame
    if len(frame) != 992:
        errors.append(f"{label} row count is {len(frame)}, expected 992")
    if frame["patient_id"].isna().any() or frame["patient_id"].duplicated().any():
        errors.append(f"{label} patient_id is not non-null and unique")
    if set(frame["patient_id"].astype(str)) != set(subjects["patient_id"].astype(str)):
        errors.append(f"{label} patient_id set differs from subjects")
    return frame


def _same_value(left: Any, right: Any) -> bool:
    if pd.isna(left) and pd.isna(right):
        return True
    return left == right


def _verify_stored_endpoints(frame: pd.DataFrame, subjects: pd.DataFrame, errors: list[str], label: str) -> None:
    if frame is None:
        return
    expected_by_horizon: dict[int, pd.DataFrame] = {}
    from src.endpoints import build_endpoint

    for horizon in SUPPORTED_HORIZONS:
        expected = build_endpoint(subjects, horizon).set_index("patient_id")
        expected_by_horizon[horizon] = expected
        state_col = f"endpoint_{horizon}_state"
        label_col = f"endpoint_{horizon}_binary_label"
        time_col = f"endpoint_{horizon}_time_to_event"
        event_col = f"endpoint_{horizon}_event_type"
        missing = [column for column in (state_col, label_col, time_col, event_col) if column not in frame]
        if missing:
            errors.append(f"{label} missing endpoint columns for {horizon}: {missing}")
            continue
        observed = frame.set_index("patient_id")
        for patient_id, row in observed.iterrows():
            if patient_id not in expected.index:
                continue
            target = expected.loc[patient_id]
            checks = ((state_col, "endpoint_state"), (label_col, "binary_label_if_evaluable"), (time_col, "time_to_event"), (event_col, "event_type"))
            for observed_column, expected_column in checks:
                if not _same_value(row[observed_column], target[expected_column]):
                    errors.append(f"{label} endpoint mismatch {patient_id}/{horizon}/{observed_column}")
                    break


def _verify_model_artifacts(repo: Path, subjects: pd.DataFrame, errors: list[str]) -> None:
    validation = repo / "data" / "validation"
    if not validation.is_dir():
        return
    for path in sorted(validation.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in {".parquet", ".csv"}:
            continue
        try:
            frame = pd.read_parquet(path) if path.suffix.lower() == ".parquet" else pd.read_csv(path)
        except Exception:
            continue
        is_model_path = any(token in path.name.lower() for token in ("oof", "fold", "split", "population", "prediction")) or "full_model" in {part.lower() for part in path.parts}
        if "patient_id" not in frame or not is_model_path:
            continue
        patient = frame["patient_id"].astype("string")
        if patient.isna().any() or (~patient.isin(subjects["patient_id"].astype("string"))).any():
            errors.append(f"model artifact has an unknown/null patient_id: {path.relative_to(repo).as_posix()}")
        if "oof" in path.name.lower():
            # A single nested-CV run has one row per patient.  Repeated-CV
            # deliberately repeats each patient once per seed/model/profile,
            # so validate the complete run identity instead of patient_id
            # alone when those dimensions are present.
            identity = ["patient_id"] + [
                column
                for column in ("seed", "model", "profile", "endpoint_horizon_days")
                if column in frame
            ]
            if frame.duplicated(identity).any():
                errors.append(
                    "OOF model artifact has duplicate run/patient rows: "
                    f"{path.relative_to(repo).as_posix()}"
                )
        split_column = next((column for column in ("split", "partition", "set", "dataset") if column in frame), None)
        fold_column = next((column for column in ("fold", "outer_fold", "fold_id") if column in frame), None)
        if split_column and fold_column:
            for _, group in frame.groupby(fold_column, dropna=False):
                buckets = {str(value).lower(): set(group.loc[group[split_column].astype(str).str.lower().eq(str(value).lower()), "patient_id"].astype(str)) for value in group[split_column].dropna().unique()}
                train = set().union(*(values for name, values in buckets.items() if "train" in name))
                test = set().union(*(values for name, values in buckets.items() if "test" in name or "valid" in name))
                if train & test:
                    errors.append(f"model artifact has patient leakage between train/test: {path.relative_to(repo).as_posix()}")


def _verify_data_size_and_raw_candidates(repo: Path, errors: list[str]) -> None:
    data_root = repo / "data"
    if data_root.is_dir():
        files = [path for path in data_root.rglob("*") if path.is_file()]
        total = sum(path.stat().st_size for path in files)
        if total >= MAX_DATA_BYTES:
            errors.append(f"repository data size is {total} bytes, limit is below {MAX_DATA_BYTES}")
        oversized = [path.relative_to(repo).as_posix() for path in files if path.stat().st_size >= MAX_DATA_FILE_BYTES]
        if oversized:
            errors.append(f"data files reach the 25 MiB limit: {oversized[:5]}")
        candidates = [
            path.relative_to(repo).as_posix()
            for path in files
            if path.suffix.lower() in RAW_CANDIDATE_SUFFIXES
            or bool({part.lower() for part in path.relative_to(repo).parts}.intersection({"raw", "cache", "tmp", "temp", "build_cache"}))
        ]
        if candidates:
            errors.append(f"prohibited raw waveform candidates under data/: {candidates[:5]}")
    try:
        result = subprocess.run(["git", "ls-files", "-z"], cwd=repo, check=True, capture_output=True)
        tracked = [item.decode("utf-8") for item in result.stdout.split(b"\0") if item]
    except (OSError, subprocess.SubprocessError):
        tracked = []
    tracked_candidates = [path for path in tracked if path.startswith("data/") and Path(path).suffix.lower() in RAW_CANDIDATE_SUFFIXES]
    if tracked_candidates:
        errors.append(f"tracked data contains prohibited raw waveform candidates: {tracked_candidates[:5]}")


def verify_phase4(repo: Path = REPO) -> list[str]:
    """Verify Phase 4 artifacts only after the full build reports COMPLETE."""

    repo = Path(repo).resolve()
    report_path = repo / "reports" / "FULL_COHORT_BUILD.json"
    if not report_path.is_file():
        return []
    report = _read_json(report_path)
    if report is None:
        return ["FULL_COHORT_BUILD.json is invalid"]
    status = str(report.get("status", "")).upper()
    if status != "COMPLETE":
        return []
    errors: list[str] = []
    full_dir = repo / "data" / "features" / "full_5min"
    manifest_path = full_dir / "manifest.json"
    payload = _read_json(manifest_path) if manifest_path.is_file() else None
    if payload is None:
        return ["full_5min manifest.json is missing or invalid"]
    window_paths = _full_shard_paths(repo, payload, "window_shards", "window_table")
    feature_paths = _full_shard_paths(repo, payload, "feature_shards", "feature_table")
    if not window_paths or not feature_paths:
        errors.append("full_5min manifest has no window/feature shards")
    _verify_declared_shards(window_paths, payload.get("window_shards"), repo, "full_5min window", errors)
    _verify_declared_shards(feature_paths, payload.get("feature_shards"), repo, "full_5min feature", errors)
    for path in [*window_paths, *feature_paths]:
        if not path.is_file():
            errors.append(f"full_5min shard is missing: {path.relative_to(repo)}")
        elif path.stat().st_size >= MAX_DATA_FILE_BYTES:
            errors.append(f"full_5min shard reaches 25 MiB: {path.relative_to(repo)}")
        else:
            _verify_parquet_compression(path, errors)
    try:
        windows = pd.concat([pd.read_parquet(path) for path in window_paths if path.is_file()], ignore_index=True) if any(path.is_file() for path in window_paths) else pd.DataFrame()
    except Exception as exc:
        errors.append(f"cannot read full_5min window shards: {exc}")
        windows = pd.DataFrame()
    try:
        features = pd.concat([pd.read_parquet(path) for path in feature_paths if path.is_file()], ignore_index=True) if any(path.is_file() for path in feature_paths) else pd.DataFrame()
    except Exception as exc:
        errors.append(f"cannot read full_5min feature shards: {exc}")
        features = pd.DataFrame()
    window_keys = _key_set(windows, "full_5min windows", errors)
    feature_keys = _key_set(features, "full_5min features", errors)
    _verify_sorted_keys(windows, "full_5min windows", errors)
    _verify_sorted_keys(features, "full_5min features", errors)
    if window_keys != feature_keys:
        errors.append("full_5min manifest/features key sets differ")
    if isinstance(payload.get("window_row_count"), int) and payload["window_row_count"] != len(windows):
        errors.append("full_5min manifest window_row_count differs from shards")
    if isinstance(payload.get("max_shard_bytes"), (int, float)) and payload["max_shard_bytes"] > MAX_DATA_FILE_BYTES:
        errors.append("full_5min manifest max_shard_bytes exceeds 25 MiB")
    if not bool(payload.get("patient_sorted_contiguous", False)):
        errors.append("full_5min manifest does not declare sorted contiguous patient shards")
    if payload.get("compression") != "zstd":
        errors.append("full_5min manifest compression is not zstd")
    if payload.get("feature_dtype") != "float64":
        errors.append("full_5min manifest feature_dtype is not float64")
    _verify_full_table_schema(windows, features, errors)
    try:
        records = pd.read_parquet(repo / "data" / "cohort" / "records.parquet")
        _verify_full_window_values(windows, records, errors)
    except Exception as exc:
        errors.append(f"cannot verify full_5min window schedule: {exc}")
    try:
        subjects = pd.read_parquet(repo / "data" / "cohort" / "subjects.parquet")
    except Exception as exc:
        errors.append(f"cannot read subjects for Phase 4 verification: {exc}")
        return errors
    patient_path = full_dir / "patient_features.parquet"
    patient_features = _verify_patient_table(patient_path, subjects, errors, "full_5min patient_features")
    if patient_features is not None:
        missing = sorted(set(FULL_AGGREGATE_NAMES) - set(patient_features.columns))
        if missing:
            errors.append(f"patient aggregation missing feature columns: {missing[:5]}")
        aggregate_columns = [column for column in patient_features if column in FULL_AGGREGATE_NAMES]
        if len(aggregate_columns) != 100:
            errors.append(f"patient aggregation has {len(aggregate_columns)} aggregate columns, expected 100")
        for column in aggregate_columns:
            if str(patient_features[column].dtype) != "float64":
                errors.append(f"patient aggregation feature {column} is not float64")
    status = _verify_patient_table(repo / "data" / "analysis" / "patient_analysis_status.parquet", subjects, errors, "patient_analysis_status")
    survival = _verify_patient_table(repo / "data" / "analysis" / "survival_ready.parquet", subjects, errors, "survival_ready")
    try:
        _verify_stored_endpoints(status, subjects, errors, "patient_analysis_status")
        _verify_stored_endpoints(survival, subjects, errors, "survival_ready")
    except Exception as exc:
        errors.append(f"cannot verify stored Phase 4 endpoints: {exc}")
    for frame, name in ((windows, "windows"), (features, "features")):
        outcome_columns = [column for column in frame if re.search(r"(?:label|outcome|followup|cause_of_death|endpoint|event_type|survival)", str(column), re.IGNORECASE)]
        if outcome_columns:
            errors.append(f"full_5min {name} contains outcome/leakage columns: {outcome_columns[:5]}")
    _verify_model_artifacts(repo, subjects, errors)
    _verify_data_size_and_raw_candidates(repo, errors)
    return errors


# Compatibility aliases for callers that name the artifact family directly.
verify_full_5min = verify_phase4
verify_phase4_artifacts = verify_phase4


def run_all(repo: Path = REPO) -> list[str]:
    errors: list[str] = []
    for verifier in (verify_hashes, verify_source_exact, verify_subjects, verify_records, verify_provenance, verify_endpoints, verify_data_contract):
        errors.extend(verifier(repo))
    errors.extend(verify_phase4(repo))
    return errors


def main() -> int:
    errors = run_all()
    if errors:
        for error in errors:
            print(f"FAIL: {error}", file=sys.stderr)
        return 1
    report = _read_json(REPO / "reports" / "FULL_COHORT_BUILD.json")
    phase4_status = str(report.get("status", "")).upper() if report else "NOT_AVAILABLE"
    suffix = "; Phase 4 checks skipped until FULL_COHORT_BUILD status is COMPLETE" if phase4_status != "COMPLETE" else "; Phase 4 checks passed"
    print("Phase 2 verification passed: hashes, source_exact, subjects, records, provenance, endpoints, and schemas" + suffix)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
