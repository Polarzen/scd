#!/usr/bin/env python3
"""Verify the self-contained MUSIC Phase 2 compact cohort state."""

from __future__ import annotations

import csv
import hashlib
import io
import sys
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
import yaml


REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
ALLOWED_STATES = {"POSITIVE", "NEGATIVE", "CENSORED", "COMPETING_EVENT", "UNKNOWN"}
ALLOWED_RECORD_TYPES = {"HOLTER", "HIGH_RESOLUTION"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
    for line_number, line in enumerate(manifest.read_text(encoding="ascii").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            expected, relative = line.split(maxsplit=1)
        except ValueError:
            errors.append(f"malformed compact hash line {line_number}")
            continue
        path = repo / relative
        if not path.is_file():
            errors.append(f"compact file missing: {relative}")
        elif sha256_file(path) != expected:
            errors.append(f"compact SHA256 mismatch: {relative}")
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
    return errors


def run_all(repo: Path = REPO) -> list[str]:
    errors: list[str] = []
    for verifier in (verify_hashes, verify_source_exact, verify_subjects, verify_records, verify_provenance, verify_endpoints, verify_data_contract):
        errors.extend(verifier(repo))
    return errors


def main() -> int:
    errors = run_all()
    if errors:
        for error in errors:
            print(f"FAIL: {error}", file=sys.stderr)
        return 1
    print("Phase 2 verification passed: hashes, source_exact, subjects, records, provenance, endpoints, and schemas")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
