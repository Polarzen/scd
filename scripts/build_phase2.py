#!/usr/bin/env python3
"""Build the complete MUSIC Phase 2 cohort state without reading ECG samples."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow
import yaml

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

try:
    from scripts.audit_music import (
        DOI,
        EXPECTED_ROOT_NAME,
        VERSION,
        audit,
        five_number,
        integer_code,
        number,
        parse_header,
        read_semicolon,
        sha256_file,
    )
except ModuleNotFoundError:
    from audit_music import (  # type: ignore
        DOI,
        EXPECTED_ROOT_NAME,
        VERSION,
        audit,
        five_number,
        integer_code,
        number,
        parse_header,
        read_semicolon,
        sha256_file,
    )


TOP_LEVEL_SOURCE_FILES = (
    "subject-info.csv",
    "subject-info_codes.csv",
    "subject-info_definitions.csv",
    "RECORDS",
    "SHA256SUMS.txt",
    "LICENSE.txt",
)
RECORD_TYPES = ("HOLTER", "HIGH_RESOLUTION")
CAUSE_CODES = {
    0: "SURVIVOR",
    1: "NON_CARDIAC_DEATH",
    3: "SCD",
    6: "PUMP_FAILURE_DEATH",
    7: "PUMP_FAILURE_DEATH",
}
EXIT_CODES = {0: "SURVIVOR", 1: "LOST_TO_FOLLOWUP", 2: "CARDIAC_TRANSPLANTATION", 3: "DEATH"}
RHYTHM_CODES = {0: "SINUS", 1: "PERMANENT_AF", 2: "ATRIAL_FLUTTER", 3: "PACEMAKER"}


def manifest_map(path: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for line in path.read_text(encoding="ascii").splitlines():
        if not line.strip():
            continue
        digest, relative = line.split(maxsplit=1)
        result[relative.lstrip("*").replace("\\", "/")] = digest.lower()
    return result


def nullable_string(value: Any) -> str | None:
    text = "" if value is None else str(value).strip()
    return text if text else None


def nullable_int(value: Any) -> int | None:
    parsed = number(value)
    if parsed is None or not parsed.is_integer():
        return None
    return int(parsed)


def official_flag(value: Any) -> bool | None:
    code = integer_code(value)
    if code == 1:
        return True
    if code == 0:
        return False
    return None


def event_source_valid(row: dict[str, str]) -> bool:
    followup = number(row.get("Follow-up period from enrollment (days)"))
    cause = integer_code(row.get("Cause of death"))
    exit_code = integer_code(row.get("Exit of the study"))
    exit_blank = not nullable_string(row.get("Exit of the study"))
    if followup is None or followup < 0 or cause not in CAUSE_CODES:
        return False
    if cause in {1, 3, 6, 7}:
        return exit_code == 3
    return cause == 0 and (exit_blank or exit_code in {0, 1, 2})


def copy_source_exact(root: Path, destination: Path, headers: list[dict[str, Any]]) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    names = set(TOP_LEVEL_SOURCE_FILES)
    names.update(path.name for pattern in ("LICENSE*", "README*") for path in root.glob(pattern) if path.is_file())
    for name in sorted(names):
        source = root / name
        if source.is_file():
            target = destination / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
    for item in headers:
        relative = Path(item["relative_path"])
        target = destination / "headers" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(item["path"], target)
    copied_dat = list(destination.rglob("*.dat"))
    if copied_dat:
        raise RuntimeError(f"Safety violation: copied .dat files: {copied_dat[:3]}")


def build_records(root: Path, headers: list[dict[str, Any]], manifest: dict[str, str], code_version: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for item in headers:
        relative_header = item["relative_path"]
        signal_files = item["signal_files"]
        signal_name = signal_files[0] if len(signal_files) == 1 else None
        signal_path = item["path"].parent / signal_name if signal_name else None
        signal_relative = signal_path.relative_to(root).as_posix() if signal_path else None
        dat_exists = bool(signal_path and signal_path.is_file())
        minimum_bytes = sum(item["expected_dat_sizes"].values()) if item["expected_dat_sizes"] else None
        dat_size = signal_path.stat().st_size if dat_exists and signal_path else None
        status: list[str] = []
        if item["anomalies"]:
            status.append("INVALID_HEADER")
        if signal_name is None:
            status.append("INVALID_HEADER")
        if not dat_exists:
            status.append("NO_DAT")
        if dat_size is not None and minimum_bytes is not None and dat_size < minimum_bytes:
            status.append("DAT_SHORTER_THAN_HEADER")
        hea_hash = manifest.get(relative_header)
        dat_hash = manifest.get(signal_relative) if signal_relative else None
        if hea_hash is None or dat_hash is None:
            status.append("OFFICIAL_SHA256_MISSING")
        rows.append(
            {
                "patient_id": item["patient_id"],
                "record_type": item["record_type"],
                "record_id": item["record_id"],
                "record_exists": dat_exists,
                "header_relative_path": relative_header,
                "signal_relative_path": signal_relative,
                "sampling_frequency": item["sampling_frequency"],
                "channel_count": item["signal_count"],
                "lead_names": item["lead_names"],
                "sample_count": item["sample_count"],
                "duration_sec": item["duration_sec"],
                "hea_file_size": item["path"].stat().st_size,
                "dat_file_size": dat_size,
                "hea_sha256": hea_hash,
                "dat_sha256": dat_hash,
                "sha256_source": "OFFICIAL_SHA256SUMS" if hea_hash is not None and dat_hash is not None else "NOT_AVAILABLE_WITHOUT_FULL_READ",
                "music_version": VERSION,
                "source_doi": DOI,
                "integrity_status": "OK" if not status else ";".join(sorted(set(status))),
                "build_code_version": code_version,
            }
        )
    frame = pd.DataFrame(rows).sort_values(["record_type", "record_id"], kind="stable").reset_index(drop=True)
    for column in ("patient_id", "record_type", "record_id", "header_relative_path", "signal_relative_path", "hea_sha256", "dat_sha256", "sha256_source", "music_version", "source_doi", "integrity_status", "build_code_version"):
        frame[column] = frame[column].astype("string")
    frame["record_exists"] = frame["record_exists"].astype("boolean")
    for column in ("channel_count", "sample_count", "hea_file_size", "dat_file_size"):
        frame[column] = frame[column].astype("Int64")
    for column in ("sampling_frequency", "duration_sec"):
        frame[column] = frame[column].astype("float64")
    return frame


def record_lookup(records: pd.DataFrame, kind: str) -> dict[str, dict[str, Any]]:
    subset = records.loc[records["record_type"] == kind]
    return {str(row["patient_id"]): row.to_dict() for _, row in subset.iterrows()}


def build_subjects(source_rows: list[dict[str, str]], records: pd.DataFrame) -> pd.DataFrame:
    holter = record_lookup(records, "HOLTER")
    high_resolution = record_lookup(records, "HIGH_RESOLUTION")
    official_columns = list(source_rows[0])
    output: list[dict[str, Any]] = []
    for source in source_rows:
        patient_id = source["Patient ID"].strip()
        h = holter.get(patient_id)
        hr = high_resolution.get(patient_id)
        metadata_holter = official_flag(source.get("Holter available"))
        metadata_hr = official_flag(source.get("Hig-resolution ECG available"))
        holter_header_exists = h is not None
        holter_dat_exists = bool(h and pd.notna(h["dat_file_size"]))
        hr_header_exists = hr is not None
        hr_dat_exists = bool(hr and pd.notna(hr["dat_file_size"]))
        has_holter = bool(h and h["integrity_status"] == "OK")
        has_hr = bool(hr and hr["integrity_status"] == "OK")
        mapping_consistent = metadata_holter == has_holter and metadata_hr == has_hr

        hr_rhythm = integer_code(source.get("ECG rhythm "))
        holter_rhythm = integer_code(source.get("Holter  rhythm "))
        selected_rhythm = holter_rhythm if holter_rhythm is not None else hr_rhythm
        af = hr_rhythm == 1 or holter_rhythm == 1
        decoded_rhythm = RHYTHM_CODES.get(selected_rhythm)
        if af:
            rhythm_group = "AF"
        elif decoded_rhythm == "SINUS":
            rhythm_group = "SINUS"
        elif decoded_rhythm == "ATRIAL_FLUTTER":
            rhythm_group = "ATRIAL_FLUTTER"
        elif decoded_rhythm == "PACEMAKER":
            rhythm_group = "PACED"
        else:
            rhythm_group = "UNKNOWN"

        followup = number(source.get("Follow-up period from enrollment (days)"))
        cause = integer_code(source.get("Cause of death"))
        exit_code = integer_code(source.get("Exit of the study"))
        primary_eligible: bool | None = False if not has_holter else None
        legacy_eligible: bool | None = False if not has_holter else None
        if not has_holter:
            hrv_eligible: bool | None = False
            primary_reason = legacy_reason = hrv_reason = "NO_HOLTER"
        else:
            primary_reason = legacy_reason = "PENDING_SIGNAL_QC"
            if af:
                hrv_eligible = False
                hrv_reason = "AF"
            else:
                hrv_eligible = None
                hrv_reason = "PENDING_SIGNAL_QC"

        raw_official = {column: nullable_string(source.get(column)) for column in official_columns}
        row: dict[str, Any] = {
            "patient_id": patient_id,
            **raw_official,
            "has_holter": has_holter,
            "has_high_resolution_ecg": has_hr,
            "holter_record_id": h["record_id"] if h else None,
            "high_resolution_record_id": hr["record_id"] if hr else None,
            "holter_sampling_frequency": h["sampling_frequency"] if h else None,
            "holter_channel_count": h["channel_count"] if h else None,
            "holter_lead_names": h["lead_names"] if h else None,
            "holter_sample_count": h["sample_count"] if h else None,
            "holter_duration_sec": h["duration_sec"] if h else None,
            "hr_sampling_frequency": hr["sampling_frequency"] if hr else None,
            "hr_channel_count": hr["channel_count"] if hr else None,
            "hr_lead_names": hr["lead_names"] if hr else None,
            "hr_sample_count": hr["sample_count"] if hr else None,
            "hr_duration_sec": hr["duration_sec"] if hr else None,
            "metadata_holter_flag": metadata_holter,
            "actual_holter_file_exists": holter_header_exists and holter_dat_exists,
            "holter_header_exists": holter_header_exists,
            "holter_dat_exists": holter_dat_exists,
            "metadata_hr_flag": metadata_hr,
            "actual_hr_file_exists": hr_header_exists and hr_dat_exists,
            "hr_header_exists": hr_header_exists,
            "hr_dat_exists": hr_dat_exists,
            "record_mapping_consistent": mapping_consistent,
            "record_integrity_status": "OK" if mapping_consistent and (not h or h["integrity_status"] == "OK") and (not hr or hr["integrity_status"] == "OK") else "METADATA_FILE_MISMATCH",
            "ecg_rhythm_raw": nullable_string(source.get("ECG rhythm ")),
            "holter_rhythm_raw": nullable_string(source.get("Holter  rhythm ")),
            "rhythm_raw": str(selected_rhythm) if selected_rhythm is not None else None,
            "rhythm_decoded": decoded_rhythm,
            "rhythm_group": rhythm_group,
            "af_flag": af,
            "pvc_count_24h": nullable_int(source.get("Number of ventricular premature beats in 24h")),
            "pvc_information_available": number(source.get("Number of ventricular premature beats in 24h")) is not None,
            "pvc_burden": None,
            "followup_days_raw": nullable_string(source.get("Follow-up period from enrollment (days)")),
            "followup_days": followup,
            "cause_of_death_raw": nullable_string(source.get("Cause of death")),
            "cause_of_death_decoded": CAUSE_CODES.get(cause),
            "exit_status_raw": nullable_string(source.get("Exit of the study")),
            "exit_status_decoded": EXIT_CODES.get(exit_code),
            "event_source_valid": event_source_valid(source),
            "eligible_primary_analysis": primary_eligible,
            "eligible_hrv_analysis": hrv_eligible,
            "eligible_legacy_analysis": legacy_eligible,
            "exclusion_reason_primary": primary_reason,
            "exclusion_reason_hrv": hrv_reason,
            "exclusion_reason_legacy": legacy_reason,
        }
        output.append(row)
    frame = pd.DataFrame(output)
    string_columns = official_columns + [
        "patient_id", "holter_record_id", "high_resolution_record_id", "record_integrity_status",
        "ecg_rhythm_raw", "holter_rhythm_raw", "rhythm_raw", "rhythm_decoded", "rhythm_group",
        "followup_days_raw", "cause_of_death_raw", "cause_of_death_decoded", "exit_status_raw",
        "exit_status_decoded", "exclusion_reason_primary", "exclusion_reason_hrv", "exclusion_reason_legacy",
    ]
    for column in dict.fromkeys(string_columns):
        frame[column] = frame[column].astype("string")
    bool_columns = [
        "has_holter", "has_high_resolution_ecg", "metadata_holter_flag", "actual_holter_file_exists",
        "holter_header_exists", "holter_dat_exists", "metadata_hr_flag", "actual_hr_file_exists",
        "hr_header_exists", "hr_dat_exists", "record_mapping_consistent", "af_flag",
        "pvc_information_available", "event_source_valid", "eligible_primary_analysis",
        "eligible_hrv_analysis", "eligible_legacy_analysis",
    ]
    for column in bool_columns:
        frame[column] = frame[column].astype("boolean")
    for column in ("holter_channel_count", "holter_sample_count", "hr_channel_count", "hr_sample_count", "pvc_count_24h"):
        frame[column] = frame[column].astype("Int64")
    for column in ("holter_sampling_frequency", "holter_duration_sec", "hr_sampling_frequency", "hr_duration_sec", "followup_days", "pvc_burden"):
        frame[column] = frame[column].astype("float64")
    return frame


def build_provenance(records: pd.DataFrame, code_version: str) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "patient_id": records["patient_id"],
            "record_id": records["record_id"],
            "record_type": records["record_type"],
            "source_dataset": "MUSIC (Sudden Cardiac Death in Chronic Heart Failure)",
            "source_version": VERSION,
            "source_doi": DOI,
            "source_relative_path": records["signal_relative_path"],
            "source_header_relative_path": records["header_relative_path"],
            "source_sha256": records["dat_sha256"],
            "header_sha256": records["hea_sha256"],
            "source_file_size": records["dat_file_size"],
            "header_file_size": records["hea_file_size"],
            "metadata_source_file": "subject-info.csv",
            "build_code_version": code_version,
        }
    )
    for column in frame.columns:
        if column not in {"source_file_size", "header_file_size"}:
            frame[column] = frame[column].astype("string")
    frame["source_file_size"] = frame["source_file_size"].astype("Int64")
    frame["header_file_size"] = frame["header_file_size"].astype("Int64")
    return frame


def write_schema(path: Path, subjects: pd.DataFrame, official_columns: list[str]) -> None:
    descriptions = {
        "patient_id": "Stable official MUSIC Patient ID; primary key.",
        "pvc_burden": "Not derivable reliably in Phase 2; always null pending an approved denominator.",
        "eligible_primary_analysis": "False if no Holter; otherwise null pending signal QC.",
        "eligible_hrv_analysis": "False for no Holter or official AF; otherwise null pending signal QC.",
        "eligible_legacy_analysis": "False if no Holter; otherwise null pending signal QC.",
    }
    columns = []
    for name in subjects.columns:
        official = name in official_columns
        columns.append(
            {
                "name": name,
                "dtype": str(subjects[name].dtype),
                "nullable": bool(subjects[name].isna().any()),
                "source": "subject-info.csv" if official else "Phase 2 derivation from official metadata/header/manifest",
                "derived": not official,
                "description": descriptions.get(name, "Official MUSIC field preserved as a nullable string." if official else "Phase 2 cohort-state field; see build_phase2.py."),
            }
        )
    payload = {"schema_version": 1, "table": "subjects", "primary_key": "patient_id", "columns": columns}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False, allow_unicode=True), encoding="utf-8")


def endpoint_counts(subjects: pd.DataFrame) -> dict[str, dict[str, int]]:
    from src.endpoints import build_endpoint

    result: dict[str, dict[str, int]] = {}
    states = ("POSITIVE", "NEGATIVE", "CENSORED", "COMPETING_EVENT", "UNKNOWN")
    for horizon in (90, 180, 365, 730):
        endpoint = build_endpoint(subjects, horizon)
        counts = endpoint["endpoint_state"].value_counts().to_dict()
        result[str(horizon)] = {state: int(counts.get(state, 0)) for state in states}
    return result


def directory_size(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file()) if path.exists() else 0


def write_compact_hashes(repo: Path) -> None:
    integrity = repo / "data" / "integrity"
    output = integrity / "compact_sha256.txt"
    candidates: list[Path] = []
    for base in (repo / "data" / "source_exact", repo / "data" / "cohort"):
        if base.exists():
            candidates.extend(path for path in base.rglob("*") if path.is_file())
    for path in (
        repo / "data" / "integrity" / "data_contract.yaml",
        repo / "config" / "reason_codes.yaml",
        repo / "config" / "cohort_schema.yaml",
        repo / "src" / "endpoints.py",
        repo / "scripts" / "build_phase2.py",
        repo / "scripts" / "verify_all.py",
        repo / "reports" / "PHASE2_COHORT_STATE.json",
        repo / "reports" / "PHASE2_COHORT_STATE.md",
        repo / "reports" / "phase2_size_report.json",
        repo / "requirements.txt",
        repo / "requirements-lock.txt",
        repo / "pytest.ini",
        repo / ".gitattributes",
    ):
        if path.is_file():
            candidates.append(path)
    candidates.extend(path for path in (repo / "tests").glob("test_*.py") if path.is_file())
    lines = [f"{sha256_file(path)}  {path.relative_to(repo).as_posix()}" for path in sorted(set(candidates), key=lambda p: p.relative_to(repo).as_posix())]
    output.write_text("\n".join(lines) + "\n", encoding="ascii")


def build(repo: Path, root: Path) -> dict[str, Any]:
    phase1_path = repo / "reports" / "DATA_AUDIT.json"
    if not phase1_path.is_file():
        raise RuntimeError("reports/DATA_AUDIT.json is missing")
    phase1 = json.loads(phase1_path.read_text(encoding="utf-8"))
    if not phase1.get("hard_gate", {}).get("passed") or phase1.get("hard_gate", {}).get("reasons"):
        raise RuntimeError("Phase 1 hard gate did not pass cleanly")
    if phase1.get("audit", {}).get("waveform_content_read") or phase1.get("audit", {}).get("feature_extraction_performed"):
        raise RuntimeError("Phase 1 safety scope is inconsistent")
    if root.name != EXPECTED_ROOT_NAME:
        raise RuntimeError(f"Refusing unexpected MUSIC root name: {root.name}")
    fresh_audit = audit(root)
    if not fresh_audit["hard_gate"]["passed"]:
        raise RuntimeError(f"Fresh source audit failed: {fresh_audit['hard_gate']['reasons']}")

    manifest = manifest_map(root / "SHA256SUMS.txt")
    headers: list[dict[str, Any]] = []
    for dirname, kind in (("Holter_ECG", "HOLTER"), ("High-resolution_ECG", "HIGH_RESOLUTION")):
        for path in sorted((root / dirname).rglob("*.hea")):
            parsed = parse_header(path, kind)
            parsed["relative_path"] = path.relative_to(root).as_posix()
            headers.append(parsed)
    code_version = "sha256:" + sha256_file(Path(__file__).resolve())[:16]
    source_rows = read_semicolon(root / "subject-info.csv")
    records = build_records(root, headers, manifest, code_version)
    subjects = build_subjects(source_rows, records)
    provenance = build_provenance(records, code_version)

    if len(subjects) != len(source_rows) or not subjects["patient_id"].is_unique:
        raise RuntimeError("Subject identity preservation failed")
    if set(records["patient_id"].dropna()) - set(subjects["patient_id"]):
        raise RuntimeError("Orphan record patient_id")
    if records.duplicated(["record_type", "record_id"]).any():
        raise RuntimeError("Duplicate record mapping")

    source_exact = repo / "data" / "source_exact"
    cohort_dir = repo / "data" / "cohort"
    integrity_dir = repo / "data" / "integrity"
    reports_dir = repo / "reports"
    config_dir = repo / "config"
    for directory in (source_exact, cohort_dir, integrity_dir, reports_dir, config_dir):
        directory.mkdir(parents=True, exist_ok=True)
    copy_source_exact(root, source_exact, headers)
    subjects.to_parquet(cohort_dir / "subjects.parquet", engine="pyarrow", compression="zstd", index=False)
    records.to_parquet(cohort_dir / "records.parquet", engine="pyarrow", compression="zstd", index=False)
    provenance.to_parquet(cohort_dir / "provenance.parquet", engine="pyarrow", compression="zstd", index=False)
    official_columns = list(source_rows[0])
    write_schema(config_dir / "cohort_schema.yaml", subjects, official_columns)

    contract = {
        "dataset_name": "MUSIC (Sudden Cardiac Death in Chronic Heart Failure)",
        "music_version": VERSION,
        "source_doi": DOI,
        "subject_count_from_source": len(source_rows),
        "subjects_primary_key": "patient_id",
        "records_foreign_key": "patient_id",
        "provenance_key": ["record_type", "record_id"],
        "required_source_files": list(TOP_LEVEL_SOURCE_FILES),
        "allowed_record_types": list(RECORD_TYPES),
        "allowed_endpoint_states": ["POSITIVE", "NEGATIVE", "CENSORED", "COMPETING_EVENT", "UNKNOWN"],
        "parquet_compression": "zstd",
        "missing_value_policy": "Use null; never coerce missing values to 0, -1, or a sentinel string.",
        "phase": 2,
        "waveform_content_included": False,
    }
    (integrity_dir / "data_contract.yaml").write_text(yaml.safe_dump(contract, sort_keys=False, allow_unicode=True), encoding="utf-8")

    endpoints = endpoint_counts(subjects)
    scd_followup = subjects.loc[subjects["cause_of_death_raw"] == "3", "followup_days"].dropna().astype(float).tolist()
    scd_bins = {
        "le_90d": sum(value <= 90 for value in scd_followup),
        "d91_180": sum(90 < value <= 180 for value in scd_followup),
        "d181_365": sum(180 < value <= 365 for value in scd_followup),
        "d366_730": sum(365 < value <= 730 for value in scd_followup),
        "gt_730d": sum(value > 730 for value in scd_followup),
    }
    mapping_inconsistent = int((~subjects["record_mapping_consistent"].fillna(False)).sum())
    report: dict[str, Any] = {
        "phase": "PHASE 2 - BUILD COMPLETE COHORT STATE",
        "generated_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "source": {"music_version": VERSION, "source_doi": DOI, "phase1_audit_passed": True},
        "safety": {"raw_write_operations": 0, "dat_waveform_content_read": False, "feature_extraction_performed": False, "phase_3_started": False},
        "cohort": {
            "official_patient_count": len(source_rows),
            "subjects_parquet_rows": len(subjects),
            "patient_ids_100_percent_preserved": set(subjects["patient_id"]) == {row["Patient ID"] for row in source_rows},
            "holter_patients": int(subjects["has_holter"].sum()),
            "no_holter_patients": int((~subjects["has_holter"]).sum()),
            "high_resolution_patients": int(subjects["has_high_resolution_ecg"].sum()),
            "no_high_resolution_patients": int((~subjects["has_high_resolution_ecg"]).sum()),
            "metadata_file_consistent_patients": len(subjects) - mapping_inconsistent,
            "metadata_file_inconsistent_patients": mapping_inconsistent,
            "af_patients": int(subjects["af_flag"].sum()),
            "rhythm_unknown_patients": int((subjects["rhythm_group"] == "UNKNOWN").sum()),
            "pvc_count_24h_available_patients": int(subjects["pvc_information_available"].sum()),
            "followup_available_patients": int(subjects["followup_days"].notna().sum()),
            "cause_of_death_available_patients": int(subjects["cause_of_death_raw"].notna().sum()),
        },
        "records": {
            "total": len(records),
            "holter": int((records["record_type"] == "HOLTER").sum()),
            "high_resolution": int((records["record_type"] == "HIGH_RESOLUTION").sum()),
            "mapping_anomalies": int(records.duplicated(["record_type", "record_id"]).sum()) + len(set(records["patient_id"]) - set(subjects["patient_id"])),
            "header_anomalies": int((records["integrity_status"].str.contains("INVALID_HEADER", na=False)).sum()),
            "missing_dat": int((records["integrity_status"].str.contains("NO_DAT", na=False)).sum()),
            "missing_hea": 0,
            "integrity_status_distribution": {str(key): int(value) for key, value in records["integrity_status"].value_counts().items()},
        },
        "endpoints": endpoints,
        "scd_interval": {
            "label": "baseline enrollment/Holter to recorded SCD outcome interval",
            "summary_days": five_number(scd_followup),
            "bins": scd_bins,
        },
        "eligibility": {
            "signal_qc_performed": False,
            "primary_pending_signal_qc": int(subjects["eligible_primary_analysis"].isna().sum()),
            "hrv_pending_signal_qc": int(subjects["eligible_hrv_analysis"].isna().sum()),
            "legacy_pending_signal_qc": int(subjects["eligible_legacy_analysis"].isna().sum()),
        },
        "unresolved_decoding": {
            "holter_rhythm_raw_codes_not_in_official_codes": sorted(set(subjects["holter_rhythm_raw"].dropna()) - {"0", "1", "2", "3"}),
            "pvc_burden": "null for all subjects; no approved denominator is available in Phase 2",
            "exit_status_blank_with_survivor_cause_count": int(((subjects["exit_status_raw"].isna()) & (subjects["cause_of_death_raw"] == "0")).sum()),
        },
    }
    report_json = reports_dir / "PHASE2_COHORT_STATE.json"
    report_json.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    endpoint_lines = []
    for horizon in (90, 180, 365, 730):
        values = endpoints[str(horizon)]
        endpoint_lines.append(f"| {horizon} | {values['POSITIVE']} | {values['NEGATIVE']} | {values['CENSORED']} | {values['COMPETING_EVENT']} | {values['UNKNOWN']} |")
    report_md = f"""# MUSIC Phase 2 — Complete Cohort State

> Phase 1 passed. This build copied official small files and headers byte-for-byte and did not open `.dat` waveform content. Phase 3 was not started.

## Cohort and records

| Item | Count |
|---|---:|
| Official patients / subjects.parquet rows | {len(source_rows)} / {len(subjects)} |
| Patient IDs preserved | {report['cohort']['patient_ids_100_percent_preserved']} |
| Holter / no Holter | {report['cohort']['holter_patients']} / {report['cohort']['no_holter_patients']} |
| High-resolution / none | {report['cohort']['high_resolution_patients']} / {report['cohort']['no_high_resolution_patients']} |
| Records total (Holter + high-resolution) | {len(records)} ({report['records']['holter']} + {report['records']['high_resolution']}) |
| Metadata/file inconsistencies | {mapping_inconsistent} |
| Header anomalies / missing dat / missing hea | {report['records']['header_anomalies']} / {report['records']['missing_dat']} / {report['records']['missing_hea']} |
| AF / rhythm unknown | {report['cohort']['af_patients']} / {report['cohort']['rhythm_unknown_patients']} |
| PVC count available | {report['cohort']['pvc_count_24h_available_patients']} |
| Follow-up / cause-of-death available | {report['cohort']['followup_available_patients']} / {report['cohort']['cause_of_death_available_patients']} |

## Endpoint audit

| Horizon (days) | POSITIVE | NEGATIVE | CENSORED | COMPETING_EVENT | UNKNOWN |
|---:|---:|---:|---:|---:|---:|
{chr(10).join(endpoint_lines)}

At 365 days, evaluable binary subjects are POSITIVE + NEGATIVE = {endpoints['365']['POSITIVE'] + endpoints['365']['NEGATIVE']}. Competing events and censored observations remain explicit and are not forced to zero.

## Recorded SCD interval

For the official SCD category, the **baseline enrollment/Holter to recorded SCD outcome interval** has n={len(scd_followup)} and min/P25/median/P75/max days = {five_number(scd_followup)}.

Bins: {scd_bins}

## Unresolved but preserved

- Official Holter rhythm raw code(s) absent from the supplied code table: {report['unresolved_decoding']['holter_rhythm_raw_codes_not_in_official_codes']}; decoded value remains null/UNKNOWN where selected.
- `pvc_burden` is null for every subject because Phase 2 has no approved denominator.
- Signal eligibility is nullable and marked `PENDING_SIGNAL_QC`; no signal-quality label was invented.

Phase 3 was not started.
"""
    (reports_dir / "PHASE2_COHORT_STATE.md").write_text(report_md, encoding="utf-8")

    size_report = {
        "source_exact": directory_size(source_exact),
        "subjects.parquet": (cohort_dir / "subjects.parquet").stat().st_size,
        "records.parquet": (cohort_dir / "records.parquet").stat().st_size,
        "provenance.parquet": (cohort_dir / "provenance.parquet").stat().st_size,
        "cohort_parquet_total": directory_size(cohort_dir),
        "config": directory_size(config_dir),
        "tests": directory_size(repo / "tests"),
        "scripts": directory_size(repo / "scripts"),
        "reports": directory_size(reports_dir),
    }
    size_path = reports_dir / "phase2_size_report.json"
    for _ in range(2):
        size_report["reports"] = directory_size(reports_dir)
        size_report["phase2_total"] = sum(size_report[key] for key in ("source_exact", "cohort_parquet_total", "config", "tests", "scripts", "reports"))
        size_path.write_text(json.dumps(size_report, indent=2) + "\n", encoding="utf-8")
    write_compact_hashes(repo)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--music-root", type=Path, help="Exact root; otherwise MUSIC_RAW_DIR is required")
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args(argv)
    value = args.music_root or (Path(os.environ["MUSIC_RAW_DIR"]) if os.environ.get("MUSIC_RAW_DIR") else None)
    if value is None:
        parser.error("MUSIC_RAW_DIR is not defined and --music-root was not supplied")
    try:
        root = value.resolve(strict=True)
        repo = args.repo_root.resolve(strict=True)
        if repo == root or root in repo.parents:
            raise RuntimeError("Repository/source path relationship is unsafe")
        report = build(repo, root)
    except Exception as exc:
        print(f"PHASE 2 BUILD ERROR: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({"subjects": report["cohort"]["subjects_parquet_rows"], "records": report["records"]["total"], "phase_3_started": False}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
