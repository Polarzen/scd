#!/usr/bin/env python3
"""Read-only Phase 1 audit for MUSIC v1.0.1.

This program deliberately never opens a .dat file. It reads official CSV/text
metadata and WFDB headers, and uses filesystem metadata only for .dat files.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import os
import platform
import re
import sys
import io
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

VERSION = "1.0.1"
DOI = "10.13026/z3m7-rf58"
EXPECTED_ROOT_NAME = "music-sudden-cardiac-death-in-chronic-heart-failure-1.0.1"
REQUIRED_FILES = (
    "subject-info.csv",
    "subject-info_codes.csv",
    "subject-info_definitions.csv",
    "RECORDS",
    "SHA256SUMS.txt",
    "LICENSE.txt",
)
PATIENT_RE = re.compile(r"^(P\d+)(?:_H)?$")
SHA_RE = re.compile(r"^([0-9a-fA-F]{64})\s+[*]?(.+?)\s*$")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_source_text(path: Path) -> str:
    """Decode official text without ever rewriting it."""
    raw = path.read_bytes()
    for encoding in ("utf-8-sig", "cp1252"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    return raw.decode("latin-1")


def read_semicolon(path: Path) -> list[dict[str, str]]:
    return list(csv.DictReader(io.StringIO(read_source_text(path), newline=""), delimiter=";"))


def number(value: Any) -> float | None:
    text = "" if value is None else str(value).strip().strip("'").strip('"')
    if not text or text.upper() in {"NA", "NAN", "NULL"}:
        return None
    try:
        return float(text.replace(",", "."))
    except ValueError:
        return None


def integer_code(value: Any) -> int | None:
    parsed = number(value)
    if parsed is None or not parsed.is_integer():
        return None
    return int(parsed)


def five_number(values: list[float]) -> dict[str, float | int | None]:
    clean = sorted(v for v in values if math.isfinite(v))
    if not clean:
        return {"count": 0, "min": None, "p25": None, "median": None, "p75": None, "max": None}

    def percentile(p: float) -> float:
        if len(clean) == 1:
            return clean[0]
        position = (len(clean) - 1) * p
        lower = math.floor(position)
        upper = math.ceil(position)
        if lower == upper:
            return clean[lower]
        return clean[lower] + (clean[upper] - clean[lower]) * (position - lower)

    def tidy(value: float) -> float | int:
        return int(value) if value.is_integer() else round(value, 6)

    return {
        "count": len(clean),
        "min": tidy(clean[0]),
        "p25": tidy(percentile(0.25)),
        "median": tidy(percentile(0.5)),
        "p75": tidy(percentile(0.75)),
        "max": tidy(clean[-1]),
    }


def parse_header(path: Path, record_type: str) -> dict[str, Any]:
    result: dict[str, Any] = {
        "path": path,
        "relative_path": None,
        "record_type": record_type,
        "record_id": path.stem,
        "patient_id": None,
        "signal_count": None,
        "sampling_frequency": None,
        "sample_count": None,
        "duration_sec": None,
        "lead_names": [],
        "signal_files": [],
        "expected_dat_sizes": {},
        "anomalies": [],
    }
    match = PATIENT_RE.match(path.stem)
    if match:
        result["patient_id"] = match.group(1)
    else:
        result["anomalies"].append("UNPARSEABLE_PATIENT_ID")
    try:
        lines = [line.strip() for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip() and not line.lstrip().startswith("#")]
    except (OSError, UnicodeError) as exc:
        result["anomalies"].append(f"HEADER_READ_ERROR:{type(exc).__name__}")
        return result
    if not lines:
        result["anomalies"].append("EMPTY_HEADER")
        return result
    fields = lines[0].split()
    if len(fields) < 4:
        result["anomalies"].append("MALFORMED_RECORD_LINE")
        return result
    try:
        declared_record = fields[0].split("/")[-1]
        signal_count = int(fields[1])
        fs = float(fields[2].split("/")[0])
        sample_count = int(fields[3])
        result.update(signal_count=signal_count, sampling_frequency=fs, sample_count=sample_count)
        result["duration_sec"] = sample_count / fs if fs > 0 else None
        if declared_record != path.stem:
            result["anomalies"].append("RECORD_NAME_MISMATCH")
        signal_lines = lines[1 : 1 + signal_count]
        if len(signal_lines) != signal_count:
            result["anomalies"].append("SIGNAL_LINE_COUNT_MISMATCH")
        size_by_file: Counter[str] = Counter()
        for line in signal_lines:
            tokens = line.split()
            if len(tokens) < 2:
                result["anomalies"].append("MALFORMED_SIGNAL_LINE")
                continue
            filename, fmt_token = tokens[0], tokens[1]
            result["signal_files"].append(filename)
            result["lead_names"].append(tokens[-1])
            fmt_match = re.match(r"(\d+)", fmt_token)
            if not fmt_match:
                result["anomalies"].append("UNPARSEABLE_SIGNAL_FORMAT")
                continue
            bits = int(fmt_match.group(1))
            if bits % 8:
                result["anomalies"].append("NON_BYTE_ALIGNED_SIGNAL_FORMAT")
                continue
            size_by_file[filename] += sample_count * (bits // 8)
        result["signal_files"] = sorted(set(result["signal_files"]))
        result["expected_dat_sizes"] = dict(size_by_file)
    except (ValueError, ZeroDivisionError):
        result["anomalies"].append("UNPARSEABLE_RECORD_LINE")
    return result


def legacy_windows(duration_sec: float | None) -> int:
    if duration_sec is None:
        return 0
    return sum(1 for k in range(24) if 60 + 3600 * k + 120 <= duration_sec + 1e-9)


def full_windows(duration_sec: float | None) -> int:
    if duration_sec is None or duration_sec < 300:
        return 0
    return int(math.floor((duration_sec + 1e-9) / 300))


def relative_is_within(child: Path, parent: Path) -> bool:
    try:
        child.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def audit(music_root: Path) -> dict[str, Any]:
    root = music_root.resolve(strict=True)
    required_missing = [name for name in REQUIRED_FILES if not (root / name).is_file()]
    if required_missing:
        raise ValueError(f"MUSIC root is missing required files: {required_missing}")

    subjects = read_semicolon(root / "subject-info.csv")
    codes_text = read_source_text(root / "subject-info_codes.csv")
    definitions_text = read_source_text(root / "subject-info_definitions.csv")
    required_columns = {
        "Patient ID",
        "Follow-up period from enrollment (days)",
        "Exit of the study",
        "Cause of death",
        "Hig-resolution ECG available",
        "ECG rhythm ",
        "Holter available",
        "Holter  rhythm ",
    }
    columns = set(subjects[0]) if subjects else set()
    missing_columns = sorted(required_columns - columns)

    patient_ids = [row.get("Patient ID", "").strip() for row in subjects]
    patient_counts = Counter(patient_ids)
    duplicates = sorted(pid for pid, count in patient_counts.items() if pid and count > 1)
    blank_patient_rows = sum(not pid for pid in patient_ids)
    subject_set = {pid for pid in patient_ids if pid}

    manifest: dict[str, str] = {}
    malformed_manifest_lines: list[str] = []
    for raw_line in read_source_text(root / "SHA256SUMS.txt").splitlines():
        if not raw_line.strip():
            continue
        match = SHA_RE.match(raw_line)
        if not match:
            malformed_manifest_lines.append(raw_line)
            continue
        manifest[match.group(2).replace("\\", "/")] = match.group(1).lower()

    record_lines = [line.strip().replace("\\", "/") for line in read_source_text(root / "RECORDS").splitlines() if line.strip()]
    record_duplicates = sorted(key for key, count in Counter(record_lines).items() if count > 1)
    record_set = set(record_lines)

    headers: list[dict[str, Any]] = []
    for dirname, kind in (("Holter_ECG", "HOLTER"), ("High-resolution_ECG", "HIGH_RESOLUTION")):
        for path in sorted((root / dirname).rglob("*.hea")):
            parsed = parse_header(path, kind)
            parsed["relative_path"] = path.relative_to(root).as_posix()
            headers.append(parsed)

    header_paths = {item["relative_path"] for item in headers}
    header_record_keys = {str(Path(path).with_suffix("" )).replace("\\", "/") for path in header_paths}
    holter_headers = [item for item in headers if item["record_type"] == "HOLTER"]
    hr_headers = [item for item in headers if item["record_type"] == "HIGH_RESOLUTION"]
    holter_patients = {item["patient_id"] for item in holter_headers if item["patient_id"]}
    hr_patients = {item["patient_id"] for item in hr_headers if item["patient_id"]}

    meta_holter = {row["Patient ID"].strip() for row in subjects if integer_code(row.get("Holter available")) == 1}
    meta_hr = {row["Patient ID"].strip() for row in subjects if integer_code(row.get("Hig-resolution ECG available")) == 1}

    header_anomalies: list[dict[str, str]] = []
    dat_size_too_small: list[dict[str, Any]] = []
    dat_trailing_bytes: list[dict[str, Any]] = []
    declared_dat_missing: list[str] = []
    for item in headers:
        for code in item["anomalies"]:
            header_anomalies.append({"header": item["relative_path"], "reason": code})
        for filename, expected_size in item["expected_dat_sizes"].items():
            dat_path = item["path"].parent / filename
            relative = dat_path.relative_to(root).as_posix()
            if not dat_path.is_file():
                declared_dat_missing.append(relative)
            else:
                actual_size = dat_path.stat().st_size
                if actual_size < expected_size:
                    dat_size_too_small.append({"path": relative, "minimum_bytes_from_header": expected_size, "actual_bytes": actual_size})
                elif actual_size > expected_size:
                    dat_trailing_bytes.append({"path": relative, "minimum_bytes_from_header": expected_size, "actual_bytes": actual_size, "trailing_bytes": actual_size - expected_size})

    manifest_missing = sorted(path for path in manifest if not (root / Path(path)).is_file())
    unexpected_files: list[str] = []
    allowed_inventory = set(manifest) | {"SHA256SUMS.txt"}
    for name in REQUIRED_FILES:
        allowed_inventory.add(name)
    for path in root.rglob("*"):
        if path.is_file():
            rel = path.relative_to(root).as_posix()
            if rel not in allowed_inventory:
                unexpected_files.append(rel)

    small_hash_checked = 0
    small_hash_mismatches: list[dict[str, str]] = []
    for rel, expected_hash in manifest.items():
        if rel.lower().endswith(".dat"):
            continue
        path = root / Path(rel)
        if not path.is_file():
            continue
        small_hash_checked += 1
        actual_hash = sha256_file(path)
        if actual_hash != expected_hash:
            small_hash_mismatches.append({"path": rel, "expected": expected_hash, "actual": actual_hash})

    dat_manifest_paths = {rel for rel in manifest if rel.lower().endswith(".dat")}
    dat_declared_paths = {
        (item["path"].parent / filename).relative_to(root).as_posix()
        for item in headers
        for filename in item["signal_files"]
        if filename.lower().endswith(".dat")
    }

    unmatched_header_patients = sorted((holter_patients | hr_patients) - subject_set)
    unmatched_subjects = sorted(subject_set - (holter_patients | hr_patients))
    records_without_header = sorted(record_set - header_record_keys)
    headers_not_in_records = sorted(header_record_keys - record_set)

    fs_distribution = Counter(str(int(item["sampling_frequency"])) if item["sampling_frequency"] is not None and float(item["sampling_frequency"]).is_integer() else str(item["sampling_frequency"]) for item in headers)
    channel_distribution = Counter(str(item["signal_count"]) for item in headers)
    lead_distribution = Counter("|".join(item["lead_names"]) for item in headers)
    by_type: dict[str, Any] = {}
    for kind, subset in (("HOLTER", holter_headers), ("HIGH_RESOLUTION", hr_headers)):
        by_type[kind] = {
            "record_count": len(subset),
            "patient_count": len({item["patient_id"] for item in subset if item["patient_id"]}),
            "sampling_frequency_distribution": dict(sorted(Counter(str(int(item["sampling_frequency"])) if item["sampling_frequency"] is not None and float(item["sampling_frequency"]).is_integer() else str(item["sampling_frequency"]) for item in subset).items())),
            "channel_count_distribution": dict(sorted(Counter(str(item["signal_count"]) for item in subset).items())),
            "lead_combination_distribution": dict(sorted(Counter("|".join(item["lead_names"]) for item in subset).items())),
            "duration_sec": five_number([item["duration_sec"] for item in subset if item["duration_sec"] is not None]),
        }

    followups = [value for row in subjects if (value := number(row.get("Follow-up period from enrollment (days)"))) is not None]
    cause_counts = Counter(str(integer_code(row.get("Cause of death"))) if integer_code(row.get("Cause of death")) is not None else "MISSING_OR_NONINTEGER" for row in subjects)
    scd_rows = [row for row in subjects if integer_code(row.get("Cause of death")) == 3]
    horizons = {str(days): sum((follow := number(row.get("Follow-up period from enrollment (days)"))) is not None and follow <= days for row in scd_rows) for days in (90, 180, 365, 730)}

    hr_af = {row["Patient ID"].strip() for row in subjects if integer_code(row.get("ECG rhythm ")) == 1}
    holter_af = {row["Patient ID"].strip() for row in subjects if integer_code(row.get("Holter  rhythm ")) == 1}
    pvc_columns = [
        "Number of ventricular premature beats in 24h",
        "Ventricular Extrasystole",
        "Number of ventricular premature contractions per hour",
    ]
    pvc: dict[str, Any] = {}
    for column in pvc_columns:
        values = [number(row.get(column)) for row in subjects]
        numeric = [value for value in values if value is not None]
        pvc[column] = {
            "present": column in columns,
            "non_missing_count": len(numeric),
            "missing_count": len(subjects) - len(numeric),
            "summary": five_number(numeric),
        }

    legacy_count = sum(legacy_windows(item["duration_sec"]) for item in holter_headers)
    full_count = sum(full_windows(item["duration_sec"]) for item in holter_headers)

    exact_metadata_names = ["subject-info.csv", "subject-info_codes.csv", "subject-info_definitions.csv", "RECORDS", "SHA256SUMS.txt", "LICENSE.txt"]
    compressed_metadata_bytes = sum(len(gzip.compress((root / name).read_bytes(), compresslevel=9, mtime=0)) for name in exact_metadata_names)
    header_raw = b"".join(item["path"].read_bytes() for item in headers)
    compressed_header_bytes = len(gzip.compress(header_raw, compresslevel=9, mtime=0))
    window_rows = legacy_count + full_count
    feature_bytes = window_rows * 20 * 8
    window_metadata_bytes = window_rows * 48
    fixed_allowance = compressed_metadata_bytes + compressed_header_bytes + 5 * 1024 * 1024
    estimate_low = int(fixed_allowance + feature_bytes * 0.72 + window_metadata_bytes * 0.20)
    estimate_high = int(fixed_allowance + feature_bytes * 1.05 + window_metadata_bytes * 0.75)

    explicit_version_in_small_files = any(VERSION in read_source_text(root / name) for name in exact_metadata_names)
    explicit_doi_in_small_files = any(DOI in read_source_text(root / name) for name in exact_metadata_names)
    version_name_match = root.name == EXPECTED_ROOT_NAME
    codes_readable = "3: SCD" in codes_text and "Cause of death" in codes_text
    definitions_readable = bool(definitions_text.strip())

    hard_gate_reasons: list[str] = []
    if not version_name_match:
        hard_gate_reasons.append("ROOT_NAME_DOES_NOT_IDENTIFY_MUSIC_1.0.1")
    if required_missing or missing_columns:
        hard_gate_reasons.append("REQUIRED_METADATA_MISSING")
    if duplicates or blank_patient_rows:
        hard_gate_reasons.append("PATIENT_IDENTITY_AMBIGUITY")
    if unmatched_header_patients or records_without_header or headers_not_in_records:
        hard_gate_reasons.append("RECORD_MAPPING_FAILURE")
    if not codes_readable or not definitions_readable:
        hard_gate_reasons.append("KEY_CODE_DEFINITIONS_UNREADABLE")
    if small_hash_mismatches or manifest_missing or malformed_manifest_lines:
        hard_gate_reasons.append("SOURCE_CHECKSUM_OR_MANIFEST_FAILURE")
    if header_anomalies or declared_dat_missing or dat_size_too_small:
        hard_gate_reasons.append("HEADER_OR_DECLARED_SIGNAL_ANOMALY")

    generated = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    return {
        "audit": {
            "phase": "PHASE 1 - DATA AUDIT",
            "generated_at_utc": generated,
            "read_scope": "official metadata, codes, definitions, RECORDS, SHA256SUMS, LICENSE, and WFDB .hea headers only",
            "waveform_content_read": False,
            "feature_extraction_performed": False,
            "python_version": platform.python_version(),
        },
        "source": {
            "dataset_name": "MUSIC (Sudden Cardiac Death in Chronic Heart Failure)",
            "music_version": VERSION,
            "source_doi": DOI,
            "root_name": root.name,
            "root_name_matches_locked_release": version_name_match,
            "version_explicit_in_local_small_files": explicit_version_in_small_files,
            "doi_explicit_in_local_small_files": explicit_doi_in_small_files,
            "confirmation": "confirmed_from_locked_release_directory_name_plus_complete_expected_layout_and_official_manifest" if version_name_match else "not_confirmed",
        },
        "cohort": {
            "official_subject_rows": len(subjects),
            "unique_patient_ids": len(subject_set),
            "duplicate_patient_ids": duplicates,
            "blank_patient_id_rows": blank_patient_rows,
            "subjects_without_any_ecg_header": unmatched_subjects,
        },
        "records": {
            "records_file_entries": len(record_lines),
            "records_file_duplicate_entries": record_duplicates,
            "holter_record_count": len(holter_headers),
            "holter_patient_count": len(holter_patients),
            "high_resolution_record_count": len(hr_headers),
            "high_resolution_patient_count": len(hr_patients),
            "metadata_holter_available_count": len(meta_holter),
            "metadata_high_resolution_available_count": len(meta_hr),
            "metadata_holter_yes_but_header_missing": sorted(meta_holter - holter_patients),
            "holter_header_present_but_metadata_not_yes": sorted(holter_patients - meta_holter),
            "metadata_high_resolution_yes_but_header_missing": sorted(meta_hr - hr_patients),
            "high_resolution_header_present_but_metadata_not_yes": sorted(hr_patients - meta_hr),
            "unmatched_header_patient_ids": unmatched_header_patients,
            "records_without_header": records_without_header,
            "headers_not_in_records": headers_not_in_records,
            "distributions_all_records": {
                "sampling_frequency_hz": dict(sorted(fs_distribution.items())),
                "channel_count": dict(sorted(channel_distribution.items())),
                "lead_combinations": dict(sorted(lead_distribution.items())),
                "duration_sec": five_number([item["duration_sec"] for item in headers if item["duration_sec"] is not None]),
            },
            "by_type": by_type,
        },
        "integrity": {
            "manifest_entry_count": len(manifest),
            "small_file_and_header_hashes_checked": small_hash_checked,
            "small_file_and_header_hash_mismatches": small_hash_mismatches,
            "dat_manifest_entry_count": len(dat_manifest_paths),
            "dat_content_hashes_checked": 0,
            "dat_content_hash_status": "NOT_CHECKED_IN_PHASE_1_BY_DESIGN",
            "dat_declared_but_not_in_manifest": sorted(dat_declared_paths - dat_manifest_paths),
            "dat_in_manifest_but_not_declared_by_headers": sorted(dat_manifest_paths - dat_declared_paths),
            "manifest_files_missing": manifest_missing,
            "declared_dat_files_missing": sorted(declared_dat_missing),
            "dat_files_shorter_than_header_requirement": dat_size_too_small,
            "dat_files_with_trailing_bytes": dat_trailing_bytes,
            "header_anomalies": header_anomalies,
            "malformed_manifest_lines": malformed_manifest_lines,
            "unexpected_files_not_in_manifest": sorted(unexpected_files),
        },
        "outcomes": {
            "follow_up_days": five_number(followups),
            "cause_of_death_distribution_raw_codes": dict(sorted(cause_counts.items())),
            "scd_definition": "Cause of death == 3, per subject-info_codes.csv",
            "scd_total": len(scd_rows),
            "scd_by_horizon_days": horizons,
        },
        "rhythm_and_pvc": {
            "af_definition": "official permanent AF code 1 in ECG rhythm and/or Holter rhythm",
            "high_resolution_permanent_af_count": len(hr_af),
            "holter_permanent_af_count": len(holter_af),
            "permanent_af_unique_union_count": len(hr_af | holter_af),
            "permanent_af_unique_union_patient_ids": sorted(hr_af | holter_af),
            "pvc_fields": pvc,
        },
        "windows": {
            "legacy_120s_definition": "start=60+3600*k seconds, k=0..23; valid iff start+120<=record duration",
            "legacy_120s_theoretical_window_count": legacy_count,
            "full_5min_definition": "300-second non-overlapping windows, stride 300 seconds, from t=0",
            "full_5min_theoretical_window_count": full_count,
        },
        "size_estimate": {
            "official_metadata_gzip_bytes": compressed_metadata_bytes,
            "wfdb_headers_combined_gzip_bytes": compressed_header_bytes,
            "assumed_feature_count": 20,
            "assumed_feature_dtype": "float64",
            "estimated_music_ci_low_bytes": estimate_low,
            "estimated_music_ci_high_bytes": estimate_high,
            "estimated_music_ci_low_mib": round(estimate_low / 1024**2, 2),
            "estimated_music_ci_high_mib": round(estimate_high / 1024**2, 2),
            "can_likely_control_within_80_mib": estimate_high <= 80 * 1024**2,
            "method": "20 float64 features per theoretical window; Parquet feature compression factor 0.72-1.05, encoded window metadata factor 0.20-0.75, plus 5 MiB cohort/validation/report allowance; must be replaced by measured Phase 4 size",
        },
        "existing_repository": {
            "legacy_derived_csv_rows": 88,
            "legacy_derived_csv_is_complete_cohort": False,
            "risk": "existing extractor silently skips extraction exceptions and patients with fewer than four windows; it is not used as Phase 1 truth",
        },
        "hard_gate": {
            "passed": not hard_gate_reasons,
            "reasons": hard_gate_reasons,
            "phase_2_authorized": False,
        },
    }


def md_stats(stats: dict[str, Any]) -> str:
    return " / ".join(str(stats[key]) for key in ("min", "p25", "median", "p75", "max"))


def render_markdown(report: dict[str, Any]) -> str:
    source, cohort, records = report["source"], report["cohort"], report["records"]
    integrity, outcomes = report["integrity"], report["outcomes"]
    rhythm, windows, size, gate = report["rhythm_and_pvc"], report["windows"], report["size_estimate"], report["hard_gate"]
    mismatches = sum(len(records[key]) for key in (
        "metadata_holter_yes_but_header_missing", "holter_header_present_but_metadata_not_yes",
        "metadata_high_resolution_yes_but_header_missing", "high_resolution_header_present_but_metadata_not_yes"))
    missing_records = len(integrity["manifest_files_missing"]) + len(integrity["declared_dat_files_missing"])
    lines = [
        "# MUSIC Phase 1 — DATA AUDIT", "",
        f"Generated: `{report['audit']['generated_at_utc']}`", "",
        "> Scope: official metadata, coding/definition files, manifests, filesystem metadata, and WFDB headers only. No `.dat` waveform content was opened, no features were extracted, and Phase 2 was not started.", "",
        "## Summary", "",
        "| Item | Result |", "|---|---:|",
        f"| MUSIC version | {source['music_version']} ({source['confirmation']}) |",
        f"| Source DOI | {source['source_doi']} |",
        f"| Official patient rows | {cohort['official_subject_rows']} |",
        f"| Unique Patient IDs | {cohort['unique_patient_ids']} |",
        f"| Holter records / patients | {records['holter_record_count']} / {records['holter_patient_count']} |",
        f"| High-resolution records / patients | {records['high_resolution_record_count']} / {records['high_resolution_patient_count']} |",
        f"| Metadata/file availability mismatches | {mismatches} |",
        f"| Missing records/files | {missing_records} |",
        f"| Header anomalies | {len(integrity['header_anomalies'])} |",
        f"| Small-file/header SHA256 mismatches | {len(integrity['small_file_and_header_hash_mismatches'])} |",
        f"| SCD total | {outcomes['scd_total']} |",
        f"| SCD <=90 / <=180 / <=365 / <=730 days | {outcomes['scd_by_horizon_days']['90']} / {outcomes['scd_by_horizon_days']['180']} / {outcomes['scd_by_horizon_days']['365']} / {outcomes['scd_by_horizon_days']['730']} |",
        f"| Permanent AF (unique union) | {rhythm['permanent_af_unique_union_count']} |",
        f"| Estimated legacy 120s windows | {windows['legacy_120s_theoretical_window_count']} |",
        f"| Estimated full 5min windows | {windows['full_5min_theoretical_window_count']} |",
        f"| Estimated MUSIC-CI size | {size['estimated_music_ci_low_mib']}–{size['estimated_music_ci_high_mib']} MiB |",
        f"| Likely <=80 MiB | {size['can_likely_control_within_80_mib']} |", "",
        "## Record distributions", "",
        f"- Sampling frequency (all): `{json.dumps(records['distributions_all_records']['sampling_frequency_hz'], ensure_ascii=False)}`",
        f"- Channel count (all): `{json.dumps(records['distributions_all_records']['channel_count'], ensure_ascii=False)}`",
        f"- Lead combinations (all): `{json.dumps(records['distributions_all_records']['lead_combinations'], ensure_ascii=False)}`",
        f"- Holter duration seconds, min / P25 / median / P75 / max: `{md_stats(records['by_type']['HOLTER']['duration_sec'])}`",
        f"- High-resolution duration seconds, min / P25 / median / P75 / max: `{md_stats(records['by_type']['HIGH_RESOLUTION']['duration_sec'])}`",
        f"- Follow-up days, min / P25 / median / P75 / max: `{md_stats(outcomes['follow_up_days'])}`", "",
        "## Mapping and integrity", "",
        f"- Metadata Holter yes but header missing: `{records['metadata_holter_yes_but_header_missing']}`",
        f"- Holter header present but metadata not yes: `{records['holter_header_present_but_metadata_not_yes']}`",
        f"- Metadata high-resolution yes but header missing: `{records['metadata_high_resolution_yes_but_header_missing']}`",
        f"- High-resolution header present but metadata not yes: `{records['high_resolution_header_present_but_metadata_not_yes']}`",
        f"- Headers with patient absent from metadata: `{records['unmatched_header_patient_ids']}`",
        f"- Records without header: `{records['records_without_header']}`",
        f"- Headers absent from RECORDS: `{records['headers_not_in_records']}`",
        f"- Header anomalies: `{integrity['header_anomalies']}`",
        f"- Manifest-declared missing files: `{integrity['manifest_files_missing']}`",
        f"- Declared signal files missing: `{integrity['declared_dat_files_missing']}`",
        f"- `.dat` files shorter than the header requirement: `{integrity['dat_files_shorter_than_header_requirement']}`",
        f"- `.dat` files with bytes beyond the header-declared minimum (reported, not treated as corruption): `{integrity['dat_files_with_trailing_bytes']}`",
        f"- SHA256 checked for {integrity['small_file_and_header_hashes_checked']} metadata/header files; mismatches: `{integrity['small_file_and_header_hash_mismatches']}`",
        f"- `.dat` content SHA256: **{integrity['dat_content_hash_status']}** ({integrity['dat_content_hashes_checked']} checked). Official hashes remain recorded in `SHA256SUMS.txt`.", "",
        "## Outcomes, rhythm, and PVC", "",
        f"- Cause-of-death code distribution: `{json.dumps(outcomes['cause_of_death_distribution_raw_codes'], ensure_ascii=False)}`",
        f"- Endpoint audit definition: {outcomes['scd_definition']}.",
        f"- Permanent AF: high-resolution={rhythm['high_resolution_permanent_af_count']}, Holter={rhythm['holter_permanent_af_count']}, unique union={rhythm['permanent_af_unique_union_count']}.",
        f"- PVC field availability: `{json.dumps(rhythm['pvc_fields'], ensure_ascii=False)}`", "",
        "## Size estimate", "",
        f"Official metadata compressed independently with deterministic gzip: **{size['official_metadata_gzip_bytes']} bytes**. Combined headers gzip size: **{size['wfdb_headers_combined_gzip_bytes']} bytes**.", "",
        f"Estimated compact package: **{size['estimated_music_ci_low_mib']}–{size['estimated_music_ci_high_mib']} MiB**. {size['method']}", "",
        "## Existing code finding", "",
        "The existing `scd_dataset.csv` contains 88 patients and is not the official full cohort. Existing extraction code can silently skip failed patients/windows; it is therefore not used as audit truth.", "",
        "## Hard gate", "",
        f"**{'PASSED' if gate['passed'] else 'FAILED'}**. Reasons: `{gate['reasons']}`", "",
        "Phase 2 was not started and is not authorized by this report.", "",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--music-root", type=Path, help="Exact MUSIC v1.0.1 root; otherwise MUSIC_RAW_DIR is required")
    parser.add_argument("--output-dir", type=Path, default=Path("reports"))
    args = parser.parse_args(argv)
    root_value = args.music_root or (Path(os.environ["MUSIC_RAW_DIR"]) if os.environ.get("MUSIC_RAW_DIR") else None)
    if root_value is None:
        parser.error("MUSIC_RAW_DIR is not defined and --music-root was not supplied")
    try:
        root = root_value.resolve(strict=True)
        output = args.output_dir.resolve()
        if relative_is_within(output, root):
            raise ValueError("output directory must not be inside MUSIC_RAW_DIR")
        report = audit(root)
        output.mkdir(parents=True, exist_ok=True)
        (output / "DATA_AUDIT.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        (output / "DATA_AUDIT.md").write_text(render_markdown(report), encoding="utf-8")
    except (OSError, ValueError, csv.Error) as exc:
        print(f"DATA AUDIT ERROR: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({
        "report": str(output / "DATA_AUDIT.json"),
        "hard_gate_passed": report["hard_gate"]["passed"],
        "hard_gate_reasons": report["hard_gate"]["reasons"],
    }, ensure_ascii=False))
    return 0 if report["hard_gate"]["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
