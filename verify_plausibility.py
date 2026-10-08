#!/usr/bin/env python3
"""Audit a rebuilt MUSIC cohort and ECG feature package against a reference.

The report separates deterministic data-contract checks from descriptive
clinical metadata. It does not encode clinical plausibility cutoffs.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


FEATURES = (
    "sig_mean", "sig_std", "sig_p2p", "sig_skew", "sig_kurt", "beats",
    "beats_per_min", "mean_rr", "sdnn", "rmssd", "pnn50", "mean_hr",
    "rr_cv", "rr_sampen", "rr_apen", "rr_dfa_alpha", "pow_lf", "pow_mf",
    "pow_hf", "pow_hf_ratio",
)
HORIZONS = (90, 180, 365, 730)
OFFICIAL_COUNTS = {
    "subjects": 992,
    "holter_records": 936,
    "high_resolution_records": 687,
    "scd_cause_code_3": 94,
    "pump_failure_cause_codes_6_or_7": 111,
    "non_cardiac_death_cause_code_1": 61,
    "survivor_exit_code_0": 695,
    "transplant_exit_code_2": 20,
    "lost_to_followup_exit_code_1": 11,
    "death_exit_code_3": 266,
}
PHYSIONET_URL = "https://physionet.org/content/music-sudden-cardiac-death/1.0.1/"


def _json_default(value: Any) -> Any:
    try:
        import numpy as np
        import pandas as pd

        if isinstance(value, (np.integer,)):
            return int(value)
        if isinstance(value, (np.floating,)):
            return None if not np.isfinite(value) else float(value)
        if isinstance(value, (np.bool_,)):
            return bool(value)
        if isinstance(value, (pd.Timestamp,)):
            return value.isoformat()
        if pd.isna(value):
            return None
    except (ImportError, TypeError, ValueError):
        pass
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"not JSON serializable: {type(value).__name__}")


class Audit:
    def __init__(self) -> None:
        self.failures: list[dict[str, Any]] = []
        self.observations: dict[str, Any] = {}
        self.warnings: list[str] = []
        self.unverified: list[str] = []
        self.artifacts: list[str] = []

    def fail(self, check: str, count: int, **details: Any) -> None:
        if int(count) > 0:
            self.failures.append({"check": check, "count": int(count), **details})

    def note(self, key: str, value: Any) -> None:
        self.observations[key] = value


def _resolve_root(path: Path) -> tuple[Path, Path]:
    root = path.expanduser().resolve()
    if (root / "data").is_dir():
        return root, root / "data"
    if root.name.lower() == "data" and root.is_dir():
        return root.parent, root
    return root, root / "data"


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _build_state(root: Path, data: Path) -> tuple[str | None, list[str]]:
    candidates = (
        data / "integrity" / "build_manifest.json",
        root / "reports" / "FULL_COHORT_BUILD.json",
    )
    states: list[str] = []
    found = False
    for path in candidates:
        document = _load_json(path)
        if document is None:
            continue
        for key in ("build_status", "status"):
            value = document.get(key)
            if isinstance(value, str) and value.strip():
                states.append(value.strip().upper())
                found = True
    if not found:
        return None, [str(path) for path in candidates]
    # Both COMPLETE (build report) and FINALIZED (integrity manifest) mean a
    # completed package. Explicit non-complete markers take precedence.
    bad = [state for state in states if state not in {"COMPLETE", "FINALIZED"}]
    if bad:
        return ", ".join(states), [f"explicit incomplete build marker: {', '.join(bad)}"]
    return ", ".join(states), []


def _read_parquet(path: Path, *, columns: list[str] | None = None):
    import pandas as pd

    if columns is None:
        return pd.read_parquet(path, engine="pyarrow")
    return pd.read_parquet(path, columns=columns, engine="pyarrow")


def _read_required(path: Path, name: str):
    if not path.is_file():
        raise FileNotFoundError(name)
    return _read_parquet(path)


def _read_shards(base: Path, manifest: dict[str, Any] | None, manifest_key: str, fallback: Path):
    import pandas as pd

    entries = manifest.get(manifest_key, []) if manifest else []
    paths: list[Path] = []
    if isinstance(entries, list) and entries:
        for item in entries:
            rel = item.get("path") if isinstance(item, dict) else item
            if isinstance(rel, str):
                paths.append(base / rel)
    if not paths:
        if fallback.is_file():
            paths = [fallback]
        elif fallback.is_dir():
            paths = sorted(fallback.glob("part-*.parquet"))
        else:
            flat = base / f"{fallback.name}.parquet"
            if flat.is_file():
                paths = [flat]
    frames = []
    absent = []
    for path in paths:
        if not path.is_file():
            absent.append(path.name)
            continue
        frames.append(_read_parquet(path))
    frame = pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()
    expected_rows = None
    if isinstance(entries, list) and entries and all(isinstance(item, dict) and isinstance(item.get("rows"), int) for item in entries):
        expected_rows = int(sum(item["rows"] for item in entries))
    return frame, {
        "listed_shards": len(paths), "read_shards": len(frames), "missing_shards": absent,
        "manifest_expected_rows": expected_rows,
        "actual_rows": int(len(frame)),
        "row_count_difference": None if expected_rows is None else int(len(frame) - expected_rows),
    }


def _load_feature_package(data: Path) -> dict[str, Any]:
    import pandas as pd

    directory = data / "features" / "full_5min"
    manifest = _load_json(directory / "manifest.json")
    feature_dir = directory / "features"
    window_dir = directory / "windows"
    raw_features, feature_manifest = _read_shards(
        directory, manifest, "feature_shards", feature_dir
    )
    windows, window_manifest = _read_shards(
        directory, manifest, "window_shards", window_dir
    )
    patient_path = directory / "patient_features.parquet"
    patient_features = _read_required(patient_path, "patient_features") if patient_path.is_file() else pd.DataFrame()

    # Some builds store all values in the window table; compact builds split
    # waveform features and QC/window metadata into manifest-listed shards.
    features = raw_features.copy()
    if features.empty and not windows.empty:
        features = windows.copy()
    join_warning = None
    if not features.empty and not windows.empty:
        keys = _common_window_keys(features, windows)
        if keys:
            extras = [c for c in windows.columns if c in {"patient_id", *keys}]
            extras += [c for c in windows.columns if c not in features.columns and c not in extras]
            window_metadata = windows.loc[:, extras].drop_duplicates(["patient_id", *keys], keep="first")
            features = features.merge(window_metadata, on=["patient_id", *keys], how="left")
        else:
            join_warning = "window feature and metadata shards have no shared window identity column"
    features = _normalize_feature_columns(features)
    windows = _normalize_feature_columns(windows)
    return {
        "directory": directory,
        "manifest": manifest,
        "features": features,
        "windows": windows,
        "patient_features": patient_features,
        "feature_shards": feature_manifest,
        "window_shards": window_manifest,
        "join_warning": join_warning,
    }


def _common_window_keys(left, right) -> list[str]:
    common = set(left.columns) & set(right.columns)
    for key in ("window_start_sec", "start_sec", "window_idx", "window_id"):
        if key in common:
            return [key]
    return []


def _normalize_feature_columns(frame):
    result = frame.copy()
    for index, name in enumerate(FEATURES, start=1):
        compact = f"feature_{index:02d}"
        if name not in result.columns and compact in result.columns:
            result[name] = result[compact]
        compact_valid = f"{compact}_valid"
        named_valid = f"{name}_valid"
        if named_valid not in result.columns and compact_valid in result.columns:
            result[named_valid] = result[compact_valid]
    return result


def _ids(frame, column: str = "patient_id"):
    if column not in frame.columns:
        return None
    return frame[column].astype("string").str.strip()


def _numeric(series):
    import numpy as np
    import pandas as pd

    if series is None:
        return pd.Series(dtype="float64")
    text = series.astype("string").str.strip()
    # The source metadata uses decimal commas in some releases. Preserve dot
    # decimals and interpret a lone comma as a decimal mark.
    text = text.str.replace(r"(?<=\d),(?=\d+$)", ".", regex=True)
    text = text.str.replace(",", "", regex=False)
    values = pd.to_numeric(text, errors="coerce").astype("float64")
    return values.where(np.isfinite(values))


def _summary(series) -> dict[str, Any]:
    values = _numeric(series)
    finite = values.dropna()
    if finite.empty:
        return {"n_numeric": 0, "n_missing_or_unparsed": int(len(values))}
    return {
        "n_numeric": int(len(finite)),
        "n_missing_or_unparsed": int(len(values) - len(finite)),
        "min": float(finite.min()),
        "p25": float(finite.quantile(0.25)),
        "median": float(finite.median()),
        "p75": float(finite.quantile(0.75)),
        "max": float(finite.max()),
    }


def _value_counts(series) -> dict[str, int]:
    if series is None:
        return {}
    cleaned = series.astype("string").fillna("<MISSING>").str.strip()
    counts = cleaned.value_counts(dropna=False)
    return {str(key): int(value) for key, value in counts.items()}


def _normalize_code(value: Any) -> str | None:
    if value is None:
        return None
    try:
        import pandas as pd

        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    text = str(value).strip()
    if not text:
        return None
    try:
        value_float = float(text.replace(",", "."))
        return str(int(value_float)) if value_float.is_integer() else text
    except ValueError:
        return text


def _code_counts(series) -> dict[str, int]:
    """Count source codes after normalizing integer-like CSV/Parquet values."""
    counts: Counter[str] = Counter()
    if series is None:
        return {}
    for value in series:
        code = _normalize_code(value)
        counts[code if code is not None else "<MISSING>"] += 1
    return dict(sorted(counts.items()))


def _metadata_and_official_counts(subjects, records, audit: Audit) -> None:
    def count_bool(frame, column: str) -> int | None:
        if column not in frame.columns:
            return None
        return int(frame[column].fillna(False).astype(bool).sum())

    record_type = records.get("record_type")
    record_counts = _value_counts(record_type)
    observed = {
        "subjects": int(len(subjects)),
        "holter_records": int(record_counts.get("HOLTER", 0)),
        "high_resolution_records": int(record_counts.get("HIGH_RESOLUTION", 0)),
    }
    cause_col = subjects.get("cause_of_death_raw", subjects.get("Cause of death"))
    exit_col = subjects.get("exit_status_raw", subjects.get("Exit of the study"))
    cause_codes = [_normalize_code(value) for value in cause_col] if cause_col is not None else []
    exit_codes = [_normalize_code(value) for value in exit_col] if exit_col is not None else []
    # In the official CSV, the 695 cause-code-0 survivors have a blank Exit
    # field. Keep that raw missingness visible, but count the documented
    # survivor category as effective exit code 0 (matching build_phase2.py's
    # event_source_valid rule that permits blank exit only for cause 0).
    effective_exit_codes = [
        "0" if exit_code is None and cause_code == "0" else exit_code
        for cause_code, exit_code in zip(cause_codes, exit_codes)
    ]
    observed.update({
        "scd_cause_code_3": sum(code == "3" for code in cause_codes),
        "pump_failure_cause_codes_6_or_7": sum(code in {"6", "7"} for code in cause_codes),
        "non_cardiac_death_cause_code_1": sum(code == "1" for code in cause_codes),
        "survivor_exit_code_0": sum(code == "0" for code in effective_exit_codes),
        "transplant_exit_code_2": sum(code == "2" for code in effective_exit_codes),
        "lost_to_followup_exit_code_1": sum(code == "1" for code in effective_exit_codes),
        "death_exit_code_3": sum(code == "3" for code in effective_exit_codes),
    })
    comparisons = {
        key: {"expected": expected, "observed": observed.get(key), "difference": None if observed.get(key) is None else observed[key] - expected}
        for key, expected in OFFICIAL_COUNTS.items()
    }
    audit.note("official_dataset_count_comparison", {
        "source": PHYSIONET_URL,
        "note": "Reference tallies are descriptive; differences are not automatically treated as data errors.",
        "comparisons": comparisons,
        "cause_code_counts": _code_counts(cause_col),
        "exit_code_counts": _code_counts(exit_col),
        "effective_exit_outcome_counts": _code_counts(effective_exit_codes),
        "exit_code_count_note": "Raw blank exit values remain <MISSING>; blank plus cause code 0 is counted as the official survivor outcome for the reference comparison.",
        "record_type_counts": record_counts,
        "has_holter_true": count_bool(subjects, "has_holter"),
        "has_high_resolution_ecg_true": count_bool(subjects, "has_high_resolution_ecg"),
    })

    metadata = {}
    for field in ("Age", "LVEF (%)", "followup_days", "Follow-up period from enrollment (days)", "holter_duration_sec"):
        if field in subjects.columns:
            metadata[field] = _summary(subjects[field])
    audit.note("metadata_ranges", metadata)


def _check_key_integrity(subjects, records, provenance, feature_package, status, survival, audit: Audit, prefix: str) -> None:
    import pandas as pd

    tables = {
        "subjects": (subjects, ["patient_id"]),
        "records_patient_type": (records, ["patient_id", "record_type"]),
        "provenance_patient_record_type": (provenance, ["patient_id", "record_id", "record_type"]),
        "patient_features": (feature_package["patient_features"], ["patient_id"]),
        "analysis_status": (status, ["patient_id"]),
        "survival_ready": (survival, ["patient_id"]),
    }
    ids = _ids(subjects)
    parent_set = set(ids.dropna().tolist()) if ids is not None else set()
    table_results: dict[str, Any] = {}
    for name, (frame, keys) in tables.items():
        if frame.empty:
            table_results[name] = {"rows": 0, "duplicate_key_rows": None, "missing_key_columns": keys}
            continue
        missing = [key for key in keys if key not in frame.columns]
        dupes = int(frame.duplicated(keys).sum()) if not missing else None
        table_results[name] = {"rows": int(len(frame)), "duplicate_key_rows": dupes, "missing_key_columns": missing}
        audit.fail(f"{prefix}.{name}.duplicate_keys", dupes or 0, rows=int(len(frame)), keys=keys)
        if "patient_id" in frame.columns and name != "subjects":
            child_ids = _ids(frame)
            orphan_rows = int((~child_ids.isin(parent_set)).sum())
            table_results[name]["orphan_patient_rows"] = orphan_rows
            audit.fail(f"{prefix}.{name}.orphan_patient_ids", orphan_rows, rows=int(len(frame)))
            if name in {"patient_features", "analysis_status", "survival_ready"}:
                child_set = set(child_ids.dropna().tolist())
                missing_subjects = len(parent_set - child_set)
                table_results[name]["missing_subject_rows"] = missing_subjects
                audit.fail(f"{prefix}.{name}.missing_subject_rows", missing_subjects)
    record_key_columns = ("patient_id", "record_id", "record_type")
    if all(column in records.columns for column in record_key_columns) and all(column in provenance.columns for column in record_key_columns):
        record_keys = set(zip(_ids(records).tolist(), records["record_id"].astype("string").tolist(), records["record_type"].astype("string").tolist()))
        provenance_keys = set(zip(_ids(provenance).tolist(), provenance["record_id"].astype("string").tolist(), provenance["record_type"].astype("string").tolist()))
        record_without_provenance = len(record_keys - provenance_keys)
        provenance_without_record = len(provenance_keys - record_keys)
        table_results["records_vs_provenance"] = {
            "records_without_provenance": record_without_provenance,
            "provenance_without_record": provenance_without_record,
        }
        audit.fail(f"{prefix}.records_without_provenance", record_without_provenance)
        audit.fail(f"{prefix}.provenance_without_record", provenance_without_record)
    features = feature_package["features"]
    keys = ["patient_id"]
    win_identity = next((key for key in ("window_start_sec", "start_sec", "window_idx", "window_id") if key in features.columns), None)
    if win_identity:
        keys.append(win_identity)
    if not features.empty:
        missing = [key for key in keys if key not in features.columns]
        duplicate_rows = int(features.duplicated(keys).sum()) if not missing else 0
        feature_ids = _ids(features)
        orphan_rows = int((~feature_ids.isin(parent_set)).sum()) if feature_ids is not None else 0
        audit.fail(f"{prefix}.window_features.missing_key_columns", len(missing), columns=missing)
        audit.fail(f"{prefix}.window_features.duplicate_keys", duplicate_rows, keys=keys, rows=int(len(features)))
        audit.fail(f"{prefix}.window_features.orphan_patient_ids", orphan_rows, rows=int(len(features)))
        table_results["window_features"] = {
            "rows": int(len(features)), "identity_keys": keys,
            "duplicate_key_rows": duplicate_rows, "orphan_patient_rows": orphan_rows,
        }
    windows = feature_package["windows"]
    if not windows.empty and "patient_id" in windows.columns:
        window_identity = next((key for key in ("window_start_sec", "start_sec", "window_idx", "window_id") if key in windows.columns), None)
        if window_identity:
            window_keys = ["patient_id", window_identity]
            window_dupes = int(windows.duplicated(window_keys).sum())
            audit.fail(f"{prefix}.window_metadata.duplicate_keys", window_dupes, keys=window_keys, rows=int(len(windows)))
            if not features.empty and window_identity in features.columns:
                feature_key_frame = features.loc[:, window_keys].drop_duplicates()
                window_key_frame = windows.loc[:, window_keys].drop_duplicates()
                key_diff = feature_key_frame.merge(window_key_frame, on=window_keys, how="outer", indicator=True)
                feature_only = int(key_diff["_merge"].eq("left_only").sum())
                metadata_only = int(key_diff["_merge"].eq("right_only").sum())
                audit.fail(f"{prefix}.window_feature_metadata_key_equality", feature_only + metadata_only,
                           feature_only=feature_only, metadata_only=metadata_only)
                table_results["window_feature_metadata_keys"] = {
                    "feature_only_rows": feature_only, "metadata_only_rows": metadata_only,
                }
    audit.note(f"{prefix}.referential_integrity", table_results)


def _check_manifest_counts(package: dict[str, Any], audit: Audit, prefix: str) -> None:
    for table_name in ("feature_shards", "window_shards"):
        details = package.get(table_name, {})
        missing = len(details.get("missing_shards", []))
        row_diff = details.get("row_count_difference")
        if not details.get("read_shards", 0):
            audit.fail(f"{prefix}.{table_name}.no_readable_shards", 1)
        audit.fail(f"{prefix}.{table_name}.missing_manifest_shards", missing, missing_shards=details.get("missing_shards", []))
        audit.fail(f"{prefix}.{table_name}.manifest_row_count", abs(row_diff or 0), expected=details.get("manifest_expected_rows"), actual=details.get("actual_rows"))


def _compare_series(a, b, atol: float = 1e-12) -> int:
    import numpy as np
    import pandas as pd

    left = a.reset_index(drop=True)
    right = b.reset_index(drop=True)
    left_num = _numeric(left)
    right_num = _numeric(right)
    left_numeric = int(left_num.notna().sum())
    right_numeric = int(right_num.notna().sum())
    left_present = int(left.notna().sum())
    right_present = int(right.notna().sum())
    if left_numeric == left_present and right_numeric == right_present:
        lv = left_num.to_numpy(dtype="float64")
        rv = right_num.to_numpy(dtype="float64")
        both_missing = np.isnan(lv) & np.isnan(rv)
        close = np.isclose(lv, rv, rtol=0.0, atol=atol, equal_nan=True)
        return int((~(both_missing | close)).sum())
    ltext = left.astype("string").fillna("<NULL>").str.strip()
    rtext = right.astype("string").fillna("<NULL>").str.strip()
    return int(ltext.ne(rtext).sum())


def _compare_cohorts(fresh_subjects, reference_subjects, fresh_records, reference_records, audit: Audit) -> None:
    import pandas as pd

    result: dict[str, Any] = {}
    if "patient_id" not in fresh_subjects.columns or "patient_id" not in reference_subjects.columns:
        audit.unverified.append("cohort equality: patient_id column missing")
        return
    fresh_ids = set(_ids(fresh_subjects).dropna().tolist())
    ref_ids = set(_ids(reference_subjects).dropna().tolist())
    result["subjects"] = {
        "fresh_rows": len(fresh_subjects), "reference_rows": len(reference_subjects),
        "fresh_only_subject_rows": len(fresh_ids - ref_ids),
        "reference_only_subject_rows": len(ref_ids - fresh_ids),
    }
    audit.fail("cohort.subject_id_set_equality", len(fresh_ids - ref_ids) + len(ref_ids - fresh_ids),
               fresh_only=len(fresh_ids - ref_ids), reference_only=len(ref_ids - fresh_ids))
    shared_ids = sorted(fresh_ids & ref_ids)
    fresh_idx = fresh_subjects.assign(patient_id=_ids(fresh_subjects)).set_index("patient_id", drop=True)
    ref_idx = reference_subjects.assign(patient_id=_ids(reference_subjects)).set_index("patient_id", drop=True)
    shared_cols = sorted(set(fresh_subjects.columns) & set(reference_subjects.columns) - {"patient_id"})
    compared_cols = [col for col in shared_cols if not re.search(r"path|sha256|checksum|hash|provenance|source_file|header_file", col, re.I)]
    missing_fresh = sorted(set(reference_subjects.columns) - set(fresh_subjects.columns))
    missing_reference = sorted(set(fresh_subjects.columns) - set(reference_subjects.columns))
    mismatches: dict[str, int] = {}
    for column in compared_cols:
        count = _compare_series(fresh_idx.loc[shared_ids, column], ref_idx.loc[shared_ids, column])
        if count:
            mismatches[column] = count
    missing_comparable = [c for c in missing_reference if not re.search(r"path|sha256|checksum|hash|provenance|source_file|header_file", c, re.I)]
    extra_comparable = [c for c in missing_fresh if not re.search(r"path|sha256|checksum|hash|provenance|source_file|header_file", c, re.I)]
    result["subjects"].update({
        "shared_fields_compared": len(compared_cols),
        "shared_field_rows_mismatched": mismatches,
        "fields_only_in_fresh": missing_reference,
        "fields_only_in_reference": missing_fresh,
        "provenance_or_path_fields_excluded": sorted(set(shared_cols) - set(compared_cols)),
    })
    audit.fail("cohort.subject_field_equality", sum(mismatches.values()), fields=len(mismatches))
    audit.fail("cohort.subject_schema_equality", len(missing_comparable) + len(extra_comparable),
               fresh_only_fields=missing_comparable, reference_only_fields=extra_comparable)

    # Compare record membership and signal-defining metadata by subject/type;
    # path, checksum, and build-provenance fields are intentionally excluded.
    record_result: dict[str, Any] = {}
    keys = ["patient_id", "record_type"]
    if all(k in fresh_records.columns and k in reference_records.columns for k in keys):
        f = fresh_records.copy(); r = reference_records.copy()
        f["patient_id"] = _ids(f); r["patient_id"] = _ids(r)
        f = f.set_index(keys); r = r.set_index(keys)
        fi, ri = set(f.index.tolist()), set(r.index.tolist())
        shared = sorted(fi & ri)
        onlyf, onlyr = len(fi - ri), len(ri - fi)
        cols = sorted(set(f.columns) & set(r.columns))
        compared = [c for c in cols if c != "build_code_version" and not re.search(r"path|sha256|checksum|hash|provenance|source_file|header_file", c, re.I)]
        field_mismatch = {c: _compare_series(f.loc[shared, c], r.loc[shared, c]) for c in compared}
        field_mismatch = {c: n for c, n in field_mismatch.items() if n}
        build_version_observation = None
        if "build_code_version" in cols:
            build_version_observation = {
                "mismatched_rows": _compare_series(f.loc[shared, "build_code_version"], r.loc[shared, "build_code_version"]),
                "fresh_value_counts": _code_counts(fresh_records["build_code_version"]),
                "reference_value_counts": _code_counts(reference_records["build_code_version"]),
                "interpretation": "Build-code version differences are provenance observations and are not record-field failures.",
            }
        record_result = {
            "fresh_record_rows": len(fresh_records), "reference_record_rows": len(reference_records),
            "fresh_only_patient_type_rows": onlyf, "reference_only_patient_type_rows": onlyr,
            "shared_fields_compared": len(compared), "shared_field_rows_mismatched": field_mismatch,
            "path_or_provenance_fields_excluded": sorted(set(cols) - set(compared)),
            "build_code_version_provenance_observation": build_version_observation,
        }
        audit.fail("cohort.record_membership_equality", onlyf + onlyr, fresh_only=onlyf, reference_only=onlyr)
        audit.fail("cohort.record_field_equality", sum(field_mismatch.values()), fields=len(field_mismatch))
    else:
        audit.unverified.append("record equality: patient_id/record_type unavailable in one root")
    result["records"] = record_result

    # Endpoint source columns are a specific label contract and receive a
    # separate summary even when other derived cohort metadata differs.
    endpoint_cols = [c for c in ("followup_days", "followup_days_raw", "cause_of_death_raw", "event_source_valid", "cause_of_death_decoded", "exit_status_raw", "exit_status_decoded") if c in fresh_subjects.columns and c in reference_subjects.columns]
    label_mismatch = {c: _compare_series(fresh_idx.loc[shared_ids, c], ref_idx.loc[shared_ids, c]) for c in endpoint_cols}
    label_mismatch = {c: n for c, n in label_mismatch.items() if n}
    result["endpoint_source_label_fields"] = {"fields_compared": endpoint_cols, "mismatched_rows_by_field": label_mismatch}
    audit.fail("cohort.endpoint_source_labels", sum(label_mismatch.values()), fields=len(label_mismatch))
    audit.note("fresh_vs_reference_cohort", result)


def _check_endpoints(subjects, status, audit: Audit, prefix: str) -> None:
    import pandas as pd

    result: dict[str, Any] = {}
    needed = {"patient_id", "followup_days", "cause_of_death_raw", "event_source_valid"}
    missing = sorted(needed - set(subjects.columns))
    if missing:
        audit.unverified.append(f"{prefix} endpoint derivation: missing source columns {missing}")
        audit.note(f"{prefix}.endpoint_states", {"unverified": True, "missing_source_columns": missing})
        return
    if "patient_id" not in status.columns:
        audit.unverified.append(f"{prefix} endpoint states: analysis status table missing patient_id")
        return
    src = subjects.copy()
    src["patient_id"] = _ids(src)
    src["_followup"] = _numeric(src["followup_days"])
    src["_cause"] = src["cause_of_death_raw"].map(_normalize_code)
    src["_source_valid"] = src["event_source_valid"].fillna(False).astype(bool)
    expected_states: dict[int, list[str]] = {}
    for horizon in HORIZONS:
        states: list[str] = []
        for followup, cause, valid in zip(src["_followup"], src["_cause"], src["_source_valid"]):
            if not valid or pd.isna(followup) or followup < 0 or cause not in {"0", "1", "3", "6", "7"}:
                state = "UNKNOWN"
            elif cause == "3":
                state = "POSITIVE" if followup <= horizon else "NEGATIVE"
            elif cause in {"1", "6", "7"}:
                state = "COMPETING_EVENT" if followup <= horizon else "NEGATIVE"
            else:
                state = "NEGATIVE" if followup >= horizon else "CENSORED"
            states.append(state)
        expected_states[horizon] = states
    observed: dict[str, Any] = {}
    indexed_source = src.set_index("patient_id", drop=True)
    status_copy = status.copy()
    status_copy["patient_id"] = _ids(status_copy)
    status_index = status_copy.set_index("patient_id", drop=True)
    common = sorted(set(indexed_source.index) & set(status_index.index))
    for horizon in HORIZONS:
        expected_map = dict(zip(src["patient_id"].tolist(), expected_states[horizon]))
        state_col = f"endpoint_{horizon}_state"
        binary_col = f"endpoint_{horizon}_binary_label"
        if state_col not in status_index.columns:
            audit.unverified.append(f"{prefix} {horizon}-day state column missing")
            continue
        actual = status_index.loc[common, state_col].astype("string").fillna("<NULL>")
        expected = pd.Series([expected_map[x] for x in common], index=actual.index, dtype="string")
        state_mismatch = int(actual.ne(expected).sum())
        labels_expected = expected.map({"POSITIVE": 1, "NEGATIVE": 0}).astype("Int64")
        label_mismatch = None
        if binary_col in status_index.columns:
            labels_actual = pd.to_numeric(status_index.loc[common, binary_col], errors="coerce").astype("Int64")
            label_mismatch = int((labels_actual.fillna(-9) != labels_expected.fillna(-9)).sum())
        states = _value_counts(status_index[state_col])
        observed[str(horizon)] = {
            "state_counts": states,
            "expected_state_counts_from_source_codes": dict(Counter(expected_states[horizon])),
            "state_mismatched_rows": state_mismatch,
            "binary_label_mismatched_rows": label_mismatch,
        }
        audit.fail(f"{prefix}.endpoint_{horizon}.state_recomputation", state_mismatch, horizon_days=horizon)
        audit.fail(f"{prefix}.endpoint_{horizon}.label_recomputation", label_mismatch or 0, horizon_days=horizon)
    if "primary_sinus_hrv_reason" in status.columns:
        reason_counts = _value_counts(status["primary_sinus_hrv_reason"])
    else:
        reason_counts = {}
    eligible_n = int(status["primary_sinus_hrv_eligible"].fillna(False).astype(bool).sum()) if "primary_sinus_hrv_eligible" in status.columns else None
    if "model_365_included" in status.columns and "endpoint_365_state" in status.columns and "primary_sinus_hrv_eligible" in status.columns:
        state = status["endpoint_365_state"].astype("string").isin(["POSITIVE", "NEGATIVE"])
        eligible = status["primary_sinus_hrv_eligible"].fillna(False).astype(bool)
        expected_include = state & eligible
        include_mismatch = int(status["model_365_included"].fillna(False).astype(bool).ne(expected_include).sum())
        audit.fail(f"{prefix}.primary_365_inclusion_logic", include_mismatch)
    else:
        include_mismatch = None
    observed["primary_eligibility"] = {
        "eligible_count": eligible_n,
        "reason_counts": reason_counts,
        "365_day_model_included_count": int(status["model_365_included"].fillna(False).astype(bool).sum()) if "model_365_included" in status.columns else None,
        "inclusion_logic_mismatched_rows": include_mismatch,
    }
    audit.note(f"{prefix}.endpoint_states", observed)


def _feature_value_checks(frame, windows, audit: Audit, prefix: str) -> None:
    import numpy as np
    import pandas as pd

    if frame.empty:
        audit.unverified.append(f"{prefix} per-window feature table missing or empty")
        return
    names = [name for name in FEATURES if name in frame.columns]
    absent = sorted(set(FEATURES) - set(names))
    if absent:
        audit.unverified.append(f"{prefix} feature columns missing: {absent}")
    metrics: dict[str, Any] = {}
    values: dict[str, Any] = {}
    for name in names:
        numeric = _numeric(frame[name])
        finite = numeric.dropna()
        raw = pd.to_numeric(frame[name], errors="coerce")
        inf_n = int(np.isinf(raw.to_numpy(dtype="float64", na_value=np.nan)).sum())
        metrics[name] = {
            **_summary(frame[name]),
            "infinite_count": inf_n,
            "valid_flag_column": f"{name}_valid" if f"{name}_valid" in frame.columns else None,
        }
        values[name] = numeric.to_numpy(dtype="float64", na_value=np.nan)
        flag_col = f"{name}_valid"
        if flag_col in frame.columns:
            valid_flag = frame[flag_col].fillna(False).astype(bool).to_numpy()
            expected_flag = numeric.notna().to_numpy()
            mismatch = int(np.count_nonzero(valid_flag != expected_flag))
            metrics[name]["valid_flag_mismatched_rows"] = mismatch
            audit.fail(f"{prefix}.{name}.validity_flag", mismatch, feature=name)

    def check_nonnegative(name: str) -> None:
        if name in values:
            count = int(np.count_nonzero(np.isfinite(values[name]) & (values[name] < -1e-12)))
            audit.fail(f"{prefix}.{name}.nonnegative", count, feature=name)
            metrics[name]["negative_count"] = count

    for name in ("mean_rr", "sdnn", "rmssd", "rr_cv", "beats", "beats_per_min", "pow_lf", "pow_mf", "pow_hf", "pow_hf_ratio"):
        check_nonnegative(name)
    if "pnn50" in values:
        v = values["pnn50"]
        bad = int(np.count_nonzero(np.isfinite(v) & ((v < -1e-12) | (v > 100.0 + 1e-12))))
        metrics["pnn50"]["outside_0_100_count"] = bad
        audit.fail(f"{prefix}.pnn50.range_0_100", bad)
    for name in ("pow_lf", "pow_mf", "pow_hf"):
        if name in values:
            v = values[name]
            bad = int(np.count_nonzero(np.isfinite(v) & (v > 1.0 + 1e-9)))
            metrics[name]["above_one_count"] = bad
            audit.fail(f"{prefix}.{name}.normalized_power_upper_bound", bad, feature=name)

    rel: dict[str, Any] = {}
    if "beats" in values and "beats_per_min" in values:
        expected_bpm = values["beats"] * 60.0 / (300.0 + 1e-8)
        bpm = values["beats_per_min"]
        present = np.isfinite(expected_bpm) & np.isfinite(bpm)
        diff = np.abs(expected_bpm[present] - bpm[present])
        bad = int(np.count_nonzero(diff > 1e-7))
        rel["beats_per_min_equals_beats_over_5min"] = {
            "compared_rows": int(present.sum()), "mismatched_rows": bad,
            "max_abs_difference": float(diff.max()) if diff.size else None,
        }
        audit.fail(f"{prefix}.beats_per_min_relationship", bad)
    if "mean_rr" in values and "mean_hr" in values:
        rr, hr = values["mean_rr"], values["mean_hr"]
        expected_hr = np.divide(60.0, rr, out=np.full_like(rr, np.nan), where=np.isfinite(rr) & (rr > 0))
        present = np.isfinite(expected_hr) & np.isfinite(hr)
        diff = np.abs(expected_hr[present] - hr[present])
        bad = int(np.count_nonzero(diff > 1e-7))
        rel["mean_hr_equals_60_over_window_mean_rr"] = {
            "compared_rows": int(present.sum()), "mismatched_rows": bad,
            "max_abs_difference": float(diff.max()) if diff.size else None,
            "interpretation": "Window-level identity uses the same valid-RR mean; aggregated mean HR is not expected to equal 60 divided by aggregated mean RR.",
        }
        audit.fail(f"{prefix}.mean_hr_relationship", bad)

    if not windows.empty:
        failure_col = "failure_reason" if "failure_reason" in windows.columns else "window_status"
        if failure_col in windows.columns:
            qc_reasons = windows.get("qc_reason")
            qc_counts = Counter()
            if qc_reasons is not None:
                for item in qc_reasons.dropna().astype(str):
                    for reason in item.split(";"):
                        reason = reason.strip()
                        if reason:
                            qc_counts[reason] += 1
            failure_counts = _value_counts(windows[failure_col])
        else:
            qc_counts, failure_counts = Counter(), {}
    else:
        qc_counts, failure_counts = Counter(), {}
    window_contract: dict[str, Any] = {}
    if not windows.empty:
        for column in ("raw_rpeak_count", "raw_rr_count", "valid_rr_count", "removed_rr_count"):
            if column in windows.columns:
                negative = int((_numeric(windows[column]) < 0).sum())
                window_contract[f"{column}_negative_rows"] = negative
                audit.fail(f"{prefix}.{column}.nonnegative", negative)
        if {"raw_rr_count", "valid_rr_count", "removed_rr_count"}.issubset(windows.columns):
            raw = _numeric(windows["raw_rr_count"])
            valid = _numeric(windows["valid_rr_count"])
            removed = _numeric(windows["removed_rr_count"])
            finite = raw.notna() & valid.notna() & removed.notna()
            count_bad = int((finite & raw.ne(valid + removed)).sum())
            window_contract["raw_rr_equals_valid_plus_removed_mismatched_rows"] = count_bad
            audit.fail(f"{prefix}.rr_count_partition", count_bad)
        if "removed_rr_ratio" in windows.columns and {"raw_rr_count", "removed_rr_count"}.issubset(windows.columns):
            ratio = _numeric(windows["removed_rr_ratio"])
            raw = _numeric(windows["raw_rr_count"])
            removed = _numeric(windows["removed_rr_count"])
            expected = np.divide(removed.to_numpy(), raw.to_numpy(), out=np.zeros(len(raw), dtype=float), where=raw.to_numpy() > 0)
            actual = ratio.to_numpy(dtype="float64", na_value=np.nan)
            finite = np.isfinite(actual) & np.isfinite(expected)
            mismatch = int(np.count_nonzero(finite & (np.abs(actual - expected) > 1e-9)))
            outside = int(np.count_nonzero(np.isfinite(actual) & ((actual < -1e-12) | (actual > 1.0 + 1e-12))))
            window_contract["removed_rr_ratio_mismatched_rows"] = mismatch
            window_contract["removed_rr_ratio_outside_0_1_rows"] = outside
            audit.fail(f"{prefix}.removed_rr_ratio.relationship", mismatch)
            audit.fail(f"{prefix}.removed_rr_ratio.range_0_1", outside)
    if "sig_std" in values:
        v = values["sig_std"]
        finite = v[np.isfinite(v)]
        uniq = int(pd.Series(np.round(finite, 12)).nunique()) if finite.size else 0
        metrics["sig_std"]["unique_values_rounded_1e_12"] = uniq
        metrics["sig_std"]["near_constant_at_1e_12"] = bool(uniq <= 1 and finite.size > 0)
        metrics["sig_std"]["normalization_reference"] = "preprocessing scales each complete ECG window by its population standard deviation; expected to be near 1 by implementation, not a clinical threshold"

    audit.note(f"{prefix}.feature_validity_ranges", {
        "rows": int(len(frame)), "features": metrics, "relationship_checks": rel,
        "window_count_contract_checks": window_contract,
        "qc_reason_counts": dict(qc_counts), "failure_reason_or_status_counts": failure_counts,
        "spectral_power_bands_hz": {"pow_lf": "0.5–4", "pow_mf": "4–15", "pow_hf": "15–40"},
    })


def _compare_patient_features(fresh, reference, audit: Audit, atol: float = 1e-10, rtol: float = 1e-8) -> None:
    import numpy as np
    import pandas as pd

    if fresh.empty or reference.empty or "patient_id" not in fresh.columns or "patient_id" not in reference.columns:
        audit.unverified.append("per-window numeric equality: feature table missing or patient_id unavailable")
        return
    f = _normalize_feature_columns(fresh)
    r = _normalize_feature_columns(reference)
    key = next((candidate for candidate in ("window_start_sec", "start_sec", "window_idx", "window_id") if candidate in f.columns and candidate in r.columns), None)
    if key is None:
        audit.unverified.append("per-window numeric equality: no shared window identity column")
        return
    feature_names = [name for name in FEATURES if name in f.columns and name in r.columns]
    if not feature_names:
        audit.unverified.append("per-window numeric equality: no shared named feature columns")
        return
    f = f.loc[:, ["patient_id", key, *feature_names]].copy()
    r = r.loc[:, ["patient_id", key, *feature_names]].copy()
    f["patient_id"] = _ids(f); r["patient_id"] = _ids(r)
    f[key] = _numeric(f[key]); r[key] = _numeric(r[key])
    keys = ["patient_id", key]
    fdup, rdup = int(f.duplicated(keys).sum()), int(r.duplicated(keys).sum())
    if fdup or rdup:
        audit.fail("feature_equality.duplicate_keys_before_join", fdup + rdup, fresh=fdup, reference=rdup)
    joined = f.merge(r, on=keys, how="outer", suffixes=("_fresh", "_reference"), indicator=True, validate="one_to_one")
    fresh_only = int((joined["_merge"] == "left_only").sum())
    ref_only = int((joined["_merge"] == "right_only").sum())
    comparable = joined["_merge"].eq("both").to_numpy()
    per_feature: dict[str, Any] = {}
    total_mismatch = 0
    for name in feature_names:
        left = _numeric(joined[f"{name}_fresh"])
        right = _numeric(joined[f"{name}_reference"])
        lv = left.to_numpy(dtype="float64", na_value=np.nan)
        rv = right.to_numpy(dtype="float64", na_value=np.nan)
        both_null = np.isnan(lv) & np.isnan(rv)
        close = np.isclose(lv, rv, rtol=rtol, atol=atol, equal_nan=True)
        bad = comparable & ~(both_null | close)
        nbad = int(np.count_nonzero(bad))
        total_mismatch += nbad
        both_finite = comparable & np.isfinite(lv) & np.isfinite(rv)
        delta = np.abs(lv[both_finite] - rv[both_finite])
        per_feature[name] = {
            "mismatched_rows": nbad,
            "compared_rows": int(comparable.sum()),
            "max_abs_difference": float(delta.max()) if delta.size else None,
        }
    audit.fail("feature_equality.window_key_set", fresh_only + ref_only, fresh_only=fresh_only, reference_only=ref_only)
    audit.fail("feature_equality.numeric_values", total_mismatch, features=len(feature_names), atol=atol, rtol=rtol)
    audit.note("fresh_vs_reference_window_features", {
        "identity_keys": keys, "fresh_rows": len(f), "reference_rows": len(r),
        "matched_rows": int(comparable.sum()), "fresh_only_rows": fresh_only,
        "reference_only_rows": ref_only, "numeric_tolerance": {"atol": atol, "rtol": rtol},
        "feature_comparisons": per_feature,
    })


def _compare_patient_aggregates(fresh, reference, audit: Audit, atol: float = 1e-10, rtol: float = 1e-8) -> None:
    import numpy as np

    if fresh.empty or reference.empty or "patient_id" not in fresh.columns or "patient_id" not in reference.columns:
        audit.unverified.append("patient aggregate equality: patient feature table missing or patient_id unavailable")
        return
    feature_columns = [
        column for column in sorted(set(fresh.columns) & set(reference.columns))
        if any(column == f"{feature}_{suffix}" for feature in FEATURES for suffix in ("mean", "std", "p10", "p50", "p90"))
        or column in {f"{feature}_valid_count" for feature in FEATURES}
    ]
    if not feature_columns:
        audit.unverified.append("patient aggregate equality: no shared aggregated feature columns")
        return
    f = fresh.loc[:, ["patient_id", *feature_columns]].copy()
    r = reference.loc[:, ["patient_id", *feature_columns]].copy()
    f["patient_id"] = _ids(f); r["patient_id"] = _ids(r)
    fdup, rdup = int(f.duplicated(["patient_id"]).sum()), int(r.duplicated(["patient_id"]).sum())
    audit.fail("aggregate_equality.duplicate_patient_keys", fdup + rdup, fresh=fdup, reference=rdup)
    joined = f.merge(r, on="patient_id", how="outer", suffixes=("_fresh", "_reference"), indicator=True)
    fresh_only = int(joined["_merge"].eq("left_only").sum())
    reference_only = int(joined["_merge"].eq("right_only").sum())
    common = joined["_merge"].eq("both").to_numpy()
    comparisons: dict[str, Any] = {}
    total = 0
    for column in feature_columns:
        left = _numeric(joined[f"{column}_fresh"]).to_numpy(dtype="float64", na_value=np.nan)
        right = _numeric(joined[f"{column}_reference"]).to_numpy(dtype="float64", na_value=np.nan)
        both_null = np.isnan(left) & np.isnan(right)
        close = np.isclose(left, right, rtol=rtol, atol=atol, equal_nan=True)
        bad = common & ~(both_null | close)
        count = int(np.count_nonzero(bad))
        total += count
        finite = common & np.isfinite(left) & np.isfinite(right)
        delta = np.abs(left[finite] - right[finite])
        comparisons[column] = {"mismatched_rows": count, "max_abs_difference": float(delta.max()) if delta.size else None}
    audit.fail("aggregate_equality.patient_key_set", fresh_only + reference_only, fresh_only=fresh_only, reference_only=reference_only)
    audit.fail("aggregate_equality.numeric_values", total, columns=len(feature_columns), atol=atol, rtol=rtol)
    audit.note("fresh_vs_reference_patient_aggregates", {
        "fresh_rows": len(f), "reference_rows": len(r), "matched_patients": int(common.sum()),
        "fresh_only_patients": fresh_only, "reference_only_patients": reference_only,
        "numeric_tolerance": {"atol": atol, "rtol": rtol}, "feature_comparisons": comparisons,
    })


def _aggregate_feature_diagnostics(patient_features, audit: Audit, prefix: str) -> None:
    import numpy as np
    import pandas as pd

    frame = patient_features
    if frame.empty:
        audit.unverified.append(f"{prefix} patient aggregation missing")
        return
    result: dict[str, Any] = {}
    for name in FEATURES:
        count_col = f"{name}_valid_count"
        value_cols = [f"{name}_{suffix}" for suffix in ("mean", "std", "p10", "p50", "p90") if f"{name}_{suffix}" in frame.columns]
        if count_col in frame.columns:
            count_values = _numeric(frame[count_col])
            count_stats = {"rows_with_positive_valid_count": int((count_values > 0).sum()), "max_valid_count": int(count_values.max()) if count_values.notna().any() else None}
        else:
            count_stats = None
        constant = {}
        for column in value_cols:
            numeric = _numeric(frame[column]).dropna()
            unique = int(numeric.nunique())
            if unique <= 1:
                constant[column] = {"unique_finite_values": unique, "n_finite": len(numeric), "value": float(numeric.iloc[0]) if len(numeric) else None}
        if count_stats is not None or constant:
            result[name] = {"valid_count": count_stats, "constant_aggregate_columns": constant}
    audit.note(f"{prefix}.patient_aggregation_validity", {
        "rows": int(len(frame)), "feature_columns_present": [c for c in frame.columns if any(c.startswith(name + "_") for name in FEATURES)],
        "diagnostics": result,
    })


def _plot_waveform_review(reference_root: Path, raw_root: Path, tables: dict[str, Any], output_dir: Path) -> tuple[str | None, str | None]:
    """Create six short, deterministic waveform/peak-review panels."""
    import importlib.util
    import numpy as np
    import pandas as pd
    try:
        import wfdb
        import matplotlib.pyplot as plt
        from matplotlib import font_manager
    except ImportError as exc:
        return None, f"可选绘图依赖不可用（{type(exc).__name__}）；数字审计已保留"

    font_family = None
    windows_font = Path(os.environ.get("WINDIR", r"C:\Windows")) / "Fonts" / "msyh.ttc"
    if windows_font.is_file():
        try:
            font_manager.fontManager.addfont(str(windows_font))
            font_family = font_manager.FontProperties(fname=str(windows_font)).get_name()
        except (OSError, RuntimeError):
            font_family = None
    if font_family is None:
        try:
            font_path = font_manager.findfont("Microsoft YaHei", fallback_to_default=False)
        except (ValueError, RuntimeError):
            font_path = font_manager.findfont("DejaVu Sans", fallback_to_default=True)
        font_family = font_manager.FontProperties(fname=font_path).get_name()
    plt.rcParams["font.family"] = [font_family]
    plt.rcParams["axes.unicode_minus"] = False

    module_path = reference_root / "src" / "full_features.py"
    if not module_path.is_file():
        return None, "src/full_features.py missing; waveform plotting skipped"
    spec = importlib.util.spec_from_file_location("_plausibility_full_features", module_path)
    if spec is None or spec.loader is None:
        return None, "could not load the fresh feature implementation; waveform plotting skipped"
    feature_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(feature_module)
    # The source contract is preprocess_ecg(sig_1d, fs), detect_r_peaks(ecg, fs).
    preprocess = getattr(feature_module, "preprocess_ecg")
    detect = getattr(feature_module, "detect_r_peaks")

    status = tables["status"].copy()
    subjects = tables["subjects"].copy()
    records = tables["records"].copy()
    if not all(column in status.columns for column in ("patient_id", "endpoint_365_state", "primary_sinus_hrv_eligible", "af_flag")):
        return None, "analysis status lacks selection columns; waveform plotting skipped"
    status["patient_id"] = _ids(status)
    status["_af"] = status["af_flag"].fillna(False).astype(bool)
    status["_eligible"] = status["primary_sinus_hrv_eligible"].fillna(False).astype(bool)
    status["_state"] = status["endpoint_365_state"].astype("string")
    holter_records = records.loc[records.get("record_type", pd.Series(index=records.index, dtype="string")).astype("string").eq("HOLTER")]
    record_by_id: dict[str, Any] = {}
    for _, row in holter_records.iterrows():
        if not pd.isna(row.get("patient_id")):
            record_by_id[str(row["patient_id"]).strip()] = row
    chosen: list[tuple[str, str]] = []
    for label, state in (("阳性", "POSITIVE"), ("阴性", "NEGATIVE")):
        candidates = status.loc[status["_eligible"] & status["_state"].eq(state)].sort_values("patient_id", kind="stable")
        for index in range(min(2, len(candidates))):
            chosen.append((f"primary {label} #{index + 1}", str(candidates.iloc[index]["patient_id"])))
    af_candidates = status.loc[status["_af"] & status["patient_id"].isin(record_by_id)].sort_values("patient_id", kind="stable")
    if not af_candidates.empty:
        chosen.append(("AF", str(af_candidates.iloc[0]["patient_id"])))
    else:
        return None, "no AF subject available; six-panel waveform review skipped"

    selected_ids = {patient_id for _, patient_id in chosen}
    non_af = status.loc[~status["_af"] & ~status["patient_id"].isin(selected_ids)].copy()
    if "n_windows_theoretical" in non_af.columns and "n_windows_qc_valid" in non_af.columns:
        theoretical = _numeric(non_af["n_windows_theoretical"]).fillna(0)
        valid = _numeric(non_af["n_windows_qc_valid"]).fillna(0)
        non_af["_qc_rate"] = np.divide(valid, theoretical, out=np.ones(len(non_af), dtype=float), where=theoretical.to_numpy() > 0)
        non_af["_qc_valid"] = valid
        sort_columns = ["_qc_rate", "_qc_valid", "patient_id"]
        non_af = non_af.sort_values(sort_columns, ascending=True, kind="stable")
    else:
        return None, "QC window counts unavailable; waveform plotting skipped"
    for index, row in non_af.iterrows():
        patient_id = str(row["patient_id"])
        if patient_id in record_by_id:
            chosen.append(("non-AF 最差 QC", patient_id))
            break
    if len(chosen) != 6:
        return None, "could not select six distinct review cases; waveform plotting skipped"

    raw_base = raw_root.expanduser().resolve()
    plotted: list[tuple[str, np.ndarray, np.ndarray, float]] = []
    for label, patient_id in chosen:
        record = record_by_id.get(patient_id)
        if record is None:
            return None, "selected case lacks a Holter mapping; waveform plotting skipped"
        relative = record.get("signal_relative_path")
        if relative is None or pd.isna(relative):
            record_id = str(record.get("record_id"))
            stem = raw_base / "Holter_ECG" / record_id
        else:
            stem = raw_base / Path(str(relative)).with_suffix("")
        stem = stem.resolve()
        try:
            stem.relative_to(raw_base)
        except ValueError:
            return None, "Holter path is outside raw-root; waveform plotting skipped"
        try:
            fs = float(record.get("sampling_frequency"))
            sample_count = int(record.get("sample_count"))
        except (TypeError, ValueError):
            return None, "selected record is missing sampling metadata; waveform plotting skipped"
        if not np.isfinite(fs) or fs <= 0:
            return None, "selected record has invalid sampling frequency; waveform plotting skipped"
        start_sample = int(round(60.0 * fs))
        full_window_samples = int(round(300.0 * fs))
        end_sample = start_sample + full_window_samples
        display_samples = int(round(20.0 * fs))
        if sample_count < int(round((60.0 + 300.0) * fs)) or sample_count < end_sample:
            return None, "first complete 300-second window is unavailable; waveform plotting skipped"
        try:
            signal = wfdb.rdrecord(str(stem), sampfrom=start_sample, sampto=end_sample, channels=[0]).p_signal
        except Exception as exc:
            return None, f"selected 300-second waveform segment could not be read ({type(exc).__name__}); waveform plotting skipped"
        if signal is None or np.asarray(signal).ndim != 2 or len(signal) != full_window_samples:
            return None, "waveform read did not return the requested exact 300-second segment; waveform plotting skipped"
        try:
            full_ecg = preprocess(np.asarray(signal[:, 0], dtype=np.float64), fs)
            full_peaks = detect(full_ecg, fs)
        except Exception as exc:
            return None, f"selected segment processing failed ({type(exc).__name__}); waveform plotting skipped"
        ecg = np.asarray(full_ecg[:display_samples])
        peaks = np.asarray(full_peaks, dtype=int)
        peaks = peaks[(peaks >= 0) & (peaks < display_samples)]
        time_axis = np.arange(len(ecg), dtype=np.float64) / fs
        plotted.append((label, time_axis, ecg, fs, peaks))

    output_dir.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(3, 2, figsize=(14, 10), constrained_layout=True)
    for axis, (label, time_axis, ecg, fs, peaks) in zip(axes.flat, plotted):
        axis.plot(time_axis, ecg, linewidth=0.55, color="#245d8f")
        valid_peaks = peaks[(peaks >= 0) & (peaks < len(ecg))]
        axis.scatter(time_axis[valid_peaks], ecg[valid_peaks], s=18, color="#d94b3d", zorder=3, label="detected R peaks")
        axis.set_title(label)
        axis.set_xlabel("相对秒")
        axis.set_ylabel("预处理 ECG")
        axis.grid(alpha=0.2)
    figure.suptitle("20 秒预处理 ECG 与自动峰检测｜非临床标注验证，仅检查明显漏检/重复")
    path = output_dir / "waveform_peak_review.png"
    figure.savefig(path, dpi=160)
    plt.close(figure)
    return str(path), None


def _load_root_tables(data: Path) -> dict[str, Any]:
    return {
        "subjects": _read_required(data / "cohort" / "subjects.parquet", "cohort/subjects.parquet"),
        "records": _read_required(data / "cohort" / "records.parquet", "cohort/records.parquet"),
        "provenance": _read_required(data / "cohort" / "provenance.parquet", "cohort/provenance.parquet"),
        "status": _read_required(data / "analysis" / "patient_analysis_status.parquet", "analysis/patient_analysis_status.parquet"),
        "survival": _read_required(data / "analysis" / "survival_ready.parquet", "analysis/survival_ready.parquet"),
        "features": _load_feature_package(data),
    }


def _markdown(report: dict[str, Any]) -> str:
    failures = report["failures"]
    obs = report["observations"]
    lines = [
        "# MUSIC 重建数据合理性审计", "",
        f"生成时间：{report['generated_at_utc']}  ",
        f"新建数据状态：{report.get('fresh_build_status') or '未确认'}  ",
        f"审计结论：{'存在客观差异/完整性问题' if failures else '未发现已编码的客观失败'}  ",
        "", "> 本报告检查数据契约、计算关系和复现一致性。数值范围是描述统计；不代表任意临床阈值。",
        "", "## 失败项", "",
    ]
    for artifact in report.get("artifacts", []):
        if str(artifact).endswith("waveform_peak_review.png"):
            lines.extend([f"人工目视复核图：`{artifact}`。", ""])
    if failures:
        for item in failures:
            details = ", ".join(f"{key}={value}" for key, value in item.items() if key not in {"check", "count"})
            lines.append(f"- `{item['check']}`：{item['count']} 项" + (f"（{details}）" if details else ""))
    else:
        lines.append("- 无。")
    lines.extend(["", "## 队列及官方来源计数", ""])
    counts = obs.get("official_dataset_count_comparison", {}).get("comparisons", {})
    if counts:
        lines.extend(["| 项目 | 来源参考 | 实际 | 差值 |", "|---|---:|---:|---:|"])
        for key, item in counts.items():
            lines.append(f"| {key} | {item['expected']} | {item['observed'] if item['observed'] is not None else '不可得'} | {item['difference'] if item['difference'] is not None else '不可得'} |")
        lines.append("")
        lines.append(f"来源：[{PHYSIONET_URL}]({PHYSIONET_URL})。Exit 原始空值保留在原始代码计数中；仅在来源 cause code 为 0 时，将空 Exit 归入 code 0 survivor 类别。人数差异只作提示，不自动判错。")
    cohort = obs.get("fresh_vs_reference_cohort", {})
    if cohort:
        subj = cohort.get("subjects", {})
        lines.extend(["", "## 新旧队列及标签一致性", "", f"- subjects：新建 {subj.get('fresh_rows')} 行，参考 {subj.get('reference_rows')} 行；新建独有 {subj.get('fresh_only_subject_rows')}，参考独有 {subj.get('reference_only_subject_rows')}。", f"- 共享且参与对比的字段数：{subj.get('shared_fields_compared')}；字段差异：{len(subj.get('shared_field_rows_mismatched', {}))} 个。路径与哈希字段忽略。"])
        label = cohort.get("endpoint_source_label_fields", {})
        lines.append(f"- endpoint 源标签字段：比较 {len(label.get('fields_compared', []))} 个；差异字段 {len(label.get('mismatched_rows_by_field', {}))} 个。")
        records = cohort.get("records", {})
        build_version = records.get("build_code_version_provenance_observation") or {}
        if build_version:
            lines.append(f"- record `build_code_version` 差异 {build_version.get('mismatched_rows', 0)} 行，单独作为构建来源信息报告，不计入队列字段失败。")
    metadata = obs.get("metadata_ranges", {})
    if metadata:
        lines.extend(["", "## 临床元数据描述范围", "", "| 字段 | 数值数 | 缺失/未解析 | 最小 | P25 | 中位数 | P75 | 最大 |", "|---|---:|---:|---:|---:|---:|---:|---:|"])
        for name, values in metadata.items():
            lines.append(f"| {name} | {values.get('n_numeric', 0)} | {values.get('n_missing_or_unparsed', 0)} | {values.get('min', '—')} | {values.get('p25', '—')} | {values.get('median', '—')} | {values.get('p75', '—')} | {values.get('max', '—')} |")
    for prefix, title in (("fresh", "新建特征"), ("reference", "参考特征")):
        section = obs.get(f"{prefix}.feature_validity_ranges")
        if not section:
            continue
        lines.extend(["", f"## {title}范围、有效性及 QC", "", f"窗口行数：{section['rows']}。"])
        lines.extend(["", "| 特征 | 有效数 | 缺失/未解析 | 最小 | 中位数 | 最大 |", "|---|---:|---:|---:|---:|---:|"])
        for name, metric in section.get("features", {}).items():
            lines.append(f"| {name} | {metric.get('n_numeric', 0)} | {metric.get('n_missing_or_unparsed', 0)} | {metric.get('min', '—')} | {metric.get('median', '—')} | {metric.get('max', '—')} |")
        qc = section.get("qc_reason_counts", {})
        failures_by_reason = section.get("failure_reason_or_status_counts", {})
        lines.extend(["", f"- QC 原因计数：`{json.dumps(qc, ensure_ascii=False, sort_keys=True)}`", f"- 读取/提取状态计数：`{json.dumps(failures_by_reason, ensure_ascii=False, sort_keys=True)}`"])
    for prefix, title in (("fresh", "新建"), ("reference", "参考")):
        endpoint = obs.get(f"{prefix}.endpoint_states")
        if not endpoint:
            continue
        lines.extend(["", f"## {title} endpoint 及主要分析人群", "", "| horizon (days) | 状态计数 | 重算状态不一致 | 标签不一致 |", "|---:|---|---:|---:|"])
        for horizon in HORIZONS:
            item = endpoint.get(str(horizon))
            if item:
                lines.append(f"| {horizon} | `{json.dumps(item['state_counts'], ensure_ascii=False, sort_keys=True)}` | {item['state_mismatched_rows']} | {item['binary_label_mismatched_rows']} |")
        eligible = endpoint.get("primary_eligibility", {})
        lines.append(f"| 主要 HRV eligible | {eligible.get('eligible_count')} | 理由：`{json.dumps(eligible.get('reason_counts', {}), ensure_ascii=False, sort_keys=True)}` | |")
    feature_compare = obs.get("fresh_vs_reference_window_features")
    if feature_compare:
        lines.extend(["", "## 窗口特征复现", "", f"- 键：`{feature_compare['identity_keys']}`；匹配窗口 {feature_compare['matched_rows']}；新建独有 {feature_compare['fresh_only_rows']}；参考独有 {feature_compare['reference_only_rows']}。", f"- 数值容差：atol={feature_compare['numeric_tolerance']['atol']}, rtol={feature_compare['numeric_tolerance']['rtol']}。"])
        different = {name: item for name, item in feature_compare["feature_comparisons"].items() if item["mismatched_rows"]}
        lines.append(f"- 有差异特征数：{len(different)}。")
    aggregate_compare = obs.get("fresh_vs_reference_patient_aggregates")
    if aggregate_compare:
        mismatched = sum(item["mismatched_rows"] > 0 for item in aggregate_compare["feature_comparisons"].values())
        lines.extend(["", "## 患者级聚合特征复现", "", f"- 匹配患者 {aggregate_compare['matched_patients']}；新建独有 {aggregate_compare['fresh_only_patients']}；参考独有 {aggregate_compare['reference_only_patients']}。", f"- 聚合特征列 {len(aggregate_compare['feature_comparisons'])} 个；差异列 {mismatched} 个；数值容差 atol={aggregate_compare['numeric_tolerance']['atol']}, rtol={aggregate_compare['numeric_tolerance']['rtol']}。"])
    lines.extend(["", "## 解释与未核验项", "", "- `sig_std` 对应逐窗归一化 ECG 标准差；接近 1 或近似常量是实现流程的结果，应视为可能冗余列。", "- `pow_lf`、`pow_mf`、`pow_hf` 是 ECG FFT 功率占比，频带分别为 0.5–4、4–15、15–40 Hz，不是 NN-HRV 频带。", "- `mean_hr = 60 / mean_rr` 逐窗由同一 NN-RR 均值生成；跨窗汇总的 mean HR 不要求等于 60 除以汇总 mean RR。", "- 可疑数值范围只按实现内的关系与数学边界报告；临床意义需由研究者判断。"])
    if report.get("unverified"):
        lines.extend(["", "未核验："])
        lines.extend(f"- {item}" for item in report["unverified"])
    return "\n".join(lines) + "\n"


def run(args: argparse.Namespace) -> int:
    import pandas as pd

    fresh_root, fresh_data = _resolve_root(args.fresh_root)
    reference_root, reference_data = _resolve_root(args.reference_root)
    output_dir = args.output_dir.expanduser().resolve()
    audit = Audit()
    state, state_errors = _build_state(fresh_root, fresh_data)
    if state_errors:
        report = {
            "schema_version": 1,
            "generated_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
            "fresh_root": str(fresh_root), "reference_root": str(reference_root),
            "fresh_build_status": state,
            "status": "NOT_RUN_INCOMPLETE_OR_UNCONFIRMED_BUILD",
            "failures": [], "observations": {}, "warnings": state_errors,
            "unverified": ["full plausibility audit requires an explicit COMPLETE or FINALIZED build marker"],
            "artifacts": [],
        }
        output_dir.mkdir(parents=True, exist_ok=True)
        json_path = output_dir / "plausibility_audit.json"
        md_path = output_dir / "plausibility_audit.md"
        json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=_json_default) + "\n", encoding="utf-8")
        md_path.write_text("# MUSIC 重建数据合理性审计\n\n审计未运行：新建目录没有显式完整构建标记，或标记未完成。\n", encoding="utf-8")
        print("状态：未运行；新建构建未标记为 COMPLETE/FINALIZED。")
        print(f"报告：{json_path}；{md_path}")
        return 2

    required_paths = [
        fresh_data / "cohort" / "subjects.parquet",
        fresh_data / "cohort" / "records.parquet",
        fresh_data / "cohort" / "provenance.parquet",
        fresh_data / "features" / "full_5min" / "patient_features.parquet",
        fresh_data / "analysis" / "patient_analysis_status.parquet",
        fresh_data / "analysis" / "survival_ready.parquet",
        reference_data / "cohort" / "subjects.parquet",
        reference_data / "cohort" / "records.parquet",
    ]
    missing = [str(path.relative_to(path.parents[2])) if path.exists() else str(path) for path in required_paths if not path.is_file()]
    if missing:
        print("审计停止：一个或多个必需表缺失；详细路径不含患者行数据。")
        print(f"缺失表数：{len(missing)}")
        return 2

    fresh = _load_root_tables(fresh_data)
    reference = _load_root_tables(reference_data)
    audit.note("fresh_counts", {
        "subjects": len(fresh["subjects"]), "records": len(fresh["records"]),
        "holter_records": int(fresh["records"].get("record_type", pd.Series(dtype="string")).astype("string").eq("HOLTER").sum()),
        "window_rows": len(fresh["features"]["features"]),
        "window_metadata_rows": len(fresh["features"]["windows"]),
        "patient_feature_rows": len(fresh["features"]["patient_features"]),
        "status_rows": len(fresh["status"]), "survival_ready_rows": len(fresh["survival"]),
        "feature_shards": fresh["features"]["feature_shards"],
        "window_shards": fresh["features"]["window_shards"],
        "manifest_present": fresh["features"]["manifest"] is not None,
    })
    audit.note("reference_counts", {
        "subjects": len(reference["subjects"]), "records": len(reference["records"]),
        "holter_records": int(reference["records"].get("record_type", pd.Series(dtype="string")).astype("string").eq("HOLTER").sum()),
        "window_rows": len(reference["features"]["features"]),
        "window_metadata_rows": len(reference["features"]["windows"]),
        "patient_feature_rows": len(reference["features"]["patient_features"]),
        "status_rows": len(reference["status"]), "survival_ready_rows": len(reference["survival"]),
        "feature_shards": reference["features"]["feature_shards"],
        "window_shards": reference["features"]["window_shards"],
        "manifest_present": reference["features"]["manifest"] is not None,
    })
    for prefix, package in (("fresh", fresh["features"]), ("reference", reference["features"])):
        if package["manifest"] is None:
            audit.fail(f"{prefix}.window_manifest.missing", 1)
        if package.get("join_warning"):
            audit.unverified.append(f"{prefix} feature/window join: {package['join_warning']}")

    _metadata_and_official_counts(fresh["subjects"], fresh["records"], audit)
    _check_key_integrity(fresh["subjects"], fresh["records"], fresh["provenance"], fresh["features"], fresh["status"], fresh["survival"], audit, "fresh")
    _check_key_integrity(reference["subjects"], reference["records"], reference["provenance"], reference["features"], reference["status"], reference["survival"], audit, "reference")
    _check_manifest_counts(fresh["features"], audit, "fresh")
    _check_manifest_counts(reference["features"], audit, "reference")
    _check_endpoints(fresh["subjects"], fresh["status"], audit, "fresh")
    _check_endpoints(reference["subjects"], reference["status"], audit, "reference")
    _feature_value_checks(fresh["features"]["features"], fresh["features"]["windows"], audit, "fresh")
    _feature_value_checks(reference["features"]["features"], reference["features"]["windows"], audit, "reference")
    _aggregate_feature_diagnostics(fresh["features"]["patient_features"], audit, "fresh")
    _aggregate_feature_diagnostics(reference["features"]["patient_features"], audit, "reference")
    _compare_cohorts(fresh["subjects"], reference["subjects"], fresh["records"], reference["records"], audit)
    _compare_patient_features(fresh["features"]["features"], reference["features"]["features"], audit)
    _compare_patient_aggregates(fresh["features"]["patient_features"], reference["features"]["patient_features"], audit)
    if args.plot_waveforms:
        image_path, warning = _plot_waveform_review(reference_root, args.raw_root, fresh, output_dir)
        if image_path:
            audit.artifacts.append(image_path)
            audit.note("waveform_peak_review", {
                "artifact": image_path, "panels": 6,
                "selection": "primary population: sorted first two POSITIVE and first two NEGATIVE; first sorted AF subject with Holter; non-AF subject with lowest qc-valid/theoretical window ratio then lowest qc-valid count",
                "segment": "read, preprocess, and detect peaks on the full first 300-second window beginning at 60 seconds; display only its first 20 seconds from channel 0",
                "review_scope": "non-clinical annotation check for obvious missed or duplicate detections",
            })
        if warning:
            audit.warnings.append(warning)

    report = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "fresh_root": str(fresh_root), "reference_root": str(reference_root),
        "fresh_build_status": state,
        "status": "FAIL" if audit.failures else "PASS_WITH_UNVERIFIED" if audit.unverified else "PASS",
        "failures": audit.failures,
        "observations": audit.observations,
        "warnings": audit.warnings,
        "unverified": audit.unverified,
        "artifacts": [],
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "plausibility_audit.json"
    md_path = output_dir / "plausibility_audit.md"
    report["artifacts"] = [str(json_path), str(md_path), *audit.artifacts]
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=_json_default) + "\n", encoding="utf-8")
    md_path.write_text(_markdown(report), encoding="utf-8")
    print(f"状态：{report['status']}；客观失败项：{len(audit.failures)}；未核验项：{len(audit.unverified)}")
    print(f"新建队列：{len(fresh['subjects'])} 人；Holter：{audit.observations['fresh_counts']['holter_records']}；窗口：{audit.observations['fresh_counts']['window_rows']}")
    print(f"报告：{json_path}；{md_path}")
    return 1 if audit.failures else 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="审计新建 MUSIC cohort/features 与原始参考包的一致性、结构和数值关系。")
    parser.add_argument("--fresh-root", required=True, type=Path, help="新建项目根目录（应有 data/ 与 COMPLETE/FINALIZED 构建标记）")
    parser.add_argument("--reference-root", required=True, type=Path, help="原始参考项目根目录")
    parser.add_argument("--output-dir", required=True, type=Path, help="JSON 与 Markdown 报告输出目录")
    parser.add_argument("--plot-waveforms", action="store_true", help="额外生成六例 ECG 峰检测人工目视复核图")
    parser.add_argument("--raw-root", type=Path, help="MUSIC 原始波形目录；与 --plot-waveforms 同用")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.plot_waveforms and args.raw_root is None:
        print("--plot-waveforms 必须同时提供 --raw-root。", file=sys.stderr)
        return 2
    try:
        return run(args)
    except Exception as exc:
        # Do not print exception text: malformed keys or data can put subject
        # identifiers in third-party error messages.
        print(f"审计停止：{type(exc).__name__}；未输出行级标识信息。", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
