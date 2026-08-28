#!/usr/bin/env python3
"""Build the complete, checkpointed MUSIC 5-minute feature package.

The builder keeps waveform reads and extraction caches outside ``data``.  Each
Holter is processed independently, with one checkpoint per patient, so a
stopped run resumes only when the patient identity, official source/header
hashes, preprocessing hash, feature schema version, and window hash all match.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import math
import os
import shutil
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import yaml

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.analysis_profiles import (  # noqa: E402
    build_analysis_population_365,
    build_patient_analysis_status,
    build_survival_ready,
    canonical_hash,
)
from src.full_aggregation import (  # noqa: E402
    AGGREGATED_FEATURE_NAMES,
    VALID_COUNT_NAMES,
    aggregate_patient_features,
)
from src.full_features import (  # noqa: E402
    FEATURE_NAMES,
    FEATURE_VALIDITY_COLUMNS,
    apply_af_policy,
    complete_window_starts,
    extract_full_windows,
    tail_seconds,
)
from src.endpoints import SUPPORTED_HORIZONS, build_endpoint  # noqa: E402


WINDOW_CONFIG_NAME = "full_5min_windows.yaml"
PREPROCESSING_CONFIG_NAME = "full_preprocessing.yaml"
FEATURE_CONFIG_NAME = "features_v2.yaml"
TARGET_SHARD_BYTES = 16 * 1024 * 1024
MAX_SHARD_BYTES = 25 * 1024 * 1024
DEFAULT_SHARD_ROWS = 40_000
CACHE_SUBDIR = Path("build_cache") / "full_5min"
FEATURE_SUBDIR = Path("data") / "features" / "full_5min"
ANALYSIS_SUBDIR = Path("data") / "analysis"


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{path.stem}.", suffix=".tmp.parquet", dir=path.parent)
    os.close(fd)
    temporary = Path(name)
    try:
        frame.to_parquet(temporary, engine="pyarrow", compression="zstd", index=False)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _read_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle) or {}
    if not isinstance(value, dict):
        raise ValueError(f"configuration must be a mapping: {path}")
    return value


def load_build_configs(repo: Path = REPO) -> dict[str, Any]:
    window_path = repo / "config" / WINDOW_CONFIG_NAME
    preprocessing_path = repo / "config" / PREPROCESSING_CONFIG_NAME
    feature_path = repo / "config" / FEATURE_CONFIG_NAME
    window = _read_yaml(window_path)
    preprocessing = _read_yaml(preprocessing_path)
    features = _read_yaml(feature_path)
    if int(window.get("window", {}).get("length_sec", -1)) != 300:
        raise ValueError("full window length must be 300 seconds")
    if int(window.get("window", {}).get("stride_sec", -1)) != 300:
        raise ValueError("full window stride must be 300 seconds")
    if int(window.get("window", {}).get("first_start_sec", -1)) != 60:
        raise ValueError("full window first start must be 60 seconds")
    if int(features.get("feature_count", -1)) != 20 or len(features.get("features", [])) != 20:
        raise ValueError("features_v2 must contain exactly 20 features")
    names = [str(item.get("feature_name")) for item in features["features"]]
    if names != list(FEATURE_NAMES):
        raise ValueError("features_v2 feature names/order do not match Phase 3")
    return {
        "window": window,
        "preprocessing": preprocessing,
        "features": features,
        "window_hash": _sha256_file(window_path),
        "preprocessing_hash": _sha256_file(preprocessing_path),
        "feature_schema_version": str(features.get("feature_schema_version", features.get("schema_version"))),
        "feature_schema_hash": _sha256_file(feature_path),
    }


def load_cohort(repo: Path = REPO) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, dict[str, Any]]]:
    subjects = pd.read_parquet(repo / "data" / "cohort" / "subjects.parquet")
    records = pd.read_parquet(repo / "data" / "cohort" / "records.parquet")
    provenance = pd.read_parquet(repo / "data" / "cohort" / "provenance.parquet")
    if "patient_id" not in subjects or subjects["patient_id"].duplicated().any():
        raise ValueError("subjects must contain unique patient_id")
    holter = records.loc[records["record_type"].astype("string").eq("HOLTER")].copy()
    if holter["patient_id"].duplicated().any():
        raise ValueError("HOLTER record mapping must be one-to-one")
    provenance_holter = provenance.loc[provenance["record_type"].astype("string").eq("HOLTER")]
    if provenance_holter.duplicated(["patient_id", "record_id"]).any():
        raise ValueError("HOLTER provenance mapping must be unique")
    provenance_map = {
        str(row["patient_id"]): row
        for row in provenance_holter.to_dict(orient="records")
    }
    records_map = {
        str(row["patient_id"]): row
        for row in holter.to_dict(orient="records")
    }
    # Merge official hashes into the record mapping without changing the frozen
    # Phase 2 tables on disk.
    for patient_id, record in records_map.items():
        source = provenance_map.get(patient_id, {})
        record["official_source_sha256"] = source.get("source_sha256")
        record["official_header_sha256"] = source.get("header_sha256")
    subjects = subjects.copy()
    subjects["patient_id"] = subjects["patient_id"].astype("string")
    return subjects, records, records_map


def checkpoint_key(
    patient_id: str,
    record: Mapping[str, Any],
    configs: Mapping[str, Any],
) -> str:
    """Compute the exact patient checkpoint identity required by the contract."""

    payload = {
        "patient_id": str(patient_id),
        "official_source_sha256": None if pd.isna(record.get("official_source_sha256")) else str(record.get("official_source_sha256")),
        "official_header_sha256": None if pd.isna(record.get("official_header_sha256")) else str(record.get("official_header_sha256")),
        "preprocessing_hash": str(configs["preprocessing_hash"]),
        "feature_schema_version": str(configs["feature_schema_version"]),
        "window_hash": str(configs["window_hash"]),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def _cache_paths(cache_root: Path, patient_id: str) -> tuple[Path, Path]:
    safe = str(patient_id).replace("/", "_").replace("\\", "_")
    return cache_root / "patients" / f"{safe}.parquet", cache_root / "checkpoints" / f"{safe}.json"


def _load_cached_patient(
    cache_root: Path,
    patient_id: str,
    key: str,
) -> pd.DataFrame | None:
    parquet_path, checkpoint_path = _cache_paths(cache_root, patient_id)
    if not parquet_path.is_file() or not checkpoint_path.is_file():
        return None
    try:
        checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        if checkpoint.get("status") != "COMPLETE" or checkpoint.get("checkpoint_key") != key:
            return None
        frame = pd.read_parquet(parquet_path)
        if "patient_id" not in frame.columns or not frame["patient_id"].astype("string").eq(str(patient_id)).all():
            return None
        if int(checkpoint.get("row_count", -1)) != len(frame):
            return None
        return frame
    except Exception:
        return None


def _record_stem(raw_root: Path, record: Mapping[str, Any]) -> Path:
    relative = record.get("signal_relative_path")
    if relative is None or pd.isna(relative):
        record_id = str(record.get("record_id"))
        return raw_root / "Holter_ECG" / record_id
    return raw_root / Path(str(relative)).with_suffix("")


def process_patient(
    *,
    patient_id: str,
    subject: Mapping[str, Any],
    record: Mapping[str, Any],
    raw_root: Path,
    configs: Mapping[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Extract one patient without passing outcome/label fields to the reader."""

    fs = float(record["sampling_frequency"])
    sample_count = int(record["sample_count"])
    duration = float(sample_count / fs)
    starts = complete_window_starts(duration)
    stem = _record_stem(raw_root, record)
    hea = Path(str(stem) + ".hea")
    dat = Path(str(stem) + ".dat")
    if not hea.is_file() or not dat.is_file():
        missing = ".hea" if not hea.is_file() else ".dat"
        raise FileNotFoundError(f"{patient_id}: missing waveform {missing} for {stem}")
    leads = record.get("lead_names")
    channel_name = str(leads[0]) if isinstance(leads, (list, tuple, np.ndarray)) and len(leads) else None
    windows = extract_full_windows(
        stem,
        fs=fs,
        sample_count=sample_count,
        patient_id=patient_id,
        channel_name=channel_name,
    )
    # AF is official subject metadata and is applied only after extraction;
    # signal/non-HRV values remain intact in the cached window table.
    windows = apply_af_policy(windows, bool(subject.get("af_flag", False)))
    windows.insert(1, "record_id", str(record["record_id"]))
    windows["tail_seconds"] = tail_seconds(duration, starts)
    return windows, {
        "patient_id": patient_id,
        "record_id": str(record["record_id"]),
        "duration_sec": duration,
        "tail_seconds": tail_seconds(duration, starts),
        "theoretical_window_count": len(starts),
        "row_count": len(windows),
        "af_flag": bool(subject.get("af_flag", False)),
        "status": "COMPLETE",
    }


def _empty_windows() -> pd.DataFrame:
    columns = [
        "patient_id", "record_id", "window_idx", "window_start_sec", "window_end_sec", "start_sample",
        "requested_samples", "actual_samples", "sampling_frequency", "channel_selected", "channel_name",
        "window_expected", "window_within_record", "waveform_read_success", "feature_extraction_success",
        "qc_valid", "window_status", "failure_reason", "qc_status", "qc_reason", "raw_rpeak_count",
        "raw_rr_count", "valid_rr_count", "removed_rr_count", "removed_rr_ratio", "tail_seconds",
        *FEATURE_NAMES, *FEATURE_VALIDITY_COLUMNS,
    ]
    return pd.DataFrame(columns=columns)


def _normalise_windows(frame: pd.DataFrame) -> pd.DataFrame:
    """Make cached/reused frames stable across pandas/pyarrow versions."""

    if frame is None or frame.empty:
        return _empty_windows() if frame is None else frame
    result = frame.copy()
    if "record_id" not in result.columns:
        result.insert(1, "record_id", pd.NA)
    if "window_idx" not in result.columns and "window_id" in result.columns:
        result["window_idx"] = result["window_id"]
    for name in FEATURE_NAMES:
        result[name] = pd.to_numeric(result[name], errors="coerce").astype("float64")
    for name in FEATURE_VALIDITY_COLUMNS:
        if name not in result.columns:
            result[name] = np.isfinite(result[name.replace("_valid", "")])
        result[name] = result[name].fillna(False).astype(bool)
    for name in ("waveform_read_success", "feature_extraction_success", "qc_valid", "window_expected", "window_within_record"):
        if name not in result.columns:
            result[name] = False
        result[name] = result[name].fillna(False).astype(bool)
    return result.sort_values(["patient_id", "window_idx"], kind="stable").reset_index(drop=True)


def _window_output_frames(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    work = _normalise_windows(frame)
    manifest_columns = [
        "patient_id", "record_id", "window_idx", "window_start_sec", "window_end_sec", "start_sample",
        "requested_samples", "actual_samples", "sampling_frequency", "channel_selected", "channel_name",
        "window_expected", "window_within_record", "waveform_read_success", "feature_extraction_success",
        "qc_valid", "window_status", "failure_reason", "qc_status", "qc_reason", "raw_rpeak_count",
        "raw_rr_count", "valid_rr_count", "removed_rr_count", "removed_rr_ratio", "tail_seconds",
    ]
    manifest = work.reindex(columns=manifest_columns).copy()
    manifest = manifest.rename(columns={"window_idx": "window_id", "window_start_sec": "start_sec", "window_end_sec": "end_sec", "sampling_frequency": "sampling_frequency"})
    manifest["window_id"] = pd.to_numeric(manifest["window_id"], errors="coerce").astype("int64")
    manifest["start_sec"] = pd.to_numeric(manifest["start_sec"], errors="coerce").astype("int64")
    manifest["end_sec"] = pd.to_numeric(manifest["end_sec"], errors="coerce").astype("int64")
    manifest["end_sample"] = (manifest["start_sample"].astype("int64") + manifest["requested_samples"].astype("int64"))
    # Keep the established Phase 3 names as aliases in the manifest.
    manifest["window_within_record"] = manifest["window_within_record"].fillna(False).astype(bool)
    manifest["waveform_read_success"] = manifest["waveform_read_success"].fillna(False).astype(bool)
    manifest["feature_extraction_success"] = manifest["feature_extraction_success"].fillna(False).astype(bool)
    manifest["qc_valid"] = manifest["qc_valid"].fillna(False).astype(bool)
    manifest["valid"] = manifest["feature_extraction_success"]
    features = work.loc[:, ["patient_id", "window_idx", "feature_extraction_success", "qc_valid", "qc_status", "qc_reason", *FEATURE_NAMES, *FEATURE_VALIDITY_COLUMNS]].copy()
    features = features.rename(columns={"window_idx": "window_id", "feature_extraction_success": "valid"})
    for name in FEATURE_NAMES:
        features[name] = pd.to_numeric(features[name], errors="coerce").astype("float64")
    for name in FEATURE_VALIDITY_COLUMNS:
        features[name] = features[name].fillna(False).astype(bool)
    features["valid"] = features["valid"].fillna(False).astype(bool)
    features["qc_valid"] = features["qc_valid"].fillna(False).astype(bool)
    return manifest, features


def _clear_output_shards(feature_dir: Path) -> None:
    feature_dir.mkdir(parents=True, exist_ok=True)
    for pattern in ("windows-*.parquet", "features-*.parquet", "manifest.json"):
        for path in feature_dir.glob(pattern):
            if path.is_file():
                path.unlink()
    for subdir in (feature_dir / "windows", feature_dir / "features"):
        subdir.mkdir(parents=True, exist_ok=True)
        for path in subdir.glob("part-*.parquet"):
            if path.is_file():
                path.unlink()


def write_window_shards(
    *,
    cache_root: Path,
    patient_ids: Sequence[str],
    output_dir: Path,
    shard_rows: int = DEFAULT_SHARD_ROWS,
) -> dict[str, Any]:
    """Write deterministic contiguous patient-range shards under 25 MiB."""

    _clear_output_shards(output_dir)
    manifest_paths: list[dict[str, Any]] = []
    feature_paths: list[dict[str, Any]] = []
    manifest_buffer: list[pd.DataFrame] = []
    feature_buffer: list[pd.DataFrame] = []
    buffered_rows = 0
    shard_index = 0

    def flush() -> None:
        nonlocal manifest_buffer, feature_buffer, buffered_rows, shard_index
        if not manifest_buffer:
            return
        manifest = pd.concat(manifest_buffer, ignore_index=True).sort_values(["patient_id", "window_id"], kind="stable")
        features = pd.concat(feature_buffer, ignore_index=True).sort_values(["patient_id", "window_id"], kind="stable")
        stem = f"{shard_index:05d}"
        manifest_path = output_dir / "windows" / f"part-{stem}.parquet"
        feature_path = output_dir / "features" / f"part-{stem}.parquet"
        _atomic_parquet(manifest, manifest_path)
        _atomic_parquet(features, feature_path)
        sizes = [manifest_path.stat().st_size, feature_path.stat().st_size]
        if max(sizes) >= MAX_SHARD_BYTES:
            raise RuntimeError(f"window shard exceeds 25 MiB: {manifest_path} / {feature_path} ({sizes})")
        manifest_paths.append({"path": manifest_path.relative_to(output_dir).as_posix(), "rows": len(manifest), "bytes": sizes[0], "patient_first": str(manifest["patient_id"].iloc[0]), "patient_last": str(manifest["patient_id"].iloc[-1])})
        feature_paths.append({"path": feature_path.relative_to(output_dir).as_posix(), "rows": len(features), "bytes": sizes[1], "patient_first": str(features["patient_id"].iloc[0]), "patient_last": str(features["patient_id"].iloc[-1])})
        shard_index += 1
        manifest_buffer = []
        feature_buffer = []
        buffered_rows = 0

    for patient_id in sorted(str(value) for value in patient_ids):
        parquet_path, _ = _cache_paths(cache_root, patient_id)
        frame = pd.read_parquet(parquet_path)
        manifest, features = _window_output_frames(frame)
        if len(manifest) != len(features):
            raise RuntimeError(f"window/feature row mismatch for {patient_id}")
        if buffered_rows and buffered_rows + len(manifest) > int(shard_rows):
            flush()
        manifest_buffer.append(manifest)
        feature_buffer.append(features)
        buffered_rows += len(manifest)
    flush()
    total_rows = int(sum(item["rows"] for item in manifest_paths))
    payload = {
        "schema_version": 1,
        "window_table": "windows/part-*.parquet",
        "feature_table": "features/part-*.parquet",
        "patient_sorted_contiguous": True,
        "window_row_count": total_rows,
        "window_shards": manifest_paths,
        "feature_shards": feature_paths,
        "max_shard_bytes": MAX_SHARD_BYTES,
        "compression": "zstd",
        "feature_dtype": "float64",
        "raw_waveform_or_cache_in_data": False,
    }
    _atomic_json(output_dir / "manifest.json", payload)
    return payload


def _patient_source_row(subjects: pd.DataFrame, patient_id: str, record: Mapping[str, Any] | None) -> dict[str, Any]:
    match = subjects.loc[subjects["patient_id"].astype("string").eq(str(patient_id))]
    if len(match) != 1:
        raise ValueError(f"expected exactly one subject row for {patient_id}")
    source = match.iloc[0].to_dict()
    if record is not None:
        source["record_id"] = str(record.get("record_id"))
        source["fs"] = record.get("sampling_frequency")
    return source


def _build_patient_features(
    *,
    subjects: pd.DataFrame,
    records_map: Mapping[str, Mapping[str, Any]],
    completed: Mapping[str, pd.DataFrame],
    failed: set[str],
    pvc_threshold: float = 0.20,
) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for source in subjects.sort_values("patient_id", kind="stable").to_dict("records"):
        patient_id = str(source["patient_id"])
        record = records_map.get(patient_id)
        frame = completed.get(patient_id, _empty_windows())
        source_row = _patient_source_row(subjects, patient_id, record)
        if patient_id in failed:
            source_row["tail_seconds"] = 0.0
        elif "tail_seconds" in frame.columns and not frame.empty:
            source_row["tail_seconds"] = float(frame["tail_seconds"].iloc[0])
        else:
            source_row["tail_seconds"] = 0.0
        if frame.empty:
            # Avoid invoking the wide pandas aggregation machinery for the 56
            # official no-Holter rows (and any isolated failures).  The empty
            # patient row is still explicit and carries null feature values.
            numeric = {
                "label": np.nan,
                "followup_days": source_row.get("followup_days", np.nan),
                "cause_of_death": pd.to_numeric(pd.Series([source_row.get("cause_of_death_raw")]), errors="coerce").iloc[0],
                "fs": source_row.get("fs", np.nan),
                "record_id": source_row.get("record_id", pd.NA),
                "n_windows_theoretical": 0,
                "n_windows_successful": 0,
                "n_windows_qc_valid": 0,
                "n_windows_used": 0,
                "window_success_rate": 0.0,
                "raw_rpeak_count_total": 0.0,
                "raw_rr_count_total": 0.0,
                "valid_rr_count_total": 0.0,
                "removed_rr_count_total": 0.0,
                "tail_seconds": float(source_row.get("tail_seconds", 0.0) or 0.0),
                "pvc_count_24h": pd.to_numeric(pd.Series([source_row.get("pvc_count_24h")]), errors="coerce").iloc[0],
                "pvc_denominator_beats": np.nan,
                "pvc_burden": np.nan,
                "high_pvc_burden": False,
            }
            empty_row = {"patient_id": patient_id, **numeric}
            empty_row.update({name: np.nan for name in AGGREGATED_FEATURE_NAMES})
            empty_row.update({name: 0 for name in VALID_COUNT_NAMES})
            one = pd.DataFrame([empty_row])
        else:
            one = aggregate_patient_features(
                frame,
                subjects=pd.DataFrame([source_row]),
                include_ineligible=True,
                min_qc_valid_windows=0,
                use_qc_valid=False,
                pvc_threshold=pvc_threshold,
            )
        # Consolidate the wide aggregate frame before appending build-state
        # columns; otherwise pandas emits one fragmentation warning per
        # patient and needlessly slows the 992-row status assembly.
        one = one.copy()
        # A failed attempt is represented separately; it must not look like a
        # processed Holter with zero valid windows in the analysis status.
        one["processed_holter"] = bool(patient_id in completed) and bool(source.get("has_holter", False))
        one["processing_status"] = "FAILED" if patient_id in failed else "COMPLETE" if patient_id in completed else "NOT_APPLICABLE"
        rows.append(one)
    result = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    subject_status = subjects.loc[:, ["patient_id", "has_holter", "af_flag"]].copy()
    subject_status["patient_id"] = subject_status["patient_id"].astype("string")
    result["patient_id"] = result["patient_id"].astype("string")
    result = result.merge(subject_status, on="patient_id", how="left", validate="one_to_one")
    processed = result["processed_holter"].fillna(False).astype(bool)
    af = result["af_flag"].fillna(False).astype(bool)
    high_pvc = result["high_pvc_burden"].fillna(False).astype(bool)
    has_qc = pd.to_numeric(result["n_windows_qc_valid"], errors="coerce").fillna(0).gt(0)
    result["primary_sinus_hrv_eligible"] = (processed & ~af & ~high_pvc & has_qc).astype("boolean")
    result["primary_sinus_hrv_reason"] = np.select(
        [~result["has_holter"].fillna(False).astype(bool), ~processed, af, high_pvc, ~has_qc],
        ["NO_HOLTER", "PROCESSING_FAILED", "AF", "HIGH_PVC_BURDEN", "NO_QC_VALID_WINDOWS"],
        default="ELIGIBLE",
    )
    return result.sort_values("patient_id", kind="stable").reset_index(drop=True)


def _write_reports(
    *,
    repo: Path,
    subjects: pd.DataFrame,
    records_map: Mapping[str, Mapping[str, Any]],
    patient_features: pd.DataFrame,
    status: pd.DataFrame,
    build_report: Mapping[str, Any],
) -> None:
    reports = repo / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    endpoint_flow: dict[str, dict[str, int]] = {}
    for horizon in sorted(SUPPORTED_HORIZONS):
        column = f"endpoint_{horizon}_state"
        counts = status[column].value_counts(dropna=False).to_dict()
        endpoint_flow[str(horizon)] = {str(key): int(value) for key, value in counts.items()}
    reason_counts = {str(key): int(value) for key, value in status["primary_sinus_hrv_reason"].value_counts(dropna=False).items()}
    cohort_flow = {
        "official_subjects": int(len(subjects)),
        "holter_records": int(len(records_map)),
        "processed_holter": int(status["processed_holter"].fillna(False).sum()),
        "processing_failures": int((status["processed_holter"].fillna(False) & status["n_windows_theoretical"].eq(0) & status["has_holter"].fillna(False)).sum()),
        "primary_sinus_hrv_reason_counts": reason_counts,
        "endpoint_states": endpoint_flow,
        "generated_at_utc": _utc_now(),
    }
    _atomic_json(reports / "COHORT_FLOW.json", cohort_flow)
    flow_rows: list[dict[str, Any]] = [
        {"stage": "ALL_PATIENTS", "n": int(len(subjects)), "excluded_n": 0, "reason": ""},
        {
            "stage": "HOLTER_AVAILABLE",
            "n": int(status["has_holter"].fillna(False).sum()),
            "excluded_n": int((~status["has_holter"].fillna(False)).sum()),
            "reason": "NO_HOLTER",
        },
        {
            "stage": "HOLTER_PROCESSED",
            "n": int(status["processed_holter"].fillna(False).sum()),
            "excluded_n": int((status["has_holter"].fillna(False) & ~status["processed_holter"].fillna(False)).sum()),
            "reason": "PROCESSING_FAILED",
        },
        {
            "stage": "SIGNAL_FEATURES_AVAILABLE",
            "n": int(status["valid_window_count"].fillna(0).gt(0).sum()),
            "excluded_n": int((status["processed_holter"].fillna(False) & status["valid_window_count"].fillna(0).eq(0)).sum()),
            "reason": "NO_QC_VALID_WINDOWS",
        },
        {
            "stage": "PRIMARY_SINUS_HRV_ELIGIBLE",
            "n": int(status["primary_sinus_hrv_eligible"].fillna(False).sum()),
            "excluded_n": int((~status["primary_sinus_hrv_eligible"].fillna(False)).sum()),
            "reason": "SEE_PRIMARY_SINUS_HRV_REASON",
        },
        {
            "stage": "ENDPOINT_365_EVALUABLE",
            "n": int(status["endpoint_365_state"].isin(["POSITIVE", "NEGATIVE"]).sum()),
            "excluded_n": int((~status["endpoint_365_state"].isin(["POSITIVE", "NEGATIVE"])).sum()),
            "reason": "CENSORED_OR_COMPETING_OR_UNKNOWN",
        },
        {
            "stage": "FINAL_MODEL_ANALYSIS_365D",
            "n": int(status["model_365_included"].fillna(False).sum()),
            "excluded_n": int((~status["model_365_included"].fillna(False)).sum()),
            "reason": "SEE_MODEL_365_EXCLUSION_REASON",
        },
    ]
    pd.DataFrame(flow_rows).to_csv(reports / "COHORT_FLOW.csv", index=False, lineterminator="\n")
    lines = [
        "# Full cohort flow",
        "",
        f"- Official subjects: **{len(subjects)}**",
        f"- Official Holter records: **{len(records_map)}**",
        f"- Processed Holters: **{int(status['processed_holter'].fillna(False).sum())}**",
        "",
        "## Primary sinus-HRV status",
        "",
        "| Reason | Count |",
        "|---|---:|",
    ]
    lines.extend(f"| {key} | {value} |" for key, value in sorted(reason_counts.items()))
    lines.extend(["", "## Dynamic endpoint states", "", "| Horizon | POSITIVE | NEGATIVE | CENSORED | COMPETING_EVENT | UNKNOWN |", "|---:|---:|---:|---:|---:|---:|"])
    for horizon in sorted(SUPPORTED_HORIZONS):
        counts = endpoint_flow[str(horizon)]
        lines.append("| {h} | {p} | {n} | {c} | {ce} | {u} |".format(h=horizon, p=counts.get("POSITIVE", 0), n=counts.get("NEGATIVE", 0), c=counts.get("CENSORED", 0), ce=counts.get("COMPETING_EVENT", 0), u=counts.get("UNKNOWN", 0)))
    (reports / "COHORT_FLOW.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    followup = pd.to_numeric(subjects.loc[subjects["cause_of_death_raw"].astype("string").eq("3"), "followup_days"], errors="coerce").dropna().astype(float)
    bins = {
        "le_90d": int((followup <= 90).sum()),
        "d91_180": int(((followup > 90) & (followup <= 180)).sum()),
        "d181_365": int(((followup > 180) & (followup <= 365)).sum()),
        "d366_730": int(((followup > 365) & (followup <= 730)).sum()),
        "gt_730d": int((followup > 730).sum()),
    }
    def five(values: pd.Series) -> str:
        if values.empty:
            return "n/a"
        q = values.quantile([0, .25, .5, .75, 1]).tolist()
        return " / ".join(f"{float(v):.6g}" for v in q)
    (reports / "SCD_INTERVAL_DISTRIBUTION.md").write_text(
        "# SCD interval distribution\n\n"
        "Baseline is the official enrollment/Holter timing field preserved by Phase 2; no event timing is inferred from waveform windows.\n\n"
        f"- SCD subjects with finite follow-up: **{len(followup)}**\n"
        f"- Min / P25 / median / P75 / max days: **{five(followup)}**\n"
        f"- Bins: `{json.dumps(bins, sort_keys=True)}`\n",
        encoding="utf-8",
    )

    payload = dict(build_report)
    payload["cohort_flow_report"] = cohort_flow
    _atomic_json(reports / "FULL_COHORT_BUILD.json", payload)
    md = [
        "# Full 5-minute cohort build",
        "",
        f"- Status: **{payload.get('status')}**",
        f"- Official subjects: **{payload.get('official_subjects')}**",
        f"- Holter records: **{payload.get('holter_records')}**",
        f"- Completed/reused Holters: **{payload.get('completed_holter')}**",
        f"- Failed Holters: **{payload.get('failed_holter')}**",
        f"- Theoretical complete windows: **{payload.get('theoretical_window_count')}**",
        f"- Successful feature rows: **{payload.get('feature_extraction_success_rows')}**",
        f"- QC-valid rows: **{payload.get('qc_valid_rows')}**",
        f"- Generated data bytes: **{payload.get('generated_data_bytes')}**",
        f"- Maximum shard bytes: **{payload.get('maximum_shard_bytes')}**",
        "",
        "Waveform reads used one exact WFDB segment per window; waveform/cache content is not written under `data/`.",
    ]
    (reports / "FULL_COHORT_BUILD.md").write_text("\n".join(md) + "\n", encoding="utf-8")


def build(
    repo: Path = REPO,
    raw_root: Path | None = None,
    *,
    resume: bool = True,
    max_patients: int | None = None,
    patient_ids: Sequence[str] | None = None,
    shard_rows: int = DEFAULT_SHARD_ROWS,
    workers: int = 1,
) -> dict[str, Any]:
    repo = Path(repo).resolve()
    raw = (Path(raw_root) if raw_root is not None else repo / "music-sudden-cardiac-death-in-chronic-heart-failure-1.0.1").resolve(strict=True)
    configs = load_build_configs(repo)
    subjects, records, records_map = load_cohort(repo)
    if len(subjects) != 992:
        raise RuntimeError(f"official patient count changed: {len(subjects)}")
    selected = sorted(str(value) for value in (patient_ids if patient_ids is not None else subjects.loc[subjects["has_holter"].fillna(False).astype(bool), "patient_id"].tolist()))
    if max_patients is not None:
        selected = selected[: int(max_patients)]
    if int(workers) < 1:
        raise ValueError("workers must be positive")
    if len(selected) != len(set(selected)):
        raise ValueError("selected patient_ids must be unique")
    cache_root = repo / CACHE_SUBDIR
    cache_root.mkdir(parents=True, exist_ok=True)
    (cache_root / "patients").mkdir(parents=True, exist_ok=True)
    (cache_root / "checkpoints").mkdir(parents=True, exist_ok=True)
    completed: dict[str, pd.DataFrame] = {}
    failed: dict[str, dict[str, Any]] = {}
    reused_count = 0
    progress = {"status": "RUNNING", "started_at_utc": _utc_now(), "official_subjects": len(subjects), "holter_selected": len(selected), "processed": 0, "reused": 0, "failed": 0}
    _atomic_json(cache_root / "progress.json", progress)

    pending: list[tuple[str, Mapping[str, Any], Mapping[str, Any], str]] = []
    terminal_count = 0
    # Resolve exact checkpoint reuse before launching workers.  This also makes
    # a resume run cheap when every patient is already complete.
    for patient_id in selected:
        if patient_id not in records_map:
            failed[patient_id] = {"patient_id": patient_id, "error_type": "MISSING_HOLTER_MAPPING", "message": "subject has Holter but no HOLTER record mapping"}
            _atomic_json(_cache_paths(cache_root, patient_id)[1], {"status": "FAILED", **failed[patient_id], "checkpoint_key": None, "failed_at_utc": _utc_now()})
            terminal_count += 1
            continue
        record = records_map[patient_id]
        key = checkpoint_key(patient_id, record, configs)
        cached = _load_cached_patient(cache_root, patient_id, key) if resume else None
        if cached is not None:
            completed[patient_id] = _normalise_windows(cached)
            reused_count += 1
            terminal_count += 1
        else:
            subject_row = subjects.loc[subjects["patient_id"].eq(patient_id)].iloc[0].to_dict()
            pending.append((patient_id, subject_row, record, key))
    progress.update({"processed": terminal_count, "reused": reused_count, "completed": len(completed), "failed": len(failed), "pending": len(pending), "updated_at_utc": _utc_now()})
    _atomic_json(cache_root / "progress.json", progress)

    def consume_result(patient_id: str, key: str, result: Any = None, error: Exception | None = None) -> None:
        nonlocal terminal_count
        if error is not None:
            failed[patient_id] = {"patient_id": patient_id, "error_type": type(error).__name__, "message": str(error)}
            _atomic_json(_cache_paths(cache_root, patient_id)[1], {"status": "FAILED", **failed[patient_id], "checkpoint_key": key, "failed_at_utc": _utc_now()})
        else:
            frame, metadata = result
            frame = _normalise_windows(frame)
            parquet_path, checkpoint_path = _cache_paths(cache_root, patient_id)
            _atomic_parquet(frame, parquet_path)
            completed[patient_id] = frame
            _atomic_json(checkpoint_path, {"status": "COMPLETE", "patient_id": patient_id, "checkpoint_key": key, "row_count": len(frame), "metadata": metadata, "completed_at_utc": _utc_now()})
        terminal_count += 1
        progress.update({"processed": terminal_count, "reused": reused_count, "completed": len(completed), "failed": len(failed), "pending": len(pending) - max(terminal_count - (len(selected) - len(pending)), 0), "last_patient_id": patient_id, "updated_at_utc": _utc_now()})
        _atomic_json(cache_root / "progress.json", progress)

    for start in range(0, len(pending), int(workers)):
        batch = pending[start : start + int(workers)]
        if int(workers) == 1:
            patient_id, subject_row, record, key = batch[0]
            try:
                value = process_patient(patient_id=patient_id, subject=subject_row, record=record, raw_root=raw, configs=configs)
                consume_result(patient_id, key, value)
            except Exception as exc:
                consume_result(patient_id, key, error=exc)
            continue
        with ThreadPoolExecutor(max_workers=min(int(workers), len(batch)), thread_name_prefix="full5m") as executor:
            futures = {
                executor.submit(process_patient, patient_id=patient_id, subject=subject_row, record=record, raw_root=raw, configs=configs): (patient_id, key)
                for patient_id, subject_row, record, key in batch
            }
            for future in as_completed(futures):
                patient_id, key = futures[future]
                try:
                    consume_result(patient_id, key, future.result())
                except Exception as exc:
                    consume_result(patient_id, key, error=exc)

    all_holter_ids = sorted(str(value) for value in subjects.loc[subjects["has_holter"].fillna(False).astype(bool), "patient_id"])
    # Patients omitted only by an explicit test limit are not treated as build
    # failures; a full invocation always selects all official Holter IDs.
    if set(selected) == set(all_holter_ids):
        missing = set(all_holter_ids) - set(completed) - set(failed)
        for patient_id in sorted(missing):
            failed[patient_id] = {"patient_id": patient_id, "error_type": "NOT_PROCESSED", "message": "patient did not reach a terminal checkpoint"}
    patient_features = _build_patient_features(subjects=subjects, records_map=records_map, completed=completed, failed=set(failed), pvc_threshold=0.20)
    features_dir = repo / FEATURE_SUBDIR
    shard_manifest = write_window_shards(cache_root=cache_root, patient_ids=sorted(completed), output_dir=features_dir, shard_rows=shard_rows)
    _atomic_parquet(patient_features, features_dir / "patient_features.parquet")

    status = build_patient_analysis_status(subjects, windows=None, patient_features=patient_features, pvc_threshold=0.20)
    analysis_dir = repo / ANALYSIS_SUBDIR
    analysis_dir.mkdir(parents=True, exist_ok=True)
    _atomic_parquet(status, analysis_dir / "patient_analysis_status.parquet")
    survival = build_survival_ready(subjects, patient_features, status, horizon_days=365)
    _atomic_parquet(survival, analysis_dir / "survival_ready.parquet")
    population = build_analysis_population_365(survival)
    population_audit = status.loc[:, [
        "patient_id", "model_365_included", "model_365_exclusion_reason",
        "endpoint_365_state", "primary_sinus_hrv_reason",
    ]].rename(columns={
        "model_365_included": "included",
        "model_365_exclusion_reason": "reason",
        "primary_sinus_hrv_reason": "signal_profile",
    })
    population_audit.to_csv(repo / "reports" / "analysis_population_365d.csv", index=False, encoding="utf-8")

    generated_paths = [path for base in (features_dir, analysis_dir) for path in base.rglob("*") if path.is_file()]
    generated_bytes = int(sum(path.stat().st_size for path in generated_paths))
    shard_sizes = [int(item["bytes"]) for collection in (shard_manifest["window_shards"], shard_manifest["feature_shards"]) for item in collection]
    theoretical = int(sum(len(complete_window_starts(float(records_map[pid]["sample_count"]) / float(records_map[pid]["sampling_frequency"]))) for pid in selected if pid in records_map))
    success_rows = int(sum(int(frame["feature_extraction_success"].fillna(False).sum()) for frame in completed.values()))
    qc_rows = int(sum(int(frame["qc_valid"].fillna(False).sum()) for frame in completed.values()))
    status_name = "COMPLETE" if not failed and set(selected) == set(all_holter_ids) else "DEGRADED"
    report = {
        "status": status_name,
        "generated_at_utc": _utc_now(),
        "official_subjects": int(len(subjects)),
        "holter_records": int(len(all_holter_ids)),
        "selected_holters": int(len(selected)),
        "completed_holter": int(len(completed)),
        "reused_checkpoints": int(reused_count),
        "failed_holter": int(len(failed)),
        "failures": list(failed.values()),
        "theoretical_window_count": theoretical,
        "feature_extraction_success_rows": success_rows,
        "qc_valid_rows": qc_rows,
        "patient_feature_rows": int(len(patient_features)),
        "patient_status_rows": int(len(status)),
        "survival_ready_rows": int(len(survival)),
        "analysis_population_365_rows": int(len(population)),
        "generated_data_bytes": generated_bytes,
        "maximum_shard_bytes": max(shard_sizes, default=0),
        "window_shards": shard_manifest["window_shards"],
        "feature_shards": shard_manifest["feature_shards"],
        "checkpoint": {
            "preprocessing_hash": configs["preprocessing_hash"],
            "feature_schema_version": configs["feature_schema_version"],
            "window_hash": configs["window_hash"],
            "key_fields": ["patient_id", "official_source_sha256", "official_header_sha256", "preprocessing_hash", "feature_schema_version", "window_hash"],
        },
        "contract": {
            "window_start_formula": "60 + 300*k",
            "window_length_sec": 300,
            "stride_sec": 300,
            "complete_windows_only": True,
            "labels_or_outcomes_in_extractor": False,
            "reader": "wfdb.rdrecord",
            "reader_channels": [0],
            "reader_padding": False,
            "feature_count": 20,
            "feature_dtype": "float64",
            "compression": "zstd",
            "raw_waveform_or_cache_in_data": False,
        },
    }
    _atomic_json(cache_root / "failures.json", {"status": status_name, "failures": list(failed.values())})
    _write_reports(repo=repo, subjects=subjects, records_map=records_map, patient_features=patient_features, status=status, build_report=report)
    progress.update({"status": status_name, "completed": len(completed), "failed": len(failed), "finished_at_utc": _utc_now()})
    _atomic_json(cache_root / "progress.json", progress)
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=REPO)
    parser.add_argument("--raw-root", type=Path, default=None)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--max-patients", type=int, default=None)
    parser.add_argument("--patient-id", action="append", dest="patient_ids", default=None)
    parser.add_argument("--shard-rows", type=int, default=DEFAULT_SHARD_ROWS)
    parser.add_argument("--workers", type=int, default=1, help="bounded patient workers; each worker reads one exact window at a time")
    args = parser.parse_args(argv)
    if args.workers < 1:
        parser.error("--workers must be positive")
    try:
        result = build(
            args.repo.resolve(),
            args.raw_root.resolve() if args.raw_root else None,
            resume=not args.no_resume,
            max_patients=args.max_patients,
            patient_ids=args.patient_ids,
            shard_rows=args.shard_rows,
            workers=args.workers,
        )
    except Exception as exc:
        print(f"FULL 5-MIN BUILD ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({key: result[key] for key in ("status", "official_subjects", "holter_records", "completed_holter", "failed_holter", "theoretical_window_count", "generated_data_bytes")}, ensure_ascii=False))
    return 0 if result["status"] == "COMPLETE" else 1


if __name__ == "__main__":
    raise SystemExit(main())
