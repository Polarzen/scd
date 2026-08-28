#!/usr/bin/env python3
"""Finalize a complete MUSIC Phase 4 build into a portable integrity package.

The finalizer is deliberately separate from the waveform builder.  It only
operates after ``FULL_COHORT_BUILD.json`` reports ``COMPLETE`` and therefore
never needs ``MUSIC_RAW_DIR``.  Its two integrity outputs are kept out of the
hash set while they are assembled; the build manifest is then included in the
compact hash list without hashing the compact list itself.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import platform
import re
import sys
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
PHASE2_COMMIT = "b6ab167f6eb7873d8a7d024f9ab996aa4a0acf8a"
PHASE3_COMMIT = "c991cebdec6605a31d2c8c110cc8947a9cd08396"
PHASE4_COMMIT = "PENDING"
MUSIC_VERSION = "1.0.1"
MUSIC_DOI = "10.13026/z3m7-rf58"
MAX_DATA_BYTES = 100 * 1024 * 1024
MAX_DATA_FILE_BYTES = 25 * 1024 * 1024
RAW_CANDIDATE_SUFFIXES = {".dat", ".mat", ".wav", ".wfdb", ".npy", ".npz", ".zip", ".tar", ".gz"}
INTEGRITY_OUTPUTS = {
    "data/integrity/compact_sha256.txt",
    "data/integrity/build_manifest.json",
}
FINALIZATION_REPORTS = {
    "reports/PHASE4_SIZE_REPORT.json",
    "reports/PHASE4_SIZE_REPORT.md",
    "reports/github_handoff.json",
}


class FinalizationError(RuntimeError):
    """Raised when a build is not safe to finalize."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise FinalizationError(f"invalid JSON: {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise FinalizationError(f"JSON object required: {path}")
    return value


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.finalize.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.finalize.tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def _complete_build_report(repo: Path) -> dict[str, Any]:
    path = repo / "reports" / "FULL_COHORT_BUILD.json"
    if not path.is_file():
        raise FinalizationError("reports/FULL_COHORT_BUILD.json is missing; run the complete Phase 4 build first")
    report = _read_json(path)
    status = str(report.get("status", "")).upper()
    if status != "COMPLETE":
        raise FinalizationError(f"Phase 4 finalization requires build status COMPLETE (found {status or 'missing'})")
    return report


def _relative(path: Path, repo: Path) -> str:
    return path.relative_to(repo).as_posix()


def _is_compact_candidate(path: Path, repo: Path) -> bool:
    relative = _relative(path, repo)
    if relative in INTEGRITY_OUTPUTS:
        return False
    parts = {part.lower() for part in path.relative_to(repo).parts}
    if parts.intersection({"build_cache", ".tmp", "tmp", "temp", "raw", "cache", "__pycache__"}):
        return False
    if path.suffix.lower() in {".tmp", ".pyc"}:
        return False
    if relative.startswith("data/") and path.suffix.lower() in RAW_CANDIDATE_SUFFIXES:
        return False
    return True


def compact_files(repo: Path) -> list[Path]:
    """Return compact data and source/config files covered by finalization."""

    paths: set[Path] = set()
    data_root = repo / "data"
    if data_root.is_dir():
        paths.update(path for path in data_root.rglob("*") if path.is_file() and _is_compact_candidate(path, repo))
    for root_name in ("config", "src", "scripts", "tests"):
        root = repo / root_name
        if root.is_dir():
            paths.update(path for path in root.rglob("*") if path.is_file() and _is_compact_candidate(path, repo))
    workflow_root = repo / ".github" / "workflows"
    if workflow_root.is_dir():
        paths.update(path for path in workflow_root.rglob("*") if path.is_file() and _is_compact_candidate(path, repo))
    reports_root = repo / "reports"
    if reports_root.is_dir():
        paths.update(
            path for path in reports_root.rglob("*")
            if path.is_file()
            and _relative(path, repo) not in FINALIZATION_REPORTS
            and _is_compact_candidate(path, repo)
        )
    for name in (
        ".gitattributes", ".gitignore", "requirements.txt", "requirements-lock.txt",
        "README.md", "DATASET.md", "REPRODUCIBILITY.md",
    ):
        path = repo / name
        if path.is_file() and _is_compact_candidate(path, repo):
            paths.add(path)
    return sorted(paths, key=lambda item: _relative(item, repo))


def _package_versions(repo: Path) -> dict[str, str]:
    versions: dict[str, str] = {}
    path = repo / "requirements-lock.txt"
    if not path.is_file():
        return versions
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        match = re.match(r"^([A-Za-z0-9_.-]+)==([^;\s]+)", line)
        if match:
            versions[match.group(1).lower().replace("_", "-")] = match.group(2)
    return versions


def _runtime_versions() -> dict[str, str]:
    names = ("numpy", "scipy", "pandas", "pyarrow", "wfdb", "scikit-learn", "joblib")
    versions: dict[str, str] = {"python": platform.python_version()}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "NOT_INSTALLED"
    return versions


def _counts(repo: Path, report: dict[str, Any]) -> dict[str, int]:
    counts: dict[str, int] = {}
    report_names = {
        "official_subjects": "official_subjects",
        "holter_records": "holter_records",
        "completed_holter": "completed_holter",
        "theoretical_window_count": "theoretical_window_count",
        "feature_extraction_success_rows": "feature_extraction_success_rows",
        "qc_valid_rows": "qc_valid_rows",
        "patient_feature_rows": "patient_feature_rows",
        "patient_status_rows": "patient_status_rows",
        "survival_ready_rows": "survival_ready_rows",
        "analysis_population_365_rows": "analysis_population_365_rows",
    }
    for target, source in report_names.items():
        if isinstance(report.get(source), (int, float)):
            counts[target] = int(report[source])
    paths = {
        "subjects": repo / "data/cohort/subjects.parquet",
        "records": repo / "data/cohort/records.parquet",
        "patient_features": repo / "data/features/full_5min/patient_features.parquet",
        "patient_status": repo / "data/analysis/patient_analysis_status.parquet",
        "survival_ready": repo / "data/analysis/survival_ready.parquet",
    }
    for name, path in paths.items():
        if path.is_file():
            try:
                counts[name] = int(len(pd.read_parquet(path, columns=["patient_id"])))
            except Exception:
                # The completed report remains the authoritative fallback for
                # a count if an optional reader dependency is unavailable.
                pass
    if "subjects" in counts:
        counts["official_subjects"] = counts["subjects"]
    records_path = paths["records"]
    if records_path.is_file():
        try:
            records = pd.read_parquet(records_path, columns=["record_type"])
            counts["record_count"] = int(len(records))
            counts["holter_records"] = int(records["record_type"].astype("string").eq("HOLTER").sum())
            counts["high_resolution_records"] = int(records["record_type"].astype("string").eq("HIGH_RESOLUTION").sum())
        except Exception:
            pass
    manifest = repo / "data/features/full_5min/manifest.json"
    if manifest.is_file():
        payload = _read_json(manifest)
        if isinstance(payload.get("window_row_count"), int):
            counts["full_window_rows"] = int(payload["window_row_count"])
        for key, target in (("window_shards", "window_shard_count"), ("feature_shards", "feature_shard_count")):
            if isinstance(payload.get(key), list):
                counts[target] = len(payload[key])
    return dict(sorted(counts.items()))


def _endpoint_state_counts(report: dict[str, Any]) -> dict[str, dict[str, int]]:
    flow = report.get("cohort_flow_report", {})
    values = flow.get("endpoint_states", {}) if isinstance(flow, dict) else {}
    result: dict[str, dict[str, int]] = {}
    if isinstance(values, dict):
        for horizon, states in values.items():
            if isinstance(states, dict):
                result[str(horizon)] = {str(state): int(count) for state, count in sorted(states.items()) if isinstance(count, (int, float))}
    return dict(sorted(result.items()))


def _data_size_report(repo: Path, report: dict[str, Any], counts: dict[str, int]) -> dict[str, Any]:
    data_root = repo / "data"
    files = sorted((path for path in data_root.rglob("*") if path.is_file()), key=lambda item: _relative(item, repo)) if data_root.is_dir() else []
    records = [{"path": _relative(path, repo), "bytes": int(path.stat().st_size)} for path in files]
    total = int(sum(item["bytes"] for item in records))
    maximum = max((item["bytes"] for item in records), default=0)
    category_bytes = {
        "cohort": sum(item["bytes"] for item in records if item["path"].startswith("data/cohort/")),
        "phase3_legacy": sum(item["bytes"] for item in records if item["path"].startswith("data/features/legacy_120s/")),
        "phase4_full_5min": sum(item["bytes"] for item in records if item["path"].startswith("data/features/full_5min/")),
        "validation": sum(item["bytes"] for item in records if item["path"].startswith("data/validation/")),
        "analysis": sum(item["bytes"] for item in records if item["path"].startswith("data/analysis/")),
        "source_exact": sum(item["bytes"] for item in records if item["path"].startswith("data/source_exact/")),
        "integrity": sum(item["bytes"] for item in records if item["path"].startswith("data/integrity/")),
    }
    repository_compact_bytes = sum(path.stat().st_size for path in compact_files(repo))
    payload: dict[str, Any] = {
        "schema_version": 1,
        "phase": 4,
        "status": "PASS" if total < MAX_DATA_BYTES and maximum < MAX_DATA_FILE_BYTES else "FAIL",
        "limits": {"repository_data_bytes_lt": MAX_DATA_BYTES, "individual_data_file_bytes_lt": MAX_DATA_FILE_BYTES},
        "repository_data_bytes": total,
        "maximum_data_file_bytes": maximum,
        "data_file_count": len(records),
        "repository_compact_bytes": int(repository_compact_bytes),
        "category_bytes": {key: int(value) for key, value in category_bytes.items()},
        "largest_30_files": sorted(records, key=lambda item: item["bytes"], reverse=True)[:30],
        "files": records,
        "counts": counts,
        "build_status": report.get("status"),
        "generated_at_utc": _utc_now(),
    }
    return payload


def _size_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Phase 4 size report",
        "",
        f"- Status: **{payload['status']}**",
        f"- Repository data bytes: **{payload['repository_data_bytes']}** (limit `< {payload['limits']['repository_data_bytes_lt']}`)",
        f"- Maximum individual data file: **{payload['maximum_data_file_bytes']}** bytes (limit `< {payload['limits']['individual_data_file_bytes_lt']}`)",
        f"- Data files: **{payload['data_file_count']}**",
        f"- Compact repository surface: **{payload['repository_compact_bytes']}** bytes",
        "",
        "## Largest 30 files",
        "",
        "| File | Bytes |",
        "|---|---:|",
    ]
    lines.extend(f"| `{item['path']}` | {item['bytes']} |" for item in payload["largest_30_files"])
    return "\n".join(lines) + "\n"


def _handoff(repo: Path, report: dict[str, Any], counts: dict[str, int], size: dict[str, Any]) -> dict[str, Any]:
    ready = size["status"] == "PASS" and str(report.get("status", "")).upper() == "COMPLETE"
    return {
        "schema_version": 1,
        "phase": "P4-D",
        "status": "READY" if ready else "BLOCKED",
        "repository_ready": ready,
        "compact_data_ready": ready,
        "raw_music_required_for_normal_ci": False,
        "default_endpoint": 365,
        "default_profile": "all20",
        "default_model": "extratrees",
        "default_seed": 42,
        "music_version": MUSIC_VERSION,
        "source_doi": MUSIC_DOI,
        "phase2_commit": PHASE2_COMMIT,
        "phase2_base_commit": PHASE2_COMMIT,
        "phase3_commit": PHASE3_COMMIT,
        "phase3_base_commit": PHASE3_COMMIT,
        "phase4_commit": PHASE4_COMMIT,
        "source": {
            "dataset_name": "MUSIC (Sudden Cardiac Death in Chronic Heart Failure)",
            "music_version": MUSIC_VERSION,
            "source_doi": MUSIC_DOI,
            "doi": MUSIC_DOI,
        },
        "lineage": {
            "phase2_commit": PHASE2_COMMIT,
            "phase3_commit": PHASE3_COMMIT,
            "phase4_commit": PHASE4_COMMIT,
        },
        "phases": {
            "phase2": {"commit": PHASE2_COMMIT},
            "phase3": {"commit": PHASE3_COMMIT},
            "phase4": {"commit": PHASE4_COMMIT},
        },
        "counts": counts,
        "endpoint_state_counts": _endpoint_state_counts(report),
        "build_status": report.get("status"),
        "size_report": "reports/PHASE4_SIZE_REPORT.json",
        "integrity_manifest": "data/integrity/build_manifest.json",
        "compact_hashes": "data/integrity/compact_sha256.txt",
        "raw_waveform_required_for_github_validation": False,
        "workflow_files": [
            ".github/workflows/ci.yml",
            ".github/workflows/model-validation.yml",
            ".github/workflows/repeated-cv.yml",
            ".github/workflows/endpoint-sensitivity.yml",
        ],
        "source_of_truth_files": [
            "data/cohort/subjects.parquet",
            "data/cohort/records.parquet",
            "data/cohort/provenance.parquet",
            "data/features/full_5min/",
            "config/",
            "src/",
        ],
        "latest_build_manifest": "data/integrity/build_manifest.json",
        "known_limitations": [
            "Raw ECG waveforms are not included; signal preprocessing changes require a local full rebuild.",
            "The 5-minute records are baseline Holter windows, not ECG immediately preceding SCD.",
            "Binary models exclude censored, competing-event, unknown-endpoint, and primary sinus-HRV-ineligible patients without removing them from the cohort facts.",
        ],
        "generated_at_utc": _utc_now(),
    }


def _build_manifest(
    repo: Path,
    report: dict[str, Any],
    counts: dict[str, int],
    size: dict[str, Any],
) -> tuple[dict[str, Any], list[Path]]:
    paths = compact_files(repo)
    files = [
        {"path": _relative(path, repo), "bytes": int(path.stat().st_size), "sha256": _sha256(path)}
        for path in paths
    ]
    hashes = {item["path"]: item["sha256"] for item in files}
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "phase": "P4-D",
        "status": "FINALIZED",
        "music_version": MUSIC_VERSION,
        "source_doi": MUSIC_DOI,
        "phase2_commit": PHASE2_COMMIT,
        "phase2_base_commit": PHASE2_COMMIT,
        "phase3_commit": PHASE3_COMMIT,
        "phase3_base_commit": PHASE3_COMMIT,
        "phase4_commit": PHASE4_COMMIT,
        "source": {
            "dataset_name": "MUSIC (Sudden Cardiac Death in Chronic Heart Failure)",
            "music_version": MUSIC_VERSION,
            "source_doi": MUSIC_DOI,
            "doi": MUSIC_DOI,
        },
        "lineage": {
            "phase2_commit": PHASE2_COMMIT,
            "phase3_commit": PHASE3_COMMIT,
            "phase4_commit": PHASE4_COMMIT,
        },
        "phases": {
            "phase2": {"commit": PHASE2_COMMIT},
            "phase3": {"commit": PHASE3_COMMIT},
            "phase4": {"commit": PHASE4_COMMIT},
        },
        "package_versions": _package_versions(repo),
        "runtime_versions": _runtime_versions(),
        "python_version": platform.python_version(),
        "software_platform": platform.platform(),
        "feature_schema_version": "full_5min_v2",
        "preprocessing_hash": _sha256(repo / "config" / "full_preprocessing.yaml"),
        "window_config_hash": _sha256(repo / "config" / "full_5min_windows.yaml"),
        "endpoint_schema_version": "music_endpoint_v1",
        "model_pipeline_version": "music_full_nested_cv_v1",
        "counts": counts,
        "endpoint_state_counts": _endpoint_state_counts(report),
        "size_report": {
            "repository_data_bytes": size["repository_data_bytes"],
            "maximum_data_file_bytes": size["maximum_data_file_bytes"],
        },
        # Keep both forms machine-readable for consumers that need a compact
        # lookup and consumers that want byte counts beside every digest.
        "hashes": hashes,
        "files": files,
        "compact_hashes_exclude": sorted(INTEGRITY_OUTPUTS),
        "build_report": "reports/FULL_COHORT_BUILD.json",
        "generated_at_utc": _utc_now(),
        "build_timestamp": report.get("generated_at_utc"),
        "build_status": report.get("status"),
    }
    return manifest, paths


def _compact_text(repo: Path, paths: Iterable[Path], build_manifest_path: Path) -> str:
    selected = list(paths) + [build_manifest_path]
    lines = [f"{_sha256(path)}  {_relative(path, repo)}" for path in sorted(selected, key=lambda item: _relative(item, repo))]
    return "\n".join(lines) + "\n"


def finalize_phase4(repo: Path = REPO) -> dict[str, Any]:
    """Finalize a complete build and return the generated build manifest."""

    repo = Path(repo).resolve()
    report = _complete_build_report(repo)

    # Import lazily so an incomplete run can fail before reading any raw path
    # or creating any finalization output.
    from scripts.verify_all import verify_phase4

    try:
        phase4_errors = verify_phase4(repo)
    except Exception as exc:
        raise FinalizationError(f"Phase 4 integrity verifier could not complete: {exc}") from exc
    if phase4_errors:
        raise FinalizationError("Phase 4 integrity checks failed: " + "; ".join(phase4_errors[:8]))

    counts = _counts(repo, report)
    size = _data_size_report(repo, report, counts)
    if size["status"] != "PASS":
        raise FinalizationError(
            "compact data size limits failed: "
            f"total={size['repository_data_bytes']} bytes, "
            f"maximum_file={size['maximum_data_file_bytes']} bytes"
        )
    reports = repo / "reports"
    _write_json(reports / "PHASE4_SIZE_REPORT.json", size)
    _write_text(reports / "PHASE4_SIZE_REPORT.md", _size_markdown(size))
    _write_json(reports / "github_handoff.json", _handoff(repo, report, counts, size))

    manifest, paths = _build_manifest(repo, report, counts, size)
    integrity = repo / "data" / "integrity"
    build_manifest_path = integrity / "build_manifest.json"
    _write_json(build_manifest_path, manifest)
    _write_text(integrity / "compact_sha256.txt", _compact_text(repo, paths, build_manifest_path))
    # The integrity outputs themselves live under data/.  Refresh the size
    # report after writing them so its file inventory describes the finalized
    # package rather than the pre-finalization working tree.
    final_size = _data_size_report(repo, report, counts)
    manifest["size_report"] = {
        "repository_data_bytes": final_size["repository_data_bytes"],
        "maximum_data_file_bytes": final_size["maximum_data_file_bytes"],
    }
    _write_json(build_manifest_path, manifest)
    _write_text(integrity / "compact_sha256.txt", _compact_text(repo, paths, build_manifest_path))
    final_size = _data_size_report(repo, report, counts)
    _write_json(reports / "PHASE4_SIZE_REPORT.json", final_size)
    _write_text(reports / "PHASE4_SIZE_REPORT.md", _size_markdown(final_size))
    _write_json(reports / "github_handoff.json", _handoff(repo, report, counts, final_size))
    return manifest


# Short alias for callers that use the action name as the function name.
finalize = finalize_phase4


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=REPO)
    args = parser.parse_args(argv)
    try:
        result = finalize_phase4(args.repo)
    except FinalizationError as exc:
        print(f"PHASE 4 FINALIZATION ERROR: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({"status": result["status"], "phase": result["phase"], "counts": result["counts"]}, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
