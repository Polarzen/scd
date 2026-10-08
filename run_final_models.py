from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import importlib.metadata
import json
import os
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SEEDS_PER_BATCH = 10
EXPECTED_SEEDS = tuple(range(100))
FIXED_COMBO = "pro_bnp_nsvt"
FIXED_WORKERS_PER_BATCH = 4
EXPECTED_SELECTOR_MEDIAN = 0.706977
EXPECTED_FIXED_MEDIAN = 0.700726

SOURCE_PATHS = (
    "analysis/validate_nested_selector.py",
    "analysis/validate_p5_combos.py",
    "analysis/validate_p3.py",
    "analysis/validate_p4_single_additions.py",
    "analysis/validate_compact_candidates.py",
    "src/endpoints.py",
    "requirements-lock.txt",
)
INPUT_PATHS = (
    "reports/FULL_COHORT_BUILD.json",
    "data/cohort/subjects.parquet",
    "data/features/full_5min/patient_features.parquet",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fingerprint_paths(project_root: Path, relative_paths: tuple[str, ...]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for relative in relative_paths:
        path = project_root / Path(relative)
        if not path.is_file():
            raise FileNotFoundError(f"required project file is missing: {path}")
        result[relative] = {"sha256": sha256_file(path), "bytes": path.stat().st_size}
    return result


def fingerprint_snapshot(project_root: Path) -> dict[str, dict[str, dict[str, Any]]]:
    return {
        "source_files": fingerprint_paths(project_root, SOURCE_PATHS),
        "input_files": fingerprint_paths(project_root, INPUT_PATHS),
    }


def check_build_complete(project_root: Path) -> dict[str, Any]:
    report_path = project_root / "reports" / "FULL_COHORT_BUILD.json"
    if not report_path.is_file():
        raise RuntimeError(f"raw build report is missing: {report_path}")
    try:
        report = json.loads(report_path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"cannot read raw build report {report_path}: {exc}") from exc
    if str(report.get("status", "")).upper() != "COMPLETE":
        raise RuntimeError(
            f"raw build is not COMPLETE (status={report.get('status')!r}); refusing to start final100"
        )
    if int(report.get("failed_holter", -1)) != 0:
        raise RuntimeError(f"raw build reports failed Holters: {report.get('failed_holter')!r}")
    if report.get("selected_holters") != report.get("completed_holter"):
        raise RuntimeError(
            "raw build selected/completed Holter counts differ: "
            f"{report.get('selected_holters')!r} vs {report.get('completed_holter')!r}"
        )
    return report


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f".{os.getpid()}.tmp")
    temp.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temp, path)


def atomic_csv(frame: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f".{os.getpid()}.tmp")
    frame.to_csv(temp, index=False, lineterminator="\n")
    os.replace(temp, path)


class Progress:
    def __init__(self, path: Path, state: dict[str, Any]):
        self.path = path
        self.state = state
        self.lock = threading.Lock()
        self.write()

    def write(self) -> None:
        atomic_json(self.path, self.state)

    def set_stage(self, stage: str, status: str, **fields: Any) -> None:
        with self.lock:
            self.state["stage"] = stage
            self.state["status"] = status
            self.state["updated_at_utc"] = utc_now()
            self.state.update(fields)
            self.write()

    def set_task(self, task_id: str, status: str, **fields: Any) -> None:
        with self.lock:
            row = self.state["tasks"].setdefault(task_id, {})
            row.update(fields)
            row["status"] = status
            row["updated_at_utc"] = utc_now()
            self.state["updated_at_utc"] = utc_now()
            self.write()


def task_environment() -> dict[str, str]:
    env = os.environ.copy()
    env.update(
        {
            "PYTHONUTF8": "1",
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "LOKY_MAX_CPU_COUNT": str(FIXED_WORKERS_PER_BATCH),
        }
    )
    return env


def _task_log_path(output_root: Path, task_id: str) -> Path:
    return output_root / "logs" / f"{task_id}.log"


def run_task(
    *,
    project_root: Path,
    progress: Progress,
    task_id: str,
    command: list[str],
    output_dir: Path | None,
    log_path: Path,
) -> dict[str, Any]:
    started = time.monotonic()
    record: dict[str, Any] = {
        "command": command,
        "cwd": str(project_root),
        "output_dir": str(output_dir) if output_dir else None,
        "log": str(log_path),
        "started_at_utc": utc_now(),
    }
    progress.set_task(task_id, "RUNNING", **record)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=False)
    try:
        with log_path.open("w", encoding="utf-8", newline="") as log_stream:
            completed = subprocess.run(
                command,
                cwd=project_root,
                env=task_environment(),
                stdout=log_stream,
                stderr=subprocess.STDOUT,
                check=False,
                shell=False,
            )
        result = {
            **record,
            "return_code": int(completed.returncode),
            "finished_at_utc": utc_now(),
            "duration_seconds": round(time.monotonic() - started, 3),
        }
        status = "COMPLETE" if completed.returncode == 0 else "FAILED"
        progress.set_task(task_id, status, **result)
        return {"task_id": task_id, **result, "status": status}
    except Exception as exc:
        result = {
            **record,
            "return_code": None,
            "error": f"{type(exc).__name__}: {exc}",
            "finished_at_utc": utc_now(),
            "duration_seconds": round(time.monotonic() - started, 3),
        }
        progress.set_task(task_id, "FAILED", **result)
        return {"task_id": task_id, **result, "status": "FAILED"}


def run_task_group(
    *,
    project_root: Path,
    progress: Progress,
    specs: list[dict[str, Any]],
    max_parallel: int,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_parallel) as executor:
        futures = {
            executor.submit(
                run_task,
                project_root=project_root,
                progress=progress,
                task_id=spec["task_id"],
                command=spec["command"],
                output_dir=spec.get("output_dir"),
                log_path=spec["log_path"],
            ): spec["task_id"]
            for spec in specs
        }
        for future in concurrent.futures.as_completed(futures):
            task_id = futures[future]
            try:
                results.append(future.result())
            except Exception as exc:  # defensive: a worker reports normal launch failures itself
                results.append(
                    {"task_id": task_id, "status": "FAILED", "error": f"{type(exc).__name__}: {exc}"}
                )
    return sorted(results, key=lambda row: row["task_id"])


def _batch_spec(
    *,
    output_root: Path,
    task_id: str,
    script: str,
    command_prefix: list[str],
    seed_start: int,
) -> dict[str, Any]:
    seed_stop = seed_start + SEEDS_PER_BATCH - 1
    family = "selector" if "nested_selector" in script else "fixed_p5"
    batch_name = f"batch_{seed_start:02d}_{seed_stop:02d}"
    output_dir = output_root / family / "batches" / batch_name
    argv = [
        sys.executable,
        script,
        *command_prefix,
        "--seed-start",
        str(seed_start),
        "--seed-stop",
        str(seed_stop),
        "--output-dir",
        str(output_dir),
    ]
    return {
        "task_id": task_id,
        "command": argv,
        "output_dir": output_dir,
        "log_path": _task_log_path(output_root, task_id),
    }


def _check_batch_files(output_root: Path, family: str) -> None:
    import pandas as pd

    expected_files = (
        ("per_seed.csv", "oof.csv", "folds.csv", "selector_contract.json")
        if family == "selector"
        else ("per_seed.csv", "oof.parquet", "folds.csv", "combo.json")
    )
    batch_dirs = sorted((output_root / family / "batches").glob("batch_*"))
    if len(batch_dirs) != 10:
        raise RuntimeError(f"{family}: expected 10 batch directories; found {len(batch_dirs)}")
    seen_seeds: list[int] = []
    for batch_dir in batch_dirs:
        missing = [name for name in expected_files if not (batch_dir / name).is_file()]
        if missing:
            raise RuntimeError(f"{batch_dir} is missing required outputs: {missing}")
        per_seed = pd.read_csv(batch_dir / "per_seed.csv")
        if "seed" not in per_seed:
            raise RuntimeError(f"{batch_dir / 'per_seed.csv'} lacks seed column")
        seeds = per_seed["seed"].astype(int).tolist()
        if len(seeds) != SEEDS_PER_BATCH or len(set(seeds)) != SEEDS_PER_BATCH:
            raise RuntimeError(f"{batch_dir} has invalid seed rows: {seeds}")
        expected_batch = list(range(int(batch_dir.name.split("_")[1]), int(batch_dir.name.split("_")[2]) + 1))
        if sorted(seeds) != expected_batch:
            raise RuntimeError(f"{batch_dir} seed range differs: expected {expected_batch}, got {sorted(seeds)}")
        seen_seeds.extend(seeds)
    if sorted(seen_seeds) != list(EXPECTED_SEEDS):
        raise RuntimeError(f"{family} batch coverage is not exactly seeds 0..99")


def merge_fixed_batches(output_root: Path) -> Path:
    import pandas as pd

    batch_dirs = sorted((output_root / "fixed_p5" / "batches").glob("batch_*"))
    if len(batch_dirs) != 10:
        raise RuntimeError(f"fixed P5: expected 10 batch dirs before merge; found {len(batch_dirs)}")
    metadata: list[dict[str, Any]] = []
    for folder in batch_dirs:
        metadata.append(json.loads((folder / "combo.json").read_text(encoding="utf-8")))
    canonical = json.dumps(metadata[0], ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    if any(json.dumps(item, ensure_ascii=False, sort_keys=True, separators=(",", ":")) != canonical for item in metadata[1:]):
        raise RuntimeError("fixed P5 batch combo.json metadata differ; refusing to merge")
    if metadata[0].get("combo") != FIXED_COMBO or int(metadata[0].get("raw_feature_count", -1)) != 24:
        raise RuntimeError("fixed P5 metadata does not describe the required 24-input pro_bnp_nsvt combo")
    if len(metadata[0].get("features", [])) != 24:
        raise RuntimeError("fixed P5 combo metadata does not have an exact 24-feature list")

    per_frames = [pd.read_csv(folder / "per_seed.csv") for folder in batch_dirs]
    oof_frames = [pd.read_parquet(folder / "oof.parquet") for folder in batch_dirs]
    fold_frames = [pd.read_csv(folder / "folds.csv") for folder in batch_dirs]
    for label, frames in (("per_seed", per_frames), ("oof", oof_frames), ("folds", fold_frames)):
        columns = list(frames[0].columns)
        if any(list(frame.columns) != columns for frame in frames[1:]):
            raise RuntimeError(f"fixed P5 {label} batch schemas differ")

    per_seed = pd.concat(per_frames, ignore_index=True).sort_values("seed", kind="stable").reset_index(drop=True)
    if per_seed["seed"].astype(int).tolist() != list(EXPECTED_SEEDS):
        raise RuntimeError("fixed P5 per_seed merge is not exactly seeds 0..99")
    oof = pd.concat(oof_frames, ignore_index=True).sort_values(["seed", "patient_id"], kind="stable").reset_index(drop=True)
    folds = pd.concat(fold_frames, ignore_index=True).sort_values(["seed", "outer_fold"], kind="stable").reset_index(drop=True)

    merged_root = output_root / "fixed_p5" / "merged_only"
    merged_dir = merged_root / FIXED_COMBO
    if merged_root.exists() and any(merged_root.iterdir()):
        raise RuntimeError(f"refusing to overwrite non-empty merged-only root: {merged_root}")
    merged_dir.mkdir(parents=True, exist_ok=True)
    atomic_csv(per_seed, merged_dir / "per_seed.csv")
    atomic_csv(folds, merged_dir / "folds.csv")
    temp_parquet = merged_dir / "oof.parquet.tmp"
    oof.to_parquet(temp_parquet, index=False, compression="zstd")
    os.replace(temp_parquet, merged_dir / "oof.parquet")
    atomic_json(merged_dir / "combo.json", metadata[0])
    return merged_root


def package_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for name in ("numpy", "pandas", "scikit-learn", "joblib", "pyarrow"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the fixed final100 selector and P5 reproductions in bounded batches.")
    parser.add_argument("--project-root", type=Path, required=True, help="fresh-run source/data directory")
    parser.add_argument("--output-root", type=Path, required=True, help="result directory outside project-root")
    parser.add_argument("--max-parallel", type=int, default=4, help="maximum simultaneous batch subprocesses (1..4)")
    parser.add_argument("--workers-per-batch", type=int, default=FIXED_WORKERS_PER_BATCH, help="fixed at 4 to cap total joblib workers at 16")
    parser.add_argument("--tasks", choices=("selector", "fixed", "both"), default="both")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not 1 <= args.max_parallel <= 4:
        raise SystemExit("--max-parallel must be between 1 and 4")
    if args.workers_per_batch != FIXED_WORKERS_PER_BATCH:
        raise SystemExit("--workers-per-batch is fixed at 4 by the final100 resource contract")
    project_root = args.project_root.resolve(strict=True)
    if not project_root.is_dir():
        raise SystemExit(f"project root is not a directory: {project_root}")
    output_root = args.output_root.resolve()
    if output_root == project_root or project_root in output_root.parents:
        raise SystemExit("output-root must be outside project-root (results cannot go into fresh-run data)")
    if output_root.exists() and any(output_root.iterdir()):
        raise SystemExit(f"output-root is not empty; default runner does not resume or reuse outputs: {output_root}")

    try:
        build_report = check_build_complete(project_root)
        snapshot = fingerprint_snapshot(project_root)
    except Exception as exc:
        raise SystemExit(str(exc)) from exc

    output_root.mkdir(parents=True, exist_ok=True)
    logs_root = output_root / "logs"
    logs_root.mkdir(parents=True, exist_ok=True)
    selected_tasks = [] if args.tasks == "both" else [args.tasks]
    if args.tasks == "both":
        selected_tasks = ["selector", "fixed"]
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "status": "PREPARED",
        "created_at_utc": utc_now(),
        "project_root": str(project_root),
        "output_root": str(output_root),
        "tasks_requested": selected_tasks,
        "target": {
            "selector": {"seeds": "0..99", "cv_auc_median_expected_6dp": EXPECTED_SELECTOR_MEDIAN},
            "fixed_p5": {"combo": FIXED_COMBO, "seeds": "0..99", "cv_auc_median_expected_6dp": EXPECTED_FIXED_MEDIAN},
        },
        "batching": {
            "seeds_per_batch": SEEDS_PER_BATCH,
            "max_parallel_batch_subprocesses": int(args.max_parallel),
            "workers_per_batch": FIXED_WORKERS_PER_BATCH,
            "max_joblib_workers": int(args.max_parallel) * FIXED_WORKERS_PER_BATCH,
        },
        "python": {"executable": sys.executable, "version": sys.version, "packages": package_versions()},
        "build_report": {
            "relative_path": "reports/FULL_COHORT_BUILD.json",
            "status": build_report.get("status"),
            "selected_holters": build_report.get("selected_holters"),
            "completed_holter": build_report.get("completed_holter"),
            "failed_holter": build_report.get("failed_holter"),
        },
        "fingerprints_before": snapshot,
        "commands": [],
    }
    atomic_json(output_root / "run_manifest.json", manifest)
    state = {
        "schema_version": 1,
        "status": "RUNNING",
        "stage": "PREPARED",
        "created_at_utc": utc_now(),
        "updated_at_utc": utc_now(),
        "tasks": {},
        "failures": [],
    }
    progress = Progress(output_root / "progress.json", state)

    try:
        if "selector" in selected_tasks:
            batch_specs: list[dict[str, Any]] = []
            for start in range(0, 100, SEEDS_PER_BATCH):
                batch_specs.append(
                    _batch_spec(
                        output_root=output_root,
                        task_id=f"selector_{start:02d}_{start + 9:02d}",
                        script="analysis/validate_nested_selector.py",
                        command_prefix=["batch"],
                        seed_start=start,
                    )
                )
            manifest["commands"].extend({"task_id": spec["task_id"], "argv": spec["command"]} for spec in batch_specs)
            atomic_json(output_root / "run_manifest.json", manifest)
            progress.set_stage("SELECTOR_BATCHES", "RUNNING", requested_batch_count=len(batch_specs))
            batch_results = run_task_group(
                project_root=project_root,
                progress=progress,
                specs=batch_specs,
                max_parallel=int(args.max_parallel),
            )
            failed = [row for row in batch_results if row["status"] != "COMPLETE"]
            if failed:
                raise RuntimeError("selector batch subprocess failures: " + ", ".join(row["task_id"] for row in failed))

            _check_batch_files(output_root, "selector")
            selector_out = output_root / "selector" / "aggregate"
            selector_command = [
                sys.executable,
                "analysis/validate_nested_selector.py",
                "aggregate",
                "--input-root",
                str(output_root / "selector" / "batches"),
                "--output-dir",
                str(selector_out),
            ]
            manifest["commands"].append({"task_id": "selector_aggregate", "argv": selector_command})
            atomic_json(output_root / "run_manifest.json", manifest)
            progress.set_stage("SELECTOR_AGGREGATE", "RUNNING")
            selector_result = run_task(
                project_root=project_root,
                progress=progress,
                task_id="selector_aggregate",
                command=selector_command,
                output_dir=selector_out,
                log_path=_task_log_path(output_root, "selector_aggregate"),
            )
            if selector_result["status"] != "COMPLETE":
                raise RuntimeError(f"selector aggregate failed; see {selector_result['log']}")

        if "fixed" in selected_tasks:
            batch_specs = []
            for start in range(0, 100, SEEDS_PER_BATCH):
                batch_specs.append(
                    _batch_spec(
                        output_root=output_root,
                        task_id=f"fixed_p5_{start:02d}_{start + 9:02d}",
                        script="analysis/validate_p5_combos.py",
                        command_prefix=["run", "--combo", FIXED_COMBO],
                        seed_start=start,
                    )
                )
            manifest["commands"].extend({"task_id": spec["task_id"], "argv": spec["command"]} for spec in batch_specs)
            atomic_json(output_root / "run_manifest.json", manifest)
            progress.set_stage("FIXED_P5_BATCHES", "RUNNING", requested_batch_count=len(batch_specs))
            batch_results = run_task_group(
                project_root=project_root,
                progress=progress,
                specs=batch_specs,
                max_parallel=int(args.max_parallel),
            )
            failed = [row for row in batch_results if row["status"] != "COMPLETE"]
            if failed:
                raise RuntimeError("fixed P5 batch subprocess failures: " + ", ".join(row["task_id"] for row in failed))
            _check_batch_files(output_root, "fixed_p5")
            progress.set_stage("FIXED_MERGE", "RUNNING")
            merged_root = merge_fixed_batches(output_root)
            summary_out = output_root / "fixed_p5" / "summary"
            summarize_command = [
                sys.executable,
                "analysis/validate_p5_combos.py",
                "summarize",
                "--input-root",
                str(merged_root),
                "--output-dir",
                str(summary_out),
            ]
            manifest["commands"].append({"task_id": "fixed_p5_summarize", "argv": summarize_command})
            atomic_json(output_root / "run_manifest.json", manifest)
            summary_result = run_task(
                project_root=project_root,
                progress=progress,
                task_id="fixed_p5_summarize",
                command=summarize_command,
                output_dir=summary_out,
                log_path=_task_log_path(output_root, "fixed_p5_summarize"),
            )
            if summary_result["status"] != "COMPLETE":
                raise RuntimeError(f"fixed P5 summarize failed; see {summary_result['log']}")

        after = fingerprint_snapshot(project_root)
        if after != snapshot:
            raise RuntimeError("source/input fingerprints changed while final100 was running")
        manifest["fingerprints_after"] = after
        manifest["status"] = "COMPLETE"
        manifest["finished_at_utc"] = utc_now()
        atomic_json(output_root / "run_manifest.json", manifest)
        progress.set_stage("COMPLETE", "COMPLETE", finished_at_utc=utc_now())
        print(f"final100 model runs complete: {output_root}")
        return 0
    except Exception as exc:
        manifest["status"] = "FAILED"
        manifest["finished_at_utc"] = utc_now()
        manifest["failure"] = f"{type(exc).__name__}: {exc}"
        try:
            manifest["fingerprints_at_failure"] = fingerprint_snapshot(project_root)
        except Exception as fingerprint_error:
            manifest["fingerprints_at_failure_error"] = f"{type(fingerprint_error).__name__}: {fingerprint_error}"
        atomic_json(output_root / "run_manifest.json", manifest)
        progress.set_stage("FAILED", "FAILED", failure=manifest["failure"], finished_at_utc=utc_now())
        print(f"final100 stopped: {manifest['failure']}", file=sys.stderr)
        print(f"partial outputs and per-task logs are retained under {output_root.parent}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
