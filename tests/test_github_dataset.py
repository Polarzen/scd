"""Repository-side checks for the compact-data GitHub handoff."""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pandas as pd
import yaml


REPO = Path(__file__).resolve().parents[1]
WORKFLOW_DIR = REPO / ".github" / "workflows"
WORKFLOWS = ("ci.yml", "model-validation.yml", "repeated-cv.yml", "endpoint-sensitivity.yml")
RAW_CANDIDATE_SUFFIXES = {".dat", ".mat", ".wav", ".wfdb", ".npy", ".npz", ".zip", ".tar", ".gz"}


def _workflow(name: str) -> tuple[dict, str]:
    path = WORKFLOW_DIR / name
    text = path.read_text(encoding="utf-8")
    document = yaml.safe_load(text)
    assert isinstance(document, dict), f"{name} is not a YAML mapping"
    return document, text


def _triggers(document: dict) -> dict:
    # YAML 1.1 loaders (including PyYAML's safe loader) treat ``on`` as a
    # boolean key; GitHub's parser correctly retains it as the event key.
    value = document.get("on", document.get(True, {}))
    return value if isinstance(value, dict) else {}


def _run_commands(document: dict) -> str:
    commands: list[str] = []
    for job in (document.get("jobs") or {}).values():
        if not isinstance(job, dict):
            continue
        steps = job.get("steps") or []
        for step in steps:
            if isinstance(step, dict) and isinstance(step.get("run"), str):
                commands.append(step["run"])
    return "\n".join(commands)


def _tracked_files() -> list[str]:
    result = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=REPO,
        check=True,
        capture_output=True,
    )
    return [item.decode("utf-8") for item in result.stdout.split(b"\0") if item]


def test_all_handoff_workflows_parse_and_have_expected_events():
    documents = {name: _workflow(name)[0] for name in WORKFLOWS}
    ci_events = _triggers(documents["ci.yml"])
    assert {"push", "pull_request", "workflow_dispatch"}.issubset(ci_events)
    for name in WORKFLOWS[1:]:
        events = _triggers(documents[name])
        assert set(events) == {"workflow_dispatch"}, name


def test_manual_model_inputs_and_artifact_contract_are_explicit():
    document, text = _workflow("model-validation.yml")
    inputs = _triggers(document)["workflow_dispatch"]["inputs"]
    assert set(inputs["endpoint"]["options"]) == {"90", "180", "365", "730"}
    assert set(inputs["profile"]["options"]) == {"all20", "physiology_only"}
    assert set(inputs["model"]["options"]) == {"extratrees", "logistic", "dummy"}
    assert all(token in text.lower() for token in ("oof", "metrics", "folds", "bootstrap", "population", "config"))
    assert "python -m src.cli validate" in text


def test_repeated_and_endpoint_matrices_are_bounded_and_machine_readable():
    repeated, repeated_text = _workflow("repeated-cv.yml")
    repeated_job = repeated["jobs"]["batches"]
    assert repeated_job["strategy"]["max-parallel"] == 5
    assert repeated_job["strategy"]["matrix"]["batch"] == list(range(1, 11))
    assert "python -m src.cli repeated-cv" in repeated_text
    assert "repeated_cv_runs.parquet" in repeated_text
    assert "repeated_cv_summary.json" in repeated_text
    assert "p97_5" in repeated_text
    assert 'f"{metric.lower()}_distribution.png"' in repeated_text

    endpoint, endpoint_text = _workflow("endpoint-sensitivity.yml")
    endpoint_job = endpoint["jobs"]["horizons"]
    assert endpoint_job["strategy"]["max-parallel"] == 4
    assert endpoint_job["strategy"]["matrix"]["endpoint"] == [90, 180, 365, 730]
    assert "python -m src.cli endpoint-sensitivity" in endpoint_text
    assert "endpoint_sensitivity.json" in endpoint_text


def test_workflow_commands_do_not_reintroduce_raw_or_rebuild_paths():
    forbidden = re.compile(r"\bMUSIC_RAW_DIR\b|\b(?:wget|curl|download|build|rebuild)\b", re.IGNORECASE)
    for name in WORKFLOWS:
        document, _ = _workflow(name)
        assert not forbidden.search(_run_commands(document)), name


def test_compact_tracked_data_has_expected_scale_and_no_waveform_candidates():
    tracked = _tracked_files()
    tracked_data = [path for path in tracked if path.startswith("data/")]
    assert "data/cohort/subjects.parquet" in tracked_data
    assert "data/cohort/records.parquet" in tracked_data
    assert "data/cohort/provenance.parquet" in tracked_data
    assert "data/features/legacy_120s/patient_features.parquet" in tracked_data
    assert len(tracked_data) >= 1_600

    total_bytes = sum((REPO / path).stat().st_size for path in tracked_data)
    assert total_bytes < 50 * 1024 * 1024
    assert not any(Path(path).suffix.lower() in RAW_CANDIDATE_SUFFIXES for path in tracked_data)

    subjects = pd.read_parquet(REPO / "data" / "cohort" / "subjects.parquet")
    records = pd.read_parquet(REPO / "data" / "cohort" / "records.parquet")
    legacy = pd.read_parquet(REPO / "data" / "features" / "legacy_120s" / "patient_features.parquet")
    assert len(subjects) >= 900
    assert subjects["patient_id"].is_unique
    assert len(records) >= 1_500
    assert len(legacy) >= 80

    full = REPO / "data" / "features" / "full_5min" / "patient_features.parquet"
    if full.is_file():
        full_features = pd.read_parquet(full)
        assert len(full_features) >= 900
        assert full_features["patient_id"].is_unique
