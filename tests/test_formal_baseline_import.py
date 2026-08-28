from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts import import_formal_baseline as adapter
from src.optimization_reporting import aggregate_optimization_artifacts


def _legacy_tree(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path]:
    monkeypatch.setattr(adapter, "EXPECTED_SEEDS", (0, 1))
    tmp_path.mkdir(parents=True, exist_ok=True)
    patient_ids = [f"P{i:04d}" for i in range(adapter.EXPECTED_PATIENT_COUNT)]
    labels = np.zeros(adapter.EXPECTED_PATIENT_COUNT, dtype=int)
    labels[: adapter.EXPECTED_POSITIVE_COUNT] = 1
    manifest = pd.DataFrame(
        {
            "patient_id": patient_ids + ["EXCLUDED"],
            "label": [*labels, np.nan],
            "included": [True] * len(patient_ids) + [False],
            "af_flag": [False] * len(patient_ids) + [True],
            "pvc_count_24h": [0.0] * len(patient_ids) + [np.nan],
        }
    )
    manifest_path = tmp_path / "b0_population_manifest.csv"
    manifest.to_csv(manifest_path, index=False)

    root = tmp_path / "legacy"
    for seed in adapter.EXPECTED_SEEDS:
        artifact = root / f"formal-repeated-all20-seed-{seed}"
        artifact.mkdir(parents=True)
        summary = {
            "model": "ExtraTrees",
            "profile": "all20",
            "seed": seed,
            "patient_count": adapter.EXPECTED_PATIENT_COUNT,
            "outer_folds": adapter.EXPECTED_OUTER_FOLDS,
            "inner_folds": adapter.EXPECTED_INNER_FOLDS,
        }
        (artifact / "repeated_per_seed.json").write_text(json.dumps(summary), encoding="utf-8")
        rank = ((np.arange(len(labels)) * 37 + seed * 11) % 100) / 100.0
        probability = 0.02 + 0.06 * rank + 0.005 * labels
        oof = pd.DataFrame(
            {
                "patient_id": patient_ids,
                "y_true": labels,
                "prediction_probability": probability,
                "prediction_label": (probability >= 0.05).astype(int),
                "outer_fold": (np.arange(len(patient_ids)) % adapter.EXPECTED_OUTER_FOLDS) + 1,
                "threshold": 0.05,
                "model": "ExtraTrees",
                "profile": "all20",
                "seed": seed,
                "endpoint_horizon_days": adapter.EXPECTED_ENDPOINT,
            }
        )
        oof.to_parquet(artifact / "repeated_oof.parquet", index=False)
    return root, manifest_path


def test_import_formal_baseline_is_aggregate_compatible(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    legacy, manifest = _legacy_tree(tmp_path, monkeypatch)
    output = tmp_path / "adapted"
    result = adapter.import_formal_baseline(legacy, output, manifest)

    assert result["candidate"] == "B0"
    assert [row["seed"] for row in result["seeds"]] == [0, 1]
    summaries = sorted(output.rglob("*_summary.json"))
    assert len(summaries) == 2
    payload = json.loads(summaries[0].read_text(encoding="utf-8"))
    assert payload["profile"] == "all20_100"
    assert payload["legacy_run_id"] == adapter.LEGACY_RUN_ID

    aggregate = aggregate_optimization_artifacts(
        artifact_dir=output,
        expected_candidates=["B0"],
        expected_seeds=[0, 1],
        baseline_candidate="B0",
    )
    assert aggregate["runs"]["seed"].tolist() == [0, 1]
    assert aggregate["runs"]["candidate"].tolist() == ["B0", "B0"]
    assert aggregate["runs"]["patient_count"].tolist() == [703, 703]


def test_import_rejects_missing_and_duplicate_seed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    legacy, manifest = _legacy_tree(tmp_path, monkeypatch)
    shutil.rmtree(legacy / "formal-repeated-all20-seed-1")
    with pytest.raises(adapter.BaselineImportError, match="seed set mismatch"):
        adapter.import_formal_baseline(legacy, tmp_path / "missing", manifest)

    legacy, manifest = _legacy_tree(tmp_path / "duplicate_case", monkeypatch)
    duplicate = legacy / "nested" / "formal-repeated-all20-seed-0"
    shutil.copytree(legacy / "formal-repeated-all20-seed-0", duplicate)
    with pytest.raises(adapter.BaselineImportError, match="duplicate legacy artifact"):
        adapter.import_formal_baseline(legacy, tmp_path / "duplicate", manifest)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("endpoint", "endpoint must be 365"),
        ("profile", "legacy all20 profile"),
        ("model", "identify ExtraTrees"),
        ("duplicate_patient", "one row per patient"),
    ],
)
def test_import_rejects_invalid_oof(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
    message: str,
) -> None:
    legacy, manifest = _legacy_tree(tmp_path, monkeypatch)
    path = legacy / "formal-repeated-all20-seed-0" / "repeated_oof.parquet"
    frame = pd.read_parquet(path)
    if mutation == "endpoint":
        frame["endpoint_horizon_days"] = 730
    elif mutation == "profile":
        frame["profile"] = "physiology_only"
    elif mutation == "model":
        frame["model"] = "logistic"
    else:
        frame.loc[1, "patient_id"] = frame.loc[0, "patient_id"]
    frame.to_parquet(path, index=False)
    with pytest.raises(adapter.BaselineImportError, match=message):
        adapter.import_formal_baseline(legacy, tmp_path / "invalid", manifest)


def test_import_rejects_wrong_current_population(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    legacy, manifest = _legacy_tree(tmp_path, monkeypatch)
    frame = pd.read_csv(manifest)
    frame.loc[0, "included"] = False
    frame.to_csv(manifest, index=False)
    with pytest.raises(adapter.BaselineImportError, match="selected population"):
        adapter.import_formal_baseline(legacy, tmp_path / "wrong_population", manifest)
