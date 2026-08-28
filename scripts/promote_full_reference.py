#!/usr/bin/env python3
"""Promote completed seed-42 model outputs into stable reference artifacts."""

from __future__ import annotations

import json
from pathlib import Path
import shutil

import pandas as pd


REPO = Path(__file__).resolve().parents[1]
VALIDATION = REPO / "data" / "validation"
MODEL_DIR = VALIDATION / "full_model"


def _read_summary(model: str) -> dict:
    path = MODEL_DIR / f"{model}_all20_365d_seed42_summary.json"
    if not path.is_file():
        raise FileNotFoundError(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("seed") != 42 or value.get("patient_count") != 703:
        raise RuntimeError(f"unexpected reference identity in {path}")
    return value


def main() -> int:
    summaries = {model: _read_summary(model) for model in ("extratrees", "logistic", "dummy")}
    source_oof = MODEL_DIR / "extratrees_all20_365d_seed42_oof.parquet"
    source_folds = MODEL_DIR / "extratrees_all20_365d_seed42_folds.csv"
    oof = pd.read_parquet(source_oof)
    if len(oof) != 703 or not oof["patient_id"].is_unique:
        raise RuntimeError("reference OOF must contain exactly one row per eligible patient")
    shutil.copyfile(source_oof, VALIDATION / "full_reference_oof.parquet")
    shutil.copyfile(source_folds, VALIDATION / "full_reference_fold_metrics.csv")
    metrics = {
        "model_pipeline_version": "full_music_v1",
        "reference": summaries["extratrees"],
        "baselines": {name: summaries[name] for name in ("logistic", "dummy")},
    }
    (VALIDATION / "full_reference_metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8"
    )
    config = {
        "model_pipeline_version": "full_music_v1",
        "endpoint": 365,
        "feature_profile": "all20",
        "model": "extratrees",
        "seed": 42,
        "outer_folds": 5,
        "inner_folds": 3,
        "n_iter": 24,
        "target_specificity": 0.70,
        "bootstrap_resamples": 2000,
        "independent_unit": "patient",
        "raw_music_required": False,
    }
    (VALIDATION / "full_reference_config.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    extra = summaries["extratrees"]
    ci = extra["bootstrap"]["ci"]
    lines = [
        "# Full cohort reference model",
        "",
        "Reference: 365-day SCD risk stratification, all20, ExtraTrees, seed 42; patient is the independent unit.",
        "",
        "| Model | ROC-AUC | AP | Brier | Sensitivity | Specificity | F1 | PPV | NPV |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name in ("extratrees", "logistic", "dummy"):
        s = summaries[name]
        lines.append(f"| {name} | {s['AUC']:.6f} | {s['AP']:.6f} | {s['Brier']:.6f} | {s['Sens']:.6f} | {s['Spec']:.6f} | {s['F1']:.6f} | {s.get('PPV')} | {s['NPV']:.6f} |")
    lines += [
        "",
        f"ExtraTrees ROC-AUC 95% patient bootstrap CI: {ci['AUC']['lower']:.6f}–{ci['AUC']['upper']:.6f}.",
        f"ExtraTrees AP 95% patient bootstrap CI: {ci['AP']['lower']:.6f}–{ci['AP']['upper']:.6f}.",
        f"ExtraTrees Brier 95% patient bootstrap CI: {ci['Brier']['lower']:.6f}–{ci['Brier']['upper']:.6f}.",
        "",
        "Metrics are descriptive validation outputs, not performance acceptance thresholds.",
    ]
    (REPO / "reports" / "REFERENCE_MODEL.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"patient_count": len(oof), "metrics": {m: summaries[m]["metrics"] for m in summaries}}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
