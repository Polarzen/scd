from pathlib import Path

import numpy as np
import pandas as pd

from src.model_optimization import prepare_optimization_bundle


ROOT = Path(__file__).resolve().parents[1]


def test_official_365_baseline_invariant_and_rhythm_safe_recovery():
    features_path = ROOT / "data" / "features" / "full_5min" / "patient_features.parquet"
    subjects_path = ROOT / "data" / "cohort" / "subjects.parquet"
    if not features_path.is_file() or not subjects_path.is_file():
        return

    baseline = prepare_optimization_bundle(features_path, subjects_path, candidate="B0")
    counts = baseline.audit["baseline_365"]
    assert baseline.audit["baseline_365_invariant_pass"]
    assert {
        key: counts[key]
        for key in (
            "full_positive",
            "baseline_positive",
            "baseline_excluded_positive",
            "baseline_excluded_positive_af",
            "baseline_excluded_positive_no_holter",
        )
    } == {
        "full_positive": 38,
        "baseline_positive": 27,
        "baseline_excluded_positive": 11,
        "baseline_excluded_positive_af": 10,
        "baseline_excluded_positive_no_holter": 1,
    }

    rhythm = prepare_optimization_bundle(features_path, subjects_path, candidate="A1")
    recovery = rhythm.audit["rhythm_safe_recovery"]
    assert recovery["positive_af_included"] == 10
    assert recovery["positive_no_holter_included"] == 0
    assert len(rhythm.frame) == 878
    assert int(rhythm.frame["af_flag"].sum()) > 0
