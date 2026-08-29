from pathlib import Path

import pandas as pd
import pytest
import yaml

from scripts.finalize_model_optimization import FinalizationError, _promotion, validate_population


def _valid_population_audit() -> dict:
    return {
        "baseline": {
            "full_positive": 38,
            "baseline_positive": 27,
            "baseline_excluded_positive": 11,
            "baseline_excluded_positive_af": 10,
            "baseline_excluded_positive_no_holter": 1,
            "rhythm_safe_modeled": 878,
            "rhythm_safe_positive": 37,
            "rhythm_safe_recovered_positive_af": 10,
        },
        "candidates": {
            "P2": {
                "patient_count": 878,
                "positive_count": 37,
                "negative_count": 841,
                "feature_count": 24,
                "af_included_count": 164,
                "af_positive_included_count": 10,
                "pvc_fields": ["pvc_count_24h"],
            }
        },
        "PVC_CONTINUOUS_UNAVAILABLE": True,
    }


def test_population_invariants_fail_closed_when_a_required_count_is_missing():
    audit = _valid_population_audit()
    del audit["baseline"]["full_positive"]

    with pytest.raises(FinalizationError, match="population invariant failed: full_positive"):
        validate_population(audit)


def test_promotion_returns_promoted_candidate_for_a_complete_winning_fixture():
    summary = pd.DataFrame(
        [
            {
                "candidate": "B0",
                "AUC_median": 0.60,
                "AUC_p2.5": 0.30,
                "AUC_p97.5": 0.90,
                "AP_median": 0.10,
                "Brier_median": 0.05,
                "BrierSkill_mean": -0.01,
                "AUC_std": 0.10,
            },
            {
                "candidate": "P2",
                "AUC_median": 0.70,
                "AUC_p2.5": 0.40,
                "AUC_p97.5": 0.78,
                "AP_median": 0.20,
                "Brier_median": 0.04,
                "BrierSkill_mean": 0.01,
                "AUC_std": 0.08,
            },
        ]
    )
    paired = {
        "comparisons": {
            "P2": {
                "candidate_better_fraction": {"AUC": 1.0, "Brier": 1.0},
            }
        }
    }
    calibration = pd.DataFrame(
        [
            {"candidate": "B0", "calibration_slope": 0.5, "calibration_intercept": 0.10},
            {"candidate": "P2", "calibration_slope": 0.9, "calibration_intercept": 0.02},
        ]
    )

    decision, checks = _promotion(summary, paired, calibration)

    assert decision == "PROMOTED_CANDIDATE"
    assert all(checks.values())


def test_model_optimization_workflow_parses_and_declares_finalize_contract():
    workflow = Path(__file__).parents[1] / ".github" / "workflows" / "model-optimization.yml"
    text = workflow.read_text(encoding="utf-8")
    document = yaml.safe_load(text)
    assert isinstance(document, dict)
    events = document.get("on", document.get(True, {}))
    assert "finalize" in events["workflow_dispatch"]["inputs"]["stage"]["options"]
    job = document["jobs"]["finalize"]
    assert job["needs"] == ["configure", "population_audit"]
    assert "python -m scripts.finalize_model_optimization" in job["steps"][-2]["run"]
    for run_id in (33193292911, 33193756146, 33171483784, 33187159259, 33187663480):
        assert str(run_id) in text
    assert "name: optimization-final" in text
