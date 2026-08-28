import pandas as pd
import pytest

from src.full_aggregation import AGGREGATED_FEATURE_NAMES
from src.full_features import FEATURE_NAMES
from src.model_optimization import (
    ALL20_100_PROFILE,
    P50_20_PROFILE,
    ROBUST40_PROFILE,
    candidate_feature_columns,
    candidate_spec,
    prepare_optimization_bundle,
    profile_feature_columns,
    validate_feature_allowlist,
)


def _fixture_frames():
    subjects = pd.DataFrame(
        {
            "patient_id": ["P1", "P2"],
            "followup_days": [500.0, 10.0],
            "cause_of_death_raw": ["0", "3"],
            "event_source_valid": [True, True],
            "has_holter": [True, True],
            "af_flag": [False, False],
            "pvc_count_24h": [2.0, 3.0],
        }
    )
    feature_values = {"patient_id": subjects["patient_id"]}
    feature_values.update({name: float(position) for position, name in enumerate(AGGREGATED_FEATURE_NAMES, start=1)})
    feature_values.update({"processed_holter": True, "n_windows_successful": 1, "primary_sinus_hrv_eligible": True})
    features = pd.DataFrame(feature_values)
    return features, subjects


def test_profile_and_candidate_schemas_are_frozen_and_deterministic():
    assert len(profile_feature_columns(ALL20_100_PROFILE)) == 100
    assert len(profile_feature_columns(P50_20_PROFILE)) == 20
    robust = profile_feature_columns(ROBUST40_PROFILE)
    assert len(robust) == 40
    assert robust[:20] == [f"{name}_p50" for name in FEATURE_NAMES]
    assert robust[20:] == [f"{name}_p90_minus_p10" for name in FEATURE_NAMES]
    assert candidate_spec("m1").model == "elasticnet"
    assert candidate_feature_columns("P2")[-1] == "pvc_count_24h"

    features, subjects = _fixture_frames()
    features["sig_mean_p90_minus_p10"] = 999.0
    first = prepare_optimization_bundle(features, subjects, candidate="M2")
    second = prepare_optimization_bundle(features, subjects, candidate="M2")
    assert first.feature_cols == second.feature_cols
    pd.testing.assert_frame_equal(first.frame, second.frame)
    assert first.frame.loc[0, "sig_mean_p90_minus_p10"] == 2.0


def test_feature_allowlist_rejects_outcome_and_identifier_injection():
    with pytest.raises(ValueError, match="outcome/ID|non-approved"):
        validate_feature_allowlist(["sig_mean_p50", "label"])
    with pytest.raises(ValueError, match="outcome/ID|non-approved"):
        validate_feature_allowlist(["sig_mean_p50", "patient_id"])
