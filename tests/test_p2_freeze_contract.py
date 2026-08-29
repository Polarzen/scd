import json
from pathlib import Path

from src.model_optimization import (
    PVC_CONTINUOUS_UNAVAILABLE,
    PVC_SOURCE_FEATURE,
    candidate_spec,
)


FREEZE_PATH = Path("config/p2_frozen_v1.json")


def test_p2_frozen_contract_matches_candidate_registry():
    freeze = json.loads(FREEZE_PATH.read_text(encoding="utf-8"))
    spec = candidate_spec("P2")

    assert freeze["freeze_id"] == "P2-FROZEN-V1"
    assert freeze["status"] == "PROMOTED_CANDIDATE_FROZEN"
    assert freeze["candidate"] == spec.name == "P2"
    assert freeze["profile"] == spec.profile == "rhythm_safe"
    assert freeze["population"] == spec.population == "rhythm_safe"
    assert freeze["model"] == spec.model == "elasticnet"
    assert freeze["feature_count"] == spec.feature_count == 24
    assert freeze["feature_columns"] == list(spec.feature_cols)
    assert freeze["feature_columns"][-2:] == ["af_flag", PVC_SOURCE_FEATURE]
    assert freeze["pvc_source_feature"] == PVC_SOURCE_FEATURE
    assert freeze["pvc_continuous_burden_available"] is False
    assert PVC_CONTINUOUS_UNAVAILABLE is True


def test_p2_freeze_prevents_silent_v1_retuning():
    freeze = json.loads(FREEZE_PATH.read_text(encoding="utf-8"))
    policy = freeze["freeze_policy"]

    assert policy["allow_refit_for_reproduction"] is True
    assert policy["allow_posthoc_defensive_validation"] is True
    assert policy["allow_model_or_feature_retuning_in_v1"] is False
    assert policy["allow_endpoint_reselection"] is False
    assert policy["allow_patient_exclusion_for_metric_improvement"] is False
    assert policy["future_model_changes_require_new_candidate_version"] is True


def test_p2_frozen_population_and_validation_contract():
    freeze = json.loads(FREEZE_PATH.read_text(encoding="utf-8"))

    assert freeze["endpoint_horizon_days"] == 365
    assert freeze["primary_endpoint"] is True
    assert freeze["patient_count"] == 878
    assert freeze["positive_count"] == 37
    assert freeze["negative_count"] == 841
    assert freeze["af_included_count"] == 164
    assert freeze["af_positive_included_count"] == 10
    assert freeze["full_365d_positive_count"] == 38
    assert freeze["unrecovered_positive_count"] == 1
    assert freeze["unrecovered_positive_reason"] == "NO_HOLTER"
    assert freeze["outer_folds"] == 5
    assert freeze["inner_folds"] == 3
    assert freeze["formal_seeds"] == {"start": 0, "stop_inclusive": 99, "count": 100}
