import numpy as np
import pandas as pd

from src.full_model import model_feature_names, prepare_model_frame


def test_prepare_model_frame_builds_dynamic_binary_endpoint_only():
    patient_ids = ["positive", "negative", "censored", "competing"]
    features = pd.DataFrame(np.arange(4 * 100, dtype=float).reshape(4, 100), columns=model_feature_names())
    features.insert(0, "patient_id", patient_ids)
    subjects = pd.DataFrame(
        {
            "patient_id": patient_ids,
            "followup_days": [20.0, 500.0, 20.0, 20.0],
            "cause_of_death_raw": ["3", "0", "0", "1"],
            "event_source_valid": [True, True, True, True],
        }
    )
    result = prepare_model_frame(features, subjects, horizon_days=365)
    assert result["patient_id"].tolist() == ["negative", "positive"]
    assert result["label"].tolist() == [0, 1]
    assert len([column for column in result if column.endswith("_mean")]) == 20


def test_feature_profiles_have_expected_aggregate_counts():
    assert len(model_feature_names("all20")) == 100
    # The current frozen manifest has two signal-quality bases; features_v2
    # manifests with additional signal-quality bases are handled dynamically.
    assert len(model_feature_names("physiology_only")) < 100
