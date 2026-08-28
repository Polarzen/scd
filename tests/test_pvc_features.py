import numpy as np
import pandas as pd

from src.full_aggregation import AGGREGATED_FEATURE_NAMES
from src.model_optimization import prepare_optimization_bundle


def _frames():
    subjects = pd.DataFrame(
        {
            "patient_id": ["P1", "P2"],
            "followup_days": [500.0, 10.0],
            "cause_of_death_raw": ["0", "3"],
            "event_source_valid": [True, True],
            "has_holter": [True, True],
            "af_flag": [False, False],
            # This is the only PVC value that P2 is allowed to use.
            "pvc_count_24h": [11.0, np.nan],
            "pvc_information_available": [True, False],
        }
    )
    feature_values = {"patient_id": ["P1", "P2"]}
    feature_values.update({name: 1.0 for name in AGGREGATED_FEATURE_NAMES})
    feature_values.update({"processed_holter": True, "n_windows_successful": 1, "primary_sinus_hrv_eligible": True})
    # Deliberately disagree with source metadata to catch accidental feature
    # table preference for the PVC value.
    feature_values.update({"pvc_count_24h": [999.0, 888.0], "pvc_burden": [0.99, 0.88], "pvc_burden_status": ["AVAILABLE", "AVAILABLE"]})
    features = pd.DataFrame(feature_values)
    return features, subjects


def test_pvc_candidates_are_identical_except_for_true_source_count():
    features, subjects = _frames()
    p1 = prepare_optimization_bundle(features, subjects, candidate="P1")
    p2 = prepare_optimization_bundle(features, subjects, candidate="P2")

    assert p1.model == p2.model == "elasticnet"
    assert p1.profile == p2.profile == "rhythm_safe"
    assert p1.population_manifest["selected_population"] == "rhythm_safe"
    assert p2.population_manifest["selected_population"] == "rhythm_safe"
    assert p1.feature_cols == p2.feature_cols[:-1]
    assert p2.feature_cols[-1] == "pvc_count_24h"
    assert p2.frame.set_index("patient_id").loc["P1", "pvc_count_24h"] == 11.0
    assert np.isnan(p2.frame.set_index("patient_id").loc["P2", "pvc_count_24h"])
    assert p1.audit["PVC_CONTINUOUS_UNAVAILABLE"] is True
    assert p2.audit["pvc"]["derived_pvc_burden_used"] is False
    assert p2.audit["pvc"]["fold_local_imputation_required"] is True
