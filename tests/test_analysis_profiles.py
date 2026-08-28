import pandas as pd

from src.analysis_profiles import build_patient_analysis_status, build_analysis_population_365
from src.full_aggregation import AGGREGATED_FEATURE_NAMES


def _subjects():
    return pd.DataFrame(
        {
            "patient_id": ["P1", "P2", "P3"],
            "has_holter": [True, True, False],
            "af_flag": [False, True, False],
            "pvc_count_24h": [1, 1, None],
            "followup_days": [500.0, 500.0, 10.0],
            "cause_of_death_raw": ["0", "0", "0"],
            "event_source_valid": [True, True, True],
        }
    )


def test_status_has_dynamic_endpoint_columns_and_primary_reasons():
    windows = pd.DataFrame(
        {
            "patient_id": ["P1"],
            "feature_extraction_success": [True],
            "qc_valid": [True],
            "raw_rpeak_count": [100],
            "raw_rr_count": [99],
            "valid_rr_count": [99],
            "removed_rr_count": [0],
        }
    )
    features = pd.DataFrame({"patient_id": ["P1", "P2", "P3"], "processed_holter": [True, True, False], "high_pvc_burden": [False, False, False], "pvc_burden": [0.01, 0.01, None], "pvc_denominator_beats": [100.0, 100.0, None], "pvc_count_24h": [1.0, 1.0, None], "n_windows_theoretical": [1, 1, 0], "n_windows_successful": [1, 1, 0], "n_windows_qc_valid": [1, 0, 0]})
    status = build_patient_analysis_status(_subjects(), windows, features)
    assert len(status) == 3
    assert {f"endpoint_{h}_state" for h in (90, 180, 365, 730)} <= set(status.columns)
    assert status.set_index("patient_id").loc["P1", "primary_sinus_hrv_eligible"]
    assert status.set_index("patient_id").loc["P2", "primary_sinus_hrv_reason"] == "AF"
    assert status.set_index("patient_id").loc["P3", "primary_sinus_hrv_reason"] == "NO_HOLTER"


def test_population_requires_all_100_finite_features_and_binary_endpoint():
    rows = {"patient_id": ["P1"], "endpoint_365_state": ["NEGATIVE"], "primary_sinus_hrv_eligible": [True]}
    rows.update({name: [1.0] for name in AGGREGATED_FEATURE_NAMES})
    population = build_analysis_population_365(pd.DataFrame(rows))
    assert len(population) == 1
    assert population.loc[0, "label_365d"] == 0
