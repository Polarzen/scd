import numpy as np
import pandas as pd

from src.full_aggregation import AGGREGATED_FEATURE_NAMES, VALID_COUNT_NAMES, aggregate_patient_features, derive_pvc_burden
from src.full_features import FEATURE_NAMES


def _windows():
    rows = []
    for index in range(3):
        row = {"patient_id": "P1", "feature_extraction_success": True, "qc_valid": index == 0, "waveform_read_success": True, "raw_rpeak_count": 100, "raw_rr_count": 99, "valid_rr_count": 99, "removed_rr_count": 0}
        for feature in FEATURE_NAMES:
            row[feature] = float(index + 1)
            row[f"{feature}_valid"] = True
        rows.append(row)
    return pd.DataFrame(rows)


def test_aggregation_uses_successful_windows_and_per_feature_bits():
    result = aggregate_patient_features(
        _windows(),
        subjects=pd.DataFrame([{"patient_id": "P1", "has_holter": True, "pvc_count_24h": 10}]),
        min_qc_valid_windows=1,
    ).iloc[0]
    assert len(AGGREGATED_FEATURE_NAMES) == 100
    assert result["sig_mean_mean"] == 2.0
    assert result["sig_mean_valid_count"] == 3
    assert result["n_windows_qc_valid"] == 1
    assert result["n_windows_successful"] == 3


def test_invalid_feature_bit_is_excluded_independently():
    windows = _windows()
    windows.loc[1, "sig_mean_valid"] = False
    result = aggregate_patient_features(windows, min_qc_valid_windows=0).iloc[0]
    assert result["sig_mean_valid_count"] == 2
    assert result["sig_mean_mean"] == 2.0


def test_pvc_burden_uses_detected_beats_and_threshold():
    burden = derive_pvc_burden(21, _windows(), threshold=0.20)
    assert burden["pvc_denominator_beats"] == 300
    assert burden["pvc_burden"] == 0.07
    assert not burden["high_pvc_burden"]
