import numpy as np

from src.full_features import (
    FEATURE_NAMES,
    REQUIRES_NN_FEATURE_NAMES,
    apply_af_policy,
    clean_rr_intervals,
    window_features,
)


def test_feature_order_and_count_are_phase3_compatible():
    rng = np.random.default_rng(4)
    values = window_features(rng.normal(size=4_000), 100.0)
    assert list(values) == list(FEATURE_NAMES)
    assert len(values) == 20


def test_rr_mask_keeps_physiological_local_median_intervals_only():
    raw = np.array([1.0, 1.05, 3.0, 0.2, 0.95, 1.1])
    valid, mask = clean_rr_intervals(raw)
    assert mask.tolist() == [True, True, False, False, True, True]
    np.testing.assert_array_equal(valid, raw[mask])


def test_af_policy_preserves_signal_and_fft_but_nulls_features_08_to_16():
    import pandas as pd

    row = {name: float(index + 1) for index, name in enumerate(FEATURE_NAMES)}
    row.update({f"{name}_valid": True for name in FEATURE_NAMES})
    row.update({"patient_id": "P1", "feature_extraction_success": True, "qc_valid": True})
    result = apply_af_policy(pd.DataFrame([row]), True).iloc[0]
    assert result["sig_mean"] == 1.0
    assert result["pow_hf"] == 19.0
    for name in REQUIRES_NN_FEATURE_NAMES:
        assert np.isnan(result[name])
        assert not result[f"{name}_valid"]
    assert not result["qc_valid"]
