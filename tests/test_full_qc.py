import numpy as np
import pandas as pd

from src.full_features import (
    AF_INCOMPATIBLE_HRV,
    INSUFFICIENT_VALID_NN,
    INVALID_RR_RATIO,
    _set_qc_fields,
)


def _row(valid=10, removed=0, raw=10, success=True):
    return {
        "feature_extraction_success": success,
        "valid_rr_count": valid,
        "raw_rr_count": raw,
        "removed_rr_ratio": removed / raw if raw else 0.0,
        "failure_reason": "",
    }


def test_qc_valid_is_separate_from_feature_extraction_success():
    row = _row(valid=9)
    _set_qc_fields(row)
    assert row["feature_extraction_success"]
    assert not row["qc_valid"]
    assert INSUFFICIENT_VALID_NN in row["qc_reason"]

    row = _row(valid=10, removed=3, raw=10)
    _set_qc_fields(row)
    assert row["feature_extraction_success"]
    assert not row["qc_valid"]
    assert INVALID_RR_RATIO in row["qc_reason"]


def test_af_qc_reason_is_explicit():
    row = _row(valid=20)
    _set_qc_fields(row, af_flag=True)
    assert not row["qc_valid"]
    assert row["qc_reason"] == AF_INCOMPATIBLE_HRV
