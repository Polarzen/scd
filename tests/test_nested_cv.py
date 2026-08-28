import numpy as np
import pandas as pd

from src.full_model import model_feature_names
from src.nested_cv import run_nested_cv


def _frame(n=24):
    rng = np.random.default_rng(7)
    frame = pd.DataFrame(rng.normal(size=(n, 100)), columns=model_feature_names())
    frame.insert(0, "patient_id", [f"P{i:03d}" for i in range(n)])
    frame["label"] = np.asarray([0, 1] * (n // 2), dtype=int)
    frame.loc[0, model_feature_names()[0]] = np.nan
    return frame


def test_nested_cv_emits_one_oof_row_per_patient_and_is_deterministic():
    frame = _frame()
    kwargs = dict(model="dummy", outer_folds=3, inner_folds=2, n_iter=1, bootstrap_resamples=25, seed=11)
    first = run_nested_cv(frame, **kwargs)
    second = run_nested_cv(frame, **kwargs)
    assert len(first.oof) == len(frame)
    assert first.oof["patient_id"].is_unique
    pd.testing.assert_frame_equal(first.oof, second.oof)
    assert set(("AUC", "AP", "Brier", "Sens", "Spec", "F1", "PPV", "NPV")) <= set(first.summary["metrics"])


def test_logistic_inner_search_and_threshold_run_on_synthetic_data():
    result = run_nested_cv(_frame(), model="logistic", outer_folds=3, inner_folds=2, n_iter=1, bootstrap_resamples=2, seed=13)
    assert len(result.folds) == 3
    assert result.summary["feature_count"] == 100
