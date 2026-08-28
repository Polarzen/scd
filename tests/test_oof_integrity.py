import numpy as np
import pandas as pd

from src.nested_cv import run_nested_cv


FEATURES = [f"x{i}" for i in range(6)]


def _frame(n: int = 24) -> pd.DataFrame:
    rng = np.random.default_rng(404)
    frame = pd.DataFrame(rng.normal(size=(n, len(FEATURES))), columns=FEATURES)
    frame.insert(0, "patient_id", [f"P{i:03d}" for i in range(n)])
    frame["label"] = np.asarray([0, 1] * (n // 2), dtype=int)
    frame["af_flag"] = np.asarray([False, True] * (n // 2))
    frame["pvc_count_24h"] = np.arange(n, dtype=float)
    frame["pvc_status"] = np.where(frame["pvc_count_24h"] % 2, "AVAILABLE", "UNAVAILABLE")
    frame["high_pvc"] = frame["pvc_count_24h"] >= 18
    return frame


def test_oof_has_exactly_one_prediction_and_preserves_requested_metadata():
    frame = _frame()
    result = run_nested_cv(
        frame,
        model="dummy",
        feature_cols=FEATURES,
        outer_folds=3,
        inner_folds=2,
        bootstrap_resamples=0,
        seed=22,
    )
    oof = result.oof

    assert len(oof) == len(frame)
    assert oof["patient_id"].is_unique
    assert oof.groupby("patient_id", sort=False).size().eq(1).all()
    assert set(oof["patient_id"]) == set(frame["patient_id"])
    assert oof["prediction_probability"].between(0.0, 1.0).all()
    assert oof["prediction_label"].isin([0, 1]).all()
    assert oof["outer_fold"].notna().all()

    expected = frame.set_index("patient_id").sort_index()
    observed = oof.set_index("patient_id").sort_index()
    for column in ("af_flag", "pvc_count_24h", "pvc_status", "high_pvc"):
        pd.testing.assert_series_equal(observed[column], expected[column], check_names=False)


def test_oof_rows_are_deterministic_for_same_seed_and_frame():
    frame = _frame()
    kwargs = dict(
        model="dummy",
        feature_cols=FEATURES,
        outer_folds=3,
        inner_folds=2,
        bootstrap_resamples=0,
        seed=22,
    )
    first = run_nested_cv(frame, **kwargs)
    second = run_nested_cv(frame, **kwargs)
    pd.testing.assert_frame_equal(first.oof, second.oof)
