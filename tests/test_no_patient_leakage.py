import numpy as np
import pandas as pd
import pytest

from src.nested_cv import make_outer_splits, run_nested_cv


FEATURES = [f"x{i}" for i in range(6)]


def _frame(n: int = 24) -> pd.DataFrame:
    rng = np.random.default_rng(202)
    frame = pd.DataFrame(rng.normal(size=(n, len(FEATURES))), columns=FEATURES)
    frame.insert(0, "patient_id", [f"P{i:03d}" for i in range(n)])
    frame["label"] = np.asarray([0, 1] * (n // 2), dtype=int)
    return frame


def test_explicit_outer_train_test_overlap_is_rejected():
    frame = _frame()
    splits = make_outer_splits(frame["label"].to_numpy(), outer_folds=3, seed=9)
    train, test = splits[0]
    bad_splits = list(splits)
    bad_splits[0] = (train, np.asarray([train[0], *test[1:]], dtype=int))

    with pytest.raises(ValueError, match="overlap"):
        run_nested_cv(
            frame,
            model="dummy",
            feature_cols=FEATURES,
            outer_folds=3,
            inner_folds=2,
            outer_splits=bad_splits,
            bootstrap_resamples=0,
        )


def test_outer_folds_have_disjoint_patients_and_cover_each_patient_once():
    frame = _frame()
    splits = make_outer_splits(frame["label"].to_numpy(), outer_folds=3, seed=12)
    patient_ids = frame["patient_id"].astype(str).to_numpy()
    seen: list[str] = []
    for train_idx, test_idx in splits:
        assert set(patient_ids[train_idx]).isdisjoint(set(patient_ids[test_idx]))
        seen.extend(patient_ids[test_idx].tolist())
    assert len(seen) == len(frame)
    assert len(set(seen)) == len(frame)

    result = run_nested_cv(
        frame,
        model="dummy",
        feature_cols=FEATURES,
        outer_folds=3,
        inner_folds=2,
        outer_splits=splits,
        bootstrap_resamples=0,
    )
    assert len(result.oof) == len(frame)
    assert result.oof["patient_id"].is_unique
    assert result.oof.groupby("patient_id", sort=False).size().eq(1).all()
    assert set(result.oof["patient_id"]) == set(frame["patient_id"])
