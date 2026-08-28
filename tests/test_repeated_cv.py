import numpy as np
import pandas as pd

from src.full_model import model_feature_names
from src.repeated_cv import run_repeated_cv


def test_repeated_cv_reports_seed_metrics_and_summary_statistics():
    rng = np.random.default_rng(3)
    frame = pd.DataFrame(rng.normal(size=(24, 100)), columns=model_feature_names())
    frame.insert(0, "patient_id", [f"P{i:03d}" for i in range(24)])
    frame["label"] = np.asarray([0, 1] * 12, dtype=int)
    result = run_repeated_cv(
        frame,
        models=["dummy"],
        seeds=[2, 5],
        outer_folds=3,
        inner_folds=2,
        n_iter=1,
        bootstrap_resamples=2,
    )
    assert len(result.per_seed) == 2
    assert result.per_seed["seed"].tolist() == [2, 5]
    assert set(result.summary["statistic"]) == {"mean", "std", "min", "p2.5", "p25", "median", "p75", "p97.5", "max"}
    assert len(result.oof) == 48
