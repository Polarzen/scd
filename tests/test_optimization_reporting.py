import json

import numpy as np
import pandas as pd

from src.metrics import compute_metrics
from src.optimization_reporting import aggregate_optimization_artifacts


def _fixtures():
    summaries = []
    oof = []
    for candidate in ("baseline", "candidate"):
        for seed in (0, 1):
            summaries.append({"candidate": candidate, "profile": "all20", "model": "logistic", "seed": seed, "endpoint_horizon_days": 365})
            values = ((0, 0.4), (1, 0.6)) if candidate == "baseline" else ((0, 0.1), (1, 0.9))
            for index, (label, probability) in enumerate(values):
                oof.append(
                    {
                        "candidate": candidate,
                        "profile": "all20",
                        "model": "logistic",
                        "seed": seed,
                        "patient_id": f"P{index}",
                        "outer_fold": index + 1,
                        "y_true": label,
                        "prediction_probability": probability,
                        "prediction_label": int(probability >= 0.5),
                        "threshold": 0.5,
                        "endpoint_horizon_days": 365,
                        "af_flag": index,
                        "high_pvc": index,
                    }
                )
    return summaries, pd.DataFrame(oof)


def test_aggregation_writes_portable_outputs_and_oriented_paired_deltas(tmp_path):
    summaries, oof = _fixtures()
    result = aggregate_optimization_artifacts(
        summaries,
        oof,
        expected_seeds={0, 1},
        baseline_candidate="baseline",
        target_dir=tmp_path,
        report_inputs={"objective": "synthetic optimization"},
    )
    comparison = result["paired_comparison"]["comparisons"]["candidate"]
    assert all(delta == 0 for delta in comparison["deltas"]["AUC"])
    assert all(delta == 0 for delta in comparison["deltas"]["AP"])
    assert all(delta < 0 for delta in comparison["deltas"]["Brier"])
    assert comparison["candidate_better_fraction"]["Brier"] == 1.0
    assert json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))["candidates"]
    for name in ("runs.csv", "runs.parquet", "summary.csv", "paired_comparison.json", "calibration_summary.csv", "calibration_bins.csv"):
        assert (tmp_path / name).exists()
    assert len(pd.read_csv(tmp_path / "calibration_bins.csv")) == 20
    selected = oof.loc[oof["candidate"].eq("candidate")]
    expected_calibration = compute_metrics(selected["y_true"], selected["prediction_probability"])
    calibration_row = result["calibration_summary"].loc[
        result["calibration_summary"]["candidate"].eq("candidate")
    ].iloc[0]
    for name in ("calibration_slope", "calibration_intercept"):
        if np.isnan(expected_calibration[name]):
            assert np.isnan(calibration_row[name])
        else:
            assert np.isclose(calibration_row[name], expected_calibration[name])
    report = (tmp_path / "reports" / "MODEL_OPTIMIZATION.md").read_text(encoding="utf-8")
    assert "PROMOTED_CANDIDATE" not in report  # no discrimination gain in this fixture
    assert report.count("## ") == 10
    assert "clinical utility" not in report.lower()
