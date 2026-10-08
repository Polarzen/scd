"""Fit paired descriptive cause-specific Cox models.

This script has an optional runtime dependency on ``lifelines``. Install that
package separately in the analysis environment before running this script;
``lifelines`` is not bundled with the project or this source publication.
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True,
                        help="portable project checkout containing data/ and analysis/")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--endpoint-summary", type=Path, required=True,
                        help="completed endpoint sensitivity summary used as the run-order guard")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    project_root = args.project_root.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    endpoint_summary = args.endpoint_summary.expanduser().resolve()
    if not project_root.is_dir():
        raise FileNotFoundError(f"project root does not exist: {project_root}")
    if not endpoint_summary.is_file():
        raise FileNotFoundError(f"endpoint sensitivity summary does not exist: {endpoint_summary}")
    root_text = str(project_root)
    if root_text not in sys.path:
        sys.path.insert(0, root_text)

    import numpy as np
    import pandas as pd
    try:
        import lifelines
        from lifelines import CoxPHFitter
    except ModuleNotFoundError as exc:
        if exc.name == "lifelines":
            raise SystemExit(
                "run_cause_specific.py requires the optional 'lifelines' package; install it separately to continue"
            ) from exc
        raise
    from analysis.validate_p3 import _number
    from src.endpoints import build_endpoint

    output_dir.mkdir(parents=True, exist_ok=True)
    protocol = {
        "status": "FROZEN_BEFORE_FITTING",
        "post_hoc": True,
        "method": "paired cause-specific Cox proportional hazards models",
        "predictors": ["age", "lvef", "nyha_III"],
        "rationale": "Low-dimensional clinical baseline contrast: age as demographic context, LVEF as cardiac function, NYHA as symptom severity; external HF guideline references 2/3/4, not selected by current outcome differences, coefficients, or performance. These do not claim to be sufficient SCD predictors.",
        "time_unit": "days from enrollment",
        "horizon": 365,
        "cause1": "SCD raw cause 3 at <=365",
        "cause2": "other death raw cause 1/6/7 at <=365",
        "censoring": "min(observed follow-up,365); other event censored at its observed time for each cause-specific hazard",
        "penalizer": 0.1,
        "l1_ratio": 0,
        "preprocessing": "median imputation without indicators; standardized 3 columns; fixed full-cohort descriptive sensitivity, no performance validation",
        "selection": "none; no tuning; no feature search",
        "package": "lifelines",
        "version": lifelines.__version__,
        "function": "CoxPHFitter.fit",
        "reported": "penalized coefficients per standardized input and direction only; no inferential P values/CI, no training AUC, no cumulative-incidence prediction",
        "limits": "Cause-specific hazards are not Fine-Gray subdistribution hazards; 365-day competing-risk predictive performance and calibrated cumulative incidence are not evaluated.",
    }
    (output_dir / "cause_specific_protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    subjects = pd.read_parquet(project_root / "data" / "cohort" / "subjects.parquet")
    subjects = subjects[subjects.has_holter].sort_values("patient_id").reset_index(drop=True)
    endpoint = build_endpoint(subjects, 365)
    assert len(subjects) == 936 and subjects.event_source_valid.all()
    assert subjects.followup_days.notna().all() and subjects.followup_days.gt(0).all()
    predictors = pd.DataFrame({
        "age": _number(subjects.Age),
        "lvef": _number(subjects["LVEF (%)"]),
        "nyha_III": np.where(
            _number(subjects["NYHA class"]).isna(),
            np.nan,
            _number(subjects["NYHA class"]).eq(3).astype(float),
        ),
    })
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler

    imputer = SimpleImputer(strategy="median", keep_empty_features=True)
    scaler = StandardScaler()
    z = scaler.fit_transform(imputer.fit_transform(predictors))
    assert z.shape == (936, 3)
    rows = []
    captured_warnings = []
    for kind, event in (
        ("SCD", endpoint.endpoint_state.eq("POSITIVE")),
        ("competing_death", endpoint.endpoint_state.eq("COMPETING_EVENT")),
    ):
        data = pd.DataFrame(z, columns=predictors.columns)
        data["days"] = subjects.followup_days.clip(upper=365).to_numpy()
        data["event"] = event.to_numpy(int)
        model = CoxPHFitter(penalizer=0.1, l1_ratio=0)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            model.fit(data, duration_col="days", event_col="event", show_progress=False)
        captured_warnings.extend(
            {"cause": kind, "category": warning.category.__name__, "message": str(warning.message)}
            for warning in caught
        )
        assert np.isfinite(model.params_).all()
        for name, value in model.params_.items():
            rows.append({
                "cause": kind,
                "predictor": name,
                "coefficient_per_standardized_input": float(value),
                "direction": "positive" if value > 0 else "negative" if value < 0 else "zero",
                "penalizer": 0.1,
                "events": int(event.sum()),
                "n": 936,
                "interpretation": "descriptive penalized cause-specific hazard; not causal effect",
            })
        model.baseline_cumulative_hazard_.to_csv(output_dir / f"cause_specific_{kind}_baseline_hazard.csv")
    pd.DataFrame(rows).to_csv(output_dir / "cause_specific_coefficients.csv", index=False)
    pd.DataFrame({
        "predictor": predictors.columns,
        "imputation_median": imputer.statistics_,
        "mean": scaler.mean_,
        "scale": scaler.scale_,
        "missing": predictors.isna().sum(),
    }).to_csv(output_dir / "cause_specific_preprocessing.csv", index=False)
    summary = {
        "status": "PASS_DESCRIPTIVE_ONLY",
        "method": protocol["method"],
        "package": lifelines.__version__,
        "n": 936,
        "SCD_events": 37,
        "competing_events": 44,
        "early_censor": 14,
        "coefficients": rows,
        "warnings": captured_warnings,
        "Fine_Gray_attempted": False,
        "Fine_Gray_reason": "No validated Fine-Gray implementation present; chose the user-permitted mature cause-specific Cox alternative instead of implementing an optimizer.",
        "cumulative_incidence_and_time_dependent_performance": "NOT COMPUTED: this descriptive fit does not constitute an internally validated competing-risk prediction pipeline; no AUC/BS fabricated.",
    }
    (output_dir / "competing_risk_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
