"""Fit and freeze the post-hoc full-development candidate model."""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True,
                        help="portable project checkout containing analysis/ and data/")
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="directory containing literature_sensitivity_summary.json and receiving artifacts")
    parser.add_argument("--protocol-dir", type=Path, required=True,
                        help="directory containing final_fit_protocol.json")
    parser.add_argument("--predictor-dictionary", type=Path, required=True,
                        help="source CSV defining the predictors used in the final candidate")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    project_root = args.project_root.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    protocol_path = args.protocol_dir.expanduser().resolve() / "final_fit_protocol.json"
    dictionary_path = args.predictor_dictionary.expanduser().resolve()
    if not project_root.is_dir():
        raise FileNotFoundError(f"project root does not exist: {project_root}")
    if not (output_dir / "literature_sensitivity_summary.json").is_file():
        raise FileNotFoundError("run the literature sensitivity first; its summary must be in --output-dir")
    if not protocol_path.is_file():
        raise FileNotFoundError(f"final-fit protocol does not exist: {protocol_path}")
    if not dictionary_path.is_file():
        raise FileNotFoundError(f"predictor dictionary does not exist: {dictionary_path}")
    root_text = str(project_root)
    if root_text not in sys.path:
        sys.path.insert(0, root_text)

    import joblib
    import numpy as np
    import pandas as pd
    import sklearn
    from scipy.special import expit
    from sklearn.model_selection import StratifiedKFold, GridSearchCV, cross_val_score
    from analysis.validate_nested_selector import build_selector_frame
    from analysis.validate_p3 import make_pipeline, C_GRID, L1_RATIO_GRID

    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    frame, recipes, _ = build_selector_frame()
    assert recipes == protocol["recipes"]
    seed = protocol["seed"]
    y = frame.label.to_numpy(int)
    splits = list(StratifiedKFold(3, shuffle=True, random_state=seed).split(np.zeros(len(y)), y))
    scores = {}
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        for name in sorted(recipes):
            estimator = make_pipeline(seed)
            estimator.set_params(clf__C=0.1, clf__l1_ratio=0.5)
            scores[name] = float(cross_val_score(
                estimator, frame[recipes[name]], y, cv=splits, scoring="average_precision",
                n_jobs=1, error_score="raise",
            ).mean())
        selected = max(sorted(scores), key=lambda name: scores[name])
        cols = recipes[selected]
        search = GridSearchCV(
            make_pipeline(seed),
            {"clf__C": list(C_GRID), "clf__l1_ratio": list(L1_RATIO_GRID)},
            cv=splits, scoring="average_precision", n_jobs=1, error_score="raise", refit=True,
        )
        search.fit(frame[cols], y)
        captured_warnings = [
            {"category": warning.category.__name__, "message": str(warning.message)}
            for warning in caught
        ]
    pipe = search.best_estimator_
    imputer = pipe.named_steps["imputer"]
    scaler = pipe.named_steps["scale"]
    classifier = pipe.named_steps["clf"]
    names = imputer.get_feature_names_out(cols)
    coef = classifier.coef_[0]
    intercept = float(classifier.intercept_[0])
    specification = {
        "created": True,
        "prespecified": False,
        "post_hoc_frozen": True,
        "selected_recipe": selected,
        "raw_predictor_order": cols,
        "transformed_predictor_order": names.tolist(),
        "C": float(search.best_params_["clf__C"]),
        "l1_ratio": float(search.best_params_["clf__l1_ratio"]),
        "n_coefficients": len(coef),
        "intercept": intercept,
        "sklearn": sklearn.__version__,
        "random_seed": seed,
        "n": 878,
        "events": 37,
        "outer_test_information_used": False,
        "training_performance_computed": False,
        "missing_indicator_original_indices": imputer.indicator_.features_.tolist(),
        "warnings": captured_warnings,
        "max_iter": classifier.max_iter,
        "tol": classifier.tol,
        "class_weight": classifier.class_weight,
        "n_iter": classifier.n_iter_.tolist(),
        "probability_definition": "conditional binary SCD status for original evaluable-cohort estimand, not cumulative incidence or validated deployment risk",
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "final_candidate_model.joblib"
    joblib.dump({
        "pipeline": pipe,
        "specification": specification,
        "current_code_root": str(project_root),
        "protocol": protocol,
    }, model_path)
    saved = joblib.load(model_path)
    p1 = saved["pipeline"].predict_proba(frame[cols])[:, 1]
    transformed = imputer.transform(frame[cols])
    p2 = expit(intercept + ((transformed - scaler.mean_) / scaler.scale_) @ coef)
    assert np.max(np.abs(p1 - p2)) < 1e-12
    specification["manual_probability_max_error"] = float(np.max(np.abs(p1 - p2)))
    pd.DataFrame({
        "predictor": names,
        "coefficient": coef,
        "scale_mean": scaler.mean_,
        "scale_sd": scaler.scale_,
        "raw_imputed_space_coefficient": coef / scaler.scale_,
    }).to_csv(output_dir / "final_candidate_model_coefficients.csv", index=False)
    rows = []
    for index, name in enumerate(names):
        raw = index < len(cols)
        source = name if raw else cols[int(imputer.indicator_.features_[index - len(cols)])]
        rows.append({
            "transformed_order": index + 1,
            "name": name,
            "source_predictor": source,
            "is_missing_indicator": not raw,
            "imputation_median": float(imputer.statistics_[index]) if raw else None,
            "scaler_mean": float(scaler.mean_[index]),
            "scaler_scale": float(scaler.scale_[index]),
            "coefficient": float(coef[index]),
        })
    pd.DataFrame(rows).to_csv(output_dir / "final_candidate_model_preprocessing.csv", index=False)
    (output_dir / "final_candidate_model_summary.json").write_text(
        json.dumps(specification, indent=2), encoding="utf-8"
    )
    import shutil
    shutil.copy2(dictionary_path, output_dir / "final_candidate_predictor_dictionary.csv")
    pd.DataFrame({"recipe": list(scores), "inner_screen_AP": list(scores.values())}).to_csv(
        output_dir / "final_candidate_screening_scores.csv", index=False
    )
    pd.DataFrame(search.cv_results_).to_csv(output_dir / "final_candidate_tuning_results.csv", index=False)
    (output_dir / "final_candidate_formula.tex").write_text(
        r"\[z_{ij}=(\widetilde{x}_{ij}-\mu_j)/\sigma_j,\quad \widehat{p}_i=\{1+\exp[-(\beta_0+\sum_j\beta_j z_{ij})]\}^{-1}.\]"
        + "\n% Values in final_candidate_model_preprocessing.csv; indicator columns precede scaling; intercept in JSON.",
        encoding="utf-8",
    )
    text = (
        "# Final candidate model specification\n\n"
        "**Post hoc frozen candidate; not prespecified or externally validated.**\n\n"
        + json.dumps(specification, indent=2)
        + "\n\n## Inputs and computation\n\n"
        "Raw predictor order is given above. Holter inputs require the exact baseline record, current "
        "src/full_features.py extraction (first signal channel, skip 60 s, complete 300 s windows), and current "
        "aggregation rules. The predictor dictionary and source code provide definitions, transformations, valid "
        "ranges and missing rules; unit uncertainties in historical clinical fields remain unresolved and must not "
        "be silently guessed.\n\n"
        "The serialized sklearn pipeline accepts an ordered DataFrame of already-defined patient-level predictors. "
        "Impute using frozen medians, append frozen missing indicators, subtract scaler mean and divide by scaler "
        "scale, multiply by coefficient vector, add the saved intercept, and apply logistic probability. CSVs "
        "contain every numerical value; final_candidate_formula.tex is the LaTeX source. No input can be reordered "
        "silently. The prediction helper validates exact columns and does not retrain.\n\n"
        "## Model rule and validation\n\n"
        "Full-development AP screening and AP tuning use the same 25 libraries, grid, preprocessing and seed 0. "
        "Tie rules were frozen before new analysis. No outer performance or library frequency was read by this "
        "script. Full-data probabilities were used only to verify serialization/manual calculation, not to calculate "
        "AUC/AP or other apparent performance. Primary validation remains the sealed procedure-level nested OOF "
        "results; these do not provide independent validation of this exact post hoc frozen artifact.\n"
    )
    (output_dir / "final_candidate_model_specification.md").write_text(text, encoding="utf-8")
    print(json.dumps(specification))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
