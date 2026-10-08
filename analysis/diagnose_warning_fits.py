"""Sequential diagnostic replay of the frozen selector with fit-level warnings.

This script does not update the model or the primary result. Select a seed
range explicitly; it writes per-fit diagnostics and compares each replay with
the archived OOF prediction for that seed. Sequential fitting makes warning
provenance observable.
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import average_precision_score
from sklearn.model_selection import ParameterGrid, StratifiedKFold


def _load_core(project_root: Path):
    root = str(project_root)
    if root not in sys.path:
        sys.path.insert(0, root)
    from analysis.validate_nested_selector import build_selector_frame, SCREEN_C, SCREEN_L1_RATIO
    from analysis.validate_p3 import make_pipeline, C_GRID, L1_RATIO_GRID, OUTER_FOLDS, INNER_FOLDS
    return build_selector_frame, SCREEN_C, SCREEN_L1_RATIO, make_pipeline, C_GRID, L1_RATIO_GRID, OUTER_FOLDS, INNER_FOLDS


def fit_one(X, y, seed, stage, recipe, outer_fold, inner_fold, C, l1_ratio, records, make_pipeline):
    estimator = make_pipeline(seed)
    estimator.set_params(clf__C=C, clf__l1_ratio=l1_ratio)
    with warnings.catch_warnings(record=True) as seen:
        warnings.simplefilter("always", ConvergenceWarning)
        estimator.fit(X, y)
    tagged = [warning for warning in seen if issubclass(warning.category, ConvergenceWarning)]
    records.append({
        "seed": seed,
        "outer_fold": outer_fold,
        "inner_fold": inner_fold,
        "stage": stage,
        "recipe": recipe,
        "C": C,
        "l1_ratio": l1_ratio,
        "max_iter": estimator.named_steps["clf"].max_iter,
        "n_iter_": int(np.max(estimator.named_steps["clf"].n_iter_)),
        "ConvergenceWarning": bool(tagged),
        "warning_count": len(tagged),
    })
    return estimator


def diagnose(seed, frame, recipes, *, SCREEN_C, SCREEN_L1_RATIO, make_pipeline,
             C_GRID, L1_RATIO_GRID, OUTER_FOLDS, INNER_FOLDS):
    records = []
    y = frame.label.to_numpy(dtype=int)
    outer = StratifiedKFold(n_splits=OUTER_FOLDS, shuffle=True, random_state=seed)
    probabilities = np.full(len(frame), np.nan)
    for outer_fold, (train, test) in enumerate(outer.split(np.zeros(len(y)), y), 1):
        y_train = y[train]
        inner = StratifiedKFold(n_splits=INNER_FOLDS, shuffle=True, random_state=seed)
        splits = list(inner.split(np.zeros(len(train)), y_train))
        screen = {}
        for recipe in sorted(recipes):
            X = frame.iloc[train][recipes[recipe]]
            fold_scores = []
            for inner_fold, (inner_train, inner_test) in enumerate(splits, 1):
                estimator = fit_one(
                    X.iloc[inner_train], y_train[inner_train], seed, "recipe_screening", recipe,
                    outer_fold, inner_fold, SCREEN_C, SCREEN_L1_RATIO, records, make_pipeline,
                )
                fold_scores.append(average_precision_score(
                    y_train[inner_test], estimator.predict_proba(X.iloc[inner_test])[:, 1]
                ))
            screen[recipe] = float(np.mean(fold_scores))
        selected = max(sorted(screen), key=lambda name: screen[name])
        X = frame.iloc[train][recipes[selected]]
        best_score = -np.inf
        best_params = None
        for params in ParameterGrid({"C": list(C_GRID), "l1_ratio": list(L1_RATIO_GRID)}):
            fold_scores = []
            for inner_fold, (inner_train, inner_test) in enumerate(splits, 1):
                estimator = fit_one(
                    X.iloc[inner_train], y_train[inner_train], seed, "hyperparameter_search", selected,
                    outer_fold, inner_fold, params["C"], params["l1_ratio"], records, make_pipeline,
                )
                fold_scores.append(average_precision_score(
                    y_train[inner_test], estimator.predict_proba(X.iloc[inner_test])[:, 1]
                ))
            score = float(np.mean(fold_scores))
            if score > best_score:
                best_score = score
                best_params = params
        assert best_params is not None
        estimator = fit_one(
            X, y_train, seed, "outer_final_refit", selected, outer_fold, 0,
            best_params["C"], best_params["l1_ratio"], records, make_pipeline,
        )
        probabilities[test] = np.clip(
            estimator.predict_proba(frame.iloc[test][recipes[selected]])[:, 1], 0, 1
        )
    assert np.isfinite(probabilities).all()
    return pd.DataFrame(records), probabilities


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True,
                        help="portable project checkout containing analysis/ and data/")
    parser.add_argument("--results-root", type=Path, required=True,
                        help="archived results directory containing final100/selector/batches")
    parser.add_argument("--seed-start", type=int, required=True)
    parser.add_argument("--seed-stop", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    project_root = args.project_root.expanduser().resolve()
    results_root = args.results_root.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if not project_root.is_dir():
        raise FileNotFoundError(f"project root does not exist: {project_root}")
    if not 0 <= args.seed_start <= args.seed_stop <= 99:
        raise ValueError("seed range must satisfy 0 <= --seed-start <= --seed-stop <= 99")
    (build_selector_frame, screen_c, screen_l1_ratio, make_pipeline, c_grid, l1_ratio_grid,
     outer_folds, inner_folds) = _load_core(project_root)
    frame, recipes, _ = build_selector_frame()
    output_dir.mkdir(parents=True, exist_ok=True)
    all_records = []
    archive_root = results_root / "final100" / "selector" / "batches"
    for seed in range(args.seed_start, args.seed_stop + 1):
        records, probabilities = diagnose(
            seed, frame, recipes,
            SCREEN_C=screen_c,
            SCREEN_L1_RATIO=screen_l1_ratio,
            make_pipeline=make_pipeline,
            C_GRID=c_grid,
            L1_RATIO_GRID=l1_ratio_grid,
            OUTER_FOLDS=outer_folds,
            INNER_FOLDS=inner_folds,
        )
        batch = (seed // 10) * 10
        archive = archive_root / f"batch_{batch:02d}_{batch + 9:02d}" / "oof.parquet"
        stored = pd.read_parquet(archive).query("seed == @seed").set_index("patient_id")
        ordered = stored.loc[frame.patient_id.astype(str)]
        np.testing.assert_allclose(probabilities, ordered.probability.to_numpy(), rtol=0, atol=1e-10)
        records.to_csv(output_dir / f"fit_diagnostics_seed_{seed:02d}.csv", index=False)
        all_records.append(records)
        print(
            f"seed={seed} fits={len(records)} warnings={int(records.warning_count.sum())} archive_match=PASS",
            flush=True,
        )
    combined = pd.concat(all_records, ignore_index=True)
    combined.to_csv(output_dir / "fit_diagnostics_summary.csv", index=False)
    print(combined.groupby("stage", sort=False).warning_count.sum().to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
