"""Repeat nested selector validation with competing deaths coded as non-SCD."""
from __future__ import annotations

import argparse
import concurrent.futures
import inspect
import json
import os
import sys
import time
import warnings
from pathlib import Path

for _key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_key] = "1"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True,
                        help="portable project checkout containing analysis/ and data/")
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def _load_core(project_root: Path):
    root = str(project_root)
    if root not in sys.path:
        sys.path.insert(0, root)
    import analysis.validate_compact_candidates as compact
    import analysis.validate_nested_selector as selector
    import analysis.validate_p3 as p3
    from src.endpoints import build_endpoint
    return compact, selector, p3, build_endpoint


def prepare(project_root: Path, output_dir: Path) -> tuple[Path, Path]:
    compact, selector, p3, original_build_endpoint = _load_core(project_root)
    primary, recipes, _ = selector.build_selector_frame()
    subjects = pd_read_parquet(p3.SUBJECTS_PATH)
    endpoint = original_build_endpoint(subjects[subjects.has_holter], 365)
    keep = endpoint.endpoint_state.isin(["POSITIVE", "NEGATIVE", "COMPETING_EVENT"])
    n = int(keep.sum())
    events = int(endpoint.endpoint_state.eq("POSITIVE").sum())
    expected = (n, events, n - events)

    def endpoint_with_competing(subject_frame, days):
        out = original_build_endpoint(subject_frame, days)
        mask = out.endpoint_state.eq("COMPETING_EVENT")
        out.loc[mask, "endpoint_state"] = "NEGATIVE"
        out.loc[mask, "binary_label_if_evaluable"] = 0
        return out

    p3.build_endpoint = endpoint_with_competing
    selector.build_endpoint = endpoint_with_competing
    env = dict(p3.__dict__)
    source = inspect.getsource(p3.build_frame)
    expected_guard = "expected = (878, 37, 841)"
    if expected_guard not in source:
        raise RuntimeError("portable P3 builder no longer has the audited cohort guard")
    source = source.replace(expected_guard, f"expected = {expected}")
    exec(compile(source, str(Path(__file__).resolve()), "exec"), env)
    compact.build_frame = env["build_frame"]

    env2 = dict(selector.__dict__)
    source2 = inspect.getsource(selector.build_selector_frame)
    selector_guard = "observed != (878, 37, 841)"
    if selector_guard not in source2:
        raise RuntimeError("portable selector no longer has the audited cohort guard")
    source2 = source2.replace(selector_guard, f"observed != {expected}")
    exec(compile(source2, str(Path(__file__).resolve()), "exec"), env2)
    sensitivity, recipes2, _ = env2["build_selector_frame"]()
    assert recipes == recipes2
    subset = sensitivity.set_index("patient_id").loc[primary.patient_id].reset_index()
    pd.testing.assert_frame_equal(primary, subset, check_dtype=True)
    assert set(sensitivity.patient_id) - set(primary.patient_id) == set(
        endpoint.loc[endpoint.endpoint_state.eq("COMPETING_EVENT"), "patient_id"]
    )
    assert expected == (
        len(sensitivity), int(sensitivity.label.sum()), int(sensitivity.label.eq(0).sum())
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    frame_path = output_dir / "endpoint_sensitivity_frame.parquet"
    protocol_path = output_dir / "endpoint_sensitivity_protocol.json"
    sensitivity.to_parquet(frame_path, index=False)
    protocol_path.write_text(json.dumps({
        "status": "FROZEN_BEFORE_RUN",
        "cohort_counts": expected,
        "competing_as_non_scd": int(endpoint.endpoint_state.eq("COMPETING_EVENT").sum()),
        "early_censor_excluded": int(endpoint.endpoint_state.eq("CENSORED").sum()),
        "recipes": recipes,
        "seeds": list(range(100)),
        "outer_folds": 5,
        "inner_folds": 3,
        "scoring": "average_precision",
        "frame_validation": "All 878 original rows exactly identical across every column; only competing-death rows added. Current source functions reused with only endpoint definition and explicit cohort guards adapted in memory; current code files not modified.",
    }, indent=2), encoding="utf-8")
    return frame_path, protocol_path


def pd_read_parquet(path: Path):
    import pandas as pd
    return pd.read_parquet(path)


def task(seed: int, project_root: str, output_dir: str) -> dict[str, object]:
    import numpy as np
    import pandas as pd
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.model_selection import StratifiedKFold, GridSearchCV, cross_val_score

    _compact, _selector, p3, _build_endpoint = _load_core(Path(project_root))
    root = Path(output_dir)
    frame = pd.read_parquet(root / "endpoint_sensitivity_frame.parquet")
    recipes = json.loads((root / "endpoint_sensitivity_protocol.json").read_text(encoding="utf-8"))["recipes"]
    y = frame.label.to_numpy(int)
    pred = np.full(len(frame), np.nan)
    rows = []
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        for fold, (tr, te) in enumerate(
            StratifiedKFold(5, shuffle=True, random_state=seed).split(np.zeros(len(y)), y), 1
        ):
            splits = list(StratifiedKFold(3, shuffle=True, random_state=seed).split(np.zeros(len(tr)), y[tr]))
            scores = {}
            for name in sorted(recipes):
                estimator = p3.make_pipeline(seed)
                estimator.set_params(clf__C=0.1, clf__l1_ratio=0.5)
                scores[name] = float(cross_val_score(
                    estimator, frame.iloc[tr][recipes[name]], y[tr], scoring="average_precision",
                    cv=splits, n_jobs=1, error_score="raise",
                ).mean())
            selected = max(sorted(scores), key=lambda name: scores[name])
            cols = recipes[selected]
            search = GridSearchCV(
                p3.make_pipeline(seed),
                {"clf__C": list(p3.C_GRID), "clf__l1_ratio": list(p3.L1_RATIO_GRID)},
                scoring="average_precision", cv=splits, n_jobs=1, error_score="raise",
            )
            search.fit(frame.iloc[tr][cols], y[tr])
            pred[te] = search.predict_proba(frame.iloc[te][cols])[:, 1]
            rows.append({
                "seed": seed, "outer_fold": fold, "selected_recipe": selected,
                "C": search.best_params_["clf__C"], "l1_ratio": search.best_params_["clf__l1_ratio"],
                "train_n": len(tr), "test_n": len(te), "train_events": int(y[tr].sum()),
                "test_events": int(y[te].sum()),
            })
    assert np.isfinite(pred).all()
    folder = root / "analysis_outputs" / "endpoint_batches"
    folder.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"patient_id": frame.patient_id, "true_label": y, "probability": pred, "seed": seed}).to_parquet(
        folder / f"seed_{seed:02d}.parquet", index=False
    )
    pd.DataFrame(rows).to_csv(folder / f"folds_{seed:02d}.csv", index=False)
    return {
        "seed": seed,
        "convergence_warnings": sum(issubclass(w.category, ConvergenceWarning) for w in caught),
        **p3.metric_row(y, pred),
    }


def main() -> int:
    import numpy as np
    import pandas as pd

    args = build_parser().parse_args()
    project_root = args.project_root.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if not project_root.is_dir():
        raise FileNotFoundError(f"project root does not exist: {project_root}")
    prepare(project_root, output_dir)
    start = time.time()
    out = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=12) as pool:
        futures = [pool.submit(task, seed, str(project_root), str(output_dir)) for seed in range(100)]
        for future in concurrent.futures.as_completed(futures):
            out.append(future.result())
            pd.DataFrame(out).to_csv(output_dir / "endpoint_sensitivity_progress.csv", index=False)
            if len(out) % 10 == 0:
                print(f"Endpoint sensitivity {len(out)}/100; {time.time() - start:.1f}s", flush=True)
    d = pd.DataFrame(out).sort_values("seed")
    d.to_csv(output_dir / "endpoint_sensitivity_per_seed.csv", index=False)
    summary = {
        "status": "PASS",
        "elapsed_seconds": time.time() - start,
        "seeds": 100,
        "outer_models": 500,
        "warnings": int(d.convergence_warnings.sum()),
        "metrics": {
            metric: {
                "median": float(d[metric].median()),
                "p2.5": float(np.percentile(d[metric], 2.5)),
                "p97.5": float(np.percentile(d[metric], 97.5)),
            }
            for metric in ("AUC", "AP", "Brier", "BSS")
        },
        "estimand": "binary observed 365-day SCD status with competing deaths as non-SCD; not a Fine-Gray model",
    }
    (output_dir / "endpoint_sensitivity_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
