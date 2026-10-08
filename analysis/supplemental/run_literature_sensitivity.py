"""Evaluate the frozen literature-constrained predictor set with nested CV."""
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
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
    parser.add_argument("--protocol-dir", type=Path, required=True,
                        help="directory with literature_protocol_freeze.json and literature_constrained_protocol.md")
    return parser


def _load_core(project_root: Path):
    root = str(project_root)
    if root not in sys.path:
        sys.path.insert(0, root)
    from analysis.validate_nested_selector import build_selector_frame
    from analysis.validate_p3 import make_pipeline, C_GRID, L1_RATIO_GRID, metric_row
    return build_selector_frame, make_pipeline, C_GRID, L1_RATIO_GRID, metric_row


def task(seed: int, project_root: str, output_dir: str, protocol_dir: str) -> dict[str, object]:
    import numpy as np
    import pandas as pd
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.model_selection import StratifiedKFold, GridSearchCV, cross_val_score

    project = Path(project_root)
    output = Path(output_dir)
    protocols = Path(protocol_dir)
    build_selector_frame, make_pipeline, C_GRID, L1_RATIO_GRID, metric_row = _load_core(project)
    frozen = json.loads((protocols / "literature_protocol_freeze.json").read_text(encoding="utf-8"))
    document_hash = hashlib.sha256(
        (protocols / "literature_constrained_protocol.md").read_bytes()
    ).hexdigest()
    assert document_hash == frozen["protocol_sha256"]
    cols = frozen["predictors"]
    frame = build_selector_frame()[0]
    y = frame.label.to_numpy(int)
    pred = np.full(len(frame), np.nan)
    rows = []
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        for fold, (tr, te) in enumerate(
            StratifiedKFold(5, shuffle=True, random_state=seed).split(np.zeros(len(y)), y), 1
        ):
            splits = list(StratifiedKFold(3, shuffle=True, random_state=seed).split(np.zeros(len(tr)), y[tr]))
            screen = make_pipeline(seed)
            screen.set_params(clf__C=0.1, clf__l1_ratio=0.5)
            screen_ap = float(cross_val_score(
                screen, frame.iloc[tr][cols], y[tr], cv=splits,
                scoring="average_precision", n_jobs=1, error_score="raise",
            ).mean())
            search = GridSearchCV(
                make_pipeline(seed),
                {"clf__C": list(C_GRID), "clf__l1_ratio": list(L1_RATIO_GRID)},
                cv=splits, scoring="average_precision", n_jobs=1, error_score="raise",
            )
            search.fit(frame.iloc[tr][cols], y[tr])
            pred[te] = search.predict_proba(frame.iloc[te][cols])[:, 1]
            rows.append({
                "seed": seed, "outer_fold": fold, "C": search.best_params_["clf__C"],
                "l1_ratio": search.best_params_["clf__l1_ratio"], "screen_AP": screen_ap,
                "test_n": len(te), "test_events": int(y[te].sum()),
            })
    assert np.isfinite(pred).all()
    batch_dir = output / "analysis_outputs" / "literature_batches"
    batch_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"patient_id": frame.patient_id, "true_label": y, "probability": pred, "seed": seed}).to_parquet(
        batch_dir / f"seed_{seed:02d}.parquet", index=False
    )
    pd.DataFrame(rows).to_csv(batch_dir / f"folds_{seed:02d}.csv", index=False)
    return {
        "seed": seed,
        "warnings": sum(issubclass(w.category, ConvergenceWarning) for w in caught),
        **metric_row(y, pred),
    }


def main() -> int:
    import numpy as np
    import pandas as pd

    args = build_parser().parse_args()
    project_root = args.project_root.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    protocol_dir = args.protocol_dir.expanduser().resolve()
    if not project_root.is_dir():
        raise FileNotFoundError(f"project root does not exist: {project_root}")
    freeze_path = protocol_dir / "literature_protocol_freeze.json"
    document_path = protocol_dir / "literature_constrained_protocol.md"
    if not freeze_path.is_file() or not document_path.is_file():
        raise FileNotFoundError(f"protocol directory must contain {freeze_path.name} and {document_path.name}")
    start = time.time()
    rows = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=12) as pool:
        futures = [
            pool.submit(task, seed, str(project_root), str(output_dir), str(protocol_dir))
            for seed in range(100)
        ]
        for future in concurrent.futures.as_completed(futures):
            rows.append(future.result())
            if len(rows) % 20 == 0:
                print("Literature sensitivity", len(rows), flush=True)
    results = pd.DataFrame(rows).sort_values("seed")
    output_dir.mkdir(parents=True, exist_ok=True)
    results.to_csv(output_dir / "literature_sensitivity_per_seed.csv", index=False)
    summary = {
        "status": "PASS",
        "post_hoc": True,
        "n": 878,
        "events": 37,
        "seeds": 100,
        "outer_models": 500,
        "elapsed_seconds": time.time() - start,
        "metrics": {
            metric: {
                "median": float(results[metric].median()),
                "p2.5": float(np.percentile(results[metric], 2.5)),
                "p97.5": float(np.percentile(results[metric], 97.5)),
            }
            for metric in ("AUC", "AP", "Brier", "BSS")
        },
        "warnings": int(results.warnings.sum()),
        "interpretation": "Limited literature-constrained clinical contrast; changes information and complexity, does not isolate historical design bias.",
    }
    (output_dir / "literature_sensitivity_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
