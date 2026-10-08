"""Patient bootstrap of the median AUC across fixed archived OOF predictions.

The bootstrap resamples patient IDs once per replicate and applies the same
draws to all 100 prediction sets. It conditions on the archived predictions;
it does not retrain or repeat model selection.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True,
                        help="portable project checkout (kept explicit for a reproducible invocation)")
    parser.add_argument("--results-root", type=Path, required=True,
                        help="archived results directory containing final100/selector")
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    project_root = args.project_root.expanduser().resolve()
    results_root = args.results_root.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if not project_root.is_dir():
        raise FileNotFoundError(f"project root does not exist: {project_root}")

    start = time.time()
    selector_root = results_root / "final100" / "selector"
    batch_paths = sorted((selector_root / "batches").glob("batch_*/oof.parquet"))
    if not batch_paths:
        raise FileNotFoundError(f"no archived OOF batches under {selector_root / 'batches'}")
    d = pd.concat([pd.read_parquet(path) for path in batch_paths], ignore_index=True)
    assert len(d) == 87800 and not d.duplicated(["patient_id", "seed"]).any()
    assert d.groupby("patient_id").true_label.nunique().eq(1).all()
    matrix = d.pivot(index="patient_id", columns="seed", values="probability").sort_index()
    assert matrix.shape == (878, 100) and matrix.columns.tolist() == list(range(100))
    y = d.groupby("patient_id").true_label.first().loc[matrix.index].to_numpy(int)
    p = matrix.to_numpy().T
    order = np.argsort(p, axis=1, kind="stable")
    scores = np.take_along_axis(p, order, axis=1)
    sy = y[order]
    first = np.empty(order.shape, int)
    last = np.empty(order.shape, int)
    for seed in range(100):
        _, idx, counts = np.unique(scores[seed], return_index=True, return_counts=True)
        first[seed] = np.repeat(idx, counts)
        last[seed] = np.repeat(idx + counts, counts)

    def aucs(weights: np.ndarray) -> np.ndarray:
        w = weights[order]
        nw = w * (1 - sy)
        pw = w * sy
        cumulative = np.pad(np.cumsum(nw, axis=1), ((0, 0), (1, 0)))
        before = np.take_along_axis(cumulative, first, axis=1)
        end = np.take_along_axis(cumulative, last, axis=1)
        return np.sum(pw * (before + 0.5 * (end - before)), axis=1) / (
            pw.sum(axis=1) * nw.sum(axis=1)
        )

    observed = float(np.median(aucs(np.ones(878))))
    rng = np.random.default_rng(20261001)
    rows = []
    for replicate in range(2000):
        sampled_ids = rng.integers(0, 878, size=878)
        weights = np.bincount(sampled_ids, minlength=878)
        if weights @ y == 0 or weights @ (1 - y) == 0:
            raise RuntimeError("one-class bootstrap sample")
        replicate_aucs = aucs(weights)
        if replicate < 10:
            for seed in (0, 17, 99):
                assert abs(replicate_aucs[seed] - roc_auc_score(y[sampled_ids], p[seed, sampled_ids])) < 1e-12
        rows.append({
            "replicate": replicate + 1,
            "median_auc": float(np.median(replicate_aucs)),
            "scd_draws": int(weights @ y),
            "non_scd_draws": int(weights @ (1 - y)),
        })

    sealed_path = selector_root / "aggregate" / "nested_selector_per_seed.csv"
    sealed = pd.read_csv(sealed_path)
    assert abs(observed - sealed.AUC.median()) < 1e-12
    output_dir.mkdir(parents=True, exist_ok=True)
    out = pd.DataFrame(rows)
    out.to_csv(output_dir / "primary_auc_patient_bootstrap.csv", index=False)
    values = out.median_auc.to_numpy()
    ci = np.percentile(values, [2.5, 97.5])
    summary = {
        "observed_median_auc": observed,
        "bootstrap_median": float(np.median(values)),
        "bootstrap_mean": float(np.mean(values)),
        "bootstrap_bias": float(np.mean(values) - observed),
        "bias_definition": "mean(T_b)-T_observed",
        "percentile_95_ci": ci.tolist(),
        "basic_95_ci": [float(2 * observed - ci[1]), float(2 * observed - ci[0])],
        "B": 2000,
        "seed": 20261001,
        "n": 878,
        "events": 37,
        "seed_predictions": 100,
        "cluster": "patient; shared resampled ID vector across all 100 seeds",
        "conditioning": "fixed previously obtained OOF predictions; no retraining or model selection within bootstrap",
        "valid_replicates": 2000,
        "validation": "30 sklearn roc_auc_score checks on actual resamples, plus sealed observed median match; exact ties handled with half-credit",
        "elapsed_seconds": time.time() - start,
    }
    (output_dir / "primary_auc_bootstrap_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figures = output_dir / "figures"
    figures.mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(values, bins=40, color="#406d92", edgecolor="white")
    ax.axvline(observed, color="black", label="Observed median AUC")
    ax.axvline(ci[0], color="#a24040", ls="--", label="Conditional percentile 95% CI")
    ax.axvline(ci[1], color="#a24040", ls="--")
    ax.set(xlabel="Median AUC across 100 fixed OOF prediction sets", ylabel="Patient bootstrap replicates")
    ax.legend(fontsize=8)
    fig.tight_layout()
    for ext in ("png", "pdf", "svg"):
        fig.savefig(figures / f"figure_primary_auc_bootstrap.{ext}", dpi=400)
    print(json.dumps(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
