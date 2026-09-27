from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from scripts.validate_compact_candidates import build_candidate_frame
from scripts.validate_p3 import ENDPOINT_DAYS, INNER_FOLDS, OUTER_FOLDS, _distribution, metric_row, validate_one_seed

SUBJECTS_PATH = REPO_ROOT / "data" / "cohort" / "subjects.parquet"
BASE = "COMPACT23"

# Each candidate adds exactly one clinically interpretable input to COMPACT23.
CANDIDATES: dict[str, tuple[str, str]] = {
    "QRS24": ("QRS duration (ms)", "qrs_ms"),
    "QTC24": ("QT corrected ", "qtc_ms"),
    "NSVT24": ("Non-sustained ventricular tachycardia (CH>10)", "nsvt"),
    "VT24": ("Ventricular Tachycardia", "ventricular_tachycardia"),
    "MI24": ("Prior Myocardial Infarction (yes=1)", "prior_mi"),
    "SYNCOPE24": ("Syncope", "syncope_any"),
    "SEX24": ("Gender (male=1)", "male"),
    "PROBNP24": ("Pro-BNP (ng/L)", "log1p_pro_bnp"),
    "CREAT24": ("Creatinine (?mol/L)", "log1p_creatinine"),
    "SODIUM24": ("Sodium (mEq/L)", "sodium"),
}

MODEL_NAMES = (BASE, *CANDIDATES.keys())


def _numeric(series: pd.Series) -> pd.Series:
    text = series.astype("string").str.strip().str.strip("'").str.strip('"')
    text = text.replace({"": pd.NA, "NA": pd.NA, "N/A": pd.NA, "<NA>": pd.NA, "None": pd.NA})
    text = text.str.replace(",", ".", regex=False)
    out = pd.to_numeric(text, errors="coerce").astype("float64")
    # MUSIC uses 999/9999-style sentinels in several clinical/Holter fields.
    out = out.mask(out >= 999)
    return out


def build_models() -> tuple[pd.DataFrame, dict[str, list[str]], dict[str, Any]]:
    frame, existing_models, meta = build_candidate_frame()
    base_cols = list(existing_models["ASSISTANT_COMPACT23"])

    subjects = pd.read_parquet(SUBJECTS_PATH)
    subjects["patient_id"] = subjects["patient_id"].astype("string")
    needed = ["patient_id", *[src for src, _ in CANDIDATES.values()]]
    missing = sorted(set(needed) - set(subjects.columns))
    if missing:
        raise ValueError(f"subjects table missing candidate columns: {missing}")
    extra = subjects.loc[:, needed].copy()

    # Parse/transform without using the outcome.
    for model, (source, dest) in CANDIDATES.items():
        raw = _numeric(extra[source])
        if model == "NSVT24":
            extra[dest] = np.where(raw.isna(), np.nan, (raw > 0).astype(float))
        elif model == "VT24":
            extra[dest] = np.where(raw.isna(), np.nan, (raw > 0).astype(float))
        elif model == "MI24":
            extra[dest] = np.where(raw.isna(), np.nan, (raw > 0).astype(float))
        elif model == "SYNCOPE24":
            extra[dest] = np.where(raw.isna(), np.nan, (raw > 0).astype(float))
        elif model == "SEX24":
            extra[dest] = raw
        elif model in {"PROBNP24", "CREAT24"}:
            if (raw.dropna() < 0).any():
                raise RuntimeError(f"negative values found in {source}")
            extra[dest] = np.log1p(raw)
        else:
            extra[dest] = raw

    extra = extra.drop(columns=[src for src, _ in CANDIDATES.values()])
    frame = frame.merge(extra, on="patient_id", how="left", validate="one_to_one")

    models: dict[str, list[str]] = {BASE: base_cols}
    model_meta: dict[str, Any] = {
        BASE: {
            "raw_feature_count": len(base_cols),
            "features": base_cols,
            "added_feature": None,
        }
    }
    for model, (_, dest) in CANDIDATES.items():
        cols = base_cols + [dest]
        models[model] = cols
        model_meta[model] = {
            "raw_feature_count": len(cols),
            "features": cols,
            "added_feature": dest,
            "missing_count": int(frame[dest].isna().sum()),
        }

    for col in {c for cols in models.values() for c in cols}:
        frame[col] = pd.to_numeric(frame[col], errors="coerce").astype("float64")

    if len(frame) != 878 or int(frame["label"].sum()) != 37:
        raise RuntimeError("optimization cohort drift")

    return frame, models, {
        "patient_count": int(len(frame)),
        "positive_count": int(frame["label"].sum()),
        "negative_count": int(len(frame) - frame["label"].sum()),
        "models": model_meta,
    }


def command_batch(args: argparse.Namespace) -> int:
    frame, models, meta = build_models()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    metrics: list[dict[str, Any]] = []
    oof: list[pd.DataFrame] = []
    folds: list[pd.DataFrame] = []
    for seed in range(args.seed_start, args.seed_stop + 1):
        for model in MODEL_NAMES:
            summary, pred, fold = validate_one_seed(
                frame, models[model], model_name=model, seed=seed
            )
            metrics.append(summary)
            oof.append(pred)
            folds.append(fold)

    pd.DataFrame(metrics).to_csv(out / "per_seed.csv", index=False)
    pd.concat(oof, ignore_index=True).to_parquet(out / "oof.parquet", index=False, compression="zstd")
    pd.concat(folds, ignore_index=True).to_csv(out / "folds.csv", index=False)
    (out / "contract.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return 0


def _bootstrap_delta(
    y: np.ndarray,
    base: np.ndarray,
    cand: np.ndarray,
    *,
    seed: int,
    n_resamples: int,
) -> dict[str, float | int]:
    rng = np.random.default_rng(seed)
    values: list[float] = []
    for _ in range(n_resamples):
        idx = rng.integers(0, len(y), size=len(y))
        yy = y[idx]
        if np.unique(yy).size < 2:
            continue
        values.append(float(roc_auc_score(yy, cand[idx]) - roc_auc_score(yy, base[idx])))
    arr = np.asarray(values)
    return {
        "lower": float(np.percentile(arr, 2.5)),
        "upper": float(np.percentile(arr, 97.5)),
        "median": float(np.median(arr)),
        "n_valid": int(len(arr)),
    }


def command_aggregate(args: argparse.Namespace) -> int:
    root = Path(args.input_root)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    per_paths = sorted(root.rglob("per_seed.csv"))
    oof_paths = sorted(root.rglob("oof.parquet"))
    contract_paths = sorted(root.rglob("contract.json"))
    if len(per_paths) != 10 or len(oof_paths) != 10:
        raise RuntimeError(f"expected 10 batches, got {len(per_paths)} metrics/{len(oof_paths)} OOF")

    per = pd.concat([pd.read_csv(p) for p in per_paths], ignore_index=True)
    oof = pd.concat([pd.read_parquet(p) for p in oof_paths], ignore_index=True)
    contract = json.loads(contract_paths[0].read_text(encoding="utf-8"))

    expected = set(range(100))
    for model in MODEL_NAMES:
        seeds = set(per.loc[per["model"].eq(model), "seed"].astype(int))
        if seeds != expected:
            raise RuntimeError(f"{model} missing seeds")

    patient = (
        oof.groupby(["patient_id", "true_label", "model"], as_index=False)["probability"]
        .mean()
        .pivot(index=["patient_id", "true_label"], columns="model", values="probability")
        .reset_index()
    )
    y = patient["true_label"].to_numpy(dtype=int)

    result: dict[str, Any] = {
        "cohort": {k: contract[k] for k in ("patient_count", "positive_count", "negative_count")},
        "models": contract["models"],
        "cv": {
            "endpoint_days": ENDPOINT_DAYS,
            "outer_folds": OUTER_FOLDS,
            "inner_folds": INNER_FOLDS,
            "seeds": "0..99",
            "model": "elastic-net logistic regression",
            "selection_metric": "average_precision",
        },
        "metrics": {},
    }

    for model in MODEL_NAMES:
        subset = per.loc[per["model"].eq(model)].sort_values("seed")
        pm = metric_row(y, patient[model].to_numpy(dtype=float))
        result["metrics"][model] = {
            "per_seed_AUC": _distribution(subset["AUC"]),
            "per_seed_AP": _distribution(subset["AP"]),
            "per_seed_Brier": _distribution(subset["Brier"]),
            "per_seed_BSS": _distribution(subset["BSS"]),
            "patient_mean_oof": pm,
        }

    base_auc_seed = per.loc[per["model"].eq(BASE)].set_index("seed")["AUC"].sort_index()
    base_patient = patient[BASE].to_numpy(dtype=float)
    for model in MODEL_NAMES:
        if model == BASE:
            continue
        cand_auc_seed = per.loc[per["model"].eq(model)].set_index("seed")["AUC"].sort_index()
        delta_seed = cand_auc_seed - base_auc_seed
        result["metrics"][model]["delta_vs_base_seed_AUC"] = _distribution(delta_seed)
        result["metrics"][model]["higher_equal_lower_vs_base"] = [
            int((delta_seed > 0).sum()),
            int((delta_seed == 0).sum()),
            int((delta_seed < 0).sum()),
        ]
        result["metrics"][model]["delta_vs_base_patient_AUC"] = float(
            result["metrics"][model]["patient_mean_oof"]["AUC"]
            - result["metrics"][BASE]["patient_mean_oof"]["AUC"]
        )
        result["metrics"][model]["delta_vs_base_patient_AUC_bootstrap"] = _bootstrap_delta(
            y,
            base_patient,
            patient[model].to_numpy(dtype=float),
            seed=args.bootstrap_seed,
            n_resamples=args.bootstrap_resamples,
        )

    # Primary ranking is repeated nested-CV median AUC; patient-mean OOF AUC is secondary.
    ranked = sorted(
        MODEL_NAMES,
        key=lambda m: (
            result["metrics"][m]["per_seed_AUC"]["median"],
            result["metrics"][m]["patient_mean_oof"]["AUC"],
        ),
        reverse=True,
    )
    result["ranking_by_repeated_cv_median_auc"] = ranked
    result["best_model"] = ranked[0]
    result["target"] = {
        "auc": 0.70,
        "best_repeated_cv_median_reaches": bool(
            result["metrics"][ranked[0]]["per_seed_AUC"]["median"] >= 0.70
        ),
        "best_patient_mean_oof_reaches": bool(
            result["metrics"][ranked[0]]["patient_mean_oof"]["AUC"] >= 0.70
        ),
    }

    rows = []
    for model in ranked:
        m = result["metrics"][model]
        rows.append({
            "model": model,
            "inputs": contract["models"][model]["raw_feature_count"],
            "added_feature": contract["models"][model].get("added_feature"),
            "auc_median_100": m["per_seed_AUC"]["median"],
            "auc_p2_5": m["per_seed_AUC"]["p2_5"],
            "auc_p97_5": m["per_seed_AUC"]["p97_5"],
            "patient_mean_auc": m["patient_mean_oof"]["AUC"],
            "ap_median_100": m["per_seed_AP"]["median"],
            "brier_median_100": m["per_seed_Brier"]["median"],
            "bss_median_100": m["per_seed_BSS"]["median"],
        })
    pd.DataFrame(rows).to_csv(out / "candidate_ranking.csv", index=False)
    patient.to_csv(out / "patient_mean_oof.csv", index=False)
    (out / "optimization_summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    lines = [
        "# Compact24 candidate optimization",
        "",
        f"Cohort: {contract['patient_count']} patients; {contract['positive_count']} SCD / {contract['negative_count']} non-SCD.",
        "",
        "| Rank | Model | Inputs | Added feature | AUC median (100) | Patient-mean OOF AUC | AP median | Brier median |",
        "|---:|---|---:|---|---:|---:|---:|---:|",
    ]
    for i, row in enumerate(rows, 1):
        lines.append(
            f"| {i} | {row['model']} | {row['inputs']} | {row['added_feature'] or '-'} | "
            f"{row['auc_median_100']:.6f} | {row['patient_mean_auc']:.6f} | "
            f"{row['ap_median_100']:.6f} | {row['brier_median_100']:.6f} |"
        )
    lines += [
        "",
        f"Best model by repeated-CV median AUC: **{ranked[0]}**.",
        f"Repeated-CV median AUC >= 0.70: **{result['target']['best_repeated_cv_median_reaches']}**.",
        f"Patient-mean OOF AUC >= 0.70: **{result['target']['best_patient_mean_oof_reaches']}**.",
        "",
        "Candidate search is exploratory model development; the winning candidate still requires locked confirmation/external validation.",
    ]
    (out / "optimization_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="command", required=True)
    b = sub.add_parser("batch")
    b.add_argument("--seed-start", type=int, required=True)
    b.add_argument("--seed-stop", type=int, required=True)
    b.add_argument("--output-dir", type=Path, required=True)
    a = sub.add_parser("aggregate")
    a.add_argument("--input-root", type=Path, required=True)
    a.add_argument("--output-dir", type=Path, required=True)
    a.add_argument("--bootstrap-resamples", type=int, default=2000)
    a.add_argument("--bootstrap-seed", type=int, default=20260927)
    return p


def main() -> int:
    args = parser().parse_args()
    return command_batch(args) if args.command == "batch" else command_aggregate(args)


if __name__ == "__main__":
    raise SystemExit(main())
