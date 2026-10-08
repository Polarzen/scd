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
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.endpoints import build_endpoint


FEATURE_PATH = REPO_ROOT / "data" / "features" / "full_5min" / "patient_features.parquet"
SUBJECTS_PATH = REPO_ROOT / "data" / "cohort" / "subjects.parquet"

AF_COMPATIBLE_BASES = (
    "sig_mean",
    "sig_std",
    "sig_p2p",
    "sig_skew",
    "sig_kurt",
    "beats",
    "beats_per_min",
    "pow_lf",
    "pow_mf",
    "pow_hf",
    "pow_hf_ratio",
)

P2_CLINICAL = ("af_flag", "pvc_count_24h")
P3_ADDED = ("age", "lvef", "nyha")

C_GRID = (0.001, 0.01, 0.1, 1.0, 3.0)
L1_RATIO_GRID = (0.0, 0.1, 0.25, 0.5, 0.75, 1.0)

OUTER_FOLDS = 5
INNER_FOLDS = 3
ENDPOINT_DAYS = 365


def _number(series: pd.Series) -> pd.Series:
    text = series.astype("string").str.strip().str.strip("'").str.strip('"')
    text = text.replace({"": pd.NA, "NA": pd.NA, "N/A": pd.NA, "<NA>": pd.NA, "None": pd.NA})
    text = text.str.replace(",", ".", regex=False)
    return pd.to_numeric(text, errors="coerce").astype("float64")


def _derived_p2_features(features: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame({"patient_id": features["patient_id"].astype("string")})
    for base in AF_COMPATIBLE_BASES:
        p10 = pd.to_numeric(features[f"{base}_p10"], errors="coerce").astype("float64")
        p50 = pd.to_numeric(features[f"{base}_p50"], errors="coerce").astype("float64")
        p90 = pd.to_numeric(features[f"{base}_p90"], errors="coerce").astype("float64")
        out[f"{base}_median"] = p50
        out[f"{base}_p90_p10"] = p90 - p10
    return out


def build_frame() -> tuple[pd.DataFrame, list[str], list[str], dict[str, Any]]:
    features = pd.read_parquet(FEATURE_PATH)
    subjects = pd.read_parquet(SUBJECTS_PATH)
    endpoint = build_endpoint(subjects, ENDPOINT_DAYS)

    required_feature_cols = {
        "patient_id",
        "processed_holter",
        "n_windows_successful",
        *[f"{base}_{suffix}" for base in AF_COMPATIBLE_BASES for suffix in ("p10", "p50", "p90")],
    }
    missing_feature_cols = sorted(required_feature_cols - set(features.columns))
    if missing_feature_cols:
        raise ValueError(f"patient feature table is missing required columns: {missing_feature_cols}")

    required_subject_cols = {
        "patient_id",
        "af_flag",
        "pvc_count_24h",
        "Age",
        "LVEF (%)",
        "NYHA class",
    }
    missing_subject_cols = sorted(required_subject_cols - set(subjects.columns))
    if missing_subject_cols:
        raise ValueError(f"subjects table is missing required columns: {missing_subject_cols}")

    eligible_endpoint = endpoint.loc[
        endpoint["endpoint_state"].isin(["POSITIVE", "NEGATIVE"]),
        ["patient_id", "binary_label_if_evaluable", "endpoint_state"],
    ].copy()
    eligible_endpoint["patient_id"] = eligible_endpoint["patient_id"].astype("string")
    eligible_endpoint["label"] = pd.to_numeric(
        eligible_endpoint["binary_label_if_evaluable"], errors="coerce"
    ).astype("Int64")

    signal_status = features.loc[
        :,
        ["patient_id", "processed_holter", "n_windows_successful"],
    ].copy()
    signal_status["patient_id"] = signal_status["patient_id"].astype("string")
    signal_status["processed_holter"] = signal_status["processed_holter"].fillna(False).astype(bool)
    signal_status["n_windows_successful"] = pd.to_numeric(
        signal_status["n_windows_successful"], errors="coerce"
    ).fillna(0)
    signal_status = signal_status.loc[
        signal_status["processed_holter"] & signal_status["n_windows_successful"].gt(0)
    ].copy()

    derived = _derived_p2_features(features)
    derived = signal_status[["patient_id"]].merge(
        derived, on="patient_id", how="inner", validate="one_to_one"
    )

    subject_vars = subjects.loc[
        :,
        ["patient_id", "af_flag", "pvc_count_24h", "Age", "LVEF (%)", "NYHA class"],
    ].copy()
    subject_vars["patient_id"] = subject_vars["patient_id"].astype("string")
    subject_vars["af_flag"] = subject_vars["af_flag"].astype("boolean").astype("Int64").astype("float64")
    subject_vars["pvc_count_24h"] = pd.to_numeric(subject_vars["pvc_count_24h"], errors="coerce").astype("float64")
    subject_vars["age"] = _number(subject_vars["Age"])
    subject_vars["lvef"] = _number(subject_vars["LVEF (%)"])
    subject_vars["nyha"] = _number(subject_vars["NYHA class"])
    subject_vars = subject_vars.drop(columns=["Age", "LVEF (%)", "NYHA class"])

    frame = eligible_endpoint.merge(derived, on="patient_id", how="inner", validate="one_to_one")
    frame = frame.merge(subject_vars, on="patient_id", how="inner", validate="one_to_one")
    frame = frame.loc[frame["label"].isin([0, 1])].copy()
    frame["label"] = frame["label"].astype(int)
    frame = frame.sort_values("patient_id", kind="stable").reset_index(drop=True)

    p2_cols = [
        column
        for base in AF_COMPATIBLE_BASES
        for column in (f"{base}_median", f"{base}_p90_p10")
    ] + list(P2_CLINICAL)
    p3_cols = p2_cols + list(P3_ADDED)

    for column in p3_cols:
        frame[column] = pd.to_numeric(frame[column], errors="coerce").astype("float64")

    counts = {
        "patient_count": int(len(frame)),
        "positive_count": int(frame["label"].sum()),
        "negative_count": int(len(frame) - frame["label"].sum()),
        "p2_raw_feature_count": len(p2_cols),
        "p3_raw_feature_count": len(p3_cols),
        "p3_added_features": list(P3_ADDED),
        "missing_counts": {column: int(frame[column].isna().sum()) for column in p3_cols},
    }

    # This hard gate ties the validation to the manuscript's current 365-day
    # binary cohort.  If the committed compact data changes, fail rather than
    # silently reporting a result from a different population.
    expected = (878, 37, 841)
    observed = (counts["patient_count"], counts["positive_count"], counts["negative_count"])
    if observed != expected:
        raise RuntimeError(f"P2/P3 cohort drift: observed {observed}, expected {expected}")

    return frame, p2_cols, p3_cols, counts


def make_pipeline(seed: int) -> Pipeline:
    return Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="median",
                    add_indicator=True,
                    keep_empty_features=True,
                ),
            ),
            ("scale", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    penalty="elasticnet",
                    solver="saga",
                    max_iter=5000,
                    tol=1e-4,
                    random_state=int(seed),
                ),
            ),
        ]
    )


def metric_row(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    prevalence = float(np.mean(y))
    brier = float(brier_score_loss(y, p))
    reference = float(prevalence * (1.0 - prevalence))
    return {
        "AUC": float(roc_auc_score(y, p)),
        "AP": float(average_precision_score(y, p)),
        "Brier": brier,
        "BSS": float(1.0 - brier / reference) if reference > 0 else float("nan"),
    }


def validate_one_seed(
    frame: pd.DataFrame,
    feature_cols: list[str],
    *,
    model_name: str,
    seed: int,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    y = frame["label"].to_numpy(dtype=int)
    splitter = StratifiedKFold(n_splits=OUTER_FOLDS, shuffle=True, random_state=int(seed))
    outer_splits = list(splitter.split(np.zeros(len(y)), y))
    probs = np.full(len(frame), np.nan, dtype=float)
    fold_rows: list[dict[str, Any]] = []

    for fold, (train_idx, test_idx) in enumerate(outer_splits, start=1):
        x_train = frame.iloc[train_idx][feature_cols]
        x_test = frame.iloc[test_idx][feature_cols]
        y_train = y[train_idx]
        y_test = y[test_idx]

        inner = StratifiedKFold(
            n_splits=INNER_FOLDS,
            shuffle=True,
            random_state=int(seed),
        )
        search = GridSearchCV(
            estimator=make_pipeline(seed),
            param_grid={
                "clf__C": list(C_GRID),
                "clf__l1_ratio": list(L1_RATIO_GRID),
            },
            scoring="average_precision",
            cv=inner,
            refit=True,
            n_jobs=-1,
            error_score="raise",
        )
        search.fit(x_train, y_train)
        fold_prob = np.asarray(search.best_estimator_.predict_proba(x_test)[:, 1], dtype=float)
        probs[test_idx] = np.clip(fold_prob, 0.0, 1.0)
        fold_metric = metric_row(y_test, probs[test_idx])
        fold_rows.append(
            {
                "model": model_name,
                "seed": int(seed),
                "outer_fold": int(fold),
                "train_n": int(len(train_idx)),
                "test_n": int(len(test_idx)),
                "train_positive": int(y_train.sum()),
                "test_positive": int(y_test.sum()),
                "best_C": float(search.best_params_["clf__C"]),
                "best_l1_ratio": float(search.best_params_["clf__l1_ratio"]),
                "inner_best_AP": float(search.best_score_),
                **fold_metric,
            }
        )

    if not np.isfinite(probs).all():
        raise AssertionError(f"{model_name}/seed{seed} has missing OOF predictions")

    metrics = metric_row(y, probs)
    summary = {
        "model": model_name,
        "seed": int(seed),
        "patient_count": int(len(frame)),
        "positive_count": int(y.sum()),
        "negative_count": int(len(y) - y.sum()),
        "raw_feature_count": int(len(feature_cols)),
        **metrics,
    }
    oof = pd.DataFrame(
        {
            "patient_id": frame["patient_id"].astype(str),
            "true_label": y,
            "probability": probs,
            "model": model_name,
            "seed": int(seed),
        }
    )
    return summary, oof, pd.DataFrame(fold_rows)


def command_batch(args: argparse.Namespace) -> int:
    frame, p2_cols, p3_cols, counts = build_frame()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)

    per_seed_rows: list[dict[str, Any]] = []
    oof_frames: list[pd.DataFrame] = []
    fold_frames: list[pd.DataFrame] = []

    for seed in range(int(args.seed_start), int(args.seed_stop) + 1):
        for model_name, cols in (("P2", p2_cols), ("P3", p3_cols)):
            summary, oof, folds = validate_one_seed(
                frame,
                cols,
                model_name=model_name,
                seed=seed,
            )
            per_seed_rows.append(summary)
            oof_frames.append(oof)
            fold_frames.append(folds)

    pd.DataFrame(per_seed_rows).to_csv(output / "per_seed.csv", index=False)
    pd.concat(oof_frames, ignore_index=True).to_parquet(
        output / "oof.parquet", index=False, compression="zstd"
    )
    pd.concat(fold_frames, ignore_index=True).to_csv(output / "folds.csv", index=False)
    (output / "cohort.json").write_text(
        json.dumps(counts, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(
        pd.DataFrame(per_seed_rows)[
            ["model", "seed", "AUC", "AP", "Brier", "BSS"]
        ].to_string(index=False)
    )
    return 0


def _distribution(values: pd.Series) -> dict[str, float]:
    x = pd.to_numeric(values, errors="coerce").dropna().to_numpy(dtype=float)
    return {
        "n": int(len(x)),
        "mean": float(np.mean(x)),
        "std": float(np.std(x, ddof=1)) if len(x) > 1 else 0.0,
        "min": float(np.min(x)),
        "p2_5": float(np.percentile(x, 2.5)),
        "median": float(np.median(x)),
        "p97_5": float(np.percentile(x, 97.5)),
        "max": float(np.max(x)),
    }


def _bootstrap_aggregate(
    patient: pd.DataFrame,
    *,
    n_resamples: int,
    seed: int,
) -> dict[str, Any]:
    y = patient["true_label"].to_numpy(dtype=int)
    p2 = patient["P2"].to_numpy(dtype=float)
    p3 = patient["P3"].to_numpy(dtype=float)
    rng = np.random.default_rng(int(seed))
    auc2: list[float] = []
    auc3: list[float] = []
    delta: list[float] = []
    n = len(y)
    for _ in range(int(n_resamples)):
        idx = rng.integers(0, n, size=n)
        yy = y[idx]
        if np.unique(yy).size < 2:
            continue
        a2 = float(roc_auc_score(yy, p2[idx]))
        a3 = float(roc_auc_score(yy, p3[idx]))
        auc2.append(a2)
        auc3.append(a3)
        delta.append(a3 - a2)

    def ci(values: list[float]) -> dict[str, float]:
        arr = np.asarray(values, dtype=float)
        return {
            "lower": float(np.percentile(arr, 2.5)),
            "upper": float(np.percentile(arr, 97.5)),
            "n_valid": int(len(arr)),
        }

    return {
        "n_resamples": int(n_resamples),
        "seed": int(seed),
        "P2_AUC": ci(auc2),
        "P3_AUC": ci(auc3),
        "delta_AUC_P3_minus_P2": ci(delta),
    }


def command_aggregate(args: argparse.Namespace) -> int:
    root = Path(args.input_root)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)

    per_seed_paths = sorted(root.rglob("per_seed.csv"))
    oof_paths = sorted(root.rglob("oof.parquet"))
    cohort_paths = sorted(root.rglob("cohort.json"))
    if len(per_seed_paths) != 10 or len(oof_paths) != 10:
        raise RuntimeError(
            f"expected 10 batch outputs; found {len(per_seed_paths)} per-seed and {len(oof_paths)} OOF files"
        )

    per_seed = pd.concat([pd.read_csv(path) for path in per_seed_paths], ignore_index=True)
    oof = pd.concat([pd.read_parquet(path) for path in oof_paths], ignore_index=True)

    expected_seeds = set(range(100))
    for model in ("P2", "P3"):
        seeds = set(per_seed.loc[per_seed["model"].eq(model), "seed"].astype(int))
        if seeds != expected_seeds:
            raise RuntimeError(f"{model} seed coverage is not exactly 0..99")
    if len(per_seed) != 200:
        raise RuntimeError(f"expected 200 model-seed rows, found {len(per_seed)}")

    summary: dict[str, Any] = {
        "cohort": json.loads(cohort_paths[0].read_text(encoding="utf-8")) if cohort_paths else {},
        "cv": {
            "endpoint_days": ENDPOINT_DAYS,
            "outer_folds": OUTER_FOLDS,
            "inner_folds": INNER_FOLDS,
            "seeds": "0..99",
            "selection_metric": "average_precision",
            "model": "elastic-net logistic regression",
            "C_grid": list(C_GRID),
            "l1_ratio_grid": list(L1_RATIO_GRID),
            "preprocessing": "training-fold median imputation + missing indicators + standardization",
        },
        "per_seed": {},
    }
    for model in ("P2", "P3"):
        subset = per_seed.loc[per_seed["model"].eq(model)].sort_values("seed")
        summary["per_seed"][model] = {
            metric: _distribution(subset[metric])
            for metric in ("AUC", "AP", "Brier", "BSS")
        }

    p2_seed = per_seed.loc[per_seed["model"].eq("P2")].set_index("seed").sort_index()
    p3_seed = per_seed.loc[per_seed["model"].eq("P3")].set_index("seed").sort_index()
    delta_auc = p3_seed["AUC"] - p2_seed["AUC"]
    delta_ap = p3_seed["AP"] - p2_seed["AP"]
    delta_brier = p3_seed["Brier"] - p2_seed["Brier"]
    summary["paired_seed_differences"] = {
        "AUC_P3_minus_P2": _distribution(delta_auc),
        "AP_P3_minus_P2": _distribution(delta_ap),
        "Brier_P3_minus_P2": _distribution(delta_brier),
        "P3_AUC_higher_seed_count": int((delta_auc > 0).sum()),
        "P3_AUC_equal_seed_count": int((delta_auc == 0).sum()),
        "P3_AUC_lower_seed_count": int((delta_auc < 0).sum()),
    }

    patient = (
        oof.groupby(["patient_id", "true_label", "model"], as_index=False)["probability"]
        .mean()
        .pivot(index=["patient_id", "true_label"], columns="model", values="probability")
        .reset_index()
    )
    if set(patient.columns) < {"patient_id", "true_label", "P2", "P3"}:
        raise RuntimeError("patient aggregation is missing P2 or P3 predictions")
    agg_metrics = {
        "P2": metric_row(patient["true_label"].to_numpy(int), patient["P2"].to_numpy(float)),
        "P3": metric_row(patient["true_label"].to_numpy(int), patient["P3"].to_numpy(float)),
    }
    agg_metrics["delta_AUC_P3_minus_P2"] = float(agg_metrics["P3"]["AUC"] - agg_metrics["P2"]["AUC"])
    summary["patient_mean_oof"] = agg_metrics
    summary["bootstrap_patient_mean_oof"] = _bootstrap_aggregate(
        patient,
        n_resamples=int(args.bootstrap_resamples),
        seed=int(args.bootstrap_seed),
    )
    summary["target_AUC_0_70"] = {
        "P3_per_seed_median_reaches_target": bool(summary["per_seed"]["P3"]["AUC"]["median"] >= 0.70),
        "P3_patient_mean_oof_reaches_target": bool(agg_metrics["P3"]["AUC"] >= 0.70),
    }

    per_seed.to_csv(output / "p2_p3_per_seed.csv", index=False)
    patient.to_csv(output / "p2_p3_patient_mean_oof.csv", index=False)
    (output / "p3_validation_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    p2_auc = summary["per_seed"]["P2"]["AUC"]
    p3_auc = summary["per_seed"]["P3"]["AUC"]
    paired = summary["paired_seed_differences"]["AUC_P3_minus_P2"]
    boot = summary["bootstrap_patient_mean_oof"]
    pm = summary["patient_mean_oof"]
    lines = [
        "# P3 complete internal validation",
        "",
        f"Cohort: {summary['cohort'].get('patient_count')} patients; "
        f"{summary['cohort'].get('positive_count')} SCD / {summary['cohort'].get('negative_count')} non-SCD.",
        "",
        "## 100 repeated nested-CV results",
        "",
        "| Model | Raw inputs | AUC median | AUC 2.5-97.5% split sensitivity | AP median | Brier median | BSS median |",
        "|---|---:|---:|---:|---:|---:|---:|",
        f"| P2 | {summary['cohort'].get('p2_raw_feature_count')} | {p2_auc['median']:.6f} | {p2_auc['p2_5']:.6f}-{p2_auc['p97_5']:.6f} | "
        f"{summary['per_seed']['P2']['AP']['median']:.6f} | {summary['per_seed']['P2']['Brier']['median']:.6f} | {summary['per_seed']['P2']['BSS']['median']:.6f} |",
        f"| P3 | {summary['cohort'].get('p3_raw_feature_count')} | {p3_auc['median']:.6f} | {p3_auc['p2_5']:.6f}-{p3_auc['p97_5']:.6f} | "
        f"{summary['per_seed']['P3']['AP']['median']:.6f} | {summary['per_seed']['P3']['Brier']['median']:.6f} | {summary['per_seed']['P3']['BSS']['median']:.6f} |",
        "",
        "## Paired P3 - P2",
        "",
        f"- Median AUC difference: {paired['median']:+.6f}",
        f"- Mean AUC difference: {paired['mean']:+.6f}",
        f"- AUC higher/equal/lower across seeds: "
        f"{summary['paired_seed_differences']['P3_AUC_higher_seed_count']}/"
        f"{summary['paired_seed_differences']['P3_AUC_equal_seed_count']}/"
        f"{summary['paired_seed_differences']['P3_AUC_lower_seed_count']}",
        "",
        "## Patient-level mean OOF prediction",
        "",
        f"- P2 AUC: {pm['P2']['AUC']:.6f} "
        f"(fixed-prediction bootstrap 95% {boot['P2_AUC']['lower']:.6f}-{boot['P2_AUC']['upper']:.6f})",
        f"- P3 AUC: {pm['P3']['AUC']:.6f} "
        f"(fixed-prediction bootstrap 95% {boot['P3_AUC']['lower']:.6f}-{boot['P3_AUC']['upper']:.6f})",
        f"- AUC difference P3-P2: {pm['delta_AUC_P3_minus_P2']:+.6f} "
        f"(paired fixed-prediction bootstrap 95% {boot['delta_AUC_P3_minus_P2']['lower']:+.6f} to "
        f"{boot['delta_AUC_P3_minus_P2']['upper']:+.6f})",
        "",
        "## 0.70 target",
        "",
        f"- 100-seed median P3 AUC >= 0.70: {summary['target_AUC_0_70']['P3_per_seed_median_reaches_target']}",
        f"- Patient-mean OOF P3 AUC >= 0.70: {summary['target_AUC_0_70']['P3_patient_mean_oof_reaches_target']}",
        "",
        "P2 = 11 AF-compatible 5-minute ECG features summarized by median and P90-P10, plus AF flag and 24 h PVC count.",
        "P3 = P2 plus Age, LVEF, and NYHA class.",
    ]
    (output / "p3_validation_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Paired P2/P3 repeated nested-CV validation")
    sub = parser.add_subparsers(dest="command", required=True)

    batch = sub.add_parser("batch")
    batch.add_argument("--seed-start", type=int, required=True)
    batch.add_argument("--seed-stop", type=int, required=True)
    batch.add_argument("--output-dir", type=Path, required=True)

    aggregate = sub.add_parser("aggregate")
    aggregate.add_argument("--input-root", type=Path, required=True)
    aggregate.add_argument("--output-dir", type=Path, required=True)
    aggregate.add_argument("--bootstrap-resamples", type=int, default=2000)
    aggregate.add_argument("--bootstrap-seed", type=int, default=20260927)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.command == "batch":
        return command_batch(args)
    if args.command == "aggregate":
        return command_aggregate(args)
    raise AssertionError(args.command)


if __name__ == "__main__":
    raise SystemExit(main())
