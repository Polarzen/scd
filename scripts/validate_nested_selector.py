from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score

from scripts.validate_compact_candidates import ASSISTANT_MODEL, build_candidate_frame
from scripts.validate_p3 import (
    C_GRID,
    ENDPOINT_DAYS,
    INNER_FOLDS,
    L1_RATIO_GRID,
    OUTER_FOLDS,
    SUBJECTS_PATH,
    _distribution,
    make_pipeline,
    metric_row,
)
from scripts.validate_p4_single_additions import CANDIDATES, candidate_value

SCREEN_C = 0.1
SCREEN_L1_RATIO = 0.5
PAIR_DROP = "sig_mean_median"
BASE_RECIPE = "BASE_COMPACT23"


def _num(series: pd.Series) -> pd.Series:
    text = series.astype("string").str.strip().str.strip("'").str.strip('"')
    text = text.replace({"": pd.NA, "NA": pd.NA, "N/A": pd.NA, "<NA>": pd.NA, "None": pd.NA})
    text = text.str.replace(",", ".", regex=False)
    return pd.to_numeric(text, errors="coerce").astype("float64")


def _flag(condition: pd.Series, valid: pd.Series) -> pd.Series:
    return pd.Series(np.where(valid, condition.astype(float), np.nan), index=condition.index, dtype="float64")


def build_selector_frame() -> tuple[pd.DataFrame, dict[str, list[str]], dict[str, Any]]:
    frame, model_features, _ = build_candidate_frame()
    base = list(model_features[ASSISTANT_MODEL])

    subjects = pd.read_parquet(SUBJECTS_PATH).copy()
    subjects["patient_id"] = subjects["patient_id"].astype("string")

    required_raw = {
        "patient_id",
        *CANDIDATES.values(),
        "Weight (kg)",
        "Height (cm)",
        "Left atrial size (mm)",
        "Normalized Troponin",
    }
    missing = sorted(required_raw - set(subjects.columns))
    if missing:
        raise RuntimeError(f"subjects table missing selector fields: {missing}")

    extra = pd.DataFrame({"patient_id": subjects["patient_id"]})
    for name in CANDIDATES:
        extra[name] = candidate_value(name, subjects)

    pro_bnp = _num(subjects["Pro-BNP (ng/L)"])
    extra["bnp_gt1000"] = _flag(pro_bnp > 1000.0, pro_bnp.notna() & (pro_bnp >= 0))

    qrs = _num(subjects["QRS duration (ms)"])
    extra["qrs_gt120"] = _flag(qrs > 120.0, qrs.between(40.0, 250.0))

    sodium = _num(subjects["Sodium (mEq/L)"])
    extra["sodium_le138"] = _flag(sodium <= 138.0, sodium.between(110.0, 170.0))

    troponin = _num(subjects["Normalized Troponin"])
    extra["troponin_positive"] = _flag(troponin > 1.0, troponin.notna() & (troponin >= 0))

    weight = _num(subjects["Weight (kg)"])
    height = _num(subjects["Height (cm)"])
    la = _num(subjects["Left atrial size (mm)"])
    bsa = np.sqrt((weight * height) / 3600.0)
    valid_bsa = weight.between(30.0, 250.0) & height.between(100.0, 230.0) & bsa.gt(0)
    la_index = la / bsa
    extra["la_index_gt26"] = _flag(la_index > 26.0, valid_bsa & la.between(10.0, 100.0))

    frame = frame.merge(extra, on="patient_id", how="left", validate="one_to_one")

    lvef = pd.to_numeric(frame["lvef"], errors="coerce").astype("float64")
    frame["lvef_le35"] = _flag(lvef <= 35.0, lvef.between(5.0, 90.0))

    recipes: dict[str, list[str]] = {BASE_RECIPE: base}

    for candidate in sorted(CANDIDATES):
        recipes[f"SINGLE_{candidate.upper()}"] = base + [candidate]

    pair_base = [column for column in base if column != PAIR_DROP]
    if len(pair_base) != 22:
        raise RuntimeError(f"expected 22 inputs after dropping {PAIR_DROP}, got {len(pair_base)}")
    for candidate in sorted(CANDIDATES):
        if candidate == "log_pro_bnp":
            continue
        recipes[f"BNP_PAIR_{candidate.upper()}"] = pair_base + ["log_pro_bnp", candidate]

    compact_ecg = [
        column
        for column in base
        if column not in {"af_flag", "log1p_pvc_count_24h", "age", "lvef", "nyha_III"}
    ]
    core17 = [column for column in compact_ecg if column != PAIR_DROP]
    if len(compact_ecg) != 18 or len(core17) != 17:
        raise RuntimeError("unexpected compact ECG feature count")

    scd5 = ["prior_mi", "la_index_gt26", "qrs_gt120", "nsvt", "bnp_gt1000"]
    recipes["MUSIC_SCD24"] = core17 + ["af_flag", "log1p_pvc_count_24h", *scd5]
    recipes["MUSIC_SCD23_NO_AF"] = core17 + ["log1p_pvc_count_24h", *scd5]
    recipes["MUSIC_SCD24_LVEF_NO_AF"] = core17 + ["log1p_pvc_count_24h", *scd5, "lvef_le35"]
    recipes["MUSIC_SCD24_TROPONIN_NO_AF"] = core17 + ["log1p_pvc_count_24h", *scd5, "troponin_positive"]
    recipes["MUSIC_SCD24_SODIUM_NO_AF"] = core17 + ["log1p_pvc_count_24h", *scd5, "sodium_le138"]

    for name, cols in recipes.items():
        if len(cols) > 24:
            raise RuntimeError(f"{name} exceeds the 24-input cap: {len(cols)}")
        if len(cols) != len(set(cols)):
            raise RuntimeError(f"{name} contains duplicate feature names")
        for column in cols:
            frame[column] = pd.to_numeric(frame[column], errors="coerce").astype("float64")

    observed = (len(frame), int(frame["label"].sum()), int((frame["label"] == 0).sum()))
    if observed != (878, 37, 841):
        raise RuntimeError(f"selector cohort drift: {observed}")

    meta = {
        "patient_count": observed[0],
        "positive_count": observed[1],
        "negative_count": observed[2],
        "endpoint_days": ENDPOINT_DAYS,
        "outer_folds": OUTER_FOLDS,
        "inner_folds": INNER_FOLDS,
        "screen_C": SCREEN_C,
        "screen_l1_ratio": SCREEN_L1_RATIO,
        "pair_drop": PAIR_DROP,
        "recipe_count": len(recipes),
        "recipes": {
            name: {
                "raw_feature_count": len(cols),
                "features": cols,
                "missing_counts": {c: int(frame[c].isna().sum()) for c in cols},
            }
            for name, cols in recipes.items()
        },
        "selection_rule": (
            "Within each outer-training fold only, screen every predeclared recipe "
            "by 3-fold average precision at fixed elastic-net settings; choose the "
            "best recipe, then tune C and l1_ratio by 3-fold AP on the same outer "
            "training data. The outer test fold is never used for recipe or "
            "hyperparameter selection."
        ),
    }
    return frame, recipes, meta


def validate_selector_seed(
    frame: pd.DataFrame,
    recipes: dict[str, list[str]],
    *,
    seed: int,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    y = frame["label"].to_numpy(dtype=int)
    outer = StratifiedKFold(n_splits=OUTER_FOLDS, shuffle=True, random_state=int(seed))
    probs = np.full(len(frame), np.nan, dtype=float)
    fold_rows: list[dict[str, Any]] = []

    for fold, (train_idx, test_idx) in enumerate(outer.split(np.zeros(len(y)), y), start=1):
        y_train = y[train_idx]
        y_test = y[test_idx]
        inner = StratifiedKFold(n_splits=INNER_FOLDS, shuffle=True, random_state=int(seed))
        inner_splits = list(inner.split(np.zeros(len(train_idx)), y_train))

        screening: dict[str, float] = {}
        for recipe_name in sorted(recipes):
            cols = recipes[recipe_name]
            estimator = make_pipeline(seed)
            estimator.set_params(clf__C=SCREEN_C, clf__l1_ratio=SCREEN_L1_RATIO)
            scores = cross_val_score(
                estimator,
                frame.iloc[train_idx][cols],
                y_train,
                scoring="average_precision",
                cv=inner_splits,
                n_jobs=-1,
                error_score="raise",
            )
            screening[recipe_name] = float(np.mean(scores))

        selected = max(sorted(screening), key=lambda name: screening[name])
        selected_cols = recipes[selected]
        search = GridSearchCV(
            estimator=make_pipeline(seed),
            param_grid={
                "clf__C": list(C_GRID),
                "clf__l1_ratio": list(L1_RATIO_GRID),
            },
            scoring="average_precision",
            cv=inner_splits,
            refit=True,
            n_jobs=-1,
            error_score="raise",
        )
        search.fit(frame.iloc[train_idx][selected_cols], y_train)
        fold_prob = search.best_estimator_.predict_proba(frame.iloc[test_idx][selected_cols])[:, 1]
        probs[test_idx] = np.clip(np.asarray(fold_prob, dtype=float), 0.0, 1.0)

        fold_rows.append(
            {
                "model": "NESTED_RECIPE_SELECTOR",
                "seed": int(seed),
                "outer_fold": int(fold),
                "train_n": int(len(train_idx)),
                "test_n": int(len(test_idx)),
                "train_positive": int(y_train.sum()),
                "test_positive": int(y_test.sum()),
                "selected_recipe": selected,
                "selected_raw_feature_count": int(len(selected_cols)),
                "screening_AP": float(screening[selected]),
                "best_C": float(search.best_params_["clf__C"]),
                "best_l1_ratio": float(search.best_params_["clf__l1_ratio"]),
                "inner_best_AP": float(search.best_score_),
                **metric_row(y_test, probs[test_idx]),
            }
        )

    if not np.isfinite(probs).all():
        raise AssertionError(f"seed {seed} has missing OOF probabilities")

    metrics = metric_row(y, probs)
    selected_counts = Counter(row["selected_recipe"] for row in fold_rows)
    summary = {
        "model": "NESTED_RECIPE_SELECTOR",
        "seed": int(seed),
        "patient_count": int(len(frame)),
        "positive_count": int(y.sum()),
        "negative_count": int(len(y) - y.sum()),
        "recipe_count": int(len(recipes)),
        "selected_recipe_counts": json.dumps(dict(sorted(selected_counts.items())), sort_keys=True),
        **metrics,
    }
    oof = pd.DataFrame(
        {
            "patient_id": frame["patient_id"].astype(str),
            "true_label": y,
            "probability": probs,
            "model": "NESTED_RECIPE_SELECTOR",
            "seed": int(seed),
        }
    )
    return summary, oof, pd.DataFrame(fold_rows)


def command_batch(args: argparse.Namespace) -> int:
    frame, recipes, meta = build_selector_frame()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    summaries, oofs, folds = [], [], []
    for seed in range(int(args.seed_start), int(args.seed_stop) + 1):
        summary, oof, fold = validate_selector_seed(frame, recipes, seed=seed)
        summaries.append(summary)
        oofs.append(oof)
        folds.append(fold)
    pd.DataFrame(summaries).to_csv(out / "per_seed.csv", index=False)
    oof_all = pd.concat(oofs, ignore_index=True)
    oof_all.to_csv(out / "oof.csv", index=False)
    oof_all.to_parquet(out / "oof.parquet", index=False, compression="zstd")
    pd.concat(folds, ignore_index=True).to_csv(out / "folds.csv", index=False)
    (out / "selector_contract.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return 0


def _bootstrap_auc(y: np.ndarray, p: np.ndarray, *, n_resamples: int, seed: int) -> dict[str, Any]:
    rng = np.random.default_rng(int(seed))
    values: list[float] = []
    n = len(y)
    for _ in range(int(n_resamples)):
        idx = rng.integers(0, n, size=n)
        yy = y[idx]
        if np.unique(yy).size < 2:
            continue
        values.append(float(roc_auc_score(yy, p[idx])))
    arr = np.asarray(values, dtype=float)
    return {
        "lower": float(np.percentile(arr, 2.5)),
        "upper": float(np.percentile(arr, 97.5)),
        "median": float(np.median(arr)),
        "n_valid": int(arr.size),
        "n_resamples": int(n_resamples),
        "seed": int(seed),
    }


def command_aggregate(args: argparse.Namespace) -> int:
    root = Path(args.input_root)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    per_paths = sorted(root.rglob("per_seed.csv"))
    oof_paths = sorted(root.rglob("oof.csv"))
    fold_paths = sorted(root.rglob("folds.csv"))
    contract_paths = sorted(root.rglob("selector_contract.json"))
    if len(per_paths) != 10 or len(oof_paths) != 10 or len(fold_paths) != 10:
        raise RuntimeError(
            f"expected 10 batches, found {len(per_paths)} metrics, {len(oof_paths)} OOF, {len(fold_paths)} folds"
        )

    per = pd.concat([pd.read_csv(p) for p in per_paths], ignore_index=True)
    oof = pd.concat([pd.read_csv(p) for p in oof_paths], ignore_index=True)
    folds = pd.concat([pd.read_csv(p) for p in fold_paths], ignore_index=True)
    if set(per["seed"].astype(int)) != set(range(100)):
        raise RuntimeError("selector seed coverage is not 0..99")

    patient = oof.groupby(["patient_id", "true_label"], as_index=False)["probability"].mean()
    y = patient["true_label"].to_numpy(dtype=int)
    p = patient["probability"].to_numpy(dtype=float)
    patient_metrics = metric_row(y, p)

    summary = {
        "contract": json.loads(contract_paths[0].read_text(encoding="utf-8")),
        "per_seed": {metric: _distribution(per[metric]) for metric in ("AUC", "AP", "Brier", "BSS")},
        "patient_mean_oof": patient_metrics,
        "bootstrap_auc": _bootstrap_auc(
            y,
            p,
            n_resamples=int(args.bootstrap_resamples),
            seed=int(args.bootstrap_seed),
        ),
        "selection_frequency": {
            str(k): int(v)
            for k, v in folds["selected_recipe"].value_counts().sort_values(ascending=False).items()
        },
        "target_0_70": {
            "cv_auc_median": float(_distribution(per["AUC"])["median"]),
            "patient_auc": float(patient_metrics["AUC"]),
            "cv_median_reaches": bool(_distribution(per["AUC"])["median"] >= 0.70),
            "patient_reaches": bool(patient_metrics["AUC"] >= 0.70),
        },
    }

    per.to_csv(out / "nested_selector_per_seed.csv", index=False)
    folds.to_csv(out / "nested_selector_folds.csv", index=False)
    patient.to_csv(out / "nested_selector_patient_mean_oof.csv", index=False)
    (out / "nested_selector_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    top = list(summary["selection_frequency"].items())[:8]
    lines = [
        "# Strict nested recipe-selection validation",
        "",
        f"Recipe library: {summary['contract']['recipe_count']} predeclared recipes; max raw inputs: 24.",
        "Recipe choice and hyperparameter tuning are confined to the outer-training fold.",
        "",
        f"100-seed CV AUC median: **{summary['per_seed']['AUC']['median']:.6f}**",
        f"Patient-mean OOF AUC: **{patient_metrics['AUC']:.6f}**",
        f"Bootstrap 95% AUC interval: **{summary['bootstrap_auc']['lower']:.6f}-{summary['bootstrap_auc']['upper']:.6f}**",
        f"CV median AUC >= 0.70: **{summary['target_0_70']['cv_median_reaches']}**",
        f"Patient AUC >= 0.70: **{summary['target_0_70']['patient_reaches']}**",
        "",
        "Most frequently selected recipes across 500 outer folds:",
    ]
    lines.extend([f"- {name}: {count}" for name, count in top])
    (out / "nested_selector_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0


def main() -> int:
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

    args = p.parse_args()
    if args.command == "batch":
        return command_batch(args)
    return command_aggregate(args)


if __name__ == "__main__":
    raise SystemExit(main())
