import warnings
from pathlib import Path
from typing import List, Tuple

import joblib
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.base import clone
from sklearn.calibration import calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore", category=UserWarning)

# =========================
# User config (edit here)
# =========================
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = str(BASE_DIR / "scd_dataset.csv")  # csv or xlsx
TARGET_COL = "scd_high_risk"  # binary target: 1 high risk, 0 low risk
ID_COLS = ["patient_id", "followup_days", "cause_of_death"]  # set [] if none

# If empty, script auto-selects all non-target, non-id columns.
HOLTER_FEATURES: List[str] = []
CLINICAL_FEATURES: List[str] = []

# Optional explicit categorical columns. If empty, auto-detected by dtype.
CATEGORICAL_COLS: List[str] = []

TEST_SIZE = 0.2
RANDOM_STATE = 42
CV_SPLITS = 5
N_BOOTSTRAP = 200  # for OR 95% CI

OUTPUT_DIR = BASE_DIR / "output_logistic"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def use_chinese_font_if_available() -> bool:
    """Try to enable a Chinese font for Matplotlib labels/titles."""
    preferred = [
        "Microsoft YaHei",
        "SimHei",
        "Noto Sans CJK SC",
        "Source Han Sans SC",
        "WenQuanYi Micro Hei",
    ]
    installed = {f.name for f in font_manager.fontManager.ttflist}
    for name in preferred:
        if name in installed:
            plt.rcParams["font.sans-serif"] = [name, "DejaVu Sans"]
            plt.rcParams["axes.unicode_minus"] = False
            return True
    return False


def load_table(path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Dataset not found: {p}")
    if p.suffix.lower() in {".csv", ".txt"}:
        return pd.read_csv(p)
    if p.suffix.lower() in {".xlsx", ".xls"}:
        return pd.read_excel(p)
    raise ValueError("Only csv/txt/xlsx/xls are supported.")


def normalize_target(y: pd.Series) -> pd.Series:
    if y.dtype.kind in {"i", "u", "f", "b"}:
        y_num = y.astype(float)
        uniq = set(pd.unique(y_num.dropna()))
        if uniq.issubset({0.0, 1.0}):
            return y_num.astype(int)

    y_str = y.astype(str).str.strip().str.lower()
    mapping = {
        "1": 1,
        "0": 0,
        "yes": 1,
        "no": 0,
        "true": 1,
        "false": 0,
        "high": 1,
        "low": 0,
        "positive": 1,
        "negative": 0,
    }
    y_map = y_str.map(mapping)
    if y_map.isna().any():
        bad = y[y_map.isna()].dropna().unique()[:10]
        raise ValueError(f"Target contains unsupported labels: {bad}")
    return y_map.astype(int)


def get_feature_cols(df: pd.DataFrame) -> List[str]:
    if HOLTER_FEATURES or CLINICAL_FEATURES:
        feats = HOLTER_FEATURES + CLINICAL_FEATURES
    else:
        drop_cols = {TARGET_COL, *ID_COLS}
        feats = [c for c in df.columns if c not in drop_cols]
    missing = [c for c in feats if c not in df.columns]
    if missing:
        raise ValueError(f"Feature columns not found in dataset: {missing}")
    return feats


def infer_cat_num_cols(df_x: pd.DataFrame) -> Tuple[List[str], List[str]]:
    if CATEGORICAL_COLS:
        cat_cols = [c for c in CATEGORICAL_COLS if c in df_x.columns]
    else:
        cat_cols = df_x.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    num_cols = [c for c in df_x.columns if c not in cat_cols]
    return cat_cols, num_cols


def build_pipeline(cat_cols: List[str], num_cols: List[str]) -> Pipeline:
    numeric_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocess = ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, num_cols),
            ("cat", categorical_pipe, cat_cols),
        ],
        remainder="drop",
    )

    model = LogisticRegression(class_weight="balanced", max_iter=5000)

    pipe = Pipeline(
        steps=[
            ("preprocess", preprocess),
            ("model", model),
        ]
    )
    return pipe


def evaluate_model(y_true: np.ndarray, y_prob: np.ndarray, out_dir: Path) -> float:
    auc = roc_auc_score(y_true, y_prob)
    pr_auc = average_precision_score(y_true, y_prob)
    brier = brier_score_loss(y_true, y_prob)
    has_zh_font = use_chinese_font_if_available()

    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    youden = tpr - fpr
    best_idx = int(np.argmax(youden))
    best_threshold = float(thresholds[best_idx])
    y_pred = (y_prob >= best_threshold).astype(int)

    print("\n=== Test performance ===")
    print(f"ROC-AUC: {auc:.4f}")
    print(f"PR-AUC: {pr_auc:.4f}")
    print(f"Brier score: {brier:.4f}")
    print(f"Best threshold (Youden): {best_threshold:.4f}")
    print("Confusion matrix:")
    print(confusion_matrix(y_true, y_pred))
    print("Classification report:")
    print(classification_report(y_true, y_pred, digits=4))

    # ROC curve
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"AUC={auc:.3f}")
    plt.plot([0, 1], [0, 1], "k--", alpha=0.7)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(out_dir / "roc_curve.png", dpi=200)
    plt.close()

    # ROC curve (Chinese text duplicate)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"AUC={auc:.3f}")
    plt.plot([0, 1], [0, 1], "k--", alpha=0.7)
    plt.xlabel("假阳性率")
    plt.ylabel("真阳性率")
    plt.title("ROC 曲线")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(out_dir / "roc_curve_zh.png", dpi=200)
    plt.close()

    # PR curve
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    plt.figure(figsize=(6, 5))
    plt.plot(recall, precision, label=f"PR-AUC={pr_auc:.3f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend(loc="lower left")
    plt.tight_layout()
    plt.savefig(out_dir / "pr_curve.png", dpi=200)
    plt.close()

    # PR curve (Chinese text duplicate)
    plt.figure(figsize=(6, 5))
    plt.plot(recall, precision, label=f"PR-AUC={pr_auc:.3f}")
    plt.xlabel("召回率")
    plt.ylabel("精确率")
    plt.title("PR 曲线")
    plt.legend(loc="lower left")
    plt.tight_layout()
    plt.savefig(out_dir / "pr_curve_zh.png", dpi=200)
    plt.close()

    # Calibration curve
    frac_pos, mean_pred = calibration_curve(y_true, y_prob, n_bins=10, strategy="quantile")
    plt.figure(figsize=(6, 5))
    plt.plot(mean_pred, frac_pos, marker="o", label="Model")
    plt.plot([0, 1], [0, 1], "k--", label="Perfect calibration")
    plt.xlabel("Mean predicted probability")
    plt.ylabel("Observed event rate")
    plt.title("Calibration Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "calibration_curve.png", dpi=200)
    plt.close()

    # Calibration curve (Chinese text duplicate)
    plt.figure(figsize=(6, 5))
    plt.plot(mean_pred, frac_pos, marker="o", label="模型")
    plt.plot([0, 1], [0, 1], "k--", label="理想校准")
    plt.xlabel("平均预测概率")
    plt.ylabel("实际事件率")
    plt.title("校准曲线")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "calibration_curve_zh.png", dpi=200)
    plt.close()

    if not has_zh_font:
        print("[WARN] 未检测到中文字体，中文图可能显示为方框。")

    return best_threshold


def to_dense_if_needed(x):
    if sparse.issparse(x):
        return x.toarray()
    return np.asarray(x)


def bootstrap_or_ci(
    transformed_x,
    y: np.ndarray,
    model_params: dict,
    n_bootstrap: int = 200,
    random_state: int = 42,
) -> np.ndarray:
    rng = np.random.default_rng(random_state)
    n = transformed_x.shape[0]

    boot_coefs = []
    base_model = LogisticRegression(**model_params)

    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, n)
        x_b = transformed_x[idx]
        y_b = y[idx]
        try:
            m = clone(base_model)
            m.fit(x_b, y_b)
            boot_coefs.append(m.coef_.ravel())
        except Exception:
            continue

    if not boot_coefs:
        raise RuntimeError("Bootstrap failed for all iterations. Try reducing model complexity.")

    return np.vstack(boot_coefs)


def main() -> None:
    df = load_table(DATA_PATH)
    if TARGET_COL not in df.columns:
        raise ValueError(f"Target column not found: {TARGET_COL}")

    feature_cols = get_feature_cols(df)
    x = df[feature_cols].copy()
    y = normalize_target(df[TARGET_COL])

    cat_cols, num_cols = infer_cat_num_cols(x)

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    pipe = build_pipeline(cat_cols=cat_cols, num_cols=num_cols)

    param_grid = {
        "model__solver": ["liblinear"],
        "model__penalty": ["l1", "l2"],
        "model__C": [0.01, 0.1, 1.0, 3.0, 10.0],
    }
    cv = StratifiedKFold(n_splits=CV_SPLITS, shuffle=True, random_state=RANDOM_STATE)

    grid = GridSearchCV(
        estimator=pipe,
        param_grid=param_grid,
        scoring="roc_auc",
        cv=cv,
        n_jobs=-1,
        verbose=1,
        refit=True,
    )

    grid.fit(x_train, y_train)
    best_model = grid.best_estimator_

    print("=== Best CV setting ===")
    print(grid.best_params_)
    print(f"Best CV ROC-AUC: {grid.best_score_:.4f}")

    # Test set evaluation
    y_prob = best_model.predict_proba(x_test)[:, 1]
    threshold = evaluate_model(y_test.to_numpy(), y_prob, OUTPUT_DIR)

    # Save model and threshold
    joblib.dump(best_model, OUTPUT_DIR / "scd_logistic_model.joblib")
    pd.Series({"threshold": threshold}).to_csv(OUTPUT_DIR / "best_threshold.csv", index=False)

    # Coef / OR / CI
    preprocess = best_model.named_steps["preprocess"]
    lr = best_model.named_steps["model"]

    feature_names = preprocess.get_feature_names_out().tolist()
    coef = lr.coef_.ravel()

    x_train_t = preprocess.transform(x_train)
    x_train_t = to_dense_if_needed(x_train_t)

    lr_params = lr.get_params()
    boot_coefs = bootstrap_or_ci(
        transformed_x=x_train_t,
        y=y_train.to_numpy(),
        model_params=lr_params,
        n_bootstrap=N_BOOTSTRAP,
        random_state=RANDOM_STATE,
    )

    coef_ci_low = np.percentile(boot_coefs, 2.5, axis=0)
    coef_ci_high = np.percentile(boot_coefs, 97.5, axis=0)

    result_df = pd.DataFrame(
        {
            "feature": feature_names,
            "coef": coef,
            "OR": np.exp(coef),
            "coef_ci_low": coef_ci_low,
            "coef_ci_high": coef_ci_high,
            "OR_ci_low": np.exp(coef_ci_low),
            "OR_ci_high": np.exp(coef_ci_high),
            "abs_coef": np.abs(coef),
        }
    ).sort_values("abs_coef", ascending=False)

    result_df.to_csv(OUTPUT_DIR / "feature_or_ci.csv", index=False, encoding="utf-8-sig")

    print("\n=== Outputs saved ===")
    print(f"Directory: {OUTPUT_DIR}")
    print("- scd_logistic_model.joblib")
    print("- best_threshold.csv")
    print("- feature_or_ci.csv")
    print("- roc_curve.png")
    print("- pr_curve.png")
    print("- calibration_curve.png")
    print("- roc_curve_zh.png")
    print("- pr_curve_zh.png")
    print("- calibration_curve_zh.png")


if __name__ == "__main__":
    main()
