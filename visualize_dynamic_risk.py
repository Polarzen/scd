# -*- coding: utf-8 -*-
"""Generate visualization figures for dynamic ECG short-term risk results."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
OUT_DIR = BASE_DIR / "output_dynamic_risk"
FIG_DIR = OUT_DIR / "figures"


def use_chinese_font_if_available() -> bool:
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


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def save_confusion_matrix(metrics: dict, out_path: Path, zh: bool = False) -> None:
    cm = np.array(metrics.get("confusion_matrix", [[0, 0], [0, 0]]), dtype=float)

    fig, ax = plt.subplots(figsize=(5.2, 4.4))
    im = ax.imshow(cm, cmap="Blues")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    if zh:
        ax.set_xticks([0, 1], labels=["预测 0", "预测 1"])
        ax.set_yticks([0, 1], labels=["真实 0", "真实 1"])
        ax.set_title("混淆矩阵（7:3 划分）")
    else:
        ax.set_xticks([0, 1], labels=["Pred 0", "Pred 1"])
        ax.set_yticks([0, 1], labels=["True 0", "True 1"])
        ax.set_title("Confusion Matrix (70/30 split)")

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            val = int(cm[i, j])
            ax.text(j, i, str(val), ha="center", va="center", color="black", fontsize=11)

    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def save_label_distribution(features_df: pd.DataFrame, out_path: Path, zh: bool = False) -> None:
    vc = features_df["label"].value_counts().sort_index()
    if zh:
        x = ["非短期", "短期"]
    else:
        x = ["Non-short-term", "Short-term"]
    y = [int(vc.get(0, 0)), int(vc.get(1, 0))]

    fig, ax = plt.subplots(figsize=(6.0, 4.2))
    bars = ax.bar(x, y, color=["#4E79A7", "#E15759"], edgecolor="black", linewidth=0.6)
    if zh:
        ax.set_title("队列标签分布")
        ax.set_ylabel("患者数")
    else:
        ax.set_title("Label Distribution in Cohort")
        ax.set_ylabel("Patients")

    for b in bars:
        h = b.get_height()
        ax.text(b.get_x() + b.get_width() / 2.0, h + 0.2, f"{int(h)}", ha="center", va="bottom", fontsize=10)

    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def save_top_feature_scores(df_score: pd.DataFrame, out_path: Path, top_n: int = 15, zh: bool = False) -> None:
    """兼容树模型(importance)与旧逻辑回归(OR)结果的特征图。"""
    fig, ax = plt.subplots(figsize=(8.4, 6.8))

    if "importance" in df_score.columns:
        d = df_score.sort_values("importance", ascending=False).head(top_n).iloc[::-1]
        vals = d["importance"].to_numpy(dtype=float)
        names = d["feature"].astype(str).to_list()

        ax.barh(names, vals, color="#4E79A7", edgecolor="black", linewidth=0.5)
        if zh:
            ax.set_title(f"重要性前 {top_n} 的特征")
            ax.set_xlabel("特征重要性")
        else:
            ax.set_title(f"Top {top_n} Features by Importance")
            ax.set_xlabel("Feature Importance")
    else:
        d = df_score.copy().head(top_n).iloc[::-1]
        vals = d["OR"].to_numpy(dtype=float)
        names = d["feature"].astype(str).to_list()
        colors = ["#D62728" if v > 1.0 else "#1F77B4" for v in vals]

        ax.barh(names, vals, color=colors, edgecolor="black", linewidth=0.5)
        ax.axvline(1.0, color="black", linestyle="--", linewidth=1.0)
        if zh:
            ax.set_title(f"按 |系数| 排名前 {top_n} 的特征（OR）")
            ax.set_xlabel("比值比（OR）")
        else:
            ax.set_title(f"Top {top_n} Features by |Coefficient| (OR)")
            ax.set_xlabel("Odds Ratio (OR)")

    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def save_cv_folds(cv_df: pd.DataFrame, out_path: Path, zh: bool = False) -> None:
    folds = cv_df["fold"].astype(int).to_numpy()
    has_f1 = "f1" in cv_df.columns
    n_cols = 4 if has_f1 else 3

    fig, axes = plt.subplots(1, n_cols, figsize=(4.4 * n_cols, 4.2), sharex=True)
    if n_cols == 1:
        axes = [axes]

    axes[0].plot(folds, cv_df["val_auc"].to_numpy(), marker="o", color="#4E79A7")
    axes[0].set_title("折次 ROC-AUC" if zh else "Fold ROC-AUC")
    axes[0].set_ylim(0.0, 1.0)
    axes[0].set_xlabel("折次" if zh else "Fold")

    axes[1].plot(folds, cv_df["val_pr_auc"].to_numpy(), marker="o", color="#E15759")
    axes[1].set_title("折次 PR-AUC" if zh else "Fold PR-AUC")
    axes[1].set_ylim(0.0, 1.0)
    axes[1].set_xlabel("折次" if zh else "Fold")

    axes[2].plot(folds, cv_df["val_brier"].to_numpy(), marker="o", color="#59A14F")
    axes[2].set_title("折次 Brier" if zh else "Fold Brier")
    axes[2].set_ylim(0.0, 1.0)
    axes[2].set_xlabel("折次" if zh else "Fold")

    if has_f1:
        axes[3].plot(folds, cv_df["f1"].to_numpy(), marker="o", color="#9C755F")
        axes[3].set_title("折次 F1" if zh else "Fold F1")
        axes[3].set_ylim(0.0, 1.0)
        axes[3].set_xlabel("折次" if zh else "Fold")

    for ax in axes:
        ax.grid(alpha=0.25, linestyle=":")

    fig.suptitle("5 折交叉验证指标" if zh else "5-Fold Cross-Validation Metrics", y=1.03, fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_cv_summary(cv_metrics: dict, out_path: Path, zh: bool = False) -> None:
    metrics = ["AUC", "PR-AUC", "Brier"]
    means = [
        float(cv_metrics.get("auc_mean", np.nan)),
        float(cv_metrics.get("pr_auc_mean", np.nan)),
        float(cv_metrics.get("brier_mean", np.nan)),
    ]
    stds = [
        float(cv_metrics.get("auc_std", 0.0)),
        float(cv_metrics.get("pr_auc_std", 0.0)),
        float(cv_metrics.get("brier_std", 0.0)),
    ]
    colors = ["#4E79A7", "#E15759", "#59A14F"]

    if "f1_mean" in cv_metrics:
        metrics.append("F1")
        means.append(float(cv_metrics.get("f1_mean", np.nan)))
        stds.append(float(cv_metrics.get("f1_std", 0.0)))
        colors.append("#9C755F")

    fig, ax = plt.subplots(figsize=(6.6, 4.4))
    ax.bar(metrics, means, yerr=stds, capsize=4, color=colors, edgecolor="black", linewidth=0.5)
    ax.set_ylim(0.0, 1.0)
    ax.set_title("交叉验证汇总（均值 ± 标准差）" if zh else "Cross-Validation Summary (mean ± std)")
    ax.set_ylabel("分数" if zh else "Score")

    for i, v in enumerate(means):
        ax.text(i, v + 0.03, f"{v:.3f}", ha="center", va="bottom", fontsize=10)

    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def save_repeated_cv_summary(rep_metrics: dict, out_path: Path, zh: bool = False) -> None:
    labels = ["AUC", "PR-AUC", "F1", "Brier"]
    mean_keys = ["auc_repeat_mean", "pr_auc_repeat_mean", "f1_repeat_mean", "brier_repeat_mean"]
    ci_keys = ["auc_ci", "pr_auc_ci", "f1_ci", "brier_ci"]
    colors = ["#4E79A7", "#E15759", "#9C755F", "#59A14F"]

    means = []
    lower_err = []
    upper_err = []
    final_labels = []
    final_colors = []

    for label, mk, ck, color in zip(labels, mean_keys, ci_keys, colors):
        if mk not in rep_metrics:
            continue
        m = float(rep_metrics.get(mk, np.nan))
        ci = rep_metrics.get(ck, [np.nan, np.nan])
        if not isinstance(ci, list) or len(ci) != 2:
            ci = [np.nan, np.nan]
        lo = float(ci[0])
        hi = float(ci[1])

        means.append(m)
        lower_err.append(max(0.0, m - lo) if np.isfinite(m) and np.isfinite(lo) else 0.0)
        upper_err.append(max(0.0, hi - m) if np.isfinite(m) and np.isfinite(hi) else 0.0)
        final_labels.append(label)
        final_colors.append(color)

    if not means:
        return

    yerr = np.vstack([np.asarray(lower_err), np.asarray(upper_err)])
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.bar(final_labels, means, yerr=yerr, capsize=5, color=final_colors, edgecolor="black", linewidth=0.5)
    ax.set_ylim(0.0, 1.0)

    ci_alpha = float(rep_metrics.get("ci_alpha", 0.95))
    title_pct = int(ci_alpha * 100)
    ax.set_title(f"重复交叉验证汇总（{title_pct}% 置信区间）" if zh else f"Repeated CV Summary ({title_pct}% CI)")
    ax.set_ylabel("分数" if zh else "Score")

    for i, v in enumerate(means):
        ax.text(i, v + 0.03, f"{v:.3f}", ha="center", va="bottom", fontsize=10)

    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    has_zh_font = use_chinese_font_if_available()

    metrics_path = OUT_DIR / "metrics.json"
    cv_metrics_path = OUT_DIR / "cv_metrics.json"
    cv_folds_path = OUT_DIR / "cv_fold_metrics.csv"
    repeated_cv_metrics_path = OUT_DIR / "repeated_cv_metrics.json"
    features_path = OUT_DIR / "dynamic_patient_features.csv"
    fi_path = OUT_DIR / "feature_importance.csv"
    or_path = OUT_DIR / "feature_or.csv"
    score_path = fi_path if fi_path.exists() else or_path

    missing = [p for p in [metrics_path, features_path, score_path] if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Missing required files: {missing}")

    metrics = load_json(metrics_path)
    features_df = pd.read_csv(features_path, encoding="utf-8-sig")
    score_df = pd.read_csv(score_path, encoding="utf-8-sig")

    save_confusion_matrix(metrics, FIG_DIR / "confusion_matrix.png", zh=False)
    save_label_distribution(features_df, FIG_DIR / "label_distribution.png", zh=False)
    save_top_feature_scores(score_df, FIG_DIR / "top_feature_importance.png", top_n=15, zh=False)

    save_confusion_matrix(metrics, FIG_DIR / "confusion_matrix_zh.png", zh=True)
    save_label_distribution(features_df, FIG_DIR / "label_distribution_zh.png", zh=True)
    save_top_feature_scores(score_df, FIG_DIR / "top_feature_importance_zh.png", top_n=15, zh=True)

    if cv_metrics_path.exists() and cv_folds_path.exists():
        cv_metrics = load_json(cv_metrics_path)
        cv_df = pd.read_csv(cv_folds_path, encoding="utf-8-sig")
        save_cv_folds(cv_df, FIG_DIR / "cv_fold_metrics.png", zh=False)
        save_cv_summary(cv_metrics, FIG_DIR / "cv_summary.png", zh=False)

        save_cv_folds(cv_df, FIG_DIR / "cv_fold_metrics_zh.png", zh=True)
        save_cv_summary(cv_metrics, FIG_DIR / "cv_summary_zh.png", zh=True)

    if repeated_cv_metrics_path.exists():
        rep_metrics = load_json(repeated_cv_metrics_path)
        save_repeated_cv_summary(rep_metrics, FIG_DIR / "repeated_cv_summary.png", zh=False)
        save_repeated_cv_summary(rep_metrics, FIG_DIR / "repeated_cv_summary_zh.png", zh=True)

    if not has_zh_font:
        print("[WARN] 未检测到中文字体，中文图可能显示为方框。")

    print(f"[INFO] Figures saved to: {FIG_DIR}")
    for p in sorted(FIG_DIR.glob("*.png")):
        print(f"[FIG] {p.name}")


if __name__ == "__main__":
    main()
