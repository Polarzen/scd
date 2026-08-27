# -*- coding: utf-8 -*-
"""
基于动态心电信号（Holter ECG）的恶性室性心律失常短期风险预测模型。

任务定义（可配置）:
- 队列模式A（默认）: 先筛选 Cause of death == 3 (SCD) 的患者，
    再以 Follow-up period from enrollment (days) <= SHORT_TERM_DAYS 定义短期风险标签
- 队列模式B: 全部患者中，短期SCD作为正类（兼容旧逻辑）

方法概述:
1) 从 subject-info.csv 读取标签与患者信息
2) 对每位患者从 Holter 原始波形中随机抽取多个时间窗
3) 提取每个时间窗的动态ECG特征（含时域、频域、非线性复杂度）
4) 在患者层面对窗口特征做聚合（mean/std/p10/p50/p90）
5) 训练并评估树模型（支持调参、交叉验证、重复交叉验证可靠性分析）

输出文件:
- output_dynamic_risk/dynamic_patient_features.csv
- output_dynamic_risk/dynamic_short_term_tree.joblib
- output_dynamic_risk/best_threshold.csv
- output_dynamic_risk/metrics.json
- output_dynamic_risk/feature_importance.csv
- output_dynamic_risk/cv_fold_metrics.csv
- output_dynamic_risk/cv_metrics.json
- output_dynamic_risk/repeated_cv_fold_metrics.csv
- output_dynamic_risk/repeated_cv_repeat_metrics.csv
- output_dynamic_risk/repeated_cv_metrics.json
"""

from __future__ import annotations

import json
import argparse
import warnings
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Tuple, TypedDict, cast

import joblib
import numpy as np
import pandas as pd
import wfdb
from scipy.interpolate import interp1d
from scipy.signal import butter, filtfilt, find_peaks, welch
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline

warnings.filterwarnings("ignore", category=UserWarning)


# -------------------------
# 路径与任务配置
# -------------------------
BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "music-sudden-cardiac-death-in-chronic-heart-failure-1.0.1"
SUBJECT_INFO_CSV = DATASET_DIR / "subject-info.csv"
HOLTER_DIR = DATASET_DIR / "Holter_ECG"

OUTPUT_DIR = BASE_DIR / "output_dynamic_risk"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 短期风险窗口（天）
SHORT_TERM_DAYS = 365

# 动态窗口设置（秒）
WINDOW_SEC = 120
MAX_WINDOWS_PER_PATIENT = 24
RANDOM_STATE = 42

# 训练设置
DEFAULT_VAL_SIZE = 0.30  # 训练:验证 = 7:3
MIN_PATIENTS_FOR_TRAIN = 80
DEFAULT_MODEL_TYPE = "et"
DEFAULT_THRESHOLD_METHOD = "target-specificity"
DEFAULT_TUNE = True
DEFAULT_TUNE_N_ITER = 24
DEFAULT_TUNE_CV_FOLDS = 4
DEFAULT_CV_REPEATS = 0
DEFAULT_CI_ALPHA = 0.95
DEFAULT_TARGET_SPECIFICITY = 0.70
DEFAULT_THRESHOLD_CALIB_SIZE = 0.25
DEFAULT_N_JOBS = -1
DEFAULT_FEATURE_WORKERS = -1
FAST_MAX_WINDOWS_PER_PATIENT = 12
DEFAULT_FAST_MODE = True
# 这些列用于标签定义或审计，训练时必须排除以避免标签泄漏
NON_FEATURE_COLUMNS = {"patient_id", "label", "followup_days", "cause_of_death"}


class TrainInfo(TypedDict):
    model_type: str
    model_params: Dict[str, object]
    threshold_method: str
    target_specificity: float
    threshold_calib_size: float


def resolve_worker_count(n_workers: int) -> int:
    """将 -1 解析为全部CPU核心，其它值限制为 >=1。"""
    cpu_count = os.cpu_count() or 1
    if n_workers == -1:
        return cpu_count
    return max(1, int(n_workers))


def model_label(model_type: str) -> str:
    if model_type == "rf":
        return "RandomForestClassifier"
    if model_type == "et":
        return "ExtraTreesClassifier"
    return model_type


def build_model(
    feature_cols: List[str],
    model_type: str = DEFAULT_MODEL_TYPE,
    model_params: Dict[str, object] | None = None,
    n_jobs: int = DEFAULT_N_JOBS,
) -> Pipeline:
    """构建用于训练与交叉验证的统一树模型管道。"""
    pre = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    steps=[
                        ("imp", SimpleImputer(strategy="median")),
                    ]
                ),
                feature_cols,
            )
        ],
        remainder="drop",
    )

    if model_type == "rf":
        estimator = RandomForestClassifier(
            n_estimators=600,
            max_depth=None,
            min_samples_leaf=2,
            min_samples_split=6,
            max_features="sqrt",
            class_weight="balanced_subsample",
            random_state=RANDOM_STATE,
            n_jobs=n_jobs,
        )
    elif model_type == "et":
        estimator = ExtraTreesClassifier(
            n_estimators=800,
            max_depth=None,
            min_samples_leaf=2,
            min_samples_split=6,
            max_features="sqrt",
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=n_jobs,
        )
    else:
        raise ValueError(f"不支持的树模型类型: {model_type}")

    if model_params:
        estimator.set_params(**model_params)

    model = Pipeline(
        steps=[
            ("pre", pre),
            (
                "clf",
                estimator,
            ),
        ]
    )
    return model


def get_param_distributions(model_type: str) -> Dict[str, List[object]]:
    if model_type == "rf":
        return {
            "clf__n_estimators": [300, 500, 800, 1000, 1400],
            "clf__max_depth": [None, 4, 6, 8, 10, 14],
            "clf__min_samples_split": [2, 4, 6, 8, 10],
            "clf__min_samples_leaf": [1, 2, 3, 4],
            "clf__max_features": ["sqrt", "log2", 0.5, 0.7],
            "clf__class_weight": ["balanced", "balanced_subsample", {0: 1, 1: 2}, {0: 1, 1: 3}],
        }
    if model_type == "et":
        return {
            "clf__n_estimators": [400, 800, 1200, 1600],
            "clf__max_depth": [None, 4, 6, 8, 10, 14],
            "clf__min_samples_split": [2, 4, 6, 8, 10],
            "clf__min_samples_leaf": [1, 2, 3, 4],
            "clf__max_features": ["sqrt", "log2", 0.5, 0.7],
            "clf__class_weight": ["balanced", {0: 1, 1: 2}, {0: 1, 1: 3}],
        }
    raise ValueError(f"不支持的树模型类型: {model_type}")


def tune_tree_model(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    feature_cols: List[str],
    model_type: str,
    tune_n_iter: int = DEFAULT_TUNE_N_ITER,
    tune_cv_folds: int = DEFAULT_TUNE_CV_FOLDS,
    n_jobs: int = DEFAULT_N_JOBS,
) -> Tuple[Pipeline, Dict[str, object], float]:
    """在训练集上随机搜索树模型超参，以PR-AUC为目标。"""
    n_pos = int(y_train.sum())
    n_neg = int(len(y_train) - n_pos)
    inner_folds = min(tune_cv_folds, n_pos, n_neg)

    base_model = build_model(feature_cols, model_type=model_type, n_jobs=n_jobs)
    if inner_folds < 2:
        print("[WARN] 调参阶段样本不足，已跳过随机搜索。")
        return base_model, {}, np.nan

    search = RandomizedSearchCV(
        estimator=base_model,
        param_distributions=get_param_distributions(model_type),
        n_iter=max(1, int(tune_n_iter)),
        scoring="average_precision",
        n_jobs=n_jobs,
        cv=StratifiedKFold(n_splits=inner_folds, shuffle=True, random_state=RANDOM_STATE),
        refit=True,
        random_state=RANDOM_STATE,
    )
    search.fit(x_train, y_train)

    best_params = {
        k.replace("clf__", "", 1): v
        for k, v in search.best_params_.items()
        if k.startswith("clf__")
    }
    return cast(Pipeline, search.best_estimator_), best_params, float(search.best_score_)


def compute_cls_stats(y_true: pd.Series | np.ndarray, pred: np.ndarray) -> Tuple[np.ndarray, float, float, float]:
    """返回混淆矩阵、灵敏度、特异度、F1。"""
    y_arr = np.asarray(y_true, dtype=int)
    p_arr = np.asarray(pred, dtype=int)
    cm = confusion_matrix(y_arr, p_arr, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    sensitivity = float(tp / (tp + fn + 1e-12))
    specificity = float(tn / (tn + fp + 1e-12))
    f1 = float(f1_score(y_arr, p_arr, zero_division=0))
    return cm, sensitivity, specificity, f1


def select_best_threshold(
    y_true: pd.Series | np.ndarray,
    prob: np.ndarray,
    method: str = DEFAULT_THRESHOLD_METHOD,
    target_specificity: float = DEFAULT_TARGET_SPECIFICITY,
) -> float:
    y_arr = np.asarray(y_true, dtype=int)
    p_arr = np.asarray(prob, dtype=np.float64)

    if method == "target-specificity":
        target_specificity = float(target_specificity)
        target_specificity = min(max(target_specificity, 0.0), 1.0)

        neg_prob = np.sort(p_arr[y_arr == 0])
        if neg_prob.size == 0:
            return 0.5

        # 使用阴性分布分位数选阈值，使训练集特异度接近目标值。
        # 由于 pred = (prob >= threshold) 判阳，阈值取 nextafter(+,inf) 可避免边界值被误判为阳性。
        if target_specificity >= 1.0:
            return float(np.nextafter(neg_prob[-1], np.inf))

        k = int(np.floor(target_specificity * neg_prob.size))
        k = min(max(k, 0), neg_prob.size - 1)
        base_thr = float(neg_prob[k])
        return float(np.nextafter(base_thr, np.inf))

    if method == "f1":
        precision, recall, thresholds = precision_recall_curve(y_arr, p_arr)
        if thresholds.size == 0:
            return 0.5
        f1_vals = 2.0 * precision[:-1] * recall[:-1] / (precision[:-1] + recall[:-1] + 1e-12)
        if f1_vals.size == 0 or np.all(~np.isfinite(f1_vals)):
            return 0.5
        best_idx = int(np.nanargmax(f1_vals))
        return float(thresholds[best_idx])

    # 兼容原有Youden阈值
    fpr, tpr, thr = roc_curve(y_arr, p_arr)
    j = tpr - fpr
    best_idx = int(np.argmax(j))
    return float(thr[best_idx])


def summarize_with_ci(values: np.ndarray, ci_alpha: float = DEFAULT_CI_ALPHA) -> Dict[str, float]:
    """对一组指标计算均值、标准差与分位数置信区间。"""
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {
            "mean": np.nan,
            "std": np.nan,
            "median": np.nan,
            "ci_low": np.nan,
            "ci_high": np.nan,
            "n": 0.0,
        }

    ci_alpha = float(ci_alpha)
    ci_alpha = min(max(ci_alpha, 0.5), 0.999)
    lo = (1.0 - ci_alpha) * 50.0
    hi = 100.0 - lo

    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0,
        "median": float(np.median(arr)),
        "ci_low": float(np.percentile(arr, lo)),
        "ci_high": float(np.percentile(arr, hi)),
        "n": float(arr.size),
    }


def split_fit_calibration(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    calib_size: float = DEFAULT_THRESHOLD_CALIB_SIZE,
    random_state: int = RANDOM_STATE,
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, bool]:
    """将训练集拆分为拟合集和阈值校准集；若条件不足则回退到使用完整训练集。"""
    if calib_size <= 0.0:
        return x_train, y_train, x_train, y_train, True

    n_pos = int(y_train.sum())
    n_neg = int(len(y_train) - n_pos)
    n_cal_pos = int(np.floor(n_pos * calib_size))
    n_cal_neg = int(np.floor(n_neg * calib_size))

    if n_cal_pos < 1 or n_cal_neg < 1:
        return x_train, y_train, x_train, y_train, True

    try:
        x_fit, x_cal, y_fit, y_cal = train_test_split(
            x_train,
            y_train,
            test_size=calib_size,
            random_state=random_state,
            stratify=y_train,
        )
    except Exception:
        return x_train, y_train, x_train, y_train, True

    return x_fit, y_fit, x_cal, y_cal, False


def prepare_features_and_labels(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
    """准备训练所需特征与标签，并完成基础质量检查。"""
    if df.empty:
        raise ValueError("动态特征表为空，无法训练。")

    if df["label"].nunique() < 2:
        raise ValueError("标签只有单一类别，无法训练二分类模型。")

    feature_cols = [c for c in df.columns if c not in NON_FEATURE_COLUMNS]
    if not feature_cols:
        raise ValueError("没有可用的训练特征。")

    # 移除常量特征，避免对模型和解释产生噪声
    non_constant_cols = []
    for c in feature_cols:
        col = pd.to_numeric(df[c], errors="coerce")
        if col.nunique(dropna=True) > 1:
            non_constant_cols.append(c)
    feature_cols = non_constant_cols

    if not feature_cols:
        raise ValueError("全部候选特征均为常量，无法训练。")

    x = df[feature_cols].copy()
    y = df["label"].astype(int)

    n_pos = int(y.sum())
    n_neg = int(len(y) - n_pos)
    if n_pos < 2 or n_neg < 2:
        raise ValueError(f"类别样本过少（n_pos={n_pos}, n_neg={n_neg}），无法训练二分类模型。")
    if n_pos < 10:
        print(f"[WARN] 正例数量较少（n_pos={n_pos}），评估方差可能较大。")

    return x, y, feature_cols


# -------------------------
# 工具函数
# -------------------------
def _to_float(val):
    """将数据集中的混合格式字符串安全转成浮点数。"""
    if pd.isna(val):
        return np.nan
    s = str(val).strip().strip("\"").strip("'")
    if s in {"", "NA", "na", "N/A", "None"}:
        return np.nan

    # 先尝试直接转
    try:
        return float(s)
    except ValueError:
        pass

    # 含小数逗号（如 31,2）
    if "," in s and "." not in s:
        parts = s.split(",")
        if len(parts) == 2 and parts[1].isdigit() and len(parts[1]) <= 3:
            s2 = s.replace(",", ".")
            try:
                return float(s2)
            except ValueError:
                pass

    # 含千分位逗号（如 1,479,583,333）
    s3 = s.replace(",", "")
    try:
        return float(s3)
    except ValueError:
        return np.nan


def _safe_float_scalar(val: object) -> float:
    """将任意标量尽量安全转为 float（失败返回 NaN）。"""
    x = _to_float(val)
    if np.isfinite(x):
        return float(x)
    return np.nan


def _safe_int_scalar(val: object, default: int = 0) -> int:
    """将任意标量尽量安全转为 int（失败返回默认值）。"""
    x = _to_float(val)
    if np.isfinite(x):
        return int(x)
    return int(default)


def bandpass_filter(x: np.ndarray, fs: float, low: float = 0.5, high: float = 40.0, order: int = 3) -> np.ndarray:
    nyq = 0.5 * fs
    hi = min(high, nyq * 0.95)
    lo = max(low, 0.05)
    if lo >= hi:
        return x
    coeffs = butter(order, [lo / nyq, hi / nyq], btype="band", output="ba")
    if coeffs is None or len(coeffs) < 2:
        return x
    b = np.asarray(coeffs[0], dtype=np.float64)
    a = np.asarray(coeffs[1], dtype=np.float64)
    return filtfilt(b, a, x)


def preprocess_ecg(sig_1d: np.ndarray, fs: float) -> np.ndarray:
    x = sig_1d.astype(np.float64)
    x = x - np.nanmedian(x)
    x = bandpass_filter(x, fs=fs)
    x = x - np.nanmedian(x)
    std = np.nanstd(x)
    if std > 1e-8:
        x = x / std
    return x


def detect_r_peaks(ecg: np.ndarray, fs: float) -> np.ndarray:
    x = ecg
    enh = x * x
    distance = int(0.25 * fs)
    threshold = np.percentile(enh, 93)
    peaks, _ = find_peaks(enh, distance=max(distance, 1), height=threshold)
    return peaks


def hrv_features(rr_sec: np.ndarray) -> Dict[str, float]:
    if rr_sec.size < 3:
        return {
            "mean_rr": np.nan,
            "sdnn": np.nan,
            "rmssd": np.nan,
            "pnn50": np.nan,
            "mean_hr": np.nan,
            "rr_cv": np.nan,
        }

    diff_rr = np.diff(rr_sec)
    mean_rr = float(np.mean(rr_sec))
    sdnn = float(np.std(rr_sec, ddof=1)) if rr_sec.size > 1 else np.nan
    rmssd = float(np.sqrt(np.mean(diff_rr ** 2))) if diff_rr.size > 0 else np.nan
    pnn50 = float(np.mean(np.abs(diff_rr) > 0.05) * 100.0) if diff_rr.size > 0 else np.nan
    mean_hr = float(60.0 / mean_rr) if mean_rr > 1e-8 else np.nan
    rr_cv = float(np.std(rr_sec) / (mean_rr + 1e-8))

    return {
        "mean_rr": mean_rr,
        "sdnn": sdnn,
        "rmssd": rmssd,
        "pnn50": pnn50,
        "mean_hr": mean_hr,
        "rr_cv": rr_cv,
    }


def sample_entropy(signal: np.ndarray, m: int = 2, r_ratio: float = 0.2) -> float:
    """Sample Entropy（SampEn）。建议输入RR序列，避免原始ECG长度导致计算过慢。"""
    x = np.asarray(signal, dtype=np.float64)
    n = len(x)
    if n < 10:
        return np.nan

    std = np.std(x)
    if not np.isfinite(std) or std < 1e-12:
        return np.nan

    r = r_ratio * std

    def _phi(mm: int) -> int:
        count = 0
        end = n - mm
        for i in range(end):
            xi = x[i:i + mm]
            for j in range(i + 1, end):
                if np.all(np.abs(xi - x[j:j + mm]) <= r):
                    count += 1
        return count

    b = _phi(m)
    a = _phi(m + 1)

    if b == 0 or a == 0:
        return np.nan
    return float(-np.log(a / b))


def approximate_entropy(signal: np.ndarray, m: int = 2, r_ratio: float = 0.2) -> float:
    """Approximate Entropy（ApEn）。"""
    x = np.asarray(signal, dtype=np.float64)
    n = len(x)
    if n < 10:
        return np.nan

    std = np.std(x)
    if not np.isfinite(std) or std < 1e-12:
        return np.nan

    r = r_ratio * std

    def _phi(mm: int) -> float:
        c_vals = []
        end = n - mm
        if end <= 0:
            return np.nan
        for i in range(end):
            xi = x[i:i + mm]
            count = 0
            for j in range(end):
                if np.all(np.abs(xi - x[j:j + mm]) <= r):
                    count += 1
            c_vals.append(count / end)

        c_arr = np.asarray(c_vals, dtype=np.float64)
        return float(np.sum(np.log(c_arr + 1e-8)) / end)

    phi_m = _phi(m)
    phi_m1 = _phi(m + 1)
    if not np.isfinite(phi_m) or not np.isfinite(phi_m1):
        return np.nan
    return float(phi_m - phi_m1)


def dfa(signal: np.ndarray) -> float:
    """Detrended Fluctuation Analysis，返回标度指数 alpha。"""
    x = np.asarray(signal, dtype=np.float64)
    n = len(x)
    if n < 16:
        return np.nan

    x = x - np.mean(x)
    y = np.cumsum(x)

    max_scale = n // 4
    if max_scale < 4:
        return np.nan

    scales = np.floor(np.logspace(np.log10(4), np.log10(max_scale), num=10)).astype(int)
    scales = np.unique(scales)
    flucts = []
    used_scales = []

    for s in scales:
        if s < 4:
            continue

        n_segments = n // s
        if n_segments < 2:
            continue

        f_nu = []
        t = np.arange(s, dtype=np.float64)
        for i in range(n_segments):
            seg = y[i * s:(i + 1) * s]
            coeffs = np.polyfit(t, seg, 1)
            trend = np.polyval(coeffs, t)
            f_nu.append(np.mean((seg - trend) ** 2))

        fn = np.sqrt(np.mean(f_nu))
        if np.isfinite(fn) and fn > 0:
            flucts.append(float(fn))
            used_scales.append(float(s))

    if len(flucts) < 2:
        return np.nan

    coeffs = np.polyfit(np.log(np.asarray(used_scales)), np.log(np.asarray(flucts)), 1)
    return float(coeffs[0])


def nonlinear_hrv_features(rr_sec: np.ndarray) -> Dict[str, float]:
    """基于RR序列的非线性特征。"""
    if rr_sec.size < 10:
        return {
            "rr_sampen": np.nan,
            "rr_apen": np.nan,
            "rr_dfa_alpha": np.nan,
        }

    return {
        "rr_sampen": sample_entropy(rr_sec, m=2, r_ratio=0.2),
        "rr_apen": approximate_entropy(rr_sec, m=2, r_ratio=0.2),
        "rr_dfa_alpha": dfa(rr_sec),
    }


def spectral_features(ecg: np.ndarray, fs: float) -> Dict[str, float]:
    n = ecg.size
    if n < 8:
        return {"pow_lf": np.nan, "pow_mf": np.nan, "pow_hf": np.nan, "pow_hf_ratio": np.nan}

    # 基于功率谱估计不同频段能量占比
    fft = np.fft.rfft(ecg)
    psd = (np.abs(fft) ** 2) / n
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)

    total = float(np.sum(psd) + 1e-12)

    def band_power(lo: float, hi: float) -> float:
        m = (freqs >= lo) & (freqs < hi)
        return float(np.sum(psd[m]) / total)

    pow_lf = band_power(0.5, 4.0)
    pow_mf = band_power(4.0, 15.0)
    pow_hf = band_power(15.0, 40.0)
    return {
        "pow_lf": pow_lf,
        "pow_mf": pow_mf,
        "pow_hf": pow_hf,
        "pow_hf_ratio": float(pow_hf / (pow_lf + pow_mf + 1e-12)),
    }


def window_features(ecg: np.ndarray, fs: float) -> Dict[str, float]:
    peaks = detect_r_peaks(ecg, fs)
    rr = np.diff(peaks) / fs if peaks.size >= 2 else np.array([])

    feats = {
        "sig_mean": float(np.mean(ecg)),
        "sig_std": float(np.std(ecg)),
        "sig_p2p": float(np.max(ecg) - np.min(ecg)),
        "sig_skew": _safe_float_scalar(pd.Series(ecg).skew()),
        "sig_kurt": _safe_float_scalar(pd.Series(ecg).kurt()),
        "beats": float(peaks.size),
        "beats_per_min": float(peaks.size * 60.0 / (ecg.size / fs + 1e-8)),
    }
    feats.update(hrv_features(rr))
    feats.update(nonlinear_hrv_features(rr))
    feats.update(spectral_features(ecg, fs))
    return feats


def aggregate_dynamic_features(df_win: pd.DataFrame) -> Dict[str, float]:
    """将一个患者多个窗口特征聚合成患者级动态特征。"""
    out: Dict[str, float] = {}
    num_cols = [c for c in df_win.columns if c not in {"patient_id", "window_idx"}]

    for c in num_cols:
        x = pd.to_numeric(df_win[c], errors="coerce").dropna().to_numpy()
        if x.size == 0:
            out[f"{c}_mean"] = np.nan
            out[f"{c}_std"] = np.nan
            out[f"{c}_p10"] = np.nan
            out[f"{c}_p50"] = np.nan
            out[f"{c}_p90"] = np.nan
            continue

        out[f"{c}_mean"] = float(np.mean(x))
        out[f"{c}_std"] = float(np.std(x, ddof=1)) if x.size > 1 else 0.0
        out[f"{c}_p10"] = float(np.percentile(x, 10))
        out[f"{c}_p50"] = float(np.percentile(x, 50))
        out[f"{c}_p90"] = float(np.percentile(x, 90))

    out["n_windows_used"] = float(len(df_win))
    return out


def read_subject_table(scd_only: bool = True) -> pd.DataFrame:
    if not SUBJECT_INFO_CSV.exists():
        raise FileNotFoundError(f"找不到 subject-info.csv: {SUBJECT_INFO_CSV}")

    df = pd.read_csv(SUBJECT_INFO_CSV, sep=";", encoding="utf-8", low_memory=False)
    df.columns = [c.strip() for c in df.columns]

    required_cols = [
        "Patient ID",
        "Follow-up period from enrollment (days)",
        "Cause of death",
    ]
    for c in required_cols:
        if c not in df.columns:
            raise ValueError(f"subject-info.csv 缺少必需列: {c}")

    df["patient_id"] = df["Patient ID"].astype(str).str.strip()
    df["followup_days"] = df["Follow-up period from enrollment (days)"].map(_to_float)
    df["cause_of_death"] = df["Cause of death"].map(_to_float)

    # 缺少随访或结局信息无法可靠定义短期标签，直接剔除
    df = df.dropna(subset=["followup_days", "cause_of_death"]).copy()

    if scd_only:
        # 用户意向: 先筛选猝死患者，再在该队列中预测短期风险
        df = df[df["cause_of_death"] == 3.0].copy()
        df["label_short_term"] = (df["followup_days"] <= float(SHORT_TERM_DAYS)).astype(int)
    else:
        # 兼容旧任务定义: 全队列中短期SCD为正类
        df["label_short_term"] = (
            (df["cause_of_death"] == 3.0) &
            (df["followup_days"] <= float(SHORT_TERM_DAYS))
        ).astype(int)

    # 仅保留有Holter记录的患者
    df["record_stem"] = df["patient_id"].apply(lambda x: HOLTER_DIR / x)
    df = df[df["record_stem"].apply(lambda p: Path(str(p) + ".hea").exists())].copy()

    return df


def sample_windows_for_record(
    record_stem: Path,
    rng: np.random.Generator,
    window_sec: int = WINDOW_SEC,
    max_windows_per_patient: int = MAX_WINDOWS_PER_PATIENT,
) -> Tuple[List[Dict[str, float]], float]:
    """读取单个Holter记录并抽样窗口，返回窗口特征列表与采样率。"""
    hdr = wfdb.rdheader(str(record_stem))
    fs_raw = getattr(hdr, "fs", None)
    sig_len_raw = getattr(hdr, "sig_len", None)
    if fs_raw is None or sig_len_raw is None:
        return [], np.nan

    try:
        fs = float(fs_raw)
        sig_len = int(sig_len_raw)
    except (TypeError, ValueError):
        return [], np.nan

    win_len = int(window_sec * fs)
    if sig_len <= win_len + 2:
        return [], fs

    n_candidates = max(1, sig_len - win_len - 1)
    n_take = min(max_windows_per_patient, n_candidates)

    # 随机无放回抽样窗口起点，增强“动态”覆盖
    starts = rng.choice(n_candidates, size=n_take, replace=False)
    starts.sort()

    features: List[Dict[str, float]] = []

    for idx, s in enumerate(starts):
        e = int(s + win_len)
        try:
            rec = wfdb.rdrecord(str(record_stem), sampfrom=int(s), sampto=e)
            p_signal = getattr(rec, "p_signal", None)
            if p_signal is None or p_signal.size == 0:
                continue
            lead = p_signal[:, 0]
            ecg = preprocess_ecg(lead, fs)
            f = window_features(ecg, fs)
            f["window_idx"] = idx
            features.append(f)
        except Exception:
            continue

    return features, fs


def _extract_patient_features_task(
    task: Tuple[int, str, str, int, float, float, int, int, int]
) -> Tuple[int, Dict[str, object] | None]:
    """单患者特征抽取任务（用于并行）。"""
    (
        patient_idx,
        pid,
        record_stem_str,
        label,
        followup_days,
        cause_of_death,
        seed,
        window_sec,
        max_windows_per_patient,
    ) = task

    try:
        rng = np.random.default_rng(seed)
        win_feats, fs = sample_windows_for_record(
            Path(record_stem_str),
            rng,
            window_sec=window_sec,
            max_windows_per_patient=max_windows_per_patient,
        )
        if len(win_feats) < 4:
            return patient_idx, None

        df_win = pd.DataFrame(win_feats)
        df_win.insert(0, "patient_id", pid)

        agg = cast(Dict[str, object], aggregate_dynamic_features(df_win))
        agg["patient_id"] = pid
        agg["label"] = int(label)
        agg["followup_days"] = float(followup_days) if np.isfinite(followup_days) else np.nan
        agg["cause_of_death"] = float(cause_of_death) if np.isfinite(cause_of_death) else np.nan
        agg["fs"] = fs
        return patient_idx, agg
    except Exception:
        return patient_idx, None


def build_dynamic_patient_table(
    max_patients: int | None = None,
    scd_only: bool = True,
    n_workers: int = DEFAULT_FEATURE_WORKERS,
    window_sec: int = WINDOW_SEC,
    max_windows_per_patient: int = MAX_WINDOWS_PER_PATIENT,
) -> pd.DataFrame:
    df_sub = read_subject_table(scd_only=scd_only)
    if max_patients is not None:
        df_sub = df_sub.head(max_patients).copy()

    tasks: List[Tuple[int, str, str, int, float, float, int, int, int]] = []
    for i, row in enumerate(df_sub.to_dict(orient="records"), start=0):
        seed = RANDOM_STATE + i * 104729
        tasks.append(
            (
                int(i),
                str(row.get("patient_id", "")),
                str(row.get("record_stem", "")),
                _safe_int_scalar(row.get("label_short_term"), default=0),
                _safe_float_scalar(row.get("followup_days")),
                _safe_float_scalar(row.get("cause_of_death")),
                int(seed),
                int(window_sec),
                int(max_windows_per_patient),
            )
        )

    worker_count = resolve_worker_count(int(n_workers))
    rows_with_idx: List[Tuple[int, Dict[str, object]]] = []

    if worker_count <= 1:
        for task in tasks:
            idx, agg = _extract_patient_features_task(task)
            if agg is not None:
                rows_with_idx.append((idx, agg))
    else:
        total = len(tasks)
        print(f"[INFO] 动态特征提取并行进程数: {worker_count}")
        with ProcessPoolExecutor(max_workers=worker_count) as ex:
            futures = [ex.submit(_extract_patient_features_task, task) for task in tasks]
            for n_done, fut in enumerate(as_completed(futures), start=1):
                idx, agg = fut.result()
                if agg is not None:
                    rows_with_idx.append((idx, agg))
                if n_done % 20 == 0 or n_done == total:
                    print(f"[INFO] 动态特征提取进度: {n_done}/{total}")

    rows_with_idx.sort(key=lambda x: x[0])
    rows = [r for _, r in rows_with_idx]

    df_pat = pd.DataFrame(rows)
    out_csv = OUTPUT_DIR / "dynamic_patient_features.csv"
    df_pat.to_csv(out_csv, index=False, encoding="utf-8-sig")
    print(f"[INFO] 已保存患者动态特征: {out_csv}")
    print(f"[INFO] shape={df_pat.shape}")

    return df_pat


def _json_safe_dict(d: Dict[str, object]) -> Dict[str, object]:
    out: Dict[str, object] = {}
    for k, v in d.items():
        if isinstance(v, np.generic):
            out[k] = v.item()
        else:
            out[k] = v
    return out


def train_and_evaluate(
    df: pd.DataFrame,
    val_size: float = DEFAULT_VAL_SIZE,
    cohort_mode: str = "scd",
    model_type: str = DEFAULT_MODEL_TYPE,
    threshold_method: str = DEFAULT_THRESHOLD_METHOD,
    target_specificity: float = DEFAULT_TARGET_SPECIFICITY,
    threshold_calib_size: float = DEFAULT_THRESHOLD_CALIB_SIZE,
    tune: bool = DEFAULT_TUNE,
    tune_n_iter: int = DEFAULT_TUNE_N_ITER,
    tune_cv_folds: int = DEFAULT_TUNE_CV_FOLDS,
    n_jobs: int = DEFAULT_N_JOBS,
) -> TrainInfo:
    x, y, feature_cols = prepare_features_and_labels(df)

    n_test = max(1, int(round(len(x) * val_size)))
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=n_test,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    x_fit, y_fit, x_cal, y_cal, calib_fallback = split_fit_calibration(
        x_train,
        y_train,
        calib_size=threshold_calib_size,
        random_state=RANDOM_STATE,
    )

    best_params: Dict[str, object] = {}
    tune_score = np.nan
    if tune:
        model, best_params, tune_score = tune_tree_model(
            x_fit,
            y_fit,
            feature_cols,
            model_type=model_type,
            tune_n_iter=tune_n_iter,
            tune_cv_folds=tune_cv_folds,
            n_jobs=n_jobs,
        )
    else:
        model = build_model(feature_cols, model_type=model_type, n_jobs=n_jobs)
        model.fit(x_fit, y_fit)

    # tune=True时best_estimator_已在训练集完成拟合；为健壮性保留一次检查
    if tune and "clf" not in model.named_steps:
        model.fit(x_fit, y_fit)

    prob_cal = model.predict_proba(x_cal)[:, 1]
    prob = model.predict_proba(x_test)[:, 1]

    auc = float(roc_auc_score(y_test, prob))
    pr_auc = float(average_precision_score(y_test, prob))
    brier = float(brier_score_loss(y_test, prob))

    best_thr = select_best_threshold(
        y_cal,
        prob_cal,
        method=threshold_method,
        target_specificity=target_specificity,
    )

    cal_pred = (prob_cal >= best_thr).astype(int)
    _, cal_sensitivity, cal_specificity, cal_f1 = compute_cls_stats(y_cal, cal_pred)

    pred = (prob >= best_thr).astype(int)
    cm, sensitivity, specificity, f1 = compute_cls_stats(y_test, pred)

    target_reached = None
    if threshold_method == "target-specificity":
        target_reached = bool(cal_specificity >= target_specificity - 1e-12)

    print("\n=== 动态短期风险模型评估 ===")
    print(f"Model: {model_label(model_type)}")
    print(f"Test ROC-AUC: {auc:.4f}")
    print(f"Test PR-AUC : {pr_auc:.4f}")
    print(f"Brier score : {brier:.4f}")
    print(f"Sensitivity : {sensitivity:.4f}")
    print(f"Specificity : {specificity:.4f}")
    print(f"F1 score    : {f1:.4f}")
    print(f"Calibration Specificity at threshold: {cal_specificity:.4f}")
    print(f"Train/Val split: {(1.0 - val_size):.1%}/{val_size:.1%}")
    print(f"Calibration split inside train: {threshold_calib_size:.1%}")
    if threshold_method == "target-specificity":
        print(f"Target specificity: {target_specificity:.3f}")
        print(f"Threshold selected on CALIBRATION ({threshold_method}): {best_thr:.4f}")
        print(f"Target reached on CALIBRATION: {target_reached}")
    else:
        print(f"Threshold selected on CALIBRATION ({threshold_method}): {best_thr:.4f}")
    if calib_fallback:
        print("[WARN] 校准集划分条件不足，阈值回退为在完整训练集上选择。")
    if tune:
        print(f"Tune PR-AUC (inner CV): {float(tune_score):.4f}")
    print("Confusion matrix:")
    print(cm)

    # 保存模型与阈值
    joblib.dump(model, OUTPUT_DIR / "dynamic_short_term_tree.joblib")
    pd.DataFrame([{"threshold": best_thr}]).to_csv(OUTPUT_DIR / "best_threshold.csv", index=False)

    # 保存指标
    metrics = {
        "model_type": model_label(model_type),
        "model_key": model_type,
        "tuned": bool(tune),
        "tune_n_iter": int(tune_n_iter),
        "tune_cv_folds": int(tune_cv_folds),
        "tune_pr_auc_cv": float(tune_score) if np.isfinite(tune_score) else None,
        "best_params": _json_safe_dict(best_params),
        "threshold_method": threshold_method,
        "target_specificity": float(target_specificity) if threshold_method == "target-specificity" else None,
        "threshold_selection_set": "train-calibration" if not calib_fallback else "train-fallback",
        "threshold_calib_size": float(threshold_calib_size),
        "threshold_calib_fallback": bool(calib_fallback),
        "calibration_threshold_specificity": cal_specificity,
        "calibration_threshold_sensitivity": cal_sensitivity,
        "calibration_threshold_f1": cal_f1,
        "target_specificity_reached_on_calibration": target_reached,
        "n_patients": int(len(df)),
        "positive_rate": float(y.mean()),
        "test_auc": auc,
        "test_pr_auc": pr_auc,
        "test_brier": brier,
        "test_sensitivity": sensitivity,
        "test_specificity": specificity,
        "test_f1": f1,
        "best_threshold": best_thr,
        "confusion_matrix": cm.tolist(),
        "val_size": float(val_size),
        "cohort_mode": cohort_mode,
        "short_term_days": SHORT_TERM_DAYS,
        "window_sec": WINDOW_SEC,
        "max_windows_per_patient": MAX_WINDOWS_PER_PATIENT,
        "model_n_jobs": int(n_jobs),
    }
    (OUTPUT_DIR / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    # 导出特征重要性
    importances = model.named_steps["clf"].feature_importances_.ravel()
    fi_df = pd.DataFrame(
        {
            "feature": feature_cols,
            "importance": importances,
        }
    ).sort_values("importance", ascending=False)
    fi_df.to_csv(OUTPUT_DIR / "feature_importance.csv", index=False, encoding="utf-8-sig")

    print("\n[INFO] 输出目录:", OUTPUT_DIR)
    print("[INFO] 文件: dynamic_short_term_tree.joblib, best_threshold.csv, metrics.json, feature_importance.csv")

    return {
        "model_type": model_type,
        "model_params": best_params,
        "threshold_method": threshold_method,
        "target_specificity": target_specificity,
        "threshold_calib_size": threshold_calib_size,
    }


def cross_validate_model(
    df: pd.DataFrame,
    cohort_mode: str = "scd",
    cv_folds: int = 5,
    model_type: str = DEFAULT_MODEL_TYPE,
    model_params: Dict[str, object] | None = None,
    threshold_method: str = DEFAULT_THRESHOLD_METHOD,
    target_specificity: float = DEFAULT_TARGET_SPECIFICITY,
    threshold_calib_size: float = DEFAULT_THRESHOLD_CALIB_SIZE,
    n_jobs: int = DEFAULT_N_JOBS,
) -> None:
    """执行分层K折交叉验证，输出逐折与汇总指标。"""
    x, y, feature_cols = prepare_features_and_labels(df)

    n_pos = int(y.sum())
    n_neg = int(len(y) - n_pos)
    max_valid_folds = min(cv_folds, n_pos, n_neg)
    if max_valid_folds < 2:
        raise ValueError(
            f"交叉验证不可用：cv_folds={cv_folds}, n_pos={n_pos}, n_neg={n_neg}。"
        )
    if max_valid_folds != cv_folds:
        print(
            f"[WARN] 由于类别样本限制，cv_folds 从 {cv_folds} 自动调整为 {max_valid_folds}。"
        )

    skf = StratifiedKFold(n_splits=max_valid_folds, shuffle=True, random_state=RANDOM_STATE)

    fold_rows: List[Dict[str, float]] = []
    for fold_idx, (tr_idx, te_idx) in enumerate(skf.split(x, y), start=1):
        x_tr = x.iloc[tr_idx]
        y_tr = y.iloc[tr_idx]
        x_te = x.iloc[te_idx]
        y_te = y.iloc[te_idx]

        x_fit, y_fit, x_cal, y_cal, calib_fallback = split_fit_calibration(
            x_tr,
            y_tr,
            calib_size=threshold_calib_size,
            random_state=RANDOM_STATE + fold_idx,
        )

        model = build_model(feature_cols, model_type=model_type, model_params=model_params, n_jobs=n_jobs)
        model.fit(x_fit, y_fit)
        prob_cal = model.predict_proba(x_cal)[:, 1]
        prob = model.predict_proba(x_te)[:, 1]

        auc = float(roc_auc_score(y_te, prob))
        pr_auc = float(average_precision_score(y_te, prob))
        brier = float(brier_score_loss(y_te, prob))

        best_thr = select_best_threshold(
            y_cal,
            prob_cal,
            method=threshold_method,
            target_specificity=target_specificity,
        )
        cal_pred = (prob_cal >= best_thr).astype(int)
        _, cal_sens, cal_spec, cal_f1 = compute_cls_stats(y_cal, cal_pred)

        pred = (prob >= best_thr).astype(int)
        cm, sens, spec, fold_f1 = compute_cls_stats(y_te, pred)
        tn, fp, fn, tp = cm.ravel()

        fold_rows.append(
            {
                "fold": float(fold_idx),
                "n_train": float(len(tr_idx)),
                "n_val": float(len(te_idx)),
                "val_positive_rate": float(y_te.mean()),
                "val_auc": auc,
                "val_pr_auc": pr_auc,
                "val_brier": brier,
                "best_threshold": best_thr,
                "threshold_calib_specificity": cal_spec,
                "threshold_calib_sensitivity": cal_sens,
                "threshold_calib_f1": cal_f1,
                "threshold_calib_fallback": float(1 if calib_fallback else 0),
                "tn": float(tn),
                "fp": float(fp),
                "fn": float(fn),
                "tp": float(tp),
                "sensitivity": sens,
                "specificity": spec,
                "f1": fold_f1,
            }
        )

    df_fold = pd.DataFrame(fold_rows)
    df_fold.to_csv(OUTPUT_DIR / "cv_fold_metrics.csv", index=False, encoding="utf-8-sig")

    summary = {
        "model_type": model_label(model_type),
        "model_key": model_type,
        "model_params": _json_safe_dict(model_params or {}),
        "threshold_method": threshold_method,
        "target_specificity": float(target_specificity) if threshold_method == "target-specificity" else None,
        "threshold_selection_set": "train-calibration-fold",
        "threshold_calib_size": float(threshold_calib_size),
        "threshold_calib_fallback_rate": float(df_fold["threshold_calib_fallback"].mean()) if len(df_fold) > 0 else np.nan,
        "cohort_mode": cohort_mode,
        "n_patients": int(len(df)),
        "positive_rate": float(y.mean()),
        "cv_folds": int(max_valid_folds),
        "auc_mean": float(df_fold["val_auc"].mean()),
        "auc_std": float(df_fold["val_auc"].std(ddof=1)),
        "pr_auc_mean": float(df_fold["val_pr_auc"].mean()),
        "pr_auc_std": float(df_fold["val_pr_auc"].std(ddof=1)),
        "brier_mean": float(df_fold["val_brier"].mean()),
        "brier_std": float(df_fold["val_brier"].std(ddof=1)),
        "sensitivity_mean": float(df_fold["sensitivity"].mean()),
        "specificity_mean": float(df_fold["specificity"].mean()),
        "f1_mean": float(df_fold["f1"].mean()),
        "f1_std": float(df_fold["f1"].std(ddof=1)),
        "short_term_days": SHORT_TERM_DAYS,
        "window_sec": WINDOW_SEC,
        "max_windows_per_patient": MAX_WINDOWS_PER_PATIENT,
        "model_n_jobs": int(n_jobs),
    }
    (OUTPUT_DIR / "cv_metrics.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\n=== 分层K折交叉验证 ===")
    print(f"Folds: {max_valid_folds}")
    print(f"CV ROC-AUC mean±std: {summary['auc_mean']:.4f} ± {summary['auc_std']:.4f}")
    print(f"CV PR-AUC  mean±std: {summary['pr_auc_mean']:.4f} ± {summary['pr_auc_std']:.4f}")
    print(f"CV Brier   mean±std: {summary['brier_mean']:.4f} ± {summary['brier_std']:.4f}")
    print(f"CV F1      mean±std: {summary['f1_mean']:.4f} ± {summary['f1_std']:.4f}")
    print("[INFO] 文件: cv_fold_metrics.csv, cv_metrics.json")


def repeated_cross_validate_model(
    df: pd.DataFrame,
    cohort_mode: str = "scd",
    cv_folds: int = 5,
    cv_repeats: int = DEFAULT_CV_REPEATS,
    model_type: str = DEFAULT_MODEL_TYPE,
    model_params: Dict[str, object] | None = None,
    threshold_method: str = DEFAULT_THRESHOLD_METHOD,
    target_specificity: float = DEFAULT_TARGET_SPECIFICITY,
    threshold_calib_size: float = DEFAULT_THRESHOLD_CALIB_SIZE,
    ci_alpha: float = DEFAULT_CI_ALPHA,
    n_jobs: int = DEFAULT_N_JOBS,
) -> None:
    """执行重复分层K折交叉验证，并给出95%CI等稳定性指标。"""
    if cv_repeats < 2:
        raise ValueError("重复交叉验证要求 cv_repeats >= 2。")

    x, y, feature_cols = prepare_features_and_labels(df)

    n_pos = int(y.sum())
    n_neg = int(len(y) - n_pos)
    max_valid_folds = min(cv_folds, n_pos, n_neg)
    if max_valid_folds < 2:
        raise ValueError(
            f"重复交叉验证不可用：cv_folds={cv_folds}, n_pos={n_pos}, n_neg={n_neg}。"
        )

    fold_rows: List[Dict[str, float]] = []
    repeat_rows: List[Dict[str, float]] = []

    for rep_idx in range(cv_repeats):
        rs = RANDOM_STATE + rep_idx
        skf = StratifiedKFold(n_splits=max_valid_folds, shuffle=True, random_state=rs)

        rep_auc: List[float] = []
        rep_pr_auc: List[float] = []
        rep_brier: List[float] = []
        rep_f1: List[float] = []
        rep_sens: List[float] = []
        rep_spec: List[float] = []

        for fold_idx, (tr_idx, te_idx) in enumerate(skf.split(x, y), start=1):
            x_tr = x.iloc[tr_idx]
            y_tr = y.iloc[tr_idx]
            x_te = x.iloc[te_idx]
            y_te = y.iloc[te_idx]

            x_fit, y_fit, x_cal, y_cal, calib_fallback = split_fit_calibration(
                x_tr,
                y_tr,
                calib_size=threshold_calib_size,
                random_state=rs + fold_idx,
            )

            model = build_model(feature_cols, model_type=model_type, model_params=model_params, n_jobs=n_jobs)
            model.fit(x_fit, y_fit)
            prob_cal = model.predict_proba(x_cal)[:, 1]
            prob = model.predict_proba(x_te)[:, 1]

            auc = float(roc_auc_score(y_te, prob))
            pr_auc = float(average_precision_score(y_te, prob))
            brier = float(brier_score_loss(y_te, prob))

            best_thr = select_best_threshold(
                y_cal,
                prob_cal,
                method=threshold_method,
                target_specificity=target_specificity,
            )
            cal_pred = (prob_cal >= best_thr).astype(int)
            _, cal_sens, cal_spec, cal_f1 = compute_cls_stats(y_cal, cal_pred)

            pred = (prob >= best_thr).astype(int)
            cm, sens, spec, fold_f1 = compute_cls_stats(y_te, pred)
            tn, fp, fn, tp = cm.ravel()

            fold_rows.append(
                {
                    "repeat": float(rep_idx + 1),
                    "repeat_seed": float(rs),
                    "fold": float(fold_idx),
                    "n_train": float(len(tr_idx)),
                    "n_val": float(len(te_idx)),
                    "val_positive_rate": float(y_te.mean()),
                    "val_auc": auc,
                    "val_pr_auc": pr_auc,
                    "val_brier": brier,
                    "best_threshold": best_thr,
                    "threshold_calib_specificity": cal_spec,
                    "threshold_calib_sensitivity": cal_sens,
                    "threshold_calib_f1": cal_f1,
                    "threshold_calib_fallback": float(1 if calib_fallback else 0),
                    "tn": float(tn),
                    "fp": float(fp),
                    "fn": float(fn),
                    "tp": float(tp),
                    "sensitivity": sens,
                    "specificity": spec,
                    "f1": fold_f1,
                }
            )

            rep_auc.append(auc)
            rep_pr_auc.append(pr_auc)
            rep_brier.append(brier)
            rep_f1.append(fold_f1)
            rep_sens.append(sens)
            rep_spec.append(spec)

        repeat_rows.append(
            {
                "repeat": float(rep_idx + 1),
                "repeat_seed": float(rs),
                "repeat_auc": float(np.mean(rep_auc)),
                "repeat_pr_auc": float(np.mean(rep_pr_auc)),
                "repeat_brier": float(np.mean(rep_brier)),
                "repeat_f1": float(np.mean(rep_f1)),
                "repeat_sensitivity": float(np.mean(rep_sens)),
                "repeat_specificity": float(np.mean(rep_spec)),
                "repeat_threshold_calib_specificity": float(
                    np.mean(
                        [
                            row["threshold_calib_specificity"]
                            for row in fold_rows
                            if int(row["repeat"]) == rep_idx + 1
                        ]
                    )
                ),
            }
        )

    df_fold = pd.DataFrame(fold_rows)
    df_repeat = pd.DataFrame(repeat_rows)

    df_fold.to_csv(OUTPUT_DIR / "repeated_cv_fold_metrics.csv", index=False, encoding="utf-8-sig")
    df_repeat.to_csv(OUTPUT_DIR / "repeated_cv_repeat_metrics.csv", index=False, encoding="utf-8-sig")

    auc_stat = summarize_with_ci(df_repeat["repeat_auc"].to_numpy(), ci_alpha=ci_alpha)
    pr_stat = summarize_with_ci(df_repeat["repeat_pr_auc"].to_numpy(), ci_alpha=ci_alpha)
    brier_stat = summarize_with_ci(df_repeat["repeat_brier"].to_numpy(), ci_alpha=ci_alpha)
    f1_stat = summarize_with_ci(df_repeat["repeat_f1"].to_numpy(), ci_alpha=ci_alpha)
    sens_stat = summarize_with_ci(df_repeat["repeat_sensitivity"].to_numpy(), ci_alpha=ci_alpha)
    spec_stat = summarize_with_ci(df_repeat["repeat_specificity"].to_numpy(), ci_alpha=ci_alpha)

    summary = {
        "model_type": model_label(model_type),
        "model_key": model_type,
        "model_params": _json_safe_dict(model_params or {}),
        "threshold_method": threshold_method,
        "target_specificity": float(target_specificity) if threshold_method == "target-specificity" else None,
        "threshold_selection_set": "train-calibration-fold",
        "threshold_calib_size": float(threshold_calib_size),
        "threshold_calib_fallback_rate": float(df_fold["threshold_calib_fallback"].mean()) if len(df_fold) > 0 else np.nan,
        "cohort_mode": cohort_mode,
        "n_patients": int(len(df)),
        "positive_rate": float(y.mean()),
        "cv_folds": int(max_valid_folds),
        "cv_repeats": int(cv_repeats),
        "total_validations": int(len(df_fold)),
        "ci_alpha": float(ci_alpha),
        "auc_repeat_mean": auc_stat["mean"],
        "auc_repeat_std": auc_stat["std"],
        "auc_ci": [auc_stat["ci_low"], auc_stat["ci_high"]],
        "pr_auc_repeat_mean": pr_stat["mean"],
        "pr_auc_repeat_std": pr_stat["std"],
        "pr_auc_ci": [pr_stat["ci_low"], pr_stat["ci_high"]],
        "brier_repeat_mean": brier_stat["mean"],
        "brier_repeat_std": brier_stat["std"],
        "brier_ci": [brier_stat["ci_low"], brier_stat["ci_high"]],
        "f1_repeat_mean": f1_stat["mean"],
        "f1_repeat_std": f1_stat["std"],
        "f1_ci": [f1_stat["ci_low"], f1_stat["ci_high"]],
        "sensitivity_repeat_mean": sens_stat["mean"],
        "sensitivity_ci": [sens_stat["ci_low"], sens_stat["ci_high"]],
        "specificity_repeat_mean": spec_stat["mean"],
        "specificity_ci": [spec_stat["ci_low"], spec_stat["ci_high"]],
        "short_term_days": SHORT_TERM_DAYS,
        "window_sec": WINDOW_SEC,
        "max_windows_per_patient": MAX_WINDOWS_PER_PATIENT,
        "model_n_jobs": int(n_jobs),
    }

    (OUTPUT_DIR / "repeated_cv_metrics.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\n=== 重复分层K折交叉验证（可靠性） ===")
    print(f"Folds x Repeats: {max_valid_folds} x {cv_repeats} = {len(df_fold)}")
    print(
        f"Repeat ROC-AUC mean={auc_stat['mean']:.4f}, "
        f"{int(ci_alpha*100)}%CI [{auc_stat['ci_low']:.4f}, {auc_stat['ci_high']:.4f}]"
    )
    print(
        f"Repeat PR-AUC  mean={pr_stat['mean']:.4f}, "
        f"{int(ci_alpha*100)}%CI [{pr_stat['ci_low']:.4f}, {pr_stat['ci_high']:.4f}]"
    )
    print(
        f"Repeat F1      mean={f1_stat['mean']:.4f}, "
        f"{int(ci_alpha*100)}%CI [{f1_stat['ci_low']:.4f}, {f1_stat['ci_high']:.4f}]"
    )
    print("[INFO] 文件: repeated_cv_fold_metrics.csv, repeated_cv_repeat_metrics.csv, repeated_cv_metrics.json")


def main() -> None:
    global WINDOW_SEC, MAX_WINDOWS_PER_PATIENT

    parser = argparse.ArgumentParser(description="动态心电短期风险预测")
    parser.add_argument("--max-patients", type=int, default=None, help="限制患者数以便快速调试")
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=DEFAULT_N_JOBS,
        help="模型训练并行线程数；-1表示使用全部CPU核心",
    )
    parser.add_argument(
        "--feature-workers",
        type=int,
        default=DEFAULT_FEATURE_WORKERS,
        help="动态特征提取并行进程数；-1表示使用全部CPU核心",
    )
    parser.add_argument(
        "--window-sec",
        type=int,
        default=WINDOW_SEC,
        help="动态窗口长度（秒），默认120",
    )
    parser.add_argument(
        "--max-windows-per-patient",
        type=int,
        default=MAX_WINDOWS_PER_PATIENT,
        help="每位患者最多抽样窗口数，默认24",
    )
    parser.add_argument(
        "--fast",
        dest="fast",
        action="store_true",
        help="快速模式：减少窗口数量并关闭调参/重复CV以显著提速（默认开启）",
    )
    parser.add_argument(
        "--full",
        dest="fast",
        action="store_false",
        help="关闭快速模式，按完整流程运行（更慢）",
    )
    parser.add_argument(
        "--reuse-features",
        action="store_true",
        help="若已存在 dynamic_patient_features.csv 且不限制样本数，则直接复用该特征表",
    )
    parser.add_argument(
        "--cohort",
        choices=["scd", "all"],
        default="scd",
        help="scd: 仅猝死患者队列（默认）; all: 全队列",
    )
    parser.add_argument(
        "--val-size",
        type=float,
        default=DEFAULT_VAL_SIZE,
        help="验证集占比，默认0.30（即训练/验证=7:3）",
    )
    parser.add_argument(
        "--cv-folds",
        type=int,
        default=5,
        help="分层K折交叉验证折数（>=2时启用，默认5）",
    )
    parser.add_argument(
        "--cv-repeats",
        type=int,
        default=DEFAULT_CV_REPEATS,
        help="重复交叉验证次数（>=2时启用；默认0表示关闭）",
    )
    parser.add_argument(
        "--ci-alpha",
        type=float,
        default=DEFAULT_CI_ALPHA,
        help="重复交叉验证置信区间置信度，默认0.95",
    )
    parser.add_argument(
        "--model",
        choices=["rf", "et"],
        default=DEFAULT_MODEL_TYPE,
        help="树模型类型：rf=随机森林, et=极端随机树",
    )
    parser.add_argument(
        "--threshold-method",
        choices=["f1", "youden", "target-specificity"],
        default=DEFAULT_THRESHOLD_METHOD,
        help="阈值选择方法：f1 / youden / target-specificity",
    )
    parser.add_argument(
        "--target-specificity",
        type=float,
        default=DEFAULT_TARGET_SPECIFICITY,
        help="固定目标特异度（仅在 threshold-method=target-specificity 时生效），默认0.70",
    )
    parser.add_argument(
        "--threshold-calib-size",
        type=float,
        default=DEFAULT_THRESHOLD_CALIB_SIZE,
        help="在训练集中用于阈值校准的比例，默认0.25",
    )
    parser.add_argument(
        "--no-tune",
        action="store_true",
        help="关闭训练集内随机搜索调参（默认开启）",
    )
    parser.add_argument(
        "--tune-n-iter",
        type=int,
        default=DEFAULT_TUNE_N_ITER,
        help="随机搜索迭代次数，默认24",
    )
    parser.add_argument(
        "--tune-cv-folds",
        type=int,
        default=DEFAULT_TUNE_CV_FOLDS,
        help="调参内层CV折数，默认4",
    )
    parser.set_defaults(fast=DEFAULT_FAST_MODE)
    args = parser.parse_args()

    max_patients = args.max_patients
    scd_only = args.cohort == "scd"
    val_size = float(args.val_size)
    cv_folds = int(args.cv_folds)
    cv_repeats = int(args.cv_repeats)
    ci_alpha = float(args.ci_alpha)
    model_type = args.model
    threshold_method = args.threshold_method
    target_specificity = float(args.target_specificity)
    threshold_calib_size = float(args.threshold_calib_size)
    tune = not bool(args.no_tune)
    tune_n_iter = int(args.tune_n_iter)
    tune_cv_folds = int(args.tune_cv_folds)
    n_jobs = int(args.n_jobs)
    feature_workers = int(args.feature_workers)
    window_sec = int(args.window_sec)
    max_windows_per_patient = int(args.max_windows_per_patient)

    if args.fast:
        tune = False
        cv_repeats = 0
        cv_folds = min(cv_folds, 3)
        max_windows_per_patient = min(max_windows_per_patient, FAST_MAX_WINDOWS_PER_PATIENT)
        print(
            "[INFO] 快速模式已启用：关闭调参与重复CV，"
            f"cv_folds<=3，max_windows_per_patient<={FAST_MAX_WINDOWS_PER_PATIENT}。"
        )

    worker_count = resolve_worker_count(feature_workers)

    WINDOW_SEC = window_sec
    MAX_WINDOWS_PER_PATIENT = max_windows_per_patient

    if not (0.05 <= val_size <= 0.50):
        raise ValueError("--val-size 建议在 [0.05, 0.50] 范围内。")
    if cv_folds < 0:
        raise ValueError("--cv-folds 不能为负数。")
    if cv_repeats < 0:
        raise ValueError("--cv-repeats 不能为负数。")
    if not (0.5 <= ci_alpha < 1.0):
        raise ValueError("--ci-alpha 必须在 [0.5, 1.0) 范围内。")
    if not (0.0 <= target_specificity <= 1.0):
        raise ValueError("--target-specificity 必须在 [0.0, 1.0] 范围内。")
    if not (0.0 <= threshold_calib_size < 0.5):
        raise ValueError("--threshold-calib-size 必须在 [0.0, 0.5) 范围内。")
    if tune_n_iter < 1:
        raise ValueError("--tune-n-iter 必须 >= 1。")
    if tune_cv_folds < 2:
        raise ValueError("--tune-cv-folds 必须 >= 2。")
    if n_jobs == 0 or n_jobs < -1:
        raise ValueError("--n-jobs 必须为 -1 或正整数。")
    if feature_workers == 0 or feature_workers < -1:
        raise ValueError("--feature-workers 必须为 -1 或正整数。")
    if window_sec < 30:
        raise ValueError("--window-sec 必须 >= 30 秒。")
    if max_windows_per_patient < 4:
        raise ValueError("--max-windows-per-patient 必须 >= 4。")

    print(f"[INFO] 配置: model_n_jobs={n_jobs}, feature_workers={worker_count}")
    print(f"[INFO] 配置: window_sec={WINDOW_SEC}, max_windows_per_patient={MAX_WINDOWS_PER_PATIENT}")

    feature_csv = OUTPUT_DIR / "dynamic_patient_features.csv"
    if args.reuse_features and feature_csv.exists() and max_patients is None:
        df_pat = pd.read_csv(feature_csv, encoding="utf-8-sig")
        print(f"[INFO] 已复用现有动态特征: {feature_csv}")
        print(f"[INFO] shape={df_pat.shape}")
    else:
        if args.reuse_features and max_patients is not None:
            print("[WARN] 指定了 --max-patients，已忽略 --reuse-features 并重建特征。")
        df_pat = build_dynamic_patient_table(
            max_patients=max_patients,
            scd_only=scd_only,
            n_workers=worker_count,
            window_sec=WINDOW_SEC,
            max_windows_per_patient=MAX_WINDOWS_PER_PATIENT,
        )

    if len(df_pat) < MIN_PATIENTS_FOR_TRAIN:
        print(
            f"[WARN] 可用患者数={len(df_pat)}，低于 MIN_PATIENTS_FOR_TRAIN={MIN_PATIENTS_FOR_TRAIN}，"
            "模型稳定性可能不足。"
        )

    if len(df_pat) >= 30 and df_pat["label"].nunique() == 2:
        train_info = train_and_evaluate(
            df_pat,
            val_size=val_size,
            cohort_mode=args.cohort,
            model_type=model_type,
            threshold_method=threshold_method,
            target_specificity=target_specificity,
            threshold_calib_size=threshold_calib_size,
            tune=tune,
            tune_n_iter=tune_n_iter,
            tune_cv_folds=tune_cv_folds,
            n_jobs=n_jobs,
        )
        if cv_folds >= 2:
            cross_validate_model(
                df_pat,
                cohort_mode=args.cohort,
                cv_folds=cv_folds,
                model_type=train_info["model_type"],
                model_params=train_info["model_params"] or None,
                threshold_method=train_info["threshold_method"],
                target_specificity=train_info["target_specificity"],
                threshold_calib_size=train_info["threshold_calib_size"],
                n_jobs=n_jobs,
            )
        else:
            print("[INFO] 已跳过交叉验证（cv_folds < 2）。")

        if cv_folds >= 2 and cv_repeats >= 2:
            repeated_cross_validate_model(
                df_pat,
                cohort_mode=args.cohort,
                cv_folds=cv_folds,
                cv_repeats=cv_repeats,
                model_type=train_info["model_type"],
                model_params=train_info["model_params"] or None,
                threshold_method=train_info["threshold_method"],
                target_specificity=train_info["target_specificity"],
                threshold_calib_size=train_info["threshold_calib_size"],
                ci_alpha=ci_alpha,
                n_jobs=n_jobs,
            )
        else:
            print("[INFO] 已跳过重复交叉验证（cv_folds < 2 或 cv_repeats < 2）。")
    else:
        print("[ERROR] 样本量过小或标签单一，无法训练二分类模型。")


if __name__ == "__main__":
    main()
