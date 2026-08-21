import argparse
import json
from contextlib import nullcontext
import random
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.under_sampling import RandomUnderSampler
from lightgbm import LGBMClassifier
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import QuantileTransformer, StandardScaler
from torch.utils.data import DataLoader, TensorDataset

import sys

ROOT = Path(__file__).resolve().parents[2]
VENV_DIR = ROOT / ".venv"
if str(VENV_DIR) not in sys.path:
    sys.path.append(str(VENV_DIR))

from augmenters_torch_ddpm import TorchDDPMConfig, TorchTabDDPMConditionalAugmenter


@dataclass
class PipelineConfig:
    data_root: str = ".venv/softsensing_data_full"
    toolset: str = "time-series-3"
    representation: str = "mean"  # mean | flatten | delta | summary_stats
    x_mmap_mode: str = "r"
    use_rep_cache: bool = True
    rep_cache_dir: str = "experiments/seagate_kqi/cache"
    taskwise_representation: bool = False

    seed: int = 42
    top_k_tasks: int = 3
    explicit_task_ids: Optional[List[int]] = None
    min_valid_per_task: int = 500

    scenarios: Tuple[str, ...] = ("S0", "S1", "S2", "S3")
    gen_ratios: Tuple[float, ...] = (0.5, 1.0, 2.0)

    drop_constant: bool = True
    imputer_strategy: str = "mean"

    lgbm_n_estimators: int = 400
    lgbm_learning_rate: float = 0.05
    lgbm_num_leaves: int = 63
    lgbm_subsample: float = 0.9
    lgbm_colsample_bytree: float = 0.9
    lgbm_device_type: str = "gpu"  # gpu | cpu
    lgbm_gpu_platform_id: int = 0
    lgbm_gpu_device_id: int = 0

    threshold_rule: str = "max_mcc"  # max_mcc | max_f1 | recall_target
    recall_target: float = 0.90
    fpr_caps: Tuple[float, ...] = (0.01, 0.02, 0.05)
    strict_eval: bool = False
    selection_metric: str = "PR_AUC"
    selection_threshold_rule: str = "max_mcc"
    paper_task_mode: str = "explicit_or_topk"

    s1_under_neg_pos_ratios: Tuple[float, ...] = (20.0, 10.0)
    s1_include_smote: bool = False
    s1_smote_k_neighbors: int = 5

    s2_train_neg_multiplier: float = 10.0
    s2_methods: Tuple[str, ...] = ("cvae", "cgan")

    cvae_latent_dim: int = 32
    cvae_hidden: int = 256
    cvae_epochs: int = 40
    cvae_batch_size: int = 256
    cvae_lr: float = 1e-3
    cvae_beta: float = 1e-3

    cgan_noise_dim: int = 64
    cgan_hidden: int = 256
    cgan_epochs: int = 80
    cgan_batch_size: int = 256
    cgan_lr: float = 2e-4
    torch_device: str = "cuda"
    torch_use_amp: bool = True
    loader_num_workers: int = 0
    loader_pin_memory: bool = True
    generator_gpu_resident: bool = True
    generator_gpu_resident_max_mb: int = 2048

    ddpm_T: int = 100
    ddpm_beta_start: float = 1e-4
    ddpm_beta_end: float = 2e-2
    ddpm_time_dim: int = 128
    ddpm_class_dim: int = 16
    ddpm_hidden_dim: int = 256
    ddpm_n_layers: int = 4
    ddpm_dropout: float = 0.1
    ddpm_epochs: int = 30
    ddpm_batch_size: int = 256
    ddpm_lr: float = 1e-3
    ddpm_device: str = "cuda"
    ddpm_train_mode: str = "conditional_mixed"  # conditional_mixed | balanced_conditional | positive_only
    strict_ddpm_train_modes: Tuple[str, ...] = ("positive_only", "conditional_mixed")
    ddpm_neg_multiplier: float = 10.0
    ddpm_max_neg: int = 50000
    ddpm_scale: str = "quantile"  # standard | quantile
    ddpm_core_variant: str = "baseline"  # baseline | tabdiff_min
    ddpm_featurewise_spike_scale: float = 0.60
    ddpm_featurewise_corr_scale: float = 0.80
    ddpm_sampler_correction_every: int = 10
    ddpm_sampler_correction_alpha: float = 0.20
    ddpm_sampler_corr_rank: int = 4
    s3_auto_mode_by_pos_rate: bool = False
    s3_positive_only_max_rate: float = 0.015
    s3_synth_filter: str = "none"  # none | knn_pos | disc_pos | hybrid_pos | knn_all_features | ig_topk | ig_weighted | random_keep
    s3_synth_filter_k: int = 5
    s3_synth_pool_multiplier: float = 1.0
    s3_ig_importance_path: str = ""
    s3_ig_top_k: int = 10
    s3_ig_keep_rates: Tuple[float, ...] = (0.25, 0.5, 0.75, 1.0)
    s3_hybrid_methods: Tuple[str, ...] = ("ig_only", "ig_clean", "ig_smote", "s1_plus_ig")
    s3_hybrid_smote_ratio: float = 0.5
    s3_hybrid_s1_ratio: float = 0.5
    s3_hybrid_clean_keep_rate: float = 0.5
    s3_postprocess: str = "none"  # none | clip_pos_q | anchor_pos_q | clip_anchor_pos_q | spike_match_pos | featurewise_quantile_pos | corr_pair_anchor_pos | corr_group_anchor_pos | latent_pca_pos
    s3_postprocess_low_q: float = 0.01
    s3_postprocess_high_q: float = 0.99
    s3_postprocess_anchor_alpha: float = 0.25
    s3_postprocess_corr_top_k: int = 6
    s3_postprocess_group_top_features: int = 8
    s3_postprocess_pca_var: float = 0.99

    output_root: str = "experiments/seagate_kqi/outputs"
    run_name: str = ""
    cars_canary_probe: bool = False
    cars_canary_q: float = 0.10
    cars_canary_max_n: int = 5000
    cars_canary_min_n: int = 20
    cars_canary_distance_top_k: int = 32
    cars_canary_distance_max_neg: int = 10000
    cars_candidate_replay: bool = False
    cars_candidate_replay_include_test: bool = False
    cars_candidate_replay_save_predictions: bool = False
    cars_bw_prototype: bool = False
    cars_bw_variants: Tuple[str, ...] = ("balanced",)
    cars_bw_min_weight: float = 0.05
    cars_bw_max_weight: float = 1.0
    cars_bw_power: float = 1.0
    cars_bw_top_k_features: int = 32
    cars_bw_sort_synthetic: bool = False
    cars_bw_normalize_mean: bool = False


SUMMARY_METRICS = ["PR_AUC", "ROC_AUC", "Recall", "FPR", "MCC"]
OPERATING_METRICS = ["F1", "Precision", "MCC", "FPR", "Threshold", "TN", "FP", "FN", "TP"]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_torch_device(device_str: str) -> torch.device:
    if device_str.startswith("cuda") and torch.cuda.is_available():
        return torch.device(device_str)
    return torch.device("cpu")


def configure_torch_runtime(device: torch.device) -> None:
    if device.type != "cuda":
        return
    torch.backends.cudnn.benchmark = True
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")


def loader_runtime_kwargs(cfg: PipelineConfig, device: torch.device) -> Dict[str, object]:
    kwargs: Dict[str, object] = {
        "num_workers": max(0, int(cfg.loader_num_workers)),
        "pin_memory": bool(cfg.loader_pin_memory and device.type == "cuda"),
    }
    if kwargs["num_workers"] > 0:
        kwargs["persistent_workers"] = True
    return kwargs


def make_grad_scaler(enabled: bool):
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        return torch.amp.GradScaler("cuda", enabled=enabled)
    return torch.cuda.amp.GradScaler(enabled=enabled)


def autocast_ctx(enabled: bool):
    if enabled and hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
        return torch.amp.autocast(device_type="cuda", dtype=torch.float16)
    return nullcontext()


def can_preload_generator_to_gpu(X: np.ndarray, cfg: PipelineConfig, device: torch.device) -> bool:
    if device.type != "cuda" or not cfg.generator_gpu_resident:
        return False
    max_bytes = int(cfg.generator_gpu_resident_max_mb) * 1024 * 1024
    return int(X.nbytes) <= max_bytes


def iter_gpu_tensor_batches(
    X: torch.Tensor,
    batch_size: int,
    shuffle: bool = True,
    drop_last: bool = False,
) -> Iterator[torch.Tensor]:
    n = int(X.shape[0])
    order = torch.randperm(n, device=X.device) if shuffle else torch.arange(n, device=X.device)
    end = n if not drop_last else (n // batch_size) * batch_size
    for start in range(0, end, batch_size):
        idx = order[start : start + batch_size]
        if drop_last and idx.numel() < batch_size:
            continue
        yield X.index_select(0, idx)


def parse_csv_floats(text: str) -> Tuple[float, ...]:
    return tuple(float(x.strip()) for x in text.split(",") if x.strip())


def parse_csv_ints(text: str) -> List[int]:
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def parse_csv_strings(text: str) -> Tuple[str, ...]:
    return tuple(x.strip() for x in text.split(",") if x.strip())


def _normalize_mmap_mode(mmap_mode: str) -> Optional[str]:
    text = str(mmap_mode).strip().lower()
    if text in {"", "none", "null", "false"}:
        return None
    return mmap_mode


def _load_numpy(path: Path, mmap_mode: Optional[str]) -> np.ndarray:
    if mmap_mode is None:
        return np.load(path)
    return np.load(path, mmap_mode=mmap_mode)


def load_toolset_arrays(data_root: str, toolset: str, mmap_mode: str):
    base = Path(data_root) / toolset
    x_mmap = _normalize_mmap_mode(mmap_mode)
    X_train = _load_numpy(base / "X_train.npy", x_mmap)
    X_val = _load_numpy(base / "X_val.npy", x_mmap)
    X_test = _load_numpy(base / "X_test.npy", x_mmap)
    y_train = np.load(base / "label_train.npy")
    y_val = np.load(base / "label_val.npy")
    y_test = np.load(base / "label_test.npy")
    return X_train, X_val, X_test, y_train, y_val, y_test


def resolve_rep_cache_dir(cfg: PipelineConfig) -> Path:
    base = Path(cfg.rep_cache_dir)
    if not base.is_absolute():
        base = ROOT / base
    return base / cfg.toolset


def rep_cache_path(cfg: PipelineConfig, split_name: str) -> Path:
    return resolve_rep_cache_dir(cfg) / f"{split_name}_{cfg.representation}_f32.npy"


def load_cached_representation(cfg: PipelineConfig, split_name: str) -> Optional[np.ndarray]:
    if not cfg.use_rep_cache:
        return None
    path = rep_cache_path(cfg, split_name)
    if not path.exists():
        return None
    return np.load(path, mmap_mode="r")


def build_or_load_representation(X: np.ndarray, cfg: PipelineConfig, split_name: str) -> np.ndarray:
    cached = load_cached_representation(cfg, split_name)
    if cached is not None:
        print(f"[INFO] rep cache hit: {split_name} -> {rep_cache_path(cfg, split_name).name}")
        return cached

    X_rep = build_representation(X, cfg.representation).astype(np.float32, copy=False)
    if not cfg.use_rep_cache:
        return X_rep

    cache_path = rep_cache_path(cfg, split_name)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache_path, X_rep)
    print(f"[INFO] rep cache saved: {split_name} -> {cache_path.name}")
    return np.load(cache_path, mmap_mode="r")


def task_stats_one_split(labels: np.ndarray, split_name: str) -> pd.DataFrame:
    n_tasks = labels.shape[1] // 2
    rows = []
    for task in range(n_tasks):
        neg = labels[:, 2 * task]
        pos = labels[:, 2 * task + 1]
        missing = ((neg == 0) & (pos == 0))
        valid = ~missing

        valid_count = int(valid.sum())
        pos_count = int((pos[valid] == 1).sum())
        neg_count = int((neg[valid] == 1).sum())

        rows.append(
            {
                "split": split_name,
                "task_id": task,
                "total": int(labels.shape[0]),
                "valid": valid_count,
                "missing": int(missing.sum()),
                "positive": pos_count,
                "negative": neg_count,
                "positive_rate": float(pos_count / max(valid_count, 1)),
            }
        )
    return pd.DataFrame(rows)


def choose_tasks(train_stats: pd.DataFrame, cfg: PipelineConfig) -> List[int]:
    if cfg.explicit_task_ids is not None and len(cfg.explicit_task_ids) > 0:
        return cfg.explicit_task_ids

    ok = train_stats[train_stats["valid"] >= cfg.min_valid_per_task].copy()
    ok = ok.sort_values(["positive", "valid"], ascending=[False, False])
    return ok.head(cfg.top_k_tasks)["task_id"].astype(int).tolist()


def extract_task_binary(labels: np.ndarray, task_id: int) -> Tuple[np.ndarray, np.ndarray]:
    neg = labels[:, 2 * task_id]
    pos = labels[:, 2 * task_id + 1]
    valid = (neg + pos) > 0
    y = (pos[valid] == 1).astype(int)
    return valid, y


def _nonpad_mask(X: np.ndarray, pad_value: float = 1.0) -> np.ndarray:
    return X[:, :, -1] != pad_value


def ts_to_tabular_mean(X: np.ndarray) -> np.ndarray:
    X_num = X[:, :, :-1].astype(np.float32)
    valid = _nonpad_mask(X).astype(np.float32)
    valid_sum = valid.sum(axis=1, keepdims=True)
    valid_sum[valid_sum == 0] = 1.0
    return (X_num * valid[:, :, None]).sum(axis=1) / valid_sum


def ts_to_tabular_flatten(X: np.ndarray) -> np.ndarray:
    X_num = X[:, :, :-1].astype(np.float32)
    valid = _nonpad_mask(X).astype(np.float32)
    X_masked = X_num * valid[:, :, None]
    return X_masked.reshape(X_masked.shape[0], -1)


def ts_to_tabular_delta(X: np.ndarray) -> np.ndarray:
    X_num = X[:, :, :-1].astype(np.float32)
    valid = _nonpad_mask(X)
    n, t, _ = X_num.shape

    first_idx = np.argmax(valid, axis=1)
    rev_idx = np.argmax(valid[:, ::-1], axis=1)
    last_idx = t - 1 - rev_idx

    has_valid = valid.any(axis=1)
    first_idx[~has_valid] = 0
    last_idx[~has_valid] = 0

    first = X_num[np.arange(n), first_idx]
    last = X_num[np.arange(n), last_idx]
    return (last - first).astype(np.float32)


def ts_to_tabular_summary_stats(X: np.ndarray) -> np.ndarray:
    X_num = X[:, :, :-1].astype(np.float32)
    valid = _nonpad_mask(X)
    n, t, f = X_num.shape
    if t <= 1:
        return ts_to_tabular_mean(X)
    valid_f = valid.astype(np.float32)
    count = valid_f.sum(axis=1)
    safe_count = count.copy()
    safe_count[safe_count == 0] = 1.0

    sum_x = (X_num * valid_f[:, :, None]).sum(axis=1)
    mean = sum_x / safe_count[:, None]

    sum_x2 = ((X_num ** 2) * valid_f[:, :, None]).sum(axis=1)
    var = np.maximum(sum_x2 / safe_count[:, None] - mean ** 2, 0.0)
    std = np.sqrt(var)

    X_for_min = np.where(valid[:, :, None], X_num, np.inf)
    X_for_max = np.where(valid[:, :, None], X_num, -np.inf)
    min_v = np.min(X_for_min, axis=1)
    max_v = np.max(X_for_max, axis=1)
    min_v[~np.isfinite(min_v)] = 0.0
    max_v[~np.isfinite(max_v)] = 0.0

    rev_idx = np.argmax(valid[:, ::-1], axis=1)
    last_idx = t - 1 - rev_idx
    has_valid = valid.any(axis=1)
    last_idx[~has_valid] = 0
    last = X_num[np.arange(n), last_idx]
    last[~has_valid] = 0.0

    time_idx = np.arange(t, dtype=np.float32)
    sum_t = (valid_f * time_idx[None, :]).sum(axis=1)
    sum_t2 = (valid_f * (time_idx[None, :] ** 2)).sum(axis=1)
    sum_tx = (X_num * valid_f[:, :, None] * time_idx[None, :, None]).sum(axis=1)
    denom = sum_t2 - (sum_t ** 2) / safe_count
    numer = sum_tx - (sum_t[:, None] * sum_x) / safe_count[:, None]
    slope = np.divide(numer, denom[:, None], out=np.zeros((n, f), dtype=np.float32), where=denom[:, None] > 1e-12)

    return np.concatenate([mean, std, min_v, max_v, last, slope], axis=1).astype(np.float32, copy=False)


def build_representation(X: np.ndarray, rep: str) -> np.ndarray:
    if rep == "mean":
        return ts_to_tabular_mean(X)
    if rep == "flatten":
        return ts_to_tabular_flatten(X)
    if rep == "delta":
        return ts_to_tabular_delta(X)
    if rep == "summary_stats":
        return ts_to_tabular_summary_stats(X)
    raise ValueError(f"Unknown representation: {rep}")


class TabularPreprocessor:
    def __init__(self, drop_constant: bool = True, imputer_strategy: str = "mean"):
        self.drop_constant = drop_constant
        self.imputer_strategy = imputer_strategy
        self.keep_mask = None
        self.imputer = None

    def fit(self, X_train: np.ndarray):
        if self.drop_constant:
            mins = np.nanmin(X_train, axis=0)
            maxs = np.nanmax(X_train, axis=0)
            self.keep_mask = mins != maxs
            if not np.any(self.keep_mask):
                self.keep_mask = np.ones(X_train.shape[1], dtype=bool)
        else:
            self.keep_mask = np.ones(X_train.shape[1], dtype=bool)

        Xk = X_train[:, self.keep_mask]
        self.imputer = SimpleImputer(strategy=self.imputer_strategy)
        self.imputer.fit(Xk)

    def transform_classifier(self, X: np.ndarray) -> np.ndarray:
        Xk = X[:, self.keep_mask]
        return self.imputer.transform(Xk)


def _fpr_from_cm(cm: np.ndarray) -> float:
    tn, fp, fn, tp = cm.ravel()
    return float(fp / max(tn + fp, 1))


def evaluate_at_threshold(y_true: np.ndarray, p: np.ndarray, threshold: float) -> Dict[str, float]:
    y_hat = (p >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_hat, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    return {
        "ROC_AUC": float(roc_auc_score(y_true, p)),
        "PR_AUC": float(average_precision_score(y_true, p)),
        "F1": float(f1_score(y_true, y_hat, zero_division=0)),
        "Recall": float(recall_score(y_true, y_hat, zero_division=0)),
        "Precision": float(precision_score(y_true, y_hat, zero_division=0)),
        "MCC": float(matthews_corrcoef(y_true, y_hat)),
        "FPR": _fpr_from_cm(cm),
        "Threshold": float(threshold),
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp),
    }


def add_recall_at_fpr_caps(
    out: Dict[str, float],
    y_ref: np.ndarray,
    p_ref: np.ndarray,
    y_eval: np.ndarray,
    p_eval: np.ndarray,
    fpr_caps: Tuple[float, ...],
) -> Dict[str, float]:
    for cap in fpr_caps:
        t_cap = threshold_for_fpr_cap(y_ref, p_ref, cap)
        m_cap = evaluate_at_threshold(y_eval, p_eval, t_cap)
        cap_key = f"Recall_at_FPR{int(cap * 1000):03d}"
        out[cap_key] = float(m_cap["Recall"])
    return out


def _run_downstream_internal(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    cfg: PipelineConfig,
    seed: int,
    threshold_rule: str,
    eval_split: str,
) -> Dict[str, float]:
    metrics, _, _ = _run_downstream_with_predictions(
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
        y_test,
        cfg,
        seed,
        threshold_rule,
        eval_split,
    )
    return metrics


def _run_downstream_with_predictions(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    cfg: PipelineConfig,
    seed: int,
    threshold_rule: str,
    eval_split: str,
    sample_weight: Optional[np.ndarray] = None,
) -> Tuple[Dict[str, float], Dict[str, np.ndarray | float], LGBMClassifier]:
    model = train_lgbm(X_train, y_train, cfg, seed, sample_weight=sample_weight)
    p_val = model.predict_proba(X_val)[:, 1]
    thr = tune_threshold(y_val, p_val, threshold_rule, cfg.recall_target)

    if eval_split == "val":
        main = evaluate_at_threshold(y_val, p_val, thr)
        metrics = add_recall_at_fpr_caps(main, y_val, p_val, y_val, p_val, cfg.fpr_caps)
        return metrics, {"p_val": p_val, "threshold": float(thr)}, model

    p_test = model.predict_proba(X_test)[:, 1]
    main = evaluate_at_threshold(y_test, p_test, thr)
    metrics = add_recall_at_fpr_caps(main, y_val, p_val, y_test, p_test, cfg.fpr_caps)
    return metrics, {"p_val": p_val, "p_test": p_test, "threshold": float(thr)}, model


def _run_downstream_val_test_with_predictions(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    cfg: PipelineConfig,
    seed: int,
    threshold_rule: str,
    include_test: bool,
    sample_weight: Optional[np.ndarray] = None,
) -> Tuple[Dict[str, float], Optional[Dict[str, float]], Dict[str, np.ndarray | float], LGBMClassifier]:
    model = train_lgbm(X_train, y_train, cfg, seed, sample_weight=sample_weight)
    p_val = model.predict_proba(X_val)[:, 1]
    thr = tune_threshold(y_val, p_val, threshold_rule, cfg.recall_target)

    val_main = evaluate_at_threshold(y_val, p_val, thr)
    val_metrics = add_recall_at_fpr_caps(val_main, y_val, p_val, y_val, p_val, cfg.fpr_caps)
    preds: Dict[str, np.ndarray | float] = {"p_val": p_val, "threshold": float(thr)}

    test_metrics = None
    if include_test:
        p_test = model.predict_proba(X_test)[:, 1]
        test_main = evaluate_at_threshold(y_test, p_test, thr)
        test_metrics = add_recall_at_fpr_caps(test_main, y_val, p_val, y_test, p_test, cfg.fpr_caps)
        preds["p_test"] = p_test

    return val_metrics, test_metrics, preds, model


def candidate_key_from_row(row: Dict[str, object]) -> str:
    scenario = str(row.get("scenario", ""))
    method = str(row.get("method", ""))
    ratio = row.get("ratio", float("nan"))
    parts = [scenario, method]
    if not pd.isna(ratio):
        parts.append(f"ratio={float(ratio):g}")
    if "s3_ig_keep_rate" in row and not pd.isna(row.get("s3_ig_keep_rate")):
        parts.append(f"keep={float(row['s3_ig_keep_rate']):g}")
    if row.get("ddpm_train_mode", ""):
        parts.append(f"mode={row['ddpm_train_mode']}")
    if row.get("s3_postprocess", ""):
        parts.append(f"post={row['s3_postprocess']}")
    return "/".join(parts)


def reconstruct_s0_s1_train(
    task_ctx: Dict[str, np.ndarray],
    cfg: PipelineConfig,
    row: Dict[str, object],
) -> Tuple[np.ndarray, np.ndarray]:
    scenario = str(row.get("scenario", ""))
    method = str(row.get("method", ""))
    ratio = float(row.get("ratio", float("nan")))
    if scenario == "S0":
        return task_ctx["X_train"], task_ctx["y_train"]
    if scenario == "S1":
        Xr, yr, _ = build_s1_resampled_train(task_ctx, cfg, method, ratio)
        return Xr, yr
    raise ValueError(f"CARS safe baseline can only reconstruct S0/S1, got {scenario}/{method}")


def choose_safe_baseline_candidate(
    candidate_rows: List[Dict[str, float]],
    cfg: PipelineConfig,
) -> Optional[Dict[str, object]]:
    safe_rows = [r for r in candidate_rows if str(r.get("scenario")) in {"S0", "S1"}]
    metric = cfg.selection_metric
    safe_rows = [r for r in safe_rows if metric in r and not pd.isna(r.get(metric))]
    if not safe_rows:
        return None
    safe_rows = sorted(
        safe_rows,
        key=lambda r: (
            float(r.get(metric, float("-inf"))),
            -float(r.get("FPR", float("inf"))),
            float(r.get("Recall", 0.0)),
        ),
        reverse=True,
    )
    return safe_rows[0]


def build_canary_set(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    safe_p_val: np.ndarray,
    model: LGBMClassifier,
    cfg: PipelineConfig,
) -> Tuple[np.ndarray, pd.DataFrame]:
    neg_idx = np.flatnonzero(y_val == 0)
    if len(neg_idx) == 0:
        return np.empty((0,), dtype=int), pd.DataFrame()

    safe_neg = safe_p_val[neg_idx].astype(np.float64, copy=False)
    score_pct = pd.Series(safe_neg).rank(method="average", pct=True).to_numpy()

    q = max(0.0, min(float(cfg.cars_canary_q), 1.0))
    min_n = max(0, min(int(cfg.cars_canary_min_n), len(neg_idx)))
    score_cut = max(min_n, int(np.ceil(len(neg_idx) * q)))
    dist_cut = score_cut
    score_order = np.argsort(-safe_neg, kind="mergesort")

    X_pos = X_train[y_train == 1]
    distance_eval = np.zeros(len(neg_idx), dtype=bool)
    dist = np.full(len(neg_idx), np.nan, dtype=np.float64)
    dist_pct = np.zeros(len(neg_idx), dtype=np.float64)
    dist_order = np.empty((0,), dtype=int)
    if len(X_pos) > 0:
        distance_max_neg = int(cfg.cars_canary_distance_max_neg)
        if distance_max_neg <= 0 or distance_max_neg >= len(neg_idx):
            distance_local = np.arange(len(neg_idx), dtype=int)
        else:
            score_seed = score_order[: min(score_cut, distance_max_neg)]
            remaining_n = max(0, distance_max_neg - len(score_seed))
            if remaining_n > 0:
                remaining = np.setdiff1d(np.arange(len(neg_idx), dtype=int), score_seed, assume_unique=False)
                rng = np.random.default_rng(int(cfg.seed) + 7919)
                sampled = rng.choice(remaining, size=min(remaining_n, len(remaining)), replace=False)
                distance_local = np.unique(np.concatenate([score_seed, sampled])).astype(int)
            else:
                distance_local = np.asarray(score_seed, dtype=int)

        weights = np.asarray(getattr(model, "feature_importances_", np.ones(X_train.shape[1])), dtype=np.float64)
        if weights.shape[0] != X_train.shape[1] or not np.isfinite(weights).all() or float(weights.sum()) <= 0:
            weights = np.ones(X_train.shape[1], dtype=np.float64)
        top_k = max(1, min(int(cfg.cars_canary_distance_top_k), X_train.shape[1]))
        top_idx = np.argsort(-weights, kind="mergesort")[:top_k]
        top_weights = weights[top_idx]
        if not np.isfinite(top_weights).all() or float(top_weights.sum()) <= 0:
            top_weights = np.ones(len(top_idx), dtype=np.float64)
        top_weights = top_weights / max(float(top_weights.sum()), 1e-12)

        X_pos_w = X_pos[:, top_idx].astype(np.float64, copy=False) * top_weights[None, :]
        X_neg_w = X_val[neg_idx[distance_local]][:, top_idx].astype(np.float64, copy=False) * top_weights[None, :]
        nn = NearestNeighbors(n_neighbors=1, metric="manhattan")
        nn.fit(X_pos_w)
        local_dist = nn.kneighbors(X_neg_w, return_distance=True)[0][:, 0].astype(np.float64, copy=False)
        dist[distance_local] = local_dist
        distance_eval[distance_local] = True
        dist_pct[distance_local] = pd.Series(-local_dist).rank(method="average", pct=True).to_numpy()
        dist_order = distance_local[np.argsort(local_dist, kind="mergesort")]

    is_score = np.zeros(len(neg_idx), dtype=bool)
    is_distance = np.zeros(len(neg_idx), dtype=bool)
    is_score[score_order[:score_cut]] = True
    if len(dist_order):
        is_distance[dist_order[: min(dist_cut, len(dist_order))]] = True
    is_canary = is_score | is_distance

    combined_rank = np.maximum(score_pct, np.nan_to_num(dist_pct, nan=0.0))
    chosen_local = np.flatnonzero(is_canary)
    max_n = int(cfg.cars_canary_max_n)
    if max_n > 0 and len(chosen_local) > max_n:
        keep_order = np.argsort(-combined_rank[chosen_local], kind="mergesort")[:max_n]
        chosen_local = chosen_local[keep_order]
        is_canary = np.zeros(len(neg_idx), dtype=bool)
        is_canary[chosen_local] = True

    canary_idx = neg_idx[chosen_local]
    table = pd.DataFrame(
        {
            "val_index": canary_idx.astype(int),
            "label": y_val[canary_idx].astype(int),
            "safe_score": safe_p_val[canary_idx].astype(float),
            "importance_weighted_distance_to_pos": dist[chosen_local].astype(float),
            "score_rank_percentile": score_pct[chosen_local].astype(float),
            "distance_rank_percentile": dist_pct[chosen_local].astype(float),
            "is_score_canary": is_score[chosen_local].astype(bool),
            "is_distance_canary": is_distance[chosen_local].astype(bool),
            "is_canary": is_canary[chosen_local].astype(bool),
            "distance_eval_candidate": distance_eval[chosen_local].astype(bool),
            "distance_top_k_features": int(cfg.cars_canary_distance_top_k),
            "distance_max_neg": int(cfg.cars_canary_distance_max_neg),
            "combined_canary_rank": combined_rank[chosen_local].astype(float),
        }
    ).sort_values(["combined_canary_rank", "safe_score"], ascending=[False, False])
    return canary_idx.astype(int), table.reset_index(drop=True)


def build_cars_canary_context(
    task_ctx: Dict[str, np.ndarray],
    cfg: PipelineConfig,
    task_candidate_rows: List[Dict[str, float]],
) -> Optional[Dict[str, object]]:
    safe_row = choose_safe_baseline_candidate(task_candidate_rows, cfg)
    if safe_row is None:
        return None
    X_safe, y_safe = reconstruct_s0_s1_train(task_ctx, cfg, safe_row)
    safe_test_metrics, preds, model = _run_downstream_with_predictions(
        X_safe,
        y_safe,
        task_ctx["X_val"],
        task_ctx["y_val"],
        task_ctx["X_test"],
        task_ctx["y_test"],
        cfg,
        cfg.seed,
        cfg.selection_threshold_rule,
        "test",
    )
    safe_p_val = np.asarray(preds["p_val"], dtype=np.float64)
    canary_idx, canary_table = build_canary_set(
        X_safe,
        y_safe,
        task_ctx["X_val"],
        task_ctx["y_val"],
        safe_p_val,
        model,
        cfg,
    )
    return {
        "safe_row": safe_row,
        "safe_candidate_key": candidate_key_from_row(safe_row),
        "safe_test_metrics": safe_test_metrics,
        "safe_p_val": safe_p_val,
        "safe_p_test": np.asarray(preds["p_test"], dtype=np.float64),
        "safe_threshold": float(preds["threshold"]),
        "val_neg_idx": np.flatnonzero(task_ctx["y_val"] == 0).astype(int),
        "canary_idx": canary_idx,
        "X_canary": task_ctx["X_val"][canary_idx] if len(canary_idx) else np.empty((0, task_ctx["X_val"].shape[1])),
        "canary_table": canary_table,
    }


def record_cars_candidate_test_prediction(
    probe_ctx: Optional[Dict[str, object]],
    candidate_meta: Dict[str, object],
    preds: Dict[str, np.ndarray | float],
) -> None:
    """Keep label-free test scores for a later sealed-policy evaluation."""
    if probe_ctx is None or "p_test" not in preds:
        return
    key = candidate_key_from_row(candidate_meta)
    store = probe_ctx.setdefault("candidate_test_predictions", {})
    if not isinstance(store, dict):
        raise TypeError("CARS candidate prediction store must be a dictionary")
    if key in store:
        raise ValueError(f"Duplicate CARS prediction key: {key}")
    store[key] = {
        "scores": np.asarray(preds["p_test"], dtype=np.float64),
        "threshold": float(preds["threshold"]),
    }


def cars_canary_inflation_metrics(
    probe_ctx: Optional[Dict[str, object]],
    aug_p_val: np.ndarray,
) -> Dict[str, float | str]:
    if not probe_ctx:
        return {}
    safe_p_val = np.asarray(probe_ctx["safe_p_val"], dtype=np.float64)
    aug_p_val = np.asarray(aug_p_val, dtype=np.float64)
    safe_thr = float(probe_ctx["safe_threshold"])
    out: Dict[str, float | str] = {
        "cars_safe_candidate_key": str(probe_ctx["safe_candidate_key"]),
    }

    val_neg_idx = np.asarray(probe_ctx.get("val_neg_idx", np.empty((0,), dtype=int)), dtype=int)
    if len(val_neg_idx):
        safe_neg = safe_p_val[val_neg_idx]
        aug_neg = aug_p_val[val_neg_idx]
        neg_shift = aug_neg - safe_neg
        neg_inflation = np.maximum(neg_shift, 0.0)
        out.update(
            {
                "cars_valneg_n": int(len(val_neg_idx)),
                "cars_valneg_score_shift_mean": float(np.mean(neg_shift)),
                "cars_valneg_score_shift_p95": float(np.percentile(neg_shift, 95)),
                "cars_valneg_inflation_mean": float(np.mean(neg_inflation)),
                "cars_valneg_inflation_p90": float(np.percentile(neg_inflation, 90)),
                "cars_valneg_inflation_p95": float(np.percentile(neg_inflation, 95)),
                "cars_valneg_inflation_max": float(np.max(neg_inflation)),
                "cars_valneg_fpr_at_safe_thr_safe": float(np.mean(safe_neg >= safe_thr)),
                "cars_valneg_fpr_at_safe_thr_aug": float(np.mean(aug_neg >= safe_thr)),
                "cars_valneg_fpr_at_safe_thr_shift": float(np.mean(aug_neg >= safe_thr) - np.mean(safe_neg >= safe_thr)),
            }
        )

    canary_idx = np.asarray(probe_ctx["canary_idx"], dtype=int)
    if len(canary_idx) == 0:
        out["cars_canary_n"] = 0
        return out

    safe_scores = safe_p_val[canary_idx]
    aug_scores = aug_p_val[canary_idx]
    shift = aug_scores - safe_scores
    inflation = np.maximum(shift, 0.0)
    out.update(
        {
        "cars_canary_n": int(len(canary_idx)),
        "cars_canary_safe_score_mean": float(np.mean(safe_scores)),
        "cars_canary_aug_score_mean": float(np.mean(aug_scores)),
        "cars_canary_score_shift_mean": float(np.mean(shift)),
        "cars_canary_score_shift_p95": float(np.percentile(shift, 95)),
        "cars_canary_inflation_mean": float(np.mean(inflation)),
        "cars_canary_inflation_p90": float(np.percentile(inflation, 90)),
        "cars_canary_inflation_p95": float(np.percentile(inflation, 95)),
        "cars_canary_inflation_max": float(np.max(inflation)),
        "cars_canary_fpr_at_safe_thr_safe": float(np.mean(safe_scores >= safe_thr)),
        "cars_canary_fpr_at_safe_thr_aug": float(np.mean(aug_scores >= safe_thr)),
        "cars_canary_fpr_at_safe_thr_shift": float(np.mean(aug_scores >= safe_thr) - np.mean(safe_scores >= safe_thr)),
        }
    )
    return out


def cars_candidate_replay_fields(
    candidate_meta: Dict[str, object],
    probe_ctx: Optional[Dict[str, object]],
    aug_p_val: np.ndarray,
    test_metrics: Optional[Dict[str, float]],
) -> Dict[str, float | str | bool]:
    out: Dict[str, float | str | bool] = {
        "cars_replay_candidate_key": candidate_key_from_row(candidate_meta),
    }
    out.update(cars_canary_inflation_metrics(probe_ctx, aug_p_val))

    if test_metrics:
        for k, v in test_metrics.items():
            out[f"cars_candidate_test_{k}"] = v
        safe_test_metrics = dict(probe_ctx.get("safe_test_metrics", {})) if probe_ctx else {}
        safe_pr = safe_test_metrics.get("PR_AUC", float("nan"))
        test_pr = test_metrics.get("PR_AUC", float("nan"))
        if not pd.isna(safe_pr) and not pd.isna(test_pr):
            delta = float(test_pr) - float(safe_pr)
            out["cars_candidate_test_safe_PR_AUC"] = float(safe_pr)
            out["cars_candidate_test_delta_vs_safe_PR_AUC"] = delta
            out["cars_candidate_test_harm_vs_safe"] = bool(delta < 0.0)
    return out


def tune_threshold(y_val: np.ndarray, p_val: np.ndarray, rule: str, recall_target: float) -> float:
    curve = threshold_metric_curve(y_val, p_val)
    if len(curve["thresholds"]) == 0:
        return 0.5

    if rule == "max_f1":
        idx = int(np.nanargmax(curve["f1"]))
        return float(curve["thresholds"][idx])

    if rule == "max_mcc":
        idx = int(np.nanargmax(curve["mcc"]))
        return float(curve["thresholds"][idx])

    if rule == "recall_target":
        ok = curve["recall"] >= float(recall_target)
        if not np.any(ok):
            return float(np.min(curve["thresholds"]))
        ok_idx = np.flatnonzero(ok)
        best_local = int(np.nanargmax(curve["precision"][ok_idx]))
        return float(curve["thresholds"][ok_idx[best_local]])

    raise ValueError(f"Unknown threshold rule: {rule}")


def threshold_for_fpr_cap(y_val: np.ndarray, p_val: np.ndarray, fpr_cap: float) -> float:
    curve = threshold_metric_curve(y_val, p_val)
    if len(curve["thresholds"]) == 0:
        return 0.5

    ok = curve["fpr"] <= float(fpr_cap) + 1e-12
    if not np.any(ok):
        return float(np.max(curve["thresholds"]))
    ok_idx = np.flatnonzero(ok)
    best_local = int(np.nanargmax(curve["recall"][ok_idx]))
    return float(curve["thresholds"][ok_idx[best_local]])


def threshold_metric_curve(y_true: np.ndarray, p: np.ndarray) -> Dict[str, np.ndarray]:
    y = np.asarray(y_true).astype(int)
    p = np.asarray(p, dtype=np.float64)
    if len(p) == 0:
        empty = np.empty((0,), dtype=np.float64)
        return {
            "thresholds": empty,
            "precision": empty,
            "recall": empty,
            "fpr": empty,
            "f1": empty,
            "mcc": empty,
        }

    order = np.argsort(-p, kind="mergesort")
    p_sorted = p[order]
    y_sorted = y[order]
    last_idx = np.r_[np.flatnonzero(p_sorted[:-1] != p_sorted[1:]), len(p_sorted) - 1]

    tp = np.cumsum(y_sorted == 1)[last_idx].astype(np.float64)
    pred_pos = (last_idx + 1).astype(np.float64)
    fp = pred_pos - tp
    pos = float((y == 1).sum())
    neg = float((y == 0).sum())
    fn = pos - tp
    tn = neg - fp

    precision = np.divide(tp, tp + fp, out=np.zeros_like(tp), where=(tp + fp) > 0)
    recall = np.divide(tp, pos, out=np.zeros_like(tp), where=pos > 0)
    fpr = np.divide(fp, neg, out=np.zeros_like(fp), where=neg > 0)
    f1 = np.divide(2.0 * precision * recall, precision + recall, out=np.zeros_like(tp), where=(precision + recall) > 0)
    denom = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = np.divide(tp * tn - fp * fn, denom, out=np.zeros_like(tp), where=denom > 0)

    return {
        "thresholds": p_sorted[last_idx],
        "precision": precision,
        "recall": recall,
        "fpr": fpr,
        "f1": f1,
        "mcc": mcc,
    }


def train_lgbm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    cfg: PipelineConfig,
    seed: int,
    sample_weight: Optional[np.ndarray] = None,
) -> LGBMClassifier:
    n_pos = int((y_train == 1).sum())
    n_neg = int((y_train == 0).sum())
    spw = float(n_neg / max(n_pos, 1))

    lgbm_extra = {}
    if cfg.lgbm_device_type == "gpu":
        lgbm_extra = {
            "device_type": "gpu",
            "gpu_platform_id": int(cfg.lgbm_gpu_platform_id),
            "gpu_device_id": int(cfg.lgbm_gpu_device_id),
        }
    else:
        lgbm_extra = {"device_type": "cpu"}

    model = LGBMClassifier(
        objective="binary",
        n_estimators=cfg.lgbm_n_estimators,
        learning_rate=cfg.lgbm_learning_rate,
        num_leaves=cfg.lgbm_num_leaves,
        subsample=cfg.lgbm_subsample,
        colsample_bytree=cfg.lgbm_colsample_bytree,
        random_state=seed,
        n_jobs=-1,
        scale_pos_weight=spw,
        verbosity=-1,
        **lgbm_extra,
    )
    if sample_weight is None:
        model.fit(X_train, y_train)
    else:
        model.fit(X_train, y_train, sample_weight=np.asarray(sample_weight, dtype=np.float64))
    return model


def run_downstream(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    cfg: PipelineConfig,
    seed: int,
) -> Dict[str, float]:
    return _run_downstream_internal(
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
        y_test,
        cfg,
        seed,
        cfg.threshold_rule,
        "test",
    )


def run_downstream_val(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    cfg: PipelineConfig,
    seed: int,
) -> Dict[str, float]:
    return _run_downstream_internal(
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
        y_test,
        cfg,
        seed,
        cfg.selection_threshold_rule,
        "val",
    )


class CVAE(nn.Module):
    def __init__(self, input_dim: int, hidden: int, latent_dim: int):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(input_dim, hidden), nn.SiLU(), nn.Linear(hidden, hidden), nn.SiLU())
        self.mu = nn.Linear(hidden, latent_dim)
        self.logvar = nn.Linear(hidden, latent_dim)
        self.dec = nn.Sequential(
            nn.Linear(latent_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, input_dim),
        )

    def encode(self, x):
        h = self.enc(x)
        return self.mu(h), self.logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_hat = self.dec(z)
        return x_hat, mu, logvar


class TabGenerator(nn.Module):
    def __init__(self, noise_dim: int, hidden: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(noise_dim, hidden),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden, hidden),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, z):
        return self.net(z)


class TabDiscriminator(nn.Module):
    def __init__(self, in_dim: int, hidden: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden, hidden),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        return self.net(x)


def train_cvae(X_pos: np.ndarray, cfg: PipelineConfig, seed: int):
    set_seed(seed)
    device = resolve_torch_device(cfg.torch_device)
    configure_torch_runtime(device)
    use_amp = bool(cfg.torch_use_amp and device.type == "cuda")
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X_pos).astype(np.float32)
    use_gpu_resident = can_preload_generator_to_gpu(Xs, cfg, device)
    if use_gpu_resident:
        X_gpu = torch.from_numpy(Xs).to(device)
        dl = None
        print(f"[INFO] cvae gpu-resident batches enabled ({Xs.nbytes / (1024 * 1024):.1f} MB)")
    else:
        ds = TensorDataset(torch.from_numpy(Xs))
        dl = DataLoader(ds, batch_size=cfg.cvae_batch_size, shuffle=True, **loader_runtime_kwargs(cfg, device))
        X_gpu = None

    model = CVAE(input_dim=Xs.shape[1], hidden=cfg.cvae_hidden, latent_dim=cfg.cvae_latent_dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.cvae_lr)
    grad_scaler = make_grad_scaler(use_amp)
    non_blocking = bool(cfg.loader_pin_memory and device.type == "cuda")

    model.train()
    for _ in range(cfg.cvae_epochs):
        if X_gpu is not None:
            batch_iter = ((xb,) for xb in iter_gpu_tensor_batches(X_gpu, cfg.cvae_batch_size, shuffle=True, drop_last=False))
        else:
            batch_iter = dl
        for (xb,) in batch_iter:
            opt.zero_grad(set_to_none=True)
            if X_gpu is None:
                xb = xb.to(device, non_blocking=non_blocking)
            with autocast_ctx(use_amp):
                x_hat, mu, logvar = model(xb)
                recon = F.mse_loss(x_hat, xb)
                kld = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
                loss = recon + cfg.cvae_beta * kld
            grad_scaler.scale(loss).backward()
            grad_scaler.step(opt)
            grad_scaler.update()

    model.eval()
    return model, scaler


@torch.no_grad()
def sample_cvae(model: CVAE, scaler: StandardScaler, n_samples: int, seed: int) -> np.ndarray:
    set_seed(seed)
    device = next(model.parameters()).device
    z = torch.randn(n_samples, model.mu.out_features, device=device)
    xs = model.dec(z).cpu().numpy()
    return scaler.inverse_transform(xs)


def train_cgan(X_pos: np.ndarray, cfg: PipelineConfig, seed: int):
    set_seed(seed)
    device = resolve_torch_device(cfg.torch_device)
    configure_torch_runtime(device)
    use_amp = bool(cfg.torch_use_amp and device.type == "cuda")

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X_pos).astype(np.float32)
    use_gpu_resident = can_preload_generator_to_gpu(Xs, cfg, device)
    if use_gpu_resident:
        X_gpu = torch.from_numpy(Xs).to(device)
        dl = None
        print(f"[INFO] cgan gpu-resident batches enabled ({Xs.nbytes / (1024 * 1024):.1f} MB)")
    else:
        ds = TensorDataset(torch.from_numpy(Xs))
        dl = DataLoader(
            ds,
            batch_size=cfg.cgan_batch_size,
            shuffle=True,
            drop_last=True,
            **loader_runtime_kwargs(cfg, device),
        )
        X_gpu = None

    G = TabGenerator(cfg.cgan_noise_dim, cfg.cgan_hidden, Xs.shape[1]).to(device)
    D = TabDiscriminator(Xs.shape[1], cfg.cgan_hidden).to(device)

    g_opt = torch.optim.Adam(G.parameters(), lr=cfg.cgan_lr, betas=(0.5, 0.999))
    d_opt = torch.optim.Adam(D.parameters(), lr=cfg.cgan_lr, betas=(0.5, 0.999))
    bce = nn.BCEWithLogitsLoss()
    g_scaler = make_grad_scaler(use_amp)
    d_scaler = make_grad_scaler(use_amp)
    non_blocking = bool(cfg.loader_pin_memory and device.type == "cuda")

    G.train()
    D.train()
    for _ in range(cfg.cgan_epochs):
        if X_gpu is not None:
            batch_iter = ((xb,) for xb in iter_gpu_tensor_batches(X_gpu, cfg.cgan_batch_size, shuffle=True, drop_last=True))
        else:
            batch_iter = dl
        for (real,) in batch_iter:
            if X_gpu is None:
                real = real.to(device, non_blocking=non_blocking)
            bsz = real.size(0)

            z = torch.randn(bsz, cfg.cgan_noise_dim, device=device)
            d_opt.zero_grad(set_to_none=True)
            with autocast_ctx(use_amp):
                fake = G(z).detach()
                d_real = D(real)
                d_fake = D(fake)
                d_loss = bce(d_real, torch.ones_like(d_real)) + bce(d_fake, torch.zeros_like(d_fake))
            d_scaler.scale(d_loss).backward()
            d_scaler.step(d_opt)
            d_scaler.update()

            z = torch.randn(bsz, cfg.cgan_noise_dim, device=device)
            g_opt.zero_grad(set_to_none=True)
            with autocast_ctx(use_amp):
                fake = G(z)
                g_loss = bce(D(fake), torch.ones((bsz, 1), device=device))
            g_scaler.scale(g_loss).backward()
            g_scaler.step(g_opt)
            g_scaler.update()

    G.eval()
    return G, scaler


@torch.no_grad()
def sample_cgan(G: TabGenerator, scaler: StandardScaler, n_samples: int, cfg: PipelineConfig, seed: int) -> np.ndarray:
    set_seed(seed)
    device = next(G.parameters()).device
    z = torch.randn(n_samples, cfg.cgan_noise_dim, device=device)
    xs = G(z).cpu().numpy()
    return scaler.inverse_transform(xs)

def real_vs_synth_auc(real_pos: np.ndarray, synth_pos: np.ndarray, cfg: PipelineConfig, seed: int) -> float:
    n = min(len(real_pos), len(synth_pos))
    if n < 30:
        return float("nan")

    rng = np.random.default_rng(seed)
    ridx = rng.choice(len(real_pos), size=n, replace=False)
    sidx = rng.choice(len(synth_pos), size=n, replace=False)

    X = np.vstack([real_pos[ridx], synth_pos[sidx]])
    y = np.concatenate([np.ones(n), np.zeros(n)]).astype(int)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, stratify=y, random_state=seed)

    imp = SimpleImputer(strategy="mean")
    Xtr = imp.fit_transform(Xtr)
    Xte = imp.transform(Xte)

    try:
        clf = train_lgbm(Xtr, ytr, cfg, seed)
        p = clf.predict_proba(Xte)[:, 1]
        return float(roc_auc_score(yte, p))
    except Exception:
        return float("nan")


def tstr_utility_pr_auc(
    X_neg_train: np.ndarray,
    X_pos_synth: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    cfg: PipelineConfig,
    seed: int,
) -> float:
    if len(X_pos_synth) < 10 or len(X_neg_train) < 100:
        return float("nan")

    n_pos = min(len(X_pos_synth), max(100, int(len(X_neg_train) / cfg.s2_train_neg_multiplier)))
    rng = np.random.default_rng(seed)
    pidx = rng.choice(len(X_pos_synth), size=n_pos, replace=len(X_pos_synth) < n_pos)

    Xtr = np.vstack([X_neg_train, X_pos_synth[pidx]])
    ytr = np.concatenate([np.zeros(len(X_neg_train), dtype=int), np.ones(n_pos, dtype=int)])

    try:
        model = train_lgbm(Xtr, ytr, cfg, seed)
        p_val = model.predict_proba(X_val)[:, 1]
        return float(average_precision_score(y_val, p_val))
    except Exception:
        return float("nan")


def build_augmented_train(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_syn_pos: np.ndarray,
    add_ratio: float,
    seed: int,
    prefer_front: bool = False,
):
    n_pos = int((y_train == 1).sum())
    n_add = int(round(n_pos * add_ratio))
    if n_add <= 0:
        return X_train, y_train, 0

    rng = np.random.default_rng(seed)
    if prefer_front and len(X_syn_pos) >= n_add:
        X_add = X_syn_pos[:n_add]
    else:
        idx = rng.choice(len(X_syn_pos), size=n_add, replace=len(X_syn_pos) < n_add)
        X_add = X_syn_pos[idx]
    y_add = np.ones(n_add, dtype=int)

    X_aug = np.vstack([X_train, X_add])
    y_aug = np.concatenate([y_train, y_add])
    return X_aug, y_aug, n_add


def random_over_resample(X_train: np.ndarray, y_train: np.ndarray, add_ratio: float, seed: int):
    n_pos = int((y_train == 1).sum())
    n_neg = int((y_train == 0).sum())
    target_pos = int(round(n_pos * (1.0 + add_ratio)))
    target_ratio = min(target_pos / max(n_neg, 1), 1.0)

    cur_ratio = n_pos / max(n_neg, 1)
    if target_ratio <= cur_ratio + 1e-12:
        return X_train, y_train, 0

    ros = RandomOverSampler(sampling_strategy=target_ratio, random_state=seed)
    X_res, y_res = ros.fit_resample(X_train, y_train)
    added = int((y_res == 1).sum() - n_pos)
    return X_res, y_res, added


def random_under_resample(X_train: np.ndarray, y_train: np.ndarray, neg_pos_ratio: float, seed: int):
    n_pos = int((y_train == 1).sum())
    n_neg = int((y_train == 0).sum())
    target_neg = int(round(n_pos * neg_pos_ratio))
    if target_neg >= n_neg:
        return X_train, y_train, 0

    sampling_strategy = n_pos / max(target_neg, 1)
    rus = RandomUnderSampler(sampling_strategy=sampling_strategy, random_state=seed)
    X_res, y_res = rus.fit_resample(X_train, y_train)
    removed = int(n_neg - (y_res == 0).sum())
    return X_res, y_res, removed


def smote_resample(
    X_train: np.ndarray,
    y_train: np.ndarray,
    add_ratio: float,
    seed: int,
    k_neighbors: int,
):
    n_pos = int((y_train == 1).sum())
    n_neg = int((y_train == 0).sum())
    if n_pos < 2:
        return X_train, y_train, 0

    target_pos = int(round(n_pos * (1.0 + add_ratio)))
    target_ratio = min(target_pos / max(n_neg, 1), 1.0)
    cur_ratio = n_pos / max(n_neg, 1)
    if target_ratio <= cur_ratio + 1e-12:
        return X_train, y_train, 0

    smote = SMOTE(
        sampling_strategy=target_ratio,
        random_state=seed,
        k_neighbors=max(1, min(int(k_neighbors), n_pos - 1)),
    )
    X_res, y_res = smote.fit_resample(X_train, y_train)
    added = int((y_res == 1).sum() - n_pos)
    return X_res, y_res, added


def sample_neg_subset(X_neg: np.ndarray, n_keep: int, seed: int) -> np.ndarray:
    if n_keep >= len(X_neg):
        return X_neg
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(X_neg), size=n_keep, replace=False)
    return X_neg[idx]


def train_positive_rate(y_train: np.ndarray) -> float:
    return float((y_train == 1).sum() / max(len(y_train), 1))


def resolve_s3_train_modes(task_ctx: Dict[str, np.ndarray], cfg: PipelineConfig) -> Tuple[str, ...]:
    if not cfg.s3_auto_mode_by_pos_rate:
        return tuple(cfg.strict_ddpm_train_modes)

    pos_rate = train_positive_rate(task_ctx["y_train"])
    preferred = "positive_only" if pos_rate <= cfg.s3_positive_only_max_rate else "conditional_mixed"
    available = tuple(x for x in cfg.strict_ddpm_train_modes if x == preferred)
    if available:
        return available
    return tuple(cfg.strict_ddpm_train_modes)


def rank_synth_by_knn_to_pos(
    X_pos: np.ndarray,
    X_syn: np.ndarray,
    k_neighbors: int,
) -> Tuple[np.ndarray, np.ndarray]:
    if len(X_syn) == 0 or len(X_pos) == 0:
        return X_syn, np.empty((0,), dtype=np.float32)

    k = max(1, min(int(k_neighbors), len(X_pos)))
    nn = NearestNeighbors(n_neighbors=k)
    nn.fit(X_pos)
    dists, _ = nn.kneighbors(X_syn)
    scores = dists.mean(axis=1).astype(np.float32, copy=False)
    order = np.argsort(scores)
    return X_syn[order], scores[order]


def rank_synth_by_disc_realness(
    X_pos: np.ndarray,
    X_syn: np.ndarray,
    cfg: PipelineConfig,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    if len(X_syn) < 30 or len(X_pos) < 30:
        return X_syn, np.full(len(X_syn), np.nan, dtype=np.float32)

    rng = np.random.default_rng(seed)
    syn_perm = rng.permutation(len(X_syn))
    syn_mid = max(1, len(syn_perm) // 2)
    syn_folds = [syn_perm[:syn_mid], syn_perm[syn_mid:]]
    scores = np.full(len(X_syn), np.nan, dtype=np.float32)

    for i, hold_idx in enumerate(syn_folds):
        if hold_idx.size == 0:
            continue
        train_idx = np.concatenate([syn_folds[j] for j in range(len(syn_folds)) if j != i])
        if train_idx.size == 0:
            train_idx = hold_idx

        n_real = min(len(X_pos), max(64, len(train_idx)))
        ridx = rng.choice(len(X_pos), size=n_real, replace=len(X_pos) < n_real)
        Xtr = np.vstack([X_pos[ridx], X_syn[train_idx]])
        ytr = np.concatenate([np.ones(len(ridx), dtype=int), np.zeros(len(train_idx), dtype=int)])

        imp = SimpleImputer(strategy="mean")
        Xtr_i = imp.fit_transform(Xtr)
        Xho_i = imp.transform(X_syn[hold_idx])

        try:
            clf = train_lgbm(Xtr_i, ytr, cfg, seed + i + 1)
            scores[hold_idx] = clf.predict_proba(Xho_i)[:, 1].astype(np.float32, copy=False)
        except Exception:
            scores[hold_idx] = 0.5

    order = np.argsort(-scores, kind="stable")
    return X_syn[order], scores[order]


def rank_synth_by_hybrid_realness(
    X_pos: np.ndarray,
    X_syn: np.ndarray,
    cfg: PipelineConfig,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    if len(X_syn) == 0:
        return X_syn, np.empty((0,), dtype=np.float32)

    _, knn_scores = rank_synth_by_knn_to_pos(X_pos, X_syn, cfg.s3_synth_filter_k)
    _, disc_scores = rank_synth_by_disc_realness(X_pos, X_syn, cfg, seed)

    knn_rank = np.argsort(np.argsort(knn_scores))
    knn_realness = 1.0 - (knn_rank.astype(np.float32) / max(len(knn_scores) - 1, 1))
    disc_realness = np.nan_to_num(disc_scores, nan=0.5, posinf=0.5, neginf=0.5).astype(np.float32, copy=False)
    hybrid = 0.5 * knn_realness + 0.5 * disc_realness
    order = np.argsort(-hybrid, kind="stable")
    return X_syn[order], hybrid[order]


_IMPORTANCE_WEIGHT_CACHE: Dict[Tuple[str, int, int, int], np.ndarray] = {}


def _resolve_optional_path(path_text: str) -> Optional[Path]:
    text = str(path_text).strip()
    if not text:
        return None
    path = Path(text)
    if not path.is_absolute():
        path = ROOT / path
    return path


def _uniform_feature_weights(n_features: int) -> np.ndarray:
    if n_features <= 0:
        return np.empty((0,), dtype=np.float32)
    return np.full(n_features, 1.0 / n_features, dtype=np.float32)


def load_importance_weights_for_s3(
    cfg: PipelineConfig,
    task_id: Optional[int],
    seed: int,
    n_features: int,
) -> np.ndarray:
    path = _resolve_optional_path(cfg.s3_ig_importance_path)
    if path is None or not path.exists():
        return _uniform_feature_weights(n_features)

    cache_key = (str(path.resolve()), int(task_id) if task_id is not None else -1, int(seed), int(n_features))
    if cache_key in _IMPORTANCE_WEIGHT_CACHE:
        return _IMPORTANCE_WEIGHT_CACHE[cache_key].copy()

    df = pd.read_csv(path)
    if "task_id" in df.columns and task_id is not None:
        task_df = df[df["task_id"].astype(str) == str(task_id)].copy()
        if not task_df.empty:
            df = task_df
    if "seed" in df.columns:
        seed_df = df[df["seed"].astype(str) == str(seed)].copy()
        if not seed_df.empty:
            df = seed_df

    if "feature_index" in df.columns:
        feature_idx = pd.to_numeric(df["feature_index"], errors="coerce")
    elif "feature_name" in df.columns:
        feature_idx = pd.to_numeric(df["feature_name"].astype(str).str.extract(r"(\d+)")[0], errors="coerce")
    else:
        feature_idx = pd.Series(np.arange(len(df)), index=df.index)

    weight_col = None
    for col in ["importance_weight_normalized", "importance_clipped", "importance_raw"]:
        if col in df.columns:
            weight_col = col
            break
    if weight_col is None:
        return _uniform_feature_weights(n_features)

    work = pd.DataFrame(
        {
            "feature_index": feature_idx,
            "weight": pd.to_numeric(df[weight_col], errors="coerce"),
        }
    ).dropna()
    work = work[(work["feature_index"] >= 0) & (work["feature_index"] < n_features)]
    if work.empty:
        return _uniform_feature_weights(n_features)

    weights = np.zeros(n_features, dtype=np.float32)
    grouped = work.groupby("feature_index")["weight"].mean()
    for idx, val in grouped.items():
        weights[int(idx)] = max(float(val), 0.0)

    total = float(weights.sum())
    if total <= 0.0:
        weights = _uniform_feature_weights(n_features)
    else:
        weights = (weights / total).astype(np.float32, copy=False)

    _IMPORTANCE_WEIGHT_CACHE[cache_key] = weights.copy()
    return weights


def rank_synth_by_weighted_knn_to_pos(
    X_pos: np.ndarray,
    X_syn: np.ndarray,
    weights: np.ndarray,
    k_neighbors: int,
    top_k: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    if len(X_syn) == 0 or len(X_pos) == 0:
        return X_syn, np.empty((0,), dtype=np.float32)

    w = np.asarray(weights, dtype=np.float32).copy()
    if w.shape[0] != X_syn.shape[1]:
        w = _uniform_feature_weights(X_syn.shape[1])
    if top_k is not None and 0 < int(top_k) < len(w):
        keep = np.argsort(-w, kind="stable")[: int(top_k)]
        masked = np.zeros_like(w)
        masked[keep] = w[keep]
        w = masked
    total = float(w.sum())
    if total <= 0.0:
        w = _uniform_feature_weights(X_syn.shape[1])
    else:
        w = (w / total).astype(np.float32, copy=False)

    scale = np.sqrt(w).astype(np.float32, copy=False)
    k = max(1, min(int(k_neighbors), len(X_pos)))
    nn = NearestNeighbors(n_neighbors=k)
    nn.fit(X_pos * scale[None, :])
    dists, _ = nn.kneighbors(X_syn * scale[None, :])
    scores = dists.mean(axis=1).astype(np.float32, copy=False)
    order = np.argsort(scores, kind="stable")
    return X_syn[order], scores[order]


def rank_synth_randomly(X_syn: np.ndarray, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    if len(X_syn) == 0:
        return X_syn, np.empty((0,), dtype=np.float32)
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(X_syn))
    scores = np.arange(len(X_syn), dtype=np.float32)
    return X_syn[order], scores


def maybe_filter_s3_synth(
    X_pos: np.ndarray,
    X_syn: np.ndarray,
    cfg: PipelineConfig,
    seed: int,
    task_id: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    if cfg.s3_synth_filter == "none":
        return X_syn, np.full(len(X_syn), np.nan, dtype=np.float32)
    if cfg.s3_synth_filter == "knn_pos":
        return rank_synth_by_knn_to_pos(X_pos, X_syn, cfg.s3_synth_filter_k)
    if cfg.s3_synth_filter == "knn_all_features":
        weights = _uniform_feature_weights(X_syn.shape[1])
        return rank_synth_by_weighted_knn_to_pos(X_pos, X_syn, weights, cfg.s3_synth_filter_k)
    if cfg.s3_synth_filter == "ig_topk":
        weights = load_importance_weights_for_s3(cfg, task_id, seed, X_syn.shape[1])
        return rank_synth_by_weighted_knn_to_pos(
            X_pos,
            X_syn,
            weights,
            cfg.s3_synth_filter_k,
            top_k=cfg.s3_ig_top_k,
        )
    if cfg.s3_synth_filter == "ig_weighted":
        weights = load_importance_weights_for_s3(cfg, task_id, seed, X_syn.shape[1])
        return rank_synth_by_weighted_knn_to_pos(X_pos, X_syn, weights, cfg.s3_synth_filter_k)
    if cfg.s3_synth_filter == "random_keep":
        return rank_synth_randomly(X_syn, seed)
    if cfg.s3_synth_filter == "disc_pos":
        return rank_synth_by_disc_realness(X_pos, X_syn, cfg, seed)
    if cfg.s3_synth_filter == "hybrid_pos":
        return rank_synth_by_hybrid_realness(X_pos, X_syn, cfg, seed)
    raise ValueError(f"Unknown s3_synth_filter: {cfg.s3_synth_filter}")


def detect_spike_binary_mask(real_mat: np.ndarray) -> np.ndarray:
    masks = np.zeros(real_mat.shape[1], dtype=bool)
    for j in range(real_mat.shape[1]):
        col = real_mat[:, j]
        zero_rate = float(np.mean(np.isclose(col, 0.0)))
        one_rate = float(np.mean(np.isclose(col, 1.0)))
        binary_mass = float(np.mean(np.isclose(col, 0.0) | np.isclose(col, 1.0)))
        uniq = np.unique(np.round(col.astype(np.float64), 6))
        is_binary_like = binary_mass >= 0.98 and len(uniq) <= 4
        is_spike_like = (zero_rate >= 0.20 or one_rate >= 0.20) and len(uniq) <= 32
        masks[j] = bool(is_binary_like or is_spike_like)
    return masks


def top_corr_pairs(real_mat: np.ndarray, top_k_pairs: int) -> List[Tuple[int, int]]:
    if real_mat.shape[1] < 2:
        return []
    corr = np.corrcoef(real_mat, rowvar=False)
    corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
    abs_corr = np.abs(corr)
    np.fill_diagonal(abs_corr, 0.0)
    tri_i, tri_j = np.triu_indices_from(abs_corr, k=1)
    scores = abs_corr[tri_i, tri_j]
    if scores.size == 0:
        return []
    order = np.argsort(scores)[::-1]
    top_n = int(max(1, min(top_k_pairs, len(order))))
    return [(int(tri_i[k]), int(tri_j[k])) for k in order[:top_n]]


def top_corr_feature_group(real_mat: np.ndarray, top_k_pairs: int, top_features: int) -> List[int]:
    pairs = top_corr_pairs(real_mat, top_k_pairs)
    feat_idx: List[int] = []
    seen = set()
    for i, j in pairs:
        for idx in (i, j):
            if idx not in seen:
                feat_idx.append(idx)
                seen.add(idx)
            if len(feat_idx) >= int(max(2, top_features)):
                return feat_idx
    return feat_idx


def build_tabdiff_controls(
    X_pos: np.ndarray,
    X_pos_scaled: np.ndarray,
    cfg: PipelineConfig,
) -> Dict[str, np.ndarray]:
    d = X_pos.shape[1]
    spike_mask = detect_spike_binary_mask(X_pos)
    corr_group_idx = top_corr_feature_group(
        X_pos,
        cfg.s3_postprocess_corr_top_k,
        cfg.s3_postprocess_group_top_features,
    )

    feature_noise_scale = np.ones(d, dtype=np.float32)
    feature_noise_scale[spike_mask] = np.minimum(
        feature_noise_scale[spike_mask],
        float(cfg.ddpm_featurewise_spike_scale),
    )
    if len(corr_group_idx) > 0:
        feature_noise_scale[corr_group_idx] = np.minimum(
            feature_noise_scale[corr_group_idx],
            float(cfg.ddpm_featurewise_corr_scale),
        )

    spike_idx = np.flatnonzero(spike_mask).astype(np.int64)
    if spike_idx.size > 0:
        spike_real = X_pos_scaled[:, spike_idx]
        spike_low = np.quantile(spike_real, cfg.s3_postprocess_low_q, axis=0).astype(np.float32)
        spike_high = np.quantile(spike_real, cfg.s3_postprocess_high_q, axis=0).astype(np.float32)
        spike_median = np.median(spike_real, axis=0).astype(np.float32)
    else:
        spike_low = np.empty((0,), dtype=np.float32)
        spike_high = np.empty((0,), dtype=np.float32)
        spike_median = np.empty((0,), dtype=np.float32)

    if len(corr_group_idx) >= 2:
        corr_real = X_pos_scaled[:, corr_group_idx].astype(np.float32, copy=False)
        corr_rank = int(max(1, min(cfg.ddpm_sampler_corr_rank, corr_real.shape[1], corr_real.shape[0])))
        pca = PCA(n_components=corr_rank, random_state=cfg.seed)
        pca.fit(corr_real)
        corr_group_mean = pca.mean_.astype(np.float32, copy=False)
        corr_group_components = pca.components_.astype(np.float32, copy=False)
    else:
        corr_group_mean = np.empty((0,), dtype=np.float32)
        corr_group_components = np.empty((0, 0), dtype=np.float32)

    return {
        "feature_noise_scale": feature_noise_scale,
        "spike_idx": spike_idx,
        "spike_low": spike_low,
        "spike_high": spike_high,
        "spike_median": spike_median,
        "corr_group_idx": np.asarray(corr_group_idx, dtype=np.int64),
        "corr_group_mean": corr_group_mean,
        "corr_group_components": corr_group_components,
    }


def postprocess_s3_synth(
    X_pos: np.ndarray,
    X_syn: np.ndarray,
    cfg: PipelineConfig,
) -> np.ndarray:
    if len(X_syn) == 0 or len(X_pos) == 0 or cfg.s3_postprocess == "none":
        return X_syn

    def rank_match_to_real(real_col: np.ndarray, syn_col: np.ndarray) -> np.ndarray:
        if len(real_col) == 0 or len(syn_col) == 0:
            return syn_col.astype(np.float32, copy=False)
        syn_order = np.argsort(syn_col, kind="stable")
        real_sorted = np.sort(real_col.astype(np.float32, copy=False))
        quantiles = (np.arange(len(syn_col), dtype=np.float32) + 0.5) / max(len(syn_col), 1)
        real_idx = np.clip((quantiles * len(real_sorted)).astype(int), 0, len(real_sorted) - 1)
        mapped_sorted = real_sorted[real_idx]
        out = np.empty(len(syn_col), dtype=np.float32)
        out[syn_order] = mapped_sorted
        return out

    def blend_pair_to_real_neighbors(real_mat: np.ndarray, syn_mat: np.ndarray, i: int, j: int, blend_alpha: float) -> None:
        real_pair = real_mat[:, [i, j]].astype(np.float32, copy=False)
        syn_pair = syn_mat[:, [i, j]].astype(np.float32, copy=False)
        mu = real_pair.mean(axis=0, keepdims=True)
        sigma = real_pair.std(axis=0, keepdims=True)
        sigma = np.where(sigma < 1e-6, 1.0, sigma)
        real_z = (real_pair - mu) / sigma
        syn_z = (syn_pair - mu) / sigma
        nn = NearestNeighbors(n_neighbors=1)
        nn.fit(real_z)
        idx = nn.kneighbors(syn_z, return_distance=False).ravel()
        target = real_pair[idx]
        syn_mat[:, [i, j]] = ((1.0 - blend_alpha) * syn_pair + blend_alpha * target).astype(np.float32, copy=False)

    def blend_group_to_real_neighbors(real_mat: np.ndarray, syn_mat: np.ndarray, feat_idx: List[int], blend_alpha: float) -> None:
        if len(feat_idx) == 0:
            return
        real_group = real_mat[:, feat_idx].astype(np.float32, copy=False)
        syn_group = syn_mat[:, feat_idx].astype(np.float32, copy=False)
        mu = real_group.mean(axis=0, keepdims=True)
        sigma = real_group.std(axis=0, keepdims=True)
        sigma = np.where(sigma < 1e-6, 1.0, sigma)
        real_z = (real_group - mu) / sigma
        syn_z = (syn_group - mu) / sigma
        nn = NearestNeighbors(n_neighbors=1)
        nn.fit(real_z)
        idx = nn.kneighbors(syn_z, return_distance=False).ravel()
        target = real_group[idx]
        syn_mat[:, feat_idx] = ((1.0 - blend_alpha) * syn_group + blend_alpha * target).astype(np.float32, copy=False)

    low_q = float(np.clip(cfg.s3_postprocess_low_q, 0.0, 1.0))
    high_q = float(np.clip(cfg.s3_postprocess_high_q, 0.0, 1.0))
    if low_q > high_q:
        low_q, high_q = high_q, low_q

    q_low = np.quantile(X_pos, low_q, axis=0)
    q_high = np.quantile(X_pos, high_q, axis=0)
    pos_median = np.median(X_pos, axis=0)

    X_out = X_syn.astype(np.float32, copy=True)
    alpha = float(np.clip(cfg.s3_postprocess_anchor_alpha, 0.0, 1.0))

    if cfg.s3_postprocess == "spike_match_pos":
        spike_mask = detect_spike_binary_mask(X_pos)
        for j in np.flatnonzero(spike_mask):
            X_out[:, j] = rank_match_to_real(X_pos[:, j], X_out[:, j])
        return X_out.astype(np.float32, copy=False)

    if cfg.s3_postprocess == "featurewise_quantile_pos":
        for j in range(X_out.shape[1]):
            X_out[:, j] = rank_match_to_real(X_pos[:, j], X_out[:, j])
        return X_out.astype(np.float32, copy=False)

    if cfg.s3_postprocess == "corr_pair_anchor_pos":
        pairs = top_corr_pairs(X_pos, cfg.s3_postprocess_corr_top_k)
        for i, j in pairs:
            blend_pair_to_real_neighbors(X_pos, X_out, i, j, alpha)
        return X_out.astype(np.float32, copy=False)

    if cfg.s3_postprocess == "corr_group_anchor_pos":
        pairs = top_corr_pairs(X_pos, cfg.s3_postprocess_corr_top_k)
        feat_idx: List[int] = []
        seen = set()
        for i, j in pairs:
            for idx in (i, j):
                if idx not in seen:
                    feat_idx.append(idx)
                    seen.add(idx)
                if len(feat_idx) >= int(max(2, cfg.s3_postprocess_group_top_features)):
                    break
            if len(feat_idx) >= int(max(2, cfg.s3_postprocess_group_top_features)):
                break
        blend_group_to_real_neighbors(X_pos, X_out, feat_idx, alpha)
        return X_out.astype(np.float32, copy=False)

    if cfg.s3_postprocess == "latent_pca_pos":
        var_keep = float(np.clip(cfg.s3_postprocess_pca_var, 0.5, 0.999))
        pca = PCA(n_components=var_keep, svd_solver="full", random_state=cfg.seed)
        pca.fit(X_pos.astype(np.float32, copy=False))
        recon = pca.inverse_transform(pca.transform(X_out)).astype(np.float32, copy=False)
        X_out = ((1.0 - alpha) * X_out + alpha * recon).astype(np.float32, copy=False)
        return X_out.astype(np.float32, copy=False)

    if cfg.s3_postprocess in {"clip_pos_q", "clip_anchor_pos_q"}:
        X_out = np.clip(X_out, q_low, q_high)

    if cfg.s3_postprocess in {"anchor_pos_q", "clip_anchor_pos_q"} and alpha > 0.0:
        X_out = (1.0 - alpha) * X_out + alpha * pos_median

    if cfg.s3_postprocess == "clip_anchor_pos_q":
        X_out = np.clip(X_out, q_low, q_high)

    return X_out.astype(np.float32, copy=False)


def prepare_s3_training_arrays(
    task_ctx: Dict[str, np.ndarray],
    cfg: PipelineConfig,
    train_mode: str,
) -> Dict[str, np.ndarray]:
    X_train = task_ctx["X_train"]
    y_train = task_ctx["y_train"]
    X_pos = X_train[y_train == 1]
    X_neg = X_train[y_train == 0]

    if train_mode == "positive_only":
        X_neg_sub = np.empty((0, X_train.shape[1]), dtype=X_train.dtype)
        X_ddpm_train = X_pos
        y_ddpm_train = np.ones(len(X_pos), dtype=int)
    elif train_mode == "balanced_conditional":
        neg_keep = int(min(len(X_neg), len(X_pos), cfg.ddpm_max_neg))
        X_neg_sub = sample_neg_subset(X_neg, neg_keep, cfg.seed)
        X_ddpm_train = np.vstack([X_pos, X_neg_sub])
        y_ddpm_train = np.concatenate([np.ones(len(X_pos), dtype=int), np.zeros(len(X_neg_sub), dtype=int)])
    else:
        neg_keep = int(min(len(X_neg), cfg.ddpm_max_neg, max(len(X_pos), len(X_pos) * cfg.ddpm_neg_multiplier)))
        X_neg_sub = sample_neg_subset(X_neg, neg_keep, cfg.seed)
        X_ddpm_train = np.vstack([X_pos, X_neg_sub])
        y_ddpm_train = np.concatenate([np.ones(len(X_pos), dtype=int), np.zeros(len(X_neg_sub), dtype=int)])

    return {
        "X_train": X_train,
        "y_train": y_train,
        "X_pos": X_pos,
        "X_neg": X_neg,
        "X_neg_sub": X_neg_sub,
        "X_ddpm_train": X_ddpm_train,
        "y_ddpm_train": y_ddpm_train,
    }


def build_s3_synth_pool(
    task_ctx: Dict[str, np.ndarray],
    cfg: PipelineConfig,
    train_mode: str,
) -> Dict[str, object]:
    arrays = prepare_s3_training_arrays(task_ctx, cfg, train_mode)
    X_train = arrays["X_train"]
    y_train = arrays["y_train"]
    X_ddpm_train = arrays["X_ddpm_train"]
    y_ddpm_train = arrays["y_ddpm_train"]

    if cfg.ddpm_scale == "quantile":
        ddpm_scaler = QuantileTransformer(
            n_quantiles=min(2000, X_ddpm_train.shape[0]),
            output_distribution="normal",
            random_state=cfg.seed,
        )
    else:
        ddpm_scaler = StandardScaler()
    X_ddpm_scaled = ddpm_scaler.fit_transform(X_ddpm_train).astype(np.float32)
    X_pos_scaled = ddpm_scaler.transform(arrays["X_pos"]).astype(np.float32)

    ddpm_cfg = TorchDDPMConfig(
        T=cfg.ddpm_T,
        beta_start=cfg.ddpm_beta_start,
        beta_end=cfg.ddpm_beta_end,
        time_dim=cfg.ddpm_time_dim,
        class_dim=cfg.ddpm_class_dim,
        hidden_dim=cfg.ddpm_hidden_dim,
        n_layers=cfg.ddpm_n_layers,
        dropout=cfg.ddpm_dropout,
        epochs=cfg.ddpm_epochs,
        batch_size=cfg.ddpm_batch_size,
        lr=cfg.ddpm_lr,
        device=cfg.ddpm_device,
        use_amp=cfg.torch_use_amp,
        num_workers=cfg.loader_num_workers,
        pin_memory=cfg.loader_pin_memory,
        gpu_resident_batches=cfg.generator_gpu_resident,
        gpu_resident_max_mb=cfg.generator_gpu_resident_max_mb,
    )
    ddpm = TorchTabDDPMConditionalAugmenter(seed=cfg.seed, cfg=ddpm_cfg)
    if cfg.ddpm_core_variant == "tabdiff_min":
        controls = build_tabdiff_controls(arrays["X_pos"], X_pos_scaled, cfg)
        ddpm.set_tabdiff_controls(
            controls["feature_noise_scale"],
            sampler_correction_every=cfg.ddpm_sampler_correction_every,
            sampler_correction_alpha=cfg.ddpm_sampler_correction_alpha,
            spike_idx=controls["spike_idx"],
            spike_low=controls["spike_low"],
            spike_high=controls["spike_high"],
            spike_median=controls["spike_median"],
            corr_group_idx=controls["corr_group_idx"],
            corr_group_mean=controls["corr_group_mean"],
            corr_group_components=controls["corr_group_components"],
        )
    ddpm.fit(X_ddpm_scaled, y_ddpm_train)

    max_ratio = max(cfg.gen_ratios)
    max_n_add = int(round((y_train == 1).sum() * max_ratio))
    synth_pool_mult = max(1.0, float(cfg.s3_synth_pool_multiplier))
    n_sample = int(round(max_n_add * synth_pool_mult))
    X_syn_sc = ddpm.sample(n_sample, y_label=1)
    X_syn = ddpm_scaler.inverse_transform(X_syn_sc)

    q_low = np.quantile(X_train, 0.01, axis=0)
    q_high = np.quantile(X_train, 0.99, axis=0)
    X_syn = np.clip(X_syn, q_low, q_high)
    X_syn = postprocess_s3_synth(arrays["X_pos"], X_syn, cfg)

    return {
        **arrays,
        "X_syn_raw": X_syn,
        "max_n_add": int(max_n_add),
    }


def _sort_priority_for_metric(selection_metric: str) -> List[str]:
    priority = [selection_metric]
    for metric in ["MCC", "PR_AUC", "ROC_AUC", "Recall", "Precision"]:
        if metric not in priority:
            priority.append(metric)
    return priority


def select_candidate_winners(df: pd.DataFrame, group_cols: List[str], selection_metric: str) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    priority = _sort_priority_for_metric(selection_metric)
    available = [c for c in priority if c in df.columns]
    ranked = df.sort_values(available, ascending=[False] * len(available))
    return ranked.groupby(group_cols, as_index=False, dropna=False).head(1).reset_index(drop=True)


def split_candidate_row(row: Dict[str, float], selection_metric: str) -> Tuple[Dict[str, float], Dict[str, float]]:
    metric_cols = set(SUMMARY_METRICS + OPERATING_METRICS)
    metric_cols.update(k for k in row.keys() if str(k).startswith("Recall_at_FPR"))
    meta = {k: v for k, v in row.items() if k not in metric_cols}
    val_metrics = {f"val_{k}": v for k, v in row.items() if k in metric_cols}
    meta["selection_metric_name"] = selection_metric
    meta["selection_metric_value"] = row.get(selection_metric, float("nan"))
    return meta, val_metrics


def summarize_paper_results(df: pd.DataFrame, run_dir: Path) -> None:
    if df.empty:
        return

    by_scenario = df.groupby(["scenario", "method"], dropna=False)[SUMMARY_METRICS].mean().reset_index()
    by_scenario.to_csv(run_dir / "paper_summary_by_scenario.csv", index=False)

    by_task = df.groupby(["task_id", "scenario", "method"], dropna=False)[SUMMARY_METRICS].mean().reset_index()
    by_task.to_csv(run_dir / "paper_summary_by_task.csv", index=False)

    mean_std = (
        df.groupby(["scenario", "method"], dropna=False)[SUMMARY_METRICS]
        .agg(["mean", "std"])
        .reset_index()
    )
    mean_std.columns = [
        "_".join([str(x) for x in col if str(x)])
        if isinstance(col, tuple)
        else str(col)
        for col in mean_std.columns
    ]
    mean_std.to_csv(run_dir / "paper_summary_mean_std.csv", index=False)


def write_strict_partial_results(
    run_dir: Path,
    candidate_rows: List[Dict[str, float]],
    winner_rows: List[Dict[str, float]],
    final_rows: List[Dict[str, float]],
) -> None:
    pd.DataFrame(candidate_rows).to_csv(run_dir / "candidate_val_results.csv", index=False)
    replay_rows = [r for r in candidate_rows if r.get("cars_replay_candidate_key")]
    if replay_rows:
        pd.DataFrame(replay_rows).to_csv(run_dir / "cars_candidate_replay.csv", index=False)
    pd.DataFrame(winner_rows).to_csv(run_dir / "selected_winners.csv", index=False)
    final_df = pd.DataFrame(final_rows)
    final_df.to_csv(run_dir / "final_test_results.csv", index=False)
    if not final_df.empty:
        summarize_paper_results(final_df, run_dir)


def write_cars_candidate_test_predictions(
    run_dir: Path,
    task_id: int,
    seed: int,
    task_ctx: Dict[str, np.ndarray],
    probe_ctx: Optional[Dict[str, object]],
) -> None:
    """Write score arrays without outcome labels for two-phase policy audits."""
    if probe_ctx is None:
        raise ValueError("Cannot save CARS predictions without a canary context")
    store = probe_ctx.get("candidate_test_predictions", {})
    if not isinstance(store, dict) or not store:
        raise ValueError("No CARS candidate test predictions were recorded")

    keys = sorted(str(key) for key in store)
    candidate_scores = np.stack(
        [np.asarray(store[key]["scores"], dtype=np.float64) for key in keys]
    )
    candidate_thresholds = np.asarray(
        [float(store[key]["threshold"]) for key in keys], dtype=np.float64
    )
    safe_scores = np.asarray(probe_ctx["safe_p_test"], dtype=np.float64)
    if candidate_scores.shape[1] != len(safe_scores):
        raise ValueError(
            "Candidate/safe test prediction length mismatch: "
            f"{candidate_scores.shape[1]} != {len(safe_scores)}"
        )
    if candidate_scores.shape[1] != len(task_ctx["X_test"]):
        raise ValueError("Prediction length does not match the task test rows")

    out_dir = run_dir / "cars_candidate_predictions"
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"task_{int(task_id)}_seed_{int(seed)}"
    pd.DataFrame(
        {
            "prediction_row": np.arange(len(keys), dtype=int),
            "cars_replay_candidate_key": keys,
        }
    ).to_csv(out_dir / f"{stem}_index.csv", index=False)
    np.savez_compressed(
        out_dir / f"{stem}_scores.npz",
        candidate_test_scores=candidate_scores,
        candidate_thresholds=candidate_thresholds,
        safe_test_scores=safe_scores,
        safe_threshold=np.asarray([float(probe_ctx["safe_threshold"])], dtype=np.float64),
        test_row_count=np.asarray([len(safe_scores)], dtype=np.int64),
    )


def build_s1_resampled_train(
    task_ctx: Dict[str, np.ndarray],
    cfg: PipelineConfig,
    method: str,
    ratio: float,
) -> Tuple[np.ndarray, np.ndarray, int]:
    if method == "random_over":
        return random_over_resample(task_ctx["X_train"], task_ctx["y_train"], ratio, cfg.seed)
    if method == "smote":
        return smote_resample(
            task_ctx["X_train"],
            task_ctx["y_train"],
            ratio,
            cfg.seed,
            cfg.s1_smote_k_neighbors,
        )
    if method.startswith("random_under_") and method.endswith("to1"):
        neg_pos_ratio = float(method.split("_")[-1].replace("to1", ""))
        Xr, yr, removed = random_under_resample(task_ctx["X_train"], task_ctx["y_train"], neg_pos_ratio, cfg.seed)
        return Xr, yr, -int(removed)
    raise ValueError(f"Unknown S1 method: {method}")


def build_s3_synth_cache(
    task_ctx: Dict[str, np.ndarray],
    cfg: PipelineConfig,
    train_mode: str,
) -> Dict[str, object]:
    pool = build_s3_synth_pool(task_ctx, cfg, train_mode)
    X_pos = pool["X_pos"]
    X_neg = pool["X_neg"]
    X_syn = pool["X_syn_raw"]
    max_n_add = int(pool["max_n_add"])

    filter_scores = np.full(len(X_syn), np.nan, dtype=np.float32)
    if cfg.s3_synth_filter != "none":
        X_syn, filter_scores = maybe_filter_s3_synth(
            X_pos,
            X_syn,
            cfg,
            cfg.seed,
            task_id=task_ctx.get("task_id"),
        )

    eval_n = max(1, min(len(X_syn), max_n_add))
    X_syn_eval = X_syn[:eval_n]

    return {
        "X_syn": X_syn,
        "X_pos": X_pos,
        "X_neg": X_neg,
        "filter_scores": filter_scores,
        "ddpm_neg_subset": int(len(pool["X_neg_sub"])),
        "real_vs_synth_auc": float(real_vs_synth_auc(X_pos, X_syn_eval, cfg, cfg.seed)),
        "tstr_pr_auc": float(tstr_utility_pr_auc(X_neg, X_syn_eval, task_ctx["X_val"], task_ctx["y_val"], cfg, cfg.seed)),
        "s3_synth_filter": cfg.s3_synth_filter,
        "s3_postprocess": cfg.s3_postprocess,
        "s3_synth_pool_size": int(len(X_syn)),
        "s3_filter_score_mean": float(np.nanmean(filter_scores)) if len(filter_scores) else float("nan"),
    }


def s3_filter_uses_keep_rates(filter_name: str) -> bool:
    return filter_name in {"knn_all_features", "ig_topk", "ig_weighted", "random_keep"}


def s3_candidate_keep_rates(cfg: PipelineConfig) -> Tuple[float, ...]:
    if not s3_filter_uses_keep_rates(cfg.s3_synth_filter):
        return (1.0,)
    rates = tuple(float(x) for x in cfg.s3_ig_keep_rates if float(x) > 0.0)
    if not rates:
        return (1.0,)
    return tuple(sorted(set(min(float(x), 1.0) for x in rates)))


def slice_s3_synth_by_keep_rate(cache: Dict[str, object], keep_rate: float) -> Tuple[np.ndarray, np.ndarray]:
    X_syn = cache["X_syn"]
    scores = cache.get("filter_scores", np.full(len(X_syn), np.nan, dtype=np.float32))
    if len(X_syn) == 0:
        return X_syn, scores
    n_keep = max(1, min(len(X_syn), int(np.ceil(len(X_syn) * float(keep_rate)))))
    return X_syn[:n_keep], scores[:n_keep]


def clean_s3_synth_by_pos_neg_margin(
    X_pos: np.ndarray,
    X_neg: np.ndarray,
    X_syn: np.ndarray,
    cfg: PipelineConfig,
    task_id: Optional[int],
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    if len(X_syn) == 0 or len(X_pos) == 0 or len(X_neg) == 0:
        return X_syn, np.full(len(X_syn), np.nan, dtype=np.float32)

    weights = load_importance_weights_for_s3(cfg, task_id, seed, X_syn.shape[1])
    if len(weights) != X_syn.shape[1]:
        weights = _uniform_feature_weights(X_syn.shape[1])
    scale = np.sqrt(weights).astype(np.float32, copy=False)
    k_pos = max(1, min(int(cfg.s3_synth_filter_k), len(X_pos)))
    k_neg = max(1, min(int(cfg.s3_synth_filter_k), len(X_neg)))

    pos_nn = NearestNeighbors(n_neighbors=k_pos)
    neg_nn = NearestNeighbors(n_neighbors=k_neg)
    pos_nn.fit(X_pos * scale[None, :])
    neg_nn.fit(X_neg * scale[None, :])
    syn_scaled = X_syn * scale[None, :]
    pos_dist = pos_nn.kneighbors(syn_scaled, return_distance=True)[0].mean(axis=1)
    neg_dist = neg_nn.kneighbors(syn_scaled, return_distance=True)[0].mean(axis=1)
    margin = (neg_dist - pos_dist).astype(np.float32, copy=False)

    keep_mask = margin >= 0.0
    min_keep = max(1, int(np.ceil(len(X_syn) * float(cfg.s3_hybrid_clean_keep_rate))))
    if int(keep_mask.sum()) < min_keep:
        keep_idx = np.argsort(-margin, kind="stable")[:min_keep]
    else:
        keep_idx = np.where(keep_mask)[0]
        keep_idx = keep_idx[np.argsort(-margin[keep_idx], kind="stable")]
    return X_syn[keep_idx], margin[keep_idx]


def _rank01(values: np.ndarray, higher_is_better: bool) -> np.ndarray:
    vals = np.asarray(values, dtype=np.float64)
    out = np.full(len(vals), 0.5, dtype=np.float64)
    finite = np.isfinite(vals)
    if int(finite.sum()) == 0:
        return out
    local = vals[finite]
    if len(local) == 1:
        out[finite] = 1.0
        return out
    order = np.argsort(-local if higher_is_better else local, kind="mergesort")
    ranks = np.empty(len(local), dtype=np.float64)
    ranks[order] = np.arange(len(local), dtype=np.float64)
    out[finite] = 1.0 - ranks / max(len(local) - 1, 1)
    return np.clip(out, 0.0, 1.0)


def cars_bw_variant_coefficients(variant: str) -> Tuple[float, float]:
    name = str(variant).strip().lower()
    if name in {"fidelity", "positive", "pos"}:
        return 1.0, 0.0
    if name in {"boundary", "safety", "canary"}:
        return 0.35, 0.65
    if name in {"aggressive", "utility"}:
        return 0.70, 0.30
    return 0.50, 0.50


def cars_bw_synthetic_weights(
    X_pos: np.ndarray,
    X_canary: np.ndarray,
    X_syn: np.ndarray,
    cfg: PipelineConfig,
    task_id: Optional[int],
    seed: int,
    variant: str,
) -> Tuple[np.ndarray, Dict[str, float | str]]:
    if len(X_syn) == 0:
        return np.empty((0,), dtype=np.float64), {"cars_bw_variant": str(variant)}

    n_features = X_syn.shape[1]
    weights = load_importance_weights_for_s3(cfg, task_id, seed, n_features)
    if len(weights) != n_features or not np.isfinite(weights).all() or float(weights.sum()) <= 0:
        weights = _uniform_feature_weights(n_features)
    top_k = max(1, min(int(cfg.cars_bw_top_k_features), n_features))
    top_idx = np.argsort(-weights, kind="mergesort")[:top_k]
    top_weights = weights[top_idx].astype(np.float64, copy=False)
    if not np.isfinite(top_weights).all() or float(top_weights.sum()) <= 0:
        top_weights = np.ones(len(top_idx), dtype=np.float64)
    scale = np.sqrt(top_weights / max(float(top_weights.sum()), 1e-12))

    syn_scaled = X_syn[:, top_idx].astype(np.float64, copy=False) * scale[None, :]

    if len(X_pos):
        pos_scaled = X_pos[:, top_idx].astype(np.float64, copy=False) * scale[None, :]
        pos_nn = NearestNeighbors(n_neighbors=1, metric="manhattan")
        pos_nn.fit(pos_scaled)
        pos_dist = pos_nn.kneighbors(syn_scaled, return_distance=True)[0][:, 0]
        fidelity = _rank01(pos_dist, higher_is_better=False)
    else:
        pos_dist = np.full(len(X_syn), np.nan, dtype=np.float64)
        fidelity = np.full(len(X_syn), 0.5, dtype=np.float64)

    if len(X_canary):
        canary_scaled = X_canary[:, top_idx].astype(np.float64, copy=False) * scale[None, :]
        canary_nn = NearestNeighbors(n_neighbors=1, metric="manhattan")
        canary_nn.fit(canary_scaled)
        canary_dist = canary_nn.kneighbors(syn_scaled, return_distance=True)[0][:, 0]
        safety = _rank01(canary_dist, higher_is_better=True)
    else:
        canary_dist = np.full(len(X_syn), np.nan, dtype=np.float64)
        safety = np.full(len(X_syn), 0.5, dtype=np.float64)

    fidelity_coef, safety_coef = cars_bw_variant_coefficients(variant)
    denom = max(fidelity_coef + safety_coef, 1e-12)
    raw = (fidelity_coef * fidelity + safety_coef * safety) / denom
    power = max(float(cfg.cars_bw_power), 1e-6)
    raw = np.clip(raw, 0.0, 1.0) ** power
    min_w = max(0.0, min(float(cfg.cars_bw_min_weight), 1.0))
    max_w = max(min_w, float(cfg.cars_bw_max_weight))
    sample_weight = min_w + (max_w - min_w) * raw
    if cfg.cars_bw_normalize_mean and len(sample_weight):
        mean_w = float(np.mean(sample_weight))
        if mean_w > 1e-12:
            sample_weight = sample_weight / mean_w
    sample_weight = np.clip(sample_weight, min_w, max_w).astype(np.float64, copy=False)

    meta: Dict[str, float | str] = {
        "cars_bw_variant": str(variant),
        "cars_bw_fidelity_coef": float(fidelity_coef),
        "cars_bw_safety_coef": float(safety_coef),
        "cars_bw_min_weight": float(min_w),
        "cars_bw_max_weight": float(max_w),
        "cars_bw_power": float(power),
        "cars_bw_normalize_mean": bool(cfg.cars_bw_normalize_mean),
        "cars_bw_top_k_features": int(top_k),
        "cars_bw_synth_weight_mean": float(np.mean(sample_weight)),
        "cars_bw_synth_weight_p10": float(np.percentile(sample_weight, 10)),
        "cars_bw_synth_weight_p50": float(np.percentile(sample_weight, 50)),
        "cars_bw_synth_weight_p90": float(np.percentile(sample_weight, 90)),
        "cars_bw_synth_weight_min": float(np.min(sample_weight)),
        "cars_bw_synth_weight_max": float(np.max(sample_weight)),
        "cars_bw_positive_fidelity_mean": float(np.mean(fidelity)),
        "cars_bw_canary_safety_mean": float(np.mean(safety)),
        "cars_bw_pos_dist_mean": float(np.nanmean(pos_dist)) if np.isfinite(pos_dist).any() else float("nan"),
        "cars_bw_canary_dist_mean": float(np.nanmean(canary_dist)) if np.isfinite(canary_dist).any() else float("nan"),
    }
    return sample_weight, meta


def build_hybrid_augmented_train(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_syn_view: np.ndarray,
    ratio: float,
    cfg: PipelineConfig,
    method: str,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, object]]:
    meta: Dict[str, object] = {
        "hybrid_base_added": 0,
        "hybrid_smote_added": 0,
        "hybrid_total_added": 0,
    }
    if method == "ig_only" or method == "ig_clean":
        X_aug, y_aug, n_add = build_augmented_train(X_train, y_train, X_syn_view, ratio, seed, prefer_front=True)
        meta["hybrid_base_added"] = int(n_add)
        meta["hybrid_total_added"] = int(n_add)
        return X_aug, y_aug, meta

    if method == "ig_smote":
        X_aug, y_aug, n_add = build_augmented_train(X_train, y_train, X_syn_view, ratio, seed, prefer_front=True)
        X_res, y_res, smote_added = smote_resample(
            X_aug,
            y_aug,
            float(cfg.s3_hybrid_smote_ratio),
            seed,
            cfg.s1_smote_k_neighbors,
        )
        meta["hybrid_base_added"] = int(n_add)
        meta["hybrid_smote_added"] = int(smote_added)
        meta["hybrid_total_added"] = int(n_add + smote_added)
        return X_res, y_res, meta

    if method == "s1_plus_ig":
        X_s1, y_s1, smote_added = smote_resample(
            X_train,
            y_train,
            float(cfg.s3_hybrid_s1_ratio),
            seed,
            cfg.s1_smote_k_neighbors,
        )
        X_aug, y_aug, n_add = build_augmented_train(X_s1, y_s1, X_syn_view, ratio, seed, prefer_front=True)
        meta["hybrid_base_added"] = int(n_add)
        meta["hybrid_smote_added"] = int(smote_added)
        meta["hybrid_total_added"] = int(n_add + smote_added)
        return X_aug, y_aug, meta

    raise ValueError(f"Unknown S3 hybrid method: {method}")


def _select_synthetic_addition_indices(
    n_syn: int,
    n_add: int,
    seed: int,
    prefer_front: bool,
) -> np.ndarray:
    if n_add <= 0 or n_syn <= 0:
        return np.empty((0,), dtype=int)
    if prefer_front and n_syn >= n_add:
        return np.arange(n_add, dtype=int)
    rng = np.random.default_rng(seed)
    return rng.choice(n_syn, size=n_add, replace=n_syn < n_add).astype(int)


def build_hybrid_augmented_train_weighted(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_syn_view: np.ndarray,
    X_syn_weights: np.ndarray,
    ratio: float,
    cfg: PipelineConfig,
    method: str,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, object]]:
    meta: Dict[str, object] = {
        "hybrid_base_added": 0,
        "hybrid_smote_added": 0,
        "hybrid_total_added": 0,
        "cars_bw_effective_added_weight_sum": 0.0,
        "cars_bw_effective_added_weight_mean": float("nan"),
    }
    n_pos = int((y_train == 1).sum())
    n_add = int(round(n_pos * ratio))
    if n_add <= 0 or len(X_syn_view) == 0:
        return X_train, y_train, np.ones(len(y_train), dtype=np.float64), meta

    syn_weights = np.asarray(X_syn_weights, dtype=np.float64)
    if len(syn_weights) != len(X_syn_view):
        syn_weights = np.ones(len(X_syn_view), dtype=np.float64)

    base_method = str(method)
    if base_method in {"ig_only", "ig_clean"}:
        X_base, y_base = X_train, y_train
        base_weight = np.ones(len(y_base), dtype=np.float64)
    elif base_method == "s1_plus_ig":
        X_base, y_base, smote_added = smote_resample(
            X_train,
            y_train,
            float(cfg.s3_hybrid_s1_ratio),
            seed,
            cfg.s1_smote_k_neighbors,
        )
        base_weight = np.ones(len(y_base), dtype=np.float64)
        meta["hybrid_smote_added"] = int(smote_added)
    else:
        X_aug, y_aug, hybrid_meta = build_hybrid_augmented_train(
            X_train,
            y_train,
            X_syn_view,
            ratio,
            cfg,
            method,
            seed,
        )
        return X_aug, y_aug, np.ones(len(y_aug), dtype=np.float64), {**meta, **hybrid_meta}

    add_idx = _select_synthetic_addition_indices(len(X_syn_view), n_add, seed, prefer_front=True)
    X_add = X_syn_view[add_idx]
    y_add = np.ones(len(add_idx), dtype=int)
    add_weight = syn_weights[add_idx]

    X_aug = np.vstack([X_base, X_add])
    y_aug = np.concatenate([y_base, y_add])
    sample_weight = np.concatenate([base_weight, add_weight])

    meta["hybrid_base_added"] = int(len(add_idx))
    meta["hybrid_total_added"] = int(len(add_idx) + int(meta["hybrid_smote_added"]))
    meta["cars_bw_effective_added_weight_sum"] = float(np.sum(add_weight))
    meta["cars_bw_effective_added_weight_mean"] = float(np.mean(add_weight)) if len(add_weight) else float("nan")
    meta["cars_bw_effective_added_weight_ratio"] = float(np.sum(add_weight) / max(n_pos, 1))
    return X_aug, y_aug, sample_weight, meta


def s3_synth_diagnostics_for_view(
    cache: Dict[str, object],
    X_syn_view: np.ndarray,
    task_ctx: Dict[str, np.ndarray],
    cfg: PipelineConfig,
) -> Tuple[float, float]:
    X_pos = cache["X_pos"]
    X_neg = cache["X_neg"]
    eval_n = max(1, min(len(X_syn_view), int(round((task_ctx["y_train"] == 1).sum() * max(cfg.gen_ratios)))))
    X_syn_eval = X_syn_view[:eval_n]
    return (
        float(real_vs_synth_auc(X_pos, X_syn_eval, cfg, cfg.seed)),
        float(tstr_utility_pr_auc(X_neg, X_syn_eval, task_ctx["X_val"], task_ctx["y_val"], cfg, cfg.seed)),
    )


def cfg_for_s3_ig(cfg: PipelineConfig) -> PipelineConfig:
    if cfg.s3_synth_filter == "none":
        return replace(cfg, s3_synth_filter="ig_weighted")
    return cfg


def run_s0(task_ctx: Dict[str, np.ndarray], cfg: PipelineConfig) -> List[Dict[str, float]]:
    metrics = run_downstream(
        task_ctx["X_train"],
        task_ctx["y_train"],
        task_ctx["X_val"],
        task_ctx["y_val"],
        task_ctx["X_test"],
        task_ctx["y_test"],
        cfg,
        cfg.seed,
    )
    return [{"scenario": "S0", "method": "none", "ratio": 0.0, "n_added": 0, **metrics}]


def run_s0_strict(task_ctx: Dict[str, np.ndarray], cfg: PipelineConfig):
    val_metrics = run_downstream_val(
        task_ctx["X_train"],
        task_ctx["y_train"],
        task_ctx["X_val"],
        task_ctx["y_val"],
        task_ctx["X_test"],
        task_ctx["y_test"],
        cfg,
        cfg.seed,
    )
    candidate_row = {"scenario": "S0", "method": "none", "ratio": 0.0, "n_added": 0, **val_metrics}
    final_metrics = _run_downstream_internal(
        task_ctx["X_train"],
        task_ctx["y_train"],
        task_ctx["X_val"],
        task_ctx["y_val"],
        task_ctx["X_test"],
        task_ctx["y_test"],
        cfg,
        cfg.seed,
        cfg.selection_threshold_rule,
        "test",
    )
    meta, val_prefixed = split_candidate_row(candidate_row, cfg.selection_metric)
    final_row = {**meta, **final_metrics, **val_prefixed}
    return [candidate_row], [candidate_row], [final_row]


def run_s1(task_ctx: Dict[str, np.ndarray], cfg: PipelineConfig) -> List[Dict[str, float]]:
    rows = []
    for r in cfg.gen_ratios:
        Xr, yr, added = random_over_resample(task_ctx["X_train"], task_ctx["y_train"], r, cfg.seed)
        m = run_downstream(Xr, yr, task_ctx["X_val"], task_ctx["y_val"], task_ctx["X_test"], task_ctx["y_test"], cfg, cfg.seed)
        rows.append({"scenario": "S1", "method": "random_over", "ratio": float(r), "n_added": int(added), **m})

    if cfg.s1_include_smote:
        for r in cfg.gen_ratios:
            Xr, yr, added = smote_resample(
                task_ctx["X_train"],
                task_ctx["y_train"],
                r,
                cfg.seed,
                cfg.s1_smote_k_neighbors,
            )
            m = run_downstream(Xr, yr, task_ctx["X_val"], task_ctx["y_val"], task_ctx["X_test"], task_ctx["y_test"], cfg, cfg.seed)
            rows.append({"scenario": "S1", "method": "smote", "ratio": float(r), "n_added": int(added), **m})

    for neg_pos in cfg.s1_under_neg_pos_ratios:
        Xr, yr, removed = random_under_resample(task_ctx["X_train"], task_ctx["y_train"], neg_pos, cfg.seed)
        m = run_downstream(Xr, yr, task_ctx["X_val"], task_ctx["y_val"], task_ctx["X_test"], task_ctx["y_test"], cfg, cfg.seed)
        rows.append(
            {
                "scenario": "S1",
                "method": f"random_under_{int(neg_pos)}to1",
                "ratio": float("nan"),
                "n_added": -int(removed),
                **m,
            }
        )
    return rows


def run_s1_strict(task_ctx: Dict[str, np.ndarray], cfg: PipelineConfig):
    candidate_rows = []

    for r in cfg.gen_ratios:
        Xr, yr, added = build_s1_resampled_train(task_ctx, cfg, "random_over", r)
        m = run_downstream_val(Xr, yr, task_ctx["X_val"], task_ctx["y_val"], task_ctx["X_test"], task_ctx["y_test"], cfg, cfg.seed)
        candidate_rows.append({"scenario": "S1", "method": "random_over", "ratio": float(r), "n_added": int(added), **m})

    if cfg.s1_include_smote:
        for r in cfg.gen_ratios:
            Xr, yr, added = build_s1_resampled_train(task_ctx, cfg, "smote", r)
            m = run_downstream_val(Xr, yr, task_ctx["X_val"], task_ctx["y_val"], task_ctx["X_test"], task_ctx["y_test"], cfg, cfg.seed)
            candidate_rows.append({"scenario": "S1", "method": "smote", "ratio": float(r), "n_added": int(added), **m})

    for neg_pos in cfg.s1_under_neg_pos_ratios:
        method = f"random_under_{int(neg_pos)}to1"
        Xr, yr, delta = build_s1_resampled_train(task_ctx, cfg, method, float("nan"))
        m = run_downstream_val(Xr, yr, task_ctx["X_val"], task_ctx["y_val"], task_ctx["X_test"], task_ctx["y_test"], cfg, cfg.seed)
        candidate_rows.append({"scenario": "S1", "method": method, "ratio": float("nan"), "n_added": int(delta), **m})

    candidate_df = pd.DataFrame(candidate_rows)
    winners = select_candidate_winners(candidate_df, ["scenario", "method"], cfg.selection_metric)

    final_rows = []
    for row in winners.to_dict("records"):
        Xr, yr, delta = build_s1_resampled_train(task_ctx, cfg, row["method"], float(row["ratio"]))
        test_metrics = _run_downstream_internal(
            Xr,
            yr,
            task_ctx["X_val"],
            task_ctx["y_val"],
            task_ctx["X_test"],
            task_ctx["y_test"],
            cfg,
            cfg.seed,
            cfg.selection_threshold_rule,
            "test",
        )
        meta, val_prefixed = split_candidate_row(row, cfg.selection_metric)
        meta["n_added"] = int(delta)
        final_rows.append({**meta, **test_metrics, **val_prefixed})

    return candidate_rows, winners.to_dict("records"), final_rows


def run_s2(task_ctx: Dict[str, np.ndarray], cfg: PipelineConfig) -> List[Dict[str, float]]:
    rows = []
    X_train = task_ctx["X_train"]
    y_train = task_ctx["y_train"]
    X_pos = X_train[y_train == 1]
    X_neg = X_train[y_train == 0]
    methods = {x.lower() for x in cfg.s2_methods}

    if len(X_pos) < 30:
        rows.append({"scenario": "S2", "method": "skip_not_enough_pos", "ratio": float("nan"), "n_added": 0})
        return rows

    max_ratio = max(cfg.gen_ratios)
    max_n_add = int(round((y_train == 1).sum() * max_ratio))

    if "cvae" in methods:
        cvae, cvae_sc = train_cvae(X_pos, cfg, cfg.seed)
        X_syn_cvae = sample_cvae(cvae, cvae_sc, max_n_add, cfg.seed)
        q_auc = real_vs_synth_auc(X_pos, X_syn_cvae, cfg, cfg.seed)
        tstr = tstr_utility_pr_auc(X_neg, X_syn_cvae, task_ctx["X_val"], task_ctx["y_val"], cfg, cfg.seed)

        for r in cfg.gen_ratios:
            X_aug, y_aug, n_add = build_augmented_train(X_train, y_train, X_syn_cvae, r, cfg.seed)
            m = run_downstream(X_aug, y_aug, task_ctx["X_val"], task_ctx["y_val"], task_ctx["X_test"], task_ctx["y_test"], cfg, cfg.seed)
            rows.append(
                {
                    "scenario": "S2",
                    "method": "cvae",
                    "ratio": float(r),
                    "n_added": int(n_add),
                    "real_vs_synth_auc": float(q_auc),
                    "tstr_pr_auc": float(tstr),
                    **m,
                }
            )

    if "cgan" in methods:
        gan_G, gan_sc = train_cgan(X_pos, cfg, cfg.seed)
        X_syn_gan = sample_cgan(gan_G, gan_sc, max_n_add, cfg, cfg.seed)
        q_auc = real_vs_synth_auc(X_pos, X_syn_gan, cfg, cfg.seed)
        tstr = tstr_utility_pr_auc(X_neg, X_syn_gan, task_ctx["X_val"], task_ctx["y_val"], cfg, cfg.seed)

        for r in cfg.gen_ratios:
            X_aug, y_aug, n_add = build_augmented_train(X_train, y_train, X_syn_gan, r, cfg.seed)
            m = run_downstream(X_aug, y_aug, task_ctx["X_val"], task_ctx["y_val"], task_ctx["X_test"], task_ctx["y_test"], cfg, cfg.seed)
            rows.append(
                {
                    "scenario": "S2",
                    "method": "cgan",
                    "ratio": float(r),
                    "n_added": int(n_add),
                    "real_vs_synth_auc": float(q_auc),
                    "tstr_pr_auc": float(tstr),
                    **m,
                }
            )
    return rows


def run_s2_strict(task_ctx: Dict[str, np.ndarray], cfg: PipelineConfig):
    candidate_rows = []
    final_rows = []
    synth_cache: Dict[str, Dict[str, object]] = {}

    X_train = task_ctx["X_train"]
    y_train = task_ctx["y_train"]
    X_pos = X_train[y_train == 1]
    X_neg = X_train[y_train == 0]
    methods = {x.lower() for x in cfg.s2_methods}

    if len(X_pos) < 30:
        skip_row = {"scenario": "S2", "method": "skip_not_enough_pos", "ratio": float("nan"), "n_added": 0}
        return [skip_row], [skip_row], []

    max_ratio = max(cfg.gen_ratios)
    max_n_add = int(round((y_train == 1).sum() * max_ratio))

    if "cvae" in methods:
        cvae, cvae_sc = train_cvae(X_pos, cfg, cfg.seed)
        X_syn_cvae = sample_cvae(cvae, cvae_sc, max_n_add, cfg.seed)
        synth_cache["cvae"] = {
            "X_syn": X_syn_cvae,
            "real_vs_synth_auc": float(real_vs_synth_auc(X_pos, X_syn_cvae, cfg, cfg.seed)),
            "tstr_pr_auc": float(tstr_utility_pr_auc(X_neg, X_syn_cvae, task_ctx["X_val"], task_ctx["y_val"], cfg, cfg.seed)),
        }
        for r in cfg.gen_ratios:
            X_aug, y_aug, n_add = build_augmented_train(X_train, y_train, X_syn_cvae, r, cfg.seed)
            m = run_downstream_val(X_aug, y_aug, task_ctx["X_val"], task_ctx["y_val"], task_ctx["X_test"], task_ctx["y_test"], cfg, cfg.seed)
            candidate_rows.append(
                {
                    "scenario": "S2",
                    "method": "cvae",
                    "ratio": float(r),
                    "n_added": int(n_add),
                    "real_vs_synth_auc": synth_cache["cvae"]["real_vs_synth_auc"],
                    "tstr_pr_auc": synth_cache["cvae"]["tstr_pr_auc"],
                    **m,
                }
            )

    if "cgan" in methods:
        gan_G, gan_sc = train_cgan(X_pos, cfg, cfg.seed)
        X_syn_gan = sample_cgan(gan_G, gan_sc, max_n_add, cfg, cfg.seed)
        synth_cache["cgan"] = {
            "X_syn": X_syn_gan,
            "real_vs_synth_auc": float(real_vs_synth_auc(X_pos, X_syn_gan, cfg, cfg.seed)),
            "tstr_pr_auc": float(tstr_utility_pr_auc(X_neg, X_syn_gan, task_ctx["X_val"], task_ctx["y_val"], cfg, cfg.seed)),
        }
        for r in cfg.gen_ratios:
            X_aug, y_aug, n_add = build_augmented_train(X_train, y_train, X_syn_gan, r, cfg.seed)
            m = run_downstream_val(X_aug, y_aug, task_ctx["X_val"], task_ctx["y_val"], task_ctx["X_test"], task_ctx["y_test"], cfg, cfg.seed)
            candidate_rows.append(
                {
                    "scenario": "S2",
                    "method": "cgan",
                    "ratio": float(r),
                    "n_added": int(n_add),
                    "real_vs_synth_auc": synth_cache["cgan"]["real_vs_synth_auc"],
                    "tstr_pr_auc": synth_cache["cgan"]["tstr_pr_auc"],
                    **m,
                }
            )

    candidate_df = pd.DataFrame(candidate_rows)
    winners = select_candidate_winners(candidate_df, ["scenario", "method"], cfg.selection_metric)

    for row in winners.to_dict("records"):
        cache = synth_cache[row["method"]]
        X_aug, y_aug, n_add = build_augmented_train(X_train, y_train, cache["X_syn"], float(row["ratio"]), cfg.seed)
        test_metrics = _run_downstream_internal(
            X_aug,
            y_aug,
            task_ctx["X_val"],
            task_ctx["y_val"],
            task_ctx["X_test"],
            task_ctx["y_test"],
            cfg,
            cfg.seed,
            cfg.selection_threshold_rule,
            "test",
        )
        meta, val_prefixed = split_candidate_row(row, cfg.selection_metric)
        meta["n_added"] = int(n_add)
        final_rows.append({**meta, **test_metrics, **val_prefixed})

    return candidate_rows, winners.to_dict("records"), final_rows

def run_s3(task_ctx: Dict[str, np.ndarray], cfg: PipelineConfig, scenario_label: str = "S3") -> List[Dict[str, float]]:
    rows = []
    X_train = task_ctx["X_train"]
    y_train = task_ctx["y_train"]

    X_pos = X_train[y_train == 1]
    X_neg = X_train[y_train == 0]
    if len(X_pos) < 30:
        rows.append({"scenario": "S3", "method": "skip_not_enough_pos", "ratio": float("nan"), "n_added": 0})
        return rows

    train_mode = resolve_s3_train_modes(task_ctx, cfg)[0] if cfg.s3_auto_mode_by_pos_rate else cfg.ddpm_train_mode

    pool = build_s3_synth_pool(task_ctx, cfg, train_mode)
    X_neg_sub = pool["X_neg_sub"]
    X_syn = pool["X_syn_raw"]

    filter_scores = np.full(len(X_syn), np.nan, dtype=np.float32)
    if cfg.s3_synth_filter != "none":
        X_syn, filter_scores = maybe_filter_s3_synth(
            X_pos,
            X_syn,
            cfg,
            cfg.seed,
            task_id=task_ctx.get("task_id"),
        )

    max_ratio = max(cfg.gen_ratios)
    max_n_add = int(round((y_train == 1).sum() * max_ratio))
    cache = {
        "X_syn": X_syn,
        "X_pos": X_pos,
        "X_neg": X_neg,
        "filter_scores": filter_scores,
    }

    for keep_rate in s3_candidate_keep_rates(cfg):
        X_syn_view, view_scores = slice_s3_synth_by_keep_rate(cache, keep_rate)
        eval_n = max(1, min(len(X_syn_view), max_n_add))
        X_syn_eval = X_syn_view[:eval_n]
        q_auc = real_vs_synth_auc(X_pos, X_syn_eval, cfg, cfg.seed)
        tstr = tstr_utility_pr_auc(X_neg, X_syn_eval, task_ctx["X_val"], task_ctx["y_val"], cfg, cfg.seed)
        for r in cfg.gen_ratios:
            X_aug, y_aug, n_add = build_augmented_train(
                X_train,
                y_train,
                X_syn_view,
                r,
                cfg.seed,
                prefer_front=(cfg.s3_synth_filter != "none"),
            )
            m = run_downstream(X_aug, y_aug, task_ctx["X_val"], task_ctx["y_val"], task_ctx["X_test"], task_ctx["y_test"], cfg, cfg.seed)
            rows.append(
                {
                    "scenario": scenario_label,
                    "method": "tabddpm_cond_y1",
                    "ratio": float(r),
                    "s3_ig_keep_rate": float(keep_rate),
                    "n_added": int(n_add),
                    "ddpm_neg_subset": int(len(X_neg_sub)),
                    "ddpm_train_mode": train_mode,
                    "real_vs_synth_auc": float(q_auc),
                    "tstr_pr_auc": float(tstr),
                    "s3_synth_filter": cfg.s3_synth_filter,
                    "s3_postprocess": cfg.s3_postprocess,
                    "s3_synth_pool_size": int(len(X_syn_view)),
                    "s3_filter_score_mean": float(np.nanmean(view_scores)) if len(view_scores) else float("nan"),
                    **m,
                }
            )
    return rows


def run_s3_strict(task_ctx: Dict[str, np.ndarray], cfg: PipelineConfig, scenario_label: str = "S3"):
    candidate_rows = []
    final_rows = []
    synth_cache: Dict[str, Dict[str, object]] = {}

    X_train = task_ctx["X_train"]
    y_train = task_ctx["y_train"]
    X_pos = X_train[y_train == 1]
    if len(X_pos) < 30:
        skip_row = {"scenario": "S3", "method": "skip_not_enough_pos", "ratio": float("nan"), "n_added": 0}
        return [skip_row], [skip_row], []

    for train_mode in resolve_s3_train_modes(task_ctx, cfg):
        cache = build_s3_synth_cache(task_ctx, cfg, train_mode)
        synth_cache[train_mode] = cache
        for keep_rate in s3_candidate_keep_rates(cfg):
            X_syn_view, view_scores = slice_s3_synth_by_keep_rate(cache, keep_rate)
            q_auc, tstr = s3_synth_diagnostics_for_view(cache, X_syn_view, task_ctx, cfg)
            for r in cfg.gen_ratios:
                X_aug, y_aug, n_add = build_augmented_train(
                    X_train,
                    y_train,
                    X_syn_view,
                    r,
                    cfg.seed,
                    prefer_front=(cfg.s3_synth_filter != "none"),
                )
                m = run_downstream_val(X_aug, y_aug, task_ctx["X_val"], task_ctx["y_val"], task_ctx["X_test"], task_ctx["y_test"], cfg, cfg.seed)
                candidate_rows.append(
                    {
                        "scenario": scenario_label,
                        "method": "tabddpm_cond_y1",
                        "ratio": float(r),
                        "s3_ig_keep_rate": float(keep_rate),
                        "n_added": int(n_add),
                        "ddpm_neg_subset": int(cache["ddpm_neg_subset"]),
                        "ddpm_train_mode": train_mode,
                        "real_vs_synth_auc": float(q_auc),
                        "tstr_pr_auc": float(tstr),
                        "s3_synth_filter": cache["s3_synth_filter"],
                        "s3_postprocess": cache["s3_postprocess"],
                        "s3_synth_pool_size": int(len(X_syn_view)),
                        "s3_filter_score_mean": float(np.nanmean(view_scores)) if len(view_scores) else float("nan"),
                        **m,
                    }
                )

    candidate_df = pd.DataFrame(candidate_rows)
    winners = select_candidate_winners(candidate_df, ["scenario", "method"], cfg.selection_metric)

    for row in winners.to_dict("records"):
        cache = synth_cache[str(row["ddpm_train_mode"])]
        X_syn_view, _ = slice_s3_synth_by_keep_rate(cache, float(row.get("s3_ig_keep_rate", 1.0)))
        X_aug, y_aug, n_add = build_augmented_train(
            X_train,
            y_train,
            X_syn_view,
            float(row["ratio"]),
            cfg.seed,
            prefer_front=(cfg.s3_synth_filter != "none"),
        )
        test_metrics = _run_downstream_internal(
            X_aug,
            y_aug,
            task_ctx["X_val"],
            task_ctx["y_val"],
            task_ctx["X_test"],
            task_ctx["y_test"],
            cfg,
            cfg.seed,
            cfg.selection_threshold_rule,
            "test",
        )
        meta, val_prefixed = split_candidate_row(row, cfg.selection_metric)
        meta["n_added"] = int(n_add)
        final_rows.append({**meta, **test_metrics, **val_prefixed})

    return candidate_rows, winners.to_dict("records"), final_rows


def run_s3_hybrid_strict(
    task_ctx: Dict[str, np.ndarray],
    cfg: PipelineConfig,
    scenario_label: str = "S3_HYBRID",
    cars_probe_ctx: Optional[Dict[str, object]] = None,
):
    candidate_rows = []
    final_rows = []
    synth_cache: Dict[str, Dict[str, object]] = {}
    clean_cache: Dict[Tuple[str, float], Tuple[np.ndarray, np.ndarray]] = {}

    X_train = task_ctx["X_train"]
    y_train = task_ctx["y_train"]
    X_pos = X_train[y_train == 1]
    if len(X_pos) < 30:
        skip_row = {"scenario": scenario_label, "method": "skip_not_enough_pos", "ratio": float("nan"), "n_added": 0}
        return [skip_row], [skip_row], []

    hybrid_cfg = cfg_for_s3_ig(cfg)
    methods = tuple(x for x in cfg.s3_hybrid_methods if x)
    if not methods:
        methods = ("ig_only",)

    for train_mode in resolve_s3_train_modes(task_ctx, hybrid_cfg):
        cache = build_s3_synth_cache(task_ctx, hybrid_cfg, train_mode)
        synth_cache[train_mode] = cache
        for keep_rate in s3_candidate_keep_rates(hybrid_cfg):
            X_syn_view, view_scores = slice_s3_synth_by_keep_rate(cache, keep_rate)
            clean_key = (train_mode, float(keep_rate))
            X_clean, clean_scores = clean_s3_synth_by_pos_neg_margin(
                cache["X_pos"],
                cache["X_neg"],
                X_syn_view,
                hybrid_cfg,
                task_ctx.get("task_id"),
                hybrid_cfg.seed,
            )
            clean_cache[clean_key] = (X_clean, clean_scores)

            for method in methods:
                X_method_view = X_clean if method == "ig_clean" else X_syn_view
                method_scores = clean_scores if method == "ig_clean" else view_scores
                q_auc, tstr = s3_synth_diagnostics_for_view(cache, X_method_view, task_ctx, hybrid_cfg)
                for r in hybrid_cfg.gen_ratios:
                    X_aug, y_aug, hybrid_meta = build_hybrid_augmented_train(
                        X_train,
                        y_train,
                        X_method_view,
                        float(r),
                        hybrid_cfg,
                        method,
                        hybrid_cfg.seed,
                    )
                    candidate_meta = {
                        "scenario": scenario_label,
                        "method": method,
                        "ratio": float(r),
                        "s3_ig_keep_rate": float(keep_rate),
                        "n_added": int(hybrid_meta["hybrid_total_added"]),
                        "ddpm_neg_subset": int(cache["ddpm_neg_subset"]),
                        "ddpm_train_mode": train_mode,
                        "real_vs_synth_auc": float(q_auc),
                        "tstr_pr_auc": float(tstr),
                        "s3_synth_filter": cache["s3_synth_filter"],
                        "s3_postprocess": cache["s3_postprocess"],
                        "s3_synth_pool_size": int(len(X_method_view)),
                        "s3_filter_score_mean": float(np.nanmean(method_scores)) if len(method_scores) else float("nan"),
                        "hybrid_base_added": int(hybrid_meta["hybrid_base_added"]),
                        "hybrid_smote_added": int(hybrid_meta["hybrid_smote_added"]),
                        "hybrid_clean_kept": int(len(X_clean)),
                    }
                    replay_fields: Dict[str, float | str | bool] = {}
                    if hybrid_cfg.cars_candidate_replay and cars_probe_ctx is not None:
                        m, test_m, preds, _ = _run_downstream_val_test_with_predictions(
                            X_aug,
                            y_aug,
                            task_ctx["X_val"],
                            task_ctx["y_val"],
                            task_ctx["X_test"],
                            task_ctx["y_test"],
                            hybrid_cfg,
                            hybrid_cfg.seed,
                            hybrid_cfg.selection_threshold_rule,
                            hybrid_cfg.cars_candidate_replay_include_test,
                        )
                        replay_fields = cars_candidate_replay_fields(
                            candidate_meta,
                            cars_probe_ctx,
                            np.asarray(preds["p_val"]),
                            test_m,
                        )
                        if hybrid_cfg.cars_candidate_replay_save_predictions:
                            record_cars_candidate_test_prediction(
                                cars_probe_ctx, candidate_meta, preds
                            )
                    else:
                        m = run_downstream_val(
                            X_aug,
                            y_aug,
                            task_ctx["X_val"],
                            task_ctx["y_val"],
                            task_ctx["X_test"],
                            task_ctx["y_test"],
                            hybrid_cfg,
                            hybrid_cfg.seed,
                        )
                    candidate_rows.append(
                        {
                            **candidate_meta,
                            **replay_fields,
                            **m,
                        }
                    )

                if hybrid_cfg.cars_bw_prototype:
                    X_canary = (
                        np.asarray(cars_probe_ctx.get("X_canary"))
                        if cars_probe_ctx is not None and "X_canary" in cars_probe_ctx
                        else np.empty((0, X_method_view.shape[1]))
                    )
                    for bw_variant in tuple(x for x in hybrid_cfg.cars_bw_variants if str(x).strip()):
                        syn_weights, bw_score_meta = cars_bw_synthetic_weights(
                            cache["X_pos"],
                            X_canary,
                            X_method_view,
                            hybrid_cfg,
                            task_ctx.get("task_id"),
                            hybrid_cfg.seed,
                            str(bw_variant),
                        )
                        X_bw_view = X_method_view
                        bw_weights = syn_weights
                        if hybrid_cfg.cars_bw_sort_synthetic and len(bw_weights):
                            bw_order = np.argsort(-bw_weights, kind="mergesort")
                            X_bw_view = X_method_view[bw_order]
                            bw_weights = bw_weights[bw_order]
                        bw_score_meta["cars_bw_sort_synthetic"] = bool(hybrid_cfg.cars_bw_sort_synthetic)
                        for r in hybrid_cfg.gen_ratios:
                            X_aug, y_aug, sample_weight, hybrid_meta = build_hybrid_augmented_train_weighted(
                                X_train,
                                y_train,
                                X_bw_view,
                                bw_weights,
                                float(r),
                                hybrid_cfg,
                                method,
                                hybrid_cfg.seed,
                            )
                            bw_method = f"{method}_bw_{str(bw_variant).strip()}"
                            candidate_meta = {
                                "scenario": scenario_label,
                                "method": bw_method,
                                "ratio": float(r),
                                "s3_ig_keep_rate": float(keep_rate),
                                "n_added": int(hybrid_meta["hybrid_total_added"]),
                                "ddpm_neg_subset": int(cache["ddpm_neg_subset"]),
                                "ddpm_train_mode": train_mode,
                                "real_vs_synth_auc": float(q_auc),
                                "tstr_pr_auc": float(tstr),
                                "s3_synth_filter": cache["s3_synth_filter"],
                                "s3_postprocess": cache["s3_postprocess"],
                                "s3_synth_pool_size": int(len(X_method_view)),
                                "s3_filter_score_mean": float(np.nanmean(method_scores)) if len(method_scores) else float("nan"),
                                "hybrid_base_added": int(hybrid_meta["hybrid_base_added"]),
                                "hybrid_smote_added": int(hybrid_meta["hybrid_smote_added"]),
                                "hybrid_clean_kept": int(len(X_clean)),
                                "cars_bw_base_method": method,
                                **bw_score_meta,
                                **hybrid_meta,
                            }
                            replay_fields = {}
                            if hybrid_cfg.cars_candidate_replay and cars_probe_ctx is not None:
                                m, test_m, preds, _ = _run_downstream_val_test_with_predictions(
                                    X_aug,
                                    y_aug,
                                    task_ctx["X_val"],
                                    task_ctx["y_val"],
                                    task_ctx["X_test"],
                                    task_ctx["y_test"],
                                    hybrid_cfg,
                                    hybrid_cfg.seed,
                                    hybrid_cfg.selection_threshold_rule,
                                    hybrid_cfg.cars_candidate_replay_include_test,
                                    sample_weight=sample_weight,
                                )
                                replay_fields = cars_candidate_replay_fields(
                                    candidate_meta,
                                    cars_probe_ctx,
                                    np.asarray(preds["p_val"]),
                                    test_m,
                                )
                                if hybrid_cfg.cars_candidate_replay_save_predictions:
                                    record_cars_candidate_test_prediction(
                                        cars_probe_ctx, candidate_meta, preds
                                    )
                            else:
                                m, _, _, _ = _run_downstream_val_test_with_predictions(
                                    X_aug,
                                    y_aug,
                                    task_ctx["X_val"],
                                    task_ctx["y_val"],
                                    task_ctx["X_test"],
                                    task_ctx["y_test"],
                                    hybrid_cfg,
                                    hybrid_cfg.seed,
                                    hybrid_cfg.selection_threshold_rule,
                                    False,
                                    sample_weight=sample_weight,
                                )
                            candidate_rows.append(
                                {
                                    **candidate_meta,
                                    **replay_fields,
                                    **m,
                                }
                            )

    candidate_df = pd.DataFrame(candidate_rows)
    winners = select_candidate_winners(candidate_df, ["scenario", "method"], hybrid_cfg.selection_metric)

    for row in winners.to_dict("records"):
        cache = synth_cache[str(row["ddpm_train_mode"])]
        keep_rate = float(row.get("s3_ig_keep_rate", 1.0))
        X_syn_view, view_scores = slice_s3_synth_by_keep_rate(cache, keep_rate)
        X_clean, clean_scores = clean_cache.get(
            (str(row["ddpm_train_mode"]), keep_rate),
            clean_s3_synth_by_pos_neg_margin(
                cache["X_pos"],
                cache["X_neg"],
                X_syn_view,
                hybrid_cfg,
                task_ctx.get("task_id"),
                hybrid_cfg.seed,
            ),
        )
        method = str(row["method"])
        base_method_raw = row.get("cars_bw_base_method", method)
        base_method = method if pd.isna(base_method_raw) else str(base_method_raw)
        if "_bw_" in base_method:
            base_method = base_method.split("_bw_", 1)[0]
        X_method_view = X_clean if base_method == "ig_clean" else X_syn_view
        bw_variant = row.get("cars_bw_variant", float("nan"))
        is_bw = not pd.isna(bw_variant)
        sample_weight = None
        if is_bw:
            X_canary = (
                np.asarray(cars_probe_ctx.get("X_canary"))
                if cars_probe_ctx is not None and "X_canary" in cars_probe_ctx
                else np.empty((0, X_method_view.shape[1]))
            )
            syn_weights, _ = cars_bw_synthetic_weights(
                cache["X_pos"],
                X_canary,
                X_method_view,
                hybrid_cfg,
                task_ctx.get("task_id"),
                hybrid_cfg.seed,
                str(bw_variant),
            )
            X_bw_view = X_method_view
            bw_weights = syn_weights
            if hybrid_cfg.cars_bw_sort_synthetic and len(bw_weights):
                bw_order = np.argsort(-bw_weights, kind="mergesort")
                X_bw_view = X_method_view[bw_order]
                bw_weights = bw_weights[bw_order]
            X_aug, y_aug, sample_weight, hybrid_meta = build_hybrid_augmented_train_weighted(
                X_train,
                y_train,
                X_bw_view,
                bw_weights,
                float(row["ratio"]),
                hybrid_cfg,
                base_method,
                hybrid_cfg.seed,
            )
        else:
            X_aug, y_aug, hybrid_meta = build_hybrid_augmented_train(
                X_train,
                y_train,
                X_method_view,
                float(row["ratio"]),
                hybrid_cfg,
                base_method,
                hybrid_cfg.seed,
            )
        test_metrics, preds, _ = _run_downstream_with_predictions(
            X_aug,
            y_aug,
            task_ctx["X_val"],
            task_ctx["y_val"],
            task_ctx["X_test"],
            task_ctx["y_test"],
            hybrid_cfg,
            hybrid_cfg.seed,
            hybrid_cfg.selection_threshold_rule,
            "test",
            sample_weight=sample_weight,
        )
        canary_metrics = cars_canary_inflation_metrics(cars_probe_ctx, np.asarray(preds["p_val"]))
        meta, val_prefixed = split_candidate_row(row, hybrid_cfg.selection_metric)
        meta["n_added"] = int(hybrid_meta["hybrid_total_added"])
        meta["hybrid_base_added"] = int(hybrid_meta["hybrid_base_added"])
        meta["hybrid_smote_added"] = int(hybrid_meta["hybrid_smote_added"])
        meta["hybrid_clean_kept"] = int(len(X_clean))
        if is_bw:
            for key in [
                "cars_bw_effective_added_weight_sum",
                "cars_bw_effective_added_weight_mean",
                "cars_bw_effective_added_weight_ratio",
            ]:
                if key in hybrid_meta:
                    meta[key] = hybrid_meta[key]
        final_rows.append({**meta, **test_metrics, **val_prefixed, **canary_metrics})

    return candidate_rows, winners.to_dict("records"), final_rows


def run_pipeline(cfg: PipelineConfig) -> Dict[str, Path]:
    if cfg.cars_candidate_replay_save_predictions and not (
        cfg.cars_canary_probe
        and cfg.cars_candidate_replay
        and cfg.cars_candidate_replay_include_test
        and "S3_HYBRID" in cfg.scenarios
    ):
        raise ValueError(
            "--cars-candidate-replay-save-predictions requires the CARS canary probe, "
            "candidate replay with test prediction enabled, and S3_HYBRID."
        )
    set_seed(cfg.seed)
    torch_dev = resolve_torch_device(cfg.torch_device)
    configure_torch_runtime(torch_dev)
    print(
        f"[INFO] torch_cuda={torch.cuda.is_available()} torch_device={torch_dev} "
        f"lgbm_device={cfg.lgbm_device_type}"
    )
    if torch.cuda.is_available():
        print(f"[INFO] gpu_name={torch.cuda.get_device_name(0)}")

    out_root = Path(cfg.output_root)
    if cfg.strict_eval:
        out_root = out_root / "paper_strict"
    base_run_name = cfg.run_name.strip() or f"{cfg.toolset}_{cfg.representation}_{time.strftime('%Y%m%d_%H%M%S')}"
    run_dir = out_root / base_run_name
    if run_dir.exists():
        run_dir = out_root / f"{base_run_name}_{time.strftime('%Y%m%d_%H%M%S')}"
    run_dir.mkdir(parents=True, exist_ok=True)

    with open(run_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(asdict(cfg), f, indent=2)

    X_train, X_val, X_test, y_train_raw, y_val_raw, y_test_raw = load_toolset_arrays(
        cfg.data_root,
        cfg.toolset,
        cfg.x_mmap_mode,
    )

    stats = pd.concat(
        [
            task_stats_one_split(y_train_raw, "train"),
            task_stats_one_split(y_val_raw, "val"),
            task_stats_one_split(y_test_raw, "test"),
        ],
        ignore_index=True,
    )
    stats.to_csv(run_dir / "task_stats.csv", index=False)

    train_stats = stats[stats["split"] == "train"].copy()
    tasks = choose_tasks(train_stats, cfg)
    print(f"[INFO] selected tasks: {tasks}")
    use_taskwise_repr = bool(cfg.taskwise_representation and len(tasks) == 1 and not cfg.use_rep_cache)
    if use_taskwise_repr:
        print("[INFO] task-wise representation mode enabled")
        X_train_rep = None
        X_val_rep = None
        X_test_rep = None
    else:
        X_train_rep = build_or_load_representation(X_train, cfg, "train")
        X_val_rep = build_or_load_representation(X_val, cfg, "val")
        X_test_rep = build_or_load_representation(X_test, cfg, "test")

    all_rows = []
    candidate_val_rows = []
    selected_winner_rows = []
    final_test_rows = []
    for task_id in tasks:
        print(f"\n[INFO] running task={task_id}")
        mtr, ytr = extract_task_binary(y_train_raw, task_id)
        mva, yva = extract_task_binary(y_val_raw, task_id)
        mte, yte = extract_task_binary(y_test_raw, task_id)
        if use_taskwise_repr:
            Xtr = build_representation(X_train[mtr], cfg.representation).astype(np.float32, copy=False)
            Xva = build_representation(X_val[mva], cfg.representation).astype(np.float32, copy=False)
            Xte = build_representation(X_test[mte], cfg.representation).astype(np.float32, copy=False)
        else:
            Xtr = X_train_rep[mtr]
            Xva = X_val_rep[mva]
            Xte = X_test_rep[mte]

        prep = TabularPreprocessor(drop_constant=cfg.drop_constant, imputer_strategy=cfg.imputer_strategy)
        prep.fit(Xtr)
        Xtr_c = prep.transform_classifier(Xtr)
        Xva_c = prep.transform_classifier(Xva)
        Xte_c = prep.transform_classifier(Xte)

        task_ctx = {
            "X_train": Xtr_c,
            "y_train": ytr,
            "X_val": Xva_c,
            "y_val": yva,
            "X_test": Xte_c,
            "y_test": yte,
            "task_id": int(task_id),
        }

        meta = {
            "toolset": cfg.toolset,
            "representation": cfg.representation,
            "task_id": int(task_id),
            "train_n": int(len(ytr)),
            "val_n": int(len(yva)),
            "test_n": int(len(yte)),
            "train_pos": int((ytr == 1).sum()),
            "val_pos": int((yva == 1).sum()),
            "test_pos": int((yte == 1).sum()),
            "seed": int(cfg.seed),
        }

        if cfg.strict_eval:
            task_candidate_rows: List[Dict[str, float]] = []
            task_selected_rows: List[Dict[str, float]] = []
            task_final_rows: List[Dict[str, float]] = []

            if "S0" in cfg.scenarios:
                print("  - S0 baseline (strict)")
                cands, wins, finals = run_s0_strict(task_ctx, cfg)
                task_candidate_rows.extend(cands)
                task_selected_rows.extend(wins)
                task_final_rows.extend(finals)
            if "S1" in cfg.scenarios:
                print("  - S1 resampling (strict)")
                cands, wins, finals = run_s1_strict(task_ctx, cfg)
                task_candidate_rows.extend(cands)
                task_selected_rows.extend(wins)
                task_final_rows.extend(finals)

            cars_probe_ctx = None
            if cfg.cars_canary_probe:
                cars_probe_ctx = build_cars_canary_context(task_ctx, cfg, task_candidate_rows)
                if cars_probe_ctx is None:
                    print("  - CARS canary probe skipped: no S0/S1 safe-baseline candidate available")
                else:
                    canary_dir = run_dir / "cars_canary_sets"
                    canary_dir.mkdir(parents=True, exist_ok=True)
                    canary_path = canary_dir / f"task_{int(task_id)}_seed_{int(cfg.seed)}_canary.csv"
                    pd.DataFrame(cars_probe_ctx["canary_table"]).to_csv(canary_path, index=False)
                    print(
                        "  - CARS canary probe: "
                        f"safe={cars_probe_ctx['safe_candidate_key']} "
                        f"canaries={len(cars_probe_ctx['canary_idx'])}"
                    )
                set_seed(cfg.seed)
            if "S2" in cfg.scenarios:
                print("  - S2 cVAE/cGAN synthesis (strict)")
                cands, wins, finals = run_s2_strict(task_ctx, cfg)
                task_candidate_rows.extend(cands)
                task_selected_rows.extend(wins)
                task_final_rows.extend(finals)
            if "S3" in cfg.scenarios:
                print("  - S3 TabDDPM synthesis (strict)")
                cands, wins, finals = run_s3_strict(task_ctx, cfg, "S3")
                task_candidate_rows.extend(cands)
                task_selected_rows.extend(wins)
                task_final_rows.extend(finals)
            if "S3_IG" in cfg.scenarios:
                print("  - S3-IG TabDDPM synthesis (strict)")
                cands, wins, finals = run_s3_strict(task_ctx, cfg_for_s3_ig(cfg), "S3_IG")
                task_candidate_rows.extend(cands)
                task_selected_rows.extend(wins)
                task_final_rows.extend(finals)
            if "S3_HYBRID" in cfg.scenarios:
                print("  - S3-HYBRID TabDDPM synthesis (strict)")
                cands, wins, finals = run_s3_hybrid_strict(task_ctx, cfg, "S3_HYBRID", cars_probe_ctx)
                task_candidate_rows.extend(cands)
                task_selected_rows.extend(wins)
                task_final_rows.extend(finals)
                if cfg.cars_candidate_replay_save_predictions:
                    write_cars_candidate_test_predictions(
                        run_dir,
                        int(task_id),
                        int(cfg.seed),
                        task_ctx,
                        cars_probe_ctx,
                    )

            for row in task_candidate_rows:
                row.update(meta)
            for row in task_selected_rows:
                row.update(meta)
            for row in task_final_rows:
                row.update(meta)

            candidate_val_rows.extend(task_candidate_rows)
            selected_winner_rows.extend(task_selected_rows)
            final_test_rows.extend(task_final_rows)
            write_strict_partial_results(run_dir, candidate_val_rows, selected_winner_rows, final_test_rows)
            if cfg.cars_canary_probe:
                canary_metric_rows = [r for r in final_test_rows if "cars_canary_n" in r]
                if canary_metric_rows:
                    pd.DataFrame(canary_metric_rows).to_csv(run_dir / "cars_canary_metrics.csv", index=False)
            print(
                f"[INFO] task={task_id} strict partial saved "
                f"candidates={len(candidate_val_rows)} winners={len(selected_winner_rows)} finals={len(final_test_rows)}",
                flush=True,
            )
        else:
            rows = []
            if "S0" in cfg.scenarios:
                print("  - S0 baseline")
                rows.extend(run_s0(task_ctx, cfg))
            if "S1" in cfg.scenarios:
                print("  - S1 resampling")
                rows.extend(run_s1(task_ctx, cfg))
            if "S2" in cfg.scenarios:
                print("  - S2 cVAE/cGAN synthesis")
                rows.extend(run_s2(task_ctx, cfg))
            if "S3" in cfg.scenarios:
                print("  - S3 TabDDPM synthesis")
                rows.extend(run_s3(task_ctx, cfg, "S3"))
            if "S3_IG" in cfg.scenarios:
                print("  - S3-IG TabDDPM synthesis")
                rows.extend(run_s3(task_ctx, cfg_for_s3_ig(cfg), "S3_IG"))
            if "S3_HYBRID" in cfg.scenarios:
                print("  - S3-HYBRID TabDDPM synthesis")
                cands, _, _ = run_s3_hybrid_strict(task_ctx, cfg, "S3_HYBRID")
                rows.extend(cands)

            for r in rows:
                r.update(meta)
            all_rows.extend(rows)

    if cfg.strict_eval:
        candidate_df = pd.DataFrame(candidate_val_rows)
        selected_df = pd.DataFrame(selected_winner_rows)
        final_df = pd.DataFrame(final_test_rows)

        candidate_df.to_csv(run_dir / "candidate_val_results.csv", index=False)
        selected_df.to_csv(run_dir / "selected_winners.csv", index=False)
        final_df.to_csv(run_dir / "final_test_results.csv", index=False)
        summarize_paper_results(final_df, run_dir)

        print(f"\n[Done] strict results saved to: {run_dir}")
        return {
            "run_dir": run_dir,
            "config": run_dir / "config.json",
            "task_stats": run_dir / "task_stats.csv",
            "candidate_val_results": run_dir / "candidate_val_results.csv",
            "selected_winners": run_dir / "selected_winners.csv",
            "final_test_results": run_dir / "final_test_results.csv",
            "paper_summary_by_scenario": run_dir / "paper_summary_by_scenario.csv",
            "paper_summary_by_task": run_dir / "paper_summary_by_task.csv",
            "paper_summary_mean_std": run_dir / "paper_summary_mean_std.csv",
        }

    df = pd.DataFrame(all_rows)
    df.to_csv(run_dir / "run_results.csv", index=False)

    summary = (
        df.groupby(["task_id", "scenario", "method"], dropna=False)[SUMMARY_METRICS]
        .mean()
        .reset_index()
    )
    summary.to_csv(run_dir / "summary_by_method.csv", index=False)

    best = df.sort_values("PR_AUC", ascending=False).groupby(["task_id"], as_index=False).head(5)
    best.to_csv(run_dir / "top5_by_task_pr_auc.csv", index=False)

    print(f"\n[Done] results saved to: {run_dir}")
    return {
        "run_dir": run_dir,
        "config": run_dir / "config.json",
        "task_stats": run_dir / "task_stats.csv",
        "run_results": run_dir / "run_results.csv",
        "summary": run_dir / "summary_by_method.csv",
        "top5": run_dir / "top5_by_task_pr_auc.csv",
    }


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Seagate single-KQI binary pipeline (S0~S3)")

    p.add_argument("--data-root", default=".venv/softsensing_data_full")
    p.add_argument("--toolset", default="time-series-3")
    p.add_argument("--representation", default="mean", choices=["mean", "flatten", "delta", "summary_stats"])
    p.add_argument("--x-mmap-mode", default="r")
    p.add_argument("--use-rep-cache", dest="use_rep_cache", action="store_true")
    p.add_argument("--no-rep-cache", dest="use_rep_cache", action="store_false")
    p.set_defaults(use_rep_cache=True)
    p.add_argument("--rep-cache-dir", default="experiments/seagate_kqi/cache")
    p.add_argument("--taskwise-representation", dest="taskwise_representation", action="store_true")
    p.add_argument("--no-taskwise-representation", dest="taskwise_representation", action="store_false")
    p.set_defaults(taskwise_representation=False)

    p.add_argument("--top-k-tasks", type=int, default=3)
    p.add_argument("--task-ids", type=str, default="")
    p.add_argument("--min-valid-per-task", type=int, default=500)

    p.add_argument("--scenarios", type=str, default="S0,S1,S2,S3")
    p.add_argument("--gen-ratios", type=str, default="0.5,1.0,2.0")

    p.add_argument("--threshold-rule", default="max_mcc", choices=["max_mcc", "max_f1", "recall_target"])
    p.add_argument("--recall-target", type=float, default=0.90)
    p.add_argument("--strict-eval", dest="strict_eval", action="store_true")
    p.add_argument("--no-strict-eval", dest="strict_eval", action="store_false")
    p.set_defaults(strict_eval=False)
    p.add_argument("--selection-metric", default="PR_AUC", choices=["PR_AUC", "ROC_AUC", "Recall", "MCC", "F1", "Precision"])
    p.add_argument(
        "--selection-threshold-rule",
        default="max_mcc",
        choices=["max_mcc", "max_f1", "recall_target"],
    )
    p.add_argument("--paper-task-mode", default="explicit_or_topk", choices=["explicit_or_topk", "topk_only"])
    p.add_argument("--s1-smote", dest="s1_include_smote", action="store_true")
    p.add_argument("--no-s1-smote", dest="s1_include_smote", action="store_false")
    p.set_defaults(s1_include_smote=False)
    p.add_argument("--s1-smote-k-neighbors", type=int, default=5)
    p.add_argument("--s2-methods", type=str, default="cvae,cgan")
    p.add_argument("--cvae-latent-dim", type=int, default=32)
    p.add_argument("--cvae-hidden", type=int, default=256)
    p.add_argument("--cvae-epochs", type=int, default=40)
    p.add_argument("--cvae-batch-size", type=int, default=256)
    p.add_argument("--cvae-lr", type=float, default=1e-3)
    p.add_argument("--cvae-beta", type=float, default=1e-3)
    p.add_argument("--cgan-noise-dim", type=int, default=64)
    p.add_argument("--cgan-hidden", type=int, default=256)
    p.add_argument("--cgan-epochs", type=int, default=80)
    p.add_argument("--cgan-batch-size", type=int, default=256)
    p.add_argument("--cgan-lr", type=float, default=2e-4)
    p.add_argument("--torch-device", default="cuda")
    p.add_argument("--torch-use-amp", dest="torch_use_amp", action="store_true")
    p.add_argument("--no-torch-amp", dest="torch_use_amp", action="store_false")
    p.set_defaults(torch_use_amp=True)
    p.add_argument("--loader-num-workers", type=int, default=0)
    p.add_argument("--loader-pin-memory", dest="loader_pin_memory", action="store_true")
    p.add_argument("--no-loader-pin-memory", dest="loader_pin_memory", action="store_false")
    p.set_defaults(loader_pin_memory=True)
    p.add_argument("--gpu-resident-generators", dest="generator_gpu_resident", action="store_true")
    p.add_argument("--no-gpu-resident-generators", dest="generator_gpu_resident", action="store_false")
    p.set_defaults(generator_gpu_resident=True)
    p.add_argument("--generator-gpu-max-mb", type=int, default=2048)
    p.add_argument("--ddpm-device", default="cuda")
    p.add_argument("--ddpm-beta-start", type=float, default=1e-4)
    p.add_argument("--ddpm-beta-end", type=float, default=2e-2)
    p.add_argument("--ddpm-time-dim", type=int, default=128)
    p.add_argument("--ddpm-class-dim", type=int, default=16)
    p.add_argument("--ddpm-hidden-dim", type=int, default=256)
    p.add_argument("--ddpm-n-layers", type=int, default=4)
    p.add_argument("--ddpm-dropout", type=float, default=0.1)
    p.add_argument("--ddpm-T", type=int, default=100)
    p.add_argument("--ddpm-epochs", type=int, default=30)
    p.add_argument("--ddpm-batch-size", type=int, default=256)
    p.add_argument("--ddpm-lr", type=float, default=1e-3)
    p.add_argument(
        "--ddpm-train-mode",
        default="conditional_mixed",
        choices=["conditional_mixed", "balanced_conditional", "positive_only"],
    )
    p.add_argument("--strict-ddpm-train-modes", default="positive_only,conditional_mixed")
    p.add_argument("--ddpm-neg-multiplier", type=float, default=10.0)
    p.add_argument("--ddpm-max-neg", type=int, default=50000)
    p.add_argument("--ddpm-scale", default="quantile", choices=["standard", "quantile"])
    p.add_argument("--ddpm-core-variant", default="baseline", choices=["baseline", "tabdiff_min"])
    p.add_argument("--ddpm-featurewise-spike-scale", type=float, default=0.60)
    p.add_argument("--ddpm-featurewise-corr-scale", type=float, default=0.80)
    p.add_argument("--ddpm-sampler-correction-every", type=int, default=10)
    p.add_argument("--ddpm-sampler-correction-alpha", type=float, default=0.20)
    p.add_argument("--ddpm-sampler-corr-rank", type=int, default=4)
    p.add_argument("--s3-auto-mode-by-pos-rate", dest="s3_auto_mode_by_pos_rate", action="store_true")
    p.add_argument("--no-s3-auto-mode-by-pos-rate", dest="s3_auto_mode_by_pos_rate", action="store_false")
    p.set_defaults(s3_auto_mode_by_pos_rate=False)
    p.add_argument("--s3-positive-only-max-rate", type=float, default=0.015)
    p.add_argument(
        "--s3-synth-filter",
        default="none",
        choices=["none", "knn_pos", "disc_pos", "hybrid_pos", "knn_all_features", "ig_topk", "ig_weighted", "random_keep"],
    )
    p.add_argument("--s3-synth-filter-k", type=int, default=5)
    p.add_argument("--s3-synth-pool-multiplier", type=float, default=1.0)
    p.add_argument("--s3-ig-importance-path", default="")
    p.add_argument("--s3-ig-top-k", type=int, default=10)
    p.add_argument("--s3-ig-keep-rate", default="0.25,0.5,0.75,1.0")
    p.add_argument("--s3-hybrid-methods", default="ig_only,ig_clean,ig_smote,s1_plus_ig")
    p.add_argument("--s3-hybrid-smote-ratio", type=float, default=0.5)
    p.add_argument("--s3-hybrid-s1-ratio", type=float, default=0.5)
    p.add_argument("--s3-hybrid-clean-keep-rate", type=float, default=0.5)
    p.add_argument(
        "--s3-postprocess",
        default="none",
        choices=[
            "none",
            "clip_pos_q",
            "anchor_pos_q",
            "clip_anchor_pos_q",
            "spike_match_pos",
            "featurewise_quantile_pos",
            "corr_pair_anchor_pos",
            "corr_group_anchor_pos",
            "latent_pca_pos",
        ],
    )
    p.add_argument("--s3-postprocess-low-q", type=float, default=0.01)
    p.add_argument("--s3-postprocess-high-q", type=float, default=0.99)
    p.add_argument("--s3-postprocess-anchor-alpha", type=float, default=0.25)
    p.add_argument("--s3-postprocess-corr-top-k", type=int, default=6)
    p.add_argument("--s3-postprocess-group-top-features", type=int, default=8)
    p.add_argument("--s3-postprocess-pca-var", type=float, default=0.99)
    p.add_argument("--lgbm-device-type", default="gpu", choices=["gpu", "cpu"])
    p.add_argument("--lgbm-gpu-platform-id", type=int, default=0)
    p.add_argument("--lgbm-gpu-device-id", type=int, default=0)
    p.add_argument("--lgbm-n-estimators", type=int, default=None)
    p.add_argument("--cars-canary-probe", action="store_true")
    p.add_argument("--cars-canary-q", type=float, default=0.10)
    p.add_argument("--cars-canary-max-n", type=int, default=5000)
    p.add_argument("--cars-canary-min-n", type=int, default=20)
    p.add_argument("--cars-canary-distance-top-k", type=int, default=32)
    p.add_argument("--cars-canary-distance-max-neg", type=int, default=10000)
    p.add_argument("--cars-candidate-replay", action="store_true")
    p.add_argument("--cars-candidate-replay-include-test", action="store_true")
    p.add_argument("--cars-candidate-replay-save-predictions", action="store_true")
    p.add_argument("--cars-bw-prototype", action="store_true")
    p.add_argument("--cars-bw-variants", default="balanced")
    p.add_argument("--cars-bw-min-weight", type=float, default=0.05)
    p.add_argument("--cars-bw-max-weight", type=float, default=1.0)
    p.add_argument("--cars-bw-power", type=float, default=1.0)
    p.add_argument("--cars-bw-top-k-features", type=int, default=32)
    p.add_argument("--cars-bw-sort-synthetic", action="store_true")
    p.add_argument("--cars-bw-normalize-mean", action="store_true")

    p.add_argument("--output-root", default="experiments/seagate_kqi/outputs")
    p.add_argument("--run-name", default="")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--require-gpu", action="store_true", default=True)
    p.add_argument("--allow-cpu", action="store_true")
    p.add_argument("--quick", action="store_true", help="reduce epochs/tasks for smoke run")
    return p


def cfg_from_args(args: argparse.Namespace) -> PipelineConfig:
    cfg = PipelineConfig(
        data_root=args.data_root,
        toolset=args.toolset,
        representation=args.representation,
        x_mmap_mode=args.x_mmap_mode,
        use_rep_cache=args.use_rep_cache,
        rep_cache_dir=args.rep_cache_dir,
        taskwise_representation=args.taskwise_representation,
        seed=args.seed,
        top_k_tasks=args.top_k_tasks,
        min_valid_per_task=args.min_valid_per_task,
        scenarios=tuple(x.strip() for x in args.scenarios.split(",") if x.strip()),
        gen_ratios=parse_csv_floats(args.gen_ratios),
        threshold_rule=args.threshold_rule,
        recall_target=args.recall_target,
        strict_eval=args.strict_eval,
        selection_metric=args.selection_metric,
        selection_threshold_rule=args.selection_threshold_rule,
        paper_task_mode=args.paper_task_mode,
        s1_include_smote=args.s1_include_smote,
        s1_smote_k_neighbors=args.s1_smote_k_neighbors,
        s2_methods=parse_csv_strings(args.s2_methods),
        cvae_latent_dim=args.cvae_latent_dim,
        cvae_hidden=args.cvae_hidden,
        cvae_epochs=args.cvae_epochs,
        cvae_batch_size=args.cvae_batch_size,
        cvae_lr=args.cvae_lr,
        cvae_beta=args.cvae_beta,
        cgan_noise_dim=args.cgan_noise_dim,
        cgan_hidden=args.cgan_hidden,
        cgan_epochs=args.cgan_epochs,
        cgan_batch_size=args.cgan_batch_size,
        cgan_lr=args.cgan_lr,
        torch_device=args.torch_device,
        torch_use_amp=args.torch_use_amp,
        loader_num_workers=args.loader_num_workers,
        loader_pin_memory=args.loader_pin_memory,
        generator_gpu_resident=args.generator_gpu_resident,
        generator_gpu_resident_max_mb=args.generator_gpu_max_mb,
        ddpm_device=args.ddpm_device,
        ddpm_T=args.ddpm_T,
        ddpm_beta_start=args.ddpm_beta_start,
        ddpm_beta_end=args.ddpm_beta_end,
        ddpm_time_dim=args.ddpm_time_dim,
        ddpm_class_dim=args.ddpm_class_dim,
        ddpm_hidden_dim=args.ddpm_hidden_dim,
        ddpm_n_layers=args.ddpm_n_layers,
        ddpm_dropout=args.ddpm_dropout,
        ddpm_epochs=args.ddpm_epochs,
        ddpm_batch_size=args.ddpm_batch_size,
        ddpm_lr=args.ddpm_lr,
        ddpm_train_mode=args.ddpm_train_mode,
        strict_ddpm_train_modes=parse_csv_strings(args.strict_ddpm_train_modes),
        ddpm_neg_multiplier=args.ddpm_neg_multiplier,
        ddpm_max_neg=args.ddpm_max_neg,
        ddpm_scale=args.ddpm_scale,
        ddpm_core_variant=args.ddpm_core_variant,
        ddpm_featurewise_spike_scale=args.ddpm_featurewise_spike_scale,
        ddpm_featurewise_corr_scale=args.ddpm_featurewise_corr_scale,
        ddpm_sampler_correction_every=args.ddpm_sampler_correction_every,
        ddpm_sampler_correction_alpha=args.ddpm_sampler_correction_alpha,
        ddpm_sampler_corr_rank=args.ddpm_sampler_corr_rank,
        s3_auto_mode_by_pos_rate=args.s3_auto_mode_by_pos_rate,
        s3_positive_only_max_rate=args.s3_positive_only_max_rate,
        s3_synth_filter=args.s3_synth_filter,
        s3_synth_filter_k=args.s3_synth_filter_k,
        s3_synth_pool_multiplier=args.s3_synth_pool_multiplier,
        s3_ig_importance_path=args.s3_ig_importance_path,
        s3_ig_top_k=args.s3_ig_top_k,
        s3_ig_keep_rates=parse_csv_floats(args.s3_ig_keep_rate),
        s3_hybrid_methods=parse_csv_strings(args.s3_hybrid_methods),
        s3_hybrid_smote_ratio=args.s3_hybrid_smote_ratio,
        s3_hybrid_s1_ratio=args.s3_hybrid_s1_ratio,
        s3_hybrid_clean_keep_rate=args.s3_hybrid_clean_keep_rate,
        s3_postprocess=args.s3_postprocess,
        s3_postprocess_low_q=args.s3_postprocess_low_q,
        s3_postprocess_high_q=args.s3_postprocess_high_q,
        s3_postprocess_anchor_alpha=args.s3_postprocess_anchor_alpha,
        s3_postprocess_corr_top_k=args.s3_postprocess_corr_top_k,
        s3_postprocess_group_top_features=args.s3_postprocess_group_top_features,
        s3_postprocess_pca_var=args.s3_postprocess_pca_var,
        lgbm_device_type=args.lgbm_device_type,
        lgbm_gpu_platform_id=args.lgbm_gpu_platform_id,
        lgbm_gpu_device_id=args.lgbm_gpu_device_id,
        output_root=args.output_root,
        run_name=args.run_name,
        cars_canary_probe=args.cars_canary_probe,
        cars_canary_q=args.cars_canary_q,
        cars_canary_max_n=args.cars_canary_max_n,
        cars_canary_min_n=args.cars_canary_min_n,
        cars_canary_distance_top_k=args.cars_canary_distance_top_k,
        cars_canary_distance_max_neg=args.cars_canary_distance_max_neg,
        cars_candidate_replay=args.cars_candidate_replay,
        cars_candidate_replay_include_test=args.cars_candidate_replay_include_test,
        cars_candidate_replay_save_predictions=args.cars_candidate_replay_save_predictions,
        cars_bw_prototype=args.cars_bw_prototype,
        cars_bw_variants=parse_csv_strings(args.cars_bw_variants),
        cars_bw_min_weight=args.cars_bw_min_weight,
        cars_bw_max_weight=args.cars_bw_max_weight,
        cars_bw_power=args.cars_bw_power,
        cars_bw_top_k_features=args.cars_bw_top_k_features,
        cars_bw_sort_synthetic=args.cars_bw_sort_synthetic,
        cars_bw_normalize_mean=args.cars_bw_normalize_mean,
    )
    if args.lgbm_n_estimators is not None:
        cfg.lgbm_n_estimators = int(args.lgbm_n_estimators)

    if args.task_ids.strip():
        cfg.explicit_task_ids = parse_csv_ints(args.task_ids)
    elif cfg.paper_task_mode == "topk_only":
        cfg.explicit_task_ids = None

    if args.quick:
        cfg.top_k_tasks = 1 if cfg.explicit_task_ids is None else cfg.top_k_tasks
        cfg.cvae_epochs = 10
        cfg.cgan_epochs = 20
        cfg.ddpm_epochs = 10
        if args.lgbm_n_estimators is None:
            cfg.lgbm_n_estimators = 150

    return cfg


def main():
    parser = build_argparser()
    args = parser.parse_args()
    if args.require_gpu and not args.allow_cpu and not torch.cuda.is_available():
        raise SystemExit("GPU is not available. Use a CUDA environment or pass --allow-cpu for a local smoke run.")
    cfg = cfg_from_args(args)
    run_pipeline(cfg)


if __name__ == "__main__":
    main()
