"""고친 확산 생성기로 합성 후보 보관 자료를 다시 만든다 — 제출한 Seagate 논문(MDPI Electronics electronics-4573451) 수정 (2026-10).

보관 파이프라인(run_pipeline.py)의 함수를 그대로 쓰고, 다음만 바꾼다.
1. 생성기: 보관 생성기 자리에 혼합형 확산 생성기(augmenters_torch_ddpm_v3.TorchTabDDPMv3Augmenter)를 쓴다.
   연속 특징은 가우시안 확산(지름길 잡음 예측, T=1000), 0/1 이진 특징은 TabDDPM 처럼 다항(범주 2개) 확산으로 다룬다.
   설정은 봉인 문서가 정한 JSON 으로 받는다. 보관 실행의 --quick(10 epoch) 같은 축소는 없다.
   생성기는 학습 자료 안에서 클래스별 20% 를 떼어 두고 그 손실로 조기 종료한다(공식 검증·테스트 자료는 쓰지 않는다).
   연속·이진 구분과 분위 변환은 떼어 둔 행을 뺀 생성기 학습행으로 맞춘다.
2. 생성 뒤 처리(`build_s3_synth_pool`): 보관 코드는 역분위변환 뒤 과제 학습행 전체의 1–99% 분위로 잘랐다(1989–1991행).
   이 단계는 과제별 특징의 26–81% 를 합성행에서 상수로 만들었으므로 뺀다. 그 밖의 순서(특징별 분위 후처리 → 가중 kNN 순위 →
   남김 비율)는 보관 코드와 같다.
3. 평가: 모든 후보와 기준선 후보를 LightGBM(보관 설정)과 XGBoost 둘 다로 학습한다. 한 번 학습한 모델로 검증 MCC 최대 임계값을 고르고
   검증·테스트 지표를 함께 저장한다. 보관 자료는 방식별 검증 우승 후보에만 테스트 결과가 있었고, 그것도 다시 학습한 모델의 값이었다.
4. 후보마다 보관 파이프라인과 같은 방식으로 판별 AUC(real_vs_synth_auc)와 TSTR(tstr_pr_auc)을 기록한다(`s3_synth_diagnostics_for_view`).
5. 과제: 보관 자료의 9개 과제에 과제 4·8 을 더한 11개 (학습 자료만으로 정하는 기준 ≥500 행을 모두 통과).
   가중 kNN 의 특징 중요도는 보관 표(9개 과제)에 같은 절차로 계산한 과제 4·8 을 더한 표를 쓴다(`importance_all11_v1`).
6. 보관: 합성 표본 풀(fp16), 생성기 EMA 가중치(fp16), 생성기 진단(떼어 둔 불량의 잡음 예측 손실과 기준선, 떼어 둔 불량 기준 품질).

격자
- 기준선 후보 15개: S0, S1 무작위 복제·SMOTE × 비율 {0.25,0.5,0.75,1,1.5,2}, 무작위 줄이기 20:1·10:1.
  보관 동일예산 실행(V10)의 기준선과 같은 15개다. 주 감사 원천(quick)의 기준선은 그중 S0, 복제·SMOTE × {0.5,1,2}, 줄이기 2개의 9개였다.
- 격자 A (주 감사 원천과 같은 구성): {ig_clean, ig_smote, s1_plus_ig} × {positive_only, conditional_mixed} × 비율 {0.5,1.0}
  × 남김 {0.25,0.75}, 중요도 가중 kNN 순위, 후처리 없음, SMOTE 비율 0.5, S1 기반 비율 0.5.
- 격자 B (동일예산 보관과 같은 구성): s1_plus_ig × positive_only × 비율 6 × 남김 4, 특징별 양성 분위 후처리.
  보관 V10 은 표현 이름이 summary_stats 였지만 time-series-2 는 시간 길이가 1 이라 mean 과 같다(run_pipeline.py 402–403행).
  격자 B 의 positive_only 생성기는 격자 A 에서 학습한 것을 다시 쓰고 표본만 새로 뽑는다.

출력: experiments/seagate_kqi/outputs/revision_2026_10/<run_name>/task{t}_seed{s}/ (사례마다 임시 폴더에 쓰고 끝나면 이름을 바꾼다)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import shutil
import sys
import time
import traceback
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from lightgbm import LGBMClassifier
from scipy.stats import ks_2samp
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import QuantileTransformer
from xgboost import XGBClassifier

HERE = Path(__file__).resolve().parent
SEAGATE = HERE.parent
ROOT = SEAGATE.parents[1]
sys.path[:0] = [str(SEAGATE), str(ROOT / ".venv"), str(HERE)]

import run_pipeline as rp  # noqa: E402
from augmenters_torch_ddpm_v3 import TorchDDPMv3Config, TorchTabDDPMv3Augmenter  # noqa: E402

TASKS_DEFAULT = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
SEEDS_DEFAULT = [42, 43, 44, 45, 46]
RATIOS_B = (0.25, 0.5, 0.75, 1.0, 1.5, 2.0)
RATIOS_A = (0.5, 1.0)
KEEP_A = (0.25, 0.75)
KEEP_B = (0.25, 0.5, 0.75, 1.0)
METHODS_A = ("ig_clean", "ig_smote", "s1_plus_ig")
MODES_A = ("positive_only", "conditional_mixed")
IMPORTANCE_CSV = "experiments/seagate_kqi/outputs/revision_2026_10/importance_all11_v1/task_feature_importance.csv"
CLASSIFIERS = ("lgbm", "xgb")
OUTPUT_PATH = "noclip_mixed"  # 자르기 없음, 이진 특징은 생성기가 0/1 로 직접 만든다
GRIDS = (
    ("A", MODES_A, RATIOS_A, KEEP_A, METHODS_A, "none"),
    ("B", ("positive_only",), RATIOS_B, KEEP_B, ("s1_plus_ig",), "featurewise_quantile_pos"),
)


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _ver(mod: str) -> str:
    try:
        return __import__(mod).__version__
    except Exception as exc:  # 기록용
        return f"unavailable: {exc}"


# ── 생성기 ──────────────────────────────────────────────────────────────────
class _GenState:
    cfg: TorchDDPMv3Config = TorchDDPMv3Config()
    no_test: bool = False   # 시험 운전(pilot)에서는 test 평가를 하지 않는다
    fitted: dict = {}       # (task, seed, mode) -> 학습된 생성기와 입력 변환
    current_key: tuple = ()


class _NoArchivedGenerator:
    def __init__(self, *args, **kwargs):
        raise RuntimeError("이 실험에서는 보관 생성기를 부르지 않는다")


def split_holdout(y: np.ndarray, seed: int, frac: float, min_holdout: int) -> np.ndarray:
    """TorchTabDDPMv3Augmenter.fit 과 같은 규칙."""
    rng = np.random.default_rng(seed + 104729)
    hold = np.zeros(len(y), dtype=bool)
    if frac > 0:
        for cls in np.unique(y):
            idx = np.flatnonzero(y == cls)
            k = int(round(len(idx) * frac))
            if k >= min_holdout and len(idx) - k >= min_holdout:
                hold[rng.choice(idx, size=k, replace=False)] = True
    return hold


def split_types(X_fit: np.ndarray):
    bin_idx, levels = [], {}
    for j in range(X_fit.shape[1]):
        u = np.unique(X_fit[:, j])
        if len(u) == 2:
            bin_idx.append(j)
            levels[j] = (float(u[0]), float(u[1]))
    cont_idx = [j for j in range(X_fit.shape[1]) if j not in levels]
    return np.array(cont_idx, dtype=int), np.array(bin_idx, dtype=int), levels


def to_bin(X: np.ndarray, bin_idx, levels) -> np.ndarray:
    B = np.zeros((len(X), len(bin_idx)), dtype=np.float32)
    for k, j in enumerate(bin_idx):
        a, b = levels[j]
        B[:, k] = (np.abs(X[:, j] - b) < np.abs(X[:, j] - a)).astype(np.float32)
    return B


def fit_generator(X_ddpm: np.ndarray, y_ddpm: np.ndarray, seed: int) -> dict:
    c = _GenState.cfg
    hold = split_holdout(y_ddpm, seed, c.holdout_frac, c.min_holdout)
    cont_idx, bin_idx, levels = split_types(X_ddpm[~hold])
    qt = None
    if len(cont_idx):
        qt = QuantileTransformer(n_quantiles=min(2000, int((~hold).sum())), output_distribution="normal", random_state=seed)
        qt.fit(X_ddpm[~hold][:, cont_idx])
    Xc = qt.transform(X_ddpm[:, cont_idx]).astype(np.float32) if qt is not None else np.zeros((len(X_ddpm), 0), np.float32)
    Xb = to_bin(X_ddpm, bin_idx, levels)
    aug = TorchTabDDPMv3Augmenter(seed=seed, cfg=c)
    t0 = time.time()
    aug.fit(Xc, Xb, y_ddpm)
    if not np.array_equal(aug.holdout_mask, hold):
        raise RuntimeError("떼어 둔 행이 실행기와 생성기에서 다르다")
    hp = hold & (y_ddpm == 1)
    return {"aug": aug, "qt": qt, "cont_idx": cont_idx, "bin_idx": bin_idx, "levels": levels, "hold": hold,
            "fit_y": np.asarray(y_ddpm).copy(), "fit_shape": list(X_ddpm.shape), "fit_seconds": round(time.time() - t0, 1),
            "holdout_pos_loss": aug.denoise_loss(Xc[hp], Xb[hp], np.ones(int(hp.sum()), dtype=int), seed=1) if hp.any() else None,
            "bin_rate_fit": to_bin(X_ddpm[(~hold) & (y_ddpm == 1)], bin_idx, levels).mean(0) if len(bin_idx) else np.zeros(0)}


def sample_generator(gen: dict, n: int) -> np.ndarray:
    C, Bs = gen["aug"].sample(n, y_label=1)
    d = gen["fit_shape"][1]
    X = np.zeros((n, d), dtype=np.float64)
    if len(gen["cont_idx"]):
        X[:, gen["cont_idx"]] = gen["qt"].inverse_transform(np.clip(C, -8, 8))
    for k, j in enumerate(gen["bin_idx"]):
        a, b = gen["levels"][j]
        X[:, j] = np.where(Bs[:, k] > 0, b, a)
    gen["last_bin_rate_syn"] = Bs.mean(0) if len(gen["bin_idx"]) else np.zeros(0)
    return X


def build_s3_synth_pool_regen(task_ctx, cfg, train_mode):
    """run_pipeline.build_s3_synth_pool 과 같은 순서. 다른 점은 생성기(혼합형 v3)와 1–99% 자르기를 뺀 것뿐이다."""
    arrays = rp.prepare_s3_training_arrays(task_ctx, cfg, train_mode)
    y_train = arrays["y_train"]
    X_ddpm, y_ddpm = arrays["X_ddpm_train"], arrays["y_ddpm_train"]
    key = _GenState.current_key
    if key in _GenState.fitted:
        gen = _GenState.fitted[key]
        if gen["fit_shape"] != list(X_ddpm.shape) or not np.array_equal(gen["fit_y"], y_ddpm):
            raise RuntimeError(f"같은 키 {key} 인데 생성기 학습 자료가 다르다")
    else:
        gen = fit_generator(X_ddpm, y_ddpm, cfg.seed)
        _GenState.fitted[key] = gen
    max_n_add = int(round((y_train == 1).sum() * max(cfg.gen_ratios)))
    n_sample = int(round(max_n_add * max(1.0, float(cfg.s3_synth_pool_multiplier))))
    X_syn = sample_generator(gen, n_sample)
    X_syn = rp.postprocess_s3_synth(arrays["X_pos"], X_syn, cfg)
    return {**arrays, "X_syn_raw": X_syn, "max_n_add": int(max_n_add)}


rp.TorchTabDDPMConditionalAugmenter = _NoArchivedGenerator
rp.build_s3_synth_pool = build_s3_synth_pool_regen


# ── 분류기 ───────────────────────────────────────────────────────────────────
def train_xgb(X: np.ndarray, y: np.ndarray, seed: int) -> XGBClassifier:
    n_pos, n_neg = int((y == 1).sum()), int((y == 0).sum())
    model = XGBClassifier(n_estimators=400, learning_rate=0.05, max_depth=6, subsample=0.9, colsample_bytree=0.9,
                          tree_method="hist", device="cuda", scale_pos_weight=float(n_neg / max(n_pos, 1)),
                          random_state=seed, n_jobs=-1, verbosity=0)
    model.fit(X, y)
    return model


def evaluate(X_tr: np.ndarray, y_tr: np.ndarray, ctx: dict, cfg, seed: int, clf: str) -> dict:
    t0 = time.time()
    model = rp.train_lgbm(X_tr, y_tr, cfg, seed) if clf == "lgbm" else train_xgb(X_tr, y_tr, seed)
    p_val = model.predict_proba(ctx["X_val"])[:, 1]
    thr = rp.tune_threshold(ctx["y_val"], p_val, cfg.selection_threshold_rule, cfg.recall_target)
    val = rp.add_recall_at_fpr_caps(rp.evaluate_at_threshold(ctx["y_val"], p_val, thr), ctx["y_val"], p_val, ctx["y_val"], p_val, cfg.fpr_caps)
    out = {f"val_{k}": float(v) for k, v in val.items()}
    if not _GenState.no_test:
        p_te = model.predict_proba(ctx["X_test"])[:, 1]
        te = rp.add_recall_at_fpr_caps(rp.evaluate_at_threshold(ctx["y_test"], p_te, thr), ctx["y_val"], p_val, ctx["y_test"], p_te, cfg.fpr_caps)
        out.update({f"test_{k}": float(v) for k, v in te.items()})
    out["threshold"] = float(thr)
    out["fit_seconds"] = round(time.time() - t0, 2)
    return out


# ── 생성기 품질 진단 (학습 자료 안에서만) ────────────────────────────────────────
def _small_lgbm(seed: int) -> LGBMClassifier:
    return LGBMClassifier(n_estimators=200, learning_rate=0.05, num_leaves=15, min_child_samples=5, deterministic=True,
                          force_row_wise=True, n_jobs=8, random_state=seed, verbosity=-1)


def c2st(real: np.ndarray, syn: np.ndarray, seed: int, reps: int = 5, folds: int = 5) -> float:
    n = len(real)
    if n < 20 or len(syn) < n:
        return float("nan")
    rng = np.random.default_rng(seed)
    aucs = []
    for r in range(reps):
        S = syn[rng.choice(len(syn), size=n, replace=False)]
        X = np.vstack([real, S])
        y = np.r_[np.ones(n, dtype=int), np.zeros(n, dtype=int)]
        p = np.zeros(len(y))
        for tr, te in StratifiedKFold(folds, shuffle=True, random_state=seed + r).split(X, y):
            p[te] = _small_lgbm(seed + r).fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
        aucs.append(roc_auc_score(y, p))
    return float(np.mean(aucs))


def generator_fidelity(X_pool: np.ndarray, arrays: dict, gen: dict, ctx: dict, cfg, seed: int) -> dict:
    Xd, yd = arrays["X_ddpm_train"], arrays["y_ddpm_train"]
    if len(Xd) != len(gen["hold"]) or not np.array_equal(gen["fit_y"], yd):
        raise RuntimeError("생성기 학습 자료와 진단 자료의 행 순서가 다르다")
    hold = gen["hold"]
    pos_fit, pos_hold = Xd[(yd == 1) & ~hold], Xd[(yd == 1) & hold]
    out = {"n_pos_fit": int(len(pos_fit)), "n_pos_hold": int(len(pos_hold))}
    if len(pos_hold) < 20:
        return out
    qt = QuantileTransformer(n_quantiles=min(2000, int((~hold).sum())), output_distribution="normal", random_state=seed)
    qt.fit(Xd[~hold])
    s_syn, s_fit, s_hold = qt.transform(X_pool), qt.transform(pos_fit), qt.transform(pos_hold)
    nn_fit = NearestNeighbors(n_neighbors=1).fit(s_fit)
    d_syn, d_hold = nn_fit.kneighbors(s_syn)[0][:, 0], nn_fit.kneighbors(s_hold)[0][:, 0]
    # 학습 음성 80% + 합성(또는 실제 학습) 불량으로 학습, 나머지 음성 20% + 떼어 둔 불량으로 평가
    neg_all = ctx["X_train"][ctx["y_train"] == 0]
    perm = np.random.default_rng(20261007 + int(ctx["task_id"])).permutation(len(neg_all))
    n_te = int(round(0.2 * len(neg_all)))
    neg_te, neg_tr = neg_all[perm[:n_te]], neg_all[perm[n_te:]]
    lcfg = replace(cfg, lgbm_device_type="cpu")

    def tstr_int(P_src):
        rng = np.random.default_rng(seed)
        P = P_src[rng.choice(len(P_src), size=len(pos_fit), replace=len(P_src) < len(pos_fit))]
        m = rp.train_lgbm(np.vstack([neg_tr, P]), np.r_[np.zeros(len(neg_tr), dtype=int), np.ones(len(P), dtype=int)], lcfg, seed)
        Xe = np.vstack([neg_te, pos_hold])
        ye = np.r_[np.zeros(len(neg_te), dtype=int), np.ones(len(pos_hold), dtype=int)]
        return float(average_precision_score(ye, m.predict_proba(Xe)[:, 1]))

    with np.errstate(invalid="ignore", divide="ignore"):
        ca = np.nan_to_num(np.corrcoef(s_syn, rowvar=False))
        cb = np.nan_to_num(np.corrcoef(s_hold, rowvar=False))
    out.update({
        "c2st_hold": c2st(pos_hold, X_pool, seed),
        "ks_hold": float(np.mean([ks_2samp(X_pool[:, j], pos_hold[:, j]).statistic for j in range(X_pool.shape[1])])),
        "corr_err_hold": float(np.linalg.norm(ca - cb) / max(np.linalg.norm(cb), 1e-12)),
        "dcr_ratio": float(np.median(d_syn) / max(np.median(d_hold), 1e-12)),
        "mem_p05": float(np.mean(d_syn < np.quantile(d_hold, 0.05))),
        "tstr_internal_syn": tstr_int(X_pool),
        "tstr_internal_real": tstr_int(pos_fit),
        "n_const_syn": int(((X_pool.max(0) == X_pool.min(0)) & (pos_fit.max(0) > pos_fit.min(0))).sum()),
    })
    return out


# ── 사례 하나 ────────────────────────────────────────────────────────────────
def run_case(task: int, seed: int, data: dict, base_cfg, out_dir: Path, say) -> None:
    cfg = replace(base_cfg, seed=seed)
    mtr, ytr = rp.extract_task_binary(data["y_tr"], task)
    mva, yva = rp.extract_task_binary(data["y_va"], task)
    mte, yte = (None, None) if _GenState.no_test else rp.extract_task_binary(data["y_te"], task)
    Xtr = np.asarray(data["rep_tr"][mtr], dtype=np.float32)
    prep = rp.TabularPreprocessor(drop_constant=True, imputer_strategy="mean")
    prep.fit(Xtr)
    ctx = {"X_train": prep.transform_classifier(Xtr), "y_train": ytr,
           "X_val": prep.transform_classifier(data["rep_va"][mva]), "y_val": yva, "task_id": int(task)}
    if not _GenState.no_test:
        ctx.update({"X_test": prep.transform_classifier(data["rep_te"][mte]), "y_test": yte})
    rows = []
    meta_case = {"task_id": task, "seed": seed, "n_train": int(len(ytr)), "n_pos_train": int(ytr.sum()),
                 "n_val": int(len(yva)), "n_pos_val": int(yva.sum()),
                 "n_test": None if _GenState.no_test else int(len(yte)), "n_pos_test": None if _GenState.no_test else int(yte.sum()),
                 "n_features": int(ctx["X_train"].shape[1]), "output_path": OUTPUT_PATH}
    say(f"과제 {task} seed {seed}: 학습 {meta_case['n_train']}(불량 {meta_case['n_pos_train']}), 특징 {meta_case['n_features']}")

    def add_row(meta: dict, X_aug: np.ndarray, y_aug: np.ndarray):
        for clf in CLASSIFIERS:
            rows.append({**meta, "classifier": clf, "n_train_aug": int(len(y_aug)), "n_pos_aug": int(y_aug.sum()),
                         **evaluate(X_aug, y_aug, ctx, cfg, seed, clf)})

    # 기준선 후보 15개
    add_row({"grid": "ref", "scenario": "S0", "method": "none", "ratio": 0.0}, ctx["X_train"], ctx["y_train"])
    for method in ("random_over", "smote"):
        for r in RATIOS_B:
            Xr, yr, _ = rp.build_s1_resampled_train(ctx, cfg, method, float(r))
            add_row({"grid": "ref", "scenario": "S1", "method": method, "ratio": float(r)}, Xr, yr)
    for neg_pos in (20.0, 10.0):
        method = f"random_under_{int(neg_pos)}to1"
        Xr, yr, _ = rp.build_s1_resampled_train(ctx, cfg, method, float("nan"))
        add_row({"grid": "ref", "scenario": "S1", "method": method, "ratio": float("nan")}, Xr, yr)
    say("  기준선 15개 끝")

    diags, pools, holds = [], {}, {}
    for grid, modes, ratios, keeps, methods, post in GRIDS:
        gcfg = rp.cfg_for_s3_ig(replace(cfg, gen_ratios=ratios, s3_ig_keep_rates=keeps, s3_hybrid_methods=methods,
                                        s3_postprocess=post, s3_synth_filter="none"))
        for mode in modes:
            _GenState.current_key = (task, seed, mode)
            cache = rp.build_s3_synth_cache(ctx, gcfg, mode)
            gen = _GenState.fitted[(task, seed, mode)]
            aug = gen["aug"]
            pools[f"{grid}_{mode}"] = cache["X_syn"].astype(np.float16)
            holds[mode] = np.flatnonzero(gen["hold"]).tolist()
            arrays = rp.prepare_s3_training_arrays(ctx, gcfg, mode)
            fid = generator_fidelity(np.asarray(cache["X_syn"], dtype=np.float32), arrays, gen, ctx, gcfg, seed)
            br_fit, br_syn = gen["bin_rate_fit"], gen.get("last_bin_rate_syn", np.zeros(0))
            d = {"grid": grid, "mode": mode, "pool_size": int(len(cache["X_syn"])),
                 "pool_real_vs_synth_auc": cache["real_vs_synth_auc"], "pool_tstr_pr_auc": cache["tstr_pr_auc"],
                 "fit_seconds": gen["fit_seconds"], "fit_rows": gen["fit_shape"][0], "fit_pos": int((gen["fit_y"] == 1).sum()),
                 "n_cont": int(len(gen["cont_idx"])), "n_bin": int(len(gen["bin_idx"])),
                 "best_step": aug.best["step"], "steps_run": aug.steps_run, "best_holdout_loss": aug.best["holdout_loss"],
                 "holdout_pos_loss": gen["holdout_pos_loss"], "loss_log": aug.loss_log,
                 "bin_rate_fit_mean": float(br_fit.mean()) if len(br_fit) else None,
                 "bin_rate_syn_mean": float(br_syn.mean()) if len(br_syn) else None,
                 "bin_rate_abs_err": float(np.abs(br_fit - br_syn).mean()) if len(br_fit) and len(br_syn) == len(br_fit) else None, **fid}
            diags.append(d)
            for keep in keeps:
                X_view, _ = rp.slice_s3_synth_by_keep_rate(cache, keep)
                X_clean, _ = rp.clean_s3_synth_by_pos_neg_margin(cache["X_pos"], cache["X_neg"], X_view, gcfg, task, seed)
                for method in methods:
                    Xm = X_clean if method == "ig_clean" else X_view
                    q_auc, tstr = rp.s3_synth_diagnostics_for_view(cache, Xm, ctx, gcfg)
                    for r in ratios:
                        X_aug, y_aug, hm = rp.build_hybrid_augmented_train(ctx["X_train"], ctx["y_train"], Xm, float(r), gcfg, method, seed)
                        add_row({"grid": grid, "scenario": "S3_HYBRID", "method": method, "ratio": float(r), "keep_rate": float(keep),
                                 "ddpm_train_mode": mode, "s3_postprocess": post, "n_added": int(hm["hybrid_total_added"]),
                                 "hybrid_base_added": int(hm["hybrid_base_added"]), "hybrid_smote_added": int(hm["hybrid_smote_added"]),
                                 "pool_view_size": int(len(Xm)), "with_replacement": bool(hm["hybrid_base_added"] > len(Xm)),
                                 "real_vs_synth_auc": float(q_auc), "tstr_pr_auc": float(tstr)}, X_aug, y_aug)
            hl = gen["holdout_pos_loss"] or {}
            say(f"  격자 {grid} {mode}: 생성기 {gen['fit_seconds']}초 (최적 {aug.best['step']}/{aug.steps_run} 단계), 떼어 둔 불량 손실 "
                f"연속 {hl.get('cont_model', float('nan')):.3f}/독립 {hl.get('cont_gauss_indep', float('nan')):.3f}, "
                f"이진 {hl.get('bin_model', float('nan')):.3f}/주변 {hl.get('bin_marginal', float('nan')):.3f}, C2ST {fid.get('c2st_hold', float('nan')):.3f}, "
                f"TSTR내부 {fid.get('tstr_internal_syn', float('nan')):.4f}/{fid.get('tstr_internal_real', float('nan')):.4f}")

    for mode in MODES_A:
        gen = _GenState.fitted[(task, seed, mode)]
        torch.save({k: v.half() for k, v in gen["aug"].ema.state_dict().items()}, out_dir / f"generator_{mode}_ema_fp16.pt")
        tf = {"cont_idx": gen["cont_idx"].tolist(), "bin_idx": gen["bin_idx"].tolist(),
              "levels": {str(k): v for k, v in gen["levels"].items()},
              "std_mu": None if gen["aug"].mu is None else gen["aug"].mu.tolist(),
              "std_sd": None if gen["aug"].sd is None else gen["aug"].sd.tolist()}
        (out_dir / f"generator_{mode}_input_transform.json").write_text(json.dumps(tf) + "\n", encoding="utf-8")
    np.savez_compressed(out_dir / "synthetic_pools_fp16.npz", **pools)
    pd.DataFrame(rows).to_csv(out_dir / "candidates.csv", index=False)
    (out_dir / "case.json").write_text(json.dumps({**meta_case, "generator_diagnostics": diags, "generator_holdout_rows": holds},
                                                  ensure_ascii=False, indent=2, default=float) + "\n", encoding="utf-8")
    for k in [k for k in _GenState.fitted if k[0] == task and k[1] == seed]:
        del _GenState.fitted[k]
    torch.cuda.empty_cache()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-name", required=True)
    ap.add_argument("--generator-config", required=True, help="TorchDDPMv3Config 값을 담은 JSON 파일")
    ap.add_argument("--lgbm-device", default="cpu", choices=["gpu", "cpu"])
    ap.add_argument("--tasks", default=",".join(map(str, TASKS_DEFAULT)))
    ap.add_argument("--seeds", default=",".join(map(str, SEEDS_DEFAULT)))
    ap.add_argument("--no-test", action="store_true", help="시험 운전: test 자료를 읽지도 평가하지도 않는다")
    args = ap.parse_args()
    out_root = SEAGATE / "outputs/revision_2026_10" / args.run_name
    out_root.mkdir(parents=True, exist_ok=True)
    gen_cfg = json.loads(Path(args.generator_config).read_text(encoding="utf-8"))
    _GenState.cfg = TorchDDPMv3Config(**gen_cfg)
    _GenState.no_test = bool(args.no_test)
    log = open(out_root / "run.log", "a", encoding="utf-8")

    def say(msg: str) -> None:
        line = f"[{datetime.now().strftime('%m-%d %H:%M:%S')}] {msg}"
        print(line, flush=True)
        log.write(line + "\n")
        log.flush()

    base_cfg = rp.PipelineConfig(
        data_root=".venv/softsensing_data_full", toolset="time-series-2", representation="mean", use_rep_cache=True,
        threshold_rule="max_mcc", selection_threshold_rule="max_mcc", selection_metric="PR_AUC", recall_target=0.9,
        s1_include_smote=True, s1_smote_k_neighbors=5, s1_under_neg_pos_ratios=(20.0, 10.0),
        strict_ddpm_train_modes=MODES_A, ddpm_scale="quantile", ddpm_neg_multiplier=3.0, ddpm_max_neg=100000,
        ddpm_core_variant="baseline", s3_positive_only_max_rate=0.015, s3_synth_filter="none", s3_synth_filter_k=5,
        s3_synth_pool_multiplier=1.0, s3_ig_importance_path=IMPORTANCE_CSV, s3_ig_top_k=10,
        s3_hybrid_smote_ratio=0.5, s3_hybrid_s1_ratio=0.5, s3_hybrid_clean_keep_rate=0.5,
        s3_postprocess_low_q=0.01, s3_postprocess_high_q=0.99, lgbm_n_estimators=400, lgbm_device_type=args.lgbm_device,
        ddpm_device="cuda", torch_device="cuda")
    code_now = {"run_regen_archive_v1.py": sha(Path(__file__).resolve()),
                "augmenters_torch_ddpm_v2.py": sha(HERE / "augmenters_torch_ddpm_v2.py"),
                "augmenters_torch_ddpm_v3.py": sha(HERE / "augmenters_torch_ddpm_v3.py"),
                "run_pipeline.py": sha(SEAGATE / "run_pipeline.py"),
                "importance_csv": sha(ROOT / IMPORTANCE_CSV),
                "generator_config": sha(Path(args.generator_config))}
    manifest = out_root / "manifest.json"
    if not manifest.exists():
        manifest.write_text(json.dumps({
            "schema_id": "seagate-revision-regen-archive-v1", "lineage": "Seagate negative-audit (제출한 논문, MDPI electronics-4573451)",
            "started_utc": datetime.now(timezone.utc).isoformat(), "argv": sys.argv, "base_cfg": asdict(base_cfg),
            "generator_cfg": gen_cfg, "output_path": OUTPUT_PATH, "lgbm_device": args.lgbm_device, "no_test": bool(args.no_test),
            "grids": {g[0]: {"modes": g[1], "ratios": g[2], "keep": g[3], "methods": g[4], "post": g[5]} for g in GRIDS},
            "classifiers": {"lgbm": f"run_pipeline.train_lgbm (400 trees, lr 0.05, 63 leaves, colsample 0.9, subsample 0.9 without "
                                    f"subsample_freq, scale_pos_weight=neg/pos, device {args.lgbm_device})",
                            "xgb": "XGBClassifier(400 trees, lr 0.05, depth 6, subsample 0.9, colsample 0.9, hist, cuda, scale_pos_weight=neg/pos)"},
            "code_sha256": code_now,
            "environment": {"python": sys.version.split()[0], "torch": torch.__version__, "torch_cuda": torch.version.cuda,
                            "numpy": np.__version__, "pandas": pd.__version__, "scikit_learn": _ver("sklearn"),
                            "imbalanced_learn": _ver("imblearn"), "lightgbm": _ver("lightgbm"), "xgboost": _ver("xgboost"),
                            "scipy": _ver("scipy"), "platform": platform.platform(), "processor": platform.processor(),
                            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None},
        }, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    else:
        prev = json.loads(manifest.read_text(encoding="utf-8"))
        if prev["code_sha256"] != code_now or prev["generator_cfg"] != gen_cfg \
                or prev.get("lgbm_device") != args.lgbm_device or prev.get("no_test") != bool(args.no_test):
            raise RuntimeError("이어 달리기인데 코드·설정이 처음 실행과 다르다")

    X_train, X_val, X_test, y_tr, y_va, y_te = rp.load_toolset_arrays(base_cfg.data_root, base_cfg.toolset, base_cfg.x_mmap_mode)
    data = {"rep_tr": rp.load_cached_representation(base_cfg, "train"),
            "rep_va": rp.build_representation(X_val, "mean").astype(np.float32),
            "rep_te": None if args.no_test else rp.build_representation(X_test, "mean").astype(np.float32),
            "y_tr": y_tr, "y_va": y_va, "y_te": None if args.no_test else y_te}
    if data["rep_tr"] is None:
        raise RuntimeError("학습 평균 표현 캐시가 없다")
    tasks = [int(x) for x in args.tasks.split(",")]
    seeds = [int(x) for x in args.seeds.split(",")]
    for task in tasks:
        for seed in seeds:
            final = out_root / f"task{task}_seed{seed}"
            if (final / "case.json").exists():
                say(f"과제 {task} seed {seed}: 이미 끝남, 건너뜀")
                continue
            tmp = out_root / f"_tmp_task{task}_seed{seed}"
            if tmp.exists():
                shutil.rmtree(tmp)
            tmp.mkdir()
            t0 = time.time()
            try:
                run_case(task, seed, data, base_cfg, tmp, say)
            except Exception:
                (tmp / "error.txt").write_text(traceback.format_exc(), encoding="utf-8")
                say(f"과제 {task} seed {seed}: 오류 — {tmp / 'error.txt'}")
                raise
            tmp.rename(final)
            say(f"과제 {task} seed {seed}: 끝 ({round((time.time() - t0) / 60, 1)}분)")
    say("전체 끝")
    return 0


if __name__ == "__main__":
    sys.exit(main())
