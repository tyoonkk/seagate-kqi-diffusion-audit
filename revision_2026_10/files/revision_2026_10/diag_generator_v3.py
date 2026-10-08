"""생성기 진단 3차 — 제출한 Seagate 논문(MDPI Electronics, electronics-4573451) 수정 실험 (2026-10-07).

목적: 새 확산 생성기(v2.1)를 보관 파이프라인에 끼우기 전에 "생성 뒤 처리 경로"를 정한다.
2차 진단(diag_generator_v2.py)에서 v2.1 은 떼어 둔 불량의 잡음 예측 손실이 자명한 기준선보다 확실히 낮아(학습됨),
특징 사이 상관과 거리(DCR)도 보관 생성기보다 나았다. 그러나 합성 불량이 실제 불량과 여전히 쉽게 구별됐다.
원고 방법 대조에서, 보관 파이프라인이 표본을 전체 학습행의 1–99% 분위로 자르고(run_pipeline.py 1989–1991),
이 때문에 과제별 특징의 26–81% 가 합성행에서 상수로 붕괴한다는 사실이 확인됐다. 이번 진단은 다음을 비교한다.

처리 경로 (생성기 출력 S 는 분위 정규 공간)
- clip       : 보관 경로. qt 역변환 뒤 과제 학습행 전체의 1–99% 분위로 자른다.
- noclip     : qt 역변환만 한다 (역변환 자체가 학습행의 관측 범위 안으로 값을 보낸다).
- noclip_snap: noclip 뒤, 생성기 학습행에서 값이 정확히 두 개뿐인 특징을 가까운 값으로 맞춘다.

생성기: archived(보관 설정, 200 epoch), v21(은닉 1024, 최대 30,000 단계, 떼어 둔 손실 조기 종료).
고전 기준: 실제 학습 불량(상한), SMOTE(k=5), 가우시안 코풀라(분위 정규 공간 다변량 정규), 독립 주변분포(열별 재표집).

측정 (모두 학습 자료 안에서만, 공식 검증·테스트 자료는 쓰지 않는다)
- c2st_hold: 떼어 둔 실제 불량 n_h 행 대 합성 n_h 행, LightGBM(CPU, 결정적) 5겹 교차검증 AUC, 합성 부분표본 5번 평균.
- c2st_fit : 생성기 학습 불량 대 합성, 같은 방식.
- ks, corr_err(분위 공간), dcr_ratio(분위 공간, 합성→학습 불량 최근접 거리 중앙값 / 떼어 둔 불량→학습 불량), mem_p05.
- tstr_int: 학습 음성의 80% + 합성 불량(학습 불량 수만큼)으로 학습, 학습 음성의 나머지 20% + 떼어 둔 불량으로 평가한 PR-AUC.
  (conditional_mixed 생성기는 학습 음성 일부를 조건 0 으로 보았으므로 평가 음성과 일부 겹칠 수 있다. 합성은 불량만 쓰므로 영향은 작다.)
- n_const_syn: 실제 학습 불량에서는 값이 변하는데 합성에서는 상수인 특징 수. frac_fractional_binary: 이진 특징 값 중 두 값이 아닌 비율.

출력: experiments/seagate_kqi/outputs/revision_2026_10/generator_diag_v3/ (diag.json, run.log, 끝에 diag.sha256). 논문 결과로 쓰지 않는다.
"""

from __future__ import annotations

import contextlib
import hashlib
import io
import json
import sys
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from lightgbm import LGBMClassifier
from scipy.stats import ks_2samp
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import QuantileTransformer

HERE = Path(__file__).resolve().parent
SEAGATE = HERE.parent
ROOT = SEAGATE.parents[1]
sys.path[:0] = [str(SEAGATE), str(ROOT / ".venv"), str(HERE)]

import run_pipeline as rp  # noqa: E402
from augmenters_torch_ddpm import TorchDDPMConfig, TorchTabDDPMConditionalAugmenter  # noqa: E402
from augmenters_torch_ddpm_v2 import TorchDDPMv2Config, TorchTabDDPMv2Augmenter  # noqa: E402

OUT = SEAGATE / "outputs/revision_2026_10/generator_diag_v3"
TASKS = [0, 5, 10]
MODES = ["positive_only", "conditional_mixed"]
SEED = 42
HOLD = 0.2
V21 = dict(T=1000, beta_start=1e-4, beta_end=2e-2, hidden_dim=1024, n_blocks=4, time_dim=128, dropout=0.1,
           train_steps=30000, batch_size=256, lr=5e-4, weight_decay=1e-4, warmup_steps=500, grad_clip=1.0,
           ema_decay=0.999, log_every=500, sample_batch=4096, device="cuda", standardize=True, holdout_frac=HOLD,
           min_holdout=20, eval_every=250, patience=8, use_bf16=True)
C2ST_REPS = 5
C2ST_FOLDS = 5


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def archived_cfg() -> TorchDDPMConfig:
    return TorchDDPMConfig(T=200, beta_start=1e-4, beta_end=0.01, time_dim=128, class_dim=16, hidden_dim=384, n_layers=6,
                           dropout=0.1, epochs=200, batch_size=256, lr=1e-3, use_amp=True, device="cuda")


def split_holdout(y: np.ndarray, seed: int) -> np.ndarray:
    """v2.1 의 fit 과 같은 규칙 (클래스별 20%, 각 쪽 최소 20행)."""
    rng = np.random.default_rng(seed + 104729)
    hold = np.zeros(len(y), dtype=bool)
    for cls in np.unique(y):
        idx = np.flatnonzero(y == cls)
        k = int(round(len(idx) * HOLD))
        if k >= 20 and len(idx) - k >= 20:
            hold[rng.choice(idx, size=k, replace=False)] = True
    return hold


def small_lgbm(seed: int) -> LGBMClassifier:
    return LGBMClassifier(n_estimators=200, learning_rate=0.05, num_leaves=15, min_child_samples=5, subsample=1.0,
                          colsample_bytree=1.0, deterministic=True, force_row_wise=True, n_jobs=8, random_state=seed,
                          verbosity=-1)


def c2st(real: np.ndarray, syn: np.ndarray, seed: int) -> tuple[float, float]:
    """실제 n 행 대 합성 n 행(부분표본) 5겹 교차검증 AUC 를 C2ST_REPS 번 평균한다. 0.5 에 가까울수록 구별이 어렵다."""
    n = len(real)
    if n < 20 or len(syn) < n:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    aucs = []
    for r in range(C2ST_REPS):
        S = syn[rng.choice(len(syn), size=n, replace=False)]
        X = np.vstack([real, S])
        y = np.r_[np.ones(n, dtype=int), np.zeros(n, dtype=int)]
        p = np.zeros(len(y))
        for tr, te in StratifiedKFold(C2ST_FOLDS, shuffle=True, random_state=seed + r).split(X, y):
            p[te] = small_lgbm(seed + r).fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
        aucs.append(roc_auc_score(y, p))
    return float(np.mean(aucs)), float(np.std(aucs))


def tstr_internal(neg_tr, neg_te, pos_syn, pos_hold, n_pos, seed) -> float:
    rng = np.random.default_rng(seed)
    P = pos_syn[rng.choice(len(pos_syn), size=n_pos, replace=len(pos_syn) < n_pos)]
    X = np.vstack([neg_tr, P])
    y = np.r_[np.zeros(len(neg_tr), dtype=int), np.ones(n_pos, dtype=int)]
    cfg = rp.PipelineConfig(lgbm_device_type="cpu")
    model = rp.train_lgbm(X, y, cfg, seed)
    Xe = np.vstack([neg_te, pos_hold])
    ye = np.r_[np.zeros(len(neg_te), dtype=int), np.ones(len(pos_hold), dtype=int)]
    return float(average_precision_score(ye, model.predict_proba(Xe)[:, 1]))


def corr_err(a: np.ndarray, b: np.ndarray) -> float:
    with np.errstate(invalid="ignore", divide="ignore"):
        ca, cb = np.nan_to_num(np.corrcoef(a, rowvar=False)), np.nan_to_num(np.corrcoef(b, rowvar=False))
    return float(np.linalg.norm(ca - cb) / max(np.linalg.norm(cb), 1e-12))


def binary_levels(X: np.ndarray) -> dict:
    lv = {}
    for j in range(X.shape[1]):
        u = np.unique(X[:, j])
        if len(u) == 2:
            lv[j] = (float(u[0]), float(u[1]))
    return lv


def snap(X: np.ndarray, levels: dict) -> np.ndarray:
    Y = X.copy()
    for j, (a, b) in levels.items():
        Y[:, j] = np.where(np.abs(Y[:, j] - a) <= np.abs(Y[:, j] - b), a, b)
    return Y


def measure(X_syn, pos_fit, pos_hold, qt, levels, neg_tr, neg_te, seed) -> dict:
    s_syn, s_fit, s_hold = qt.transform(X_syn), qt.transform(pos_fit), qt.transform(pos_hold)
    nn_fit = NearestNeighbors(n_neighbors=1).fit(s_fit)
    d_syn = nn_fit.kneighbors(s_syn)[0][:, 0]
    d_hold = nn_fit.kneighbors(s_hold)[0][:, 0]
    var_fit = pos_fit.max(0) > pos_fit.min(0)
    const_syn = X_syn.max(0) == X_syn.min(0)
    if levels:
        cols = np.array(sorted(levels))
        a = np.array([levels[j][0] for j in cols])
        b = np.array([levels[j][1] for j in cols])
        V = X_syn[:, cols]
        frac_frac = float(np.mean(~(np.isclose(V, a) | np.isclose(V, b))))
    else:
        frac_frac = float("nan")
    h_auc, h_sd = c2st(pos_hold, X_syn, seed)
    f_auc, f_sd = c2st(pos_fit, X_syn, seed + 100)
    return {
        "c2st_hold": h_auc, "c2st_hold_sd": h_sd, "c2st_fit": f_auc, "c2st_fit_sd": f_sd,
        "ks": float(np.mean([ks_2samp(X_syn[:, j], pos_hold[:, j]).statistic for j in range(X_syn.shape[1])])),
        "corr_err": corr_err(s_syn, s_hold),
        "dcr_ratio": float(np.median(d_syn) / max(np.median(d_hold), 1e-12)),
        "mem_p05": float(np.mean(d_syn < np.quantile(d_hold, 0.05))),
        "tstr_int": tstr_internal(neg_tr, neg_te, X_syn, pos_hold, len(pos_fit), seed),
        "n_const_syn": int((const_syn & var_fit).sum()),
        "frac_fractional_binary": frac_frac,
        "n_syn": int(len(X_syn)),
    }


def main() -> int:
    if OUT.exists():
        print(f"출력 폴더가 이미 있다: {OUT}")
        return 1
    OUT.mkdir(parents=True)
    log = open(OUT / "run.log", "w", encoding="utf-8")

    def say(msg: str) -> None:
        line = f"[{datetime.now().strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        log.write(line + "\n")
        log.flush()

    cfg = rp.PipelineConfig(data_root=".venv/softsensing_data_full", toolset="time-series-2", representation="mean",
                            use_rep_cache=True, seed=SEED, ddpm_neg_multiplier=3.0, ddpm_max_neg=100000)
    X_train, X_val, X_test, y_tr_raw, y_va_raw, _ = rp.load_toolset_arrays(cfg.data_root, cfg.toolset, cfg.x_mmap_mode)
    rep_tr = rp.load_cached_representation(cfg, "train")
    res = {"schema_id": "seagate-revision-generator-diag-v3", "lineage": "Seagate negative-audit (제출한 논문)",
           "tasks": TASKS, "modes": MODES, "seed": SEED, "holdout_frac": HOLD, "v21": V21, "archived_cfg": asdict(archived_cfg()),
           "c2st": {"reps": C2ST_REPS, "folds": C2ST_FOLDS, "model": "LGBM 200 trees lr 0.05 leaves 15 min_child 5 deterministic CPU"},
           "rows": [],
           "code_sha256": {"augmenters_torch_ddpm_v2.py": sha(HERE / "augmenters_torch_ddpm_v2.py"),
                           "augmenters_torch_ddpm.py": sha(ROOT / ".venv/augmenters_torch_ddpm.py"),
                           "diag_generator_v3.py": sha(Path(__file__).resolve()),
                           "run_pipeline.py": sha(SEAGATE / "run_pipeline.py")}}
    for task in TASKS:
        mtr, ytr = rp.extract_task_binary(y_tr_raw, task)
        Xtr = np.asarray(rep_tr[mtr], dtype=np.float32)
        prep = rp.TabularPreprocessor(drop_constant=True, imputer_strategy="mean")
        prep.fit(Xtr)
        ctx = {"X_train": prep.transform_classifier(Xtr), "y_train": ytr, "task_id": task}
        lo, hi = np.quantile(ctx["X_train"], 0.01, axis=0), np.quantile(ctx["X_train"], 0.99, axis=0)
        neg_all = ctx["X_train"][ytr == 0]
        rng_neg = np.random.default_rng(20261007 + task)
        perm = rng_neg.permutation(len(neg_all))
        n_te = int(round(0.2 * len(neg_all)))
        neg_te, neg_tr = neg_all[perm[:n_te]], neg_all[perm[n_te:]]
        for mode in MODES:
            arr = rp.prepare_s3_training_arrays(ctx, rp.PipelineConfig(**{**asdict(cfg), "ddpm_train_mode": mode}), mode)
            Xd, yd = arr["X_ddpm_train"], arr["y_ddpm_train"]
            hold = split_holdout(yd, SEED)
            pos_fit, pos_hold = Xd[(yd == 1) & ~hold], Xd[(yd == 1) & hold]
            qt = QuantileTransformer(n_quantiles=min(2000, int((~hold).sum())), output_distribution="normal", random_state=SEED)
            qt.fit(Xd[~hold])
            Xs = qt.transform(Xd).astype(np.float32)
            levels = binary_levels(Xd[~hold])
            n_syn = int(min(6000, max(500, 2 * len(pos_fit))))
            say(f"과제 {task} {mode}: 생성기 학습 {int((~hold).sum())}행 (불량 {len(pos_fit)}), 떼어 둔 불량 {len(pos_hold)}, "
                f"특징 {Xd.shape[1]}, 이진 특징 {len(levels)}, 합성 {n_syn}")
            variants = {}
            # 보관 생성기
            t0 = time.time()
            a = TorchTabDDPMConditionalAugmenter(seed=SEED, cfg=archived_cfg())
            with contextlib.redirect_stdout(io.StringIO()):
                a.fit(Xs[~hold], yd[~hold])
            S = np.clip(a.sample(n_syn, y_label=1), -8, 8)
            inv = qt.inverse_transform(S)
            variants["archived|clip"] = (np.clip(inv, lo, hi), {"fit_seconds": round(time.time() - t0, 1)})
            variants["archived|noclip"] = (inv, {})
            # v2.1
            t0 = time.time()
            b = TorchTabDDPMv2Augmenter(seed=SEED, cfg=TorchDDPMv2Config(**V21))
            b.fit(Xs, yd)
            if not np.array_equal(b.holdout_mask, hold):
                raise RuntimeError("떼어 둔 행이 진단과 생성기에서 다르다")
            fit_s = round(time.time() - t0, 1)
            hl = b.denoise_loss(Xs[hold & (yd == 1)], np.ones(int((hold & (yd == 1)).sum()), dtype=int), seed=1)
            S = np.clip(b.sample(n_syn, y_label=1), -8, 8)
            inv = qt.inverse_transform(S)
            extra = {"fit_seconds": fit_s, "best_step": b.best["step"], "steps_run": b.steps_run,
                     "best_holdout_loss": b.best["holdout_loss"], "holdout_pos_loss": hl}
            variants["v21|clip"] = (np.clip(inv, lo, hi), extra)
            variants["v21|noclip"] = (inv, {})
            variants["v21|noclip_snap"] = (snap(inv, levels), {})
            say(f"  v21 학습 {fit_s}초, 최적 단계 {b.best['step']}/{b.steps_run}, 떼어 둔 불량 손실 "
                f"{ {k: round(v, 3) for k, v in hl.items()} }")
            # 고전 기준
            rng = np.random.default_rng(SEED)
            variants["ref|real_fit"] = (pos_fit, {})
            k = min(5, len(pos_fit) - 1)
            nn = NearestNeighbors(n_neighbors=k + 1).fit(pos_fit)
            base = rng.integers(0, len(pos_fit), n_syn)
            nb = nn.kneighbors(pos_fit[base])[1][np.arange(n_syn), rng.integers(1, k + 1, n_syn)]
            lam = rng.random((n_syn, 1))
            variants["ref|smote"] = (pos_fit[base] + lam * (pos_fit[nb] - pos_fit[base]), {})
            Z = qt.transform(pos_fit)
            mu, cov = Z.mean(0), np.cov(Z, rowvar=False) + 1e-6 * np.eye(Z.shape[1])
            variants["ref|gauss_copula"] = (qt.inverse_transform(np.clip(rng.multivariate_normal(mu, cov, size=n_syn, method="eigh"), -8, 8)), {})
            variants["ref|indep_marg"] = (np.column_stack([pos_fit[rng.integers(0, len(pos_fit), n_syn), j] for j in range(pos_fit.shape[1])]), {})
            for name, (X_syn, ex) in variants.items():
                t1 = time.time()
                m = measure(np.asarray(X_syn, dtype=np.float32), pos_fit, pos_hold, qt, levels, neg_tr, neg_te, SEED)
                res["rows"].append({"task": task, "mode": mode, "variant": name, **ex, **m, "measure_seconds": round(time.time() - t1, 1)})
                say(f"  {name:22s} C2ST(떼어둔) {m['c2st_hold']:.3f}±{m['c2st_hold_sd']:.3f}  C2ST(학습) {m['c2st_fit']:.3f}  "
                    f"KS {m['ks']:.3f}  상관오차 {m['corr_err']:.3f}  DCR비 {m['dcr_ratio']:.2f}  기억 {m['mem_p05']:.3f}  "
                    f"TSTR내부 {m['tstr_int']:.4f}  상수화 {m['n_const_syn']}  이진중간값 {m['frac_fractional_binary']:.4f}")
            (OUT / "diag.json").write_text(json.dumps(res, ensure_ascii=False, indent=2, default=float) + "\n", encoding="utf-8")
            torch.cuda.empty_cache()
    res["finished_utc"] = datetime.now(timezone.utc).isoformat()
    raw = (json.dumps(res, ensure_ascii=False, indent=2, default=float) + "\n").encode("utf-8")
    (OUT / "diag.json").write_bytes(raw)
    (OUT / "diag.sha256").write_text(hashlib.sha256(raw).hexdigest() + "\n", encoding="utf-8")
    say("끝")
    return 0


if __name__ == "__main__":
    sys.exit(main())
