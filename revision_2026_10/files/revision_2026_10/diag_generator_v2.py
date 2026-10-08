"""생성기 진단 v2 — 제출한 Seagate 논문 MDPI 수정 (2026-10).

v1 에서 알게 된 점: 학습 불량과 검증 불량 사이의 분포 이동이 커서 실제 학습 불량조차 검증 불량과 판별 AUC 1.0 이다.
그래서 생성 품질은 공식 검증 자료가 아니라 **학습 불량 안에서 떼어 둔 20%** 와 비교해 잰다 (같은 분포).
고친 생성기 v2.1 은 그 떼어 둔 자료로 조기 종료한다. 공식 검증 자료는 효용 지표(TSTR, 보관 정의)에만 쓴다.

비교: 보관 생성기(같은 떼어 둔 자료 기준으로 다시 잼) 대 v2.1 (표준화, 조기 종료, bf16, 은닉 1024 와 512)
과제 0, 5, 10 / seed 42 / 학습 모드 positive_only, conditional_mixed
출력: experiments/seagate_kqi/outputs/revision_2026_10/generator_diag_v2/ (이미 있으면 실행하지 않음)
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
from scipy.stats import ks_2samp
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import QuantileTransformer

HERE = Path(__file__).resolve().parent
SEAGATE = HERE.parent
ROOT = SEAGATE.parents[1]
sys.path[:0] = [str(SEAGATE), str(ROOT / ".venv"), str(HERE)]

import run_pipeline as rp  # noqa: E402
from augmenters_torch_ddpm import TorchDDPMConfig, TorchTabDDPMConditionalAugmenter  # noqa: E402
from augmenters_torch_ddpm_v2 import TorchDDPMv2Config, TorchTabDDPMv2Augmenter  # noqa: E402

OUT = SEAGATE / "outputs/revision_2026_10/generator_diag_v2"
TASKS = [0, 5, 10]
MODES = ["positive_only", "conditional_mixed"]
SEED = 42
HOLD = 0.2

V21 = {"standardize": True, "holdout_frac": HOLD, "eval_every": 250, "patience": 8, "train_steps": 10000,
       "use_bf16": True, "hidden_dim": 1024, "n_blocks": 4, "lr": 5e-4, "dropout": 0.1}
V21_SMALL = {**V21, "hidden_dim": 512}


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def archived_cfg() -> TorchDDPMConfig:
    return TorchDDPMConfig(T=200, beta_start=1e-4, beta_end=0.01, time_dim=128, class_dim=16, hidden_dim=384, n_layers=6,
                           dropout=0.1, epochs=200, batch_size=256, lr=1e-3, use_amp=True, device="cuda")


def split_holdout(y: np.ndarray, seed: int) -> np.ndarray:
    """v2.1 의 fit 과 같은 규칙으로 떼어 둘 행을 고른다 (클래스별 20%, 각 쪽 최소 20행)."""
    rng = np.random.default_rng(seed + 104729)
    hold = np.zeros(len(y), dtype=bool)
    for cls in np.unique(y):
        idx = np.flatnonzero(y == cls)
        k = int(round(len(idx) * HOLD))
        if k >= 20 and len(idx) - k >= 20:
            hold[rng.choice(idx, size=k, replace=False)] = True
    return hold


def fidelity(X_syn: np.ndarray, X_fit_pos: np.ndarray, X_hold_pos: np.ndarray, qt, cfg, ctx) -> dict:
    """같은 분포의 떼어 둔 불량을 기준으로 잰 품질. X 는 모두 원래 단위."""
    n = min(len(X_syn), ctx["max_n_add"])
    s_syn, s_fit, s_hold = qt.transform(X_syn), qt.transform(X_fit_pos), qt.transform(X_hold_pos)
    d_syn = NearestNeighbors(n_neighbors=1).fit(s_fit).kneighbors(s_syn)[0][:, 0]
    d_hold = NearestNeighbors(n_neighbors=1).fit(s_fit).kneighbors(s_hold)[0][:, 0]

    def corr_err(a, b):
        ca, cb = np.nan_to_num(np.corrcoef(a, rowvar=False)), np.nan_to_num(np.corrcoef(b, rowvar=False))
        return float(np.linalg.norm(ca - cb) / max(np.linalg.norm(cb), 1e-12))

    return {
        "rvs_auc_vs_holdout": float(rp.real_vs_synth_auc(X_hold_pos, X_syn[:n], cfg, cfg.seed)),
        "rvs_auc_vs_fit": float(rp.real_vs_synth_auc(X_fit_pos, X_syn[:n], cfg, cfg.seed)),
        "mean_ks_vs_holdout": float(np.mean([ks_2samp(X_syn[:, j], X_hold_pos[:, j]).statistic for j in range(X_syn.shape[1])])),
        "corr_err_vs_holdout": corr_err(s_syn, s_hold),
        "dcr_ratio": float(np.median(d_syn) / max(np.median(d_hold), 1e-12)),
        "frac_syn_closer_than_holdout_p05": float(np.mean(d_syn < np.quantile(d_hold, 0.05))),
        "tstr_pr_auc_val": float(rp.tstr_utility_pr_auc(ctx["X_neg"], X_syn[:n], ctx["X_val"], ctx["y_val"], cfg, cfg.seed)),
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
    rep_va = rp.build_representation(X_val, "mean").astype(np.float32)
    res = {"schema_id": "seagate-revision-generator-diag-v2", "lineage": "Seagate negative-audit (제출한 논문)",
           "tasks": TASKS, "modes": MODES, "seed": SEED, "holdout_frac": HOLD, "configs": {"v21": V21, "v21_small": V21_SMALL},
           "archived_cfg": asdict(archived_cfg()), "rows": [],
           "code_sha256": {"augmenters_torch_ddpm_v2.py": sha(HERE / "augmenters_torch_ddpm_v2.py"),
                           "diag_generator_v2.py": sha(Path(__file__).resolve()),
                           "run_pipeline.py": sha(SEAGATE / "run_pipeline.py")}}
    for task in TASKS:
        mtr, ytr = rp.extract_task_binary(y_tr_raw, task)
        mva, yva = rp.extract_task_binary(y_va_raw, task)
        Xtr = np.asarray(rep_tr[mtr], dtype=np.float32)
        prep = rp.TabularPreprocessor(drop_constant=True, imputer_strategy="mean")
        prep.fit(Xtr)
        ctx = {"X_train": prep.transform_classifier(Xtr), "y_train": ytr, "X_val": prep.transform_classifier(rep_va[mva]),
               "y_val": yva, "task_id": task}
        ctx["X_neg"] = ctx["X_train"][ytr == 0]
        ctx["max_n_add"] = int(round(int(ytr.sum()) * 2.0))
        lo, hi = np.quantile(ctx["X_train"], 0.01, axis=0), np.quantile(ctx["X_train"], 0.99, axis=0)
        for mode in MODES:
            arr = rp.prepare_s3_training_arrays(ctx, rp.PipelineConfig(**{**asdict(cfg), "ddpm_train_mode": mode}), mode)
            Xd, yd = arr["X_ddpm_train"], arr["y_ddpm_train"]
            hold = split_holdout(yd, SEED)
            pos_fit, pos_hold = Xd[(yd == 1) & ~hold], Xd[(yd == 1) & hold]
            if len(pos_hold) < 20:
                say(f"과제 {task} {mode}: 떼어 둘 불량이 부족하다 ({len(pos_hold)})")
                continue
            # 분위 변환은 떼어 두지 않은 학습 부분으로만 맞춘다 (떼어 둔 자료의 정보가 새지 않게)
            qt = QuantileTransformer(n_quantiles=min(2000, int((~hold).sum())), output_distribution="normal", random_state=SEED)
            qt.fit(Xd[~hold])
            Xs = qt.transform(Xd).astype(np.float32)
            n_syn = int(min(6000, max(500, ctx["max_n_add"])))
            say(f"과제 {task} {mode}: 학습 {len(Xd)}행 (불량 {int(yd.sum())}, 떼어 둔 불량 {len(pos_hold)}), 특징 {Xd.shape[1]}")
            gens = {}
            # 보관 생성기: 떼어 두지 않은 부분으로만 학습해 같은 기준으로 비교
            t0 = time.time()
            a = TorchTabDDPMConditionalAugmenter(seed=SEED, cfg=archived_cfg())
            with contextlib.redirect_stdout(io.StringIO()):
                a.fit(Xs[~hold], yd[~hold])
            gens["archived"] = (a.sample(n_syn, y_label=1), round(time.time() - t0, 1), {})
            for name, conf in (("v21", V21), ("v21_small", V21_SMALL)):
                t0 = time.time()
                b = TorchTabDDPMv2Augmenter(seed=SEED, cfg=TorchDDPMv2Config(**conf))
                b.fit(Xs, yd)
                if not np.array_equal(b.holdout_mask, hold):
                    raise RuntimeError("떼어 둔 행이 진단과 생성기에서 다르다")
                fit_s = round(time.time() - t0, 1)
                t1 = time.time()
                S = b.sample(n_syn, y_label=1)
                hl = b.denoise_loss(Xs[hold & (yd == 1)], np.ones(int((hold & (yd == 1)).sum()), dtype=int), seed=1)
                gens[name] = (S, fit_s, {"best_step": b.best["step"], "steps_run": b.steps_run, "best_holdout_loss": b.best["holdout_loss"],
                                         "holdout_pos_loss": hl, "sample_seconds": round(time.time() - t1, 1)})
            for name, (S, fit_s, extra) in gens.items():
                X_syn = np.clip(qt.inverse_transform(np.clip(S, -8, 8)), lo, hi).astype(np.float32)
                f = fidelity(X_syn, pos_fit, pos_hold, qt, cfg, ctx)
                row = {"task": task, "mode": mode, "generator": name, "fit_seconds": fit_s, **extra, **f}
                res["rows"].append(row)
                say(f"  {name}: {fit_s}초, 판별AUC(떼어둔) {f['rvs_auc_vs_holdout']:.4f}, 판별AUC(학습) {f['rvs_auc_vs_fit']:.4f}, "
                    f"KS {f['mean_ks_vs_holdout']:.4f}, 상관오차 {f['corr_err_vs_holdout']:.3f}, DCR비 {f['dcr_ratio']:.3f}, "
                    f"TSTR {f['tstr_pr_auc_val']:.4f} {('| 최적 단계 ' + str(extra['best_step']) + ', 떼어둔 손실 ' + str({k: round(v, 3) for k, v in extra['holdout_pos_loss'].items()})) if extra else ''}")
            # 기준: 떼어 두지 않은 실제 불량을 '합성'처럼 넣었을 때 (같은 분포의 상한)
            ref = fidelity(pos_fit, pos_fit, pos_hold, qt, cfg, ctx)
            res["rows"].append({"task": task, "mode": mode, "generator": "real_fit_pos", **ref})
            say(f"  기준(실제 불량): 판별AUC(떼어둔) {ref['rvs_auc_vs_holdout']:.4f}, KS {ref['mean_ks_vs_holdout']:.4f}, "
                f"상관오차 {ref['corr_err_vs_holdout']:.3f}, TSTR {ref['tstr_pr_auc_val']:.4f}")
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
