"""생성기 진단 4차 — 제출한 Seagate 논문(MDPI electronics-4573451) 수정 실험 (2026-10-07).

3차 진단에서 가우시안 확산(v2.1)은 이진 특징의 드문 값을 만들지 못해 과제 10 에서 떼어 둔 불량과 완벽히 구별됐다(C2ST 1.000).
실제 학습 불량에서 드문 이진 값만 지워도 C2ST 가 0.562 에서 0.997 로 오른다는 점으로 원인을 확인했다.
이번에는 TabDDPM 처럼 이진 특징을 다항(범주 2개) 확산으로 다루는 혼합형 생성기(augmenters_torch_ddpm_v3.py)를 같은 기준으로 잰다.

- 과제 0·5·10, positive_only·conditional_mixed, seed 42. 떼어 두기·음성 나누기·측정 함수는 3차와 같다(diag_generator_v3 를 그대로 쓴다).
- 생성기 입력: 생성기 학습행(떼어 둔 행 제외)에서 값이 정확히 두 개인 특징은 이진(작은 값 0, 큰 값 1), 나머지는 분위 정규화 연속 특징.
- 측정용 분위 변환은 3차와 같게 모든 특징에 맞춘 것을 쓴다(3차 결과와 바로 비교하기 위해).
출력: experiments/seagate_kqi/outputs/revision_2026_10/generator_diag_v4/. 논문 결과로 쓰지 않는다.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from sklearn.preprocessing import QuantileTransformer

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("d3", HERE / "diag_generator_v3.py")
d3 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(d3)
rp = d3.rp
from augmenters_torch_ddpm_v3 import TorchDDPMv3Config, TorchTabDDPMv3Augmenter  # noqa: E402

OUT = d3.SEAGATE / "outputs/revision_2026_10/generator_diag_v4"
TASKS = [0, 5, 10]
MODES = ["positive_only", "conditional_mixed"]
V3 = dict(T=1000, beta_start=1e-4, beta_end=2e-2, hidden_dim=1024, n_blocks=4, time_dim=128, dropout=0.1, train_steps=40000,
          batch_size=256, lr=5e-4, weight_decay=1e-4, warmup_steps=500, grad_clip=1.0, ema_decay=0.999, log_every=500,
          sample_batch=4096, device="cuda", standardize=True, holdout_frac=d3.HOLD, min_holdout=20, eval_every=250, patience=8,
          use_bf16=True)


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
                            use_rep_cache=True, seed=d3.SEED, ddpm_neg_multiplier=3.0, ddpm_max_neg=100000)
    X_train, X_val, X_test, y_tr_raw, y_va_raw, _ = rp.load_toolset_arrays(cfg.data_root, cfg.toolset, cfg.x_mmap_mode)
    rep_tr = rp.load_cached_representation(cfg, "train")
    res = {"schema_id": "seagate-revision-generator-diag-v4", "lineage": "Seagate negative-audit (제출한 논문)", "tasks": TASKS,
           "modes": MODES, "seed": d3.SEED, "v3": V3, "rows": [],
           "code_sha256": {"diag_generator_v4.py": d3.sha(Path(__file__).resolve()), "diag_generator_v3.py": d3.sha(HERE / "diag_generator_v3.py"),
                           "augmenters_torch_ddpm_v3.py": d3.sha(HERE / "augmenters_torch_ddpm_v3.py"),
                           "augmenters_torch_ddpm_v2.py": d3.sha(HERE / "augmenters_torch_ddpm_v2.py")}}
    for task in TASKS:
        mtr, ytr = rp.extract_task_binary(y_tr_raw, task)
        Xtr = np.asarray(rep_tr[mtr], dtype=np.float32)
        prep = rp.TabularPreprocessor(drop_constant=True, imputer_strategy="mean")
        prep.fit(Xtr)
        ctx = {"X_train": prep.transform_classifier(Xtr), "y_train": ytr, "task_id": task}
        neg_all = ctx["X_train"][ytr == 0]
        perm = np.random.default_rng(20261007 + task).permutation(len(neg_all))
        n_te = int(round(0.2 * len(neg_all)))
        neg_te, neg_tr = neg_all[perm[:n_te]], neg_all[perm[n_te:]]
        for mode in MODES:
            arr = rp.prepare_s3_training_arrays(ctx, rp.PipelineConfig(**{**asdict(cfg), "ddpm_train_mode": mode}), mode)
            Xd, yd = arr["X_ddpm_train"], arr["y_ddpm_train"]
            hold = d3.split_holdout(yd, d3.SEED)
            pos_fit, pos_hold = Xd[(yd == 1) & ~hold], Xd[(yd == 1) & hold]
            qt_meas = QuantileTransformer(n_quantiles=min(2000, int((~hold).sum())), output_distribution="normal", random_state=d3.SEED)
            qt_meas.fit(Xd[~hold])
            levels_meas = d3.binary_levels(Xd[~hold])
            cont_idx, bin_idx, levels = split_types(Xd[~hold])
            qt_c = QuantileTransformer(n_quantiles=min(2000, int((~hold).sum())), output_distribution="normal", random_state=d3.SEED)
            qt_c.fit(Xd[~hold][:, cont_idx])
            Xc = qt_c.transform(Xd[:, cont_idx]).astype(np.float32)
            Xb = to_bin(Xd, bin_idx, levels)
            n_syn = int(min(6000, max(500, 2 * len(pos_fit))))
            say(f"과제 {task} {mode}: 생성기 학습 {int((~hold).sum())}행 (불량 {len(pos_fit)}), 떼어 둔 불량 {len(pos_hold)}, "
                f"연속 {len(cont_idx)} · 이진 {len(bin_idx)}, 합성 {n_syn}")
            t0 = time.time()
            g = TorchTabDDPMv3Augmenter(seed=d3.SEED, cfg=TorchDDPMv3Config(**V3))
            g.fit(Xc, Xb, yd)
            if not np.array_equal(g.holdout_mask, hold):
                raise RuntimeError("떼어 둔 행이 진단과 생성기에서 다르다")
            fit_s = round(time.time() - t0, 1)
            hp = hold & (yd == 1)
            hl = g.denoise_loss(Xc[hp], Xb[hp], np.ones(int(hp.sum()), dtype=int), seed=1)
            C, Bs = g.sample(n_syn, y_label=1)
            X_syn = np.zeros((n_syn, Xd.shape[1]), dtype=np.float32)
            X_syn[:, cont_idx] = qt_c.inverse_transform(np.clip(C, -8, 8))
            for k, j in enumerate(bin_idx):
                a, b = levels[j]
                X_syn[:, j] = np.where(Bs[:, k] > 0, b, a)
            m = d3.measure(X_syn, pos_fit, pos_hold, qt_meas, levels_meas, neg_tr, neg_te, d3.SEED)
            real_rate = to_bin(pos_fit, bin_idx, levels).mean(0) if len(bin_idx) else np.zeros(0)
            rate_err = float(np.abs(real_rate - Bs.mean(0)).mean()) if len(bin_idx) else float("nan")
            row = {"task": task, "mode": mode, "variant": "v3_mixed", "fit_seconds": fit_s, "best_step": g.best["step"], "steps_run": g.steps_run,
                   "best_holdout_loss": g.best["holdout_loss"], "holdout_pos_loss": hl, "n_cont": int(len(cont_idx)), "n_bin": int(len(bin_idx)),
                   "bin_rate_real_mean": float(real_rate.mean()) if len(bin_idx) else None,
                   "bin_rate_syn_mean": float(Bs.mean()) if len(bin_idx) else None, "bin_rate_abs_err": rate_err, **m}
            res["rows"].append(row)
            say(f"  v3 혼합형: 학습 {fit_s}초, 최적 {g.best['step']}/{g.steps_run}, 떼어 둔 불량 손실 연속 {hl['cont_model']:.3f}"
                f"(0-예측 {hl['cont_zero']:.3f}, 독립 {hl['cont_gauss_indep']:.3f}) 이진 {hl['bin_model']:.3f}(주변 {hl['bin_marginal']:.3f}, 복사 {hl['bin_copy']:.3f})")
            say(f"    C2ST(떼어둔) {m['c2st_hold']:.3f}±{m['c2st_hold_sd']:.3f}  C2ST(학습) {m['c2st_fit']:.3f}  KS {m['ks']:.3f}  상관오차 {m['corr_err']:.3f}  "
                f"DCR비 {m['dcr_ratio']:.2f}  기억 {m['mem_p05']:.3f}  TSTR내부 {m['tstr_int']:.4f}  상수화 {m['n_const_syn']}  "
                f"1비율 실제 {row['bin_rate_real_mean']:.4f} 합성 {row['bin_rate_syn_mean']:.4f} (특징별 오차 {rate_err:.4f})")
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
