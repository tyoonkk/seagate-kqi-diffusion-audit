"""생성기 진단 3차-b — 제출한 Seagate 논문(MDPI electronics-4573451) 수정 실험 (2026-10-07).

3차 진단에서 고친 생성기(v2.1)는 떼어 둔 불량의 잡음 예측 손실은 기준선보다 확실히 낮았지만, 작은 과제(과제 0, 학습 불량 121행)에서
표본의 주변분포가 실제보다 좁고(KS 0.114 대 실제 0.058) 실제 불량에서는 값이 변하는 특징 40개가 합성에서 상수가 됐다.
같은 학습 모델에서 **표본 추출 방식만** 바꿔 이 문제가 추출 단계에서 생기는지 본다. 학습은 바꾸지 않는다.

추출 방식
- bf16_post : 지금 방식. bf16 자동 혼합 정밀도, 분산 β̃_t (사후 분산, Ho et al. 2020 의 fixedsmall)
- fp32_post : 32비트, 분산 β̃_t
- fp32_beta : 32비트, 분산 β_t (Ho et al. 2020 의 fixedlarge)

과제 0·5·10 positive_only, seed 42, 자르기 없음(noclip). 측정은 diag_generator_v3.measure 를 그대로 쓴다.
추가로 분위 정규 공간에서 특징별 표준편차 비(합성/실제 학습 불량)의 중앙값을 적는다(1 보다 작으면 좁게 뽑힌 것).
출력: experiments/seagate_kqi/outputs/revision_2026_10/generator_diag_v3b/. 논문 결과로 쓰지 않는다.
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

OUT = d3.SEAGATE / "outputs/revision_2026_10/generator_diag_v3b"
TASKS = [0, 5, 10]
MODE = "positive_only"
VARIANTS = {"bf16_post": (True, "post"), "fp32_post": (False, "post"), "fp32_beta": (False, "beta")}


@torch.no_grad()
def sample_variant(aug, n: int, use_bf16: bool, var: str, seed_offset: int = 7919) -> np.ndarray:
    """augmenters_torch_ddpm_v2.TorchTabDDPMv2Augmenter.sample 과 같은 계산에 정밀도와 분산 선택만 더했다."""
    dev = aug.device
    g = torch.Generator(device=dev)
    g.manual_seed(aug.seed + seed_offset)
    out = []
    d = aug.ema.inp.in_features
    for start in range(0, n, aug.cfg.sample_batch):
        b = min(aug.cfg.sample_batch, n - start)
        x = torch.randn((b, d), device=dev, generator=g)
        y = torch.full((b,), 1, device=dev, dtype=torch.long)
        for ti in range(aug.cfg.T - 1, -1, -1):
            t = torch.full((b,), ti, device=dev, dtype=torch.long)
            if use_bf16 and dev.type == "cuda":
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    eps_hat = aug.eps_pred(aug.ema, x, t, y).float()
            else:
                eps_hat = aug.eps_pred(aug.ema, x, t, y)
            a, ab, bt = aug.alphas[ti], aug.alpha_bars[ti], aug.betas[ti]
            mean = (x - bt / torch.sqrt(1.0 - ab) * eps_hat) / torch.sqrt(a)
            if ti > 0:
                sig = torch.sqrt(aug.post_var[ti]) if var == "post" else torch.sqrt(bt)
                x = mean + sig * torch.randn((b, d), device=dev, generator=g)
            else:
                x = mean
        out.append(x.float().cpu().numpy())
    S = np.vstack(out)
    if getattr(aug, "mu", None) is not None:
        S = S * aug.sd + aug.mu
    return S


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
    res = {"schema_id": "seagate-revision-generator-diag-v3b", "lineage": "Seagate negative-audit (제출한 논문)", "tasks": TASKS,
           "mode": MODE, "seed": d3.SEED, "v21": d3.V21, "variants": {k: list(v) for k, v in VARIANTS.items()}, "rows": [],
           "code_sha256": {"diag_generator_v3b.py": d3.sha(Path(__file__).resolve()), "diag_generator_v3.py": d3.sha(HERE / "diag_generator_v3.py"),
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
        arr = rp.prepare_s3_training_arrays(ctx, rp.PipelineConfig(**{**asdict(cfg), "ddpm_train_mode": MODE}), MODE)
        Xd, yd = arr["X_ddpm_train"], arr["y_ddpm_train"]
        hold = d3.split_holdout(yd, d3.SEED)
        pos_fit, pos_hold = Xd[(yd == 1) & ~hold], Xd[(yd == 1) & hold]
        qt = QuantileTransformer(n_quantiles=min(2000, int((~hold).sum())), output_distribution="normal", random_state=d3.SEED)
        qt.fit(Xd[~hold])
        Xs = qt.transform(Xd).astype(np.float32)
        levels = d3.binary_levels(Xd[~hold])
        n_syn = int(min(6000, max(500, 2 * len(pos_fit))))
        t0 = time.time()
        b = d3.TorchTabDDPMv2Augmenter(seed=d3.SEED, cfg=d3.TorchDDPMv2Config(**d3.V21))
        b.fit(Xs, yd)
        if not np.array_equal(b.holdout_mask, hold):
            raise RuntimeError("떼어 둔 행이 다르다")
        say(f"과제 {task}: 학습 {round(time.time() - t0, 1)}초, 최적 {b.best['step']}/{b.steps_run}")
        z_fit = qt.transform(pos_fit)
        sd_fit = z_fit.std(0)
        for name, (bf16, var) in VARIANTS.items():
            t1 = time.time()
            S = np.clip(sample_variant(b, n_syn, bf16, var), -8, 8)
            sd_ratio = np.median(S.std(0)[sd_fit > 1e-6] / sd_fit[sd_fit > 1e-6])
            X_syn = qt.inverse_transform(S).astype(np.float32)
            m = d3.measure(X_syn, pos_fit, pos_hold, qt, levels, neg_tr, neg_te, d3.SEED)
            res["rows"].append({"task": task, "variant": name, "sd_ratio_median": float(sd_ratio), "sample_seconds": round(time.time() - t1, 1), **m})
            say(f"  {name:10s} 표준편차비 {sd_ratio:.3f}  C2ST(떼어둔) {m['c2st_hold']:.3f}  KS {m['ks']:.3f}  상관오차 {m['corr_err']:.3f}  "
                f"DCR비 {m['dcr_ratio']:.2f}  기억 {m['mem_p05']:.3f}  TSTR내부 {m['tstr_int']:.4f}  상수화 {m['n_const_syn']}")
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
