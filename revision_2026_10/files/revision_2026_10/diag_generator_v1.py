"""보관 확산 생성기 대 고친 생성기(v2) 진단 — 제출한 Seagate 논문 MDPI 수정 (2026-10).

같은 과제·seed·전처리에서 두 생성기를 학습하고 다음을 잰다.
- 잡음 예측 손실: 학습 불량과 검증 불량(보지 않은 자료)에서, 0 예측·가우시안 독립 예측 기준선과 함께
- 합성 불량 품질: 판별 AUC(학습 불량 기준 = 보관 파이프라인 정의, 검증 불량 기준), TSTR PR-AUC(보관 정의),
  특징별 KS 거리, 상관행렬 오차, 가장 가까운 학습 불량까지 거리(복사 여부)
- 후처리(특징별 분위 맞추기) 전과 후를 모두 잰다. 보관 파이프라인은 후처리 뒤에 지표를 계산했다.

출력: experiments/seagate_kqi/outputs/revision_2026_10/generator_diag_v1/ (이미 있으면 실행하지 않음)
실행 위치: 저장소 최상위 (data_root 가 상대 경로)
"""

from __future__ import annotations

import contextlib
import hashlib
import io
import json
import platform
import sys
import time
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

OUT = SEAGATE / "outputs/revision_2026_10/generator_diag_v1"
TASKS = [0, 5, 10]
SEED = 42


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def archived_cfg() -> TorchDDPMConfig:
    """동일 예산 보관 원천(v10) 실행의 명령줄 값: hidden 384, 6층, T=200, β 끝 0.01, 200 epoch."""
    return TorchDDPMConfig(T=200, beta_start=1e-4, beta_end=0.01, time_dim=128, class_dim=16, hidden_dim=384, n_layers=6,
                           dropout=0.1, epochs=200, batch_size=256, lr=1e-3, use_amp=True, device="cuda")


@torch.no_grad()
def archived_denoise_loss(aug: TorchTabDDPMConditionalAugmenter, Xs: np.ndarray, seed: int, n_draws: int = 8) -> dict:
    dev = next(aug.model.parameters()).device
    X = torch.from_numpy(np.ascontiguousarray(Xs, dtype=np.float32)).to(dev)
    Y = torch.ones(X.shape[0], dtype=torch.long, device=dev)
    g = torch.Generator(device=dev)
    g.manual_seed(seed)
    scale = aug.feature_noise_scale_t if aug.feature_noise_scale_t is not None else torch.ones(1, X.shape[1], device=dev)
    tot = {"model": 0.0, "zero": 0.0, "gauss_indep": 0.0}
    cnt = 0
    aug.model.eval()
    for _ in range(n_draws):
        t = torch.randint(0, aug.cfg.T, (X.shape[0],), device=dev, generator=g)
        eps = torch.randn(X.shape, device=dev, generator=g) * scale
        ab = aug.alpha_bars[t].view(-1, 1)
        x_t = torch.sqrt(ab) * X + torch.sqrt(1.0 - ab) * eps
        pred = aug.model(x_t, t, Y).float()
        tot["model"] += float(((pred - eps) ** 2).sum())
        tot["zero"] += float((eps ** 2).sum())
        tot["gauss_indep"] += float(((torch.sqrt(1.0 - ab) * x_t - eps) ** 2).sum())
        cnt += eps.numel()
    return {k: v / cnt for k, v in tot.items()}


def nn_dist(ref: np.ndarray, q: np.ndarray) -> np.ndarray:
    nn = NearestNeighbors(n_neighbors=1).fit(ref)
    return nn.kneighbors(q, return_distance=True)[0][:, 0]


def corr_err(a: np.ndarray, b: np.ndarray) -> float:
    ca, cb = np.corrcoef(a, rowvar=False), np.corrcoef(b, rowvar=False)
    ca, cb = np.nan_to_num(ca), np.nan_to_num(cb)
    return float(np.linalg.norm(ca - cb) / max(np.linalg.norm(cb), 1e-12))


def mean_ks(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean([ks_2samp(a[:, j], b[:, j]).statistic for j in range(a.shape[1])]))


def quality(name: str, X_syn: np.ndarray, X_syn_s: np.ndarray, ctx: dict, cfg) -> dict:
    """X_syn: 원래 단위, X_syn_s: 분위(정규) 공간."""
    X_pos, X_vpos, X_neg = ctx["X_pos"], ctx["X_vpos"], ctx["X_neg"]
    qt = ctx["qt"]
    n_eval = min(len(X_syn), ctx["max_n_add"])
    t0 = time.time()
    out = {
        "variant": name,
        "rvs_auc_train_pos": float(rp.real_vs_synth_auc(X_pos, X_syn[:n_eval], cfg, cfg.seed)),
        "rvs_auc_val_pos": float(rp.real_vs_synth_auc(X_vpos, X_syn[:n_eval], cfg, cfg.seed)),
        "tstr_pr_auc": float(rp.tstr_utility_pr_auc(X_neg, X_syn[:n_eval], ctx["X_val"], ctx["y_val"], cfg, cfg.seed)),
        "mean_ks_vs_val_pos": mean_ks(X_syn, X_vpos),
        "corr_err_vs_val_pos": corr_err(X_syn_s, qt.transform(X_vpos)),
    }
    d_syn = nn_dist(ctx["Xs_pos"], X_syn_s)
    d_val = nn_dist(ctx["Xs_pos"], qt.transform(X_vpos))
    out["dcr_median_syn"] = float(np.median(d_syn))
    out["dcr_median_val"] = float(np.median(d_val))
    out["dcr_ratio"] = float(np.median(d_syn) / max(np.median(d_val), 1e-12))
    out["frac_syn_closer_than_val_p05"] = float(np.mean(d_syn < np.quantile(d_val, 0.05)))
    out["seconds"] = round(time.time() - t0, 1)
    return out


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
                            use_rep_cache=True, seed=SEED, s3_postprocess="featurewise_quantile_pos",
                            s3_postprocess_low_q=0.01, s3_postprocess_high_q=0.99, ddpm_train_mode="positive_only")
    X_train, X_val, X_test, y_tr_raw, y_va_raw, y_te_raw = rp.load_toolset_arrays(cfg.data_root, cfg.toolset, cfg.x_mmap_mode)
    rep_tr = rp.load_cached_representation(cfg, "train")
    if rep_tr is None:
        raise RuntimeError("학습 평균 표현 캐시가 없다")
    rep_va = rp.build_representation(X_val, "mean").astype(np.float32)
    say(f"표현: train {rep_tr.shape}, val {rep_va.shape}")

    results = {"schema_id": "seagate-revision-generator-diag-v1", "lineage": "Seagate negative-audit (제출한 논문)",
               "tasks": TASKS, "seed": SEED, "representation": "mean", "train_mode": "positive_only",
               "archived_cfg": archived_cfg().__dict__, "v2_cfg": TorchDDPMv2Config().__dict__, "per_task": {},
               "code_sha256": {"augmenters_torch_ddpm.py": sha(ROOT / ".venv/augmenters_torch_ddpm.py"),
                               "augmenters_torch_ddpm_v2.py": sha(HERE / "augmenters_torch_ddpm_v2.py"),
                               "run_pipeline.py": sha(SEAGATE / "run_pipeline.py"),
                               "diag_generator_v1.py": sha(Path(__file__).resolve())},
               "environment": {"python": sys.version.split()[0], "torch": torch.__version__,
                               "cuda": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                               "platform": platform.platform()}}
    for task in TASKS:
        say(f"=== 과제 {task} ===")
        mtr, ytr = rp.extract_task_binary(y_tr_raw, task)
        mva, yva = rp.extract_task_binary(y_va_raw, task)
        Xtr = np.asarray(rep_tr[mtr], dtype=np.float32)
        Xva = rep_va[mva]
        prep = rp.TabularPreprocessor(drop_constant=True, imputer_strategy="mean")
        prep.fit(Xtr)
        Xtr_c, Xva_c = prep.transform_classifier(Xtr), prep.transform_classifier(Xva)
        X_pos, X_neg = Xtr_c[ytr == 1], Xtr_c[ytr == 0]
        X_vpos = Xva_c[yva == 1]
        qt = QuantileTransformer(n_quantiles=min(2000, len(X_pos)), output_distribution="normal", random_state=SEED)
        Xs_pos = qt.fit_transform(X_pos).astype(np.float32)
        Xs_vpos = qt.transform(X_vpos).astype(np.float32)
        max_n_add = int(round(len(X_pos) * 2.0))
        n_syn = int(min(6000, max(500, max_n_add)))
        lo, hi = np.quantile(Xtr_c, 0.01, axis=0), np.quantile(Xtr_c, 0.99, axis=0)
        ctx = {"X_pos": X_pos, "X_vpos": X_vpos, "X_neg": X_neg, "X_val": Xva_c, "y_val": yva, "qt": qt,
               "Xs_pos": Xs_pos, "max_n_add": max_n_add}
        say(f"학습 불량 {len(X_pos)}, 검증 불량 {len(X_vpos)}, 학습 정상 {len(X_neg)}, 특징 {X_pos.shape[1]}, 합성 {n_syn}")
        rec = {"n_pos_train": int(len(X_pos)), "n_pos_val": int(len(X_vpos)), "n_features": int(X_pos.shape[1])}

        # 보관 생성기
        t0 = time.time()
        aug_a = TorchTabDDPMConditionalAugmenter(seed=SEED, cfg=archived_cfg())
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            aug_a.fit(Xs_pos, np.ones(len(Xs_pos), dtype=int))
        (OUT / f"task{task}_archived_fit.log").write_text(buf.getvalue(), encoding="utf-8")
        rec["archived_train_seconds"] = round(time.time() - t0, 1)
        rec["archived_loss_train_pos"] = archived_denoise_loss(aug_a, Xs_pos, seed=1)
        rec["archived_loss_val_pos"] = archived_denoise_loss(aug_a, Xs_vpos, seed=1)
        Xa_s = aug_a.sample(n_syn, y_label=1).astype(np.float32)
        say(f"보관 생성기: 학습 {rec['archived_train_seconds']}초, 검증 손실 {rec['archived_loss_val_pos']}")

        # 고친 생성기
        t0 = time.time()
        aug_b = TorchTabDDPMv2Augmenter(seed=SEED, cfg=TorchDDPMv2Config())
        aug_b.fit(Xs_pos, np.ones(len(Xs_pos), dtype=int))
        rec["v2_train_seconds"] = round(time.time() - t0, 1)
        rec["v2_loss_log"] = aug_b.loss_log
        rec["v2_loss_train_pos"] = aug_b.denoise_loss(Xs_pos, np.ones(len(Xs_pos), dtype=int), seed=1)
        rec["v2_loss_val_pos"] = aug_b.denoise_loss(Xs_vpos, np.ones(len(Xs_vpos), dtype=int), seed=1)
        t1 = time.time()
        Xb_s = aug_b.sample(n_syn, y_label=1).astype(np.float32)
        rec["v2_sample_seconds"] = round(time.time() - t1, 1)
        say(f"고친 생성기: 학습 {rec['v2_train_seconds']}초, 검증 손실 {rec['v2_loss_val_pos']}")

        # 품질: 원래 단위로 되돌리고 보관 파이프라인처럼 1~99% 로 자른 뒤, 후처리 전·후
        qual = []
        for name, Xs in (("archived", Xa_s), ("v2", Xb_s)):
            X_raw = np.clip(qt.inverse_transform(np.clip(Xs, -8, 8)), lo, hi).astype(np.float32)
            X_pp = rp.postprocess_s3_synth(X_pos, X_raw, cfg).astype(np.float32)
            for tag, Xo in (("raw", X_raw), ("postprocessed", X_pp)):
                q = quality(f"{name}_{tag}", Xo, qt.transform(Xo).astype(np.float32), ctx, cfg)
                say(f"  {q['variant']}: 판별AUC(학습) {q['rvs_auc_train_pos']:.4f}, 판별AUC(검증) {q['rvs_auc_val_pos']:.4f}, "
                    f"TSTR {q['tstr_pr_auc']:.4f}, KS {q['mean_ks_vs_val_pos']:.4f}, 상관오차 {q['corr_err_vs_val_pos']:.3f}, DCR비 {q['dcr_ratio']:.3f}")
                qual.append(q)
        # 기준: 학습 불량 자체를 '합성'처럼 넣었을 때(실제 자료의 상한)
        ref = quality("real_train_pos_as_synth", X_pos, Xs_pos, ctx, cfg)
        qual.append(ref)
        say(f"  기준(실제 학습 불량): 판별AUC(검증) {ref['rvs_auc_val_pos']:.4f}, TSTR {ref['tstr_pr_auc']:.4f}, KS {ref['mean_ks_vs_val_pos']:.4f}, 상관오차 {ref['corr_err_vs_val_pos']:.3f}")
        rec["quality"] = qual
        results["per_task"][str(task)] = rec
        (OUT / "diag.json").write_text(json.dumps(results, ensure_ascii=False, indent=2, default=float) + "\n", encoding="utf-8")
        torch.cuda.empty_cache()

    results["finished_utc"] = datetime.now(timezone.utc).isoformat()
    raw = (json.dumps(results, ensure_ascii=False, indent=2, default=float) + "\n").encode("utf-8")
    (OUT / "diag.json").write_bytes(raw)
    (OUT / "diag.sha256").write_text(hashlib.sha256(raw).hexdigest() + "\n", encoding="utf-8")
    say("끝")
    return 0


if __name__ == "__main__":
    sys.exit(main())
