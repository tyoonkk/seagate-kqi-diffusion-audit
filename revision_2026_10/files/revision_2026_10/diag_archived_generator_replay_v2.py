"""보관 생성기를 재생성 실험과 같은 떼어 두기 분할로 다시 학습해 같은 품질 지표를 잰다 — 제출한 Seagate 논문(MDPI electronics-4573451) 수정.

v2 (2026-10-07): v1 과 계산은 같고, 이어 돌리기만 더했다. v1 실행(outputs/revision_2026_10/archived_generator_replay_v1)은
결과와 무관하게 도구의 백그라운드 시간 제한(30분)에 걸려 127행에서 멈췄다. v1 출력은 고치지 않고 그대로 두며, v2 는 새 폴더에서
55개 사례 전체를 처음부터 다시 계산한다. 다시 멈추면 같은 코드 해시일 때만 끝난 행을 건너뛰고 이어 간다(manifest.json 대조).
v1 과 겹치는 행은 재현성 확인(같은 seed 의 GPU 학습이 같은 값을 내는지)에 쓴다.

목적: 논문에 "보관 생성기 대 고친 생성기"를 11개 과제 × seed 5개 × 두 모드에서 같은 기준으로 나란히 보고하기 위해서다.
고친 생성기 값은 재생성 실행(regen_archive_v1_run1)의 case.json 에 있다. 이 스크립트는 보관 설정만 다시 학습한다.

- 보관 설정 두 가지: quick(10 epoch, 주 감사 후보원)과 full(200 epoch, 동일예산 아카이브). 둘 다 T=200, β 1e-4→0.01, EpsMLP 384×6.
- 생성기 입력: 보관 파이프라인처럼 분위 변환하되, 떼어 둔 행을 빼고 맞춘다(보관 실행은 전체 행에 맞췄다).
  출력은 보관 코드(run_pipeline.build_s3_synth_pool)대로 역분위변환 뒤 학습 자료 1–99% 자르기. ±8 자르기 같은 다른 처리는 하지 않는다.
- 재생성 실행과 같은 행인지 case.json 의 격자 A 진단(fit_rows, fit_pos, n_pos_hold)으로 확인하고, 다르면 멈춘다.
- 떼어 둔 행: 재생성 실행이 case.json 에 남긴 `generator_holdout_rows` 를 그대로 쓴다(같은 행).
- 측정: 재생성 실행기의 generator_fidelity 와 같은 함수(C2ST, KS, 상관 오차, DCR, 기억, 내부 TSTR, 상수화)와
  떼어 둔 불량의 잡음 예측 손실(보관 망, 0-예측, 독립 가우시안 기준).
test 자료는 쓰지 않는다. 출력: outputs/revision_2026_10/archived_generator_replay_v1/ (rows.csv, run.log, 끝에 sha256)
"""

from __future__ import annotations

import contextlib
import hashlib
import importlib.util
import io
import json
import sys
import time
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.preprocessing import QuantileTransformer

HERE = Path(__file__).resolve().parent
SEAGATE = HERE.parent
ROOT = SEAGATE.parents[1]
spec = importlib.util.spec_from_file_location("R", HERE / "run_regen_archive_v1.py")
R = importlib.util.module_from_spec(spec)
spec.loader.exec_module(R)
rp = R.rp
from augmenters_torch_ddpm import TorchDDPMConfig, TorchTabDDPMConditionalAugmenter  # noqa: E402  (보관 생성기)

RUN = SEAGATE / "outputs/revision_2026_10/regen_archive_v1_run1"
OUT = SEAGATE / "outputs/revision_2026_10/archived_generator_replay_v2"
SETTINGS = {"quick_10ep": 10, "full_200ep": 200}


def archived_cfg(epochs: int) -> TorchDDPMConfig:
    return TorchDDPMConfig(T=200, beta_start=1e-4, beta_end=0.01, time_dim=128, class_dim=16, hidden_dim=384, n_layers=6,
                           dropout=0.1, epochs=epochs, batch_size=256, lr=1e-3, use_amp=True, device="cuda")


@torch.no_grad()
def archived_holdout_loss(aug, Xs_hold: np.ndarray, y_hold: np.ndarray, seed: int = 1, n_draws: int = 8) -> dict:
    """보관 망의 떼어 둔 자료 잡음 예측 손실과 기준선 (보관 일정 T=200 위에서)."""
    dev = next(aug.model.parameters()).device
    T = aug.cfg.T
    betas = torch.linspace(aug.cfg.beta_start, aug.cfg.beta_end, T, device=dev)
    ab = torch.cumprod(1.0 - betas, dim=0)
    X = torch.from_numpy(np.asarray(Xs_hold, dtype=np.float32)).to(dev)
    Y = torch.from_numpy(np.asarray(y_hold).astype(np.int64)).to(dev)
    g = torch.Generator(device=dev)
    g.manual_seed(seed)
    tot = {"model": 0.0, "zero": 0.0, "gauss_indep": 0.0}
    cnt = 0
    aug.model.eval()
    for _ in range(n_draws):
        t = torch.randint(0, T, (len(X),), device=dev, generator=g)
        eps = torch.randn(X.shape, device=dev, generator=g)
        a = ab[t].view(-1, 1)
        xt = torch.sqrt(a) * X + torch.sqrt(1 - a) * eps
        pred = aug.model(xt, t, Y)
        tot["model"] += float(F.mse_loss(pred.float(), eps, reduction="sum"))
        tot["zero"] += float((eps ** 2).sum())
        tot["gauss_indep"] += float(((torch.sqrt(1 - a) * xt - eps) ** 2).sum())
        cnt += eps.numel()
    return {k: v / cnt for k, v in tot.items()}


def main() -> int:
    import argparse
    import os
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(OUT), help="시험 실행 때만 바꾼다")
    ap.add_argument("--limit", type=int, default=0, help="앞에서 몇 사례만 (시험 실행)")
    args = ap.parse_args()
    out_dir = Path(args.out)
    os.chdir(ROOT)
    code_sha = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    man_p = out_dir / "manifest.json"
    done, rows = set(), []
    if out_dir.exists():
        man = json.loads(man_p.read_text(encoding="utf-8")) if man_p.exists() else {}
        if man.get("code_sha256") != code_sha or (out_dir / "rows.sha256").exists():
            print(f"이어 갈 수 없는 출력 폴더다(코드가 다르거나 이미 끝남): {out_dir}")
            return 1
        if (out_dir / "rows.csv").exists():
            prev = pd.read_csv(out_dir / "rows.csv")
            rows = prev.to_dict("records")
            done = {(int(r["task_id"]), int(r["seed"]), r["mode"], r["setting"]) for r in rows}
    else:
        out_dir.mkdir(parents=True)
        man_p.write_text(json.dumps({"code_sha256": code_sha, "started_utc": datetime.now(timezone.utc).isoformat(),
                                     "limit": args.limit}, indent=2) + "\n", encoding="utf-8")
    log = open(out_dir / "run.log", "a", encoding="utf-8")

    def say(msg):
        line = f"[{datetime.now().strftime('%m-%d %H:%M:%S')}] {msg}"
        print(line, flush=True)
        log.write(line + "\n")
        log.flush()

    base = rp.PipelineConfig(data_root=".venv/softsensing_data_full", toolset="time-series-2", representation="mean", use_rep_cache=True,
                             ddpm_neg_multiplier=3.0, ddpm_max_neg=100000, lgbm_device_type="cpu", lgbm_n_estimators=400)
    _, _, _, y_tr, _, _ = rp.load_toolset_arrays(base.data_root, base.toolset, base.x_mmap_mode)
    rep_tr = rp.load_cached_representation(base, "train")
    cases = sorted(RUN.glob("task*_seed*/case.json"))
    if args.limit:
        cases = cases[: args.limit]
    for case in cases:
        meta = json.loads(case.read_text(encoding="utf-8"))
        task, seed = int(meta["task_id"]), int(meta["seed"])
        mtr, ytr = rp.extract_task_binary(y_tr, task)
        Xtr = np.asarray(rep_tr[mtr], dtype=np.float32)
        prep = rp.TabularPreprocessor(drop_constant=True, imputer_strategy="mean")
        prep.fit(Xtr)
        ctx = {"X_train": prep.transform_classifier(Xtr), "y_train": ytr, "task_id": task}
        cfg = replace(base, seed=seed)
        lo, hi = np.quantile(ctx["X_train"], 0.01, axis=0), np.quantile(ctx["X_train"], 0.99, axis=0)
        for mode in ("positive_only", "conditional_mixed"):
            arrays = rp.prepare_s3_training_arrays(ctx, cfg, mode)
            Xd, yd = arrays["X_ddpm_train"], arrays["y_ddpm_train"]
            hold = np.zeros(len(yd), dtype=bool)
            hold[np.asarray(meta["generator_holdout_rows"][mode], dtype=int)] = True
            ref = [d for d in meta["generator_diagnostics"] if d["grid"] == "A" and d["mode"] == mode][0]
            got = (len(yd), int(yd.sum()), int((hold & (yd == 1)).sum()))
            if got != (ref["fit_rows"], ref["fit_pos"], ref["n_pos_hold"]):
                raise RuntimeError(f"과제 {task} seed {seed} {mode}: 재생성 실행과 행이 다르다 {got} != {(ref['fit_rows'], ref['fit_pos'], ref['n_pos_hold'])}")
            qt = QuantileTransformer(n_quantiles=min(2000, int((~hold).sum())), output_distribution="normal", random_state=seed)
            qt.fit(Xd[~hold])
            Xs = qt.transform(Xd).astype(np.float32)
            n_pool = int(round(int(ytr.sum()) * 1.0))
            for name, ep in SETTINGS.items():
                if (task, seed, mode, name) in done:
                    continue
                t0 = time.time()
                aug = TorchTabDDPMConditionalAugmenter(seed=seed, cfg=archived_cfg(ep))
                with contextlib.redirect_stdout(io.StringIO()):
                    aug.fit(Xs[~hold], yd[~hold])
                fit_s = round(time.time() - t0, 1)
                hp = hold & (yd == 1)
                hl = archived_holdout_loss(aug, Xs[hp], yd[hp])
                X_syn = np.clip(qt.inverse_transform(aug.sample(n_pool, y_label=1)), lo, hi).astype(np.float32)
                gen = {"hold": hold, "fit_y": yd}
                fid = R.generator_fidelity(X_syn, arrays, gen, ctx, cfg, seed)
                rows.append({"task_id": task, "seed": seed, "mode": mode, "setting": name, "fit_seconds": fit_s, "total_seconds": round(time.time() - t0, 1),
                             "loss_model": hl["model"], "loss_zero": hl["zero"], "loss_gauss": hl["gauss_indep"], **fid})
                say(f"과제 {task} seed {seed} {mode} {name}: 손실 {hl['model']:.3f} (0-예측 {hl['zero']:.3f}, 독립 {hl['gauss_indep']:.3f}) "
                    f"C2ST {fid.get('c2st_hold', float('nan')):.3f} TSTR내부 {fid.get('tstr_internal_syn', float('nan')):.4f}")
                pd.DataFrame(rows).to_csv(out_dir / "rows.csv", index=False)
                torch.cuda.empty_cache()
    raw = (out_dir / "rows.csv").read_bytes()
    (out_dir / "rows.sha256").write_text(hashlib.sha256(raw).hexdigest() + "\n", encoding="utf-8")
    (out_dir / "meta.json").write_text(json.dumps({"finished_utc": datetime.now(timezone.utc).isoformat(), "settings": SETTINGS, "limit": args.limit,
                                               "code_sha256": code_sha}, indent=2) + "\n",
                                   encoding="utf-8")
    say("끝")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
