"""다시 만든 보관 자료의 생성기 진단을 표로 모은다 — 제출한 Seagate 논문(MDPI electronics-4573451) 수정 (2026-10).

입력: 실행기(run_regen_archive_v1.py)가 사례마다 남긴 case.json 의 generator_diagnostics. test 자료는 쓰지 않는다.
출력(한 번만 쓰는 폴더):
- generator_diagnostics_rows.csv : 사례 × 격자 × 모드 한 행
- generator_diagnostics_by_task.csv : 과제 × 격자 × 모드 요약 (seed 5개 평균·최소·최대)
- learning_check.csv : 사전 등록 3절 기준 1 을 만족하지 못한 생성기 목록 (없으면 빈 표).
  기준: 떼어 둔 불량에서 연속 부분 손실 < 특징 독립 가우시안 기준, 그리고 이진 부분 손실 < min(주변 비율 기준, 복사 기준).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

COLS = ["best_step", "steps_run", "fit_seconds", "loss_cont_model", "loss_cont_zero", "loss_cont_gauss", "loss_bin_model",
        "loss_bin_marginal", "loss_bin_copy", "bin_rate_fit_mean", "bin_rate_syn_mean", "bin_rate_abs_err", "c2st_hold", "ks_hold",
        "corr_err_hold", "dcr_ratio", "mem_p05", "tstr_internal_syn", "tstr_internal_real", "n_const_syn"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()
    run_dir, out = Path(args.run_dir), Path(args.out_dir)
    if out.exists():
        print(f"출력 폴더가 이미 있다: {out}")
        return 1
    rows = []
    for case in sorted(run_dir.glob("task*_seed*/case.json")):
        meta = json.loads(case.read_text(encoding="utf-8"))
        for d in meta["generator_diagnostics"]:
            hl = d.get("holdout_pos_loss") or {}
            rows.append({"task_id": meta["task_id"], "seed": meta["seed"], "grid": d["grid"], "mode": d["mode"],
                         "n_pos_fit": d.get("n_pos_fit"), "n_pos_hold": d.get("n_pos_hold"), "pool_size": d.get("pool_size"),
                         "n_cont": d.get("n_cont"), "n_bin": d.get("n_bin"),
                         "best_step": d.get("best_step"), "steps_run": d.get("steps_run"), "fit_seconds": d.get("fit_seconds"),
                         "loss_cont_model": hl.get("cont_model"), "loss_cont_zero": hl.get("cont_zero"), "loss_cont_gauss": hl.get("cont_gauss_indep"),
                         "loss_bin_model": hl.get("bin_model"), "loss_bin_marginal": hl.get("bin_marginal"), "loss_bin_copy": hl.get("bin_copy"),
                         "bin_rate_fit_mean": d.get("bin_rate_fit_mean"), "bin_rate_syn_mean": d.get("bin_rate_syn_mean"),
                         "bin_rate_abs_err": d.get("bin_rate_abs_err"),
                         **{k: d.get(k) for k in ["c2st_hold", "ks_hold", "corr_err_hold", "dcr_ratio", "mem_p05",
                                                  "tstr_internal_syn", "tstr_internal_real", "n_const_syn"]}})
    if not rows:
        raise SystemExit("case.json 이 없다")
    out.mkdir(parents=True)
    df = pd.DataFrame(rows).sort_values(["task_id", "grid", "mode", "seed"])
    df.to_csv(out / "generator_diagnostics_rows.csv", index=False)
    agg = df.groupby(["task_id", "grid", "mode"]).agg(
        n_seeds=("seed", "nunique"), n_pos_fit=("n_pos_fit", "first"), n_pos_hold=("n_pos_hold", "first"),
        **{f"{c}_mean": (c, "mean") for c in COLS}, **{f"{c}_min": (c, "min") for c in COLS}, **{f"{c}_max": (c, "max") for c in COLS}
    ).reset_index()
    agg.to_csv(out / "generator_diagnostics_by_task.csv", index=False)
    cont_ok = (df["loss_cont_model"] < df["loss_cont_gauss"]) | df["loss_cont_model"].isna()
    bin_ok = (df["loss_bin_model"] < df[["loss_bin_marginal", "loss_bin_copy"]].min(axis=1)) | df["loss_bin_model"].isna()
    fail = df[~(cont_ok & bin_ok)]
    fail.to_csv(out / "learning_check.csv", index=False)
    print(f"생성기 {len(df)}개 중 학습 확인 기준을 만족하지 못한 것 {len(fail)}개")
    print(agg[["task_id", "grid", "mode", "n_pos_fit", "best_step_mean", "loss_cont_model_mean", "loss_cont_gauss_mean",
               "loss_bin_model_mean", "loss_bin_marginal_mean", "c2st_hold_mean",
               "dcr_ratio_mean", "tstr_internal_syn_mean", "tstr_internal_real_mean"]].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
