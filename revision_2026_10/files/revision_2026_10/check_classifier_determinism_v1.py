"""분류기 결정성 확인 기록 — 계열 A (MDPI electronics-4573451) 수정, 원고 4.8절 근거.

같은 입력으로 같은 분류기를 두 번 학습해 검증 자료 예측 확률이 같은지 본다.
- LightGBM CPU (재생성 실험 설정: run_pipeline.train_lgbm, device cpu)
- LightGBM GPU (보관 실험 설정: device gpu)
- XGBoost CUDA (재생성 실험 설정: run_regen_archive_v1.train_xgb)
과제 0, seed 42, 재생성 실험과 같은 전처리, 학습 자료는 S0(실제 행). test 자료는 쓰지 않는다.
출력(한 번만 씀): outputs/revision_2026_10/classifier_determinism_v1/result.json
"""

from __future__ import annotations

import contextlib
import hashlib
import importlib.util
import io
import json
import os
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
SEAGATE = HERE.parent
ROOT = SEAGATE.parents[1]
OUT = SEAGATE / "outputs/revision_2026_10/classifier_determinism_v1"
spec = importlib.util.spec_from_file_location("R", HERE / "run_regen_archive_v1.py")
R = importlib.util.module_from_spec(spec)
with contextlib.redirect_stdout(io.StringIO()):
    spec.loader.exec_module(R)
rp = R.rp


def main() -> int:
    os.chdir(ROOT)
    if OUT.exists():
        print(f"출력 폴더가 이미 있다: {OUT}")
        return 1
    OUT.mkdir(parents=True)
    task, seed = 0, 42
    base = rp.PipelineConfig(data_root=".venv/softsensing_data_full", toolset="time-series-2", representation="mean", use_rep_cache=True,
                             lgbm_n_estimators=400, seed=seed)
    _, X_val, _, y_tr, y_va, _ = rp.load_toolset_arrays(base.data_root, base.toolset, base.x_mmap_mode)
    rep_tr = rp.load_cached_representation(base, "train")      # 재생성 실행과 같은 방식(run_regen_archive_v1.main)
    rep_va = rp.build_representation(X_val, "mean").astype(np.float32)
    mtr, ytr = rp.extract_task_binary(y_tr, task)
    mva, _ = rp.extract_task_binary(y_va, task)
    prep = rp.TabularPreprocessor(drop_constant=True, imputer_strategy="mean")
    prep.fit(np.asarray(rep_tr[mtr], dtype=np.float32))
    Xtr = prep.transform_classifier(np.asarray(rep_tr[mtr], dtype=np.float32))
    Xva = prep.transform_classifier(np.asarray(rep_va[mva], dtype=np.float32))
    res = {}
    fits = {"lightgbm_cpu": lambda: rp.train_lgbm(Xtr, ytr, replace(base, lgbm_device_type="cpu"), seed),
            "lightgbm_gpu": lambda: rp.train_lgbm(Xtr, ytr, replace(base, lgbm_device_type="gpu"), seed),
            "xgboost_cuda": lambda: R.train_xgb(Xtr, ytr, seed)}
    for name, fit in fits.items():
        try:
            p = [fit().predict_proba(Xva)[:, 1] for _ in range(2)]
            res[name] = {"max_abs_diff": float(np.max(np.abs(p[0] - p[1]))), "identical": bool(np.array_equal(p[0], p[1])),
                         "n_val_rows": int(len(Xva))}
        except Exception as exc:  # GPU 판이 없으면 그 사실을 기록한다
            res[name] = {"error": f"{type(exc).__name__}: {exc}"}
        print(name, res[name])
    import lightgbm
    import xgboost
    out = {"task_id": task, "seed": seed, "train_rows": int(len(ytr)), "results": res,
           "versions": {"lightgbm": lightgbm.__version__, "xgboost": xgboost.__version__},
           "created_utc": datetime.now(timezone.utc).isoformat(),
           "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (OUT / "result.json").write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
