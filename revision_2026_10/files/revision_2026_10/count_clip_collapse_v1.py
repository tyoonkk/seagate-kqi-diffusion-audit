"""보관 파이프라인의 생성 뒤 1–99% 자르기로 상수가 되는 특징의 비율 — 계열 A (MDPI electronics-4573451) 수정, 원고 4.3절 근거.

보관 코드(run_pipeline.build_s3_synth_pool)는 생성 표본을 역분위변환한 뒤 과제 학습행 전체(양성+음성)의 1·99 분위로 자른다.
두 분위가 같은 특징은 모든 합성행에서 한 값이 된다. 과제마다(상수열 제거 뒤 특징 기준) 그 수와 비율을 센다.
같은 계산을 2026-10-07 팩트체크에서 했지만(26–81%) 스크립트가 남지 않아, 기록으로 다시 계산한다. test 자료는 쓰지 않는다.
출력(한 번만 씀): outputs/revision_2026_10/clip_collapse_v1/clip_collapse.csv, meta.json
"""

from __future__ import annotations

import contextlib
import hashlib
import importlib.util
import io
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
SEAGATE = HERE.parent
ROOT = SEAGATE.parents[1]
OUT = SEAGATE / "outputs/revision_2026_10/clip_collapse_v1"
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
    base = rp.PipelineConfig(data_root=".venv/softsensing_data_full", toolset="time-series-2", representation="mean", use_rep_cache=True)
    _, _, _, y_tr, _, _ = rp.load_toolset_arrays(base.data_root, base.toolset, base.x_mmap_mode)
    rep_tr = rp.load_cached_representation(base, "train")
    rows = []
    for task in range(11):
        mtr, ytr = rp.extract_task_binary(y_tr, task)
        Xtr = np.asarray(rep_tr[mtr], dtype=np.float32)
        prep = rp.TabularPreprocessor(drop_constant=True, imputer_strategy="mean")
        prep.fit(Xtr)
        X = prep.transform_classifier(Xtr)
        lo, hi = np.quantile(X, 0.01, axis=0), np.quantile(X, 0.99, axis=0)   # run_pipeline 1989–1991 과 같은 계산
        collapsed = lo == hi
        pos = X[ytr == 1]
        out_of_range = ((pos < lo) | (pos > hi)).any(axis=1)
        rows.append({"task_id": task, "n_features": int(X.shape[1]), "n_collapsed": int(collapsed.sum()),
                     "pct_collapsed": float(100 * collapsed.mean()),
                     "pct_pos_rows_outside_range": float(100 * out_of_range.mean())})
        print(rows[-1])
    d = pd.DataFrame(rows)
    d.to_csv(OUT / "clip_collapse.csv", index=False)
    (OUT / "meta.json").write_text(json.dumps({"created_utc": datetime.now(timezone.utc).isoformat(),
                                               "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                                               "csv_sha256": hashlib.sha256((OUT / "clip_collapse.csv").read_bytes()).hexdigest()},
                                              indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
