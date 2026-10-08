"""수정 원고 표 1 (과제 목록) 수치를 원자료에서 만든다 — 계열 A (MDPI electronics-4573451) 수정, 심사위원 1 의견 3.

열: 분할별 유효 행 수와 불량 행 수(개수), 학습 불량 비율(%), 상수 열을 지운 뒤 학습에 쓰는 특징 수(개수), 그중 0/1 이진 특징 비율(%).
출력: table1_inventory_r1.csv, table1_inventory_r1.json (원고 생성기가 읽는다). 기존 표 1 수치와 대조해 다르면 멈춘다.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
SEAGATE = HERE.parents[1]
ROOT = SEAGATE.parents[1]
sys.path[:0] = [str(SEAGATE), str(ROOT / ".venv")]
import run_pipeline as rp  # noqa: E402

OLD_TABLE1 = {0: (4714, 151, 941, 67, 1046, 50), 1: (22059, 584, 2951, 76, 2574, 113), 2: (62068, 1766, 11954, 188, 6891, 115),
              3: (19863, 383, 4639, 91, 3889, 108), 4: (7220, 169, 1351, 27, 1328, 51), 5: (23007, 730, 2438, 63, 2776, 91),
              6: (43809, 1695, 6602, 214, 5618, 199), 7: (62750, 1565, 9114, 221, 7625, 230), 8: (18853, 546, 2814, 71, 2282, 27),
              9: (19957, 192, 3050, 35, 2914, 43), 10: (266230, 2798, 49235, 505, 42655, 489)}


def main() -> int:
    import os
    os.chdir(ROOT)
    cfg = rp.PipelineConfig(data_root=".venv/softsensing_data_full", toolset="time-series-2", representation="mean", use_rep_cache=True)
    Xtr, Xva, Xte, ytr, yva, yte = rp.load_toolset_arrays(cfg.data_root, cfg.toolset, cfg.x_mmap_mode)
    rep = rp.load_cached_representation(cfg, "train")
    rows = []
    for t in range(11):
        m_tr, y_tr = rp.extract_task_binary(ytr, t)
        _, y_va = rp.extract_task_binary(yva, t)
        _, y_te = rp.extract_task_binary(yte, t)
        X = np.asarray(rep[m_tr], dtype=np.float32)
        prep = rp.TabularPreprocessor(drop_constant=True, imputer_strategy="mean")
        prep.fit(X)
        Xc = prep.transform_classifier(X)
        binary = np.array([np.isin(np.unique(Xc[:, j]), [0.0, 1.0]).all() for j in range(Xc.shape[1])])
        r = {"task_id": t, "train_rows": int(len(y_tr)), "train_pos": int(y_tr.sum()), "val_rows": int(len(y_va)), "val_pos": int(y_va.sum()),
             "test_rows": int(len(y_te)), "test_pos": int(y_te.sum()), "train_pos_rate_pct": round(100 * float(y_tr.mean()), 2),
             "n_features": int(Xc.shape[1]), "binary_feature_pct": round(100 * float(binary.mean()), 1)}
        old = OLD_TABLE1[t]
        if (r["train_rows"], r["train_pos"], r["val_rows"], r["val_pos"], r["test_rows"], r["test_pos"]) != old:
            raise SystemExit(f"과제 {t}: 기존 표 1 과 다르다 {old}")
        rows.append(r)
    df = pd.DataFrame(rows)
    df.to_csv(HERE / "table1_inventory_r1.csv", index=False)
    raw = (json.dumps(rows, indent=2) + "\n").encode("utf-8")
    (HERE / "table1_inventory_r1.json").write_bytes(raw)
    print(df.to_string(index=False))
    print("sha256", hashlib.sha256(raw).hexdigest()[:16])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
