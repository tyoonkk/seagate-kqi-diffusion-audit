"""time-series-2 의 범주형 원-핫 블록을 학습 자료에서 찾는다 — 제출한 Seagate 논문(MDPI electronics-4573451) 수정 (2026-10).

공개 자료는 범주형 변수를 원-핫으로 펼쳐 열 이름 없이 내보냈다(Seagate README: 이진 특징은 원-핫 범주형). 같은 변수의 열은 붙어 있고,
한 행에서 1 이 많아야 하나다(변수가 없는 행은 모두 0). 원래 열 순서를 따라가며, 모든 값이 0/1 인 열을 현재 블록에 더했을 때
어떤 학습행에서도 1 이 두 개 겹치지 않으면 같은 블록으로 묶는다. 겹치면 새 블록을 시작한다.
(서로 다른 변수가 우연히 한 번도 함께 1 이 아니면 한 블록으로 묶일 수 있다. 그래도 블록 안에서 1 이 많아야 하나라는 성질은 지켜진다.)

학습 분할(457,163 행)의 평균 표현만 쓴다. 출력: outputs/revision_2026_10/categorical_blocks_v1/blocks.json (+ sha256)
"""

from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
SEAGATE = HERE.parent
ROOT = SEAGATE.parents[1]
sys.path[:0] = [str(SEAGATE), str(ROOT / ".venv")]
import run_pipeline as rp  # noqa: E402

OUT = SEAGATE / "outputs/revision_2026_10/categorical_blocks_v1"


def main() -> int:
    import os
    os.chdir(ROOT)
    if OUT.exists():
        print(f"출력 폴더가 이미 있다: {OUT}")
        return 1
    cfg = rp.PipelineConfig(data_root=".venv/softsensing_data_full", toolset="time-series-2", representation="mean", use_rep_cache=True)
    rep = np.asarray(rp.load_cached_representation(cfg, "train"), dtype=np.float32)
    n, d = rep.shape
    is01 = np.array([bool(np.all((rep[:, j] == 0) | (rep[:, j] == 1))) for j in range(d)])
    block = np.full(d, -1, dtype=int)
    bid, run_any, last = -1, None, -2
    for j in range(d):
        if not is01[j]:
            run_any, last = None, -2
            continue
        col = rep[:, j] > 0
        if run_any is not None and j == last + 1 and not np.any(run_any & col):
            run_any |= col
        else:
            bid += 1
            run_any = col.copy()
        block[j] = bid
        last = j
    blocks = []
    for b in range(bid + 1):
        cols = np.flatnonzero(block == b)
        s = (rep[:, cols] > 0).sum(1)
        blocks.append({"block": b, "first_col": int(cols[0]), "last_col": int(cols[-1]), "size": int(len(cols)),
                       "share_rows_with_one": round(float((s == 1).mean()), 6), "max_ones_in_row": int(s.max())})
    OUT.mkdir(parents=True)
    rec = {"schema_id": "seagate-ts2-categorical-blocks-v1", "created_utc": datetime.now(timezone.utc).isoformat(),
           "source": "time-series-2 학습 분할 평균 표현 캐시", "n_rows": int(n), "n_cols": int(d),
           "n_binary_cols": int(is01.sum()), "n_blocks": int(bid + 1),
           "mean_ones_per_row": round(float((rep[:, is01] > 0).sum(1).mean()), 6),
           "block_of_column": block.tolist(), "blocks": blocks,
           "code_sha256": hashlib.sha256(Path(__file__).resolve().read_bytes()).hexdigest()}
    raw = (json.dumps(rec, ensure_ascii=False, indent=1) + "\n").encode("utf-8")
    (OUT / "blocks.json").write_bytes(raw)
    (OUT / "blocks.sha256").write_text(hashlib.sha256(raw).hexdigest() + "\n", encoding="utf-8")
    print(f"0/1 열 {int(is01.sum())}개, 블록 {bid + 1}개, 행당 1 의 평균 {rec['mean_ones_per_row']}")
    for b in blocks:
        print(b)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
