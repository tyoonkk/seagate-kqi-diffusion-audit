"""재생성 실행 진행 상황을 한 번에 보여 준다 (읽기 전용) — 계열 A 수정 실험.

사용: python regen_status.py  → 끝난 사례 수, 사례별 걸린 시간, 남은 사례, 예상 종료 시각, 오류 여부
"""

from __future__ import annotations

import re
import sys
from datetime import datetime, timedelta
from pathlib import Path

RUN = Path(__file__).resolve().parents[1] / "outputs/revision_2026_10/regen_archive_v1_run1"
GROUPS = {"A": [10, 9, 4, 0, 3], "B": [7, 2, 6, 1, 5, 8]}
SEEDS = [42, 43, 44, 45, 46]


def main() -> int:
    log = (RUN / "run.log").read_text(encoding="utf-8").splitlines() if (RUN / "run.log").exists() else []
    starts, ends = {}, {}
    for line in log:
        m = re.match(r"\[(\d\d-\d\d \d\d:\d\d:\d\d)\] 과제 (\d+) seed (\d+): (학습|끝)", line)
        if m:
            ts = datetime.strptime("2026-" + m.group(1), "%Y-%m-%d %H:%M:%S")
            key = (int(m.group(2)), int(m.group(3)))
            (starts if m.group(4) == "학습" else ends)[key] = ts
    done = sorted(k for k in ends)
    errors = [l for l in log if "오류" in l]
    now = datetime.now()
    print(f"끝난 사례 {len(done)}/55, 오류 {len(errors)}건")
    for g, tasks in GROUPS.items():
        todo = [(t, s) for t in tasks for s in SEEDS]
        fin = [k for k in todo if k in ends]
        durs = [(ends[k] - starts[k]).total_seconds() / 60 for k in fin if k in starts]
        cur = [k for k in todo if k in starts and k not in ends]
        print(f"  프로세스 {g}: {len(fin)}/{len(todo)} 끝" + (f", 진행 중 과제 {cur[-1][0]} seed {cur[-1][1]}" if cur else ""))
        for k in fin:
            print(f"    과제 {k[0]} seed {k[1]}: {round((ends[k] - starts[k]).total_seconds() / 60, 1)}분")
    if errors:
        print("오류 줄:", *errors[-3:], sep="\n  ")
    return 0


if __name__ == "__main__":
    sys.exit(main())
