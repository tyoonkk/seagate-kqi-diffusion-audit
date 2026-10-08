"""재생성 실험 동결 기록을 만든다 — 제출한 Seagate 논문(MDPI electronics-4573451) 수정 (2026-10).

사전 등록 문서와 실행·분석에 쓰는 모든 코드·설정·입력 표의 SHA-256 을 한 파일에 적는다. 이 파일을 만든 뒤에는
기록된 파일을 고치지 않는다. 고쳐야 하면 새 판(v2)을 만들고 그 사실을 남긴다.
"""

from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
SEAGATE = HERE.parent
ROOT = SEAGATE.parents[1]
OUT = HERE / "REGEN_FREEZE_V1.json"

FILES = [
    HERE / "REGEN_PROTOCOL_V1_KO.md",
    HERE / "run_regen_archive_v1.py",
    HERE / "augmenters_torch_ddpm_v2.py",
    HERE / "augmenters_torch_ddpm_v3.py",
    HERE / "generator_config_v3_final.json",
    HERE / "build_regen_audits_v1.py",
    HERE / "validate_regen_audits_v1.py",
    HERE / "build_regen_generator_table_v1.py",
    SEAGATE / "run_pipeline.py",
    SEAGATE / "build_ieee_task_independent_nested_audit.py",
    SEAGATE / "build_ieee_v10_uniform_grid_loto_audit.py",
    SEAGATE / "build_ieee_matched_objective_gate_audit.py",
    SEAGATE / "analyze_task10_v9_targeted_sweep.py",
    SEAGATE / "outputs/revision_2026_10/importance_all11_v1/task_feature_importance.csv",
    SEAGATE / "outputs/revision_2026_10/importance_all11_v1/MERGE_RECORD.json",
]


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main() -> int:
    if OUT.exists():
        print(f"동결 기록이 이미 있다: {OUT}")
        return 1
    missing = [str(p) for p in FILES if not p.exists()]
    if missing:
        print("없는 파일:", missing)
        return 1
    rec = {"schema_id": "seagate-revision-regen-freeze-v1", "lineage": "Seagate negative-audit (제출한 논문, MDPI electronics-4573451)",
           "frozen_utc": datetime.now(timezone.utc).isoformat(),
           "files": {str(p.relative_to(ROOT)).replace("\\", "/"): sha(p) for p in FILES},
           "note": "동결 뒤 이 파일들을 고치지 않는다. 결과를 보기 전에 만든 기록이다."}
    OUT.write_text(json.dumps(rec, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(rec, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
