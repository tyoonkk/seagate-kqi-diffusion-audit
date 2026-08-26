# Matched-objective gate audit — 사전 고정 스펙 (결과 열람 전 작성)

작성: 2026-08-20, Claude Code (연구 담당; Codex 부재 기간이라 자기검증 보완용으로
이 스펙을 실행 **전에** 고정한다)
목적: IEEE Access revision 의 R1 예상 지적 — primary negative (nested) 와
secondary positive (equal-budget LOTO) 가 서로 다른 candidate archive /
gate family / selection objective 를 써서 결과 차이의 원인을 분해할 수 없다 —
에 대한 attribution 분석.

## 1. 무엇을 계산하는가

입력 (기존 아카이브, 수정 금지):
- experiments/seagate_kqi/artifacts/condition_aware_main/
  selector_v12_final_validation_report_run1/
  selector_v12_final_candidates_with_task_context.csv  (45 case)
- 동일 폴더의 selector_v12_final_selection_results.csv (legacy 재현 검증용)

분석 단위: V10 equal-budget archive 의 45 case (9 task x 5 seed).
각 case = 24-candidate grid 의 validation-best hybrid 1개 + safe baseline.
결정은 이진: hybrid 배치 또는 reference fallback.

Gate family 두 가지:
- legacy5: build_ieee_v10_uniform_grid_loto_audit.py 의 5개 gate 를 문자 그대로 재사용
  (safe_only, strict_certificate, v11_grid_or_profile_strict,
   v11_rare_signal_aggressive_else_strict, v12_signal_aggressive_else_strict)
- nested7-veto: build_ieee_task_independent_nested_audit.py 의 7개 gate 를
  case-level veto 로 적용. 원본은 "gate 통과 후보 중 val-best 선택"이고,
  여기서는 후보가 이미 val-argmax 로 고정돼 있으므로 "고정 후보가 gate 를
  통과하면 배치, 아니면 fallback". 원본보다 보수적인 semantics 임을 명시 보고.

Selection objective 세 가지 (outer LOTO, 각 fold 의 개발 8-task 데이터만 사용):
- conservative: negative_row_count==0 AND negative_task_count==0 인 gate 만 자격,
  자격자 중 dev task-macro 최대. 자격자 없으면 reference_only.
  (nested 스크립트 choose_gate 와 동일 tie-break)
- mean_only: dev task-macro 최대. (동일 tie-break)
- harm_first: 기존 LOTO 스크립트 choose_rule 과 동일
  (harm asc, macro desc, min-task desc, coverage desc, name asc).

총 6 arm = 2 family x 3 objective. **6개 전부 보고한다. 선택 보고 금지.**

고정 상수: EXPECTED_TASKS=(0,1,2,3,5,6,7,9,10), SEEDS=(42..46),
BOOTSTRAP_SEED=20260714, REPS=10000 (기존 두 스크립트와 동일).

출력: artifacts/condition_aware_main/ieee_matched_objective_gate_audit_run1/
(새 디렉토리; 기존 run1 산출물 덮어쓰기 금지)

무결성 fail-closed:
- 45행, task/seed 집합 일치, task-seed 중복 없음 아니면 중단
- legacy5 를 full archive 에 적용한 결과가
  legacy_rule_summary.csv 의 macro/harm/coverage 를 재현하지 못하면 중단

## 2. 결과 열람 전 해석 규칙

이미 아카이브된 legacy_rule_summary.csv 로부터 논리적으로 예측되는 것:
harm 0 gate 가 3개 존재하므로 conservative 는 legacy5 에서 abstain 하지 않는다.
이 예측이 맞든 틀리든 아래 규칙대로 보고한다.

- (i) conservative 가 V10 archive 에서 gate 를 선택하고 heldout macro > 0:
  "objective 가 아니라 archive/gate-family 가 negative/positive 차이의
  원인"이라는 attribution 으로 보고. primary negative 주장 유지.
  초록의 conservative abstention 문장에 archive 범위 한정어 추가.
- (ii) conservative 가 V10 에서도 abstain: 원 논문 서사 강화로 보고.
- (iii) nested7-veto 가 V10 에서 harm 을 만들면: gate family 이식 실패의
  추가 증거로 보고.
- (iv) 어떤 결과든 revision 에는 "새 retrospective sensitivity,
  historically test-exposed gate family, prospective 아님" 라벨을 붙인다.
- (v) 결과가 어느 방향이든 primary(nested V9)와 secondary(LOTO V10)의
  기존 수치·역할은 변경하지 않는다. 이 분석은 추가 감사이지 대체가 아니다.

## 3. 한계 (사전 인지)

- V10 archive 의 gate family 는 historically test-exposed. 이 분석은 그
  오염을 제거하지 못하며, objective/family 축의 교란만 분해한다.
- 후보 단위 nested 재실행은 24후보 중 23개의 test 결과 부재로 불가능
  (synthetic matrix 미보존). 이 사실 자체를 revision 한계 문단에 명시.
