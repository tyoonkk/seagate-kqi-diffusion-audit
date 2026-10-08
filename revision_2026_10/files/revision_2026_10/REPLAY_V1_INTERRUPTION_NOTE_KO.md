# 보관 생성기 재학습 진단 v1 중단 기록 — 계열 A (제출한 Seagate 논문, MDPI electronics-4573451) 수정

작성: 2026-10-07

- `diag_archived_generator_replay_v1.py` 본 실행(출력 `outputs/revision_2026_10/archived_generator_replay_v1/`)은
  도구(Claude Code)의 백그라운드 명령 기본 시간 제한 30분에 걸려 23:39 경 강제로 멈췄다. 실행자가 시간 제한을 늘리지 않은 실수다.
  결과를 보고 멈춘 것이 아니다. 멈출 때까지 127행(220행 중)이 `rows.csv` 에 써졌고, `rows.sha256`·`meta.json` 은 없다.
- v1 출력 폴더는 고치지 않고 그대로 둔다.
- 계산은 그대로 두고 이어 돌리기(같은 코드 해시일 때만, `manifest.json` 대조)만 더한 `diag_archived_generator_replay_v2.py` 를 만들어
  새 폴더 `outputs/revision_2026_10/archived_generator_replay_v2/` 에서 55개 사례 전체를 처음부터 다시 계산한다.
  두 판의 손실 함수와 계산 부분은 같다(diff 확인).
- 재현성: v2 시험 실행(과제 0 seed 42, 4행, 임시 폴더)의 값이 v1 의 같은 행과 완전히 같았다(손실·C2ST·TSTR·상수 특징 수의 최대 차 0.0).
  본 실행이 끝나면 v1 의 127행 전체를 v2 와 대조해 이 기록 끝에 덧붙인다.
- 원고 수치는 v2 에서만 가져온다(`paper/mdpi_electronics_2026/build_r1_text_numbers.py`).

## 대조 결과 (v2 끝난 뒤 덧붙임, 2026-10-08)

- v2 는 220행(55 사례 × 두 모드 × 두 보관 설정)을 모두 끝냈다. rows.sha256 = 23d6241813b31079aaf5170dc71ad3b818190dd745cedc5cdfad5c75519835ef.
- v1 의 127행 전체를 v2 의 같은 행(과제·seed·모드·설정)과 대조했다. 손실(모델·0-예측·독립 가우시안), C2ST, KS, 상관 오차, DCR,
  내부 TSTR(합성·실제), 상수 특징 수의 최대 절대 차가 모두 0.0 이다. 보관 생성기 재학습은 이 환경에서 결정적이며, v1 중단은 결과에 영향이 없다.
