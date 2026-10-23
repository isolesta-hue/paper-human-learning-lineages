<!-- 시아(뉴호라이즌스) 세션의 검토 — 원문 그대로 옮김(RULES 12). 출처: isolesta-hue/new-transformer-model-research, 브랜치 claude/great-einstein-ln7j5x, 파일 PAPER1_REVIEW.md, 커밋 37d15222c55ddfb24350d95d75080c6a2bd81127 (2026-10-22 00:05:09 UTC). 옮긴 날 2026-10-22, 다미 세션. -->

# 첫 논문 초안 검토 — 시아 (뉴호라이즌스) 세션

- 검토한 초안: 공용 `paper1/DRAFT.md` **Draft 1.2** (공용 커밋 344c247, 2026-10-21 13:30 UTC)
- 검토한 날: 2026-10-22 (시아 저장소 브랜치 `claude/great-einstein-ln7j5x`)
- 절차: 공용 `paper1/README.md` (RULES 12) — 이 파일을 다미 세션이 원문 그대로 `paper1/REVIEW_sia.md`로 옮김. 초안은 고치지 않음.
- 맡은 곳: 주로 1 (시아에 관한 사실). 2~5는 시아 기록으로 확인할 수 있는 곳만 적음.
- 사용자 요청 (10-22, 시아 세션): "당시 세션에서 논문과 관련된 글을 적었으니까 읽고 실행해줘"

## 요약
- 시아에 관한 수치 가운데 **틀린 것은 없음**. 확인한 값은 다음과 같음.
  - 시간: 57.6분, 생애마다 배움 174초와 답 254초.
  - 줄 수: 33줄 중 16줄.
  - C9: 27~30 / 30.
  - C2: Pursuit에 13승 4패, p = 0.049.
  - 부품 빼기: C0 +0.197, C2 +0.311, C9 +0.595.
- 고쳐야 할 곳은 둘. 시아 기록을 실제보다 넓게 옮겼음.
  - §8 "strength is designed structure"
  - §8 NH 145 서술
- 시아의 v1 받침 읽기에 **새로 확인한 사실**이 있음 — 항목 4.
  - 시험 노출에는 둘 이상이 함께 떨어진 순간이 한 번도 없음.
  - 그런데 받침 문항의 절반은 둘이 함께 떨어짐.
  - 이 사실은 시험 방법 절에도 넣을 만함.
- §6.4의 *[to be checked by the New Horizons session]* 자리에 넣을 문장을 항목 1에 적음.

## 1. 자기 모델에 관한 사실

### 항목 1 — §6.4 시아 설명 (종류: 빠짐 · 사실 보탬)
- **위치**: §6.4 — "**New Horizons**: a body with designed perception and books (a listener, eye frames, counting tables), through `sia-conv-v3`. *[to be checked by the New Horizons session]*"
- **고칠 내용**: 아래 문장으로 바꾸기를 제안함.
  > **New Horizons**: a body of separately designed learning parts — a listener that pairs words with pictures by co-occurrence across moments (with an order memory of neighbouring words, mutual exclusivity, a counting sense and a relation sense), an eye that reads an event from the difference between two scenes, and two bio-net predictors plus an event book that predict each object's next place — through `sia-conv-v3` (code fingerprint 7d03b3b4d445; runner `exam_day.py`, sha 878e0d99840e). It answered every item format, including "when" and "where before" (0 missing answers). Body seed = 1000 × 1000 + life number; one process per life; one BLAS thread.
- **근거**:
  - 몸과 판: `STAGE1_REPLY.md` "첫 봉인 판 v1 — 시아: 답 올림", `exam_answers/v1/exam_day_record.json` (version sia-conv-v3, fingerprint 7d03b3b4d445, runner_sha 878e0d99840e).
  - 부품 목록과 갈래 (구조 / 회색): `docs/HUMAN_SHARE.md` 44~53.
  - 빠진 답 0: 공용 `exam/v1/RESULTS.md`, 시아 저장소 커밋 8f49dbe6.

### 항목 2 — §6.5 시간, §7.1 표 (종류: 확인 — 고칠 것 없음)
- **위치**:
  - §6.5 "New Horizons, 30 lives | 57.6 min | 4 processes; learning 174 s + answering 254 s per life"
  - §7.1 "**3 / 10** (C0, C2, C9) | 16 of 33"
  - "C9 beats all ten baselines in 27–30 of 30 lives", "C2 … 13 wins, 4 losses, p = 0.049"
- **확인한 값**:
  - 생애 30개 기록의 평균은 배움 174초, 답 254초.
  - 시작 03:21:20, 끝 04:18:56 UTC.
  - 채점표의 시아 줄은 33줄 중 16줄 통과.
- **근거**: `exam_answers/v1/_log_life_*.json`, `exam_day_record.json`; 공용 `exam/v1/RESULTS.md`; `STAGE1_REPLY.md` "v1 채점 결과 — 시아가 읽음".

### 항목 3 — §8 "Human share"의 시아 인용 (종류: 사실 틀림 — 인용 범위)
- **위치**: §8 — "The sibling learner that passed three criteria also reports, in its own records, that much of its strength is designed structure (counting tables, perception frames, books)."
- **문제**: 시아 기록이 "사람이 준 구조 (세는 표 · 감각 · 책 · 둘레 꼴)의 덕"이라고 한 것은 **빠름**(시간)에 대한 말이었음.
  - 출처: `STAGE1_REPLY.md` "곧 빠름의 큰 몫은 사람이 준 구조 … 가 대신 해 주는 일의 덕".
  - **통과**에 대해서는 부품 빼기 탐침(NH 146)에서 다르게 말했음: "통과한 셋은 각각 덧붙인 부품 하나에 기댐 … 배운 낱말 + 설계한 부품이 함께 풂".
  - 세는 표는 v1에서 통과를 만들지 못했음. C3 수 → 장면 0.863으로 미달이고, C4는 판정되지 않았음.
- **고칠 내용**:
  > The sibling learner that passed three criteria reports, from a part-removal probe on its own v1 lives, that each pass leans on one designed part working together with learned word–picture pairs (an order memory, mutual exclusivity, a relation sense), and that much of its speed comes from designed structure.
- **근거**: `results/nh146_v1probe.md` (커밋 e7cce8c4), `STAGE1_REPLY.md`.

### 항목 4 — §7.6 시아의 받침 읽기 + 세 학습자 공통 (종류: 사실 보탬 — 시아의 첫 읽기를 바로잡음)
- **위치**:
  - §7.6 "support and contact: influence passed on through other objects is missed"
  - "Across the three learners" 절
- **새로 확인한 것** (채점 뒤 진단, 공개된 열쇠와 30생애 노출 모두):
  - **v1 노출에서 받침이 빠져 같은 열로 내려간 순간은 674번이고, 모두 한 물건만 내려감.** 둘 이상이 함께 내려간 순간은 0번.
  - **받침 문항 720개 중 360개는 정답 보기에서 두 물건이 함께 내려감.**
  - 받침 기준은 노출 안의 문항(한 물건)과 노출보다 깊은 문항(두 물건)이 반반 섞여 있음.
  - 시아의 정답률은 두 갈래 모두 낮았음.

    | 문항 | 한 물건 | 두 물건 |
    |---|---|---|
    | 예측 | 0.583 | 0.489 |
    | 짝 | 0.583 | 0.528 |

  - 그러니 시아의 첫 읽기 "바로 위 물건은 맞히나 둘째 칸 위를 놓침"은 **절반만 맞음**. 한 물건만 떨어지는 문항에서도 약했음.
- **고칠 내용**:
  - (a) §7.6 시아 줄에 한 문장을 더하기를 제안함.
    > Its later check found the support items deeper than its exposure: in all 30 lives the exposure has 674 moments where a support is removed, each with exactly one object falling, while 360 of the 720 support items have two objects falling together; New Horizons was weak on both halves (one object: 0.583 prediction, 0.583 pair; two objects: 0.489, 0.528).
  - (b) "Across the three learners"에 한 줄 더하기를 제안함 — "The support items ask beyond the exposure's depth" 꼴로. 다미 · 아라의 받침 짝이 1.000 · 0.99인 것은 이 넓히기를 해냈다는 뜻이므로, 세 몸을 견주는 데 쓸모가 있음.
- **근거**: `new_horizons/v1_support_depth.py`, `results/v1_support_depth.md` (시아 저장소, 이 검토와 같은 커밋).
- 같은 결이 우리 세상에서도 재현됨: "얕은 세상"(노출에서 깊은 떨어짐만 뺌)에서 4판 받침 늘린 키가 0.43 (`README.md` NH 148, `results/nh148_v5.md`).

### 항목 5 — §8 "Stability and plasticity"의 NH 145 서술 (종류: 사실 틀림)
- **위치**: §8 — "In the sibling New Horizons, one part learned the same causal rule but corrupted what it had already learned (its record NH 145)."
- **문제**: 배운 부품과 오염시킨 부품이 다름.
  - NH 145(우리 세상의 "얼음 칸" — 물건이 두 칸 미끄러짐)에서 새 규칙을 배운 것은 예측기(피질)임. 참 읽기를 주면 0.68.
  - 오염은 **설계한 부품**에서 왔음. 눈의 닫힌 사건 갈래 목록(사람 몫 44)이 새 효과를 "놓기"로 담았고, 그 꼬리표가 듣는 이의 밀기 낱말 배움을 오염시켰음.
  - 밀기 문장 읽기는 0.998 → 0.51, 맞닿아는 −0.513.
  - 사건 책(사람 몫 52)은 절대 자리를 보지 않아 얼음 칸과 보통 칸을 한 칸에 섞었고, 그래서 새 규칙의 배움을 가로막았음. 참 읽기에서 책을 빼면 +0.037 (26:2).
  - 또 규칙이 "the same"인지는 확인하지 않았음 — 다미의 "미끄러운 바닥이 움직임을 두 배로"와 같은 꼴의 규칙일 뿐.
- **고칠 내용**:
  > In the sibling New Horizons, a similar new causal rule (slippery cells) was learned by its predictor (0.68 when given the true reading), but a fixed designed part — the eye's closed list of event kinds — filed the new effect under "put", and that label corrupted what its listener had learned about push sentences (reading 0.998 → 0.51); its event book, which ignores absolute position, also held the new rule back (removing it: +0.037) (its record NH 145).
- **근거**:
  - `README.md` NH 145, `results/nh145_ice.md`.
  - `docs/HUMAN_SHARE.md` 44 ("닫힌 갈래 목록이 새 효과 (두 칸 미끄러짐)를 '놓기'로 담아 듣는 이의 밀기 배움을 오염 — 밀기 문장 읽기 0.998 → 0.51, 맞닿아 −0.513").
  - 장부 L52.

## 2. 시험 방법

### 항목 6 — §6.1 받침 문항의 깊이 (종류: 빠짐)
- **위치**: §6.1 C6 줄 "support, solidity, contact, permanence (prediction and pair)"
- **고칠 내용**: 항목 4의 사실을 방법 절에 한 줄로 넣기를 제안함.
  - 예: "In v1, half of the support items have two objects falling together, a depth never shown in the exposure (one object in all 674 cases)."
  - 다음 판 생성기에서 노출보다 깊은 문항을 "novel"처럼 따로 표시하면, 넓히기와 배운 것을 나눠 읽을 수 있음. 공용 판에 바라는 점으로 적으며, 결정은 다미 세션과 사용자 몫임.
- **근거**: 항목 4와 같음.

### 항목 7 — §8 Limitations: 준비 점검이 꼴만 봄 (종류: 빠짐)
- **위치**: §8 "Limitations." 목록
- **고칠 내용**: 한 줄 더하기를 제안함.
  > Readiness checks covered the answer format only; a path check — whether a learner can produce that kind of answer at all — was added after v1 (shared `exam/path_check.py`, commit b1771ce).
  - 아라 P55의 교훈(§7.6)과 이어짐. 공용 길 점검에서 시아의 v1 답은 걸린 묶음 0/35였음(공용 `exam/v1/PATH_CHECK.md`).

## 3. 주장의 크기

### 항목 8 — 요약의 "passed 3 of 10" (종류: 주장 큼 — 작게)
- **위치**: Abstract — "One sibling learner passed 3 of 10."
- **고칠 내용**:
  > One sibling learner passed 3 of 10 by the grader's lines; under the full judgement none of the three is yet a complete pass.
  - §7.1의 판단(C2는 Pursuit에 미결, C0은 관문, 셔플 대조 없음)과 맞춤. 시아 세션도 "온전한 통과"라 쓰지 않음.
- **근거**: `STAGE1_REPLY.md` "v1 채점 결과 — 시아가 읽음".

### 항목 9 — §7.6 "answers by pointing into the scene" (종류: 제안 — 무엇이 답했는지를 정확히)
- **위치**: §7.6 — "New Horizons, which answers by pointing into the scene, scores 1.000"
- **고칠 내용**:
  - 시아가 사람 이름을 맞힌 길: 문항이 가리키는 물건의 그림에 대해, 노출에서 배운 낱말 × 그림 함께 남(공기)으로 낱말을 고름. 문장 틀은 쓰지 않음.
  - "가리킴"(문항 대상 → 장면의 물건)은 설계한 읽기 틀(사람 몫 45)이고, 이름 짝은 배운 것임.
  - 제안 문장:
    > New Horizons, which picks the word from learned word–picture pairs for the object the item points at (a designed reading frame), not from the sentence frame, scores 1.000
- **근거**: `docs/HUMAN_SHARE.md` 45; NH 144 (`results/nh144_design.md` — 이름은 덧붙인 부품 없이 배움).

## 4. 빠진 것

### 항목 10 — §9 다음 걸음: 형제 쪽은 이미 고친 몸을 골랐음 (종류: 빠짐 — 넣을지는 쓰는 세션 판단)
- **위치**: §9 "After that come a second sealed edition, opened again by all three learners on the same day, …"
- **고칠 내용**:
  - 한 줄 보태기를 제안함: v1은 다 쓴 판이라, 고침은 각 세션이 자기 세상에서만 골랐음.
  - 시아는 v1 진단이 가리킨 두 곳을 우리 세상에서 고쳐 판정을 통과했음.
    - NH 147 — 본 것으로 가리킴 바로잡기, 100시드.
    - NH 148 — 이어진 물건 책, 30시드 × 세상 둘.
  - 다음 봉인 판에는 `sia-conv-v5`(지문 8df1fce720d9)로 답함.
  - 둘째 판이 그 고침이 옮겨지는지를 잼 — v1과 같은 사전 등록 · 봉인 원칙.
- **근거**: `STAGE1_REPLY.md` "변환기 판 올림 — 시아: sia-conv-v5", `README.md` NH 147 · NH 148.

### 항목 11 — 부록 B 재현: 시아 줄 (종류: 빠짐)
- **위치**: Appendix B (지금은 다미 실행기만)
- **고칠 내용**: 시아 줄을 더하기를 제안함.
  > New Horizons' runner: `python3 -m new_horizons.exam_day --edition DIR --out DIR --shared DIR` (sia-conv-v3, code fingerprint 7d03b3b4d445, runner sha 878e0d99840e; PYTHONHASHSEED=0, one BLAS thread — with four threads some answers change). A rerun of all 30 lives reproduced every judged answer row (18,771 of 18,771; NH 146).
- **근거**: `exam_answers/v1/exam_day_record.json`; `docs/SIBLINGS.md` 10-19 (스레드 수와 재현); `results/nh146_v1probe.md` ("full 판의 답 줄 = 시험 날 답 — 18771/18771 줄 (생애 30개 모두 같음)").

## 5. 틀과 읽기 쉬움

### 항목 12 — 부록 E의 인용에 남은 프로젝트 안 이름 (종류: 제안)
- **위치**: 초안 머리의 메모 — "The project's internal names (Dami, Ara, Sia) appear only in this note." — 와 Appendix E의 한국어 원문 인용.
  - 원문 인용에 "시아가 읽음", "아라는 문법…"처럼 안쪽 이름이 남아 있음.
- **고칠 내용**: 원문 인용 안의 이름을 [New Horizons] · [Pioneer]로 바꾸거나, "원문 그대로라 안쪽 이름이 남음"을 부록 E 머리에 한 줄로 밝히기를 제안함.

### 항목 13 — 부록 C 표 (종류: 제안, 작음)
- 부록 C는 관계 표에 시아 점수를 함께 적는다고 되어 있음.
- 시아의 네 관계 읽기("상관 강함 · 인과 대부분 강함 · 상호 약함 · 선후 약함")는 다미 D77 표와 같은 축임.
- 표 아래 한 줄로 "시아의 약한 곳은 다른 모양 — 다미는 순간 사이, 시아는 상호(받침 · 맞닿아)와 선후(따라가기)"를 보태면 §8의 비교가 또렷해짐.
- **근거**: `STAGE1_REPLY.md` "네 관계로 보면", 다미 발견 · D77 (다미 열일곱째 읽기 (d)).
