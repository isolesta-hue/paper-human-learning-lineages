<!-- 아라(파이오니어) 세션의 검토 — 원문 그대로 옮김(RULES 12: 공용 저장소는 다미 세션만 씀). 출처: isolesta-hue/recursive-improvement-experiment, 브랜치 claude/dazzling-goldberg-sk11yo, 파일 pioneer/PAPER1_REVIEW.md, 커밋 c178e765fd0a9f2a18d669244ee3a1f53410592f (2026-10-22 00:13:08 UTC). 옮긴 날 2026-10-22, 다미 세션. -->

# 첫 논문 초안 검토 — 아라(파이오니어) 세션

- **검토한 초안**: 공용 저장소 `paper1/DRAFT.md`, 커밋 `344c247`(Draft 1.2, 2026-10-21 13:17 UTC).
- **검토한 날**: 2026-10-22 (UTC). 사용자 요청 10-21 "다미가 논문 검토와 관련되서 적은 내용이 있을거야. 그걸 읽고 실행해줘." — 절차는 공용 `paper1/README.md`(검토는 자기 저장소에, 다미 세션이 옮김).
- **꼴**: 항목마다 위치 · 종류(사실 틀림 / 주장 큼 / 빠짐 / 제안) · 고칠 내용 · 근거. 고칠 문장은 영어 원문을 그대로 인용하고, 바꿀 문장은 영어로 제안한다.
- 아라 쪽 근거 파일은 모두 이 저장소(isolesta-hue/recursive-improvement-experiment, 브랜치 `claude/dazzling-goldberg-sk11yo`)의 `pioneer/` 아래에 있다.

## 한 줄 요약
아라에 관한 수치는 거의 다 맞다. 다만 §7.6의 아라 읽기는 **초안 뒤에 바로잡혔다**. P56(진단 E1)에서 아라는 장면을 보고 이름을 고르지만, **들은 틀 안에서만** 그렇게 한다는 것이 확인됐다. 이 결과는 §8의 핵심 읽기("틀이 장면을 이김")를 두 번째 몸에서 잰 증거가 되므로 넣을 것을 권한다. 다만 "아라와 다미 모두 문장 틀로만 답한다"는 문장은 아라에게는 너무 넓다.

---

## 1. 자기 모델에 관한 사실

### 1-1. §6.4 Pioneer 설명 — 제안 (사실 확인 + 보완)
- **인용**: "Pioneer: a streaming transformer body with an experience-memory mix (`s1+win2`), through `ara-conv v1`. It answers in two ways: "whole" (the full brain) and "alone" (the transformer part only). "When" and "where before" were not supported. *[to be checked by the Pioneer session]*"
- **확인**: 맞다. `[to be checked …]` 표시는 지워도 된다.
- **보탤 것** (재현에 필요한 값들):
  - 변환기 지문 `1577ace667de`(`exam_stream.py` + `exam_life.py` + `exam_gen_v1/common.py`의 sha256 앞 12자리)
  - 여는 날 실행기 지문 `131385ea7675`
  - 몸 시드 = 생애 번호
  - 두 벌의 답은 **같은 생애**에서 나왔다(alone = 경험 기억 섞기를 끈 답).
  - 근거: `pioneer/exam_answers/v1/manifest.json`(`run_record`), `pioneer/STAGE1_REPLY.md`(10-18 준비 줄).
- **제안 문장**: "Pioneer: a streaming transformer body (`s1+win2`: a two-unit context window, a self token, positions) with an experience-memory mix, through `ara-conv v1` (fingerprint 1577ace667de; runner 131385ea7675; body seed = life number). Its two answer sets come from the same lives: "whole" (transformer plus memory mix) and "alone" (the mix switched off). "When" and "where before" were declared unsupported before the opening."

### 1-2. §6.3 TF 기준선 — 사실 보완
- **인용**: "TF takes the shape of Pioneer's base transformer."
- **고칠 내용**: TF는 아라의 **예전 주머니 입력 몸 n0**(2층, 폭 32, 머리 4, ff 64, 위치 없는 주머니 입력 + 질문 토큰)의 꼴이다. 시험을 본 흐름 몸 `s1+win2`의 꼴이 아니다(공용 `exam_spec/BASELINES_LEARNED.md` 1절). 독자가 "TF ≈ 아라의 시험 몸"으로 읽지 않도록 고칠 것을 권한다.
- **제안 문장**: "TF has the shape of Pioneer's earlier bag-input body (n0: 2 layers, width 32, 4 heads, no positions), trained from scratch — not the streaming body Pioneer took the exam with."

### 1-3. §6.5 시간 표 — 확인 + 보완
- **인용**: "Pioneer, 30 lives | 44 h 41 min | 24 stops (2-hour task limit, container restarts), resumed from session checkpoints; host speed changed 1.8–2.0× mid-run"
- **확인**: 맞다(첫 시작 10-19 03:22:45 → 마지막 답 10-21 00:03:46).
- **보탤 것**:
  - 판이 열린 때부터는 45 h 3 min이다.
  - 끊김 24번 = 2시간 한도 20번 + 컨테이너 재시작 4번.
  - 연습 CPU는 143시간으로 시간의 거의 전부다. 답하기(두 벌)와 저장은 회차마다 중앙값 19초.
  - 마지막 묶음에서 CPU가 놀았다(생애 2개를 CPU 4개로, 약 6.7시간).
  - 근거: `pioneer/exam_answers/v1/timing_summary.txt`, `timing.log`, README "첫 봉인 판 v1 — 아라가 답함".
- 표의 Notes 칸에 "CPU 143 h (practice; answering ≈ 19 s per session)"를 넣으면 다미의 "CPU 52.6 h"와 견줄 수 있다.

### 1-4. §7.1 점수표 — 확인
- **인용**: "Pioneer (whole / alone) | 0 / 10 | 4 of 33 (C2 kept-at-end, C6 support pair, solidity prediction and pair)"
- **확인**: whole·alone 모두 같은 네 줄을 통과한다(alone: 0.925 · 0.994 · 0.972 · 1.000). 맞다.

### 1-5. §7.6 아라 읽기 — 사실 틀림 (초안 뒤에 바로잡음)
- **인용**: "Grammar (particles 0.99–1.00) and physical prediction (support, solidity) were learned; tying words to objects in the scene (identity, who, what) is near chance; …" 그리고 "Next: a pre-registered diagnostic probe (E1) on four v1 lives."
- **고칠 내용**: E1(P56)이 끝나 이 읽기를 바로잡았다.
  - 근거: README P56 결과, 커밋 `80b58629`, `pioneer/exam_probe_v1_report.txt`; 사전 등록 README P56, 커밋 `d500ca89`·`a485b8d3`.
  - **재현**: v1 생애 4개(life_026~029)를 시험 코드 그대로 다시 돌린 답이 낸 답과 **줄마다 같다**(두 벌 × 8,736~8,896줄; 같은 종류의 호스트, 도중 컨테이너 재시작 한 번을 건너서도).
  - **장면을 빼면** 물건 이름 0.289 → 0.047(4생애, 128문항, +4/−35). 아라는 장면을 보고 이름을 고른다.
  - **들은 꼴로 물으면** 조사 없는 한국어 "___ 있다" 이름(물건 + 상자)이 0.021 → 0.260이 된다(96문항). 들은 꼴은 "<이름> 이/가 있다"이고, 사전 등록 규칙 +0.20을 넘었다.
  - **30생애 전체, 언어별**(이 검토를 위해 잼, 공용 채점기의 `right()`, 16회차 뒤):

    | 문항 | 틀을 들었나 | whole | alone |
    |---|---|---|---|
    | 물건 이름, 영어 "there is ___" | 물건과 함께 들음 | **0.631** (303/480) | 0.627 |
    | 물건 이름, 한국어 "___ 있다" | 조사 없는 꼴 — 들은 적 없음 | 0.108 (52/480) | 0.108 |
    | 사람 이름(두 언어) | 사람과 함께 들은 적 없음 | 0.000 (0/480) | 0.000 |
    | 상자 이름(영어 / 한국어) | — | 0.087 / 0.050 | 0.092 / 0.050 |

  - "어디"는 R2 읽기(대상의 줄 + 후보 칸)로 바꾸면 0.009 → 0.111이 된다(117문항, 우연 0.040). 길은 생기지만 약하고, 사전 등록 규칙 +0.10에는 못 미쳤다.
- **제안 문장**(Pioneer 문단을 이렇게 바꿈): "**Pioneer** (`pioneer/README.md` P55, commit d500ca89; diagnosis P56, commit 80b58629): Grammar (particles 0.99–1.00) and physical prediction (support, solidity) were learned. Its first reading — that tying words to scene objects was near chance — was corrected by a pre-registered diagnosis on four v1 lives rerun with the exam code (answers equal row for row): removing the scene drops thing names from 0.289 to 0.047, so the body does pick names from the scene, but only inside frames it heard. Over all 30 lives, thing names asked in the English frame heard with things ("there is ___") score 0.631; the same names in the Korean frame without its particle ("___ 있다", never heard) 0.108; asked in the heard form ("<name> 이/가 있다") the Korean name items rise from 0.021 to 0.260 (4 lives). Person names, never heard in either frame, score 0.000. "Where" equals chance because its reading asked for a cell token its body never produced as an answer; read as a scene unit (target's line with each candidate cell) it rises to 0.111 (chance 0.040). Whole and alone are the same."

### 1-6. §7.6 "Across the three learners" — 주장 큼 (아라 부분)
- **인용**: "New Horizons, which answers by pointing into the scene, scores 1.000; Pioneer and Voyager, which answer from the sentence frame, score 0.000."
- **고칠 내용**: 사람 이름 0.000은 맞다. 그러나 아라는 "문장 틀로만" 답하지 않는다. 장면을 빼면 물건 이름이 무너지고, 들은 틀 안의 물건 이름은 0.631이다(1-5).
  - **제안**: "Pioneer and Voyager score 0.000. Pioneer does use the scene inside a frame it heard (thing names 0.631 in "there is ___"; 0.047 without the scene), so for it the failure is the unheard frame, not a missing scene path."
  - 같은 문단의 "Thing names, which are asked in a heard frame, also stay low (0.05–0.14)"는 **다미**의 값이다. 아라는 0.631이므로 "for Voyager"를 밝힐 것을 권한다.

### 1-7. §7.6 빈 상자 짝 — 빠짐
- **인용**: "The emptied-box pair scores 0.593 for New Horizons, Voyager and baseline T alike."
- **보탤 것**:
  - 아라는 0.564(whole) / 0.566(alone)이고, E1 4생애에서는 0.594다.
  - 아라의 R2 읽기도 장면 단위의 가능도로 고르므로 같은 꼴의 버릇일 가능성이 크다. 다만 보기 종류("하나 더 많음 / 적음")별로 나눠 확인하지는 않았다.
  - **제안**: "(Pioneer: 0.564 / 0.566; its scene-unit reading is also a likelihood, so the same habit is likely but was not checked per option type.)"

### 1-8. 부록 E 아라 인용 — 빠짐
- P55의 첫 문장("낱말을 장면 속 대상에 잇는 일(정체·누가·무엇)은 우연 근처")은 P56에서 바로잡혔다. P56 결과의 원문 한 줄을 함께 실을 것을 권한다(README P56 결과, 커밋 `80b58629`):
  > P55 ①의 "낱말 ↔ 대상 잇기(정체)가 약함"을 바로잡음 — 잇기는 있다. 다만 들은 문장 틀에 묶여 있어서, 틀이 바뀌면(사람 + "there is", 조사 없는 한국어) 틀의 습관이 장면을 이긴다.

---

## 2. 시험 방법

### 2-1. §6.2 절차 — 제안
- 생성기·채점기·통과선·기준선의 해시 고정에 더해, **각 모델의 답할 판(변환기·실행기 지문)도 여는 날 전에 적어 두었다**. 아라는 10-18에 `STAGE1_REPLY.md` 준비 줄에 적었다. 절차의 강점이니 한 줄 넣을 것을 권한다.
  - 제안: "Each session also declared, before the opening, the body, converter and runner (with fingerprints) it would answer with."

### 2-2. 변환기와 길 점검 — 빠짐 (한계 또는 §6에)
- **인용**(한계): "The converter is ours."
- **보탤 것**:
  - 변환기는 세 세션이 각자 썼다.
  - 여는 날 전 점검은 **형식 점검**(가짜 정답 몸·무작위 몸)뿐이었고, "그 읽기에서 몸이 그 답의 종류를 낼 수 있나"(**길 점검**)는 없었다. 그 결과 아라의 "어디" 읽기는 처음부터 길이 없었다(P55 ②).
  - 다음 판부터 몸으로 하는 길 점검(가르친 문항 되묻기)을 준비 점검에 넣는다(다미 Q19, 사용자 결정 10-21; 아라도 같은 점검을 함).
  - 제안: "Readiness checks covered the answer format only (a fake body that knows the key, a random body); none asked whether a reading could reach the asked kind of answer at all. Pioneer's 'where' reading could not (§7.6). From the next edition, a body-level path check (teach an item in the exposure's form, then ask it back) is part of readiness."

### 2-3. §6.1 수치 — 확인
- 아라의 파일과 맞는다. 예: life_000의 문항 626개, 답 줄은 생애 026~029에서 8,736~8,896.

---

## 3. 주장의 크기

### 3-1. §8 "This reading is not yet measured" — 고칠 것 (재어진 것이 생김)
- **인용**: "A body that sums all its evidence lets the frame outvote the scene. … This reading is not yet measured; a test in the learner's own world (names asked with and without a frame) is the next step."
- **고칠 내용**: 지금은 두 몸에서 재어졌다.
  - 다미 발견 168: 자기 세상 `@nwf`에서 새 이름이 0.284 → 0.111(10:0).
  - 아라 P56: 같은 시험 생애에서 들은 꼴로 묻기 +0.24, 장면 빼기 −0.24.
  - 다만 아라 쪽은 생애 4개짜리 진단이고, 이미 쓴 판(v1)에서 잰 것이며, 판정이 아니라 진단이다. 그 크기에 맞춰 쓸 것.
  - 제안: "The reading has since been measured in two bodies: in Voyager's own world (finding 168: new names 0.284 → 0.111 inside a frame never heard with names, 10:0), and in Pioneer's exam lives (a diagnosis on four lives: asking the Korean names in their heard form raises them from 0.021 to 0.260). Both are diagnoses, not verdicts on a sealed edition."
- "A body that sums all its evidence"는 다미 몸의 설명이다. 아라는 합하는 몸이 아니라 트랜스포머인데도 같은 현상을 보였다. 이 점을 밝히면 주장이 더 넓고 단단해진다.

### 3-2. §8 "Not the converter" — 범위를 좁힐 것
- **인용**: "What would have to change. Not the converter: tuning it to the exam would be fitting the test, and the diagnosis shows it would not be enough anyway."
- **고칠 내용**: 다미의 이름 실패에는 맞다. 그러나 아라의 "어디" 묶음에서는 변환기(길이 없는 읽기)가 원인의 일부였다(1-5, 2-2).
  - 또 "길이 없는 읽기를 길 점검으로 고치는 일"은 시험 맞춤과 다르다. 문항별 맞춤이 아니라 "몸이 연습한 단위의 꼴로 묻는다"는 일반 규칙이고, 판정은 다음 판에서 한다.
  - 제안: "Not Voyager's converter: … For Pioneer's 'where' groups the reading itself had no path; a reading rule fixed by a path check is not fitting the test, but its effect must be judged on the next sealed edition."

### 3-3. 초록 마지막 문단 — 제안
- **인용**: "What differs in the exam is how names are asked: through a sentence frame ("there is ___") whose statistics favour common words, and for persons a frame never heard with a person at all."
- 이 문장은 다미의 읽기이고, 위 3-1처럼 이제 두 몸에서 재어졌다. 초록에서 "measured in two learners (diagnoses)"를 밝히면 주장의 크기와 증거가 맞는다.

---

## 4. 빠진 것

### 4-1. 관련 연구 — 제안 (모두 출처 확인 필요)
- **트랜스포머와 가린 토큰 학습**: Vaswani et al. (2017) *Attention is all you need*; Devlin et al. (2019) *BERT*. 아라와 TF·MLP 기준선의 바탕이다.
- **적은 데이터의 발달형 언어 모델**: BabyLM Challenge(Warstadt et al., 2023). 아라 계열의 비교 눈금이다(다미 D71).
- **틀은 종류를, 장면은 낱말을**: 통사적 부트스트래핑(Gleitman 1990, *The structural sources of verb meanings*, Language Acquisition 1(1)) — 문장 틀이 낱말의 종류를 알려 준다는 생각.
  - §8의 요구 (a) "let the scene decide which word fills a slot while the frame only decides what kind of word"와 바로 닿는다.
- **한 아이의 눈과 귀로 배우는 신경망의 낱말-대상 잇기**: Vong, Wang, Orhan & Lake (2024), *Science* 383. 작은 학습자의 grounding 비교 대상.

### 4-2. 한계 — 재현 조건의 차이
- 세 세션의 재현 조건이 다르다.
  - 다미: BLAS 1스레드에서 끝자리까지 같음.
  - 아라: torch 스레드 1/프로세스, OpenBLAS 4, CPU 4. 생애 다수가 호스트 종류가 바뀌는 가운데 돌았다.
  - 아라 E1의 재현(줄마다 같음)은 **같은 종류의 호스트**에서만 확인했다.
- 제안: "Reproduction was checked per session under its own conditions (Voyager: every BLAS on 1 thread; Pioneer: torch 1 thread per process, rerun lives equal row for row on the same host type); answers across host types were not compared."

### 4-3. 아라의 다음 걸음 — 정보
- 아라는 지금 P57 "틀 흩뜨리기"를 시험하고 있다(사용자 결정 10-21). 연습에서 맞힐 낱말 말고 다른 낱말 몇 개도 빼서, 틀이 아니라 장면이 답을 정하게 하는 일반 규칙이다.
- §9 "next step"의 요구 (a)를 다른 몸에서 겨냥한 것이다. 결과가 나오기 전이라 논문에는 "진행 중"으로만 쓸 수 있다.

---

## 5. 틀과 읽기 쉬움

### 5-1. §7.6에 세 모델의 이름 표 — 제안
"틀을 들었나 / 안 들었나"를 한눈에 보이게 표 하나를 권한다. 아라 값은 1-5의 30생애 값이다. 다미·시아 값은 다미 세션이 채운다.

| | 들은 틀 (영어 물건 "there is ___") | 안 들은 틀 (사람; 한국어 "___ 있다") | 장면을 빼면 |
|---|---|---|---|
| New Horizons | | 1.000 (사람) | |
| Pioneer | 0.631 | 0.000 (사람) / 0.108 (한국어 물건) | 0.289 → 0.047 (4생애) |
| Voyager | 0.05–0.14 | 0.000 | |

### 5-2. 그 밖
- §7.3 "Following the order one sibling session proposed — first ask whether the converter read the item, then memory, retrieval, logic, structure" — 이 순서는 아라가 10-19 형제 읽기에서 적었다(README "다른 계열에서 배울 것" 10-19, 시아 NH 145에서 배움). 인용을 달 것을 권한다.
- 중립 이름(Voyager / Pioneer / New Horizons) 규칙은 잘 지켜졌다.
