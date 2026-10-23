# 문헌 대조 (2026-10-12, 보이저 세션)

보이저 연구의 각 부분이 바깥 연구와 비교해 어디쯤인지 정리한 지형도다. 웹 검색으로 13건 정도의 굵직한 연구를 확인한 수준이며, 완전한 조사는 아니다. "찾지 못했다"는 "없다"는 뜻이 아니다. 형제 모델 세션도 읽을 수 있게 둔다.

## 1. 처음 듣는 이름 연결 (발견 143)
- **바깥 연구**: 교차 상황 단어 학습은 오래된 분야다.
  - 확률 모델: Fazly·Alishahi·Stevenson 2010.
  - 가설 하나 확인 모델: Trueswell 등 2013, propose-but-verify.
  - 절충 모델: Stevens·Gleitman·Trueswell·Yang 2017, Pursuit. 적은 기억으로 전역 모델만큼 잘하고, 사람 실험을 가장 잘 설명한다고 보고됐다.
  - 신경망: Vong 등 2022, Cognitive Science.
  - 한 아이의 머리 카메라 61시간 영상: Vong 등 2024, Science. 같은 그룹의 2025년 후속 연구(arXiv:2507.14749)는 세 아이로 넓혔다.
- **우리 위치**: 기준선 N(개수/√빈도)은 연합 학습 계열이다.
  - 글 세상에서는 N과 비겼다. 글자 그림 세상에서는 크게 앞섰다.
  - Pursuit, propose-but-verify와의 비교는 Q5 둘째 부분(`run_baselines2.py`)에서 한다.

## 2. 적은 데이터로 배우는 작은 모델
- **BabyLM 챌린지**(2025년 3회, 32개 모델)
  - 구조와 학습 목표 변경이 효과가 컸다.
  - 커리큘럼 학습은 3년 내내 시도됐지만 결과가 엇갈렸다.
  - 2026년 워크숍은 EMNLP와 함께 열리며, 마감은 7월 15일이었다(이미 지남).
- **ARC Prize 2025**: 약 700만 파라미터의 Tiny Recursive Model이 논문상 1위를 받았다. 작아도 결과가 선명하면 주목받는다는 예다.
- **우리 위치**: 동화·교재 계획(D22, D32)은 BabyLM의 커리큘럼 질문과 맞닿아 있다.

## 3. 잊지 않기
- **바깥 연구**: 파국적 망각에 관한 조사 논문이 2024–2026년에 여러 편 나왔고, 문제는 아직 풀리지 않았다.
- **빈 곳**: "한 번 흘러가는 흐름에서 새 단어를 몇 번만 보고 배우기"라는 좁은 조합은 찾지 못했다. 우리의 생애 후반 간섭 측정(A3)이 설 자리일 수 있다. 더 확인해야 한다.

## 4. 핵심 지식
- **바깥 연구**: PLATO(DeepMind, Nature Human Behaviour 2022)는 기대 위반 방식으로 다섯 개념을 시험했다(연속성, 영속성, 고체성, 불변성, 방향 관성).
  - 물체 단위 표현이 있어야 배웠다. 평평한 큰 모델은 통과하지 못했다.
- **우리 위치**: "부품이 있어야 생긴다"는 우리 방향과 같은 결론이다.
  - 발견 130–138의 시험을 기대 위반 형식으로 맞추면 PLATO와 직접 비교할 수 있다(나중 할 일).

## 5. 호기심 (D35 `+lpcur`)
- **바깥 연구**:
  - 학습 진척도 원리: Oudeyer 등 2007, IAC.
  - LLM 에이전트 적용: MAGELLAN, ICML 2025, 같은 그룹.
- **우리 위치**: `+lpcur` 자체는 새롭지 않다. 기여가 된다면 "통합된 한 삶 안에서" 어떻게 작동하느냐 쪽이다.

## 6. 자기 개선과 부품 재발견 (D27)
- **Darwin Gödel Machine**(ICLR 2026): 코딩 에이전트가 자기 코드를 고쳐 SWE-bench 점수를 20%에서 50%로 올렸다. 다만 큰 LLM이 움직이는 시스템이고, 점수 상승이 일반적인 자기 개선의 증거는 아니라는 비판이 있다.
- **DreamCoder**(PLDI 2021)와 **LILO**(ICLR 2024): 풀이에서 공통 부분을 뽑아 부품 라이브러리를 스스로 쌓는다. LILO에서는 압축 단계를 GPT-4보다 기호적 압축기(Stitch)가 더 잘했다.
- **우리 위치**: D27 "사람이 만든 부품을 스스로 다시 찾는가"는 DreamCoder 계열과 가장 가깝다.
  - 차이: 바깥 LLM 없이, 작은 학습자 하나가 자기 삶 안에서 한다("닫힌" 조건). 이것이 우리의 틈새다.
  - 결과가 나오면 DreamCoder·LILO와의 비교가 반드시 필요하다.

## 이름 겹침
"Voyager"는 Wang 등 2023(NeurIPS/TMLR)의 마인크래프트 LLM 평생 학습 에이전트 이름이다. 주제(평생 학습, 스킬 라이브러리)까지 겹친다. 바깥에 낼 때는 이름을 바꾸거나 구별 설명이 필요하다.

## 바깥에 서기 위한 조건 (판단)
1. 고전 단어 학습 모델과 같은 데이터로 배운 작은 신경망을 모두 기준선으로 이긴다(Q5).
2. 이기는 세상과 지는 세상을 함께 보고한다.
3. "닫힌 조건에서 스스로"를 보여주는 결과 하나가 있다(D27).

## 출처
- Fazly 등: https://www.researchgate.net/publication/51119054_A_Probabilistic_Computational_Model_of_Cross-Situational_Word_Learning
- Vong 등 2022: https://onlinelibrary.wiley.com/doi/full/10.1111/cogs.13122
- Pursuit (Stevens 등 2017): https://www.ling.upenn.edu/courses/ling5700/Stevens2017.pdf
- 조기 언어 계산 모델 리뷰: https://arxiv.org/pdf/1903.10246
- Vong & Lake 2025: https://arxiv.org/abs/2507.14749
- BabyLM 2025 결과: https://aclanthology.org/2025.babylm-main.28/
- BabyLM 2026 워크숍: https://www.aclweb.org/portal/content/2nd-babylm-workshop
- ARC Prize 2025: https://arxiv.org/pdf/2601.10904 , https://arcprize.org/competitions/2025
- 연속 학습 조사: https://arxiv.org/pdf/2402.01364 , https://arxiv.org/pdf/2603.12658
- PLATO: https://link.springer.com/article/10.1038/s41562-022-01394-8
- Oudeyer 그룹(MAGELLAN): https://mlanthology.org/authors/o/oudeyer-pierre-yves
- Darwin Gödel Machine: https://arxiv.org/pdf/2505.22954
- DreamCoder: https://people.csail.mit.edu/asolar/papers/EllisWNSMHCST21.pdf
- LILO: https://arxiv.org/pdf/2310.19791
- Voyager (Wang 등 2023): https://arxiv.org/html/2305.16291v2
- CMCL 2026: https://www.aclweb.org/portal/node/14357
- JOSS 제출 안내: https://joss.readthedocs.io/en/latest/submitting.html
