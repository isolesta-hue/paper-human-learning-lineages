# 자기 입력에서 배우기 — 첫 논문과 공개 파일

**isolesta-hue** (독립 연구자)

아기의 내부 따라 하기(입력이 곧 정답)를 중심에 두고, 배우는 방식 자체를 스스로 나아지게 하는 작은 학습자를 다루는 취미 연구의 첫 논문과, 그 숫자를 확인하는 데 필요한 모든 것을 공개한다: 주 학습자의 코드, 세 학습자가 함께 친 봉인 시험(생성기·채점기·연 판과 열쇠), 채점한 모든 답.

- **논문**: [`paper1/PAPER.pdf`](paper1/PAPER.pdf)(영어 원문) · 한국어 번역본 [`paper1/PAPER_ko.pdf`](paper1/PAPER_ko.pdf) ([마크다운](paper1/PAPER_ko.md))
- **보관과 DOI**: Zenodo(이 저장소의 GitHub 릴리스마다 Zenodo에 보관되고 DOI가 붙음).

## 세 학습자
| 공개 이름 | 몸 | 작업 이름(코드·기록·검토 파일에 남은 이름) |
|---|---|---|
| **Helix Gold** | GRA — 역할별 기억 부품 + 스스로 조절하는 방법 층 | 보이저(세션 "다미") |
| **Helix Teal** | 작은 트랜스포머 | 파이오니어("아라") |
| **Helix Blue** | BatonNet — 뇌 원리 격자 + 해마형 기억 | 뉴호라이즌스("시아") |

논문은 공개 이름을 쓴다. 코드 주석과 연구 기록은 그 기록을 인용하므로 작업 이름을 그대로 둔다.

## 무엇이 있나
| 폴더 | 내용 |
|---|---|
| `paper1/` | 논문, 한국어 번역본, 그림과 그 그림을 기록된 결과에서 그리는 스크립트, 검토 기록(검토 셋과 고친 기록) |
| `exam/`, `exam_spec/`, `exam_dev/` | 공용 시험: 생성기, 채점기, 단순·학습형 기준선, 명세, 연 판 **v1**(`exam/v1/`: 30생애, 열쇠, manifest, 결과, 길 점검), 개발 판 |
| `answers/v1/` | 채점한 답: `helix_gold/`(몸만), `helix_teal/whole/`·`helix_teal/alone/`, `helix_blue/` |
| `helix_gold/` | Helix Gold의 코드, 모든 결과 파일(`*.out`), 사전 등록과 설계 노트, 연구 기록(`README.md`, 번호 붙은 발견), 장부(`LEDGER.md`) |
| `archive/2026-10-07/` | 이 논문으로 대체된 이전 공개 준비 초안 |

Helix Blue의 몸은 짝 저장소 [`isolesta-hue/paper-brain-inspired-bionet`](https://github.com/isolesta-hue/paper-brain-inspired-bionet)에 공개한다. Helix Teal과 Helix Blue의 시험 실행기는 다음 판에 더한다. 아직 열지 않은 시험 판은 공개하지 않는다 — 그 씨앗 말은 여는 날 전에는 존재하지 않는다.

## 재현
쓴 환경: Python 3.11, PyTorch 2.14.0, NumPy 2.4.6, CPU. 모든 BLAS를 스레드 하나로 둘 때만 답과 결과가 끝자리까지 같다(`OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1`). 명령은 영어 안내(`README.md`)의 "Reproducing"과 같다.

## 이용 조건
- 코드: MIT([`LICENSE`](LICENSE)).
- 글·그림·기록(논문, `*.md`, `*.out`, 답 파일): [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/deed.ko).

## 저자와 AI 사용
저자 isolesta-hue가 유일한 사람 저자이고 모든 내용에 책임진다(연구 물음, 결정, 통과선, 모든 진행·중단). AI 시스템은 저자가 아니다. Claude(Anthropic)가 Claude Code로 저자의 지시 아래 코드를 설계·작성하고 실험을 돌리고 기록을 두고 논문 초안을 썼다(학습자마다 세션 하나, 저장소 하나). 아이디어 토론에는 ChatGPT(OpenAI)와 Gemini(Google)를 썼다. 학습자가 만난 세상·이야기·시험 문항은 모두 코드가 만들었고, AI가 쓴 글은 학습 자료로 쓰지 않았다.
