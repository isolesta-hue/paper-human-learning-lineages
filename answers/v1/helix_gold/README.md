# 다미의 첫 봉인 판 v1 답

- 판: 공용 `exam/v1/`(커밋 05c07fc, manifest sha256 앞 12자리 `4ac9e0977857`). 30생애, 답 264,336줄(빠짐 0, 같은 문항·회차 겹침 0).
- 실행: `run_exam_day.py`(시험 코드 지문 `0551dd3a140d`, 변환기 `conv-v1`), 몸 시드 = 생애 번호. 10-19 03:01:49 ~ 23:07:49 UTC.
- 파일: `life_XXX.jsonl.gz`(생애별 답, `gzip -n`), `SHA256SUMS`(풀었을 때의 원본 sha256), `runs/life_XXX.json`(생애 기록: 줄 수, 답하는 동안 몸 그대로, CPU·벽시계·메모리), `piece.log`(조각 기록).
- 채점기는 풀어 놓은 jsonl을 읽는다: `gunzip -k life_*.jsonl.gz` 뒤 `sha256sum -c SHA256SUMS`.

## 재현 조건 (2026-10-20, 시아의 형제 알림 10-19 "수치 라이브러리의 스레드 수가 답을 바꿀 수 있음"을 받아 확인)

- 만든 조건: torch 스레드 1(생애 프로세스마다, `run_exam.py`의 `torch.set_num_threads(1)`), numpy의 OpenBLAS는 기본(환경 변수 `OMP_NUM_THREADS`·`OPENBLAS_NUM_THREADS`·`MKL_NUM_THREADS` 없음), CPU 4(x86_64, Intel Xeon 2.10GHz), 한 번에 생애 2~4개.
- 확인(`check_exam_threads.py`, 결과 `check_exam_threads.out`): life_000의 앞 2회차(답 1,097줄)를 (가) 같은 조건, (나) OpenBLAS·OpenMP·MKL 스레드 1(파이썬을 띄우기 전에 정함)로 다시 돌림 — **둘 다 기록된 답과 점수가 끝자리까지 같음**(1,097/1,097, 가장 큰 차이 0). 그때의 판: Python 3.11.15, numpy 2.4.6(OpenBLAS 0.3.31), torch 2.14.0+cu130.
- 범위: 한 생애의 16회차 중 앞 2회차만 잼. 다미의 큰 행렬 계산은 거의 모두 torch(스레드 1로 고정)이고 numpy 행렬 곱은 세상 생성 쪽 몇 곳뿐이라 스레드 수에 기대지 않는 것으로 보이지만, 16회차 전체를 스레드 1로 다시 돌려 보지는 않았다.
