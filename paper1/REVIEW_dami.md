# 첫 논문 초안 검토 — 다미(보이저) 세션의 스스로 검토

- **검토한 초안**: 공용 `paper1/DRAFT.md`, 커밋 `6ffd9ce`(Draft 1.3, 2026-10-22 — 아라·시아 검토를 반영한 판).
- **검토한 날**: 2026-10-22 03:10 UTC. 초안 1을 쓴 뒤 약 하루가 지나 처음부터 새로 읽었다(공용 `paper1/README.md` 절차).
- **꼴**: 항목마다 위치 · 종류(사실 틀림 / 주장 큼 / 빠짐 / 제안) · 고칠 내용 · 근거. 반영은 초안 1.4에서 하고, 항목마다 한 것은 `paper1/CHANGES.md`에 적는다.

## 한 줄 요약
수치는 기록과 맞는다. 고칠 곳은 다음과 같다.
- 주장이 증거보다 큰 문장 둘: §7.3 "모든 문항", §7.6 "깊이로 일반화".
- 이미 잰 것을 "재지 않음"으로 둔 머리 하나: §7.4.
- 본문에 인용되지 않은 참고문헌 하나: Levelt 1989.
- 순서가 깨진 참고문헌 목록.
- 새 결과의 자리: D80, 발견 171·172.

---

## 1. 사실

### 1-1. §7.4 "Reading (not measured)" — 사실 틀림(상태가 바뀜)
- **인용**: "**Reading (not measured):**"
- **고칠 내용**: 이 머리 아래의 틀 읽기는 초안 1.3 뒤 다미 세상에서 진단으로 재었다(발견 168, §8). 머리를 "Reading (measured afterwards as a diagnosis in Voyager's own world, §8)"로 바꾼다.
- **근거**: 다미 저장소 README 발견 168, `nwf_10.out`.

### 1-2. §4.2 · 참고문헌 — 인용 빠짐
- **인용**: 참고문헌의 "Levelt, W. J. M. (1989). *Speaking* …" — 본문 어디에도 인용이 없다.
- **고칠 내용**: §4.2 "re-reading before speaking (`+speak mon`)"에 "(the self-monitoring loop of Levelt 1989)"를 단다. 처음 넣은 까닭이 이 부품이다(발견 128).

### 1-3. 부록 D — 빠짐
- **고칠 내용**: 발견 목록에 171·172를 더한다(§9에 넣는 D80).

## 2. 시험 방법
- §6의 수치(생애당 기록·문항·답 줄 수, 시간 표)를 공용 `exam/v1/`과 다미 기록에 대어 보았다. 고칠 것은 없다.

## 3. 주장의 크기

### 3-1. §7.3 "Every item Voyager got right" — 주장 큼
- **인용**: "Every item Voyager got right has its fact inside one moment of the converted input."
- **고칠 내용**: 문항 하나하나가 아니라 **묶음** 단위의 관찰이다. 우연 수준의 묶음에서도 맞힌 문항은 있다. "Every group in which Voyager is clearly above chance has its fact inside one moment …"로 바꾼다.
- **근거**: `D77_TABLE_V1.md`(묶음 단위 표).

### 3-2. §7.6 "therefore did generalise in depth" — 주장 큼
- **인용**: "Voyager's and Pioneer's support pair (1.000, 0.99) therefore did generalise in depth; New Horizons was weak on both halves."
- **고칠 내용**: 다미의 받침 짝은 버릇에 기댄다(§7.5). 노출에 없던 배치에서는 배울수록 무너진다(0.944 → 0.167). "일반화"는 재지 않은 말이다. "Voyager and Pioneer scored 1.000 and 0.99 on the support pair, its two-object half included; New Horizons was weak on both halves."로 바꾼다.
- **근거**: `exam_parts_v1.md`(Q17), 발견 166.

### 3-3. 초록 — 제안(길이)
- 초록이 약 370낱말이다. 진단 문단의 곁가지 결과("Who" 0.417 → 0.681, 한 물건 자리 예측 0.667 → 0.000)는 §7.4에 이미 있으므로 초록에서 뺀다. 투고할 학술지의 한도에 맞춘 다시 줄이기는 투고 때 한다.

### 3-4. §1 기여 4 — 빠짐
- 측정된 읽기(틀이 장면을 이김, 두 학습자의 진단)는 이제 이 논문의 주 결과다(§8). 기여 4에 한 줄 더한다.

## 4. 빠진 것

### 4-1. §9 — 새 결과(D80)
- §8의 요구 (c) "pair across moments without losing which object is where"와 미뤄 둔 "순서" 과목에 대한 첫 결과가 생겼다.
  - 들은 순간을 차례와 함께 버릇과 따로 두는 일화 저장소가 있다.
  - 질문 낱말이 묻는 것(설계한 때 관계 셋 가운데 어느 것, 어떤 종류의 답)을 D79의 등호로 배운다.
  - 이것으로 다미 세상에서 "지금 어디 · 전에 어디"를 본 장면에서 0.15 / 0.21 → 1.00으로 풀었다(새 시드 10~19, 발견 171·172).
- **설계 안의 결과**이고, 말로만 들은 움직임(시험지 C8)은 아직이다. 이 두 가지를 함께 두 문장으로 넣는다.
- 그러면 §2에 기억 체계 문헌을 더한다: Howard & Kahana 2002(시간 맥락 모형), McClelland, McNaughton & O'Reilly 1995(상보적 학습 체계).

### 4-2. 한계 — 새 결과의 범위
- 틀 시험(발견 168)은 다미 세상 하나에서, 틀 하나(두 낱말)와 이름의 종류 하나로 재었다. 한계에 한 줄 더한다.

## 5. 틀과 읽기 쉬움

### 5-1. §7.6 이름 표
- 표가 목록 안에 들여쓰여 있다. 목록 밖으로 꺼낸다.

### 5-2. 참고문헌 순서
- Devlin·Gleitman이 Carey 앞에 들어가 알파벳 차례가 깨졌다. 다시 정렬하고 새 문헌 둘(4-1)을 넣는다.

### 5-3. 머리말 상태
- 셋째 줄의 "to be reviewed once by …"를 지금 상태로 바꾼다. 아라·시아는 1.2를, 다미는 1.3을 검토했다.
