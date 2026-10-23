"""Reads diag_joined.py's answers (finding 166 next (a); pre-registration DIAG_JOINED.md -- these readings are fixed before
any diagnosis life runs). Judgement = the grader's own (exam.grade.right at the item's judged session, exam.checks.group,
exam.checks.pass_line); the recorded conv-v1 answers of the same lives (the graded v1 answers) are the comparison.
  J0 checks, per life (any failure -> no table): every exposure line rebuilt from the joined units; the answer rows are the
     recorded rows exactly (no duplicate); score keys = candidates; the learner unchanged while answering.
  J1 per group (criterion, kind, split, order), items pooled over the lives: n, chance, conv-v1, conv-v1j, the difference,
     the items whose judgement changes (a = right with conv-v1 only, b = right with conv-v1j only) and the two-sided sign
     test p; label "오름" = difference >= +0.02 and p < 0.005, "내림" = <= -0.02 and p < 0.005, else "-".
  J2 the question (finding 166 (1)) -- the name items of criterion 0 (person + box + thing) pooled:
     "나눔이 주된 원인" when conv-v1j >= 0.50 and "오름"; "나눔이 일부 원인" when "오름" and < 0.50;
     "나눔은 원인 아님" otherwise (the failure is then the body's: memory -> retrieval -> logic -> structure).
  J3 the other groups whose fact spans moments in conv-v1's form (criteria 1, 2, 3, 4, 5, 6 contact / permanence, 7, 8,
     9): their labels = the split's share there (reported, no verdict).
  J4 the groups whose fact sits inside one moment in conv-v1's form (criterion 0 particles; criterion 6 support / solidity):
     a "내림" = the cost of joining (a joined unit is a bag: which picture is at which cell can mix) -- reported.
  J5 cost: CPU per life (the exam day's conv-v1 CPU for the same life and the 10-20 rerun's are on other hosts; host
     speed varies about 1.6x -- compared, not judged).
Caveats (declared before): three lives, items pooled (the sign test counts items, not lives); the keys are public, so the
variant was fixed before the run and nothing is tuned after; diagnosis only -- no v1 score changes.
    python3 analyse_joined.py EDITION_DIR RECORDED_ANSWERS_DIR JOINED_DIR [--shared SHARED_REPO] [--out FILE.md]"""
import glob
import json
import math
import os
import sys
from collections import defaultdict

SPANS = {0: False, 1: True, 2: True, 3: True, 4: True, 5: True, 7: True, 8: True, 9: True}   # criterion 6 by kind below


def sign_p(a, b):
    n, k = a + b, min(a, b)
    if n == 0:
        return 1.0
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)


def label(d, p):
    if d >= 0.02 and p < 0.005:
        return "오름"
    if d <= -0.02 and p < 0.005:
        return "내림"
    return "-"


def spans(g):
    c, k = g[0], g[1]
    if c == 0:
        return not k.startswith("particle")        # names span moments (sentence / scene line); particles do not
    if c == 6:
        return k.split()[0] in ("contact", "permanence")
    return SPANS[c]


def main(edition, recorded, joined, shared, out_md):
    sys.path.insert(0, shared)
    from exam.checks import group, pass_line
    from exam.grade import judged_session, load, right
    lives = sorted(os.path.basename(f)[:-5] for f in glob.glob(os.path.join(joined, "life_*.json")))
    bad = []
    for life in lives:
        s = json.load(open(os.path.join(joined, f"{life}.json")))
        ok = (s["rebuild_ok"] and s["same_rows_as_recorded"] and s["duplicated"] == 0 and s["keys_match"]
              and s["unchanged_while_answering"] and s["sessions"] == "all")
        print(f"J0 {life}: rebuild {s['rebuild_ok']} ({s['units']} units / {s['exposure_lines']} lines), rows {s['rows']} "
              f"same as recorded {s['same_rows_as_recorded']}, keys {s['keys_match']}, unchanged {s['unchanged_while_answering']}, "
              f"cpu {s['cpu_s'] / 60:.1f} min -> {'ok' if ok else 'FAILED'}")
        if not ok:
            bad.append(life)
    if bad:
        sys.exit(f"J0 failed for {bad}: no table")
    cnt = defaultdict(lambda: defaultdict(float))
    for life in lives:
        items = {it["id"]: it for it in load(os.path.join(edition, "lives", life, "items.jsonl"))}
        rec = {(r["id"], r["after_session"]): r for r in load(os.path.join(recorded, f"{life}.jsonl"))}
        new = {(r["id"], r["after_session"]): r for r in load(os.path.join(joined, f"{life}.jsonl"))}
        for k in load(os.path.join(edition, "keys", f"{life}.jsonl")):
            it, g, js = items[k["id"]], group(k), judged_session(k["when"])
            a, b = right(it, k, rec.get((k["id"], js))), right(it, k, new.get((k["id"], js)))
            c = cnt[g]
            c["n"] += 1
            c["chance"] += 0.5 if it["format"] == "pair" else 1.0 / max(1, len(it.get("options") or it["candidates"]))
            c["v1"] += a
            c["v1j"] += b
            c["a"] += a and not b
            c["b"] += b and not a
    lines =["| 묶음 | 판정 | 순간 사이? | n | 우연 | conv-v1(시험 답) | conv-v1j | 차이 | 바뀐 문항 (v1만 / v1j만) | p | 표시 |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    names = defaultdict(float)
    for g in sorted(cnt, key=lambda x: (x[0], x[1], x[2], x[3])):
        c = cnt[g]
        n = c["n"]
        d = (c["v1j"] - c["v1"]) / n
        p = sign_p(int(c["a"]), int(c["b"]))
        lab = label(d, p)
        if g[0] == 0 and g[1].startswith("name"):
            for key in ("n", "chance", "v1", "v1j", "a", "b"):
                names[key] += c[key]
        name = f"C{g[0]} {g[1]}" + (f" {g[2]}" if g[2] else "") + (" (순서)" if g[3] else "")
        lines.append(f"| {name} | {'판정' if pass_line(g) is not None else '보고'} | {'예' if spans(g) else '아니오'} | {int(n)} | "
                     f"{c['chance'] / n:.3f} | {c['v1'] / n:.3f} | {c['v1j'] / n:.3f} | {d:+.3f} | {int(c['a'])} / {int(c['b'])} | "
                     f"{p:.4f} | {lab} |")
    n = names["n"]
    d = (names["v1j"] - names["v1"]) / n
    p = sign_p(int(names["a"]), int(names["b"]))
    lab = label(d, p)
    v1j = names["v1j"] / n
    j2 = "나눔이 주된 원인" if (v1j >= 0.50 and lab == "오름") else "나눔이 일부 원인" if lab == "오름" else "나눔은 원인 아님"
    text = (f"conv-v1j 진단 생애 — 생애 {len(lives)}개({', '.join(lives)}), 채점 회차의 답, 시험 답(conv-v1)과 같은 생애·같은 몸 시드\n\n"
            + "\n".join(lines)
            + f"\n\nJ2 이름 문항(사람 + 상자 + 물건) 모음: n {int(n)}, 우연 {names['chance'] / n:.3f}, conv-v1 {names['v1'] / n:.3f}, "
              f"conv-v1j {v1j:.3f}, 차이 {d:+.3f}, 바뀐 문항 {int(names['a'])} / {int(names['b'])}, p {p:.2g} -> {lab} -> **{j2}**\n")
    print(text)
    if out_md:
        open(out_md, "w").write(text)
    print("exit 0")


if __name__ == "__main__":
    a = sys.argv[1:]
    shared = a[a.index("--shared") + 1] if "--shared" in a else "/home/user/shared-curriculum"
    out_md = a[a.index("--out") + 1] if "--out" in a else None
    main(a[0], a[1], a[2], shared, out_md)
