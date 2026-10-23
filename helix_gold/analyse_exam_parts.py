"""Reads diag_exam_parts.py's probe files AFTER grading (Q17, 2026-10-20). Written and fixed before any key of edition v1
is read (the keys stay sealed until the three models' answers are in) -- the readings below do not change after.
Judgement = the grader's own (exam.grade.right on the same judged session, exam.checks.group, exam.checks.pass_line).
Readings (D67 diagnosis -- the body is judged as a whole, the parts are diagnosis):
  R0 checks: per life, the probe life's answer rows equal the recorded ones (all, exact); every judged row's "real"
     variant equals the exam's answer; the split error and the recomposed sum's deviation < 1e-3; the body unchanged while
     answering and around the newborn pass. Any failure -> no table.
  R1 per group (criterion, kind, split, order), items pooled over the probed lives: n, the exam's answer ("real"), the
     recomposed sum ("full" -- items whose judgement differs from "real" are counted), the newborn body ("exp 0", same
     rule) and chance (1 / candidates; 1/2 for pair).
  R2 per group and part (habit = sim + assoc, sim, assoc, pattern, chunk, side): accuracy WITHOUT the part. "기댐" = drop
     >= 0.02 and the items whose judgement changes go one way (two-sided sign test p < 0.005); "해침" = rise >= 0.02 with
     p < 0.005; else "-".
  R3 per group and part: accuracy with the part ALONE ("혼자" = within 0.02 of the exam's answer).
  R4 the kind gate: accuracy without it (labels as R2, the gate as the part).
  R5 per judged group: the parts it leans on (R2), the parts that alone reach it (R3), and "구조만" when exp 0 >= 0.80,
     "배움 필요" when real - exp 0 >= 0.20, else "-".
Caveats (declared before): answer-time removal only -- the life learned with every part (Sia's NH 144 removed parts for
whole lives, which is a different measure); a handful of lives, items pooled (the sign test counts items, not lives).
--newborn-test: a mechanics test on a development edition (its keys are open) -- the newborn rows stand in for the
judged ones; never used on v1.
    python3 analyse_exam_parts.py EDITION_DIR PROBE_DIR [--shared SHARED_REPO] [--out FILE.md] [--newborn-test]"""
import glob
import json
import math
import os
import sys
from collections import defaultdict

PARTS = ("habit", "sim", "assoc", "pattern", "chunk", "side")


def sign_p(a, b):
    """two-sided exact binomial (1/2) p for a vs b."""
    n, k = a + b, min(a, b)
    if n == 0:
        return 1.0
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)


def label(d, a, b, up="해침", down="기댐"):
    p = sign_p(a, b)
    if d <= -0.02 and p < 0.005:
        return down
    if d >= 0.02 and p < 0.005:
        return up
    return "-"


def main(edition, probe, shared, out_md, newborn_test=False):
    sys.path.insert(0, shared)
    from exam.checks import group, pass_line
    from exam.grade import judged_session, load, right
    lives = sorted(os.path.basename(f)[:-5] for f in glob.glob(os.path.join(probe, "life_*.json")))
    bad = []
    for life in lives:
        s = json.load(open(os.path.join(probe, f"{life}.json")))
        ok = (s.get("compared") is False or (s["same_rows"] and s["exact"] == s["rows"])) and \
            s["real_equal"] == s["judged_rows"] and s["largest_split_err"] < 1e-3 and s["largest_full_dev"] < 1e-3 and \
            s["unchanged_while_answering"] and s["pre_unchanged"]
        print(f"R0 {life}: rows {s['rows']} exact {s['exact']} real=exam {s['real_equal']}/{s['judged_rows']} split err "
              f"{s['largest_split_err']:.1e} sum dev {s['largest_full_dev']:.1e} unchanged {s['unchanged_while_answering']}/"
              f"{s['pre_unchanged']} -> {'ok' if ok else 'FAILED'}")
        if not ok:
            bad.append(life)
    if bad:
        sys.exit(f"R0 failed for {bad}: no table")
    cnt = defaultdict(lambda: defaultdict(int))          # group -> counter
    flips = defaultdict(lambda: defaultdict(lambda: [0, 0]))   # group -> variant -> [real right & variant wrong, reverse]
    spoke = defaultdict(set)                             # group -> the parts that ever spoke (nonzero) on its rows
    for life in lives:
        items = {it["id"]: it for it in load(os.path.join(edition, "lives", life, "items.jsonl"))}
        rows = {(r["id"], r["after_session"]): r for r in load(os.path.join(probe, f"{life}.parts.jsonl"))}
        for k in load(os.path.join(edition, "keys", f"{life}.jsonl")):
            it, g, js = items[k["id"]], group(k), judged_session(k["when"])
            pr, p0 = rows.get((k["id"], js)), rows.get((k["id"], -1))
            if newborn_test:                       # mechanics test on a development life: the newborn rows stand in
                pr = p0
            c = cnt[g]
            c["n"] += 1
            c["chance"] += 0.5 if it["format"] == "pair" else 1.0 / max(1, len(it.get("options") or it["candidates"]))
            if pr is None:
                c["missing"] += 1
                continue
            judge = {v: right(it, k, {"scores": sc}) for v, sc in pr["variants"].items()}
            spoke[g].update(pr.get("spoke", []))
            c["real"] += judge["real"]
            c["full≠real"] += judge["full"] != judge["real"]
            c["exp0"] += right(it, k, {"scores": p0["variants"]["real"]}) if p0 else 0
            for v, ok in judge.items():
                c["v:" + v] += ok
                f = flips[g][v]
                f[0] += judge["real"] and not ok
                f[1] += ok and not judge["real"]
    lines = ["| 묶음 | 판정 | n | 시험 답 | 경험 0 | 우연 | " + " | ".join(f"−{p}" for p in PARTS) + " | 종류 문 없이 | "
             + " | ".join(f"{p}만" for p in PARTS) + " | R5 |",
             "|" + "---|" * (7 + 2 * len(PARTS) + 1)]
    for g in sorted(cnt, key=lambda x: (x[0], x[1], x[2], x[3])):
        c, n = cnt[g], cnt[g]["n"]
        real = c["real"] / n
        cells, lean, alone = [], [], []
        sp = spoke[g] | ({"habit"} if spoke[g] & {"sim", "assoc"} else set())
        for p in PARTS:
            if f"v:-{p}" not in c or p not in sp:   # not in the body, or never spoke on these rows
                cells.append("·")
                continue
            acc = c[f"v:-{p}"] / n
            a, b = flips[g][f"-{p}"]
            lab = label(acc - real, b, a)
            if lab == "기댐":
                lean.append(p)
            cells.append(f"{acc:.3f}{'' if lab == '-' else ' ' + lab}")
        a, b = flips[g]["nogate"]
        gacc = c["v:nogate"] / n
        glab = label(gacc - real, b, a)
        solo = []
        for p in PARTS:
            if f"v:{p}" not in c or p not in sp:
                solo.append("·")
                continue
            acc = c[f"v:{p}"] / n
            if acc >= real - 0.02:
                alone.append(p)
            solo.append(f"{acc:.3f}")
        e0 = c["exp0"] / n
        r5 = ("구조만" if e0 >= 0.80 else "배움 필요" if real - e0 >= 0.20 else "-") + \
             (f"; 기댐 {'+'.join(lean)}" if lean else "") + (f"; 혼자 {'+'.join(alone)}" if alone else "")
        name = f"C{g[0]} {g[1]}" + (f" {g[2]}" if g[2] else "") + (" (순서)" if g[3] else "")
        judged = "판정" if pass_line(g) is not None else "보고"
        lines.append(f"| {name} | {judged} | {n} | {real:.3f}" + (f" (full≠ {c['full≠real']})" if c["full≠real"] else "")
                     + f" | {e0:.3f} | {c['chance'] / n:.3f} | " + " | ".join(cells)
                     + f" | {gacc:.3f}{'' if glab == '-' else ' ' + glab} | " + " | ".join(solo) + f" | {r5} |")
    text = (f"부품별 탐침 — 생애 {len(lives)}개({', '.join(lives)}), 채점 회차의 답, 답할 때 빼기만(Q17)\n\n" + "\n".join(lines) + "\n")
    print(text)
    if out_md:
        open(out_md, "w").write(text)
    print("exit 0")


if __name__ == "__main__":
    a = sys.argv[1:]
    shared = a[a.index("--shared") + 1] if "--shared" in a else "/home/user/shared-curriculum"
    out_md = a[a.index("--out") + 1] if "--out" in a else None
    main(a[0], a[1], shared, out_md, newborn_test="--newborn-test" in a)
