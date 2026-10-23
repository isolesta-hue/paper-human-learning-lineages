"""Path check (2026-10-21; the user's decision in the Voyager session, after Pioneer's v1 reading P55: "format checks with a
perfect fake body look only at the form, not at whether the body can produce that kind of answer at all"). A readiness check
for the next edition, run on a development edition (open keys) with a body's answer files -- or, as a diagnosis only, on a
spent edition. It does not change any grade. Per judged group (exam.checks.group, pass_line not None), items pooled over
the lives, at each item's judged session (exam.grade.judged_session), it asks whether the body's answer DEPENDS ON THE ITEM:
  n, accuracy (exam.grade.right), chance (1 / candidates; 1/2 for pair);
  no answer  -- the share of items with no row, empty scores or a tie for the top (the grader counts them wrong);
  same top   -- per life, the share of the group's items whose top choice is the life's single most frequent top choice
                (pair: the label judged impossible), averaged over lives by item count -- 1.0 = one answer whatever is asked;
  answer rank -- the median normalised rank of the right candidate (0 = top, 1 = bottom; choice / blank / slot only).
Verdict, fixed before this file was first run:
  "막힘: 답 없음"            no answer >= 0.50
  "막힘: 문항과 무관한 답"    same top >= 0.90 and accuracy < chance + 0.05
  "열림"                     otherwise (the path exists -- whether the body learns the rest is the exam's question)
A closed path means the reading rule (the converter) or the body's answer form cannot produce item-dependent answers of
that kind: fix the reading or the body's form first (in the body's own world), before reading the group as "not learned".
LIMIT (found when the first version was checked against a known case, 2026-10-21): from answer files alone this catches
only two kinds of closed path -- no answer, and one answer whatever is asked. It does NOT catch Pioneer's v1 "where"
(its reading asked for a cell token its body never produced as an answer: the scores still vary from item to item, so the
answers look like "not learned"). That kind needs a body-level test -- teach the item itself (question and answer, in the
body's own heard form) and ask it back: a body that cannot answer an item it was literally taught has no path for that
kind of answer under that reading. This file is the cheap first screen; the taught-item test is the path check proper.
    python3 -m exam.path_check --edition DIR --answers NAME=DIR [NAME=DIR ...] [--out FILE.md]"""
import argparse
import os
from collections import Counter, defaultdict

from .checks import group, pass_line
from .grade import judged_session, load, right


def key_of(ans):
    if isinstance(ans, list):
        return ",".join(str(int(v)) for v in ans)
    return str(ans)


def check_set(edition, adir):
    acc = defaultdict(lambda: defaultdict(float))
    tops = defaultdict(lambda: defaultdict(list))         # group -> life -> [top choice or None]
    ranks = defaultdict(list)
    for life in sorted(os.listdir(os.path.join(edition, "lives"))):
        items = {it["id"]: it for it in load(os.path.join(edition, "lives", life, "items.jsonl"))}
        path = os.path.join(adir, f"{life}.jsonl")
        rows = {(r["id"], int(r["after_session"])): r for r in load(path)} if os.path.exists(path) else {}
        for k in load(os.path.join(edition, "keys", f"{life}.jsonl")):
            g = group(k)
            if pass_line(g) is None:
                continue
            it, row = items[k["id"]], rows.get((k["id"], judged_session(k["when"])))
            c = acc[g]
            c["n"] += 1
            c["right"] += right(it, k, row)
            c["chance"] += 0.5 if it["format"] == "pair" else 1.0 / max(1, len(it.get("options") or it["candidates"]))
            sc = (row or {}).get("scores") or {}
            if not sc and row is not None and row.get("answer") is not None:     # an answer-form row (the baselines)
                a_ = row["answer"]
                tops[g][life].append(a_.get("impossible") if isinstance(a_, dict) else key_of(a_))
                continue
            if not sc:
                c["none"] += 1
                tops[g][life].append(None)
                continue
            vals = {a: float(v) for a, v in sc.items()}
            if it["format"] == "pair":
                lo = min(vals.values())
                low = [a for a, v in vals.items() if v == lo]
                top = low[0] if len(low) == 1 else None
            else:
                hi = max(vals.values())
                best = [a for a, v in vals.items() if v == hi]
                top = best[0] if len(best) == 1 else None
                ans = key_of(k["answer"])
                if ans in vals and len(vals) > 1:
                    better = sum(1 for v in vals.values() if v > vals[ans])
                    ranks[g].append(better / (len(vals) - 1))
            if top is None:
                c["none"] += 1
            tops[g][life].append(top)
    out = {}
    for g, c in acc.items():
        n = c["n"]
        same_num = same_den = 0
        for life, ts in tops[g].items():
            got = [t for t in ts if t is not None]
            if got:
                same_num += Counter(got).most_common(1)[0][1]
            same_den += len(ts)
        rk = sorted(ranks[g])
        out[g] = dict(n=int(n), acc=c["right"] / n, chance=c["chance"] / n, none=c["none"] / n,
                      same=same_num / same_den if same_den else float("nan"),
                      rank=rk[len(rk) // 2] if rk else float("nan"))
    return out


def verdict(d):
    if d["none"] >= 0.50:
        return "막힘: 답 없음"
    if d["same"] >= 0.90 and d["acc"] < d["chance"] + 0.05:
        return "막힘: 문항과 무관한 답"
    return "열림"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--edition", required=True)
    ap.add_argument("--answers", nargs="+", required=True)
    ap.add_argument("--out")
    a = ap.parse_args()
    lines = []
    for spec in a.answers:
        name, adir = spec.split("=", 1)
        res = check_set(a.edition, adir)
        lines += [f"\n### {name}\n", "| 묶음 | n | 정답률 | 우연 | 답 없음 | 같은 답 | 정답 순위(중앙) | 길 |", "|---|---|---|---|---|---|---|---|"]
        closed = 0
        for g in sorted(res, key=lambda x: (x[0], x[1], x[2], x[3])):
            d = res[g]
            v = verdict(d)
            closed += v != "열림"
            name_g = f"C{g[0]} {g[1]}" + (f" {g[2]}" if g[2] else "")
            rank = "–" if d["rank"] != d["rank"] else f"{d['rank']:.2f}"
            lines.append(f"| {name_g} | {d['n']} | {d['acc']:.3f} | {d['chance']:.3f} | {d['none']:.2f} | {d['same']:.2f} | "
                         f"{rank} | {v} |")
        lines.append(f"\n막힌 묶음 {closed} / {len(res)}")
    text = "\n".join(lines) + "\n"
    print(text)
    if a.out:
        with open(a.out, "w", encoding="utf-8") as f:
            f.write(text)


if __name__ == "__main__":
    main()
