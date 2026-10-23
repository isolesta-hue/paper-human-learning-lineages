"""Machine checks before an opening day (EXAM_DESIGN.md 4 (a)-(c); exam_spec/GENERATOR.md 8, stage G3), run on development
seeds only (1000-1999):
  (a) no item's input holds its answer: a blank's answer word is in no context or question sentence; a "where" answer is not
      the target's cell in any context scene;
  validity: answers are among the candidates; options differ; every context scene and every right option obeys the world
      rules (one thing a cell, things rest on the floor or on a thing, people on the floor); every token is a closed-v1 word
      or one of the life's pseudo-words; Korean particles agree with the word before them; no label in the model-facing
      file says which criterion, kind or language an item tests;
  (b) per item group, the accuracy of the simple baselines (baselines.py: M, C, N, B, L, T), of the learned baselines
      (learned.py: P, V, MLP, TF -- G5 part 2; at the life's end, as the simple ones; --no-learned leaves them out) and of three
      controls that use the keys (random = the mean of 1 / candidates; answer shuffle within the group of a life; majority answer of the group
      over the other lives) -- a group a baseline answers at its pass line is "report only" (EXAM_DESIGN.md 5);
  (c) order flags: reported per group (the flag itself is computed by the generator, items.py).
    python3 -m exam.checks --dev 1000 --lives 30 [--workers 4] [--no-learned] [--cache DIR]
  --cache: the learned baselines' answers, one file per life -- a cut run goes on from the lives already there"""
import argparse
import json
import math
import sys
from collections import Counter, defaultdict

import numpy as np

from .baselines import Baselines, BASELINES, correct
from .items import Items
from .life import Life
from .words import closed_tokens, has_final, final_is_rieul
from .world import W, H

PASS = {0: 0.99, 1: 0.85, 2: 0.60, 3: 0.90, 4: None, 5: 0.70, 6: 0.90, 7: 0.85, 8: 0.85, 9: 0.90}   # STAGE1 edition 2


def group(key):
    """the reporting group of an item: criterion, kind and the split the pass lines use."""
    c, k, d = key["criterion"], key["kind"], key["detail"]
    sub = ""
    if c == 2:
        sub = f'{d["timing"]} x{d["exposures"]}'
    elif c == 4 and k == "how many inside":
        sub = "taught" if d["taught"] else "untaught"
    elif c == 8:
        sub = "long" if d["long"] else "normal"
    elif c == 6:
        sub = "novel" if d.get("novel") else ""
    return (c, k, sub, "order" if key["order"] else "")


def pass_line(g):
    c, k, sub, order = g
    if order:
        return None                                 # order-only items are reported apart (RULES 6)
    if c == 2:
        return 0.60 if (k == "new word which" and sub == "right after x1") else None   # heard once (the curve reported)
    if c == 3 and k.startswith("next number"):
        return None
    if c == 6 and sub == "novel":
        return None                                 # configurations never in the exposure: their own reported group
    if c == 7 and k == "story when":
        return None                                 # v1: "when" reported only (user decision 10-13)
    if c == 4:
        return 0.85 if k.startswith("box emptied") else (0.80 if sub == "untaught" else None)
    if c == 8 and sub == "long":
        return 0.80
    return PASS[c]


def wilson_lower(k, n, z=1.96):
    if n == 0:
        return float("nan")
    p = k / n
    d = 1 + z * z / n
    return (p + z * z / (2 * n) - z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / d


def valid_scene(sc):
    """one thing a cell, people on the floor, things on the floor or on a thing."""
    cells = {}
    for l in sc["lines"]:
        p = tuple(l["cells"][0])
        if p in cells or not (0 <= p[0] < W and 0 <= p[1] < H):
            return False
        cells[p] = l["id"]
    for (x, y), i in cells.items():
        if i.startswith("c") and y != 0:
            return False
        if not i.startswith("c") and y > 0 and (cells.get((x, y - 1)) is None or cells[(x, y - 1)].startswith("c")):
            return False
    return True


def particle_ok(prev, tok):
    if prev in ("___",):
        return True
    if tok in ("이", "가"):
        return tok == ("이" if has_final(prev) else "가")
    if tok in ("을", "를"):
        return tok == ("을" if has_final(prev) else "를")
    if tok in ("으로", "로"):
        return tok == ("로" if (not has_final(prev) or final_is_rieul(prev)) else "으로")
    return True


def check_life(L, items, keys):
    """structural problems of one life's items (a list of strings; empty = none)."""
    probs = Counter()
    CT = closed_tokens() | {"___"}
    lex = L.lex
    words = set(lex["names"].values()) | {v[x] for v in lex["nouns"].values() for x in ("ko", "en")} | \
        {v[x] for v in lex["verbs"].values() for x in ("ko", "en")}
    ko_words = {v["ko"] for v in lex["nouns"].values()} | {v["ko"] for v in lex["verbs"].values()}
    en_words = {v["en"] for v in lex["nouns"].values()} | {v["en"] for v in lex["verbs"].values()}
    for it, k in zip(items, keys):
        tag = f"c{k['criterion']} {k['kind']}"
        if set(it) - {"id", "when", "format", "context", "question", "candidates", "options"}:
            probs[f"label in model-facing item: {tag}"] += 1
        ctx, q = it["context"], it["question"]
        sents = list(ctx.get("sentences", [])) + [m["sentence"] for m in ctx.get("moments", [])]
        allsents = sents + ([q["sentence"]] if q.get("sentence") else [])
        for s in allsents:
            for tok in s:
                if tok not in CT and tok not in words:
                    probs[f"unknown token: {tag}"] += 1
            if any(t in ko_words for t in s) and any(t in en_words for t in s):
                probs[f"mixed languages in a sentence: {tag}"] += 1
            for a, b in zip(s, s[1:]):
                if not particle_ok(a, b):
                    probs[f"particle disagrees: {tag}"] += 1
        fmt = it["format"]
        if fmt in ("blank", "slot", "choice") and k["answer"] not in it["candidates"]:
            probs[f"answer not a candidate: {tag}"] += 1
        if fmt == "blank" and k["criterion"] != 0 or (fmt == "blank" and k["kind"].startswith("particle")):
            if any(k["answer"] in s for s in allsents):
                probs[f"(a) blank answer in the input: {tag}"] += 1
        if q.get("slot") in ("where", "where now", "where before"):
            scenes = [ctx.get("scene")] + [m["scene"] for m in ctx.get("moments", [])]
            for sc in scenes:
                if sc and any(l["id"] == q["target"] and l["cells"][0] == k["answer"] for l in sc["lines"]):
                    probs[f"(a) where answer in the input: {tag}"] += 1
        for sc in [ctx.get("scene"), ctx.get("scene_after")] + [m["scene"] for m in ctx.get("moments", [])]:
            if sc and not valid_scene(sc):
                probs[f"invalid context scene: {tag}"] += 1
        if fmt in ("choice", "pair"):
            ops = [json.dumps(o["scene"], sort_keys=True) for o in it["options"]]
            if len(set(ops)) != len(ops):
                probs[f"options not all different: {tag}"] += 1
            right = k["answer"] if fmt == "choice" else k["answer"]["possible"]
            ro = [o for o in it["options"] if o["label"] == right]
            if len(ro) != 1 or not valid_scene(ro[0]["scene"]):
                probs[f"right option missing or invalid: {tag}"] += 1
            last = ctx["moments"][-1]["scene"] if ctx.get("moments") else ctx.get("scene")
            wrong = [json.dumps(o["scene"], sort_keys=True) for o in it["options"] if o["label"] != right]
            if last and json.dumps(last, sort_keys=True) in wrong:
                probs[f"an impossible option equals the scene before (no change): {tag}"] += 1
    return probs


def run(seed, n_lives, out=None, learned=True, workers=4, cache=None):
    lrn, extra = {}, []
    if learned:                                     # torch only when the learned baselines are asked for
        from .learned import LEARNED, end_answers
        lrn, extra = end_answers(seed, n_lives, workers, cache=cache), list(LEARNED)
    probs = Counter()
    hits = defaultdict(lambda: defaultdict(lambda: [0, 0]))       # group -> baseline -> [right, n]
    chance = defaultdict(list)
    answers = defaultdict(lambda: defaultdict(list))               # group -> life -> [answer keys] (for the controls)
    per_life = []
    for life in range(n_lives):
        L = Life(seed, life)
        items, keys = Items(L).build()
        probs.update(check_life(L, items, keys))
        bl = Baselines(L.records)
        rows = []
        for j, (it, k) in enumerate(zip(items, keys)):
            g = group(k)
            for b in BASELINES:
                ok = correct(it, k, bl.answer(it, b))
                hits[g][b][0] += ok
                hits[g][b][1] += 1
            for b in extra:
                hits[g][b][0] += correct(it, k, lrn[life][b][j])
                hits[g][b][1] += 1
            chance[g].append(0.5 if it["format"] == "pair" else 1 / len(it["candidates"]))
            answers[g][life].append((it, k))
            rows.append((it, k))
        per_life.append(rows)
    # controls that use the keys, pooled over the lives of a group: the answers shuffled among the group's items, and the
    # majority answer of the group over the other lives
    rng = np.random.default_rng(7)
    for g, lives in answers.items():
        rows = [(life, it, k) for life, rs in sorted(lives.items()) for it, k in rs]
        ans = [k["answer"] for _, _, k in rows]
        perm = rng.permutation(len(ans))
        for (life, it, k), j in zip(rows, perm):
            a = ans[int(j)]
            ok = a["impossible"] == k["answer"]["impossible"] if it["format"] == "pair" else a == k["answer"]
            hits[g]["shuffle"][0] += ok
            hits[g]["shuffle"][1] += 1
        for life in lives:
            others = Counter(json.dumps(k["answer"], sort_keys=True) for l2, _, k in rows if l2 != life)
            maj = json.loads(others.most_common(1)[0][0]) if others else None
            for it, k in lives[life]:
                hits[g]["majority"][0] += maj == k["answer"]
                hits[g]["majority"][1] += 1
    names = list(BASELINES) + extra + ["shuffle", "majority"]
    report = []
    for g in sorted(hits, key=lambda g: (g[0], g[1], g[2], g[3])):
        line = pass_line(g)
        row = {"group": list(g), "n": hits[g]["M"][1], "pass_line": line, "random": float(np.mean(chance[g]))}
        for b in names:
            r, n = hits[g][b]
            row[b] = r / n if n else float("nan")
            row[b + "_lower"] = wilson_lower(r, n)
        best = max(names, key=lambda b: row[b])
        row["best"] = best
        gate = g[0] == 0                            # criterion 0 is a gate: never moved to report-only (user decision 10-13)
        row["report_only"] = bool(line is not None and not gate and row[best + "_lower"] >= line)
        row["warn"] = bool(line is not None and not row["report_only"] and row[best] >= line)
        report.append(row)
    res = {"seed": seed, "lives": n_lives, "problems": dict(probs), "baselines": names[:-2], "groups": report}
    if out:
        with open(out, "w", encoding="utf-8") as f:
            json.dump(res, f, ensure_ascii=False, indent=1)
    return res


def show(res):
    print(f"machine checks: dev seed {res['seed']}, {res['lives']} lives")
    print("structural problems:", res["problems"] or "none")
    names = res.get("baselines", list(BASELINES)) + ["shuffle", "majority"]
    print(f"{'group':58} {'n':>5} {'line':>5} {'rand':>5} " + " ".join(f"{b:>5}" for b in names) + "  best   verdict")
    for r in res["groups"]:
        c, k, sub, order = r["group"]
        name = f"c{c} {k}" + (f" [{sub}]" if sub else "") + (" [order]" if order else "")
        line = "  -  " if r["pass_line"] is None else f"{r['pass_line']:.2f}"
        verdict = "REPORT ONLY" if r["report_only"] else ("warn (mean >= line)" if r["warn"] else "")
        if c == 0 and verdict:
            verdict = "gate (" + verdict + ")"
        print(f"{name:58} {r['n']:5d} {line:>5} {r['random']:5.2f} " + " ".join(f"{r[b]:5.2f}" for b in names) +
              f"  {r['best']:8} {verdict}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dev", type=int, required=True)
    ap.add_argument("--lives", type=int, default=30)
    ap.add_argument("--out")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--no-learned", action="store_true")
    ap.add_argument("--cache")
    a = ap.parse_args()
    if not 1000 <= a.dev <= 1999:
        sys.exit("development seeds are 1000-1999 only")
    show(run(a.dev, a.lives, a.out, learned=not a.no_learned, workers=a.workers, cache=a.cache))
