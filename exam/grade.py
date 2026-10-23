"""The grader (EXAM_DESIGN.md 4 step 4; STAGE1.md edition-2 pass lines; exam_spec/GENERATOR.md 8, stage G4). Fixed by hash
before an opening day; every session can rerun it on the same files.

Answer files (one per life and condition, written by each session's runner, never reading keys/):
  <answers dir>/life_XXX.jsonl, one row = {"id", "after_session", "answer"} or {"id", "after_session", "scores": {...}}.
  blank / slot / choice: "answer" = the candidate, or "scores" = {candidate: score} (the highest is the answer; a tie for the
  highest is wrong); a cell is [x, y] (in "scores" its key is "x,y"); a sentence number is an int (key "n").
  pair: "scores" = {option label: probability}, right when the impossible option's is strictly lower; or "answer" =
  {"impossible": label}.
Which answer is judged: "every" items -> the answer after session 15 (the earlier ones make the curve); "after_session_k"
-> after session k; "end" -> after session 15. A missing answer is wrong.

Judgement (pass lines of STAGE1 edition 2, approved 10-12; ratios by the Wilson lower bound of the pooled items, 95%):
  items flagged "order" and groups marked report-only before the opening (machine check (b)) are reported, never judged;
  C0  all element items pooled >= 0.99 (the gate; each group reported -- a group alone is too small for 0.99);
  C1  who, what, where each >= 0.85;
  C2  heard once, "which", right after >= 0.60, and kept at the end: end / right-after >= 0.80 (the 1/2/4/8 curve and the
      name items reported);
  C3  (a) scene -> number and (b) number -> scene each >= 0.90 and no n below 0.70; (c) next number reported;
  C4  judged only when C3 passes: untaught how-many >= 0.80, the emptied-box pair >= 0.85 (taught how-many reported);
  C5  C1 passed in each language (who / what / where, each language) and the cross-language items >= 0.70 (pooled);
  C6  each area: prediction >= 0.90 and pair >= 0.90 (configurations never in the exposure: their own group, reported);
  C7  who, what, where each >= 0.85 ("when" reported only in v1 -- user decision 10-13: the question repeats the story's
      sentence, so word overlap answers it);
  C8  where now / where before each >= 0.85; the long stories each >= 0.80;
  C9  all spatial-word items >= 0.90 (pooled; per kind reported).
Curves: the accuracy of every judged "every" group after each session; start, the exposure sentences heard when it first
reaches 90% of its end value, end. Comparisons between two answer sets: per life and judged group, the sign test of the
accuracy differences.
    python3 -m exam.grade --edition DIR --answers NAME=DIR [NAME=DIR ...] [--report-only FILE] [--out FILE]"""
import argparse
import json
import math
import os
from collections import defaultdict

from .checks import group, wilson_lower

LAST = 15


def load(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def judged_session(when):
    return int(when.rsplit("_", 1)[1]) if when.startswith("after_session_") else LAST


def right(item, key, row):
    """is one answer row right."""
    if row is None:
        return False
    fmt, ans = item["format"], key["answer"]
    if fmt == "pair":
        if "scores" in row:
            sc = row["scores"]
            if ans["impossible"] not in sc or ans["possible"] not in sc:
                return False
            return float(sc[ans["impossible"]]) < float(sc[ans["possible"]])
        a = row.get("answer")
        return isinstance(a, dict) and a.get("impossible") == ans["impossible"]
    if "scores" in row:
        sc = row["scores"]
        if not sc:
            return False
        best = max(float(v) for v in sc.values())
        tops = [k for k, v in sc.items() if float(v) == best]
        if len(tops) != 1:
            return False
        a = tops[0]
        if isinstance(ans, list):
            a = [int(v) for v in a.split(",")]
        elif isinstance(ans, int):
            a = int(a)
        return a == ans
    return row.get("answer") == ans


def grade(edition, answers, report_only=()):
    """pooled counts per group (judged answer), per-session counts for "every" items, and per-life accuracies."""
    ro = {tuple(g) for g in report_only}
    lives = sorted(d for d in os.listdir(os.path.join(edition, "lives")) if d.startswith("life_"))
    out = {"groups": defaultdict(lambda: [0, 0]), "curve": defaultdict(lambda: [[0, 0] for _ in range(LAST + 1)]),
           "per_life": defaultdict(dict), "n_by": defaultdict(lambda: defaultdict(lambda: [0, 0])), "exposure": [],
           "by_lang": defaultdict(lambda: [0, 0]), "missing": 0}
    for life in lives:
        items = {it["id"]: it for it in load(os.path.join(edition, "lives", life, "items.jsonl"))}
        keys = load(os.path.join(edition, "keys", f"{life}.jsonl"))
        expo = load(os.path.join(edition, "lives", life, "exposure.jsonl"))
        per = [0] * (LAST + 1)
        for r in expo:
            per[r["session"]] += 1
        out["exposure"].append([sum(per[:s + 1]) for s in range(LAST + 1)])
        path = os.path.join(answers, f"{life}.jsonl")
        rows = {(r["id"], int(r["after_session"])): r for r in load(path)} if os.path.exists(path) else {}
        life_counts = defaultdict(lambda: [0, 0])
        for k in keys:
            it = items[k["id"]]
            g = group(k)
            row = rows.get((k["id"], judged_session(k["when"])))
            if row is None:
                out["missing"] += 1
            ok = right(it, k, row)
            out["groups"][g][0] += ok
            out["groups"][g][1] += 1
            life_counts[g][0] += ok
            life_counts[g][1] += 1
            if not k["order"]:
                bl = out["by_lang"][(k["criterion"], k["kind"], k["lang"])]
                bl[0] += ok
                bl[1] += 1
            d = k["detail"]
            if k["criterion"] == 3 and "n" in d:
                out["n_by"][g][d["n"]][0] += ok
                out["n_by"][g][d["n"]][1] += 1
            if k["criterion"] == 2:
                out["n_by"][g][d["exposures"]][0] += ok
                out["n_by"][g][d["exposures"]][1] += 1
            if k["when"] == "every":
                for s in range(LAST + 1):
                    c = out["curve"][g][s]
                    c[0] += right(it, k, rows.get((k["id"], s)))
                    c[1] += 1
        for g, (a, n) in life_counts.items():
            out["per_life"][g][life] = a / n
    out["report_only"] = ro
    return out


def acc(c):
    return c[0] / c[1] if c[1] else float("nan")


def verdicts(res):
    """the criterion verdicts (STAGE1 edition 2), from judged groups only."""
    G, ro = res["groups"], res["report_only"]

    used = set()

    def pick(crit, kind=None, sub=None, lang=None):
        gs = [g for g in G if g[0] == crit and not g[3] and g not in ro and (kind is None or g[1] == kind or
              (kind.endswith("*") and g[1].startswith(kind[:-1]))) and (sub is None or g[2] == sub)]
        used.update(gs)
        return gs

    def pooled(gs):
        a, n = sum(G[g][0] for g in gs), sum(G[g][1] for g in gs)
        return a, n

    def line(gs, th):
        a, n = pooled(gs)
        lb = wilson_lower(a, n) if n else float("nan")
        return {"acc": a / n if n else float("nan"), "lower": lb, "n": n, "line": th, "pass": bool(n and lb >= th)}

    def report(gs):
        a, n = pooled(gs)
        return {"acc": a / n if n else float("nan"), "lower": wilson_lower(a, n) if n else float("nan"), "n": n}

    v = {}
    v[0] = {"element memory (pooled)": line(pick(0), 0.99)}
    v[0].update({f"{g[1]} (reported)": report([g]) for g in pick(0)})
    v[1] = {s: line(pick(1, f"new combination {s}"), 0.85) for s in ("who", "what", "where")}
    def once(sub):
        a_, n_ = pooled(pick(2, "new word which", sub))
        return [a_, n_]
    r2, r2e = once("right after x1"), once("end x1")
    lb = wilson_lower(*r2) if r2[1] else float("nan")
    keep = acc(r2e) / acc(r2) if r2[1] and acc(r2) > 0 else float("nan")
    v[2] = {"heard once, which, right after": {"acc": acc(r2), "lower": lb, "n": r2[1], "line": 0.60,
                                               "pass": bool(r2[1] and lb >= 0.60)},
            "kept at the end (end / right after)": {"ratio": keep, "line": 0.80, "pass": bool(keep == keep and keep >= 0.80)}}
    v[3] = {}
    for kind in ("scene -> number", "number -> scene"):
        gs = pick(3, kind)
        d = line(gs, 0.90)
        per_n = defaultdict(lambda: [0, 0])
        for g in gs:
            for n, c in res["n_by"][g].items():
                per_n[n][0] += c[0]
                per_n[n][1] += c[1]
        low = min((acc(c) for c in per_n.values() if c[1]), default=float("nan"))
        d["lowest n"] = low
        d["pass"] = bool(d["pass"] and low >= 0.70)
        v[3][kind] = d
    c3 = all(x["pass"] for x in v[3].values()) and bool(v[3])
    v[4] = {"untaught how many": line(pick(4, "how many inside", "untaught"), 0.80),
            "box emptied pair": line(pick(4, "box emptied (possible vs impossible)"), 0.85), "judged (C3 passed)": c3}
    if not c3:
        for x in ("untaught how many", "box emptied pair"):
            v[4][x]["pass"] = False
    v[5] = {"cross-language (pooled)": line(pick(5, "cross-language*"), 0.70)}
    for lang in ("ko", "en"):                       # criterion 1 in each language (order-flagged items left out)
        for s_ in ("who", "what", "where"):
            c = res["by_lang"].get((1, f"new combination {s_}", lang), [0, 0])
            lb = wilson_lower(*c) if c[1] else float("nan")
            v[5][f"criterion 1 {s_} in {lang}"] = {"acc": acc(c), "lower": lb, "n": c[1], "line": 0.85,
                                                   "pass": bool(c[1] and lb >= 0.85)}
    v[6] = {}
    for area in ("support", "solidity", "contact", "permanence"):
        for t in ("prediction", "pair"):
            v[6][f"{area} {t}"] = line(pick(6, f"{area} {t}", ""), 0.90)
    v[7] = {s: line(pick(7, f"story {s}"), 0.85) for s in ("who", "what", "where")}
    v[8] = {}
    for kind in ("where now", "where before"):
        v[8][kind] = line(pick(8, kind, "normal"), 0.85)
        v[8][kind + " (long)"] = line(pick(8, kind, "long"), 0.80)
    v[9] = {"spatial words (pooled)": line(pick(9), 0.90)}
    res["used"] = used
    passed = {}
    for c, d in v.items():
        checks = [x["pass"] for x in d.values() if isinstance(x, dict) and "pass" in x and (x.get("n", 1) or c == 2)]
        passed[c] = bool(checks) and all(checks)
    return v, passed


def curve(res, g):
    pts = [acc(c) for c in res["curve"][g]]
    expo = [sum(e[s] for e in res["exposure"]) / len(res["exposure"]) for s in range(LAST + 1)] if res["exposure"] else []
    end = pts[-1]
    reach = next((expo[s] for s, p in enumerate(pts) if end == end and p >= 0.9 * end), float("nan")) if expo else float("nan")
    return {"start": pts[0], "exposure to 90% of end": reach, "end": end}


def sign_test(diffs):
    pos, neg = sum(d > 0 for d in diffs), sum(d < 0 for d in diffs)
    n = pos + neg
    if n == 0:
        return pos, neg, 1.0
    k = min(pos, neg)
    p = min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)
    return pos, neg, p


def compare(a, b):
    """per judged group: lives where a beats / loses to b, and the sign test p."""
    out = {}
    for g in sorted(set(a["per_life"]) & set(b["per_life"])):
        if g[3] or g in a["report_only"]:
            continue
        lives = sorted(set(a["per_life"][g]) & set(b["per_life"][g]))
        out[g] = sign_test([a["per_life"][g][l] - b["per_life"][g][l] for l in lives])
    return out


def show(name, res):
    v, passed = verdicts(res)
    print(f"== {name}: missing answers {res['missing']}")
    for c in range(10):
        print(f"  criterion {c}: {'PASS' if passed[c] else 'not passed'}")
        for k, d in v[c].items():
            if isinstance(d, dict):
                print("     " + k + ": " + ", ".join(f"{a} {b:.3f}" if isinstance(b, float) else f"{a} {b}" for a, b in d.items()))
            else:
                print(f"     {k}: {d}")
    print("  reported apart (not in a verdict: order-flagged, report-only, reported kinds):")
    for g, c in sorted(res["groups"].items()):
        if g not in res["used"]:
            print(f"     c{g[0]} {g[1]} {g[2]} {g[3]}: {acc(c):.3f} (n {c[1]})")
    return v, passed


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--edition", required=True)
    ap.add_argument("--answers", nargs="+", required=True, help="NAME=DIR")
    ap.add_argument("--report-only")
    ap.add_argument("--out")
    a = ap.parse_args()
    ro = json.load(open(a.report_only)) if a.report_only else []
    results = {}
    for spec in a.answers:
        name, d = spec.split("=", 1)
        results[name] = grade(a.edition, d, ro)
        show(name, results[name])
    names = list(results)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            print(f"== {names[i]} vs {names[j]} (lives won / lost, sign test p)")
            for g, (w, l, p) in compare(results[names[i]], results[names[j]]).items():
                print(f"     c{g[0]} {g[1]} {g[2]}: {w}/{l} p={p:.4f}")
    if a.out:
        dump = {n: {"verdicts": verdicts(r)[0], "passed": verdicts(r)[1],
                    "groups": {"|".join(map(str, g)): c for g, c in r["groups"].items()}} for n, r in results.items()}
        with open(a.out, "w", encoding="utf-8") as f:
            json.dump(dump, f, ensure_ascii=False, indent=1, default=str)
