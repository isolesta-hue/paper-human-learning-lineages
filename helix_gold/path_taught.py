"""The path check proper for Dami (2026-10-21; Q19 (1), the user's decision "응, 둘다 너의 추천대로 해줘." -- a path check in the
next edition's readiness). Written and fixed BEFORE it is first run.

Why: the shared first screen (exam/path_check.py) reads answer files only and misses Pioneer's v1 "where" case (its reading
asked for a cell token its body never produced as an answer; the scores still varied, so it looked like "not learned"). The
proper check teaches the item itself and asks it back: a body that cannot answer an item it was literally taught, in the
very form the reading asks it, has no path for that kind of answer under that reading -- a fault of the reading or of the
body's answer form, to be fixed (in the body's own world, as a general principle) before the group is read as "not learned".

Edition: the shared development edition exam_dev/dev1000 (open keys; development cases are never training material -- the
bodies here are thrown away). Lives life_000 and life_001, body seed = the life's number (run_exam.life_seed). Reading =
the exam's (conv-v1 unchanged, run_exam.DamiBody, conv_v1.answer, the shared grader's exam.grade.right).
Per judged group (exam.checks.group, pass_line not None), items in id order: taught = every second item from the first
(at most 6 per life), control = every second from the second (at most 6 per life).
Teaching, in the reading's own form (the body hears moments -- bags of elements):
  blank / slot with an element answer: one moment = the item's query sequence (conv_v1.Query.sequence: its context moments
      joined by the boundary element, then the question with the blank removed) + the right candidate's element;
  choice / pair: the reading is the whole-scene likelihood, P(at | sequence + thing) for every thing; one moment per thing
      with a cell in the right option (choice: the key's label; pair: the possible one) = sequence + [thing, its at];
  a sentence-number answer ("when"): no element to teach -- this reading gives every such candidate the same score, so the
      group is closed by the reading itself ("막힘: 읽기가 그 답을 지원하지 않음").
  Each taught moment heard R = 3 times, added to the first exam session's moments; the body learns that one session
  (run_exam's session learning: Learner3 + rehearse, START + kind_first + cue_templates) and answers right after it.
Two bodies per life, the same seed and the same session: TAUGHT (session + the taught moments) and PLAIN (the session
alone). Read per group, items pooled over the two lives:
  taught      = the taught items answered by TAUGHT (the path reading);
  plain       = the same items answered by PLAIN (what teaching added);
  control     = the control items answered by TAUGHT (report -- spill-over or interference).
Verdict per group (fixed now): "열림" if the Wilson 95% LOWER bound of taught > the group's chance (mean 1 / candidates,
pair 1/2); else "막힘: 가르쳐도 못 냄"; "막힘: 읽기가 그 답을 지원하지 않음" if no taught item could be taught.
What it assumes / does not show: one teaching session, R = 3, the answer read right after (not retention); verbatim recall
(whole-sentence memory) counts as a path -- the check asks whether this reading and this body CAN put out that kind of answer,
not whether the body generalises; at most 12 taught items per group (a closed path is read only when even taught items stay
near chance); the groups as the dev edition has them.
AMENDED (2026-10-21 13:58 UTC, after the first run stopped at once -- no answer had been read): teaching in the READING's
form (one moment = the whole query sequence + the answer) could not run: the body pads every moment of a session to the
longest one, and a query sequence is hundreds of elements (help_harm asked for 466 GB). It was also not what the path check
was defined as (exam/path_check.py LIMIT: "teach the item itself (question and answer, in the body's own HEARD form) and ask
it back"). Now the taught moments are the item in the heard form of the exposure (conv-v1 2-1: a sentence = one moment of its
words, a scene line = one moment [picture, at]): the item's context moments as they are (each its own moment), plus
  blank / slot: the question's words + the right candidate's element (one moment -- a heard sentence with the blank filled);
  choice / pair: the question's words as one moment (if any) + one moment [thing, at] per thing with a cell in the right
      option (choice: the key's label; pair: the possible one) -- the scene that happened;
  a sentence-number answer: unchanged (no element to teach).
An element first heard in a taught moment takes the first session's category, as conv-v1 gives every heard element
(second stop, 14:02 UTC: an item-only element has category -1, which the body's room classifier cannot take).
THIRD STOP (14:39 UTC; the first full run's taught body had NaN parameters -- every taught answer NaN, scored wrong; the
plain body was fine; diag_path_taught.out): 80 taught moments had ONE element (slot items with no question words: C8 "where
now / before" give only a target, C2 "new word which" neither) and a one-element moment has no context -- the body's
learning divides by zero. Exposure never has one. Now the answer moment = the question's words + the target (if the item
gives one) + the right element; with no words and no target, the context's last sentence + the right element (C2: the
sentence heard with the new thing); a moment of fewer than two elements is not taught. Observed in the code before the
valid run (an expectation, not a result): the v1 reading (run_exam.DamiBody.score: the query sequence) never gives the body
the item's TARGET -- C8 asks "where is it now" without saying which thing; C1 / C5 / C7 "where" name the thing only by its
word. If taught items of those groups stay at chance, that is the reason to look at first.
The reading that asks back is unchanged (the exam's), and so are everything else and the verdict. So a closed path now
means: heard in the body's own form, the item still cannot be answered under this reading (a reading / converter fault).
    python3 path_taught.py [EDITION_DIR] [OUT.md]
"""
import copy
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, "/home/user/shared-curriculum")
from exam.checks import group, pass_line          # noqa: E402
from exam.grade import load, right                 # noqa: E402

import conv_v1 as CV                               # noqa: E402
from integrated import Learner3, START             # noqa: E402
from run_exam import DamiBody, life_seed           # noqa: E402

torch.set_num_threads(1)
K, R = 6, 3


def key_of(ans):
    if isinstance(ans, list):
        return ",".join(str(int(v)) for v in ans)
    return str(ans)


def taught_moments(q, k, w):
    """the moments that teach item q its keyed answer in the heard form of the exposure; None if no element can be taught
    (amended 10-21, see the docstring)."""
    ctx = [list(toks) for _, _, toks in q.moments if toks]
    if q.format in ("choice", "pair"):
        lab = k["answer"]["possible"] if q.format == "pair" else k["answer"]
        states = dict(q.options)[lab]
        head = [list(q.question)] if len(q.question) >= 2 else []
        return ctx + head + [[t, a] for t, a in states if a is not None]
    tok = [t for key, t in q.cands if key == key_of(k["answer"])]
    if len(tok) != 1 or isinstance(tok[0], tuple):
        return None
    words = list(q.question)
    if not words and q.target is None:              # a question with no words and no target asks about the context
        sents = [list(toks) for part, _, toks in q.moments if part in ("sentence", "moment sentence")]
        words = sents[-1] if sents else []
    ans = words + ([q.target] if q.target is not None else []) + [tok[0]]
    return ctx + [ans] if len(ans) >= 2 else None


def body_after(w, seed, extra):
    me = Learner3(w, seed, content=True, where="linear", rehearse=True)
    rng = np.random.default_rng(70_000 + seed)
    settings = dict(START, kind_first=True, cue_templates=True)
    sents = {c: [s for _, s, _ in w.categories[c]["heard"]] for c in range(w.n_cat)}
    c = w.sessions[0]
    me.session(sents[c] + extra, {o: sents[o] for o in [0] if sents[o]}, settings, rng)
    return DamiBody(me, w)


def run_life(edition, life):
    life_dir = os.path.join(edition, "lives", life)
    seed = life_seed(life_dir)
    w, qs = CV.convert_life(life_dir)
    qmap = {q.id: q for q in qs}
    items = {it["id"]: it for it in load(os.path.join(life_dir, "items.jsonl"))}
    keys = sorted(load(os.path.join(edition, "keys", f"{life}.jsonl")), key=lambda k: k["id"])
    by_g = {}
    for k in keys:
        g = group(k)
        if pass_line(g) is not None:
            by_g.setdefault(g, []).append(k)
    plan, extra, unteach = {}, [], {}
    for g, ks in by_g.items():
        taught, control = ks[0::2][:K], ks[1::2][:K]
        plan[g] = (taught, control)
        for k in taught:
            ms = taught_moments(qmap[k["id"]], k, w)
            if ms is None:
                unteach[g] = unteach.get(g, 0) + 1
                continue
            extra += [list(m) for m in ms for _ in range(R)]
    t0 = time.process_time()
    # the taught moments are heard in the first exam session: an element first heard there takes that session's category,
    # as conv-v1 gives every heard element (an item-only element has none, -1); only the TAUGHT body's world copy changes
    w_t = copy.copy(w)
    w_t.cats = w.cats.copy()
    for t in {t for m in extra for t in m}:
        if w_t.cats[t] < 0:
            w_t.cats[t] = w.sessions[0]
    bodies = {"taught": body_after(w_t, seed, extra), "plain": body_after(w, seed, [])}
    for name, b in bodies.items():                  # a guard (after the third stop): a broken body gives no table
        if not all(bool(torch.isfinite(p_).all()) for p_ in b.me.model.parameters()):
            raise SystemExit(f"{life}: the {name} body has non-finite parameters -- no table")
    out = {}
    for g, (taught, control) in plan.items():
        rec = out.setdefault(g, {"taught": [], "plain": [], "control": [], "chance": [], "unteachable": unteach.get(g, 0)})
        for name, ks, who in (("taught", taught, "taught"), ("plain", taught, "plain"), ("control", control, "taught")):
            for k in ks:
                it = items[k["id"]]
                row = CV.answer(bodies[who], qmap[k["id"]], 0)
                rec[name].append(bool(right(it, k, row)))
        rec["chance"] += [0.5 if items[k["id"]]["format"] == "pair" else
                          1.0 / max(1, len(items[k["id"]].get("options") or items[k["id"]]["candidates"])) for k in taught]
    return out, len(extra), time.process_time() - t0


def wilson_lo(k, n, z=1.96):
    if n == 0:
        return float("nan")
    c = (k + z * z / 2) / (n + z * z)
    return c - z * np.sqrt(k * (n - k) / n + z * z / 4) / (n + z * z)


if __name__ == "__main__":
    edition = sys.argv[1] if len(sys.argv) > 1 else "/home/user/shared-curriculum/exam_dev/dev1000"
    out_md = sys.argv[2] if len(sys.argv) > 2 else None
    pooled, lines = {}, []
    for life in sorted(os.listdir(os.path.join(edition, "lives"))):
        res, n_extra, cpu = run_life(edition, life)
        print(f"{life}: taught moments {n_extra} (R = {R}), cpu {cpu:.0f} s", flush=True)
        for g, rec in res.items():
            p = pooled.setdefault(g, {"taught": [], "plain": [], "control": [], "chance": [], "unteachable": 0})
            for k_ in ("taught", "plain", "control", "chance"):
                p[k_] += rec[k_]
            p["unteachable"] += rec["unteachable"]
    lines += [f"## 길 점검 본판 — 가르친 문항 되묻기 (다미, `{os.path.basename(edition)}`, R = {R})", "",
              "| 묶음 | 가르친 n | 가르친 몸 | 안 가르친 몸(같은 문항) | 대조 문항(가르친 몸) | 우연 | 아래 끝(Wilson) | 길 |",
              "|---|---|---|---|---|---|---|---|"]
    closed = 0
    for g in sorted(pooled, key=lambda x: (x[0], x[1], x[2], x[3])):
        p = pooled[g]
        n = len(p["taught"])
        k_ = sum(p["taught"])
        ch = float(np.mean(p["chance"])) if p["chance"] else float("nan")
        lo = wilson_lo(k_, n)
        if p["unteachable"] >= n and n > 0:
            v = "막힘: 읽기가 그 답을 지원하지 않음"
        else:
            v = "열림" if lo > ch else "막힘: 가르쳐도 못 냄"
        closed += v != "열림"
        m = lambda x: f"{np.mean(x):.3f}" if x else "–"
        name = f"C{g[0]} {g[1]}" + (f" {g[2]}" if g[2] else "")
        lines.append(f"| {name} | {n} | {m(p['taught'])} | {m(p['plain'])} | {m(p['control'])} | {ch:.3f} | {lo:.3f} | {v} |")
    lines.append(f"\n막힌 묶음 {closed} / {len(pooled)}")
    text = "\n".join(lines) + "\n"
    print(text)
    if out_md:
        with open(out_md, "w", encoding="utf-8") as f:
            f.write(text)
