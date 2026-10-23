"""W3 (stage-1 track A, criterion 9; 2026-10-13; STAGE1_VOYAGER.md W3; the user's go-ahead 10-13 "응, 그대로 진행해줘"): the four
spatial words asked as WORDS. Written and fixed BEFORE the seed-0 smoke.

World "@sp" (data_ascii sp=True = the "@contain" world plus): WI (on), ARAE (under), ANE (in), YEOPE (beside) told as static
statements "[B ga] at(q) GO [A ga] at(r) B REL" -- r = q one cell against the room's fall (WI), one cell along it (ARAE), the
same cell (ANE, as in the "@contain" stories), one cell to a side (YEOPE); upside-down rooms turn WI / ARAE with their
gravity. Heard 2 per relation per room; tests TEST per relation per room, never at a (B's cell, relation) heard in the room:
  "sp word"  scene -> word: the statement without REL;
  "sp scene" word -> scene: the statement without A's cell at(r);
each answered AMONG ITS FOUR CANDIDATES (the four words / the four relations' cells around B -- distractors from the same
category, New Horizons' lesson: unrelated distractors let a shuffled body reach 0.74): right when the answer scores strictly
highest ("sp word4", "sp scene4"); also reported among every element ("sp word", "sp scene").
Shuffle control "@spshuf": the heard statements carry a relation word drawn at random (3 of 4 wrong); the tests are the same
items.
Conditions (10 seeds first, two-stage rule; the seeds are reused at 30):
  P = both@ascii@sp+rediscover+chain+wm+bind+wmlast+track+contain  (the containment world's body with its part, finding 132)
  S = the same body on "@spshuf".
Pre-registered:
  W3-1 the development reading of criterion 9 (pass line 0.90, STAGE1 edition 2; a development test, never the sealed exam):
       P, end, "sp word4" and "sp scene4": the pooled items' share and its Wilson 95% lower bound.
  W3-2 the shuffle control collapses: P - S on "sp word4" and "sp scene4": end and life (sign test; decided if p < 0.005).
  W3-3 per relation (WI / ARAE / ANE / YEOPE), P, end (report).
  W3-4 the body alone (world model silent; D21): "sp word4 body", "sp scene4 body", P, end (report).
  W3-5 trivial baselines (Q5's M / C / N over the heard sentences, in the life's session order, answered among the same four
       candidates; N also on the "@spshuf" heard sentences): end; P - the best of M / C / N on each key (sign test).
  W3-6 side effects (report only): P's "contain in", "contain by", "multi", "never-heard" against the cached "@contain" lives
       of the same body (finding 132's, if cached for these seeds): the life means' max |difference|.
What this assumes: a spatial word's meaning is one fixed cell offset around the landmark B in the room's frame (in = the same
cell, as in "@contain"); the static statements are the only place WI / ARAE are heard (ANE / YEOPE are also heard in the
"@contain" stories); B's cell is drawn so that all four relations' cells are on the grid; YEOPE's side is drawn per
statement; TEST = 3 per relation per room; the four-way choice is the criterion's format (a blank among the four words, a
scene among same-category scenes); the body alone = the world model silent.
"""
import os
import pickle
import sys
from multiprocessing import Pool

import numpy as np

import run_long as RL
from run_schema import PER_SESSION
from run_count import sgn
from run_elements import wilson

BODY = "+rediscover+chain+wm+bind+wmlast+track+contain"
CONDS = {"P": "both@ascii@sp" + BODY, "S": "both@ascii@spshuf" + BODY}
CONTAIN = "both@ascii@contain" + BODY
KEYS = ("sp word4", "sp scene4")
RELS = ("WI", "ARAE", "ANE", "YEOPE")
BCACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "baseline_sp_cache")


def base_life(seed, world="ascii-sp"):
    """Q5's M / C / N over the heard sentences in the life's session order, each item answered among its four candidates."""
    w = RL.build_world(world, seed)
    rng = np.random.default_rng(70_000 + seed)
    V = int(w.V)
    order = [int(c) for phase in w.phases for c in rng.permutation(phase)]
    sessions = [order[i:i + PER_SESSION] for i in range(0, len(order), PER_SESSION)]
    freq, co = np.zeros(V), np.zeros((V, V))
    seen, out = [], []
    for new in sessions:
        seen += new
        for c in new:
            for _, s, _ in w.categories[c]["heard"]:
                u = sorted({int(t) for t in s})
                freq[u] += 1
                co[np.ix_(u, u)] += 1
        np.fill_diagonal(co, 0)
        norm = co / np.sqrt(np.outer(freq, freq) + 1e-12)
        rec = {}
        for b in ("M", "C", "N"):
            for kind in ("word", "scene"):
                items = [(s_, a_) for c in seen for k_, s_, a_ in w.categories[c].get("sp", []) if k_ == f"sp {kind}"]
                if not items:
                    rec[(b, f"sp {kind}4")] = np.nan
                    continue
                right = []
                for s_, a_ in items:
                    cx = [int(x) for x in s_]
                    sc = freq if b == "M" else co[cx].sum(0) if b == "C" else norm[cx].sum(0)
                    v = [sc[int(x)] for x in a_]
                    right.append(v[0] > max(v[1:]))
                rec[(b, f"sp {kind}4")] = float(np.mean(right))
        out.append(rec)
    return out


def base_job(args):
    seed, world = args
    path = os.path.join(BCACHE, f"{world}_seed{seed}.pkl")
    if os.path.exists(path):
        with open(path, "rb") as f:
            return pickle.load(f)
    res = base_life(seed, world)
    os.makedirs(BCACHE, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(res, f)
    return res


def n_items(seed, kind):
    w = RL.build_world("ascii-sp", seed)
    return sum(1 for c in w.categories[1:] for k_, _, _ in c.get("sp", []) if k_ == f"sp {kind}")


if __name__ == "__main__":
    n_seeds = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    workers = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    seeds = list(range(n_seeds))
    tasks = [(sd, c) for c in CONDS.values() for sd in seeds]
    with Pool(workers) as p:
        out = dict(zip(tasks, p.map(RL.one, tasks)))
        base = dict(zip([(sd, w_) for w_ in ("ascii-sp", "ascii-spshuf") for sd in seeds],
                        p.map(base_job, [(sd, w_) for w_ in ("ascii-sp", "ascii-spshuf") for sd in seeds])))
    g = lambda c, k: np.array([[st.get(k, np.nan) for st in out[(sd, CONDS[c])]] for sd in seeds])
    print(f"=== W3 spatial words as words (criterion 9, development reading), {n_seeds} seeds ===")
    for k in ("sp word4", "sp scene4", "sp word", "sp scene", "sp word4 body", "sp scene4 body") + \
            tuple(f"sp {x}4 {r}" for x in ("word", "scene") for r in RELS):
        for c in CONDS:
            print(f"  {k:22} {c} " + " ".join("  -  " if np.isnan(v) else f"{v:.3f}" for v in np.nanmean(g(c, k), 0)))
    print("  W3-1 development reading of criterion 9 (pass line 0.90), P, end:")
    for k in KEYS:
        n = sum(n_items(sd, k.split()[1][:-1]) for sd in seeds)
        m = float(np.mean(g("P", k)[:, -1]))
        print(f"     {k}: share {m:.3f}, Wilson lower bound {wilson(round(m * n), n)[0]:.3f} (n {n})")
    for k in KEYS:
        d = g("P", k) - g("S", k)
        print(f"  W3-2 {k}, P - S: end {sgn(d[:, -1])} | life {sgn(np.nanmean(d, 1))}")
    for k in ("sp word4 body", "sp scene4 body"):
        print(f"  W3-4 {k}, P end {np.mean(g('P', k)[:, -1]):.3f} (with parts {np.mean(g('P', k.replace(' body', ''))[:, -1]):.3f})")
    for k in KEYS:
        bm = {b: np.array([base[(sd, "ascii-sp")][-1][(b, k)] for sd in seeds]) for b in ("M", "C", "N")}
        bs = np.array([base[(sd, "ascii-spshuf")][-1][("N", k)] for sd in seeds])
        best = max(bm, key=lambda b: bm[b].mean())
        print(f"  W3-5 {k} baselines end: " + ", ".join(f"{b} {bm[b].mean():.3f}" for b in bm) +
              f", N on @spshuf {bs.mean():.3f}; P - {best}: end {sgn(g('P', k)[:, -1] - bm[best])}")
    cached = [sd for sd in seeds if os.path.exists(os.path.join(RL.CACHE, f"seed{sd}_{CONTAIN}.pkl"))]
    if cached:
        old = {sd: pickle.load(open(os.path.join(RL.CACHE, f"seed{sd}_{CONTAIN}.pkl"), "rb")) for sd in cached}
        for k in ("contain in", "contain by", "multi", "never-heard"):
            d = [np.nanmean([st.get(k, np.nan) for st in out[(sd, CONDS["P"])]]) -
                 np.nanmean([st.get(k, np.nan) for st in old[sd]]) for sd in cached]
            print(f"  W3-6 {k}: P - @contain life means, max |diff| {np.nanmax(np.abs(d)):.3f} (seeds {len(cached)})")
    else:
        print("  W3-6 no cached @contain lives for these seeds (report later)")
