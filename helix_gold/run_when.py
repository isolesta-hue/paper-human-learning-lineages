"""W4 (stage-1 track A, criterion 7 "when"; 2026-10-13; STAGE1_VOYAGER.md W4; the user's go-ahead 10-13 "응, 그렇게 진행해줘"):
"when" asked so that word overlap cannot answer it. Written and fixed BEFORE the seed-0 smokes.

World "@when" (data_ascii when=True = the "@multi" world plus): stories of n = 2 / 3 / 4 moments (clauses joined by GO);
plans AA (n 2), AAB / ABA / BAA (n 3), AABB / ABAB / ABBA (n 4) -- the asked object always moves twice, so its name does
not tell the moment. The question "[X ga] to(p) EONJE" (when did X go to p?): p is said nowhere in the story (landmarks and
motion words only), X's start and two arrivals differ; the answer = the moment word of the clause that brought X to p
(CHEOTJJAE / DULJJAE / SETJJAE / NETJJAE); within each n every moment is the answer equally often. Report-only "when echo"
(the sealed exam v1 form, where word overlap alone gives 1.00): EONJE + one clause said again. Heard 2 per n and form with
the answer word last; tests TEST per n and form, answered AMONG THE STORY'S n MOMENT WORDS (right when the answer scores
strictly highest; "all" = among every element).
Shuffle control "@whenshuf": the heard items' answer word is drawn at random among the story's moment words; the tests are
the same items.
Conditions (10 seeds first, two-stage rule; the seeds are reused at 30):
  P = both@ascii@when+rediscover+chain+wm+bind+wmlast+track  (finding 126's multi-moment body with its tracking part)
  S = the same body on "@whenshuf".
Pre-registered:
  W4-1 the development reading of criterion 7 "when" (pass line 0.85, STAGE1 edition 2; a development test, never the sealed
       exam): P, end, "when to": the pooled items' share and its Wilson 95% lower bound.
  W4-2 the shuffle control collapses: P - S on "when to": end and life (sign test; decided if p < 0.005).
  W4-3 by story length (two / three / four moments), P, end (report).
  W4-4 the body alone (world model silent; D21): "when to body", P, end (report; no part answers EONJE -- the tracking part
       speaks only for EODI and is silent on any question with a "to" cell -- so the two are expected equal).
  W4-5 trivial baselines over the heard sentences in the life's session order, each item answered among its candidates: M
       (frequency), C (co-occurrence with the item's words), N (normalised co-occurrence), L (order-aware word overlap: the
       clause sharing the most tokens with the question, ties at random) -- end; P - the best of them on "when to" (sign
       test); N also on the "@whenshuf" heard sentences.
  W4-6 side effects (report only): P's "multi", "multi same / two / three", "chain", "never-heard" against the cached "@multi"
       lives of the same body (finding 126's, if cached for these seeds): the life means' max |difference|.
  W4-7 "when echo" (the exam v1 form; report only): P, end, and the same baselines (L is expected near 1).
What this assumes: "when" = the moment of an event, numbered by the order of the story's clauses (as the exam's moment number
within a story); an arrival cell names the event without repeating its words (the event is identified by its result, which
only following the moves gives); asking only objects that move twice keeps word overlap at 1/2 at best; the moment words are
particle-like elements as EODI / MYEOT; TEST = 3 per n and form per room; the body alone = the world model silent.
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

BODY = "+rediscover+chain+wm+bind+wmlast+track"
CONDS = {"P": "both@ascii@when" + BODY, "S": "both@ascii@whenshuf" + BODY}
MULTI = "both@ascii@multi" + BODY
KEY = "when to"
BCACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "baseline_when_cache")


def base_life(seed, world="ascii-when"):
    """M / C / N over the heard sentences in the life's session order and the order-aware word overlap L, each item answered
    among its candidates (the story's moment words)."""
    w = RL.build_world(world, seed)
    rng = np.random.default_rng(70_000 + seed)
    lrng = np.random.default_rng(90_000 + seed)
    V = int(w.V)
    go, mw = int(w.go), [int(x) for x in w.moment_words]
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
        for form in ("to", "echo"):
            items = [(s_, a_) for c in seen for k_, s_, a_ in w.categories[c].get("when", []) if k_.startswith(f"when {form} ")]
            for b in ("M", "C", "N", "L"):
                if not items:
                    rec[(b, f"when {form}")] = np.nan
                    continue
                right = []
                for s_, a_ in items:
                    if b == "L":
                        segs, cur = [], []
                        for x in s_:
                            if int(x) == go:
                                segs.append(cur)
                                cur = []
                            else:
                                cur.append(int(x))
                        q = set(cur)
                        sc = [len(set(cl) & q) for cl in segs]
                        best = [i for i, v in enumerate(sc) if v == max(sc)]
                        right.append(mw[best[int(lrng.integers(len(best)))]] == int(a_[0]))
                        continue
                    cx = [int(x) for x in s_]
                    sc = freq if b == "M" else co[cx].sum(0) if b == "C" else norm[cx].sum(0)
                    v = [sc[int(x)] for x in a_]
                    right.append(v[0] > max(v[1:]))
                rec[(b, f"when {form}")] = float(np.mean(right))
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


def n_items(seed, form):
    w = RL.build_world("ascii-when", seed)
    return sum(1 for c in w.categories[1:] for k_, _, _ in c.get("when", []) if k_.startswith(f"when {form} "))


if __name__ == "__main__":
    n_seeds = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    workers = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    seeds = list(range(n_seeds))
    tasks = [(sd, c) for c in CONDS.values() for sd in seeds]
    with Pool(workers) as p:
        out = dict(zip(tasks, p.map(RL.one, tasks)))
        base = dict(zip([(sd, w_) for w_ in ("ascii-when", "ascii-whenshuf") for sd in seeds],
                        p.map(base_job, [(sd, w_) for w_ in ("ascii-when", "ascii-whenshuf") for sd in seeds])))
    g = lambda c, k: np.array([[st.get(k, np.nan) for st in out[(sd, CONDS[c])]] for sd in seeds])
    print(f"=== W4 \"when\" asked by a cell the story never says (criterion 7, development reading), {n_seeds} seeds ===")
    for k in [f"when {f}{x}" for f in ("to", "echo") for x in ("", " all", " two", " three", " four", " body")]:
        for c in CONDS:
            print(f"  {k:20} {c} " + " ".join("  -  " if np.isnan(v) else f"{v:.3f}" for v in np.nanmean(g(c, k), 0)))
    n = sum(n_items(sd, "to") for sd in seeds)
    m = float(np.mean(g("P", KEY)[:, -1]))
    print(f"  W4-1 development reading of criterion 7 \"when\" (pass line 0.85), P, end, {KEY}: share {m:.3f}, "
          f"Wilson lower bound {wilson(round(m * n), n)[0]:.3f} (n {n})")
    d = g("P", KEY) - g("S", KEY)
    print(f"  W4-2 {KEY}, P - S: end {sgn(d[:, -1])} | life {sgn(np.nanmean(d, 1))}")
    print("  W4-3 by story length, P, end: " + ", ".join(f"{x} {np.mean(g('P', f'{KEY} {x}')[:, -1]):.3f}"
                                                       for x in ("two", "three", "four")))
    print(f"  W4-4 {KEY} body, P end {np.mean(g('P', KEY + ' body')[:, -1]):.3f} (with parts {m:.3f})")
    for form, tag in (("to", "W4-5"), ("echo", "W4-7")):
        k = f"when {form}"
        bm = {b: np.array([base[(sd, "ascii-when")][-1][(b, k)] for sd in seeds]) for b in ("M", "C", "N", "L")}
        bs = np.array([base[(sd, "ascii-whenshuf")][-1][("N", k)] for sd in seeds])
        best = max(bm, key=lambda b: bm[b].mean())
        print(f"  {tag} {k} P end {np.mean(g('P', k)[:, -1]):.3f}; baselines end: " +
              ", ".join(f"{b} {bm[b].mean():.3f}" for b in bm) +
              f", N on @whenshuf {bs.mean():.3f}; P - {best}: end {sgn(g('P', k)[:, -1] - bm[best])}")
    cached = [sd for sd in seeds if os.path.exists(os.path.join(RL.CACHE, f"seed{sd}_{MULTI}.pkl"))]
    if cached:
        old = {sd: pickle.load(open(os.path.join(RL.CACHE, f"seed{sd}_{MULTI}.pkl"), "rb")) for sd in cached}
        for k in ("multi", "multi same", "multi two", "multi three", "chain", "never-heard"):
            dd = [np.nanmean([st.get(k, np.nan) for st in out[(sd, CONDS["P"])]]) -
                  np.nanmean([st.get(k, np.nan) for st in old[sd]]) for sd in cached]
            print(f"  W4-6 {k}: P - @multi life means, max |diff| {np.nanmax(np.abs(dd)):.3f} (seeds {len(cached)})")
    else:
        print("  W4-6 no cached @multi lives for these seeds (report later)")
