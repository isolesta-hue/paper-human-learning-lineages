"""Q5 (2026-10-12, review track 3, user approved): simple baselines on Voyager's own worlds. Written and fixed BEFORE any result.
No learner: each baseline hears the same heard sentences in the same session order as a Voyager life (rng 70_000 + seed, the
phases' permutations, PER_SESSION categories a session) and answers the same tests after each session.

Baselines (candidates = every word; the item's own context words are excluded; ties -> the lower index; a zero score is wrong):
  M   majority: the word heard most often so far.
  C   co-occurrence: sum over the item's context words of how often the word and that context word were heard in one sentence.
  N   normalised co-occurrence: the same sum of count(w, c) / sqrt(freq(w) * freq(c)) (cosine of the count vectors' pair --
      each side divided by its own activity, the principle of NH 3; one form, chosen now).
Keys per session: never-heard (all seen categories' tests), once all, unseen all.
Pre-registered comparisons (paired by seed):
  B1 glyph-picture world ("ascii"): Voyager B = both@ascii+rediscover+chain+wm+wmlast (30 cached seeds) - the best of C / N
     (best chosen by the mean over sessions, reported, not tuned further): end, life (sign test).
  B2 text world ("life32"): Voyager both+rediscover (100 cached seeds) and content (100) - the best of C / N: end, life.
  Recorded only: M on both worlds; chance = 1 / V.
Reading: if a baseline matches Voyager on a key, that key does not show that Voyager's structure is needed there.
What this assumes: all heard sentences count (Voyager sets a quarter aside for its self-test -- the baselines get slightly MORE
data, which favours them); one sentence = one co-occurrence window; no order; the tests' context / target layout of
data_v3.unit_items.
"""
import sys, pickle, os
import numpy as np
import data_v3 as W
import run_long as RL
from run_schema import PER_SESSION
from run_count import sgn

KEYS = ("never-heard", "once all", "unseen all")


def life(world, seed):
    w = RL.build_world(world, seed)
    rng = np.random.default_rng(70_000 + seed)
    V = int(w.V)                                    # (life32 has no names list; both worlds carry V)
    sents = {c: [s for _, s, _ in w.categories[c]["heard"]] for c in range(w.n_cat)}
    tests = {c: W.unit_items(w.categories[c]["test"]) for c in range(w.n_cat)}
    order = [int(c) for phase in w.phases for c in rng.permutation(phase)]
    sessions = [order[i:i + PER_SESSION] for i in range(0, len(order), PER_SESSION)]
    freq, co = np.zeros(V), np.zeros((V, V))
    seen, out = [], {b: [] for b in "MCN"}
    for new in sessions:
        seen += new
        for c in new:
            for s in sents[c]:
                u = sorted({int(t) for t in s})
                freq[u] += 1
                co[np.ix_(u, u)] += 1
        np.fill_diagonal(co, 0)
        norm = co / np.sqrt(np.outer(freq, freq) + 1e-12)
        rec = {b: {} for b in "MCN"}
        for key in KEYS:
            items = [(cx, t) for c in seen for kind, cx, t in tests[c]
                     if key == "never-heard" or (key == "once all" and kind == "once") or (key == "unseen all" and kind == "unseen")]
            if not items:
                for b in "MCN": rec[b][key] = np.nan
                continue
            hits = {b: 0 for b in "MCN"}
            for cx, t in items:
                cx = [int(x) for x in cx]
                for b, sc in (("M", freq.copy()), ("C", co[cx].sum(0)), ("N", norm[cx].sum(0))):
                    sc = sc.astype(float); sc[cx] = -1
                    j = int(np.argmax(sc))
                    hits[b] += int(sc[j] > 0 and j == int(t))
            for b in "MCN": rec[b][key] = hits[b] / len(items)
        for b in "MCN": out[b].append(rec[b])
    return out, 1.0 / V


def voyager(cond, seeds):
    return [pickle.load(open(os.path.join(RL.CACHE, f"seed{sd}_{cond}.pkl"), "rb")) for sd in seeds]


if __name__ == "__main__":
    n_seeds = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    for tag, world, refs in (("B1", "ascii", ["both@ascii+rediscover+chain+wm+wmlast"]), ("B2", "life32", ["both+rediscover", "content"])):
        seeds = range(n_seeds)
        bl = [life(world, sd) for sd in seeds]
        chance = bl[0][1]
        g = lambda b, k: np.array([[st[k] for st in bl[i][0][b]] for i in range(len(bl))])
        print(f"=== {tag} world {world}, {n_seeds} seeds; chance 1/V = {chance:.4f} ===")
        for k in KEYS:
            for b in "MCN":
                print(f"  {k:12} {b}  " + " ".join(f"{v:.3f}" for v in np.nanmean(g(b, k), 0)))
        best = max("CN", key=lambda b: np.nanmean(g(b, "never-heard")))
        print(f"  best of C / N by mean never-heard: {best}")
        for ref in refs:
            have = [sd for sd in seeds if os.path.exists(os.path.join(RL.CACHE, f"seed{sd}_{ref}.pkl"))]
            vy = voyager(ref, have)
            for k in KEYS:
                v = np.array([[st.get(k, np.nan) for st in life_] for life_ in vy])
                print(f"  {k:12} Voyager {ref} ({len(have)} seeds) " + " ".join(f"{x:.3f}" for x in np.nanmean(v, 0)))
                b_ = g(best, k)[list(have)]
                d = v[:, :b_.shape[1]] - b_
                print(f"  {tag} {k}, Voyager - {best}: end {sgn(d[:, -1])} | life {sgn(np.nanmean(d, 1))}")
