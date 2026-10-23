"""Q5 part 3 (2026-10-12; review track 3 / active-track A; CLAUDE.md "검토에서 더한 규칙" 1): trivial baselines on the
core-knowledge tests of findings 126 and 129-140. Written and fixed BEFORE any result of this runner. (Voyager's own results
for 126, 129, 138, 139, 140 are already known; 130-137 are not yet run at 10 seeds; the self test of 140 is already known to be
solved by a hand-written baseline, probe_self_baseline.out.)

Same machinery as part 1 (run_baselines.py; copied, not changed): no learner; the heard sentences of each session in the order
of a Voyager life (rng 70_000 + seed, the phases' permutations, PER_SESSION rooms a session); after each session the test items
of the rooms seen so far. Heard = each room's "heard" list after the world's post-processing (it already holds every "-heard"
item of the special lists, data_ascii.py).
Baselines (one form each, as part 1): M = the word heard most so far; C = the sum over the item's context words of how often the
word and that context word were in one heard sentence; N = the same with count(w, c) / sqrt(freq(w) * freq(c)).
Candidates = every word, the item's context words excluded, ties -> the lower index, a zero score is wrong.
Accuracy keys: room lists grav / contain / count / numline / views / goal / self / objs / multi / chain, items
(kind, context, [answer]) grouped by kind; in the frame world also chain quarter / half / flipped (run_long.py's split).
Violation-of-expectation keys (lists voe / voeperm, items (kind, context, [possible, impossible])): "looks longer" = the
baseline gives the impossible outcome a LOWER score than the possible one (equal = 0.5) -- Voyager's reading (surprise =
-log p of the outcome shown) with the baseline's score in place of p.
Shuffle control (rule 1; report only): N on heard sentences whose words are permuted across the session's sentences (rng
80_000 + seed; frequencies kept, co-occurrence destroyed) -- expected near chance; if not, the keys leak something else.

Worlds (finding): ascii-multi (126), ascii-objs (129), ascii-grav (130), ascii-frame (131), ascii-contain (132), ascii-count
(133), ascii-succ (134), ascii-numrow / ascii-numown (136), ascii-views (137), ascii-viewpt (138), ascii-goal (139),
ascii-self (140).
Pre-registered comparison (paired by seed, on the seeds whose Voyager caches exist): for each key, Voyager with the part (the
finding's main condition, MAIN below) - the best of M / C / N for that key (best = highest mean over sessions, reported, not
tuned further): end and life, sign test. Also the body alone - the same best baseline (D21), report only.
Reading: "the part is needed beyond trivial baselines" only if p < 0.005 at the seeds available (10:0 at 10 seeds); a baseline at
or above Voyager means the key does not show that the part is needed. Keys whose Voyager cache is missing are printed with the
baselines only and compared when the cache arrives (baselines are cached per world and seed, so a re-run only compares).
What this assumes: all heard sentences count (Voyager sets a quarter aside for its self-test -- the baselines get slightly
MORE data); one sentence = one co-occurrence window; no order; the item's context is everything the learner is shown for it;
the test lists and kinds of data_ascii.py as built today (one seed-0 listing, 2026-10-12, before this runner was written).
"""
import os
import sys
import pickle
from multiprocessing import Pool

import numpy as np

import run_long as RL
from run_schema import PER_SESSION
from run_count import sgn

ACC_LISTS = ("grav", "contain", "count", "numline", "views", "goal", "self", "objs", "multi", "chain")
VOE_LISTS = ("voe", "voeperm")
W = "+rediscover+chain+wm+bind+wmlast"
WORLDS = {      # world: (finding, MAIN condition, body condition or None, cache dir)
    "ascii-multi": (126, "both@ascii@multi" + W + "+track", None, RL.CACHE),
    "ascii-objs": (129, "both@ascii@objs" + W + "+track+props", None, RL.CACHE),
    "ascii-grav": (130, "both@ascii@grav" + W + "+track+grav", None, RL.CACHE),
    "ascii-frame": (131, "both@ascii@frame+rediscover+chain+wm+wmframe", None, RL.CACHE),
    "ascii-contain": (132, "both@ascii@contain" + W + "+track+contain", None, RL.CACHE),
    "ascii-count": (133, "both@ascii@count" + W + "+track+count", None, RL.CACHE),
    "ascii-succ": (134, "both@ascii@succ" + W + "+track+count+succ", None, RL.CACHE),
    "ascii-numrow": (136, "both@ascii@numrow" + W, "both@ascii@numrow+rediscover+chain", RL.CACHE),
    "ascii-numown": (136, "both@ascii@numown" + W + "+scale", None, RL.CACHE),
    "ascii-views": (137, "both@ascii@views" + W + "+views", "both@ascii@views+rediscover+chain", RL.CACHE),
    "ascii-viewpt": (138, "both@ascii@viewpt" + W + "+views+vturn", "both@ascii@viewpt+rediscover+chain", RL.CACHE),
    "ascii-goal": (139, "both@ascii@goal" + W + "+goal", "both@ascii@goal+rediscover+chain", RL.CACHE),
    "ascii-self": (140, "both@ascii@self" + W + "+self", "both@ascii@self+rediscover+chain",
                   os.path.join(os.path.dirname(os.path.abspath(__file__)), "self_budget_cache")),
}
BCACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "baseline3_cache")


def qt(f_):
    return 2 if f_ is True else 0 if f_ is False else int(f_)


def pick(sc, cx):
    sc = np.asarray(sc, dtype=float).copy()
    sc[cx] = -1
    j = int(np.argmax(sc))
    return j if sc[j] > 0 else -1


def life(world, seed, shuffle=False):
    w = RL.build_world(world, seed)
    rng = np.random.default_rng(70_000 + seed)
    V = int(w.V)
    order = [int(c) for phase in w.phases for c in rng.permutation(phase)]
    sessions = [order[i:i + PER_SESSION] for i in range(0, len(order), PER_SESSION)]
    r2 = np.random.default_rng(80_000 + seed)
    freq, co = np.zeros(V), np.zeros((V, V))
    seen, out = [], []
    for new in sessions:
        seen += new
        block = [[int(t) for t in s] for c in new for _, s, _ in w.categories[c]["heard"]]
        if shuffle:
            flat = r2.permutation([t for s in block for t in s]).tolist()
            nb, i = [], 0
            for s in block:
                nb.append(flat[i:i + len(s)]); i += len(s)
            block = nb
        for s in block:
            u = sorted(set(s))
            freq[u] += 1
            co[np.ix_(u, u)] += 1
        np.fill_diagonal(co, 0)
        norm = co / np.sqrt(np.outer(freq, freq) + 1e-12)
        scorer = {"M": lambda cx: freq, "C": lambda cx: co[cx].sum(0), "N": lambda cx: norm[cx].sum(0)}
        acc, voe = {}, {}
        for c in seen:
            cat = w.categories[c]
            for L in ACC_LISTS:
                for kind, s, ans in cat.get(L, []):
                    keys = [kind]
                    if world == "ascii-frame" and L == "chain":
                        f_ = cat.get("flipped", False)
                        keys += (["chain quarter"] if qt(f_) in (1, 3) else ["chain half"] if qt(f_) == 2 else [])
                        keys += (["chain flipped"] if f_ else [])
                    for k in keys:
                        acc.setdefault(k, []).append(([int(x) for x in s], int(ans[0])))
            for L in VOE_LISTS:
                for kind, s, ans in cat.get(L, []):
                    voe.setdefault(kind, []).append(([int(x) for x in s], int(ans[0]), int(ans[1])))
        rec = {}
        for b in ("M", "C", "N"):
            for k, items in acc.items():
                rec[(b, k)] = float(np.mean([pick(scorer[b](cx), cx) == t for cx, t in items]))
            for k, items in voe.items():
                v = []
                for cx, good, bad in items:
                    sc = scorer[b](cx)
                    v.append(1.0 if sc[bad] < sc[good] else 0.5 if sc[bad] == sc[good] else 0.0)
                rec[(b, k)] = float(np.mean(v))
        out.append(rec)
    return out, 1.0 / V


def job(args):
    world, seed, shuffle = args
    path = os.path.join(BCACHE, f"{world}_seed{seed}{'_shuf' if shuffle else ''}.pkl")
    if os.path.exists(path):
        with open(path, "rb") as f:
            return pickle.load(f)
    res = life(world, seed, shuffle)
    os.makedirs(BCACHE, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(res, f)
    return res


def voyager(cache, cond, seeds):
    have = [sd for sd in seeds if os.path.exists(os.path.join(cache, f"seed{sd}_{cond}.pkl"))]
    return have, [pickle.load(open(os.path.join(cache, f"seed{sd}_{cond}.pkl"), "rb")) for sd in have]


if __name__ == "__main__":
    n_seeds = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    workers = int(sys.argv[2]) if len(sys.argv) > 2 else 2
    only = sys.argv[3].split(",") if len(sys.argv) > 3 else list(WORLDS)
    seeds = list(range(n_seeds))
    tasks = [(wn, sd, sh) for wn in only for sd in seeds for sh in (False, True)]
    with Pool(workers) as p:
        res = dict(zip(tasks, p.map(job, tasks)))
    for wn in only:
        fnd, main, body, cache = WORLDS[wn]
        bl = [res[(wn, sd, False)][0] for sd in seeds]
        sh = [res[(wn, sd, True)][0] for sd in seeds]
        chance = res[(wn, 0, False)][1]
        keys = sorted({k for lf in bl for st in lf for (_, k) in st})
        print(f"=== finding {fnd}: world {wn}, {n_seeds} seeds; chance 1/V = {chance:.4f} ===", flush=True)
        hv, vy = voyager(cache, main, seeds)
        hb, vb = voyager(cache, body, seeds) if body else ([], [])
        g = lambda lives, b, k: np.array([[st.get((b, k), np.nan) for st in lf] for lf in lives])
        for k in keys:
            for b in ("M", "C", "N"):
                print(f"  {k:16} {b}      " + " ".join(f"{v:.3f}" for v in np.nanmean(g(bl, b, k), 0)))
            print(f"  {k:16} N/shuf " + " ".join(f"{v:.3f}" for v in np.nanmean(g(sh, 'N', k), 0)))
            best = max("MCN", key=lambda b: np.nanmean(g(bl, b, k)))
            for tag, have, lives in (("main", hv, vy), ("body", hb, vb)):
                if not have:
                    if tag == "main":
                        print(f"  {k:16} Voyager {tag}: no cache yet ({main})")
                    continue
                v = np.array([[st.get(k, np.nan) for st in lf] for lf in lives])
                if np.all(np.isnan(v)):
                    print(f"  {k:16} Voyager {tag}: key not in the cache")
                    continue
                print(f"  {k:16} Voyager {tag} ({len(have)} seeds) " + " ".join(f"{x:.3f}" for x in np.nanmean(v, 0)))
                bb = g(bl, best, k)[have]
                d = v[:, :bb.shape[1]] - bb
                print(f"  {k:16} Voyager {tag} - best baseline {best}: end {sgn(d[:, -1])} | life {sgn(np.nanmean(d, 1))}",
                      flush=True)
