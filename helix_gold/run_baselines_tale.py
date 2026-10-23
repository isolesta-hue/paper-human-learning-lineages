"""D22 rule-1 check (2026-10-13; CLAUDE.md "검토에서 더한 규칙" 1: trivial baselines and shuffle before claiming a part is needed).
Written AFTER tale_30.out was read -- so it asks no new question: it only checks the pre-registered T1-T3 claims against the
trivial baselines, with the machinery of Q5 part 3 (run_baselines3.py, copied in shape, run_baselines3.pick used as is).

No learner: the heard sentences of each session in the order of a Voyager life (rng 70_000 + seed, the phases' permutations,
PER_SESSION rooms a session), world "ascii-tale" (the tales -- marked and unmarked, and the v1b tales told in the session
before their room -- are in the rooms' heard lists exactly as Voyager hears them). Baselines M (the word heard most), C (sum of
co-occurrence counts with the item's context words), N (normalised counts); candidates = every word but the item's context
words, ties -> the lower index, a zero score is wrong. Shuffle: N on the session's words permuted across its sentences
(rng 80_000 + seed). Keys exactly as run_long.life computes them for "@tale": "tale in", "tale transfer" (room list "tale"),
"tale real clean / drop / turned" (grav floor + stack of the rooms by tale_drop / flipped), "tale preview marked / unmarked"
(grav floor + stack of a room c with pre_drop False / True while c - 1 is seen and c is not).
Reading: "the tale part is needed beyond trivial baselines" for T1 (preview unmarked, TA), T2 (preview marked, T1) and T3
(tale in, T1 and TA) only if part - the best baseline (highest mean over sessions) gives p < 0.005 (sign test over 30 seeds,
end where the key has an end, else life). Voyager's lives are read from the cache (run_tale.CONDS).
What this assumes: every heard sentence counts (Voyager sets a quarter aside -- the baselines get slightly more data); one
sentence = one co-occurrence window; the item's context is everything the learner is shown for it.
"""
import os
import sys
import pickle
from multiprocessing import Pool

import numpy as np

import run_long as RL
from run_schema import PER_SESSION
from run_count import sgn
from run_baselines3 import pick
import run_tale as T

BCACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "baseline_tale_cache")
KEYS = ("tale in", "tale transfer", "tale real clean", "tale real drop", "tale real turned", "tale preview marked",
        "tale preview unmarked")


def items_of(w, seen):
    out = {}
    for kt in ("in", "transfer"):
        out[f"tale {kt}"] = [(s_, ans[0]) for c in seen for k_, s_, ans in w.categories[c].get("tale", []) if k_ == f"tale {kt}"]
    groups = {"clean": [c for c in seen if w.categories[c].get("tale_drop") is False],
              "drop": [c for c in seen if w.categories[c].get("tale_drop") is True],
              "turned": [c for c in seen if w.categories[c].get("flipped") is True]}
    gr = lambda cs: [(s_, ans[0]) for c in cs for k_, s_, ans in w.categories[c].get("grav", []) if k_ in ("grav floor", "grav stack")]
    for g, cs in groups.items():
        out[f"tale real {g}"] = gr(cs)
    for g, dropv in (("marked", False), ("unmarked", True)):
        out[f"tale preview {g}"] = gr([c for c in range(1, w.n_cat)
                                       if w.categories[c].get("pre_drop") is dropv and c - 1 in seen and c not in seen])
    return out


def life(seed, shuffle=False):
    w = RL.build_world("ascii-tale", seed)
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
                nb.append(flat[i:i + len(s)])
                i += len(s)
            block = nb
        for s in block:
            u = sorted(set(s))
            freq[u] += 1
            co[np.ix_(u, u)] += 1
        np.fill_diagonal(co, 0)
        norm = co / np.sqrt(np.outer(freq, freq) + 1e-12)
        scorer = {"M": lambda cx: freq, "C": lambda cx: co[cx].sum(0), "N": lambda cx: norm[cx].sum(0)}
        its = items_of(w, seen)
        rec = {}
        for b in (("N",) if shuffle else ("M", "C", "N")):
            for k, items in its.items():
                rec[(b, k)] = float(np.mean([pick(scorer[b]([int(x) for x in cx]), [int(x) for x in cx]) == int(t)
                                             for cx, t in items])) if items else np.nan
        out.append(rec)
    return out


def job(args):
    seed, shuffle = args
    path = os.path.join(BCACHE, f"ascii-tale_seed{seed}{'_shuf' if shuffle else ''}.pkl")
    if os.path.exists(path):
        with open(path, "rb") as f:
            return pickle.load(f)
    res = life(seed, shuffle)
    os.makedirs(BCACHE, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(res, f)
    return res


if __name__ == "__main__":
    n_seeds = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    workers = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    seeds = list(range(n_seeds))
    tasks = [(sd, sh) for sd in seeds for sh in (False, True)]
    with Pool(workers) as p:
        res = dict(zip(tasks, p.map(job, tasks)))
    vy = {c: [pickle.load(open(os.path.join(RL.CACHE, f"seed{sd}_{T.CONDS[c]}.pkl"), "rb")) for sd in seeds] for c in T.CONDS}
    gb = lambda b, k, sh=False: np.array([[st.get((b, k), np.nan) for st in res[(sd, sh)]] for sd in seeds])
    gv = lambda c, k: np.array([[st.get(k, np.nan) for st in vy[c][sd]] for sd in range(n_seeds)])
    print(f"=== D22 rule-1 check: trivial baselines on the tale keys, {n_seeds} seeds ===")
    for k in KEYS:
        for b in ("M", "C", "N"):
            print(f"  {k:22} {b}      " + " ".join("  -  " if np.isnan(v) else f"{v:.3f}" for v in np.nanmean(gb(b, k), 0)))
        print(f"  {k:22} N/shuf " + " ".join("  -  " if np.isnan(v) else f"{v:.3f}" for v in np.nanmean(gb('N', k, True), 0)))
    for tag, k, c in (("T1", "tale preview unmarked", "TA"), ("T2", "tale preview marked", "T1"), ("T3", "tale in", "T1"),
                      ("T3", "tale in", "TA")):
        best = max("MCN", key=lambda b: np.nanmean(gb(b, k)))
        d = gv(c, k)[:, :gb(best, k).shape[1]] - gb(best, k)
        life_d = np.nanmean(d, 1)
        end_ok = not np.all(np.isnan(d[:, -1]))
        print(f"  {tag} {k}, {c} - best baseline {best} (baseline life mean {np.nanmean(gb(best, k)):.3f}): "
              + (f"end {sgn(d[:, -1])} | " if end_ok else "") + f"life {sgn(life_d)}")
