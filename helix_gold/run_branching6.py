"""New learning vs keeping the old: a judging logic, and splitting into a stable and a plastic copy (2026-09-25; the user:
people choose between leaning on what they know and quickly picking up the new, depending on the case -- both are used,
so a judging logic is needed; and AI can do what people cannot: think in parts).

Finding 67: every branching learner learned the newest categories 0.08-0.10 worse than fixed settings -- finding 50's judge
averages over categories, old ones outnumber new ones as the life goes on, so it favours keeping the old (more review,
lower learning rate). People shift between exploiting and exploring, and between stability and plasticity, with the
changeability of their world (Aston-Jones & Cohen 2005; Yu & Dayan 2005). AI: continual learners with a fast plastic
and a slow stable memory (fast and slow weights, Hinton & Plaut 1987; CLS-ER, Arani et al. 2022).
  sh              -- finding 67's successive halving (reference)
  sh-balanced     -- the judge counts new learning and keeping the old half each
  sh+split-age    -- the main learner stays as sh (stable); a plastic copy is forked from it at the start of every session
                     and learns the session with the start settings; questions about categories new this session go to
                     the plastic copy, the rest to the stable one (routing by a fixed rule)
  sh+split-judged -- the same, but routing per bucket (new / old categories) is decided by the record: in each session's
                     trial the plastic settings also run, and per bucket the plastic copy is used while it has beaten the
                     chosen branch on the held-out sentences at least as often as it lost (finding 61's stakes rule)
  fixed           -- reference
Platform as run_branching5.py. Seeds: 30 first; 100 if the key comparisons (vs sh, life) are undecided.
"""
import os
import pickle
import sys
from multiprocessing import Pool

import numpy as np
import torch

import data_v2 as D
from run_unified_continual import sign_p
import run_branching5 as B

CONDITIONS = ["fixed", "sh", "sh-balanced", "sh+split-age", "sh+split-judged"]


def life(cond, d, seed):
    B.JUDGE_MODE = "balanced" if cond == "sh-balanced" else "mean"
    base = "fixed" if cond == "fixed" else "sh"
    split = cond.split("split-")[1] if "split-" in cond else None
    return B.life(base, d, seed, split=split)


CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache_b6")


def one_seed(seed):
    path = os.path.join(CACHE, f"seed{seed}.pkl")
    if os.path.exists(path):
        with open(path, "rb") as f:
            return pickle.load(f)
    torch.set_num_threads(1)
    d = D.build("hard", seed)
    out = {c: life(c, d, seed) for c in CONDITIONS}
    with open(path, "wb") as f:
        pickle.dump(out, f)
    return out


def main():
    n_seeds = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    workers = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    os.makedirs(CACHE, exist_ok=True)
    with Pool(workers) as pool:
        results = pool.map(one_seed, range(n_seeds))
    n_steps = len(results[0][CONDITIONS[0]])
    n_sess = n_steps - B.CYCLES

    def m(vals):
        vals = np.array(vals, dtype=float)
        return f"{np.nanmean(vals):.3f}" if (~np.isnan(vals)).any() else "  -  "

    print(f"=== {n_seeds} seeds; hard level; five knobs; successive halving ===")
    for key in ["never-heard", "newest", "older", "review", "lr", "trial epochs", "plastic new", "plastic old"]:
        print(f"\n--- {key} ---")
        for c in CONDITIONS:
            print(f"{c:16} " + " ".join(f"{m([r[c][i][key] for r in results]):>7}" for i in range(n_steps)))
    print("\n--- paired ---")
    for g, base in [("sh", "fixed"), ("sh-balanced", "sh"), ("sh+split-age", "sh"), ("sh+split-judged", "sh"),
                    ("sh+split-judged", "sh+split-age"), ("sh+split-age", "fixed")]:
        cells = []
        for label, get in [("life", lambda r, gg: np.mean([st["never-heard"] for st in r[gg]])),
                           ("end", lambda r, gg: r[gg][-1]["never-heard"]),
                           ("newest", lambda r, gg: np.mean([st["newest"] for st in r[gg][1:n_sess]])),
                           ("older", lambda r, gg: np.nanmean([st["older"] for st in r[gg][1:n_sess]]))]:
            dd = np.array([get(r, g) - get(r, base) for r in results])
            n_pos, n_neg = int((dd > 1e-9).sum()), int((dd < -1e-9).sum())
            cells.append(f"{label} {dd.mean():+.3f} ({n_pos}/{n_neg}{'*' if sign_p(n_pos, n_neg) < 0.01 else ''})")
        print(f"{g:16} vs {base:13} " + "  ".join(cells))


if __name__ == "__main__":
    main()
