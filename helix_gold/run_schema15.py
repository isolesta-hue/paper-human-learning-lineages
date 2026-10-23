"""Problem templates, first experiment: several question formats (2026-09-25; the user's proposal -- before applying a
template to a question, recognise the pattern of the question itself; learn that "a question made by blanking a word out
of a sentence" exists, and later use that format itself).

Until now there was one format (hide a word, produce it), built into the code by the developer. Here the same knowledge
(habit, templates) is asked in three formats, made from the same test items:
  blank      -- produce the hidden word (as before)
  options    -- the answer + 3 other known words of its category: pick one (recognition -- easier than recall for people)
  candidate  -- one word, the answer half of the time: yes or no? (a decision: how sure must I be to say yes?)
Procedures the system has: F = the best word of all, C = the best of the given options, V_r = "yes" iff the candidate
ranks in the top r (r = 1, 2, 3, 5, 10). Learners (all masked, finding 57):
  fixed            -- one habit for every question: F everywhere (V1 for yes/no) -- a student who answers every question as
                      a fill-in
  dev              -- the developer's hand-made dispatch: blank -> F, options -> C, candidate -> V1
  learned          -- every session the teacher gives practice questions of all formats on the new sentences just heard
                      (before practice); the system tries every procedure on each (cheap, finding 58) and gives each question
                      SHAPE the procedure right most often so far. It is not told the formats' names; it learns which
                      procedure a shape calls for, including how sure to be before saying yes (r).
  learned+practice -- the same, and it USES the options format itself: it makes its own options questions from what it
                      heard (answer + 3 known words of the category) and practises picking among them (a contrastive loss;
                      retrieval practice with competitive lures helps people, Little et al. 2012).
World: schema. Seeds: two stages (CLAUDE.md). Key comparisons fixed before looking: learned vs dev (options, candidate, end),
learned+practice vs learned (life).
"""
import os
import pickle
import sys
from multiprocessing import Pool

import numpy as np
import torch

from run_unified_continual import sign_p
import run_schema3 as S3

WORLDS8 = ["schema"]
LEARNERS = ["fixed", "dev", "learned", "learned+practice"]
CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache15")
KEY = [("learned+practice", "learned")]


def life(world_name, learner, seed):
    return S3.life(world_name, "norm+explain", seed, evidence=True, independent=True, mask_context=True, formats=learner)


def one_seed(seed):
    path = os.path.join(CACHE, f"seed{seed}.pkl")
    if os.path.exists(path):
        with open(path, "rb") as f:
            return pickle.load(f)
    torch.set_num_threads(1)
    out = {(wn, ln): life(wn, ln, seed) for wn in WORLDS8 for ln in LEARNERS}
    with open(path, "wb") as f:
        pickle.dump(out, f)
    return out


def life_mean(r, g):
    return np.mean([st["never-heard"] for st in r[g]])


def decided(results, ga, gb):
    dd = np.array([life_mean(r, ga) - life_mean(r, gb) for r in results])
    n_pos, n_neg = int((dd > 1e-9).sum()), int((dd < -1e-9).sum())
    half = 2.756 * dd.std(ddof=1) / np.sqrt(len(dd))          # 99% t interval, 29 degrees of freedom
    return sign_p(n_pos, n_neg) < 0.005 or (dd.mean() - half > -0.002 and dd.mean() + half < 0.002)


def main():
    n_seeds = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    workers = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    os.makedirs(CACHE, exist_ok=True)
    with Pool(workers) as pool:
        results = pool.map(one_seed, range(n_seeds))
    n_sess = len(results[0][(WORLDS8[0], LEARNERS[0])])
    print(f"=== {n_seeds} seeds; data_v3 schema / random / local ===")
    for key, label in [("curve once", "newest ONE-SHOT pairs, epoch 0"), ("curve unseen", "newest UNSEEN pairs, epoch 0")]:
        print(f"\n--- {label} (sessions 1..{n_sess}) ---")
        for wn in WORLDS8:
            for ln in LEARNERS:
                print(f"{wn:6} {ln:17} " + " ".join(f"{np.nanmean([r[(wn, ln)][i][key][0] for r in results]):.3f}" for i in range(n_sess)))
    for key in ["never-heard", "once all", "fmt blank", "fmt options", "fmt candidate", "verify r"]:
        print(f"\n--- {key} (sessions 1..{n_sess}) ---")
        for wn in WORLDS8:
            for ln in LEARNERS:
                row = [np.nanmean([r[(wn, ln)][i][key] for r in results]) for i in range(n_sess)]
                if not all(np.isnan(row)):
                    print(f"{wn:6} {ln:17} " + " ".join(f"{v:6.3f}" for v in row))

    print("\n--- paired ---")
    def cmp(ga, gb, label):
        cells = []
        for lab, get in [("life", life_mean),
                         ("end", lambda r, g: r[g][-1]["never-heard"]),
                         ("pairs end", lambda r, g: r[g][-1]["pairs all"]),
                         ("once end", lambda r, g: r[g][-1]["once all"]),
                         ("unseen end", lambda r, g: r[g][-1]["unseen all"]),
                         ("blank end", lambda r, g: r[g][-1]["fmt blank"]),
                         ("options end", lambda r, g: r[g][-1]["fmt options"]),
                         ("yes/no end", lambda r, g: r[g][-1]["fmt candidate"]),
                         ("yes/no life", lambda r, g: np.mean([st["fmt candidate"] for st in r[g]]))]:
            dd = np.array([get(r, ga) - get(r, gb) for r in results])
            n_pos, n_neg = int((dd > 1e-9).sum()), int((dd < -1e-9).sum())
            cells.append(f"{lab} {dd.mean():+.3f} ({n_pos}/{n_neg}{'*' if sign_p(n_pos, n_neg) < 0.01 else ''})")
        print(f"{label:28} " + "  ".join(cells))
    for wn in WORLDS8:
        cmp((wn, "dev"), (wn, "fixed"), f"{wn}: dev vs fixed")
        cmp((wn, "learned"), (wn, "dev"), f"{wn}: learned vs dev")
        cmp((wn, "learned+practice"), (wn, "learned"), f"{wn}: +practice vs learned")
    if n_seeds < 100:
        und = [f"{wn}: {a} vs {b}" for wn in WORLDS8 for a, b in KEY if not decided(results, (wn, a), (wn, b))]
        print("\nstage rule: " + ("all key comparisons decided -- stop" if not und else "undecided -> extend to 100: " + "; ".join(und)))


if __name__ == "__main__":
    main()
