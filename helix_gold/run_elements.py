"""Track A, W1 (2026-10-12; repo STAGE1_VOYAGER.md): stage-1 criterion 0 -- element memory, per subject world. Written and
fixed BEFORE running.

Criterion 0 (shared-curriculum STAGE1.md edition 2, user-approved 10-12): "element memory check (a gate for every subject)
>= 0.99"; a model that does not pass is marked "memory stage" for that subject (diagnosis order memory -> recall -> logic ->
structure). Voyager's element memory (Module A) is memorised from every element's features at birth, before any life, so the
check needs no life: build the world, build the learner exactly as run_long.life() does for the glyph-picture worlds
(Learner3(w, seed, content=True, where="linear")), and run data_ascii.element_check (every element recalled from memory through
the frozen decoder -- and the "where" channel through its own linear decoder -- and recognised: the nearest element to the
recalled features is the element itself), reported by element type (w.origin).

Worlds = every glyph-picture world a stage-1 subject lives in today: ascii, ascii-kor (criteria 1, 5), ascii-multi (8),
ascii-tale (8, D22), ascii-objs / ascii-grav / ascii-contain (6), ascii-count / ascii-succ / ascii-numrow / ascii-numown (3, 4),
ascii-views / ascii-viewpt (9, spatial), ascii-goal / ascii-self (D2), ascii-frame (6, frames).
Seeds 0-9 (first stage, as the 10-12 amendment). Reported per world: mean over seeds per element type, and the pooled share of
all elements with its Wilson 95% interval. Pre-registered reading: a world passes criterion 0 if the Wilson LOWER bound of the
pooled share >= 0.99 (STAGE1: judged by the lower end of the 95% interval); a type below 0.99 is named (memory stage for it).
Baselines / shuffle (rule 1) do not apply: this is a memory gate of the learner's own store, not an answer that a count could
give; the check is the same one finding 90 (B0) used.
What this assumes: element memory is fixed at birth (run_long.life() builds the learner the same way; nothing later rewrites
Module A -- the outer-layer checksum, memory_hash, guards it); the element list of each world as built today; nearest-element
recognition in raw feature space as the recall criterion (data_ascii.element_check, unchanged).
"""
import sys
from multiprocessing import Pool

import numpy as np
import torch

import run_long as RL
import data_ascii as A
from integrated import Learner3

WORLDS = ("ascii", "ascii-kor", "ascii-multi", "ascii-tale", "ascii-objs", "ascii-grav", "ascii-contain", "ascii-count",
          "ascii-succ", "ascii-numrow", "ascii-numown", "ascii-views", "ascii-viewpt", "ascii-goal", "ascii-self", "ascii-frame")


def job(args):
    world, seed = args
    torch.set_num_threads(1)
    w = RL.build_world(world, seed)
    me = Learner3(w, seed, content=True, where="linear")
    res = A.element_check(w, me.memory, seed, linear_pos=me.where_decoder if hasattr(me, "where_decoder") else None)
    return res, int(w.V)


def wilson(k, n, z=1.96):
    c = (k + z * z / 2) / (n + z * z)
    h = z * np.sqrt(k * (n - k) / n + z * z / 4) / (n + z * z)
    return c - h, c + h


if __name__ == "__main__":
    n_seeds = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    workers = int(sys.argv[2]) if len(sys.argv) > 2 else 2
    only = sys.argv[3].split(",") if len(sys.argv) > 3 else list(WORLDS)
    tasks = [(wn, sd) for wn in only for sd in range(n_seeds)]
    with Pool(workers) as p:
        out = dict(zip(tasks, p.map(job, tasks)))
    print(f"=== criterion 0, element memory per subject world ({n_seeds} seeds) ===")
    for wn in only:
        rs = [out[(wn, sd)] for sd in range(n_seeds)]
        types = sorted({k for r, _ in rs for k in r if k != "all"})
        k = sum(r["all"] * V for r, V in rs)
        n = sum(V for _, V in rs)
        lo, hi = wilson(round(k), n)
        below = [t for t in types if np.mean([r.get(t, np.nan) for r, _ in rs]) < 0.99]
        print(f"  {wn:14} all {k / n:.4f} (Wilson 95% [{lo:.4f}, {hi:.4f}]) -> {'PASS' if lo >= 0.99 else 'memory stage'}"
              + (f"; types below 0.99: {below}" if below else ""))
        print("      " + ", ".join(f"{t} {np.nanmean([r.get(t, np.nan) for r, _ in rs]):.3f}" for t in types), flush=True)
