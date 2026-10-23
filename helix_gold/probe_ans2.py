"""Ledger D1b, short follow-ups to finding 135 (2026-10-09, the user's choice: very short runs, record, then fold D1b into
D13). Record only; no learner, no life. Pre-registered (written before any run):
  B  "which is more" learned only from heard comparisons (no counts): from N heard pairs (two dot pictures, k != m drawn
     from 1..20, the word says which side is more) the comparison axis = mean(features(more) - features(less)); tested on
     fresh pictures of the finding-135 pairs (same ratio / same difference). N = 6 and N = 24. Weber-like = same ratio ~
     same accuracy and same difference falling with size; reported both ways.
  C  9 x 9 pictures (81 cells, the resolution rung (5)), a retina built the same way (one-hot glyph cells -> fixed random
     projection -> sigmoid), k = 1..60: finding 135's A1 (spread, cv) and A2 with the pairs scaled x 4 / x 3 (1:2 -> 6:12,
     12:24, 24:48; 2:3 -> 8:12, 16:24, 32:48; 3:4 -> 9:12, 18:24, 36:48; difference 4 -> 4:8, 20:24, 44:48).
10 seeds each; separate random streams.
"""
import numpy as np
import data_ascii as DA
from probe_ans import RATIO, DIFF4, GL_DOT, GL_BG

def see_cells(cells, P):
    one_hot = np.zeros((cells.size, len(DA.GLYPHS)))
    one_hot[np.arange(cells.size), cells] = 1
    return 1 / (1 + np.exp(-(one_hot.ravel() @ P) * 2))

def pic(rng, k, n_cells, P):
    s = np.full(n_cells, GL_BG)
    s[rng.choice(n_cells, k, replace=False)] = GL_DOT
    return see_cells(s, P)

def retina(rng, n_cells):
    return rng.normal(0, 1 / np.sqrt(n_cells), (n_cells * len(DA.GLYPHS), DA.RAW_DIM))

def acc_pairs(score, rng, pairs, n_cells, P, n=200):
    out = {}
    for a, b in pairs:
        fa = np.array([score(pic(rng, a, n_cells, P)) for _ in range(n)])
        fb = np.array([score(pic(rng, b, n_cells, P)) for _ in range(n)])
        out[(a, b)] = float((fb[:, None] > fa[None, :]).mean())
    return out

def part_b(seed, n_heard):
    rng = np.random.default_rng(seed * 7919 + 33_000_101 + n_heard)
    P = retina(rng, 25)
    d = []
    for _ in range(n_heard):
        k, m = rng.choice(np.arange(1, 21), 2, replace=False)
        d.append(pic(rng, max(k, m), 25, P) - pic(rng, min(k, m), 25, P))
    w = np.mean(d, 0)
    return acc_pairs(lambda f: f @ w, rng, [p for ps in list(RATIO.values()) + [DIFF4] for p in ps], 25, P)

R9 = {"1:2": [(6, 12), (12, 24), (24, 48)], "2:3": [(8, 12), (16, 24), (32, 48)], "3:4": [(9, 12), (18, 24), (36, 48)]}
D9 = [(4, 8), (20, 24), (44, 48)]
K9 = [1, 2, 4, 6, 8, 12, 16, 24, 32, 40, 48, 56, 60]

def part_c(seed):
    rng = np.random.default_rng(seed * 7919 + 33_000_201)
    P = retina(rng, 81)
    ks = range(1, 61)
    tr = {k: np.array([pic(rng, k, 81, P) for _ in range(100)]) for k in ks}
    X = np.vstack([np.c_[tr[k], np.ones(100)] for k in ks]); y = np.concatenate([np.full(100, k, float) for k in ks])
    w = np.linalg.lstsq(X, y, rcond=None)[0]
    score = lambda f: np.r_[f, 1] @ w
    sd = {k: np.std([score(pic(rng, k, 81, P)) for _ in range(200)]) for k in K9}
    return sd, acc_pairs(score, rng, [p for ps in list(R9.values()) + [D9] for p in ps], 81, P)

if __name__ == "__main__":
    print("=== D1b short follow-ups (10 seeds; record only) ===")
    for n in (6, 24):
        res = [part_b(s, n) for s in range(10)]
        print(f"  B  heard comparisons N = {n}")
        for name, ps in list(RATIO.items()) + [("diff 4", DIFF4)]:
            print(f"      {name:7} " + "  ".join(f"{a}:{b} {np.mean([r[(a, b)] for r in res]):.3f}" for a, b in ps))
    res = [part_c(s) for s in range(10)]
    print("  C  9 x 9 retina, k: sd / cv")
    print("      " + "  ".join(f"{k}:{np.mean([r[0][k] for r in res]):.2f}/{np.mean([r[0][k] for r in res]) / k:.3f}" for k in K9))
    for name, ps in list(R9.items()) + [("diff 4", D9)]:
        print(f"      {name:7} " + "  ".join(f"{a}:{b} {np.mean([r[1][(a, b)] for r in res]):.3f}" for a, b in ps))
