"""Ledger D13, memory-first check before any design (2026-10-09): can Voyager's element memory (Module A, the same
memorize() as every life) hold the cells of a number line? A line of N cells (x = 0..N-1, centred) added to the glyph
world's elements; its features use the world's own per-axis position code (first half carries x with a fixed +-1 pattern,
second half the line's own constant), at STEP s; record: recall-and-recognise rate (data_ascii.element_check) of the line
cells and of all other elements (they must stay 100%). Variants (fixed before running): N = 5, 9; s = 0.2 (the world's
STEP; clipped to [0, 1] beyond 2 cells from the centre) and s = 0.1 (fits 9 cells unclipped). 5 seeds. Record only.
"""
import numpy as np, torch
import data_ascii as DA
from run_seed_transfer import memorize

def check(seed, n, s):
    w = DA.build(seed)
    rng = np.random.default_rng(seed * 104_729 + 44_000_003)
    xy = DA.WHERE_DIMS - DA.ROLE_DIMS
    su = rng.choice([-1, 1], xy // 2)
    line = []
    for i in range(n):
        x = i - (n - 1) / 2
        v = np.full(DA.RAW_DIM, 0.5)
        v[:xy // 2] = 0.5 + s * x * su
        v[xy // 2:xy] = 0.5 + 0.3 * rng.choice([-1, 1], xy // 2)        # "this is the number line" (a room-like constant)
        v[:xy] += rng.normal(0, DA.POS_NOISE, xy)
        line.append(np.clip(v, 0, 1))
    V0 = w.V
    w.vecs = np.vstack([w.vecs, np.array(line)])
    w.origin = list(w.origin) + ["line"] * n
    w.V = len(w.vecs)
    torch.manual_seed(seed)
    mem, _ = memorize(torch.tensor(w.vecs), seed)
    return DA.element_check(w, mem, seed)

if __name__ == "__main__":
    print("=== D13 memory-first: number-line cells in Voyager's element memory (5 seeds) ===")
    for n, s in ((5, 0.2), (9, 0.2), (9, 0.1)):
        res = [check(sd, n, s) for sd in range(5)]
        line = [r["line"] for r in res]
        others = [np.mean([v for k, v in r.items() if k not in ("line", "all")]) for r in res]
        print(f"  N={n} STEP={s}: line cells recognised {np.mean(line):.3f} (min {min(line):.3f}) | other elements {np.mean(others):.3f}")
