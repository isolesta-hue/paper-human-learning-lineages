"""Ledger D1b, first step (2026-10-09): does Voyager's sense organ already carry an approximate number sense? Record only --
no learner, no life, no body part (Stoianov & Zorzi 2012: number sense emerged in a generative model of dot images).

Stimuli: 5 x 5 glyph pictures with k marks ("*") at random cells, the rest "." (k = 1..20), seen through Voyager's fixed random
retina (data_ascii.retina / see -- the same sense organ as the glyph world). A separate random stream; nothing in the worlds
changes. Per seed (10 seeds = 10 retinas): 200 training pictures per k -> one least-squares readout "how many" (features ->
k, with a bias; the simplest learned "more" axis, no constants); 200 fresh test pictures per k.
Pre-registered (written before any run):
  A1 scalar variability: per k, the readout's spread sd(k) and the coefficient of variation cv(k) = sd(k) / k. Weber-like
     (ratio) coding predicts cv roughly constant across k; difference coding predicts sd constant (cv falling with k).
  A2 "which is more" (two-alternative, test pictures, all pairs): accuracy for pairs of the same RATIO at different sizes
     (1:2 -> 3:6, 5:10, 8:16; 2:3 -> 4:6, 6:9, 10:15; 3:4 -> 6:8, 9:12, 12:16) and the same DIFFERENCE (4 -> 2:6, 6:10,
     12:16). Weber-like: same ratio ~ same accuracy, same difference falls with size. Reported as is, both ways.
Reading: either answer is informative -- if the senses already give a ratio code, D1b's body part is only the comparison;
if not, the approximate code has to come from somewhere else (Module A's memory, or a learned compression), to be tested next.
"""
import numpy as np
import data_ascii as DA

GL_DOT, GL_BG = DA.GLYPHS.index("*"), DA.GLYPHS.index(".")
KS = range(1, 21)
RATIO = {"1:2": [(3, 6), (5, 10), (8, 16)], "2:3": [(4, 6), (6, 9), (10, 15)], "3:4": [(6, 8), (9, 12), (12, 16)]}
DIFF4 = [(2, 6), (6, 10), (12, 16)]


def pics(rng, k, n, P):
    out = []
    for _ in range(n):
        s = np.full(DA.SPRITE * DA.SPRITE, GL_BG)
        s[rng.choice(DA.SPRITE * DA.SPRITE, k, replace=False)] = GL_DOT
        out.append(DA.see(s.reshape(DA.SPRITE, DA.SPRITE), P))
    return np.array(out)


def one(seed):
    rng = np.random.default_rng(seed * 7919 + 33_000_001)
    P = DA.retina(rng)
    tr = {k: pics(rng, k, 200, P) for k in KS}
    te = {k: pics(rng, k, 200, P) for k in KS}
    X = np.vstack([np.c_[tr[k], np.ones(200)] for k in KS])
    y = np.concatenate([np.full(200, k, float) for k in KS])
    w = np.linalg.lstsq(X, y, rcond=None)[0]
    pred = {k: np.c_[te[k], np.ones(200)] @ w for k in KS}
    sd = {k: pred[k].std() for k in KS}
    acc = lambda a, b: float((pred[b][:, None] > pred[a][None, :]).mean())
    return sd, {p: acc(*p) for ps in list(RATIO.values()) + [DIFF4] for p in ps}, {k: pred[k].mean() for k in KS}


if __name__ == "__main__":
    res = [one(s) for s in range(10)]
    print("=== D1b first step: approximate number in the retina (10 seeds; record only) ===")
    print("  A1  k   mean readout   sd     cv = sd / k")
    for k in KS:
        m = np.mean([r[2][k] for r in res]); s = np.mean([r[0][k] for r in res])
        print(f"      {k:2}  {m:6.2f}        {s:5.2f}  {s / k:5.3f}")
    print("  A2 which is more (accuracy, mean over seeds)")
    for name, ps in list(RATIO.items()) + [("diff 4", DIFF4)]:
        print(f"      {name:7} " + "  ".join(f"{a}:{b} {np.mean([r[1][(a, b)] for r in res]):.3f}" for a, b in ps))
