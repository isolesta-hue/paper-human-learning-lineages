"""Finding 118 (2026-10-05, from findings 115 and 117): does the world-shaped eye judge the whole-sentence memory too harshly
because its background words include other pairs' members?

Finding 115: late in the plain world the self-experiment drives the whole-sentence memory weight to ~0.3 and the lead is
lost; finding 117 ruled out the method layer's eye. The eye's question generator (world_items) swaps the background for
known words of the same category and kind -- pair members included -- whereas the outside test's background is partnerless
filler words; a pair member in the background makes the whole-sentence memory pull in a wrong sentence. Fix with what the
learner knows (+bgpool): only words with no strong partner in its own record (either way, P >= STRONG) may be background.
Record only (+whatif2: the what-if on the world-shaped questions and on the outside test, finding 114).

Conditions (10 seeds): both+rediscover+whatif2 (plain, cached), both+rediscover+whatif2+bgpool, and the same in the rung-1
world (@share). Pre-registered:
  G1 per world and part: the outside gain of the weight the eye chooses, sessions 2-16 and 9-16, old eye vs new eye
     (paired by seed), and each against 0.
  G2 rank correlation of the eye with the outside over the weights, by session.
  G3 the weight each eye chooses for the whole-sentence memory, sessions 9-16 (the old eye: ~0.3 in +tune lives).
"""
import sys
from multiprocessing import Pool

import numpy as np

import run_long as RL
from run_unified_continual import sign_p

WORLDS = {"plain": ("both+rediscover+whatif2", "both+rediscover+whatif2+bgpool"),
          "share": ("both+rediscover@share+whatif2", "both+rediscover@share+whatif2+bgpool")}


def rcorr(a, b):
    a, b = np.argsort(np.argsort(a)).astype(float), np.argsort(np.argsort(b)).astype(float)
    return np.nan if a.std() == 0 or b.std() == 0 else float(np.corrcoef(a, b)[0, 1])


def sgn(x):
    x = np.asarray(x)
    w, lo = int((x > 1e-12).sum()), int((x < -1e-12).sum())
    return f"{x.mean():+.4f} ({w}:{lo}, p={sign_p(w, lo):.4f})"


if __name__ == "__main__":
    n_seeds = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    workers = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    tasks = [(sd, c) for pr in WORLDS.values() for c in pr for sd in range(n_seeds)]
    with Pool(workers) as p:
        out = dict(zip(tasks, p.map(RL.one, tasks)))
    print(f"=== {n_seeds} seeds; the world-shaped eye with a partnerless background (record only) ===")
    for world, (old, new) in WORLDS.items():
        print(f"\n##### {world}")
        for part in ("chunk", "assoc"):
            gains, chosen = {}, {}
            for c in (old, new):
                g, ch, rc = [], [], []
                for sd in range(n_seeds):
                    l = out[(sd, c)]
                    gg, cc, rr = [], [], []
                    for i in range(1, len(l)):
                        best = max(RL.CF_W, key=lambda wt: l[i][f"cf world {part} {wt}"])
                        gg.append(l[i][f"cf out {part} {best}"] - l[i][f"cf out {part} 1.0"])
                        cc.append(best)
                        rr.append(rcorr([l[i][f"cf world {part} {wt}"] for wt in RL.CF_W],
                                        [l[i][f"cf out {part} {wt}"] for wt in RL.CF_W]))
                    g.append(gg); ch.append(cc); rc.append(rr)
                gains[c], chosen[c] = np.array(g), np.array(ch)
                tag = "new" if c == new else "old"
                print(f"  {part:5} {tag} eye: gain 2-16 {sgn(gains[c].mean(1))} | 9-16 {sgn(gains[c][:, 7:].mean(1))}"
                      f" | chosen w s9-16 {chosen[c][:, 7:].mean():.2f} | rank corr by session "
                      + " ".join(f"{v:+.2f}" for v in np.nanmean(np.array(rc), axis=0)))
            d = gains[new] - gains[old]
            print(f"  {part:5} G1 new - old: 2-16 {sgn(d.mean(1))} | 9-16 {sgn(d[:, 7:].mean(1))}")
