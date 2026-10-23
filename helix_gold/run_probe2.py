"""Finding 100 (the late probe; finding 99's follow-up, the user's order: check first, then act). Record-only ("+probe2"):
every branch practised to the LAST rung, raw per-branch scores kept. Three causes, told apart before running:
  C1 too early     -- the eyes' rank correlation with the outside test is clearly higher here than at the first rung (99);
  C2 noisy eyes    -- an eye's own reliability (two draws / two halves of the held-out sentences) is low;
  C3 no lever      -- the outside test's OWN reliability over the branches (two halves of its items) is low: the knob
                      choices barely change the outcome, so no eye can rank them (the body gives the method layer no lever).
Also the average of the two imagination eyes (no constant), the fresh eye (new categories' held-out) vs the outside test on
the new categories, and the spread of the outside test over the branches. 10 seeds per world; behaviour unchanged (checked).
"""
import sys
from multiprocessing import Pool

import numpy as np

import run_long as RL

WORLDS = {"glyph": "both@ascii+fix-routeknobs", "text": "both+world+recomb-side-routeknobs"}


def rank(v):
    return np.argsort(np.argsort(v)).astype(float)


def rcorr(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    m = ~(np.isnan(a) | np.isnan(b))
    if m.sum() < 3:
        return np.nan
    a, b = rank(a[m]), rank(b[m])
    if a.std() == 0 or b.std() == 0:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])


def main():
    n_seeds = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    workers = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    extra = sys.argv[3] if len(sys.argv) > 3 else ""          # finding 101: "+attend" (the reference stays without it)
    worlds = {k: v for k, v in WORLDS.items() if len(sys.argv) <= 4 or k == sys.argv[4]}
    tasks = [(sd, c + extra + "+probe2") for c in worlds.values() for sd in range(n_seeds)]
    tasks += [(sd, c + extra) for c in worlds.values() for sd in range(n_seeds)]   # the references, in the pool too
    with Pool(workers) as pool:
        out = dict(zip(tasks, pool.map(RL.one, tasks, chunksize=1)))
    for name, c in worlds.items():
        res = [out[(sd, c + extra + "+probe2")] for sd in range(n_seeds)]
        refs = [out[(sd, c + extra)] for sd in range(n_seeds)]
        same = all(abs(a["never-heard"] - b["never-heard"]) < 1e-9 for r, q in zip(res, refs) for a, b in zip(r, q))
        print(f"\n=== {name} world, {n_seeds} seeds; record-only check: {same} ===")
        rows = {}
        for r in res:
            for i, st in enumerate(r):
                raw = st.get("probe2")
                if not raw:
                    continue
                avg = (np.array(raw["recomb"]) + np.array(raw["world"])) / 2
                vals = {
                    "C1 recomb ~ true": rcorr(raw["recomb"], raw["true"]),
                    "C1 world ~ true": rcorr(raw["world"], raw["true"]),
                    "C1 heard ~ true": rcorr(raw["heard"], raw["true"]),
                    "C1 avg(recomb,world) ~ true": rcorr(avg, raw["true"]),
                    "   fresh ~ true new": rcorr(raw["fresh"], raw["true new"]),
                    "C2 recomb reliability": rcorr(raw["recomb"], raw["recomb 2"]),
                    "C2 world reliability": rcorr(raw["world"], raw["world 2"]),
                    "C2 heard reliability": rcorr(raw["heard A"], raw["heard B"]),
                    "C3 outside test reliability": rcorr(raw["true A"], raw["true B"]),
                    "   outside spread (max-min)": float(np.max(raw["true"]) - np.min(raw["true"])),
                    "   new spread (max-min)": float(np.max(raw["true new"]) - np.min(raw["true new"])),
                }
                for k, v in vals.items():
                    rows.setdefault(k, {}).setdefault(i, []).append(v)
        sess = sorted({i for d in rows.values() for i in d})
        print(f"{'session':30} " + " ".join(f"{i + 1:6d}" for i in sess) + "   life")
        for k, d in rows.items():
            per = [np.nanmean(d.get(i, [np.nan])) for i in sess]
            print(f"{k:30} " + " ".join(f"{x:6.2f}" for x in per) + f"  {np.nanmean([x for v in d.values() for x in v]):+.2f}")


if __name__ == "__main__":
    main()
