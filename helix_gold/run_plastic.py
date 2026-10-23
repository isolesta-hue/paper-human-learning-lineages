"""Finding 105: Pioneer's idea in Voyager (the user, 2026-10-02) -- old memory changes slowly, new memory fast, and a steady
direction seeps into the old ("plastic": per word the habit row's step = 1 / (1 + age) x the optimizer's step, age = sessions
the word was heard in first-hand -- the running-mean step, no constant; full step for a row that moved the same way three
sessions in a row). As part of the body ("+plastic", fixed on) and, by the new principle, as a method-layer knob off at birth
("-plasticknob"). 30 seeds, cache_long_v2 (references reused).
Key comparisons, fixed before running:
  P1 (glyph body) content@ascii+fix+plastic vs content@ascii+fix, end and life; once-heard s13-16 (the late collapse);
  P2 (text body) content+world+recomb-side+plastic vs content+world+recomb-side (must not lose);
  P3 (glyph knob) both@ascii+fix-routeknobs+bundle-plasticknob vs both@ascii+fix-routeknobs+bundle, end and life;
  P4 (text knob) both+world+recomb-side-routeknobs+bundle-plasticknob vs both+world+recomb-side-routeknobs+bundle
  (the method layer carries the mode bundles from finding 104 on -- adopted with the user, 2026-10-02).
Also: the share of lives with the knob on, per session.
"""
import sys
from multiprocessing import Pool

import numpy as np

from run_unified_continual import sign_p
import run_long as RL

CONDS = ["content@ascii+fix", "content@ascii+fix+plastic", "both@ascii+fix-routeknobs+bundle", "both@ascii+fix-routeknobs+bundle-plasticknob",
         "content+world+recomb-side", "content+world+recomb-side+plastic", "both+world+recomb-side-routeknobs+bundle",
         "both+world+recomb-side-routeknobs+bundle-plasticknob"]


def main():
    n_seeds = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    workers = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    tasks = [(sd, c) for c in CONDS for sd in range(n_seeds)]
    with Pool(workers) as pool:
        out = dict(zip(tasks, pool.map(RL.one, tasks, chunksize=1)))
    res = [{c: out[(sd, c)] for c in CONDS} for sd in range(n_seeds)]
    n = len(res[0][CONDS[0]])

    def sgn(dd):
        n_pos, n_neg = int((dd > 1e-9).sum()), int((dd < -1e-9).sum())
        return f"{dd.mean():+.3f} ({n_pos}/{n_neg}, p={sign_p(n_pos, n_neg):.4f})"

    print(f"=== {n_seeds} seeds; old slow, new fast (Pioneer's idea) ===")
    for key in ["never-heard", "once all", "unseen all", "pairs all", "plastic"]:
        print(f"\n--- {key} ---")
        for c in CONDS:
            print(f"{c:48} " + " ".join(f"{np.nanmean([r[c][i].get(key, np.nan) for r in res]):6.3f}" for i in range(n)))

    def pair(g, b_, name, key="never-heard"):
        dl = np.array([np.nanmean([s[key] for s in r[g]]) - np.nanmean([s[key] for s in r[b_]]) for r in res])
        de = np.array([r[g][-1][key] - r[b_][-1][key] for r in res])
        late = np.array([np.mean([r[g][i][key] - r[b_][i][key] for i in range(12, 16)]) for r in res])
        print(f"{name} [{key}] {g} vs {b_}: life {sgn(dl)}  end {sgn(de)}  s13-16 {sgn(late)}")
    print()
    pair("content@ascii+fix+plastic", "content@ascii+fix", "P1")
    pair("content@ascii+fix+plastic", "content@ascii+fix", "P1", "once all")
    pair("content+world+recomb-side+plastic", "content+world+recomb-side", "P2")
    pair("both@ascii+fix-routeknobs+bundle-plasticknob", "both@ascii+fix-routeknobs+bundle", "P3")
    pair("both+world+recomb-side-routeknobs+bundle-plasticknob", "both+world+recomb-side-routeknobs+bundle", "P4")


if __name__ == "__main__":
    main()
