"""D80 v1.1 (2026-10-22; the user's decision "응, 너의 추천대로 진행해줘. 이번 아이디어의 효과가 예상보다 더 유의미하게 관측되는거
같네."). Written and fixed BEFORE any life of these seeds.

What changed (finding 171's reading): v1 read a picture's time line in the channel of the USUAL top answer, so for "where
now" questions whose usual top answer was a picture the store was never opened ("now" 0.589). In v1.1 the D79 equals sign
also learns, per question word set, the channel of the answers heard with it (run_long.EpisodeStore.learn; mode "EQ1" reads
the time line in that learned channel). Everything else is v1's: the "@seq" world, the episode store, the three time
relations, the Laplace belief > 1/2, answer time only, the usual kind gate (a value it does not allow is not given).
NEW seeds 10-19 (seeds 0-9 were seen in v1); the same condition both@ascii@seq+rediscover; the same body answers U, E, EQ
(v1's rules, reported as a replication) and EQ1.
Pre-registered:
  V1 now: EQ1 "seq now" (right after the room arrived, items pooled) >= 0.80 AND EQ1 - U sign test over seeds p < 0.005.
  V2 before: EQ1 "seq before" >= 0.80 AND EQ1 - U p < 0.005 (v1's EQ was 1.000 -- a replication on new seeds).
  V3 old keys at the last session (motion, names, scene): an EQ1 DROP = EQ1 - U <= -0.05 with p < 0.005.
  Verdict: "성공" = V1 and V2 and no V3 drop; otherwise "실패" (what failed is named).
  Report: v1's readings on the new seeds (E "now" >= 0.80?, EQ "before"), the learned meanings and channels per seed.
What it cannot show (as v1): moves told only in words (the exam's C8) -- v2; Voyager's own world only; the time relations
are designed, what is learned is a question's meaning and its answer channel.
    python3 run_d80b.py [first_seed=10] [n_seeds=10] [workers=3]
"""
import sys
from multiprocessing import Pool

import numpy as np

import run_long as RL
from run_count import sgn
from run_d80 import COND, last, pooled, test

RULES = ("U", "E", "EQ", "EQ1")

if __name__ == "__main__":
    first = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    n_seeds = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    workers = int(sys.argv[3]) if len(sys.argv) > 3 else 3
    seeds = list(range(first, first + n_seeds))
    with Pool(workers) as p:
        lives = p.map(RL.one, [(sd, COND) for sd in seeds], chunksize=1)
    if any(lf is None for lf in lives):
        print("PIECE DEADLINE: some lives not finished -- no table", flush=True)
        sys.exit(0)
    print(f"=== D80 v1.1 (the equals sign also learns the answer's channel): \"@seq\", seeds {seeds[0]}-{seeds[-1]}, {COND} ===")
    v = {(k, r): np.array([pooled(lf, k, r) for lf in lives]) for k in ("seq now", "seq before") for r in RULES}
    for k in ("seq now", "seq before"):
        print(f"  {k} (right after the room arrived): " + " / ".join(f"{r} {v[(k, r)].mean():.3f}" for r in RULES)
              + f"; EQ1 - U {sgn(v[(k, 'EQ1')] - v[(k, 'U')])}; EQ1 - EQ {sgn(v[(k, 'EQ1')] - v[(k, 'EQ')])}")
    rel = {}
    for lf in lives:
        for k_, x in lf[-1].items():
            if k_.startswith("seq rel "):
                rel.setdefault(k_[8:], []).append(("last", "before", "first")[int(x)])
    chans = {}
    for lf in lives:
        for k_, x in lf[-1].items():
            if k_.startswith("seq chan "):
                chans.setdefault(k_[9:], []).append(("where", "word", "picture")[int(x)])
    print("  learned answer channels at the last session: " +
          "; ".join(f"{k_}: {sorted(set(x))} ({len(x)} seeds)" for k_, x in sorted(chans.items())))
    print("  learned meanings at the last session: " +
          "; ".join(f"{k_}: {sorted(set(x))} ({len(x)} seeds)" for k_, x in sorted(rel.items())))
    print("  V3 old keys at the last session (U / EQ1):")
    drop = []
    for k in ("old motion", "old name", "old scene"):
        u = np.array([last(lf, k, "U") for lf in lives])
        x = np.array([last(lf, k, "EQ1") for lf in lives])
        m, p_ = test(x - u)
        d = m <= -0.05 and p_ < 0.005
        drop.append(d)
        print(f"    {k}: U {np.nanmean(u):.3f} | EQ1 {np.nanmean(x):.3f} ({sgn(x - u)}){' DROP' if d else ''}")
    m1, p1 = test(v[("seq now", "EQ1")] - v[("seq now", "U")])
    m2, p2 = test(v[("seq before", "EQ1")] - v[("seq before", "U")])
    v1 = v[("seq now", "EQ1")].mean() >= 0.80 and m1 > 0 and p1 < 0.005
    v2 = v[("seq before", "EQ1")].mean() >= 0.80 and m2 > 0 and p2 < 0.005
    failed = [n for n, ok in (("V1 now", v1), ("V2 before", v2), ("V3 old keys", not any(drop))) if not ok]
    print(f"  replication of v1 on new seeds: E now {v[('seq now', 'E')].mean():.3f}, EQ before {v[('seq before', 'EQ')].mean():.3f}")
    print(f"  V1 now EQ1 {v[('seq now', 'EQ1')].mean():.3f} (p = {p1:.4f}); V2 before EQ1 {v[('seq before', 'EQ1')].mean():.3f} "
          f"(p = {p2:.4f}); V3 drops {sum(drop)} -> verdict: {'성공' if not failed else '실패 (' + ', '.join(failed) + ')'}")
