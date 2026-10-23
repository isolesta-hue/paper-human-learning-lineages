"""D72 subject 3 (2026-10-18; D72_DESIGN.md; the user's decision Q15 "그럼 너의 추천대로 하자." -- subjects 3 and 4 before
the first edition, one after the other). Written and fixed BEFORE any life of this test (the world check
check_d72c_world.out and the seed-0 smoke read the world's counts only -- no outcome key).

The brain: the same frozen brain as subjects 1 and 2 (D72 principle 10): "+rediscover+chain+wm+wmctx+wmlast".
Conditions (seeds 0-9):
  ICE   = both@ascii@d72ice+rediscover+chain+wm+wmctx+wmlast     subject 3, a new causal rule (a slippery floor): OUTSIDE
  CHAIN = both@ascii@d72chain+rediscover+chain+wm+wmctx+wmlast   the designed control (cached by run_d72b.py, finding 164)
  BASE  = both@ascii+rediscover+chain+wm+wmctx+wmlast            no subject (cached by run_d72.py)
Keys (run_long.d72_keys): "d72 k{j}" times floor j was in a heard moment so far; "d72 ice {j}" (a motion word + a start
never heard with it and that floor in the room + the floor picture -> the arrival cell among four: twice the word's move
(right), the word's own end (its usual meaning), two other motion words' ends); " abstain" the share of items whose four
candidates all tie; " body" = the world model silent (diagnosis only, D67). Chance 0.25.
Readings (pre-registered; the same as run_d72b.py for its outside subject):
  R1 curve by k, buckets 0 / 1 / 2 / 3-4 / 5-8 / 9-16 / 17+; "reaches the line" = the first bucket with mean >= 0.80 over
     >= 10 points.
  R2 per seed (a) END mean - 0.25 and (b) END mean - the seed's own k = 0 mean (the body knows every motion word's usual
     move, so its prior may sit BELOW chance on the lure) -- sign tests.
  R3 design reliance: per seed, the mean over points with 1 <= k <= 8, CHAIN - ICE -- sign test (CHAIN's premise held in
     finding 164).
  R4 body alone at END (ICE).
  R5 what the subject costs: ICE - BASE over every END key both carry except "d72 ...": keys decided (10:0, |mean| >= 0.02)
     down / up. Expected before any life (written in data_ascii): a floor moment enters the motion word's book as a move of
     2d.
  R6 cost: CPU per life (D51 side files).
  R8 abstaining: the share of items whose four candidates all tie, by k bucket.
DECISION (written before; "decided" = sign test p < 0.005, i.e. 10:0): "배움" when R2 (a) AND (b) are decided above AND R1
  reaches the line by k <= 16; "느린 배움" when (a) and (b) are decided but the line is not reached by 16; "아직" otherwise.
  Reliance: "설계에 기댐" when R3 is decided positive with mean >= +0.10; "설계 밖이 더 빠름" when decided negative with mean
  <= -0.10; else "차이 없음/판정 안 남".
  What it changes: the five-subject yardstick (D72) for the general learning mechanism that comes after the five subjects
  (Q14 1); nothing in the brain is changed by it (principle 10).
What it assumes: the frozen brain; the floors are new drawn pictures memorised at birth (criterion 0 gate,
check_d72c_elements.out); within-life curves.
    python3 run_d72c.py [N_SEEDS=10] [WORKERS=4]"""
import json
import os
import sys

import numpy as np

import run_long as RL
from run_d72 import BASE, BASE_PARTS, BUCKETS, PARTS, sign
from run_d72b import CHAIN, curve, end_mean, item_k, item_keys, k0_mean, points

ICE = "both@ascii@d72ice" + BASE_PARTS
CONDS = (ICE, CHAIN, BASE)


def flag_check():
    bad = []
    for c in CONDS:
        on = {p for p in PARTS if f"+{p}+" in c + "+"}
        if on != {"wm", "wmctx", "wmlast"}:
            bad.append((c, sorted(on)))
        for tag, mine in (("@d72ice", ICE), ("@d72chain", CHAIN)):
            if (tag in c) != (c == mine):
                bad.append((c, tag))
        if any(t in c for t in ("@d72cat", "@d72verb", "@d72neg")):
            bad.append((c, "another subject"))
    return bad


if __name__ == "__main__":
    n_seeds = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    workers = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    bad = flag_check()
    if bad:
        sys.exit(f"FLAG CHECK FAILED: {bad}")
    seeds = list(range(n_seeds))
    tasks = [(sd, c) for c in CONDS for sd in seeds]
    out = dict(zip(tasks, RL.map_lives(tasks, workers)))
    R = {c: [out[(sd, c)] for sd in seeds] for c in CONDS}
    print(f"=== D72 subject 3 (ice, outside) vs control (chain, cached) and base (cached), {n_seeds} seeds (development test) ===")
    label = lambda lo, hi: f"{lo}" if lo == hi else f"{lo}+" if hi > 100 else f"{lo}-{hi}"
    rows, rch = curve(points(R[ICE], "ice"))
    print("R1 ICE: " + " | ".join(f"k {label(lo, hi)}: {m:.3f} ({n})" for lo, hi, m, n in rows) + " | " +
          (f"reaches 0.80 at k >= {rch}" if rch is not None else "does not reach 0.80"))
    ma, pa_, na_, pa = sign([end_mean(r, "ice") - 0.25 for r in R[ICE]])
    mb, pb_, nb_, pb = sign([end_mean(r, "ice") - k0_mean(r, "ice") for r in R[ICE]])
    print(f"R2 ICE END {ma + 0.25:.3f}: (a) - chance {ma:+.3f} ({pa_}:{na_}, p = {pa:.4f}); "
          f"(b) - own k = 0 {mb:+.3f} ({pb_}:{nb_}, p = {pb:.4f})")
    both = pa < 0.005 and ma > 0 and pb < 0.005 and mb > 0
    verdict = "배움" if both and rch is not None and rch <= 16 else "느린 배움" if both else "아직"
    print(f"R4 ICE body alone END: {np.nanmean([end_mean(r, 'ice', ' body') for r in R[ICE]]):.3f} (whole brain "
          f"{ma + 0.25:.3f})")

    def mid(r, kind):
        v = [x for _, k, x in points([r], kind) if 1 <= k <= 8]
        return np.mean(v) if v else np.nan
    m, pos, neg, p = sign([mid(rc, "chain") - mid(ri, "ice") for rc, ri in zip(R[CHAIN], R[ICE])])
    print(f"R3 CHAIN - ICE over 1 <= k <= 8: {m:+.3f} ({pos}:{neg}, p = {p:.4f})")
    reliance = ("설계에 기댐" if p < 0.005 and m >= 0.10 else "설계 밖이 더 빠름" if p < 0.005 and m <= -0.10 else
                "차이 없음/판정 안 남")
    keys = sorted(k for k in set(R[ICE][0][-1]) & set(R[BASE][0][-1]) if not k.startswith("d72")
                  and isinstance(R[ICE][0][-1][k], (int, float)))
    down, up = [], []
    for k in keys:
        mm, pp, nn, pv = sign([R[ICE][i][-1].get(k, np.nan) - R[BASE][i][-1].get(k, np.nan) for i in range(n_seeds)])
        if pv < 0.005 and abs(mm) >= 0.02:
            (up if mm > 0 else down).append((k, round(mm, 3)))
    print(f"R5 ICE - BASE over {len(keys)} END keys: decided down {len(down)} {down[:8]}, up {len(up)} {up[:8]}")
    ab = [(item_k(s, k), s.get(k + " abstain", np.nan)) for r in R[ICE] for s in r for k in item_keys(s, "ice")]
    print("R8 ICE abstaining (all four tie): " + " | ".join(
        (lambda v: f"k {label(lo, hi)}: {np.mean(v):.3f}" if v else f"k {label(lo, hi)}: -")(
            [a for kk, a in ab if lo <= kk <= hi and not np.isnan(a)]) for lo, hi in BUCKETS))
    cpu = []
    for sd in seeds:
        f = os.path.join(RL.CACHE + "_cpu", f"seed{sd}_{ICE}.json")
        if os.path.exists(f):
            d = json.load(open(f))
            cpu.append((d.get("cpu_cum_s", [0])[-1] / 60, d.get("peak_rss_gb", np.nan)))
    if cpu:
        print(f"R6 {ICE}: CPU per life {np.mean([a for a, _ in cpu]):.1f} min, peak {np.nanmax([b for _, b in cpu]):.2f} GB")
    print(f"DECISION: ICE {verdict}, reliance {reliance}")
    print("exit 0")
