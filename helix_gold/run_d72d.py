"""D72 subject 4 (2026-10-22; D72_DESIGN.md "과목 4", rule fixed 10-18 before the order device existed; the user's decision
"응, 너의 의견대로 (가)와 (나)를 순서대로 진행해줘."). Written and fixed BEFORE any life of this test (the world check
check_d72d_world.out and the seed-0 smoke read the world's counts only -- no outcome key).

The brain: the frozen brain of subjects 1-3 (D72 principle 10) WITH the order device D80 v1.1 (the episode store, answer
rule EQ1, answer time only -- the habits learn as before): "+rediscover+chain+wm+wmctx+wmlast+epi".
Conditions (seeds 0-9):
  NEXT  = both@ascii@d72next+rediscover+chain+wm+wmctx+wmlast+epi  subject 4, a new order rule: OUTSIDE
  CHAIN = both@ascii@d72chain+rediscover+chain+wm+wmctx+wmlast     the designed control (cached by run_d72b.py, finding 164)
  BASE  = both@ascii+rediscover+chain+wm+wmctx+wmlast              no subject (cached by run_d72.py)
The rule (data_ascii, d72="next"): a bell moment [BELL, picture, at(p)] + others is followed at once by [picture, from(p),
to(p + up)] + others, no motion word, up turned with the room; for every bell pair three bell-less pairs whose verbless move
goes to each other neighbour, so every neighbour is the verbless move's end equally often (check_d72d_world.out: 44 per
direction per seed) -- "up" is told only by the bell one moment before.
Keys (run_long.d72_keys): "d72 k{j}" times bell j was in a heard moment so far; "d72 next {j}" (the bell moment as heard, at a
cell never heard with that bell in the room -> to(?) among one up (right), stay, two other neighbours); " abstain" all four
tie; " body" = the world model silent (diagnosis only, D67); " eq1" = the same items answered with the order device. Chance
0.25.
Readings (pre-registered; R1-R8 as run_d72c.py):
  R1 curve by k, buckets 0 / 1 / 2 / 3-4 / 5-8 / 9-16 / 17+; "reaches the line" = the first bucket with mean >= 0.80 over
     >= 10 points.
  R2 per seed (a) END mean - 0.25 and (b) END mean - the seed's own k = 0 mean -- sign tests.
  R3 design reliance: per seed, the mean over points with 1 <= k <= 8, CHAIN - NEXT -- sign test.
  R4 body alone at END (NEXT).
  R5 what the subject costs: NEXT - BASE over every END key both carry except "d72 ...": keys decided (10:0, |mean| >= 0.02)
     down / up. Expected before any life: the verbless moves add from -> to pairs to every neighbour equally to the habits;
     the world model's books read only moments with a motion word, so none of them enters a book -- no decided key, or small
     drops in the motion keys.
  R6 cost: CPU per life (D51 side files).
  R8 abstaining by k bucket.
  R9 the order device: (session, item key) values where " eq1" differs from the usual answer, over all seeds. Expected 0 --
     the device speaks only for a question with exactly one picture whose question word set has a learned meaning, and the
     static check before any life found no such item in any D72 world (check_d72d_world.out: subject items hold the
     room's other pictures; no word set is ever learned -- subject 1's one-picture items carry no word).
DECISION (written before; "decided" = sign test p < 0.005, i.e. 10:0): as run_d72c.py -- "배움" when R2 (a) AND (b) are
  decided above AND R1 reaches the line by k <= 16; "느린 배움" when (a) and (b) are decided but the line is not reached by
  16; "아직" otherwise. Reliance: "설계에 기댐" when R3 is decided positive with mean >= +0.10; "설계 밖이 더 빠름" when decided
  negative with mean <= -0.10; else "차이 없음/판정 안 남".
  Expected before any life: "아직" -- the store keeps the order, but no part reads what follows a moment, and the body has no
  order (Q15). What a "배움" would mean: a path that is not order (a shortcut in this world) -- the world is then fixed
  before the subject is used again (principle 4).
Subjects 1-3 with the device brain (D72_DESIGN subject 4: "같은 뇌로 과목 1~3도 다시 잼"): NOT re-run -- by the static check the
  device cannot answer any of their items, so their stored curves are this brain's curves; R9 checks the same on subject 4
  inside the lives.
What it assumes: the frozen brain; the bells are new words memorised at birth (criterion 0 gate, check_d72d_elements.out);
within-life curves.
    python3 run_d72d.py [N_SEEDS=10] [WORKERS=3]"""
import json
import os
import sys

import numpy as np

import run_long as RL
from run_d72 import BASE, BASE_PARTS, BUCKETS, PARTS, sign
from run_d72b import CHAIN

NEXT = "both@ascii@d72next" + BASE_PARTS + "+epi"
CONDS = (NEXT, CHAIN, BASE)
TAGS = (" body", " abstain", " eq1", " eq1 abstain")


def flag_check():
    bad = []
    for c in CONDS:
        on = {p for p in PARTS if f"+{p}+" in c + "+"}
        if on != {"wm", "wmctx", "wmlast"}:
            bad.append((c, sorted(on)))
        for tag, mine in (("@d72next", NEXT), ("@d72chain", CHAIN), ("+epi", NEXT)):
            if (tag in c) != (c == mine):
                bad.append((c, tag))
        if any(t in c for t in ("@d72cat", "@d72verb", "@d72neg", "@d72ice")):
            bad.append((c, "another subject"))
    return bad


def item_k(s, key):
    tail = key.split()[2]
    if key.startswith("d72 chain"):
        return min(s.get(f"d72 k{tail[0]}", np.nan), s.get(f"d72 k{tail[1]}", np.nan))
    return s.get(f"d72 k{tail}", np.nan)


def item_keys(s, kind):
    return sorted(k for k in s if k.startswith(f"d72 {kind} ") and not k.endswith(TAGS))


def points(lives, kind):
    return [(sd, item_k(s, k), s[k]) for sd, r in enumerate(lives) for s in r for k in item_keys(s, kind)
            if not np.isnan(s[k])]


def curve(pts):
    rows = []
    for lo, hi in BUCKETS:
        v = [x for _, k, x in pts if lo <= k <= hi]
        rows.append((lo, hi, float(np.mean(v)) if v else float("nan"), len(v)))
    return rows, next((lo for lo, hi, m, n in rows if n >= 10 and m >= 0.80), None)


def end_mean(r, kind, tag=""):
    v = [r[-1].get(k + tag, np.nan) for k in item_keys(r[-1], kind)]
    return float(np.nanmean(v)) if v else float("nan")


def k0_mean(r, kind):
    v = [x for _, k, x in points([r], kind) if k == 0]
    return float(np.mean(v)) if v else float("nan")


if __name__ == "__main__":
    n_seeds = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    workers = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    bad = flag_check()
    if bad:
        sys.exit(f"FLAG CHECK FAILED: {bad}")
    seeds = list(range(n_seeds))
    tasks = [(sd, c) for c in CONDS for sd in seeds]
    res = RL.map_lives(tasks, workers)
    if any(r is None for r in res):
        print("PIECE DEADLINE: some lives not finished -- no table", flush=True)
        sys.exit(0)
    out = dict(zip(tasks, res))
    R = {c: [out[(sd, c)] for sd in seeds] for c in CONDS}
    print(f"=== D72 subject 4 (next, outside) vs control (chain, cached) and base (cached), {n_seeds} seeds ===")
    label = lambda lo, hi: f"{lo}" if lo == hi else f"{lo}+" if hi > 100 else f"{lo}-{hi}"
    rows, rch = curve(points(R[NEXT], "next"))
    print("R1 NEXT: " + " | ".join(f"k {label(lo, hi)}: {m:.3f} ({n})" for lo, hi, m, n in rows) + " | " +
          (f"reaches 0.80 at k >= {rch}" if rch is not None else "does not reach 0.80"))
    ma, pa_, na_, pa = sign([end_mean(r, "next") - 0.25 for r in R[NEXT]])
    mb, pb_, nb_, pb = sign([end_mean(r, "next") - k0_mean(r, "next") for r in R[NEXT]])
    print(f"R2 NEXT END {ma + 0.25:.3f}: (a) - chance {ma:+.3f} ({pa_}:{na_}, p = {pa:.4f}); "
          f"(b) - own k = 0 {mb:+.3f} ({pb_}:{nb_}, p = {pb:.4f})")
    both = pa < 0.005 and ma > 0 and pb < 0.005 and mb > 0
    verdict = "배움" if both and rch is not None and rch <= 16 else "느린 배움" if both else "아직"
    print(f"R4 NEXT body alone END: {np.nanmean([end_mean(r, 'next', ' body') for r in R[NEXT]]):.3f} (whole brain "
          f"{ma + 0.25:.3f})")

    def mid(r, kind):
        v = [x for _, k, x in points([r], kind) if 1 <= k <= 8]
        return np.mean(v) if v else np.nan
    m, pos, neg, p = sign([mid(rc, "chain") - mid(rn, "next") for rc, rn in zip(R[CHAIN], R[NEXT])])
    print(f"R3 CHAIN - NEXT over 1 <= k <= 8: {m:+.3f} ({pos}:{neg}, p = {p:.4f})")
    reliance = ("설계에 기댐" if p < 0.005 and m >= 0.10 else "설계 밖이 더 빠름" if p < 0.005 and m <= -0.10 else
                "차이 없음/판정 안 남")
    keys = sorted(k for k in set(R[NEXT][0][-1]) & set(R[BASE][0][-1]) if not k.startswith("d72")
                  and isinstance(R[NEXT][0][-1][k], (int, float)))
    down, up = [], []
    for k in keys:
        mm, pp, nn, pv = sign([R[NEXT][i][-1].get(k, np.nan) - R[BASE][i][-1].get(k, np.nan) for i in range(n_seeds)])
        if pv < 0.005 and abs(mm) >= 0.02:
            (up if mm > 0 else down).append((k, round(mm, 3)))
    print(f"R5 NEXT - BASE over {len(keys)} END keys: decided down {len(down)} {down[:8]}, up {len(up)} {up[:8]}")
    ab = [(item_k(s, k), s.get(k + " abstain", np.nan)) for r in R[NEXT] for s in r for k in item_keys(s, "next")]
    print("R8 NEXT abstaining (all four tie): " + " | ".join(
        (lambda v: f"k {label(lo, hi)}: {np.mean(v):.3f}" if v else f"k {label(lo, hi)}: -")(
            [a for kk, a in ab if lo <= kk <= hi and not np.isnan(a)]) for lo, hi in BUCKETS))
    pairs = [(s[k], s.get(k + " eq1", np.nan)) for r in R[NEXT] for s in r for k in item_keys(s, "next")]
    diff = sum(1 for a, b in pairs if not (a == b or (np.isnan(a) and np.isnan(b))))
    print(f"R9 the order device: values where EQ1 differs from the usual answer {diff} of {len(pairs)} (expected 0)")
    cpu = []
    for sd in seeds:
        f = os.path.join(RL.CACHE + "_cpu", f"seed{sd}_{NEXT}.json")
        if os.path.exists(f):
            d = json.load(open(f))
            cpu.append((d.get("cpu_cum_s", [0])[-1] / 60, d.get("peak_rss_gb", np.nan)))
    if cpu:
        print(f"R6 {NEXT}: CPU per life {np.mean([a for a, _ in cpu]):.1f} min, peak {np.nanmax([b for _, b in cpu]):.2f} GB")
    print(f"DECISION: NEXT {verdict}, reliance {reliance}")
    print("exit 0")
