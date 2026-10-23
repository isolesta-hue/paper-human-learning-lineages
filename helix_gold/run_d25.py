"""D25 edition 1 (2026-10-13; repo D25_DESIGN.md, user-approved the same day with its four points): the parts' confident
answers moved into the body's practice through a filter and a temporary store ("sleep"), d25.py. Written and fixed BEFORE
the seed-0 smoke of any D25 condition. Decision record for a later check of the choices: ledger D43.

Store (d25.py): per room a fixed pool of up to 12 NEW two-verb questions made from the room's heard material (a thing of
the room, two motion words heard there, a start cell heard there, one background picture; never a chain test core); in a
world with tales the same questions with the tale marker in front ("tale" frame). After each session the world model
answers every pool question of the rooms met so far; a question it answers (it fires = it is sure) is a candidate; counters:
the same answer in how many sleeps in a row (repeat; also "still there after time"), the frame tag.
Conditions (+d25X on the base condition; X = the store's mode):
  N  the store sleeps and records, nothing transferred (the control; also checks that the store alone changes nothing);
  U  every candidate confirmed once, any frame, no usefulness check (unfiltered);
  F  fixed automatic: repeat >= 2, reality frame, and useful (the body alone gets it wrong now);
  K  F with the repeat threshold a method-layer knob (1 / 2 / 3, start 2);
  M  K plus the method layer's direct marks (keep now / forget; one-shot) -- the design's adopted option;
  E  as many extra practice items as U transfers, drawn from the session's own heard items (volume control).
Worlds: GAIN on "ascii", base B = both@ascii+rediscover+chain+wm+wmlast (finding 143/144's Voyager B): N U F K M E.
SAFETY on "@tale", base T = both@ascii@tale+rediscover+chain+wm+bind+wmlast+track+grav+tkA (finding 148's TA): N U F M.
10 seeds first (two-stage rule: decided if p < 0.005, else 30; the 10 seeds are reused).
Pre-registered:
  G1 gain, body alone (world model silent) on the chain questions ("d25 chain body"): M - N, F - N, K - N, U - N; end and life.
  G2 content vs volume: U - E on "d25 chain body"; end and life.
  G3 cost (report, a sign test each): M - N on never-heard (combined) and "d25 never-heard body", "d25 once body",
     "d25 unseen body"; the transferred counts.
  G4 the store alone: N vs the cached B lives (same seeds) on every key B has -- max |difference| (expected 0).
  S1 safety, @tale, body alone: "grav body unturned", "tale preview marked body", "tale preview unmarked body":
     U - N (expected harm if tale-frame answers leak), M - N and F - N (expected none); end / life sign tests; and
     "d25 transferred tale" (U only should be > 0).
  S2 combined answers on @tale ("tale in", "tale real clean", "tale preview unmarked"): M - N max |difference| (report).
Rule 1: the chain questions' trivial baselines are 0.000 (finding 145, part 3) -- a body-alone gain is above them if > 0.
What this assumes: a candidate = a two-verb question the world model rolls out (its only general rollout); "sure" = it fires;
12 questions a room; repeat = the same answer in consecutive sleeps; useful = the body alone wrong at selection; a transferred
question stays in the practice for the life with weight 1, its answer fixed at transfer; E's volume follows U's count;
marks are one-shot; the knob ladder 1 / 2 / 3; if the method layer switches its self-made side part on (finding 85), the
transferred questions are practised there like every self-made item.
"""
import os
import sys
import pickle
from multiprocessing import Pool

import numpy as np

import run_long as RL
from run_count import sgn

B = "both@ascii+rediscover+chain+wm+wmlast"
T = "both@ascii@tale+rediscover+chain+wm+bind+wmlast+track+grav+tkA"
GAIN = {m: B + f"+d25{m}" for m in "NUFKME"}
SAFE = {m: T + f"+d25{m}" for m in "NUFM"}

if __name__ == "__main__":
    n_seeds = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    workers = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    seeds = list(range(n_seeds))
    tasks = [(sd, c) for c in list(GAIN.values()) + list(SAFE.values()) for sd in seeds]
    with Pool(workers) as p:
        out = dict(zip(tasks, p.map(RL.one, tasks)))
    g = lambda c, k: np.array([[st.get(k, np.nan) for st in out[(sd, c)]] for sd in seeds])
    print(f"=== D25 edition 1: parts -> body through a filtered store, {n_seeds} seeds ===")
    for k in ("d25 chain body", "chain", "never-heard", "d25 never-heard body", "d25 once body", "d25 unseen body",
              "d25 candidates", "d25 transferred", "d25 k", "d25 mark keep", "d25 mark forget"):
        for m, c in GAIN.items():
            print(f"  {k:21} {m} " + " ".join("  -  " if np.isnan(v) else f"{v:.3f}" for v in np.nanmean(g(c, k), 0)))
    for tag, a, b in (("G1", "M", "N"), ("G1", "F", "N"), ("G1", "K", "N"), ("G1", "U", "N"), ("G2", "U", "E")):
        d = g(GAIN[a], "d25 chain body") - g(GAIN[b], "d25 chain body")
        print(f"  {tag} chain body, {a} - {b}: end {sgn(d[:, -1])} | life {sgn(np.nanmean(d, 1))}")
    for k in ("never-heard", "d25 never-heard body", "d25 once body", "d25 unseen body"):
        d = g(GAIN["M"], k) - g(GAIN["N"], k)
        print(f"  G3 {k}, M - N: end {sgn(d[:, -1])} | life {sgn(np.nanmean(d, 1))}")
    old = [pickle.load(open(os.path.join(RL.CACHE, f"seed{sd}_{B}.pkl"), "rb")) for sd in seeds]
    diffs = [abs(a[k] - b[k]) for o, sd in zip(old, seeds) for a, b in zip(o, out[(sd, GAIN["N"])]) for k in a
             if k in b and isinstance(a[k], float) and not np.isnan(a[k])]
    print(f"  G4 the store alone, N vs cached B: max |diff| {max(diffs) if diffs else float('nan'):.2e}")
    for k in ("grav body unturned", "tale preview marked body", "tale preview unmarked body", "d25 chain body",
              "d25 transferred", "d25 transferred tale", "tale in", "tale real clean", "tale preview unmarked"):
        for m, c in SAFE.items():
            print(f"  {k:26} {m} " + " ".join("  -  " if np.isnan(v) else f"{v:.3f}" for v in np.nanmean(g(c, k), 0)))
    for k in ("grav body unturned", "tale preview marked body", "tale preview unmarked body"):
        for a in "UMF":
            d = g(SAFE[a], k) - g(SAFE["N"], k)
            end = f"end {sgn(d[:, -1])} | " if not np.all(np.isnan(d[:, -1])) else ""
            print(f"  S1 {k}, {a} - N: {end}life {sgn(np.nanmean(d, 1))}")
    for k in ("tale in", "tale real clean", "tale preview unmarked"):
        d = g(SAFE["M"], k) - g(SAFE["N"], k)
        print(f"  S2 {k}, M - N: max |diff| {np.nanmax(np.abs(d)) if not np.all(np.isnan(d)) else float('nan'):.3f}")
