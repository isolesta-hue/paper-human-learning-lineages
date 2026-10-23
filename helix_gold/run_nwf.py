"""Q14 frame test (2026-10-21): does a sentence frame never heard with names outvote the scene when a new name is asked inside
it? Written and fixed BEFORE the seed-0 smoke. Run first: the world check (check_nwf_world.py: every old world unchanged, 35
flags x seeds 0-2; the "@nwf" shapes) and the criterion-0 gate on "@nwf" (elements_10_nwf.out).

Why: finding 167's correction (10-21). In its own world Voyager learns new names across situations (finding 149: a naming
moment holds 1-3 other pictures; in a different sentence 0.26 -> 0.70 over 1 -> 8 hearings), but in the sealed exam v1 its name
items were 0.000-0.056 and joining the record into one unit did not help (finding 167). One difference: this world asks a name
in a frameless moment, the exam asks it inside a sentence frame ("there is ___"), and person names were never heard after
"there is" (17th sibling reading (a): 0 of 12,195). Pioneer's P55 reads its own 0.000 the same way. The guess: the frame
words' statistics outvote the scene. It decides how the user's Q14 requirement (2) is worded (finding 167's correction).

World "@nwf" (data_ascii nwf=True; check_nwf_world.py): the "@nw" world unchanged (the same rooms, names heard exactly 1 / 2 /
4 / 8 times, never in noise) plus, per room, 6 frame moments [THERE, IS, a common word, its picture, a place] + 1-3 of the
room's pictures; 8 common things shared by every room, drawn Zipf-like (p ~ 1 / rank). The frame is never heard with a room's
new names. Tests (the room's list "nwf"): every "@nw" item that gives the picture and asks the name (another place, fresh
background; 3 per object), bare ("nwb{n}") and the same item plus THERE, IS ("nwf{n}"); the frame's own use, 3 per room:
a common picture -> its word, bare ("nwbc") and framed ("nwfc"). Scored as finding 149 (whole vocabulary, top answer).
Keys per session (run_long.nwf_keys): counts for the rooms that arrived this session (" new" = right after exposure, each item
once in a life) and for all rooms met (" all").

Condition (one; 10 seeds first, 30 if undecided -- the two-stage rule): P = both@ascii@nwf+rediscover+chain+wm+wmlast (finding
149's P; P and the body alone tied on names there). No baseline: the question is within one life (the same items with and
without the frame), not whether the learner beats counting.

Measures per seed (items pooled over the sessions' " new" counts and over n = 1 / 2 / 4 / 8): B = bare right / items, F =
framed right / items, D = F - B. Per n the same (report).
Pre-registered:
  Q1 the verdict (all three on the pooled D):
     "지지 -- 틀이 장면을 이김": mean D <= -0.10 AND the seeds' sign test p < 0.005 AND, of the items the bare got right but the
         framed missed ("flips", pooled over seeds), >= 0.50 were answered with a common word or a frame word (the frame's
         own continuation won);
     "틀이 떨어뜨리지만 다른 길": mean D <= -0.10 AND p < 0.005 AND that share < 0.50 (the drop is not the frame's continuation
         -- e.g. dilution of the scene by two more words);
     "기각 -- 틀이 장면을 이기지 못함": mean D > -0.05;
     "미정" otherwise (-> 30 seeds).
  Q2 report: per n, B, F, D (sign test); flips b+f- vs b-f+ pooled (binomial sign test); the flips' answers (common / frame /
     another name / other) and all framed errors' answers; the room-names reading (right when the answer beats the room's
     three other new names -- the exam's candidate form) bare vs framed.
  Q3 check that the frame was learned (report): the frame's own use at the last session over all rooms, nwfc vs nwbc. If the
     body never learned "THERE, IS -> a common word" (nwfc not above nwbc and below 0.50), a "기각" is read as "this body did
     not learn the frame at all", not as "a learned frame does not outvote the scene".
What this assumes: a frame = two words heard only with common things (the exam's "there is" had 12,195 continuations, all
things); the bag-of-moments body has no word order, so the frame is its words, not their order; one frame, one kind of name;
Zipf 1/rank over 8 common things, 6 frame moments per room (186 per life); whole-vocabulary top answer, ties to the lower
index (as finding 149); flips classified by the framed top answer. What it does not test: the exam's converter, its many
persons per scene, its English and Korean frames, its picture features (conv-v1) -- a "지지" here says the mechanism exists in
this body, not that it is the whole of the exam's failure.
    python3 run_nwf.py [n_seeds=10] [workers=2] [first_seed=0]
"""
import sys
from multiprocessing import Pool

import numpy as np
from scipy import stats

import run_long as RL
from run_count import sgn

COND = "both@ascii@nwf+rediscover+chain+wm+wmlast"
NS = (1, 2, 4, 8)
KINDS = ("common", "frame", "name", "other")


def pooled(lf, key, tag=" new"):
    """sum over the sessions of a count key."""
    return float(sum(st.get(f"{key}{tag}", 0.0) for st in lf))


def per_seed(lf, k="nwfb"):
    n = pooled(lf, f"{k} n")
    b, f = pooled(lf, f"{k} bare right"), pooled(lf, f"{k} framed right")
    return (b / n, f / n, (f - b) / n) if n else (np.nan, np.nan, np.nan)


def binom_p(a, b):
    return float(stats.binomtest(int(a), int(a + b), 0.5).pvalue) if a + b else 1.0


if __name__ == "__main__":
    n_seeds = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    workers = int(sys.argv[2]) if len(sys.argv) > 2 else 2
    first = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    seeds = list(range(first, first + n_seeds))
    with Pool(workers) as p:
        lives = p.map(RL.one, [(sd, COND) for sd in seeds], chunksize=1)
    if any(lf is None for lf in lives):
        print("PIECE DEADLINE: some lives not finished -- no table", flush=True)
        sys.exit(0)
    print(f"=== Q14 frame test (\"@nwf\"): new names asked without / with a frame never heard with names, {len(seeds)} seeds "
          f"({seeds[0]}-{seeds[-1]}), {COND} ===")
    print("  B / F = bare / framed right, items pooled over the sessions right after exposure (each item once); D = F - B")
    rows = np.array([per_seed(lf) for lf in lives])
    for sd, (b, f, d) in zip(seeds, rows):
        print(f"  seed {sd:2}: B {b:.3f} F {f:.3f} D {d:+.3f}")
    B, F, D = rows[:, 0], rows[:, 1], rows[:, 2]
    w_, l_ = int((D > 1e-12).sum()), int((D < -1e-12).sum())
    p_seeds = binom_p(w_, l_)
    print(f"  pooled n: B {B.mean():.3f} F {F.mean():.3f} D {sgn(D)}")
    for n in NS:
        r = np.array([per_seed(lf, f"nwfb{n}") for lf in lives])
        print(f"  n={n}: B {r[:, 0].mean():.3f} F {r[:, 1].mean():.3f} D {sgn(r[:, 2])}")
    bp = sum(pooled(lf, "nwfb b+f-") for lf in lives)
    bm = sum(pooled(lf, "nwfb b-f+") for lf in lives)
    print(f"  flips pooled over seeds: bare right & framed wrong {bp:.0f}, the reverse {bm:.0f} (p = {binom_p(bp, bm):.4g})")
    fl = {k: sum(pooled(lf, f"nwfb flip {k}") for lf in lives) for k in KINDS}
    er = {k: sum(pooled(lf, f"nwfb framed err {k}") for lf in lives) for k in KINDS}
    share = (fl["common"] + fl["frame"]) / bp if bp else np.nan
    print("  the flips' framed answers: " + ", ".join(f"{k} {v:.0f}" for k, v in fl.items()) +
          f" -> common or frame word {share:.3f}")
    print("  all framed errors' answers: " + ", ".join(f"{k} {v:.0f}" for k, v in er.items()))
    n_all = sum(pooled(lf, "nwfb n") for lf in lives)
    print(f"  room-names reading (the answer beats the room's 3 other new names; chance 0.25): bare "
          f"{sum(pooled(lf, 'nwfb bare four') for lf in lives) / n_all:.3f}, framed "
          f"{sum(pooled(lf, 'nwfb framed four') for lf in lives) / n_all:.3f}")
    cb = [lf[-1].get("nwbc right all", np.nan) / lf[-1].get("nwbc n all", np.nan) for lf in lives]
    cf = [lf[-1].get("nwfc right all", np.nan) / lf[-1].get("nwfc n all", np.nan) for lf in lives]
    print(f"  Q3 the frame's own use at the last session (all rooms): bare common picture -> word {np.nanmean(cb):.3f}, "
          f"framed {np.nanmean(cf):.3f} (framed - bare {sgn(np.array(cf) - np.array(cb))})")
    frame_learned = not (np.nanmean(cf) <= np.nanmean(cb) and np.nanmean(cf) < 0.50)
    if D.mean() <= -0.10 and p_seeds < 0.005:
        verdict = "지지 -- 틀이 장면을 이김" if share >= 0.50 else "틀이 떨어뜨리지만 다른 길"
    elif D.mean() > -0.05:
        verdict = "기각 -- 틀이 장면을 이기지 못함" + ("" if frame_learned else " (이 몸은 틀 자체를 배우지 못함 -- Q3)")
    else:
        verdict = "미정 (-> 30 seeds)"
    print(f"  Q1 verdict: mean D {D.mean():+.3f}, seeds {w_}:{l_} p = {p_seeds:.4f}, flips to common / frame word "
          f"{share:.3f} -> {verdict}")
