"""D72 subject 1 + the design-inside control (2026-10-18; D72_DESIGN.md; the user 10-18 "너의 추천대로 진행해줘" -- the
learning-power test runs before the first sealed edition, W7b after it). Written and fixed BEFORE any life of this test
(the world side's seed-0 check, smoke_d72.out, read the world's counts only, no outcome key).

What it decides (Q11): the direction of track B. The parts pass every criterion they were made for and the body alone is
near chance (SCORE_TABLE.md), so the score table cannot tell "knowledge the design gave" from "learned from little
experience" -- the baby compass (D71) asks for the second. A subject no part was made for, against one a part was made for,
read as learning curves inside one life: if the outside subject is not learned or is learned far slower, the next work is
a general learning mechanism (not another part); if it is learned at the inside pace, the current order stands. The curves
are also the yardstick ("아기 눈금") the four later subjects are read against, the same frozen brain throughout.

The brain (frozen for all five subjects, D72 principle 10): the parts that FIT the world (findings 159 and 161 -- the
self-trust gate is not a default) -- the glyph world has motion words and upside-down rooms, no Korean particles, no
multi-moment stories, no "describe" items, so FIT = wm + wmctx + wmlast on the base "+rediscover+chain" (run_w7pilot rule).
Conditions (seeds 0-9):
  CAT  = both@ascii@d72cat+rediscover+chain+wm+wmctx+wmlast    subject 1, a new category rule (shape families): OUTSIDE
  VERB = both@ascii@d72verb+rediscover+chain+wm+wmctx+wmlast   four new motion words: INSIDE (the world-model book's job)
  BASE = both@ascii+rediscover+chain+wm+wmctx+wmlast           no subject: what the subjects cost the rest
Keys (run_long.d72_keys, every session): "d72 k{j}" = times word j was said so far (densities: every room / every 2nd /
4th / 8th room, drawn per phase and per seed); CAT "d72 word {j}" (a NOVEL thing's picture -> its family's word, four-way)
and "d72 pic {j}" (word -> the novel picture of its family, four-way), asked from birth (k = 0 is the curve's first point);
VERB "d72 verb {j}" (a start never heard with that word in that room -> the arrival cell, four-way), in the rooms met.
Chance 0.25 everywhere. " body" = the same with every part silent (diagnosis only, D67).
Readings (pre-registered):
  R1 curves: every (seed, session, word) point by times heard k, buckets 0 / 1 / 2 / 3-4 / 5-8 / 9-16 / 17+: mean score and
     points; CAT word, CAT pic, CAT (their mean), VERB. "Reaches the line" = the first bucket with mean >= 0.80 (D72's
     line) over >= 10 points.
  R2 learned at all: per seed, the END mean over the four words (CAT: of word and pic) minus chance 0.25 -- sign test.
  R3 design reliance: per seed, the mean over points with 1 <= k <= 8, VERB - CAT -- sign test.
  R4 body alone at END (CAT, VERB): where the learning sits (diagnosis).
  R5 what the subject costs: CAT - BASE and VERB - BASE over every END key both carry except "d72 ...": keys decided
     (10:0, |mean| >= 0.02) down / up.
  R6 cost: CPU per life per condition (D51 side files).
DECISION (written before; "decided" = sign test p < 0.005, i.e. 10:0):
  per subject: "배움" when R2 is decided above chance AND R1 reaches the line by k <= 16; "느린 배움" when R2 is decided
  but the line is not reached by 16; "아직" otherwise.
  reliance: "설계에 기댐" when R3 is decided positive (VERB faster) with mean >= +0.10; "설계 밖이 더 빠름" when decided
  negative with mean <= -0.10; else "차이 없음/판정 안 남".
  -> CAT "아직" or "느린 배움" together with "설계에 기댐": track B's next work is a general learning mechanism (written as
  the finding, proposed to the user); otherwise the order stands. An undecided R3 with |mean| >= 0.10 goes to 30 seeds only
  if it alone changes that verdict (Q11). R5 keys decided down are written as the subject's interference (the curves are
  then read with that caveat).
What it assumes: the frozen brain above; the subject's words are new elements memorised at birth (criterion 0 gate passed
for both worlds, check_d72_elements.out); the CAT test pictures are novel (made for the tests only, in no sentence); one
word per density; within-life curves (a point's k is the world's count, not the learner's rehearsals).
    python3 run_d72.py [N_SEEDS=10] [WORKERS=4]"""
import json
import os
import sys

import numpy as np

import run_long as RL

BASE_PARTS = "+rediscover+chain+wm+wmctx+wmlast"
CAT = "both@ascii@d72cat" + BASE_PARTS
VERB = "both@ascii@d72verb" + BASE_PARTS
BASE = "both@ascii" + BASE_PARTS
CONDS = (CAT, VERB, BASE)
BUCKETS = [(0, 0), (1, 1), (2, 2), (3, 4), (5, 8), (9, 16), (17, 10 ** 6)]
PARTS = ["wm", "wmlast", "wmctx", "wmrev", "bind", "track", "props", "grav", "wmframe", "contain", "count", "succ", "scale",
         "views", "vturn", "goal", "self", "gate", "scene", "rehearse"]


def flag_check():
    """the substring-collision check: exactly wm, wmctx, wmlast among the parts; the subject only in its condition."""
    bad = []
    for c in CONDS:
        on = {p for p in PARTS if f"+{p}+" in c + "+"}
        if on != {"wm", "wmctx", "wmlast"}:
            bad.append((c, sorted(on)))
        if ("@d72cat" in c) != (c == CAT) or ("@d72verb" in c) != (c == VERB):
            bad.append((c, "subject"))
    return bad


def sign_p(n_pos, n_neg):
    from math import comb
    n = n_pos + n_neg
    if n == 0:
        return 1.0
    k = min(n_pos, n_neg)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)


def sign(d):
    d = np.asarray(d, float)
    d = d[~np.isnan(d)]
    pos, neg = int((d > 0).sum()), int((d < 0).sum())
    return float(np.mean(d)) if len(d) else float("nan"), pos, neg, sign_p(pos, neg)


def score(s, j, kind, tag=""):
    if kind == "cat":
        a, b = s.get(f"d72 word {j}{tag}", np.nan), s.get(f"d72 pic {j}{tag}", np.nan)
        return np.nanmean([a, b]) if not (np.isnan(a) and np.isnan(b)) else np.nan
    return s.get(f"d72 {kind} {j}{tag}", np.nan)


def points(lives, kind):
    """(seed, k, score) for every session and word with a score."""
    out = []
    for sd, r in enumerate(lives):
        for s in r:
            for j in range(4):
                v = score(s, j, kind)
                if not np.isnan(v):
                    out.append((sd, s.get(f"d72 k{j}", np.nan), v))
    return out


def curve(pts):
    rows = []
    for lo, hi in BUCKETS:
        v = [x for _, k, x in pts if lo <= k <= hi]
        rows.append((lo, hi, float(np.mean(v)) if v else float("nan"), len(v)))
    reach = next((lo for lo, hi, m, n in rows if n >= 10 and m >= 0.80), None)
    return rows, reach


def end_mean(r, kind, tag=""):
    return np.nanmean([score(r[-1], j, kind, tag) for j in range(4)])


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
    print(f"=== D72 subject 1 (cat, outside) + control (verb, inside) + base, {n_seeds} seeds (development test) ===")
    verdicts, reach = {}, {}
    for name, c, kinds in (("CAT", CAT, ("word", "pic", "cat")), ("VERB", VERB, ("verb",))):
        for kind in kinds:
            rows, rch = curve(points(R[c], kind))
            label = lambda lo, hi: f"{lo}" if lo == hi else f"{lo}+" if hi > 100 else f"{lo}-{hi}"
            line = " | ".join(f"k {label(lo, hi)}: {m:.3f} ({n})" for lo, hi, m, n in rows)
            print(f"R1 {name} {kind}: {line} | " + (f"reaches 0.80 at k >= {rch}" if rch is not None else
                                                     "does not reach 0.80"))
            if kind in ("cat", "verb"):
                reach[name] = rch
        kind = "cat" if name == "CAT" else "verb"
        m, pos, neg, p = sign([end_mean(r, kind) - 0.25 for r in R[c]])
        print(f"R2 {name} END mean - chance: {m:+.3f} ({pos}:{neg}, p = {p:.4f}); END {m + 0.25:.3f}")
        learned = p < 0.005 and m > 0
        verdicts[name] = ("배움" if learned and reach[name] is not None and reach[name] <= 16 else
                          "느린 배움" if learned else "아직")
        mb = np.nanmean([end_mean(r, kind, " body") for r in R[c]])
        print(f"R4 {name} body alone END: {mb:.3f} (whole brain {m + 0.25:.3f})")

    def mid(r, kind):
        v = [x for _, k, x in points([r], kind) if 1 <= k <= 8]
        return np.mean(v) if v else np.nan
    m, pos, neg, p = sign([mid(rv, "verb") - mid(rc, "cat") for rv, rc in zip(R[VERB], R[CAT])])
    print(f"R3 VERB - CAT over 1 <= k <= 8: {m:+.3f} ({pos}:{neg}, p = {p:.4f})")
    reliance = ("설계에 기댐" if p < 0.005 and m >= 0.10 else "설계 밖이 더 빠름" if p < 0.005 and m <= -0.10 else
                "차이 없음/판정 안 남")
    for name, c in (("CAT", CAT), ("VERB", VERB)):
        keys = sorted(k for k in set(R[c][0][-1]) & set(R[BASE][0][-1]) if not k.startswith("d72")
                      and isinstance(R[c][0][-1][k], (int, float)))
        down, up = [], []
        for k in keys:
            mm, pp, nn, pv = sign([R[c][i][-1].get(k, np.nan) - R[BASE][i][-1].get(k, np.nan) for i in range(n_seeds)])
            if pv < 0.005 and abs(mm) >= 0.02:
                (up if mm > 0 else down).append((k, round(mm, 3)))
        print(f"R5 {name} - BASE over {len(keys)} END keys: decided down {len(down)} {down[:8]}, up {len(up)} {up[:8]}")
    for c in CONDS:
        cpu = []
        for sd in seeds:
            f = os.path.join(RL.CACHE + "_cpu", f"seed{sd}_{c}.json")
            if os.path.exists(f):
                d = json.load(open(f))
                cpu.append((d.get("cpu_cum_s", [0])[-1] / 60, d.get("peak_rss_gb", np.nan)))
        if cpu:
            print(f"R6 {c}: CPU per life {np.mean([a for a, _ in cpu]):.1f} min, peak {np.nanmax([b for _, b in cpu]):.2f} GB")
    print(f"DECISION: CAT {verdicts['CAT']}, VERB {verdicts['VERB']}, reliance {reliance}")
    print("exit 0")
