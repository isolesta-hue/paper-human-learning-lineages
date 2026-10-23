"""D72 subject 2 + the new designed control (2026-10-18; D72_DESIGN.md; the user's decision Q14 "세가지 다 너의 추천대로
해보자." -- the control must be one a part actually acts on: a question chaining two new motion words; subject 2 before
10-20 16:00Z). Written and fixed BEFORE any life of this test (the world check check_d72b_world.out and the seed-0 smoke
read the worlds' counts and whether the world model speaks on the chain questions -- no outcome key).

The brain: the same frozen brain as subject 1 (D72 principle 10): "+rediscover+chain+wm+wmctx+wmlast" on the glyph world.
Conditions (seeds 0-9):
  NEG   = both@ascii@d72neg+rediscover+chain+wm+wmctx+wmlast     subject 2, a new grammar marker (negation): OUTSIDE
  CHAIN = both@ascii@d72chain+rediscover+chain+wm+wmctx+wmlast   the "verb" world + chain questions of two new words: INSIDE
  BASE  = both@ascii+rediscover+chain+wm+wmctx+wmlast            no subject (the lives cached by run_d72.py -- the world
                                                                 check shows the base world unchanged)
  VERB  = both@ascii@d72verb+rediscover+chain+wm+wmctx+wmlast    subject 1's control (cached; read only for R7)
Keys (run_long.d72_keys, unchanged): "d72 k{j}" times word j was said so far (NEG: negator j; CHAIN: new motion word j);
NEG "d72 neg {j}" (a motion word + a start never heard with it and that negator in the room + the negator -> the arrival
cell among four: stays (right), the motion word's own end, two other motion words' ends); CHAIN "d72 chain {a}{b}" (two
new words never heard together + a start -> the end among the ends of four pairs from that start). Chance 0.25. " body" =
the world model silent (diagnosis only, D67). A CHAIN point's k = min(k_a, k_b) (the books need both words).
Readings (pre-registered):
  R1 curves: every (seed, session, item key) point by k, buckets 0 / 1 / 2 / 3-4 / 5-8 / 9-16 / 17+; "reaches the line" =
     the first bucket with mean >= 0.80 over >= 10 points.
  R2 learned at all: per seed, (a) the END mean minus chance 0.25 and (b) the END mean minus the seed's own k = 0 mean
     (the starting point -- for NEG the right cell shares its x, y with the start in the sentence, so the body's prior may
     sit off chance before any negator is heard; a seed with no k = 0 point is left out of (b)) -- sign tests.
  R3 design reliance: per seed, the mean over points with 1 <= k <= 8, CHAIN - NEG -- sign test.
  R4 body alone at END (NEG, CHAIN) and, per seed, CHAIN whole - CHAIN body at END: the PREMISE of the control -- if the
     world model changes no CHAIN answer in any seed (whole == body everywhere), the control did not engage the part and no
     reliance verdict is written (subject 1's lesson, finding 162).
  R5 what the subject costs: NEG - BASE and CHAIN - BASE over every END key both carry except "d72 ...": keys decided
     (10:0, |mean| >= 0.02) down / up. Expected before any life (written in data_ascii): a negated moment enters the
     motion word's book as a move of zero -- motion keys may go down in NEG.
  R6 cost: CPU per life per condition (D51 side files).
  R7 same learning: CHAIN's every non-chain END key equals the cached VERB life's (the chain questions are records only) --
     the count of seeds that differ (0 expected).
  R8 abstaining (diagnosis; added before any life after the subject-1 diagnosis diag_d72cat.py -- its picture -> word
     items had all four candidates blocked by the kind gate, a tie the strict rule scores wrong): the share of items whose
     four candidates all tie, by k bucket, whole brain (" abstain" keys; " body abstain" for the body, not read). A score below chance
     is read together with it.
DECISION (written before; "decided" = sign test p < 0.005, i.e. 10:0):
  per subject: "배움" when R2 (a) AND (b) are decided above AND R1 reaches the line by k <= 16; "느린 배움" when (a) and
  (b) are decided but the line is not reached by 16; "아직" otherwise.
  reliance (only when R4's premise holds): "설계에 기댐" when R3 is decided positive with mean >= +0.10; "설계 밖이 더
  빠름" when decided negative with mean <= -0.10; else "차이 없음/판정 안 남".
  What it changes: the five-subject yardstick (D72) for the general learning mechanism that comes after (Q14 1) -- which
  kinds of rule the frozen brain learns from little experience; nothing in the brain is changed by it (principle 10).
What it assumes: the frozen brain; the negators are new word elements memorised at birth (criterion 0 gate,
check_d72b_elements.out); the chain questions' two words were never heard together; within-life curves.
    python3 run_d72b.py [N_SEEDS=10] [WORKERS=4]"""
import json
import os
import sys

import numpy as np

import run_long as RL
from run_d72 import BASE, BASE_PARTS, BUCKETS, PARTS, VERB, sign

NEG = "both@ascii@d72neg" + BASE_PARTS
CHAIN = "both@ascii@d72chain" + BASE_PARTS
CONDS = (NEG, CHAIN, BASE, VERB)


def flag_check():
    """the substring-collision check: exactly wm, wmctx, wmlast among the parts; each subject only in its condition."""
    bad = []
    for c in CONDS:
        on = {p for p in PARTS if f"+{p}+" in c + "+"}
        if on != {"wm", "wmctx", "wmlast"}:
            bad.append((c, sorted(on)))
        for tag, mine in (("@d72neg", NEG), ("@d72chain", CHAIN), ("@d72verb", VERB)):
            if (tag in c) != (c == mine):
                bad.append((c, tag))
        if "@d72cat" in c:
            bad.append((c, "@d72cat"))
    return bad


def deep_eq(a, b):
    """equality that counts NaN == NaN at any depth (a life's dict-valued keys hold NaN: plain == calls them different --
    the subject-1 diagnosis's first equality check failed on that alone)"""
    if isinstance(a, dict) and isinstance(b, dict):
        return set(a) == set(b) and all(deep_eq(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return len(a) == len(b) and all(deep_eq(x, y) for x, y in zip(a, b))
    if isinstance(a, float) and isinstance(b, float) and np.isnan(a) and np.isnan(b):
        return True
    return a == b


def item_k(s, key):
    """the point's k: NEG "d72 neg j" -> k_j; CHAIN "d72 chain ab" -> min(k_a, k_b)."""
    tail = key.split()[2]
    if key.startswith("d72 chain"):
        return min(s.get(f"d72 k{tail[0]}", np.nan), s.get(f"d72 k{tail[1]}", np.nan))
    return s.get(f"d72 k{tail}", np.nan)


def item_keys(s, kind):
    return sorted(k for k in s if k.startswith(f"d72 {kind} ") and not k.endswith((" body", " abstain")))


def points(lives, kind):
    out = []
    for sd, r in enumerate(lives):
        for s in r:
            for key in item_keys(s, kind):
                v = s[key]
                if not np.isnan(v):
                    out.append((sd, item_k(s, key), v))
    return out


def curve(pts):
    rows = []
    for lo, hi in BUCKETS:
        v = [x for _, k, x in pts if lo <= k <= hi]
        rows.append((lo, hi, float(np.mean(v)) if v else float("nan"), len(v)))
    reach = next((lo for lo, hi, m, n in rows if n >= 10 and m >= 0.80), None)
    return rows, reach


def end_mean(r, kind, tag=""):
    keys = item_keys(r[-1], kind)
    v = [r[-1].get(k + tag, np.nan) for k in keys]
    return float(np.nanmean(v)) if v else float("nan")


def k0_mean(r, kind):
    v = [x for _, k, x in points([r], kind) if k == 0]
    return float(np.mean(v)) if v else float("nan")


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
    print(f"=== D72 subject 2 (neg, outside) + control (chain, inside) vs base, {n_seeds} seeds (development test) ===")
    verdicts, mids = {}, {}
    label = lambda lo, hi: f"{lo}" if lo == hi else f"{lo}+" if hi > 100 else f"{lo}-{hi}"
    for name, c, kind in (("NEG", NEG, "neg"), ("CHAIN", CHAIN, "chain")):
        rows, rch = curve(points(R[c], kind))
        print(f"R1 {name}: " + " | ".join(f"k {label(lo, hi)}: {m:.3f} ({n})" for lo, hi, m, n in rows) + " | " +
              (f"reaches 0.80 at k >= {rch}" if rch is not None else "does not reach 0.80"))
        ma, pa_, na_, pa = sign([end_mean(r, kind) - 0.25 for r in R[c]])
        mb, pb_, nb_, pb = sign([end_mean(r, kind) - k0_mean(r, kind) for r in R[c]])
        print(f"R2 {name} END {ma + 0.25:.3f}: (a) - chance {ma:+.3f} ({pa_}:{na_}, p = {pa:.4f}); "
              f"(b) - own k = 0 {mb:+.3f} ({pb_}:{nb_}, p = {pb:.4f})")
        both = pa < 0.005 and ma > 0 and pb < 0.005 and mb > 0
        verdicts[name] = "배움" if both and rch is not None and rch <= 16 else "느린 배움" if both else "아직"
        body = [end_mean(r, kind, " body") for r in R[c]]
        print(f"R4 {name} body alone END: {np.nanmean(body):.3f} (whole brain {ma + 0.25:.3f})")
        mids[name] = [np.mean([x for _, k, x in points([r], kind) if 1 <= k <= 8] or [np.nan]) for r in R[c]]
    md, pd_, nd_, pdv = sign([end_mean(r, "chain") - end_mean(r, "chain", " body") for r in R[CHAIN]])
    premise = (pd_ + nd_) > 0
    print(f"R4 CHAIN whole - body at END: {md:+.3f} ({pd_}:{nd_}, p = {pdv:.4f}); the world model changes CHAIN answers: "
          f"{'yes' if premise else 'NO -- premise failed'}")
    m, pos, neg, p = sign([a - b for a, b in zip(mids["CHAIN"], mids["NEG"])])
    print(f"R3 CHAIN - NEG over 1 <= k <= 8: {m:+.3f} ({pos}:{neg}, p = {p:.4f})")
    reliance = ("판정 안 함(대조가 부품을 움직이지 않음)" if not premise else
                "설계에 기댐" if p < 0.005 and m >= 0.10 else "설계 밖이 더 빠름" if p < 0.005 and m <= -0.10 else
                "차이 없음/판정 안 남")
    for name, c in (("NEG", NEG), ("CHAIN", CHAIN)):
        keys = sorted(k for k in set(R[c][0][-1]) & set(R[BASE][0][-1]) if not k.startswith("d72")
                      and isinstance(R[c][0][-1][k], (int, float)))
        down, up = [], []
        for k in keys:
            mm, pp, nn, pv = sign([R[c][i][-1].get(k, np.nan) - R[BASE][i][-1].get(k, np.nan) for i in range(n_seeds)])
            if pv < 0.005 and abs(mm) >= 0.02:
                (up if mm > 0 else down).append((k, round(mm, 3)))
        print(f"R5 {name} - BASE over {len(keys)} END keys: decided down {len(down)} {down[:8]}, up {len(up)} {up[:8]}")
    differ = 0
    for i in range(n_seeds):
        a, b = R[CHAIN][i], R[VERB][i]
        differ += not (len(a) == len(b) and all(deep_eq({k: x.get(k) for k in y}, y) for x, y in zip(a, b)))
    print(f"R7 CHAIN = VERB on every key the VERB life has: seeds that differ {differ} (0 expected)")
    for name, c, kind in (("NEG", NEG, "neg"), ("CHAIN", CHAIN, "chain")):
        ab = [(item_k(s, k), s.get(k + " abstain", np.nan)) for r in R[c] for s in r for k in item_keys(s, kind)]
        line = []
        for lo, hi in BUCKETS:
            v = [a for kk, a in ab if lo <= kk <= hi and not np.isnan(a)]
            line.append(f"k {label(lo, hi)}: {np.mean(v):.3f}" if v else f"k {label(lo, hi)}: -")
        print(f"R8 {name} abstaining (all four tie): " + " | ".join(line))
    for c in (NEG, CHAIN):
        cpu = []
        for sd in seeds:
            f = os.path.join(RL.CACHE + "_cpu", f"seed{sd}_{c}.json")
            if os.path.exists(f):
                d = json.load(open(f))
                cpu.append((d.get("cpu_cum_s", [0])[-1] / 60, d.get("peak_rss_gb", np.nan)))
        if cpu:
            print(f"R6 {c}: CPU per life {np.mean([a for a, _ in cpu]):.1f} min, peak {np.nanmax([b for _, b in cpu]):.2f} GB")
    print(f"DECISION: NEG {verdicts['NEG']}, CHAIN {verdicts['CHAIN']}, reliance {reliance}")
    print("exit 0")
