"""Finding 121 (2026-10-07, the user's order step 2 -- the eye-disagreement alarm -- and the user's ideas of 2026-10-06: the
difference between same-role variants is itself information; tag every record and learn which tags predict outcomes).
Analysis only, on cached lives -- no new life is run.

Finding 119: one eye grading another inherits the grader's bias (the held-out eye agreed with the outside on the connection
table only 17% of the time). What the learner CAN see without the outside is whether its own eyes agree. Question: does
agreement between eyes tell when a choice can be trusted -- is it a usable tag?

Lives: the record-only what-if lives (the learner itself is identical with or without the records): plain and rung-1 worlds,
10 seeds -- both+rediscover(+@share)+whatif2 (eyes S = held-out own sentences, W = world-shaped questions) and the same
seeds with +bgpool (eye B = world-shaped with a partnerless background, finding 118; same learner, so the three eyes judge
the same learner at the same session); glyph world (S and W only). Outside: the same what-if on the outside test (cf out).

Per session 2-16 and part (whole-sentence memory "chunk", connection table "assoc"): each eye's choice = the weight in
CF_W with its best score (ties: 1, the current weight); its direction = down / same / up relative to 1. The outside gain
of a weight w = out(w) - out(1).
Pre-registered:
  A1 agreement rates (same direction) per pair of eyes, per part and world.
  A2 the outside gain of W's choice when S agrees with W vs when it disagrees (and the same with B), and the share of
     sessions where the gain is >= 0.
  A3 offline policies (single-session counterfactual gain, mean over sessions 2-16 and 9-16, sign test over seeds):
     (i) follow W (as +tune), (ii) follow W only if S agrees in direction, else 1, (iii) the same with B,
     (iv) majority: the direction two or more eyes share (W's weight if W is in it, else the other eye's), else 1.
     Compared to (i). Offline = at a fixed learner, one session at a time -- a first look at whether agreement is worth
     a tag, not the full dynamic effect.
"""
import pickle

import numpy as np

import run_long as RL
from run_unified_continual import sign_p

CF = RL.CF_W
WORLDS = {"plain": ("both+rediscover+whatif2", "both+rediscover+whatif2+bgpool"),
          "share": ("both+rediscover@share+whatif2", "both+rediscover@share+whatif2+bgpool"),
          "glyph": ("both@ascii+rediscover+whatif2", None)}


def choice(st, eye, part):
    vals = [st.get(f"cf {eye} {part} {w}", np.nan) for w in CF]
    if any(np.isnan(v) for v in vals):
        return None
    best = max(vals)
    return 1.0 if vals[CF.index(1.0)] >= best - 1e-12 else CF[int(np.argmax(vals))]


def direction(w):
    return 0 if w == 1.0 else (1 if w > 1.0 else -1)


def gain(st, part, w):
    return st[f"cf out {part} {w}"] - st[f"cf out {part} 1.0"]


def sgn(x):
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    w, lo = int((x > 1e-12).sum()), int((x < -1e-12).sum())
    return f"{x.mean():+.4f} ({w}:{lo}, p={sign_p(w, lo):.3f})" if len(x) else "  -  "


if __name__ == "__main__":
    print("=== finding 121: do the learner's own eyes agree, and does agreement mark trustworthy choices? (cached lives) ===")
    for world, (cw, cb) in WORLDS.items():
        seeds = range(10)
        lw = {sd: pickle.load(open(f"{RL.CACHE}/seed{sd}_{cw}.pkl", "rb")) for sd in seeds}
        lb = {sd: pickle.load(open(f"{RL.CACHE}/seed{sd}_{cb}.pkl", "rb")) for sd in seeds} if cb else None
        print(f"\n##### {world}")
        for part in ("chunk", "assoc"):
            agree = {"S-W": [], "B-W": [], "S-B": []}
            gw = {"S agrees": [], "S disagrees": [], "B agrees": [], "B disagrees": []}
            pol = {k: {sd: [] for sd in seeds} for k in ("i", "ii", "iii", "iv")}
            late = {k: {sd: [] for sd in seeds} for k in ("i", "ii", "iii", "iv")}
            for sd in seeds:
                for i in range(1, 16):
                    st = lw[sd][i]
                    s, w = choice(st, "self", part), choice(st, "world", part)
                    if s is None or w is None or f"cf out {part} 1.0" not in st:
                        continue
                    b = choice(lb[sd][i], "world", part) if lb else None
                    ds, dw = direction(s), direction(w)
                    agree["S-W"].append(ds == dw)
                    gw["S agrees" if ds == dw else "S disagrees"].append(gain(st, part, w))
                    if b is not None:
                        db = direction(b)
                        agree["B-W"].append(db == dw)
                        agree["S-B"].append(ds == db)
                        gw["B agrees" if db == dw else "B disagrees"].append(gain(st, part, w))
                    picks = {"i": w, "ii": w if ds == dw else 1.0}
                    if b is not None:
                        picks["iii"] = w if direction(b) == dw else 1.0
                        votes = [(ds, s), (dw, w), (direction(b), b)]
                        maj = [d for d in (-1, 0, 1) if sum(v[0] == d for v in votes) >= 2]
                        picks["iv"] = (w if maj and maj[0] == dw else
                                       next(x for d_, x in votes if maj and d_ == maj[0]) if maj else 1.0)
                    for k, x in picks.items():
                        pol[k][sd].append(gain(st, part, x))
                        if i >= 8:
                            late[k][sd].append(gain(st, part, x))
            print(f"  [{part}] A1 agreement " + " | ".join(f"{k} {np.mean(v):.2f} (n={len(v)})" for k, v in agree.items() if v))
            print(f"  [{part}] A2 outside gain of W's choice: " + " | ".join(
                f"{k} {np.mean(v):+.4f} (>=0 {np.mean(np.asarray(v) >= -1e-12):.2f}, n={len(v)})" for k, v in gw.items() if v))
            for name, P in (("s2-16", pol), ("s9-16", late)):
                base = np.array([np.mean(P["i"][sd]) for sd in seeds])
                row = [f"(i) W {sgn(base)}"]
                for k in ("ii", "iii", "iv"):
                    if all(P[k][sd] for sd in seeds):
                        row.append(f"({k}) - (i) {sgn(np.array([np.mean(P[k][sd]) for sd in seeds]) - base)}")
                print(f"  [{part}] A3 {name}: " + " | ".join(row))
