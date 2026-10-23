"""W7 pilot (2026-10-16): the parts that FIT the world vs ALL parts on -- a short test before the first score table.

The user (10-16): "부품은 알맞게 키는게 맞는거 같은데, 인간의 뇌를 보면 다른 기능의 부품이 작업에 참여해 일정부분 도움을
주는 경우가 있어서 다 키는 경우의 결과가 어떨지 좀 궁금해. 짧게 두 경우를 테스트하는걸 먼저 해줘." Written and fixed
BEFORE any life of this pilot. No part and no world changes: only two new condition names.

Parts (every positive part flag the code has; controls "rand / rev / all / one / id", the measurement modes "+wmtest /
+wmmeasure" and the tale modes are not parts): wm, wmlast, wmctx, wmrev, bind, track, props, grav, wmframe, contain, count,
succ, scale, views, vturn, goal, self (17), on the base "+rediscover+chain".
FIT (the rule, fixed here -- a part fits a world when the world holds the items it was made for):
  wm, wmlast   every world (motion words)          wmctx   every world (the upside-down rooms, finding 122)
  wmrev        worlds with "describe" items (scene -> motion word; every Korean-story world)
  bind         Korean-particle worlds              track   multi-moment story worlds
  contain / count / grav / props / succ / scale / views + vturn / goal / self / wmframe
               only in the world of that subject (@contain-family / @count-family / @grav / @props / @succ / @numrow /
               @views-@viewpt / @goal / @self / @frame)
ALL = FIT + every other part.
Worlds (two, chosen where another subject's part might help -- the user's question): "@ten" (W5: numbers 1-10; "next number"
was chance 0.106 -- could the chant's successor part or the number line's scale help?) and "@sp" (W3: spatial words; "아직"
-- could gravity / support, viewpoints or counting help?). Seeds 0-2 (a pilot: direction, not a verdict).
  @ten FIT = wm wmlast wmctx wmrev bind track count contain             ALL adds props grav wmframe succ scale views vturn
                                                                            goal self (9)
  @sp  FIT = wm wmlast wmctx wmrev bind track contain                   ALL adds props grav wmframe count succ scale views
                                                                            vturn goal self (10)
  reference (cached, no new life): the world's W-runner condition P (W5 / W3) -- FIT differs from it by wmctx + wmrev (+ contain
  on @ten, which W5's P lacked -- finding 145's lesson).
Readings (report only, pre-registered):
  R1 per world, the main keys (W5: "ten count", "ten pick", "ten next", "ten box new", "voe ten ..."; W3: "sp word4",
     "sp scene4"; both: "contain ...", "story", "never-heard", "chain", "multi", "describe"): end values FIT / ALL / P and
     ALL - FIT per seed.
  R2 every key both lives carry: how many keys move with ALL - FIT 3:0 in one direction AND |mean| >= 0.02 (help / harm).
  R3 cost: ALL / FIT in reference-job units (D51), peak memory.
DECISION FOR W7b (fixed here): the judged column stays FIT (the user's lean, "알맞게 키는게 맞는거 같은데"). If R2 shows a
key helped by ALL (3:0, mean >= +0.02), W7b adds an ALL column for every world (the human-brain-like help is then measured at
10 seeds); if ALL only harms or changes nothing, W7b runs FIT only and this pilot is written as a finding (which parts stay
silent, which interfere). Three seeds cannot decide anything (the smallest sign-test p is 0.25) -- a direction only.
What this assumes: the FIT rule above (a human choice, written in HUMAN_SHARE.md's sense); two worlds stand for the rest;
the parts are switched on from the first session (no waking).
    python3 run_w7pilot.py [N_SEEDS=3] [WORKERS=4]
"""
import json
import os
import pickle
import sys

import numpy as np

import run_long as RL

BASE = "+rediscover+chain"
FIT = {"ten": "+wm+wmctx+wmrev+bind+wmlast+track+count+contain",
       "sp": "+wm+wmctx+wmrev+bind+wmlast+track+contain"}
EXTRA = {"ten": "+props+grav+wmframe+succ+scale+views+vturn+goal+self",
         "sp": "+props+grav+wmframe+count+succ+scale+views+vturn+goal+self"}
REF = {"ten": "both@ascii@ten+rediscover+chain+wm+bind+wmlast+track+count",
       "sp": "both@ascii@sp+rediscover+chain+wm+bind+wmlast+track+contain"}
PARTS = ["wm", "wmlast", "wmctx", "wmrev", "bind", "track", "props", "grav", "wmframe", "contain", "count", "succ", "scale",
         "views", "vturn", "goal", "self"]
NOT_PARTS = ["bindrand", "trackone", "propsrand", "gravall", "containall", "countrand", "succrev", "scalerev", "viewsrand",
             "goalrand", "selfid", "wmtest", "wmmeasure", "tk1", "tkA", "tkig"]
MAIN = {"ten": ("ten count", "ten pick", "ten next", "ten box new", "voe ten"),
        "sp": ("sp word4", "sp scene4")}
COMMON = ("contain", "story", "never-heard", "chain", "multi", "describe")


def cond(world, which):
    return f"both@ascii@{world}{BASE}{FIT[world]}" + (EXTRA[world] if which == "ALL" else "")


def flag_check():
    """the substring-collision check: every part flag is on exactly when intended; no control / mode flag is on."""
    bad = []
    for world in FIT:
        for which in ("FIT", "ALL"):
            c = cond(world, which)
            want = set((FIT[world] + (EXTRA[world] if which == "ALL" else "")).strip("+").split("+"))
            for p in PARTS:
                if (f"+{p}" in c) != (p in want) and not (p == "wm" and "+wm" in c):
                    bad.append((c, p))
            for p in NOT_PARTS:
                if f"+{p}" in c:
                    bad.append((c, p))
    return bad


def life_end_mean(stats, k):
    v = [st.get(k, np.nan) for st in stats]
    return v[-1], np.nanmean(v) if np.any(~np.isnan(np.array(v, dtype=float))) else np.nan


def units(c, sd):
    f = os.path.join(RL.CACHE + "_cpu", f"seed{sd}_{c}.json")
    if not os.path.exists(f):
        return np.nan, np.nan
    d = json.load(open(f))
    return d["cpu_cum_s"][-1] / d["ref_cpu_s"], d.get("peak_rss_gb", np.nan)


if __name__ == "__main__":
    n_seeds = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    workers = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    bad = flag_check()
    assert not bad, f"flag collision: {bad}"
    seeds = list(range(n_seeds))
    tasks = [(sd, cond(w_, x_)) for w_ in FIT for x_ in ("FIT", "ALL") for sd in seeds]
    out = dict(zip(tasks, RL.map_lives(tasks, workers)))
    print(f"=== W7 pilot: parts that FIT the world vs ALL parts on, {n_seeds} seeds (direction only) ===")
    print("flag check: no collision")
    for w_ in FIT:
        F, A = cond(w_, "FIT"), cond(w_, "ALL")
        ref = {sd: pickle.load(open(os.path.join(RL.CACHE, f"seed{sd}_{REF[w_]}.pkl"), "rb")) for sd in seeds
               if os.path.exists(os.path.join(RL.CACHE, f"seed{sd}_{REF[w_]}.pkl"))}
        num = (int, float, np.integer, np.floating)     # stat values that are numbers (a few keys hold words, e.g. a mode)
        keys = sorted(k for k in out[(0, F)][-1] if isinstance(k, str) and k in out[(0, A)][-1]
                      and isinstance(out[(0, F)][-1][k], num) and isinstance(out[(0, A)][-1][k], num))
        print(f"--- @{w_}: FIT {FIT[w_]} | ALL adds {EXTRA[w_]} | reference P = W-runner condition ({len(ref)} seeds cached)")
        print(f"  R1 main keys (end; mean over seeds)        FIT     ALL     P(ref)  ALL-FIT per seed")
        for k in keys:
            if not (k.startswith(MAIN[w_]) or k.startswith(COMMON)) or "body" in k:
                continue
            f_ = np.array([life_end_mean(out[(sd, F)], k)[0] for sd in seeds], dtype=float)
            a_ = np.array([life_end_mean(out[(sd, A)], k)[0] for sd in seeds], dtype=float)
            r_ = np.array([life_end_mean(ref[sd], k)[0] if sd in ref else np.nan for sd in seeds], dtype=float)
            if np.all(np.isnan(f_)) and np.all(np.isnan(a_)):
                continue
            d_ = a_ - f_
            print(f"  {k:40s} {np.nanmean(f_):.3f}   {np.nanmean(a_):.3f}   {np.nanmean(r_):.3f}   "
                  + " ".join(f"{x:+.3f}" for x in d_))
        moved = []
        for k in keys:
            d_ = np.array([life_end_mean(out[(sd, A)], k)[0] - life_end_mean(out[(sd, F)], k)[0] for sd in seeds],
                          dtype=float)
            if np.any(np.isnan(d_)):
                continue
            if (np.all(d_ > 0) or np.all(d_ < 0)) and abs(d_.mean()) >= 0.02:
                moved.append((k, d_.mean()))
        helped = [(k, m) for k, m in moved if m > 0]
        harmed = [(k, m) for k, m in moved if m < 0]
        print(f"  R2 keys moved by ALL ({len(seeds)}:0, |mean| >= 0.02) of {len(keys)}: helped {len(helped)}, harmed {len(harmed)}")
        for k, m in sorted(moved, key=lambda x: -abs(x[1]))[:25]:
            print(f"     {'+' if m > 0 else '-'} {k:40s} {m:+.3f}")
        uf = [units(F, sd) for sd in seeds]
        ua = [units(A, sd) for sd in seeds]
        print(f"  R3 cost: ALL / FIT units {np.nanmean([u[0] for u in ua]) / np.nanmean([u[0] for u in uf]):.2f} "
              f"(FIT {np.nanmean([u[0] for u in uf]):.0f}, ALL {np.nanmean([u[0] for u in ua]):.0f}); "
              f"peak GB FIT {np.nanmax([u[1] for u in uf]):.2f}, ALL {np.nanmax([u[1] for u in ua]):.2f}")
