"""Voyager's per-subject exposure numbers (shared EXAM_DESIGN 8-1): heard items of a subject until a key first reaches 0.90.

Analysis only: reads cached lives (cache_long_v2), rebuilds each life's world to count what was heard; nothing is trained.

Exposure = the subject's heard items (by their kind tag in the world) in every room met up to and including the session where
the key is first >= 0.90 (rooms come two per session in the life's own order, rebuilt exactly as run_long.life does). The
body's own practice and replay are not counted (EXAM_DESIGN 10: repetitions inside the body are each model's method, not a
shared unit). Per seed: the first such session; reported over seeds as the median and the range, and how many seeds never
reached 0.90 in the life ("not reached": the end value, never the end exposure as a lower bound -- New Horizons' wish in 8-1).

Subjects (the condition = the full body of that world's finding; keys as pre-registered by its runner):
  1   sentence understanding  @kor   "story"                     story-heard           (100 seeds)
  1.8 movement rule           @kor   "never-heard"               R0-R4 motion items     (100 seeds)
  3a  counting, scene -> n    @ten   "ten count"                 ten-heard x 3/5 (3 count scenes of 5 per room, edition 3)
  3b  counting, n -> scene    @ten   "ten pick"                  the same count scenes
  4   put / take, new pairs   @ten   "ten box new"               ten-heard x 2/5 (2 box stories)
  6   gravity / support       @grav  ("grav floor"+"grav stack")/2, "grav step"   grav-heard
  6v  violation of expectation @grav "voe float", "voe through"  grav-heard
  9a  spatial word, scene->word @sp  "sp word4"                  sp-heard
  9b  spatial word, word->scene @sp  "sp scene4"                 sp-heard
  2   new word: designed exposures 1 / 2 / 4 / 8 (@nw, finding 149) -- the end values of "nw<k> new"
"""
import glob
import re
import sys

import numpy as np

import integrated
from run_long import CACHE
from run_schema import PER_SESSION

B = "+rediscover+chain+wm+bind+wmlast"
SUBJ = [  # (label, world flag, cond, key or (keys to average), kind tag test, share of those items)
    ("1 sentence (story)", "ascii-kor", "both@ascii@kor" + B, "story", lambda t: t == "story-heard", 1.0),
    ("1.8 movement rule (never-heard)", "ascii-kor", "both@ascii@kor" + B, "never-heard", lambda t: re.fullmatch(r"R\d", t), 1.0),
    ("3a count scene->n (ten count)", "ascii-ten", "both@ascii@ten" + B + "+track+count", "ten count",
     lambda t: t == "ten-heard", 3 / 5),
    ("3b count n->scene (ten pick)", "ascii-ten", "both@ascii@ten" + B + "+track+count", "ten pick",
     lambda t: t == "ten-heard", 3 / 5),
    ("4 put/take new (ten box new)", "ascii-ten", "both@ascii@ten" + B + "+track+count", "ten box new",
     lambda t: t == "ten-heard", 2 / 5),
    ("6 gravity fall (floor+stack)", "ascii-grav", "both@ascii@grav" + B + "+track+grav", ("grav floor", "grav stack"),
     lambda t: t == "grav-heard", 1.0),
    ("6 gravity one-cell (grav step)", "ascii-grav", "both@ascii@grav" + B + "+track+grav", "grav step",
     lambda t: t == "grav-heard", 1.0),
    ("6v voe float", "ascii-grav", "both@ascii@grav" + B + "+track+grav", "voe float", lambda t: t == "grav-heard", 1.0),
    ("6v voe through", "ascii-grav", "both@ascii@grav" + B + "+track+grav", "voe through", lambda t: t == "grav-heard", 1.0),
    ("9a spatial scene->word (sp word4)", "ascii-sp", "both@ascii@sp" + B + "+track+contain", "sp word4",
     lambda t: t == "sp-heard", 1.0),
    ("9b spatial word->scene (sp scene4)", "ascii-sp", "both@ascii@sp" + B + "+track+contain", "sp scene4",
     lambda t: t == "sp-heard", 1.0),
]


def sessions_of(w, seed):
    rng = np.random.default_rng(70_000 + seed)          # as run_long.life: the room order's own generator
    order = [int(c) for phase in w.phases for c in rng.permutation(phase)]
    return [order[i:i + PER_SESSION] for i in range(0, len(order), PER_SESSION)]


def seeds_of(cond):
    out = []
    for f in glob.glob(f"{CACHE}/seed*_{cond}.pkl"):
        m = re.match(r"seed(\d+)_", f.split("/")[-1])
        out.append(int(m.group(1)))
    return sorted(out)


def one(label, world, cond, key, is_kind, share):
    import pickle
    rows, never, ends = [], 0, []
    for sd in seeds_of(cond):
        steps = pickle.load(open(f"{CACHE}/seed{sd}_{cond}.pkl", "rb"))
        w = integrated.build_world(world, sd)
        sess = sessions_of(w, sd)
        assert len(sess) == len(steps), (label, sd, len(sess), len(steps))
        per_room = {c: sum(1 for t in w.categories[c]["heard"] if is_kind(t[0])) * share for c in range(w.n_cat)}
        cum, seen = [], 0.0
        for s in sess:
            seen += sum(per_room[c] for c in s)
            cum.append(seen)
        val = lambda st: (np.mean([st.get(k, np.nan) for k in key]) if isinstance(key, tuple) else st.get(key, np.nan))
        v = [val(st) for st in steps]
        ends.append(v[-1])
        hit = next((i for i, x in enumerate(v) if x == x and x >= 0.90 - 1e-9), None)   # 0.9 from means can be 0.8999...
        if hit is None:
            never += 1
        else:
            rows.append((cum[hit], hit + 1))
        print(f"  {label} seed {sd}: " + " ".join("--" if x != x else f"{x:.2f}" for x in v)
              + f" | first >= 0.90: {'never' if hit is None else f'session {hit + 1}, {cum[hit]:.0f} heard'}", flush=True)
    n = len(ends)
    if rows:
        e = np.array([r[0] for r in rows])
        s_ = np.array([r[1] for r in rows])
        txt = (f"median {np.median(e):.0f} heard [range {e.min():.0f}-{e.max():.0f}], session median {np.median(s_):.0f}"
               f" [{s_.min()}-{s_.max()}]")
    else:
        txt = "not reached"
    print(f"== {label}: {n} seeds; reached {n - never}/{n}; {txt}; end mean {np.nanmean(ends):.3f}"
          f" (total heard in the life: {cum[-1]:.0f})", flush=True)


def newword():
    cond = "both@ascii@nw" + "+rediscover+chain+wm+wmlast"
    import pickle
    ends = {k: [] for k in (1, 2, 4, 8)}
    for sd in seeds_of(cond):
        steps = pickle.load(open(f"{CACHE}/seed{sd}_{cond}.pkl", "rb"))
        for k in ends:
            ends[k].append(steps[-1].get(f"nw{k} new", np.nan))
    print("== 2 new word (designed exposures, end of life): " + ", ".join(
        f"{k} heard {np.nanmean(v):.3f} (n={len(v)})" for k, v in ends.items()), flush=True)


if __name__ == "__main__":
    for row in SUBJ:
        one(*row)
    newword()
