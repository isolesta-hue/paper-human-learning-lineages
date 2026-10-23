"""D75 step 1 (2026-10-17; D75_DESIGN.md; the user: "너의 추천대로 (가)부터 해보자"): can Dami learn movement from the
scenes before and after, WITHOUT role markers? Written and fixed BEFORE any life of this test (the seed-0 smoke,
smoke_scene.out, read the scene part's event counts only, no outcome key).

World "@scene" (data_ascii.build(scene=True)): the "@walk" world's walk told as a child sees it -- every move is a sentence
[the mover's name, the motion word] followed by the scene after it, one moment [picture, at(cell)] per thing of the room;
no from / to markers anywhere in the moves (rooms upside down in phase 2, as every world). "@sceneshuf": the sentence's
motion word drawn at random among the room's while the scene shows the true move (the control: nothing to learn from the
pairing). The part "+scene" (scene_part.py): scene memory, change = event, the most constant move per word (the room's
own counts first), roll-out; its cell speaks last where it fires (as finding 125's rule).
Conditions (seeds 0-9):
  P = both@ascii@scene+rediscover+scene       the principle
  S = both@ascii@sceneshuf+rediscover+scene   the shuffle control (same part, moves paired with random words)
Key "scene" (four- or five-way: right when the answer scores strictly highest among the arrival cells of the room's motion
words from that start; chance 1/4 or 1/5): in every walking room met, per motion word, a start never heard with that word
there -- [picture, at(start), name, word] -> at(arrival). " body" = the same with the scene part (and any world model)
silent: the body alone (D21, diagnosis only).
Readings (END = the life's last session; LIFE = mean over sessions):
  R1 P END "scene": mean, pooled items' Wilson 95% lower bound, chance.
  R2 P - S, END and LIFE: mean, seeds better / worse, sign test p.
  R3 P - P body (END): the part's share.
  R4 the scene part's record: events, words.
  R5 cost: CPU per life (side files).
DECISION (written before): "배움(표지 없이)" when R2 END is decided better (p < 0.005) and R1's lower bound >= 0.85 (the
pass line of STAGE1 criterion 1's "where"); "약함" when R2 is decided better but the lower bound is under 0.85; else "아님".
What it assumes (written before): the own-cell frame only (D75 step 2 adds the other-thing frame); the mover in a test is
the one picture in the question; the room is the picture's category; a naming moment with background pictures is not an
observation.  python3 run_d75.py N_SEEDS WORKERS"""
import json
import os
import sys

import numpy as np

import run_long as RL

P = "both@ascii@scene+rediscover+scene"
SH = "both@ascii@sceneshuf+rediscover+scene"


def sign_p(n_pos, n_neg):
    from math import comb
    n = n_pos + n_neg
    if n == 0:
        return 1.0
    k = min(n_pos, n_neg)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)


def wilson_lb(k, n, z=1.96):
    if n == 0:
        return float("nan")
    c = (k + z * z / 2) / (n + z * z)
    h = z * np.sqrt(k * (n - k) / n + z * z / 4) / (n + z * z)
    return float(c - h)


def series(r, key):
    return [s.get(key, np.nan) for s in r]


if __name__ == "__main__":
    n_seeds = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    workers = int(sys.argv[2]) if len(sys.argv) > 2 else 2
    for c in (P, SH):
        assert c.count("+scene") == 1 and ("@sceneshuf" in c) == (c == SH)
    seeds = list(range(n_seeds))
    tasks = [(sd, c) for c in (P, SH) for sd in seeds]
    out = dict(zip(tasks, RL.map_lives(tasks, workers)))
    R = {c: [out[(sd, c)] for sd in seeds] for c in (P, SH)}
    print(f"=== D75 step 1: movement from before / after scenes, no role markers ({n_seeds} seeds, development test) ===")
    end = {c: np.array([r[-1].get("scene", np.nan) for r in R[c]]) for c in (P, SH)}
    life = {c: np.array([np.nanmean(series(r, "scene")) for r in R[c]]) for c in (P, SH)}
    body = np.array([r[-1].get("scene body", np.nan) for r in R[P]])
    # pooled items at the end: the key is a mean over the life's items; the item count is the same per seed's world
    import data_ascii as A
    n_items = [sum(len(cat.get("scene", [])) for cat in A.build(sd, scene=True).categories) for sd in seeds]
    k_right = int(round(float(np.nansum(end[P] * np.array(n_items)))))
    lb = wilson_lb(k_right, int(sum(n_items)))
    print(f"R1 P END scene: mean {np.nanmean(end[P]):.3f}, pooled {k_right}/{sum(n_items)} Wilson 95% LB {lb:.3f}; "
          f"per seed {np.round(end[P], 3).tolist()}; chance 0.20-0.25")
    for tag, a, b in (("END", end[P], end[SH]), ("LIFE", life[P], life[SH])):
        d = a - b
        pos, neg = int((d > 0).sum()), int((d < 0).sum())
        print(f"R2 P - S {tag}: mean {np.nanmean(d):+.3f} ({pos}:{neg}, p = {sign_p(pos, neg):.4f}); S {tag} {np.nanmean(b):.3f}")
    d3 = end[P] - body
    print(f"R3 P - P body END: mean {np.nanmean(d3):+.3f}; body {np.nanmean(body):.3f}")
    print(f"R4 record (P END): events {np.nanmean(series([r[-1:] for r in R[P]][0], 'scene events')):.0f} (seed 0), "
          f"words {np.nanmean([r[-1].get('scene words', np.nan) for r in R[P]]):.1f}")
    cpu = []
    for sd in seeds:
        f = os.path.join(RL.CACHE + "_cpu", f"seed{sd}_{P}.json")
        if os.path.exists(f):
            cpu.append(json.load(open(f)).get("cpu_cum_s", [0])[-1] / 60)
    print(f"R5 cost: P CPU per life {np.mean(cpu) if cpu else float('nan'):.1f} min")
    d = end[P] - end[SH]
    pos, neg = int((d > 0).sum()), int((d < 0).sum())
    better = sign_p(pos, neg) < 0.005 and np.nanmean(d) > 0
    verdict = "배움(표지 없이)" if better and lb >= 0.85 else "약함" if better else "아님"
    print(f"DECISION: {verdict}")
    print("exit 0")
