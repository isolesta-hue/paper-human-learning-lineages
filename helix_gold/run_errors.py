"""Finding 112 (2026-10-03, the user's decision: fill the four gaps humans and LLMs have and Voyager lacks, tied to the
emotion modes): step 1 -- the error-type eye, record only.

The method layer so far sees one number (the judge's score). Every big gain of this project came after WE split the wrong
answers by kind (finding 90: 30-43% of the glyph answers were not positions; 92; 101). The eye moves that diagnosis into the
learner: each wrong answer gets an OWN label from what the learner itself knows -- kind (input channel tag; glyph world),
topic (its own classifier), pulled (heard together with a context word: its own Hebbian record), near (among the hidden
one's 5 nearest in its own memory space), other -- and, for checking only, the DEVELOPER's truth (world labels: topic,
background, near cell). Recorded on the outside test items at the end of every session; nothing the learner does changes.

Conditions (10 seeds, a probe): both+rediscover+errors (text), both@ascii+rediscover+errors (glyph); references: the cached
both+rediscover, both@ascii+rediscover.
Pre-registered:
  E0 regression -- scores equal the cached references exactly.
  E1 validity -- per pair (own topic / dev topic, own pulled / dev background, own near / dev near cell; glyph only for the
     last): precision and recall of the own label against the developer's, pooled over seeds and sessions. The eye is usable
     for the dashboard if precision >= 0.8 on the labels that carry most of the errors.
  E2 the picture -- the own-label shares of all items by session, per world: which kinds of error dominate when, and
     whether they move over the life (a label that never moves cannot trigger a mode).
  E3 the answering process (the user, 2026-10-03: the learner should see every number of its own thinking -- an advantage
     over humans, who cannot watch their own brain): the answer's score split EXACTLY into its parts (habit similarity,
     habit connection table, templates, whole-sentence memory, side part; kind gate as a mask; split error must be ~1e-5),
     and for every wrong answer the part that made it win (largest share of score(answer) - score(hidden)) -- by session,
     with the mean margins. This is the "logic" step of our diagnosis order, done by the learner.
"""
import sys
from multiprocessing import Pool

import numpy as np

import run_long as RL

PAIRS = {"text": ("both+rediscover+errors", "both+rediscover"), "glyph": ("both@ascii+rediscover+errors", "both@ascii+rediscover")}

if __name__ == "__main__":
    n_seeds = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    workers = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    tasks = [(sd, c) for pr in PAIRS.values() for c in pr for sd in range(n_seeds)]
    with Pool(workers) as p:
        out = dict(zip(tasks, p.map(RL.one, tasks)))
    print(f"=== {n_seeds} seeds; the error-type eye (record only) ===")
    for world, (c, ref) in PAIRS.items():
        same = all(np.allclose([s["never-heard"] for s in out[(sd, c)]], [s["never-heard"] for s in out[(sd, ref)]],
                               atol=1e-12) for sd in range(n_seeds))
        lives = [out[(sd, c)] for sd in range(n_seeds)]
        n = len(lives[0])
        print(f"\n##### {world}: E0 regression identical: {same}")
        print("E2 own-label shares of all items by session:")
        for k in ("wrong",) + RL.ERR_OWN:
            print(f"  {k:7} " + " ".join(f"{np.mean([l[i][f'err {k}'] for l in lives]):.3f}" for i in range(n)))
        print("E3 the part that made the wrong answer win (share of wrong answers) by session:")
        for q in RL.PARTS + ("gate",):
            print(f"  {q:7} " + " ".join(f"{np.mean([l[i][f'cause {q}'] for l in lives]):.3f}" for i in range(n)))
        print("   mean margin (answer - hidden) per part, last session: "
              + " ".join(f"{q} {np.mean([l[-1][f'margin {q}'] for l in lives]):+.2f}" for q in RL.PARTS))
        print(f"   largest split error {max(l[i]['parts split error'] for l in lives for i in range(n)):.1e}")
        print("E1 own label vs developer truth (pooled):")
        for o, d in (("topic", "topic"), ("pulled", "background"), ("near", "near")):
            both = sum(l[i][f"errv {o} {d}"] for l in lives for i in range(n))
            own = sum(l[i][f"errv own {o}"] for l in lives for i in range(n))
            dev = sum(l[i][f"errv dev {d}"] for l in lives for i in range(n))
            print(f"  own {o:6} vs dev {d:10}: precision {both / max(own, 1):.3f}  recall {both / max(dev, 1):.3f}"
                  f"  (own {own:.0f}, dev {dev:.0f})")
