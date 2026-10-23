"""D80 v2 (2026-10-22; D80_DESIGN.md "v2"; the user's decision "응, 너의 의견대로 (가)와 (나)를 순서대로 진행해줘." -- (나)
after D72 subject 4). Written and fixed BEFORE any life of these seeds (check_seq2_world.out reads the world and the device
logic without any body).

What is new: moves TOLD only in words. The "@seq2" world is the "@seq" world with the scene lines after the walk's last
SEQ_TOLD = 4 moves removed -- those moves are heard only as the sentence [name, motion word] (the exam's C8 tells moves in
words). v1.1's store puts only SEEN moments into a time line, so it cannot know where a thing went after a told move. v2:
a second store whose where line also takes a told sentence -- the scene part (D75, "+scene") gives the thing it moves
(the sentence word whose counted movers are the most constant) and the move (its counted move in that room); the expected
cell = the line's last cell + the move; the cell seen right after the sentence takes its place (what is seen replaces what
was said). The store learns its own question meanings with the same equals sign (v1.1, with the answer channel). The
scene part's observation rule now needs a moment without a word (a question-and-answer moment is said, not seen; no
"@scene" observation holds a word -- check_seq2_world.out).
Condition: both@ascii@seq2+rediscover+scene, NEW seeds 20-29 (seeds 0-19 were seen in v1 / v1.1). The same body answers
every item three ways: U (usual: the body and the scene part), EQ1 (v1.1: the seen-only store), EQ2 (v2: the told-aware
store).
Pre-registered:
  W1 told now: EQ2 "seq now told" (right after the room arrived, items pooled) >= 0.80 AND EQ2 - EQ1 sign test p < 0.005.
  W2 told before: EQ2 "seq before told" >= 0.80 AND EQ2 - EQ1 p < 0.005.
  W3 seen items ("seq now", "seq before", right after the room arrived): an EQ2 DROP = EQ2 - EQ1 <= -0.05 with p < 0.005.
  W4 old keys at the last session (motion, names, scene): an EQ2 DROP = EQ2 - U <= -0.05 with p < 0.005.
  Verdict: "성공" = W1 and W2 and no W3 / W4 drop; otherwise "실패" (what failed is named).
  Report: U on the told items; the learned meanings of both stores; the scene part's counted events.
  Expected before any life (check_seq2_world.out, no body): the told-aware line gives the right cell for nearly every told
  item, the seen-only line for few -- the body decides only through the learned question meanings and the kind gate.
What it cannot show: the exam's C8 moves are mostly relative to another thing (put on, push to -- D75 step 2's frame) and
its story comes inside the question, not in the life; the scene part's frame (own cell) and the three time relations are
designed; Voyager's own world only.
    python3 run_d80c.py [first_seed=20] [n_seeds=10] [workers=3]"""
import sys

import numpy as np

import run_long as RL
from run_count import sgn
from run_d80 import last, pooled, test

COND = "both@ascii@seq2+rediscover+scene"
RULES = ("U", "EQ1", "EQ2")

if __name__ == "__main__":
    first = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    n_seeds = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    workers = int(sys.argv[3]) if len(sys.argv) > 3 else 3
    seeds = list(range(first, first + n_seeds))
    lives = RL.map_lives([(sd, COND) for sd in seeds], workers)
    if any(lf is None for lf in lives):
        print("PIECE DEADLINE: some lives not finished -- no table", flush=True)
        sys.exit(0)
    print(f"=== D80 v2 (moves told only in words -> expected cells): \"@seq2\", seeds {seeds[0]}-{seeds[-1]}, {COND} ===")
    kinds = ("seq now told", "seq before told", "seq now", "seq before")
    v = {(k, r): np.array([pooled(lf, k, r) for lf in lives]) for k in kinds for r in RULES}
    for k in kinds:
        print(f"  {k} (right after the room arrived): " + " / ".join(f"{r} {np.nanmean(v[(k, r)]):.3f}" for r in RULES)
              + f"; EQ2 - EQ1 {sgn(v[(k, 'EQ2')] - v[(k, 'EQ1')])}; EQ2 - U {sgn(v[(k, 'EQ2')] - v[(k, 'U')])}")
    for nm in ("seq", "seq2"):
        rel = {}
        for lf in lives:
            for k_, x in lf[-1].items():
                if k_.startswith(f"{nm} rel "):
                    rel.setdefault(k_[len(nm) + 5:], []).append(("last", "before", "first")[int(x)])
        print(f"  learned meanings ({'seen-only' if nm == 'seq' else 'told-aware'} store) at the last session: " +
              "; ".join(f"{k_}: {sorted(set(x))} ({len(x)} seeds)" for k_, x in sorted(rel.items())))
    print(f"  scene part events at the last session: {np.mean([lf[-1].get('scene events', np.nan) for lf in lives]):.0f} "
          f"(mean over seeds)")
    w1m, w1p = test(v[("seq now told", "EQ2")] - v[("seq now told", "EQ1")])
    w2m, w2p = test(v[("seq before told", "EQ2")] - v[("seq before told", "EQ1")])
    W1 = np.nanmean(v[("seq now told", "EQ2")]) >= 0.80 and w1m > 0 and w1p < 0.005
    W2 = np.nanmean(v[("seq before told", "EQ2")]) >= 0.80 and w2m > 0 and w2p < 0.005
    drops3 = []
    for k in ("seq now", "seq before"):
        m, p_ = test(v[(k, "EQ2")] - v[(k, "EQ1")])
        if m <= -0.05 and p_ < 0.005:
            drops3.append(k)
    print("  W4 old keys at the last session (U / EQ2):")
    drops4 = []
    for k in ("old motion", "old name", "old scene"):
        u = np.array([last(lf, k, "U") for lf in lives])
        x = np.array([last(lf, k, "EQ2") for lf in lives])
        m, p_ = test(x - u)
        d = m <= -0.05 and p_ < 0.005
        if d:
            drops4.append(k)
        print(f"    {k}: U {np.nanmean(u):.3f} | EQ2 {np.nanmean(x):.3f} ({sgn(x - u)}){' DROP' if d else ''}")
    failed = [n for n, ok in (("W1 told now", W1), ("W2 told before", W2), ("W3 seen items", not drops3),
                              ("W4 old keys", not drops4)) if not ok]
    print(f"  W1 told now EQ2 {np.nanmean(v[('seq now told', 'EQ2')]):.3f} (p = {w1p:.4f}); W2 told before EQ2 "
          f"{np.nanmean(v[('seq before told', 'EQ2')]):.3f} (p = {w2p:.4f}); W3 drops {drops3}; W4 drops {drops4} "
          f"-> verdict: {'성공' if not failed else '실패 (' + ', '.join(failed) + ')'}")
    print("exit 0")
