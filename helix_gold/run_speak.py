"""Stage-1 path step 3b (2026-10-08, the user's question: run the understanding logic backwards -- scene -> words; decision
"그렇게 진행해줘"). Record only: the learner is the finding-123/125 body (Korean particle stories, world model with binding,
the decided world model speaks last); the speaking probe (run_long.speak_probe, "+speak") changes nothing it learns
(checked: seed 0, max difference 0 against the life without the probe).

Items "speak" (data_ascii, @kor; a separate random stream; nothing else changes, checked seeds 0, 3, 5): a scene -- an
object's name, from(a), to(b) -- told as one sentence "[mover P1] [landmark P2] VERB". The learner fills it with what it
learned: P2 = the particle it found marks the start (finding 123), P1 = the other particle it heard bound to names, the
landmark = a noun it remembers named at the start cell, the motion word by (body) its body's best motion word or (book)
the books read backwards (step 3a). "mon" = self-monitoring: the sentence is read back with its own understanding and
re-chosen until it lands on the seen end (Levelt 1989). Graded by the world (phrase order free: several sentences are
right). Control "swap": the particles exchanged.

Smoke (seed 0): v1 said a noun of ANOTHER room named at the same cell (positions are shared by the rooms) -- 0.75 -> 0.11
over the life while its read-back passed every time (the comprehension cannot tell the rooms' nouns apart at one cell:
self-monitoring cannot catch what its own understanding cannot see); fixed before any 30-seed run: the mover's room's
nouns first. v2: body 0.50-0.67, body mon 1.00, book 1.00, swap 0.00; the learner unchanged (max difference 0).

Condition (30 seeds): both@ascii@kor+rediscover+chain+wm+bind+wmlast+speak.
Pre-registered (paired within each life; end = session 16, life = mean over sessions with speak items):
  S1 book mon - body: end, life (the full backwards path vs the body alone).
  S2 the monitor's own gain: body mon - body; book mon - book.
  S3 swap (expected ~0: the learned particles carry the roles).
  S4 recorded: landmark found, read-back right, learned particles right (GA / ESEO).
"""
import sys
from multiprocessing import Pool

import numpy as np

import run_long as RL
from run_unified_continual import sign_p

C = "both@ascii@kor+rediscover+chain+wm+bind+wmlast+speak"


def sgn(x):
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    w, lo = int((x > 1e-12).sum()), int((x < -1e-12).sum())
    return f"{x.mean():+.4f} ({w}:{lo}, p={sign_p(w, lo):.4f})"


if __name__ == "__main__":
    n_seeds = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    workers = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    with Pool(workers) as p:
        out = dict(zip(range(n_seeds), p.map(RL.one, [(sd, C) for sd in range(n_seeds)])))
    print(f"=== {n_seeds} seeds; scene -> one Korean sentence (stage-1 step 3b, record only) ===")
    g = lambda k: np.array([[st.get(f"speak {k}", np.nan) for st in out[sd]] for sd in range(n_seeds)])
    for k in ("body", "body mon", "book", "book mon", "swap", "readback", "lm found", "particles ok"):
        print(f"  {k:13} " + " ".join("  -  " if np.isnan(v) else f"{v:.3f}" for v in np.nanmean(g(k), 0)))
    for tag, a, b in (("S1 book mon - body", "book mon", "body"), ("S2 body mon - body", "body mon", "body"),
                      ("S2 book mon - book", "book mon", "book")):
        d = g(a) - g(b)
        print(f"  {tag}: end {sgn(d[:, -1])} | life {sgn(np.nanmean(d, 1))}")
    print(f"  S3 swap: end mean {np.nanmean(g('swap')[:, -1]):.3f}, life mean {np.nanmean(g('swap')):.3f}")
