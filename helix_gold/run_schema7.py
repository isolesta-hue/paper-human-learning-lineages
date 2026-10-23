"""Content layer, seventh experiment: choosing templates by context -- the templates' own coordinate system
(finding 57 follow-up, 2026-09-24; design agreed with the user).

Finding 57: a reality filter for the whole template channel moved the right way (more trust in the schema world, less in
the random one) but one global trust cannot tell "which game is this": when the rules changed, the new templates' success
and the old templates' failure mixed, and adapting its speed did nothing. The user's picture: playing baseball with the
rules of basketball -- people first narrow the candidates by the cues of the place (a field, a ball), try the familiar one,
move on if it does not fit, and settle on the one that fits. So far every template votes on every new word -- more
candidates, more chance hits (the multiple-comparisons problem).
  place   -- the place axis of a coordinate system for templates: a template remembers the categories whose pairs
             confirmed it (a category's place = the centre of its words' memory vectors). Where a context word lives, a
             template counts by how near its nearest confirmed place is, relative to the nearest template's
             (exp(-(d_k^2 - d_min^2) / 2h^2), h = the typical distance between neighbouring categories -- no free constant).
             A template confirmed everywhere applies everywhere; one confirmed only in one region applies there.
  trial   -- trying each template on the current situation: on the new sentences just heard (before practice), each
             template alone -- how many answers does it turn wrong -> right (helped) and right -> wrong (harmed)? Its weight
             this session x (helped + 1) / (helped + harmed + 2). No memory across sessions: a person tries the familiar
             candidate first because trying costs time; for this system trying every template costs little, so all are
             tried (the constraint that makes people search in order is not shared -- what remains shared is that a test on
             few sentences can pass a wrong template by chance).
  place+trial -- both.
Worlds: schema / random / local (every super-category has its own R0-R2, R3 shared: 13 directions -- place should matter) /
back (rule set A, then B, then A again: place does not tell A from B -- only trying does).
Learners (all on the standardised habit, independent evidence, context mask; finding 57): D-norm / masked / masked+filter
(finding 57's global filter) / place / trial / place+trial.
"""
import sys
from multiprocessing import Pool

import numpy as np
import torch

from run_unified_continual import sign_p
import run_schema3 as S3

WORLDS7 = ["schema", "random", "local", "back"]
LEARNERS = ["D-norm", "masked", "masked+filter", "place", "trial", "place+trial"]


def life(world_name, learner, seed):
    if learner == "D-norm":
        return S3.life(world_name, "D-norm", seed)
    kw = dict(evidence=True, independent=True, mask_context=True)
    if learner == "masked+filter":
        kw["reality"] = "fixed"
    elif learner != "masked":
        kw["select"] = learner
    return S3.life(world_name, "norm+explain", seed, **kw)


def one_seed(seed):
    torch.set_num_threads(1)
    return {(wn, ln): life(wn, ln, seed) for wn in WORLDS7 for ln in LEARNERS}


def main():
    n_seeds = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    workers = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    with Pool(workers) as pool:
        results = pool.map(one_seed, range(n_seeds))
    n_sess = len(results[0][(WORLDS7[0], LEARNERS[0])])
    print(f"=== {n_seeds} seeds; data_v3 schema / random / local (own rules per super-category) / back (A, B, A) ===")
    for key, label in [("curve once", "newest ONE-SHOT pairs, epoch 0"), ("curve unseen", "newest UNSEEN pairs, epoch 0")]:
        print(f"\n--- {label} (sessions 1..{n_sess}) ---")
        for wn in WORLDS7:
            for ln in LEARNERS:
                print(f"{wn:6} {ln:14} " + " ".join(f"{np.nanmean([r[(wn, ln)][i][key][0] for r in results]):.3f}" for i in range(n_sess)))
    for key in ["never-heard", "pairs all", "once all", "unseen all", "used", "junk used", "rel",
                "place right", "place wrong", "trial right", "trial wrong"]:
        print(f"\n--- {key} (sessions 1..{n_sess}) ---")
        for wn in WORLDS7:
            for ln in LEARNERS:
                row = [np.nanmean([r[(wn, ln)][i][key] for r in results]) for i in range(n_sess)]
                if not all(np.isnan(row)):
                    print(f"{wn:6} {ln:14} " + " ".join(f"{v:6.3f}" for v in row))

    print("\n--- paired ---")
    def cmp(ga, gb, label):
        cells = []
        for lab, get in [("life", lambda r, g: np.mean([st["never-heard"] for st in r[g]])),
                         ("end", lambda r, g: r[g][-1]["never-heard"]),
                         ("2nd half", lambda r, g: np.mean([st["never-heard"] for st in r[g][4:]])),
                         ("pairs end", lambda r, g: r[g][-1]["pairs all"]),
                         ("once end", lambda r, g: r[g][-1]["once all"]),
                         ("once@0 s2-8", lambda r, g: np.mean([st["curve once"][0] for st in r[g][1:]]))]:
            dd = np.array([get(r, ga) - get(r, gb) for r in results])
            n_pos, n_neg = int((dd > 1e-9).sum()), int((dd < -1e-9).sum())
            cells.append(f"{lab} {dd.mean():+.3f} ({n_pos}/{n_neg}{'*' if sign_p(n_pos, n_neg) < 0.01 else ''})")
        print(f"{label:34} " + "  ".join(cells))
    for wn in WORLDS7:
        cmp((wn, "masked"), (wn, "D-norm"), f"{wn}: masked vs D-norm")
        cmp((wn, "masked+filter"), (wn, "masked"), f"{wn}: filter vs masked")
        cmp((wn, "place"), (wn, "masked"), f"{wn}: place vs masked")
        cmp((wn, "trial"), (wn, "masked"), f"{wn}: trial vs masked")
        cmp((wn, "trial"), (wn, "masked+filter"), f"{wn}: trial vs filter")
        cmp((wn, "place+trial"), (wn, "trial"), f"{wn}: place+trial vs trial")
        cmp((wn, "place+trial"), (wn, "D-norm"), f"{wn}: place+trial vs D-norm")


if __name__ == "__main__":
    main()
