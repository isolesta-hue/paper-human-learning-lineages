"""D26 (1) S1 (2026-10-10; the continuity principle -- "the same Voyager grows", user decision; D26_GROWTH.md). Written and
fixed BEFORE any multi-seed run.

Found while designing word growth: the body answers among ALL words of the world (lab2.acc takes the argmax over V) and its
practice loss uses all V words as the denominator, so a word never heard is already pushed away as a wrong answer, session
after session, before it is ever met -- and it can be answered. "+knownonly" (run_schema3.only_known / classify): only the
words heard so far are answer candidates, in answering and in practice; only the rooms already met are classes. This is
the step that should make appending words later the same as having their places from birth (S2, exact check next).
Inertness audit (probe_inert.py, seed 0, 6 sessions; the D26 S2 check): v1 of "+knownonly" (mask only) still moved the
never-heard words' z_work rows, assoc columns and skill every session -- the habit was standardised over ALL V words, so
every word of the world was part of every answer's scale. Fixed before any multi-seed run: the standardisation over the
heard words only (_tail_known, habit, assoc_share). v2 audit: every value of every word not yet heard stays bit-equal to
its birth value in all 6 sessions (the only reported matches are on dimension 32 = LATENT = n_cat, not word rows), so
having a word's place from birth is the same as appending it when first heard. v1 caches moved to ../knownonly_v1_cache/.
SEED-0 SMOKE of v2 (2026-10-11): never-heard end 0.710 vs B 0.766 (-0.056), once all 0.324 vs 0.361, chain 0.33-0.87 in
sessions 1-6 (B 1.00) -- against the prediction. Suspected route (recorded, not yet tested): test items whose ANSWER word
was never heard cannot be answered under "+knownonly" (a baby cannot name what it never heard; the old body could pick
it). Added before the 30-seed run (record only): K4 the share of never-heard / chain items whose answer word is not yet
heard, and K5 THE FAIR COMPARISON -- accuracy on the items whose answer word WAS heard ("never-heard answerable", "chain
answerable"), knownonly vs Br = B+"+knownrec" (the same life as B, records added only; checked equal to B's cache on
B's keys at seed 0). Prediction: the loss disappears on answerable items (K5 ~0 or up). The smoke's cache was moved (the
record changed) to ../knownonly_v2_cache/.
K4/K5 SEED-0 CHECK (2026-10-11, check_known3.out; Br = B's cache exactly on B's keys): the chain loss is ALL unheard
answers (chain answerable 1.000 = 1.000; 67% -> 0% of chain items had an unheard answer in sessions 1-7); the never-heard
loss is NOT (0% of its items have an unheard answer; answerable 0.710 vs 0.766) -- "+knownonly" changes learning itself.
Candidate routes (to be told apart after the 30 seeds): (a) fewer negatives -- the never-heard words, pushed away as wrong
answers, spread the body's vector space (as more negatives help contrastive learning); (b) the scale -- standardising over
fewer words changes the habit's size against the other parts (templates, whole-sentence memory) added after it.
Conditions (30 seeds; B = both@ascii+rediscover+chain+wm+wmlast, finding 127's body, cached): B+"+knownonly" vs B.
Pre-registered (end = session 16, life = mean over sessions):
  K1 never-heard, knownonly - B: end, life (expected ~0 or a little up: no never-heard word answered).
  K2 once all (heard once), unseen all, older, newest: knownonly - B: end, life (either sign; recorded).
  K3 chain, story, describe: max |difference| and end sign test (expected ~0: answered by the world model mostly).
What this assumes: "heard" = in the learner's known set (heard sentences of the sessions so far; test-only words are not
heard and can never be answered); the mask is exact (-1e9 logits), no partial credit for unheard words; the world's word list
itself is still fixed at birth (growth proper is S2).
"""
import sys
from multiprocessing import Pool

import numpy as np

import run_long as RL
from run_count import sgn

B = "both@ascii+rediscover+chain+wm+wmlast"
K = B + "+knownonly"
BR = B + "+knownrec"                                # B with the K4 / K5 records only (no mask)
KEYS = ("never-heard", "once all", "unseen all", "older", "newest", "chain", "story", "describe",
        "unheard answer never-heard", "unheard answer chain", "never-heard answerable", "chain answerable")

if __name__ == "__main__":
    n_seeds = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    workers = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    tasks = [(sd, c) for c in (K, B, BR) for sd in range(n_seeds)]
    with Pool(workers) as p:
        out = dict(zip(tasks, p.map(RL.one, tasks)))
    print(f"=== {n_seeds} seeds; D26 (1) S1 only heard words are answer candidates ===")
    g = lambda c, k: np.array([[st.get(k, np.nan) for st in out[(sd, c)]] for sd in range(n_seeds)])
    for k in KEYS:
        for tag, c in (("knownonly", K), ("B", B), ("Br", BR)):
            print(f"  {k:12} {tag:9} " + " ".join("  -  " if np.isnan(x) else f"{x:.3f}" for x in np.nanmean(g(c, k), 0)))
    for k in ("never-heard answerable", "chain answerable"):
        d = g(K, k) - g(BR, k)
        print(f"  K5 {k:22} knownonly - Br: end {sgn(d[:, -1])} | life {sgn(np.nanmean(d, 1))}")
    for k in KEYS[:8]:
        d = g(K, k) - g(B, k)
        lab = "K1" if k == "never-heard" else "K3" if k in ("chain", "story", "describe") else "K2"
        print(f"  {lab} {k:12} knownonly - B: end {sgn(d[:, -1])} | life {sgn(np.nanmean(d, 1))} | max |diff| {np.nanmax(np.abs(d)):.3f}")
