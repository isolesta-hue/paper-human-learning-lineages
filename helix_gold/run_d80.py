"""D80 v1 -- experience that holds order (D80_DESIGN.md; the user's question and decisions 10-21: "응, 기록하고 D79 결과 보고
진행해줘.", "응, 둘다 너의 추천대로 해줘." -- the next D79 attempt is (다) here). Written and fixed BEFORE the seed-0 smoke.
Run first: check_seq_world.py (every old world unchanged, 36 flags x seeds 0-2; "@seq" = "@scene" + the start scene + the
question moments + tests, every question moment and test true from what was seen -- PASS) and the criterion-0 gate on
"@seq" (elements_10_seq.out: PASS 0.9998).

Condition: both@ascii@seq+rediscover, seeds 0-9 (the body without designed parts, as the exam's). The episode store and
the three answer rules act only when answering (run_long.EpisodeStore, seq_keys): the same body answers every item three
ways -- U (usual), E ((가)(나): the last value of the question's picture's time line), EQ ((가)(나)(다): the time relation
the D79 equals sign learned for the question's word set; none learned: usual).
Measures per seed: "seq now" / "seq before" right after the room arrived (" new"), items pooled over the sessions by count;
the old keys at the last session (" all"): motion tests (R*, once, unseen), names (name*), scene tests (four-way).
Pre-registered (D80_DESIGN.md):
  R1 (가)(나) "now": E >= 0.80 (mean over seeds) AND E - U sign test over seeds p < 0.005 -> "(가)(나) 성공", else "실패".
  R2 (다) "before": EQ >= 0.80 AND EQ - E p < 0.005 -> R2 holds.
  R3 old keys: a DROP = rule - U <= -0.05 (mean) with p < 0.005. EQ's drop refutes (다); E's drop is reported (expected on
     the scene tests: recency without the question's meaning spoils a hypothetical question).
  Verdict: (가)(나) by R1; (다) "성공" = R2 holds and no EQ drop, else "실패" -> D79 is put aside (the user's decision).
What it assumes / cannot show: the stamp is the order index (an approximation of a slowly drifting context); the three time
relations are designed (what is learned is which relation a question word means); only SEEN moments enter a time line --
moves told only in words (the exam's C8) are v2's (sentence -> expected scene, D75's scene part); Voyager's own world only.
    python3 run_d80.py [n_seeds=10] [workers=3]
"""
import sys
from multiprocessing import Pool

import numpy as np

import run_long as RL
from run_count import sgn
from run_nwf import binom_p

COND = "both@ascii@seq+rediscover"
RULES = ("U", "E", "EQ")


def pooled(lf, key, rule, tag=" new"):
    num = den = 0.0
    for st in lf:
        n = st.get(f"{key} n{tag}", 0.0)
        a = st.get(f"{key} {rule}{tag}", np.nan)
        if n and a == a:
            num += a * n
            den += n
    return num / den if den else np.nan


def last(lf, key, rule):
    return lf[-1].get(f"{key} {rule} all", np.nan)


def test(d):
    d = np.asarray(d, dtype=float)
    d = d[~np.isnan(d)]
    w_, l_ = int((d > 1e-12).sum()), int((d < -1e-12).sum())
    return d.mean() if len(d) else np.nan, binom_p(w_, l_)


if __name__ == "__main__":
    n_seeds = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    workers = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    seeds = list(range(n_seeds))
    with Pool(workers) as p:
        lives = p.map(RL.one, [(sd, COND) for sd in seeds], chunksize=1)
    if any(lf is None for lf in lives):
        print("PIECE DEADLINE: some lives not finished -- no table", flush=True)
        sys.exit(0)
    print(f"=== D80 v1, experience that holds order: \"@seq\", {n_seeds} seeds, {COND}; U usual / E (가)(나) / EQ (가)(나)(다) ===")
    v = {(k, r): np.array([pooled(lf, k, r) for lf in lives]) for k in ("seq now", "seq before") for r in RULES}
    for k in ("seq now", "seq before"):
        print(f"  {k} (right after the room arrived): " + " / ".join(f"{r} {v[(k, r)].mean():.3f}" for r in RULES)
              + f"; E - U {sgn(v[(k, 'E')] - v[(k, 'U')])}; EQ - U {sgn(v[(k, 'EQ')] - v[(k, 'U')])}; "
              f"EQ - E {sgn(v[(k, 'EQ')] - v[(k, 'E')])}")
    rel = {}
    for lf in lives:
        for k_, x in lf[-1].items():
            if k_.startswith("seq rel "):
                rel.setdefault(k_[8:], []).append(("last", "before", "first")[int(x)])
    print("  learned meanings at the last session (word set: relation per seed): " +
          "; ".join(f"{k_}: {sorted(set(x))} ({len(x)} seeds)" for k_, x in sorted(rel.items())))
    print("  R3 old keys at the last session (U / E / EQ):")
    drops = {"E": [], "EQ": []}
    for k in ("old motion", "old name", "old scene"):
        u = np.array([last(lf, k, "U") for lf in lives])
        line = f"    {k}: U {np.nanmean(u):.3f}"
        for r in ("E", "EQ"):
            x = np.array([last(lf, k, r) for lf in lives])
            m, p_ = test(x - u)
            drop = m <= -0.05 and p_ < 0.005
            drops[r].append(drop)
            line += f" | {r} {np.nanmean(x):.3f} ({sgn(x - u)}){' DROP' if drop else ''}"
        print(line)
    e_now, u_now = v[("seq now", "E")], v[("seq now", "U")]
    m1, p1 = test(e_now - u_now)
    r1 = e_now.mean() >= 0.80 and m1 > 0 and p1 < 0.005
    eq_b, e_b = v[("seq before", "EQ")], v[("seq before", "E")]
    m2, p2 = test(eq_b - e_b)
    r2 = eq_b.mean() >= 0.80 and m2 > 0 and p2 < 0.005
    da = r2 and not any(drops["EQ"])
    print(f"  R1 now: E {e_now.mean():.3f}, E - U {m1:+.3f} p = {p1:.4f} -> (가)(나) {'성공' if r1 else '실패'}")
    print(f"  R2 before: EQ {eq_b.mean():.3f}, EQ - E {m2:+.3f} p = {p2:.4f} -> {'holds' if r2 else 'does not hold'}; "
          f"EQ old-key drops {sum(drops['EQ'])}, E old-key drops {sum(drops['E'])} (reported) -> (다) {'성공' if da else '실패 -> D79 접어 둠'}")
