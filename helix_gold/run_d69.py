"""D69 (2026-10-16): the self-trust gate -- can each part find out by itself whether it belongs in the world?

The user, after finding 159 (every part on harms where the world has no items for a part): "인간의 경우와는 다르구나. 이건
인간쪽이 더 범용 구조라는 건가. 다 켜기는 폐기하자. 대신 너가 낸 새 연구 후보로 한번 실험을 해보는게 좋을거 같아."
Written and fixed BEFORE any life of this test (the seed-0 smoke reads the gate's records only, no outcome key).

The gate ("+gate"; worldmodel.GATED, grade_parts, part_ok): the tracker's trust rule widened to every part. In each session
the learner's prepare() predicts every NEW sentence before learning it (context = all but the last element, answer = the
last element), by the world model as it speaks now and by the same with the part added (a quiet part) or taken out (a
speaking part); where the two differ, the part's record counts whether its prediction was the heard one. A part speaks when
(right + 1) / (differed + 2) > 1/2; no record = quiet. Every other part of the brain is unchanged; "+gate" off = the code
before (regression check_d69_reg.out).
GATE EDITION 2 (2026-10-16 ~15:20Z, BEFORE ANY GATED LIFE -- only FIT lives without the gate had started): edition 1 judged
each part against every other part on. The mechanism peek (peek_d69_ed1.out: the @ten ALL+gate seed-0 life cut after
session 8, records only) showed that parts which give the same answer on heard stories (the particles and the tracker)
then never differ, so 5 of the 6 FIT parts had no record and stayed quiet -- the test would have measured that flaw, not
the principle. Edition 2 judges a part against the brain as it speaks now (see above); a part that needs another
(counting needs the tracker) is judged once the other speaks. The readings, the rule and the conditions are unchanged.
GATE EDITION 3 (same day, still BEFORE ANY GATED LIFE): the edition-2 peek showed the same empty records; diag_d69.py found
why -- predicting a session's sentences BEFORE it is learned can never answer its stories, because a story room's things
are named in the same session. So the sentences are graded right AFTER the session is taken in, as the tracker's own trust
is (heard stories predicted from the story alone, then compared). On the state after session 4, grading the heard stories
gave the tracker 30 / 30 and the particles 10 / 29 (diag only). Readings, rule and conditions unchanged.

Worlds and conditions (run_w7pilot.py's FIT rule and part lists, unchanged; seeds 0-9):
  F   = FIT                    (seeds 0-2 cached by the pilot)
  FG  = FIT + "+gate"          -- does the gate cost anything where every part fits?
  AG  = ALL + "+gate"          -- does the gate find the fitting parts by itself?
  A   = ALL (no gate)          reference only, the pilot's seeds 0-2 (cached; "다 켜기는 폐기" -- not run further)
  worlds "@ten" (W5 numbers) and "@sp" (W3 spatial words), as the pilot.
Readings (pre-registered; "decided" = sign test p < 0.005 over the 10 seeds, i.e. 10:0, with |mean| >= 0.02):
  G1 (main) per world, the pilot's R1 keys (MAIN + COMMON prefixes, body keys excluded): AG - F at the end per seed.
     Verdict "회복" (the gate recovers FIT by itself) when no R1 key is decided worse; else "아직" with the keys listed.
     Also: R1 keys decided BETTER under AG (another part helping once it is gated -- the user's first question again);
     and on seeds 0-2, for the keys the pilot found harmed by ALL, the recovered share (AG - A) / (F - A), median (report).
  G2 per world, the same keys: FG - F. Verdict "해 없음" when no R1 key is decided worse; else "손해" with the keys.
  G3 the mechanism, per world and condition (AG, FG): for every part, the share of seeds in which it speaks at the end, the
     median session it first speaks, its end record (differed, right; medians). Expected (written before): in AG the FIT
     parts speak and the others stay quiet; reported as two counts -- FIT parts quiet at the end / non-FIT parts speaking.
  G4 every numeric key both carry: how many move decided (10:0, |mean| >= 0.02) with AG - F and FG - F, each direction.
  G5 cost: AG / F and FG / F in reference-job units (D51), peak memory.
Decision (fixed here): G1 "회복" and G2 "해 없음" in both worlds -> the gate is written as a finding and proposed to the
user as a default for every life (the human choice "which parts" moves into learning -- D67's learning share, D69); G1
"아직" -> the finding says which parts the gate keeps wrongly (G3) and why, nothing becomes default. A key with |mean| >=
0.02 but not decided is reported as undecided; 30 seeds only if such a key alone stands between the verdicts.
What this assumes: the gate's rule above (Laplace 1/2, the tracker's); every part judged on top of all the others that
are on (not in combinations); heard sentences end with what they tell (the last element) -- a part that never changes a
prediction of this life's own sentences stays quiet even if it would help on a test item (that is G2's risk).
    python3 run_d69.py [N_SEEDS=10] [WORKERS=4]
"""
import json
import os
import pickle
import sys

import numpy as np

import run_long as RL
import run_w7pilot as P
import worldmodel as WM
from run_count import sign_p

COND = {"F": lambda w_: P.cond(w_, "FIT"), "FG": lambda w_: P.cond(w_, "FIT") + "+gate",
        "AG": lambda w_: P.cond(w_, "ALL") + "+gate", "A": lambda w_: P.cond(w_, "ALL")}
NUM = (int, float, np.integer, np.floating)


def end(stats, k):
    v = stats[-1].get(k, np.nan) if stats else np.nan
    return float(v) if isinstance(v, NUM) else np.nan


def decided(d):
    d = np.asarray(d, dtype=float)
    d = d[~np.isnan(d)]
    if len(d) == 0:
        return 0, np.nan, np.nan
    p = sign_p(int((d > 1e-12).sum()), int((d < -1e-12).sum()))
    m = float(d.mean())
    return (1 if m > 0 else -1) if (p < 0.005 and abs(m) >= 0.02) else 0, m, p


def r1_keys(world, stats):
    return sorted(k for k in stats[-1] if isinstance(k, str) and isinstance(stats[-1][k], NUM) and "body" not in k
                  and (k.startswith(P.MAIN[world]) or k.startswith(P.COMMON)))


def units(c, sd):
    f = os.path.join(RL.CACHE + "_cpu", f"seed{sd}_{c}.json")
    if not os.path.exists(f):
        return np.nan, np.nan
    d = json.load(open(f))
    return d["cpu_cum_s"][-1] / d["ref_cpu_s"], d.get("peak_rss_gb", np.nan)


if __name__ == "__main__":
    n_seeds = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    workers = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    bad = P.flag_check()
    assert not bad, f"flag collision: {bad}"
    for w_ in P.FIT:
        for x in ("FG", "AG"):
            assert COND[x](w_).count("+gate") == 1 and "+gate" not in COND["F"](w_) + COND["A"](w_)
    seeds = list(range(n_seeds))
    tasks = [(sd, COND[x](w_)) for w_ in P.FIT for x in ("F", "FG", "AG") for sd in seeds]
    out = dict(zip(tasks, RL.map_lives(tasks, workers)))
    print(f"=== D69 self-trust gate: FIT / FIT+gate / ALL+gate, {n_seeds} seeds (development test) ===")
    print("flag check: no collision")
    for w_ in P.FIT:
        S = {x: [out[(sd, COND[x](w_))] for sd in seeds] for x in ("F", "FG", "AG")}
        keys = r1_keys(w_, S["F"][0])
        print(f"--- @{w_}: FIT {P.FIT[w_]} | ALL adds {P.EXTRA[w_]}")
        verdict = {}
        for x, name in (("AG", "G1 ALL+gate - FIT"), ("FG", "G2 FIT+gate - FIT")):
            worse, better, undecided = [], [], []
            print(f"  {name} (R1 keys, end; mean F / mean {x} / mean diff / sign p)")
            for k in keys:
                d = [end(S[x][i], k) - end(S["F"][i], k) for i in range(n_seeds)]
                dec, m, p = decided(d)
                print(f"    {k:36s} {np.nanmean([end(s, k) for s in S['F']]):.3f} "
                      f"{np.nanmean([end(s, k) for s in S[x]]):.3f} {m:+.3f} p={p:.3f}" + ("  WORSE" if dec < 0 else
                                                                                         "  BETTER" if dec > 0 else ""))
                (worse if dec < 0 else better if dec > 0 else undecided if abs(m) >= 0.02 else []).append(k)
            ok = not worse
            verdict[x] = ok
            lab = ("회복" if ok else "아직") if x == "AG" else ("해 없음" if ok else "손해")
            print(f"  {name}: {lab} -- decided worse {worse if worse else 'none'}; decided better {better if better else 'none'};"
                  f" undecided |mean| >= 0.02: {undecided if undecided else 'none'}")
        ref = [sd for sd in range(3) if os.path.exists(os.path.join(RL.CACHE, f"seed{sd}_{COND['A'](w_)}.pkl"))]
        if ref:
            A = {sd: pickle.load(open(os.path.join(RL.CACHE, f"seed{sd}_{COND['A'](w_)}.pkl"), "rb")) for sd in ref}
            shares = []
            for k in keys:
                dA = [end(A[sd], k) - end(S["F"][sd], k) for sd in ref]
                if all(x_ < 0 for x_ in dA) and abs(np.mean(dA)) >= 0.02:
                    for sd in ref:
                        gap = end(S["F"][sd], k) - end(A[sd], k)
                        if gap > 1e-9:
                            shares.append((end(S["AG"][sd], k) - end(A[sd], k)) / gap)
            print(f"  G1 recovered share on the pilot's harmed R1 keys (seeds {ref}): median "
                  f"{np.median(shares) if shares else np.nan:.2f} over {len(shares)} key-seeds")
        for x in ("AG", "FG"):
            fit = set(P.FIT[w_].strip("+").split("+"))
            q_fit, s_non = [], []
            print(f"  G3 {x}: part  speaks-at-end share | first session it speaks (median) | end record differed / right")
            for p_, _ in WM.GATED:
                if f"gate {p_} speaks" not in S[x][0][-1]:
                    continue
                sp = [s[-1].get(f"gate {p_} speaks", np.nan) for s in S[x]]
                first = [next((i + 1 for i, st in enumerate(s) if st.get(f"gate {p_} speaks") == 1.0), np.nan) for s in S[x]]
                dif = [s[-1].get(f"gate {p_} differed", np.nan) for s in S[x]]
                rt = [s[-1].get(f"gate {p_} right", np.nan) for s in S[x]]
                share = float(np.nanmean(sp))
                print(f"    {p_:8s} {'FIT' if p_ in fit else 'non':3s} {share:.2f} | {np.nanmedian(first):.0f} | "
                      f"{np.nanmedian(dif):.0f} / {np.nanmedian(rt):.0f}")
                if p_ in fit and share < 0.5:
                    q_fit.append(p_)
                if p_ not in fit and share >= 0.5:
                    s_non.append(p_)
            print(f"  G3 {x}: FIT parts quiet at the end (in most seeds): {q_fit if q_fit else 'none'}; "
                  f"non-FIT parts speaking: {s_non if s_non else 'none'}")
        all_keys = sorted(k for k in S["F"][0][-1] if isinstance(k, str) and isinstance(S["F"][0][-1][k], NUM)
                          and not k.startswith("gate "))
        for x in ("AG", "FG"):
            mv = [decided([end(S[x][i], k) - end(S["F"][i], k) for i in range(n_seeds)])[0] for k in all_keys]
            print(f"  G4 {x} - F over {len(all_keys)} keys: decided up {sum(1 for v in mv if v > 0)}, "
                  f"down {sum(1 for v in mv if v < 0)}")
        uf = [units(COND["F"](w_), sd) for sd in seeds]
        for x in ("AG", "FG"):
            ux = [units(COND[x](w_), sd) for sd in seeds]
            print(f"  G5 cost {x} / F units {np.nanmean([u[0] for u in ux]) / np.nanmean([u[0] for u in uf]):.2f}; "
                  f"peak GB F {np.nanmax([u[1] for u in uf]):.2f}, {x} {np.nanmax([u[1] for u in ux]):.2f}")
        print(f"  DECISION @{w_}: G1 {'회복' if verdict['AG'] else '아직'}, G2 {'해 없음' if verdict['FG'] else '손해'}")
