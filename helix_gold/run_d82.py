"""D82 first step (2026-10-22; D82_DESIGN.md; the user's idea "많은 인간이 스스로 모른다는것을 몰라서 학습에 영향이 가고 학습
방법도 달라진다" -> D82, and the decision "응, 그럼 너의 추천대로 하자."). Written and fixed BEFORE any life of this test.

Question: does Voyager know what it does not know? The four D72 subjects outside the design were never learned (findings
162-165, 173); the in-design control was learned from the first hearing (164). If the learner's own scores carry a sense
of not knowing, its confidence is lower where it fails.
Nothing in the learner changes. "+d82" only records, for every D72 item, the confidence of the answer -- the top
candidate's share when the item's four candidates' scores are normalised among themselves (0.25 = all four tie, the body
abstains) -- and whether it was right (run_long.d72_keys, conf=True). The same conditions and seeds as the stored lives:
  CAT   both@ascii@d72cat+rediscover+chain+wm+wmctx+wmlast+d82        subject 1 (new category word), outside
  NEG   both@ascii@d72neg+...+d82                                     subject 2 (negation marker), outside
  ICE   both@ascii@d72ice+...+d82                                     subject 3 (slippery floor), outside
  NEXT  both@ascii@d72next+...+epi+d82                                subject 4 (next moment after a bell), outside
  CHAIN both@ascii@d72chain+...+d82                                   two new verbs chained, in design (its "d72 verb"
                                                                      one-step items: the body's own, reported)
seeds 0-9 (50 lives).
Pre-registered:
  R0 the regression: every "d72 ..." key of every session equals the stored life's (the same seed, the condition without
     "+d82") -- otherwise no table (the flag would not be record-only).
  At the last session (END), per kind: accuracy, mean confidence, overconfidence = confidence - accuracy, the share at
  0.25 (abstaining), and AUROC (items pooled over seeds; how well confidence tells right from wrong).
  K1 between subjects: per seed, the mean END confidence over the four outside subjects' items minus CHAIN's (chain items)
     -- decided when the sign test gives p < 0.005 (10:0).
  K2 between items: AUROC over the outside subjects' END items pooled (seeds and subjects) >= 0.60.
  K3 calibration: pooled outside overconfidence <= +0.10.
  Verdict: "모름을 앎" = K1 decided with mean <= -0.10 AND K3; "모름을 모름" = K1 not decided negative AND outside
  overconfidence > +0.25; otherwise "일부" (which of K1-K3 held is named).
  Report: confidence and accuracy by hearings k (outside subjects; the fluency illusion would be confidence rising with
  hearing while accuracy does not), per kind; CHAIN's own calibration.
  Expected before any life: NEXT and CAT's word items abstain often (confidence 0.25 -- a mark of not knowing that the
  rules score as wrong); ICE answers the word's usual move with confidence (a lure) -- confident errors.
What it cannot show: a confidence read from the scores, not a part that uses it (the next step, if any, is the user's);
Voyager's own world only.
    python3 run_d82.py [N_SEEDS=10] [WORKERS=3]"""
import json
import os
import pickle
import sys

import numpy as np

import run_long as RL
from run_d72 import BUCKETS, sign
from run_d72b import CHAIN, NEG
from run_d72 import CAT
from run_d72c import ICE
from run_d72d import NEXT

PLAIN = {"cat": CAT, "neg": NEG, "ice": ICE, "next": NEXT, "chain": CHAIN}
CONDS = {k: c + "+d82" for k, c in PLAIN.items()}
OUTSIDE = {"cat": ("word", "pic"), "neg": ("neg",), "ice": ("ice",), "next": ("next",)}


def same(a, b):
    if isinstance(a, float) and isinstance(b, float) and np.isnan(a) and np.isnan(b):
        return True
    return a == b


def auroc(pairs):
    """P(confidence of a right item > that of a wrong one), ties half."""
    r = np.array([c for c, ok in pairs if ok > 0.5])
    w = np.array([c for c, ok in pairs if ok <= 0.5])
    if len(r) == 0 or len(w) == 0:
        return float("nan")
    gt = (r[:, None] > w[None, :]).mean()
    eq = (r[:, None] == w[None, :]).mean()
    return float(gt + 0.5 * eq)


def end_pairs(life, kinds):
    out = []
    for k, v in life[-1].items():
        if k.startswith("d82 d72 ") and k.endswith(" pairs") and k.split()[2] in kinds:
            out += v
    return out


def k_pairs(life, kinds):
    """(k, confidence, right) over every session: k = the times the item's word was heard so far."""
    out = []
    for s in life:
        for k, v in s.items():
            if k.startswith("d82 d72 ") and k.endswith(" pairs") and k.split()[2] in kinds:
                j = k.split()[3]
                kk = s.get(f"d72 k{j}", np.nan) if j.isdigit() else np.nan
                out += [(kk, c, ok) for c, ok in v]
    return out


if __name__ == "__main__":
    n_seeds = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    workers = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    seeds = list(range(n_seeds))
    tasks = [(sd, c) for c in CONDS.values() for sd in seeds]
    res = RL.map_lives(tasks, workers)
    if any(r is None for r in res):
        print("PIECE DEADLINE: some lives not finished -- no table", flush=True)
        sys.exit(0)
    out = dict(zip(tasks, res))
    R = {k: [out[(sd, c)] for sd in seeds] for k, c in CONDS.items()}
    # R0: record only
    bad = []
    for k, c in PLAIN.items():
        stored = []
        for sd in seeds:                            # the stored life itself, never a new one
            f = os.path.join(RL.CACHE, f"seed{sd}_{c}.pkl")
            stored.append(pickle.load(open(f, "rb")) if os.path.exists(f) else [])
        for sd, (a, b) in enumerate(zip(R[k], stored)):
            if len(a) != len(b):
                bad.append((k, sd, "sessions"))
                continue
            for i, (sa, sb) in enumerate(zip(a, b)):
                for key in set(sa) | set(sb):
                    if key.startswith("d72 ") and not same(sa.get(key, "missing"), sb.get(key, "missing")):
                        bad.append((k, sd, i, key))
    print(f"=== D82 first step: Voyager's confidence on the D72 subjects, {n_seeds} seeds ===")
    print(f"R0 record only (every d72 key equal to the stored life): {'PASS' if not bad else 'FAIL ' + str(bad[:6])}")
    if bad:
        sys.exit("R0 failed -- no table")
    print("END per kind: accuracy | confidence | overconfidence | share at 0.25 | AUROC (pooled items)")
    for sub, kinds in list(OUTSIDE.items()) + [("chain", ("chain", "verb"))]:
        for kind in kinds:
            pr = [p for life in R[sub] for p in end_pairs(life, (kind,))]
            if not pr:
                continue
            c = np.array([a for a, _ in pr]); ok = np.array([b for _, b in pr])
            print(f"  {sub:5s} {kind:5s}: {ok.mean():.3f} | {c.mean():.3f} | {c.mean() - ok.mean():+.3f} | "
                  f"{np.mean(np.isclose(c, 0.25)):.3f} | {auroc(pr):.3f}  (n {len(pr)})")
    k1 = []
    for i in range(n_seeds):
        o = [a for sub, kinds in OUTSIDE.items() for a, _ in end_pairs(R[sub][i], kinds)]
        ch = [a for a, _ in end_pairs(R["chain"][i], ("chain",))]
        k1.append(np.mean(o) - np.mean(ch))
    m1, pos, neg, p1 = sign(k1)
    pooled = [p for sub, kinds in OUTSIDE.items() for life in R[sub] for p in end_pairs(life, kinds)]
    pc = np.array([a for a, _ in pooled]); pk = np.array([b for _, b in pooled])
    over = pc.mean() - pk.mean()
    au = auroc(pooled)
    K1 = p1 < 0.005 and m1 <= -0.10
    K2 = au >= 0.60
    K3 = over <= 0.10
    print(f"K1 outside - in-design confidence (per seed): {m1:+.3f} ({pos}:{neg}, p = {p1:.4f}) -> {'held' if K1 else 'not'}")
    print(f"K2 AUROC outside (pooled {len(pooled)} items): {au:.3f} -> {'held' if K2 else 'not'}")
    print(f"K3 outside overconfidence: confidence {pc.mean():.3f} - accuracy {pk.mean():.3f} = {over:+.3f} -> "
          f"{'held' if K3 else 'not'}")
    label = lambda lo, hi: f"{lo}" if lo == hi else f"{lo}+" if hi > 100 else f"{lo}-{hi}"
    for sub, kinds in OUTSIDE.items():
        kp = [x for life in R[sub] for x in k_pairs(life, kinds)]
        row = []
        for lo, hi in BUCKETS:
            v = [(c, ok) for kk, c, ok in kp if lo <= kk <= hi]
            row.append(f"k {label(lo, hi)}: {np.mean([c for c, _ in v]):.2f}/{np.mean([ok for _, ok in v]):.2f}" if v
                       else f"k {label(lo, hi)}: -")
        print(f"  by hearings, {sub} (confidence/accuracy): " + " | ".join(row))
    if K1 and K3:
        verdict = "모름을 앎"
    elif not (p1 < 0.005 and m1 < 0) and over > 0.25:
        verdict = "모름을 모름"
    else:
        verdict = "일부 (" + ", ".join(n for n, ok in (("K1", K1), ("K2", K2), ("K3", K3)) if ok) + " held)"
    print(f"DECISION: {verdict}")
    cpu = []
    for c in CONDS.values():
        for sd in seeds:
            f = os.path.join(RL.CACHE + "_cpu", f"seed{sd}_{c}.json")
            if os.path.exists(f):
                cpu.append(json.load(open(f)).get("cpu_cum_s", [0])[-1] / 60)
    if cpu:
        print(f"cost: CPU per life {np.mean(cpu):.1f} min")
    print("exit 0")
