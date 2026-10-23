"""D83 (1) (2026-10-23): a part speaks where it has been useful -- can the parts learn WHERE to speak by themselves?

The user (10-22): "이 인간몫 부품이나 구조라는건 일종의 하네스나 스킬이나 에이전트인건가? 그렇다면 필요한 하네스나 스킬이나 에이전트를
만들고 사용하는 llm의 기술에서 무언가 아이디어를 얻을수도 있을꺼 같은데." -> recorded; order (10-23) "너가 제시한 의견대로 진행하면
될거같아" = (1) first. Design and pre-registration: D83_DESIGN.md. Written and fixed BEFORE any life of this test (the seed-0 peeks
read the records only, no outcome key).

The gate ("+useful"; worldmodel.grade_use, use_say, _use_term): Toolformer's rule (Schick et al. 2023) per question form (the set
of element kinds a question holds). After each session is taken in, every new sentence is predicted (all but the last element ->
the last element) by the world model as it speaks now in that sentence's form, with each part's say added and taken out; +1 right,
0 silent, -1 wrong over the cells the kind gate lets answer -- every sentence heard so far, read again each session (edition 3,
D83_DESIGN.md). A part speaks in a form where its net is > 0; where it made no difference the last decision stands; a form never
decided falls back to the all-forms decision; none = quiet.
References: "+usequiet" (records kept, nobody speaks), "+useshuf" (in each form a part reads another part's record).

Conditions (run_w7pilot.py's FIT / ALL rule and part lists, unchanged; worlds "@ten" and "@sp"):
  F   = FIT                        seeds 0-9 (cached, D69)
  AG  = ALL + "+gate"              seeds 0-9 (cached, D69) -- compared only
  A   = ALL                        seeds 0-2 (cached, finding 159) -- reference
  AU  = ALL + "+useful"            seeds 0-9  NEW
  FU  = FIT + "+useful"            seeds 0-9  NEW
  QU  = ALL + "+useful+usequiet"   seeds 0-2  NEW (simple reference: always quiet)
  SU  = ALL + "+useful+useshuf"    seeds 0-2  NEW (shuffle control)
Readings (pre-registered; "decided" = sign test p < 0.005 over the 10 seeds, i.e. 10:0, with |mean| >= 0.02):
  U1 (main) per world, R1 keys (the pilot's MAIN + COMMON prefixes, body keys excluded): AU - F at the end. "회복" when no R1 key
     is decided worse; else "아직" with the keys. Keys decided better are listed too.
  U2 per world, the same keys: FU - F. "해 없음" when none decided worse; else "손해".
  U3 mechanism, AU and FU: per part -- forms with a record, forms it speaks in, all-forms net / changed, answer rows it spoke in;
     the share of decisions made by the all-forms record. Two counts: FIT parts that never spoke / non-FIT parts that spoke.
     The story keys D69 broke ("story", "story wm fired", "story wm right"): F / AG / AU side by side.
  U4 every numeric key both carry: decided moves AU - F and FU - F, each direction.
  U5 cost AU / F and FU / F (reference-job units, D51), peak memory.
  REF (rule 1; seeds 0-2, direction only): on the R1 keys A harmed in the pilot plus D69's story keys -- for AU, QU, SU: how many
     keys sit 3:0 below F with mean <= -0.02. If AU is "회복" but QU or SU also has none, the finding does not say the records
     decide when to speak.
Decision (fixed here): U1 "회복" and U2 "해 없음" in both worlds -> written as a finding and proposed to the user as a default;
else the finding says which parts the gate keeps or quiets wrongly (U3) and why; nothing becomes default. A key with |mean| >=
0.02 but not decided is reported as undecided; 30 seeds only if such a key alone stands between the verdicts.
    python3 run_d83.py [N_SEEDS=10] [WORKERS=3]
"""
import os
import pickle
import re
import sys

import numpy as np

import run_d69 as D69
import run_long as RL
import run_w7pilot as P
import worldmodel as WM

COND = {"F": lambda w_: P.cond(w_, "FIT"), "AG": lambda w_: P.cond(w_, "ALL") + "+gate", "A": lambda w_: P.cond(w_, "ALL"),
        "AU": lambda w_: P.cond(w_, "ALL") + "+useful", "FU": lambda w_: P.cond(w_, "FIT") + "+useful",
        "QU": lambda w_: P.cond(w_, "ALL") + "+useful+usequiet", "SU": lambda w_: P.cond(w_, "ALL") + "+useful+useshuf"}
STORY = ("story", "story wm fired", "story wm right")
end, decided, r1_keys, units = D69.end, D69.decided, D69.r1_keys, D69.units


def flag_check():
    """rule 4: the new flags against every existing flag literal and against each other (substring both ways)."""
    lits = set()
    for f in ("run_long.py", "integrated.py", "worldmodel.py"):
        lits |= set(re.findall(r'"([+@\-][A-Za-z0-9_]+)" in cond', open(f).read()))
    new = ("+useful", "+usequiet", "+useshuf")
    return [(a, b) for a in new for b in lits | set(new) if a != b and (a in b or b in a)] + P.flag_check()


if __name__ == "__main__":
    n_seeds = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    workers = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    bad = flag_check()
    assert not bad, f"flag collision: {bad}"
    seeds, ref_seeds = list(range(n_seeds)), [0, 1, 2]
    # interleaved so that a "@ten" life (the larger) runs next to a "@sp" life
    tasks = ([(sd, COND[x](w_)) for sd in seeds for x in ("AU", "FU") for w_ in P.FIT] +
             [(sd, COND[x](w_)) for sd in ref_seeds for x in ("QU", "SU") for w_ in P.FIT] +
             [(sd, COND[x](w_)) for x in ("F", "AG") for w_ in P.FIT for sd in seeds])
    out = dict(zip(tasks, RL.map_lives(tasks, workers)))
    print(f"=== D83 (1) usefulness gate: ALL+useful / FIT+useful vs FIT, {n_seeds} seeds (development test) ===")
    print("flag check: no collision")
    for w_ in P.FIT:
        S = {x: [out[(sd, COND[x](w_))] for sd in seeds] for x in ("F", "AG", "AU", "FU")}
        R = {x: {sd: out[(sd, COND[x](w_))] for sd in ref_seeds} for x in ("QU", "SU")}
        keys = r1_keys(w_, S["F"][0])
        fit = set(P.FIT[w_].strip("+").split("+"))
        print(f"--- @{w_}: FIT {P.FIT[w_]} | ALL adds {P.EXTRA[w_]}")
        verdict = {}
        for x, name in (("AU", "U1 ALL+useful - FIT"), ("FU", "U2 FIT+useful - FIT")):
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
            lab = ("회복" if ok else "아직") if x == "AU" else ("해 없음" if ok else "손해")
            print(f"  {name}: {lab} -- decided worse {worse if worse else 'none'}; decided better {better if better else 'none'};"
                  f" undecided |mean| >= 0.02: {undecided if undecided else 'none'}")
        for x in ("AU", "FU"):
            q_fit, s_non = [], []
            print(f"  U3 {x}: part  forms recorded | forms it speaks in | all-forms net / changed | answer rows spoken (medians)")
            for p_, _ in WM.GATED:
                if f"use {p_} net" not in S[x][0][-1]:
                    continue
                med = {f: np.nanmedian([s[-1].get(f"use {p_} {f}", np.nan) for s in S[x]])
                       for f in ("forms", "says in", "net", "changed", "said rows")}
                spoke = float(np.mean([s[-1].get(f"use {p_} said rows", 0) > 0 for s in S[x]]))
                print(f"    {p_:8s} {'FIT' if p_ in fit else 'non':3s} {med['forms']:.0f} | {med['says in']:.0f} | "
                      f"{med['net']:+.0f} / {med['changed']:.0f} | {med['said rows']:.0f} (spoke in {spoke:.0%} of seeds)")
                if p_ in fit and spoke < 0.5:
                    q_fit.append(p_)
                if p_ not in fit and spoke >= 0.5:
                    s_non.append(p_)
            bo = np.nanmedian([s[-1].get("use backoff share", np.nan) for s in S[x]])
            print(f"  U3 {x}: FIT parts that never spoke (most seeds): {q_fit if q_fit else 'none'}; non-FIT parts that spoke: "
                  f"{s_non if s_non else 'none'}; decisions by the all-forms record {bo:.3f}")
        print("  U3 story keys (end, mean): " + "; ".join(
            f"{k}: F {np.nanmean([end(s, k) for s in S['F']]):.3f} AG {np.nanmean([end(s, k) for s in S['AG']]):.3f} "
            f"AU {np.nanmean([end(s, k) for s in S['AU']]):.3f}" for k in STORY if k in S["F"][0][-1]))
        all_keys = sorted(k for k in S["F"][0][-1] if isinstance(k, str) and isinstance(S["F"][0][-1][k], D69.NUM)
                          and not k.startswith(("gate ", "use ")))
        for x in ("AU", "FU"):
            mv = [decided([end(S[x][i], k) - end(S["F"][i], k) for i in range(n_seeds)])[0] for k in all_keys]
            print(f"  U4 {x} - F over {len(all_keys)} keys: decided up {sum(1 for v in mv if v > 0)}, "
                  f"down {sum(1 for v in mv if v < 0)}")
        uf = [units(COND["F"](w_), sd) for sd in seeds]
        for x in ("AU", "FU"):
            ux = [units(COND[x](w_), sd) for sd in seeds]
            print(f"  U5 cost {x} / F units {np.nanmean([u[0] for u in ux]) / np.nanmean([u[0] for u in uf]):.2f}; "
                  f"peak GB F {np.nanmax([u[1] for u in uf]):.2f}, {x} {np.nanmax([u[1] for u in ux]):.2f}")
        refA = [sd for sd in ref_seeds if os.path.exists(os.path.join(RL.CACHE, f"seed{sd}_{COND['A'](w_)}.pkl"))]
        A = {sd: pickle.load(open(os.path.join(RL.CACHE, f"seed{sd}_{COND['A'](w_)}.pkl"), "rb")) for sd in refA}
        harmed = [k for k in keys if refA and all(end(A[sd], k) - end(S["F"][sd], k) < 0 for sd in refA)
                  and np.mean([end(A[sd], k) - end(S["F"][sd], k) for sd in refA]) <= -0.02]
        harmed += [k for k in STORY if k in keys and k not in harmed]
        cnt = {}
        for x, src in (("AU", {sd: S["AU"][sd] for sd in ref_seeds}), ("QU", R["QU"]), ("SU", R["SU"])):
            below = [k for k in harmed if all(end(src[sd], k) - end(S["F"][sd], k) < 0 for sd in ref_seeds)
                     and np.mean([end(src[sd], k) - end(S["F"][sd], k) for sd in ref_seeds]) <= -0.02]
            cnt[x] = below
            print(f"  REF {x} (seeds 0-2): of {len(harmed)} keys (A-harmed + story), 3:0 below F: {len(below)} {below}")
        records_decide = not (verdict["AU"] and (not cnt["QU"] or not cnt["SU"]))
        print(f"  DECISION @{w_}: U1 {'회복' if verdict['AU'] else '아직'}, U2 {'해 없음' if verdict['FU'] else '손해'}; "
              f"references {'leave the records deciding' if records_decide else 'match AU -- do not say the records decide'}")
