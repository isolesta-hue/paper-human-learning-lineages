"""D79 v1 -- the relational equals sign (the user's idea and decision 10-21: "응, 기록하고 한번 설계와 테스트를 해봤으면 좋겠어.
보편적이라기 보단 조금 특정 상황에 대한 최적화된 인간의 사고 로직으로 보이지만, 한번 살펴볼만해 보여."). Written and fixed
BEFORE any "+eq" life.

The rule (run_long.d79_logits; answer time only -- learning is untouched): for an item whose usual top answer is a word and
which shows a picture, the candidates are the words the usual answer allows (its kind gate and known mask: the frame decides
the KIND, Q14 requirement (2)), and among them the word x is chosen whose own memory accounts best for every picture seen,
R(x) = sum over the item's pictures e of log p(e | [x]) (left = the unknown x, right = what is seen; "=" = make them agree).
Every other item keeps the usual answer.

World and lives: "@nwf" (finding 168), P + "+eq" = both@ascii@nwf+rediscover+chain+wm+wmlast+eq, seeds 0-9. Because the rule
acts only when answering, an "+eq" life learns exactly as the stored P life of the same seed: every usual ("operational")
key of it must equal the stored life's (E0) -- and each item is answered both ways by the same body (paired).
Pre-registered:
  E0 regression: every key of the stored P life equals the same key of the "+eq" life (seeds 0-9). If not: no table.
  E1 the frame (requirement (2)): under the rule, D_eq = framed - bare (picture -> new name, the sessions right after
     exposure, items pooled, n = 1 / 2 / 4 / 8 pooled; as finding 168) -- "틀 영향 사라짐" if the mean over seeds of D_eq
     > -0.05. Report F_eq - F (the framed items, rule vs usual) and B_eq - B, sign tests over seeds.
  E2 one hearing: the bare picture -> new name heard ONCE under the rule, mean over seeds >= 0.40 (the line told to the user
     10-21; note the usual value here is finding 168's 0.092 -- the 0.26 of finding 149 pooled both directions, and the rule
     does not touch the name -> picture direction). Report B_eq(n) - B(n), sign tests.
  E3 old keys (side effects, same lives): the word-answer keys answered both ways at the last session -- "name{n} same"
     (the old-style name test, n pooled by the mean), "nwbc all" / "nwfc all" (the frame's own common words) -- a DROP if
     rule - usual <= -0.05 with a sign test p < 0.005 over seeds. (Items whose answer is a place or a picture are not touched
     by the rule: no side effect there by construction; E0 shows the learning is the same.)
  Verdict: "성공" = E1 and E2 hold and no E3 drop; "부분" = exactly one of E1 / E2 holds and no drop; "실패" = neither holds, or
     an E3 drop.
What this assumes: v1 = answer time only (no solving-and-recording yet -- the learning side of D79 is v2); the right side =
the pictures only (the "what" stream; the "where" stream does not name a thing); the sum of log-probabilities (independent
pictures); every allowed word is a candidate; a bag-of-moments body has no word order. What it cannot show: whether the rule
transfers to the exam form (the next sealed edition), or to function words (particles -- decided by the sentence, not the
scene: a known risk of this v1, to be checked before the rule enters the body).
AMENDED (2026-10-21 18:58 UTC, after the first read stopped at E0 -- no table had been printed): E0 compared values with ==,
which calls two NaN-filled dicts different ("curve once" {0: nan, ...}: 25 such, all of them); now a NaN-aware deep
comparison (NaN == NaN at any depth). Nothing else changed.
    python3 run_d79.py [n_seeds=10] [workers=3]
"""
import sys
from multiprocessing import Pool

import numpy as np

import run_long as RL
from run_count import sgn
from run_nwf import COND as P, NS, binom_p, pooled

E = P + "+eq"


def same(a, b):
    """equal at any depth, NaN == NaN (the E0 comparison, amended 18:58 UTC)."""
    if isinstance(a, float) and isinstance(b, float) and a != a and b != b:
        return True
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(same(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return len(a) == len(b) and all(same(x, y) for x, y in zip(a, b))
    return a == b


def ratio(lf, k, tag):
    n = pooled(lf, f"{k} n", tag)
    return (pooled(lf, f"{k} bare right", tag) / n, pooled(lf, f"{k} framed right", tag) / n) if n else (np.nan, np.nan)


def last(lf, k):
    return lf[-1].get(k, np.nan)


if __name__ == "__main__":
    n_seeds = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    workers = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    seeds = list(range(n_seeds))
    with Pool(workers) as p:
        lives = p.map(RL.one, [(sd, E) for sd in seeds], chunksize=1)
    if any(lf is None for lf in lives):
        print("PIECE DEADLINE: some lives not finished -- no table", flush=True)
        sys.exit(0)
    stored = [RL.one((sd, P)) for sd in seeds]     # cached (finding 168)
    print(f"=== D79 v1, the relational equals sign at answer time: \"@nwf\", {n_seeds} seeds, {E} ===")
    bad = []
    for sd, a, b in zip(seeds, stored, lives):
        for i, (sa, sb) in enumerate(zip(a, b)):
            for k, v in sa.items():
                vb = sb.get(k, "missing")
                if not same(v, vb):
                    bad.append((sd, i, k, v, vb))
        if len(a) != len(b):
            bad.append((sd, "sessions", len(a), len(b)))
    print(f"  E0 regression (every key of the stored P life == the +eq life): {'PASS' if not bad else 'FAIL'} "
          f"({len(bad)} differences{'; first ' + str(bad[:3]) if bad else ''})")
    if bad:
        sys.exit(0)
    B, F = np.array([ratio(lf, "nwfb", " new") for lf in lives]).T
    Be, Fe = np.array([ratio(lf, "nwfb", " new eq") for lf in lives]).T
    print("  picture -> new name, right after exposure (items pooled; usual / rule):")
    print(f"    bare {B.mean():.3f} / {Be.mean():.3f} (rule - usual {sgn(Be - B)}); framed {F.mean():.3f} / {Fe.mean():.3f} "
          f"(rule - usual {sgn(Fe - F)})")
    print(f"    framed - bare: usual {sgn(F - B)} | rule {sgn(Fe - Be)}")
    per_n = {}
    for n in NS:
        b, f = np.array([ratio(lf, f"nwfb{n}", " new") for lf in lives]).T
        be, fe = np.array([ratio(lf, f"nwfb{n}", " new eq") for lf in lives]).T
        per_n[n] = (b, f, be, fe)
        print(f"    n={n}: bare {b.mean():.3f} / {be.mean():.3f} ({sgn(be - b)}); framed {f.mean():.3f} / {fe.mean():.3f} "
              f"({sgn(fe - f)}); framed - bare under the rule {sgn(fe - be)}")
    bp = sum(pooled(lf, "nwfb b+f-", " new eq") for lf in lives)
    bm = sum(pooled(lf, "nwfb b-f+", " new eq") for lf in lives)
    fl = {k: sum(pooled(lf, f"nwfb flip {k}", " new eq") for lf in lives) for k in ("common", "frame", "name", "other")}
    er = {k: sum(pooled(lf, f"nwfb framed err {k}", " new eq") for lf in lives) for k in ("common", "frame", "name", "other")}
    print(f"    under the rule: flips bare right & framed wrong {bp:.0f}, the reverse {bm:.0f} (p = {binom_p(bp, bm):.3g}); "
          f"flip answers {fl}; all framed errors {er}")
    n_all = sum(pooled(lf, "nwfb n", " new") for lf in lives)
    for tag, nm in ((" new", "usual"), (" new eq", "rule")):
        print(f"    room-names reading ({nm}): bare {sum(pooled(lf, 'nwfb bare four', tag) for lf in lives) / n_all:.3f}, "
              f"framed {sum(pooled(lf, 'nwfb framed four', tag) for lf in lives) / n_all:.3f}")
    for n in NS:
        a_ = np.array([np.nanmean([st.get(f"nw{n} new", np.nan) for st in lf]) for lf in lives])
        b_ = np.array([np.nanmean([st.get(f"nw{n} new eq", np.nan) for st in lf]) for lf in lives])
        print(f"    nw{n} new (both directions, mean over sessions): usual {a_.mean():.3f} / rule {b_.mean():.3f} ({sgn(b_ - a_)})")
    print("  E3 old word-answer keys at the last session (usual / rule):")
    drop = []
    ns_u = np.array([np.nanmean([last(lf, f"name{n} same") for n in NS]) for lf in lives])
    ns_e = np.array([np.nanmean([last(lf, f"name{n} same eq") for n in NS]) for lf in lives])
    rows = [("name same (n pooled)", ns_u, ns_e)]
    for k in ("nwbc", "nwfc"):
        u = np.array([last(lf, f"{k} right all") / last(lf, f"{k} n all") for lf in lives])
        e = np.array([last(lf, f"{k} right all eq") / last(lf, f"{k} n all eq") for lf in lives])
        rows.append((f"{k} all", u, e))
    for name, u, e in rows:
        d = e - u
        w_, l_ = int((d > 1e-12).sum()), int((d < -1e-12).sum())
        is_drop = d.mean() <= -0.05 and binom_p(w_, l_) < 0.005
        drop.append(is_drop)
        print(f"    {name}: {u.mean():.3f} / {e.mean():.3f} ({sgn(d)}){' -> DROP' if is_drop else ''}")
    e1 = (Fe - Be).mean() > -0.05
    e2 = per_n[1][2].mean() >= 0.40
    if any(drop) or not (e1 or e2):
        verdict = "실패"
    elif e1 and e2:
        verdict = "성공"
    else:
        verdict = "부분"
    print(f"  E1 framed - bare under the rule {(Fe - Be).mean():+.3f} -> {'틀 영향 사라짐' if e1 else '틀 영향 남음'}; "
          f"E2 one hearing under the rule {per_n[1][2].mean():.3f} -> {'0.40 넘음' if e2 else '0.40 못 미침'}; "
          f"E3 {'옛 열쇠 떨어짐' if any(drop) else '떨어진 옛 열쇠 없음'} -> verdict: {verdict}")
