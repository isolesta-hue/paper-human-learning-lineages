"""W6 (stage-1 track A, criterion 5 "two languages"; 2026-10-15; W6_DESIGN.md, STAGE1_VOYAGER.md W6; the user's go-ahead
10-15 "응, 너의 의견대로 하자" and, on the converter's correction, "응, 그렇게 진행해줘"; the user's caution 10-15: "테스트에
신경쓰다가 범용 로직이 아닌 테스트에 최적화된 로직을 설계하는 실수는 경계하고" -- so W6 widens the world and the tests only:
no part changes; what the body cannot do is written "아직" and goes to track B, where a fix must be a general principle).
Written and fixed BEFORE the seed-0 smokes.

World "@bi" (data_ascii bi=True = the "@kor" world, unchanged, plus): every object gets a second, English-like name (a new
word, random features); heard, each story in ONE language (the exam's exposure form -- no paired translations): English
naming moments of the same form and counts as the Korean ones (picture, name, where) and English stories "m VERB FROM lm"
+ the scene (2 per motion word). Motion words are shared (Voyager's verbs are textbook formulas, no language's words).
Tests per motion room, one event per motion word (no scene):
  "bi story ko"   "[m GA] [lm ESEO] VERB" (phrase order free)            -> the end cell   (criterion 1, Korean)
  "bi story ko d" + the distractor "[lm2 REUL]"                          -> the end cell   (criterion 1, Korean)
  "bi story en"   "m VERB FROM lm"                                       -> the end cell   (criterion 1, English-like)
  "bi story en d" "m VERB lm2 FROM lm" (only word order tells m from lm2) -> the end cell  (ORDER-ONLY: report)
  "bi where X>Y"  the plain story in X, GO, then ko "[m GA] EODI" / en "WHERE IS m"        -> the end cell
  "bi what X>Y"   the plain story in X, GO, then ko "[MUEOS GA] [lm ESEO] VERB" / en "WHAT VERB FROM lm"
                                                                          -> m's name in Y among the room's four names in Y
  X != Y = criterion 5's "a fact told in one language, asked in the other"; X = Y = the same item without the crossing.
  Cells are answered among the 25 "to" cells (the exam's cell slot), names among the four (chance 0.25).
Shuffle control "@bishuf": every English naming moment shows a RANDOM object of the room (its picture and place, drawn per
moment) -- an English name points at no object; Korean is unchanged.
Conditions (10 seeds first, two-stage rule, early stop Q10; the seeds are reused at 30):
  P = both@ascii@bi+rediscover+chain+wm+bind+wmlast+track  (finding 125's Korean body + finding 126's tracking, for GO)
  S = the same body on "@bishuf".
Pre-registered (P at the end unless said; "reading" = the development reading, never the sealed exam):
  W6-1 criterion 1 per language: Korean = "bi story ko all" ("ko" + "ko d"), English-like = "bi story en". "통과" only if
       the pooled share's Wilson 95% lower bound >= 0.85 AND P - the best of M / C / N is decided (p < 0.005); for the
       English-like reading ALSO P - S decided (the Korean part is the same in S by design, so P - S is reported only).
  W6-2 criterion 5, told in one language and asked in the other: "bi cross" (where + what, ko>en + en>ko pooled). "통과"
       only if the Wilson lower bound >= 0.70 AND P - S AND P - the best of M / C / N decided (p < 0.005). Per form and
       per direction reported.
  W6-3 the same items without the crossing ("bi same", X = Y) and the cost of crossing (same - cross, sign test): report.
  W6-4 the order-only English items "bi story en d": report.
  W6-5 the body alone (world model silent; D21): every key with " body" (report, end).
  W6-6 trivial baselines over the heard sentences in the life's session order, each item answered among its candidates:
       M (frequency), C (co-occurrence with the item's words), N (normalised co-occurrence) -- end; N also on "@bishuf".
  W6-7 side effects (report only): P's "story", "never-heard", "once", "unseen", "chain", "multi" against the cached
       "@multi" lives of the same body (edition 2; edition 1 compared with "@kor" lives without "+track"): the life
       means' max |difference|.
EXPECTED (written before any result, not a reading): "+bind" binds a particle to the element right BEFORE it and finds
the start and the mover by particles only (finding 123); so in English-like sentences FROM is bound to the verb and the
subject has no mark -> the world model stays silent -> near chance; and the world model follows a state keyed by the
Korean name, so an English question about the same object is not recognised -> crossing near chance. If so: "아직", and
the track-B candidates (general principles only): (a) which side a function word binds to, and its role, learned from
the learner's own record (any particle / preposition / notation); (b) two names heard with the same picture and place
are one object (synonyms, nicknames, other scripts; D55 index, D60 source).
EDITION 2 OF THE WORLD (2026-10-15, after the seed-0 smoke of edition 1; no part, no reading changed): edition 1 was built
on "@kor", which has no heard GO-joined (multi-moment) stories -- so the tracking part (+track, finding 126), which earns its
trust on those (Laplace, (hit + 1) / (n + 2) > 0.5), never spoke: in the smoke every "bi where" / "bi what" key equalled
its body-alone key, even Korean-to-Korean (0.406 at the end). Edition 2 builds "@bi" on "@multi" (the world the tracking
part was made in), so the same-language items can use it and the cost of crossing is measured against a working
reference. W6-7 now compares with the cached "@multi" lives of exactly P's body. The English material is unchanged (no
English multi-moment heard stories -- an asymmetry written here: the Korean multi form came first).
What this assumes: verbs shared by the two languages (textbook formulas); a fixed English-like order (subject, verb,
object, prepositional phrase last); naming moments without function words in both languages; one-moment questions; GO =
the moment boundary (in the exam, separate sentence records joined by the converter's boundary mark); the "what"
candidates = the room's four objects; one story = one language; one event per motion word per room (the life's cost --
learned in W5); English names have random features while the Korean names carry the object's own features (the B-stage
world's design, unchanged) -- an asymmetry written here, not tuned away.
"""
import os
import pickle
import sys

import numpy as np

import run_long as RL
from run_schema import PER_SESSION
from run_count import sgn, sign_p
from run_elements import wilson

BODY = "+rediscover+chain+wm+bind+wmlast+track"
CONDS = {"P": "both@ascii@bi" + BODY, "S": "both@ascii@bishuf" + BODY}
OLD = "both@ascii@multi+rediscover+chain+wm+bind+wmlast+track"   # edition 2: the same body on "@multi"
BCACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "baseline_bi_cache")


def keys_of(k_):
    """the pooled keys an item counts in (as run_long.bi_keys)"""
    out = [k_]
    if k_ in ("bi story ko", "bi story ko d"):
        out.append("bi story ko all")
    if k_.startswith(("bi where ", "bi what ")):
        x_, y_ = k_.split()[-1].split(">")
        side = "cross" if x_ != y_ else "same"
        out += [f"bi {side} {k_.split()[1]}", f"bi {side}"]
    return out


def base_life(seed, world="ascii-bi"):
    """M / C / N over the heard sentences in the life's session order, each item answered among its candidates (the 25
    "to" cells, the answer first; "what": the room's four names in the asked language)."""
    w = RL.build_world(world, seed)
    rng = np.random.default_rng(70_000 + seed)
    V = int(w.V)
    to_cells = sorted(int(t) for t, r in w.role_of.items() if r == "to")
    order = [int(c) for phase in w.phases for c in rng.permutation(phase)]
    sessions = [order[i:i + PER_SESSION] for i in range(0, len(order), PER_SESSION)]
    freq, co = np.zeros(V), np.zeros((V, V))
    seen, out = [], []
    for new in sessions:
        seen += new
        for c in new:
            for _, s, _ in w.categories[c]["heard"]:
                u = sorted({int(t) for t in s})
                freq[u] += 1
                co[np.ix_(u, u)] += 1
        np.fill_diagonal(co, 0)
        norm = co / np.sqrt(np.outer(freq, freq) + 1e-12)
        rec = {}
        for b in ("M", "C", "N"):
            groups = {}
            for c in seen:
                for k_, s_, a_ in w.categories[c].get("bi", []):
                    cands = [int(x) for x in a_] if k_.startswith("bi what ") else \
                        [int(a_[0])] + [t for t in to_cells if t != int(a_[0])]
                    cx = [int(x) for x in s_]
                    sc = freq if b == "M" else co[cx].sum(0) if b == "C" else norm[cx].sum(0)
                    v = [float(sc[x]) for x in cands]
                    r_ = float(v[0] > max(v[1:]))
                    for key in keys_of(k_):
                        groups.setdefault(key, []).append(r_)
            for k, v in groups.items():
                rec[(b, k)] = float(np.mean(v))
        out.append(rec)
    return out


def base_job(args):
    seed, world = args
    path = os.path.join(BCACHE, f"{world}_seed{seed}.pkl")
    if os.path.exists(path):
        with open(path, "rb") as f:
            return pickle.load(f)
    res = base_life(seed, world)
    os.makedirs(BCACHE, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(res, f)
    return res


def n_items(seed, key):
    w = RL.build_world("ascii-bi", seed)
    return sum(1 for c in w.categories[1:] for k_, _, _ in c.get("bi", []) if key in keys_of(k_))


def p_of(d):
    d = np.asarray(d, dtype=float)
    d = d[~np.isnan(d)]
    return sign_p(int((d > 1e-12).sum()), int((d < -1e-12).sum()))


if __name__ == "__main__":
    n_seeds = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    workers = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    seeds = list(range(n_seeds))
    tasks = [(sd, c) for c in CONDS.values() for sd in seeds]
    worlds = [(sd, w_) for w_ in ("ascii-bi", "ascii-bishuf") for sd in seeds]
    out = dict(zip(tasks, RL.map_lives(tasks, workers)))      # D57: a dead worker fails loudly; chunksize 1
    base = dict(zip(worlds, RL.map_lives(worlds, workers, fn=base_job, peak=lambda t: 1.0)))
    g = lambda c, k: np.array([[st.get(k, np.nan) for st in out[(sd, CONDS[c])]] for sd in seeds])
    end = lambda c, k: float(np.nanmean(g(c, k)[:, -1]))
    print(f"=== W6 two languages (criteria 1 per language and 5, development reading), {n_seeds} seeds ===")
    forms = ["bi story ko", "bi story ko d", "bi story ko all", "bi story en", "bi story en d"] + \
        [f"bi {f} {x}>{y}" for f in ("where", "what") for x in ("ko", "en") for y in ("ko", "en")] + \
        [f"bi {s} {f}" for s in ("cross", "same") for f in ("where", "what")] + ["bi cross", "bi same"]
    for k in forms:
        for c in CONDS:
            for tag in ("", " body"):
                v = g(c, k + tag)
                if np.all(np.isnan(v)):
                    continue
                print(f"  {k + tag:24} {c} " + " ".join("  -  " if np.isnan(x) else f"{x:.3f}" for x in np.nanmean(v, 0)))
    bm = {k: {b: np.array([base[(sd, "ascii-bi")][-1].get((b, k), np.nan) for sd in seeds]) for b in ("M", "C", "N")}
          for k in forms}
    bs = {k: np.array([base[(sd, "ascii-bishuf")][-1].get(("N", k), np.nan) for sd in seeds]) for k in forms}

    def reading(tag_, key, line, need_s):
        n = sum(n_items(sd, key) for sd in seeds)
        m = end("P", key)
        lo = wilson(round(m * n), n)[0]
        dS = g("P", key)[:, -1] - g("S", key)[:, -1]
        best = max(("M", "C", "N"), key=lambda b: np.nanmean(bm[key][b]))
        dB = g("P", key)[:, -1] - bm[key][best]
        print(f"  {tag_} {key} (pass line {line}): pooled {m:.3f}, Wilson lower bound {lo:.3f} (n {n}); P - S end {sgn(dS)}"
              f"{'' if need_s else ' (report)'}; best baseline {best} {np.nanmean(bm[key][best]):.3f}, P - {best} end {sgn(dB)}")
        ok = lo >= line and p_of(dB) < 0.005 and np.nanmean(dB) > 0
        if need_s:
            ok = ok and p_of(dS) < 0.005 and np.nanmean(dS) > 0
        print(f"  reading {key}: {'통과' if ok else '아직'} (line met {lo >= line})")
    reading("W6-1", "bi story ko all", 0.85, False)
    reading("W6-1", "bi story en", 0.85, True)
    reading("W6-2", "bi cross", 0.70, True)
    for k in ("bi cross where", "bi cross what", "bi where ko>en", "bi where en>ko", "bi what ko>en", "bi what en>ko"):
        d = g("P", k) - g("S", k)
        print(f"  W6-2 {k}: P end {end('P', k):.3f}, S {end('S', k):.3f}; P - S end {sgn(d[:, -1])} | life "
              f"{sgn(np.nanmean(d, 1))}")
    for s_, c_ in (("bi same", "bi cross"), ("bi same where", "bi cross where"), ("bi same what", "bi cross what")):
        d = g("P", s_)[:, -1] - g("P", c_)[:, -1]
        print(f"  W6-3 {s_} {end('P', s_):.3f} vs {c_} {end('P', c_):.3f}: cost of crossing (same - cross) end {sgn(d)}")
    print(f"  W6-4 bi story en d (order only, report): P end {end('P', 'bi story en d'):.3f}, S {end('S', 'bi story en d'):.3f}")
    for key in ("bi story ko all", "bi story en", "bi story en d", "bi cross", "bi same"):
        print(f"  W6-5 {key} body, P end {end('P', key + ' body'):.3f} (with parts {end('P', key):.3f})")
    for key in ("bi story ko all", "bi story en", "bi story en d", "bi cross where", "bi cross what", "bi same where",
                "bi same what"):
        print(f"  W6-6 {key}: P end {end('P', key):.3f}; baselines end: " +
              ", ".join(f"{b} {np.nanmean(bm[key][b]):.3f}" for b in ("M", "C", "N")) +
              f", N on @bishuf {np.nanmean(bs[key]):.3f}")
    cached = [sd for sd in seeds if os.path.exists(os.path.join(RL.CACHE, f"seed{sd}_{OLD}.pkl"))]
    if cached:
        old = {sd: pickle.load(open(os.path.join(RL.CACHE, f"seed{sd}_{OLD}.pkl"), "rb")) for sd in cached}
        for k in ("story", "never-heard", "once", "unseen", "chain", "multi"):
            dd = [np.nanmean([st.get(k, np.nan) for st in out[(sd, CONDS["P"])]]) -
                  np.nanmean([st.get(k, np.nan) for st in old[sd]]) for sd in cached]
            print(f"  W6-7 {k}: P - @multi life means (same body), mean {np.nanmean(dd):+.3f}, max |diff| "
                  f"{np.nanmax(np.abs(dd)):.3f} (seeds {len(cached)})")
    else:
        print("  W6-7 no cached @multi lives for these seeds (report later)")
