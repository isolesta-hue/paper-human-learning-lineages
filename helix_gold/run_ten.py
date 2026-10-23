"""W5 (stage-1 track A, criteria 3 "count to 10" and 4 "add / take away"; 2026-10-14; STAGE1_VOYAGER.md W5; the user's
go-ahead 10-14 "w5는 너의 의견대로 진행해줘"): numbers 1-10 and putting in / taking out, in the forms of the sealed exam's
items (not its content). Written and fixed BEFORE the seed-0 smokes.

World "@ten" (data_ascii ten=True = the "@count" world with the numbers 1-10 instead of 1-3): things are put into a container
one by one ("[X ga] at(q) [C ane]") and some taken out ("[X ga] at(r)", elsewhere); "[C ane] MYEOT" (how many in C?) is
answered by a number word HANA .. YEOL. The things are drawn uniformly from a fixed pool of 12 counting things plus the
room's own objects (edition 2, below). Heard per motion room (the exam's exposure forms): 5 count stories with distinct n
(edition 2; edition 1 had one per count 1-10) and 4 box stories, put a <= 5 / take
b < a, never the held pairs (3, 1) (4, 2) (5, 3). Tests per motion room:
  "ten count n"  (criterion 3 (a), things -> number word): put n, 2 per n, answered among the ten number words;
  "ten pick n"   (criterion 3 (b), number word -> things): the word for n scored after each of four stories (n and three
                 other counts, the same container and cell), right when it scores strictly highest after the story with n;
  "ten next n"   (criterion 3 (c), report only): told "[C ane] NUM_n", one more put in -> NUM_n+1 (n = 1-9);
  "ten box g"    (criterion 4, number words): put a / take b -> a - b; g = held (the three held pairs), far (a = 6-10, 3 a
                 room), taught (heard pairs, 2 a room; report); "new" = held + far (combinations never heard);
  "voe ten g"    (criterion 4, things): the right count / one more or one fewer -- the wrong one more surprising (ties half).
Shuffle control "@tenshuf": the heard answers drawn at random among the ten words; the tests are the same items.
Conditions (10 seeds first, two-stage rule; the seeds are reused at 30):
  P = both@ascii@ten+rediscover+chain+wm+bind+wmlast+track+count  (finding 133's body with its counting part)
  S = the same body on "@tenshuf".
Pre-registered (P at the end unless said; "reading" = the development reading, never the sealed exam):
  W5-1 criterion 3 (a) "ten count": per n the pooled share; reading "통과" only if the pooled share's Wilson 95% lower bound
       >= 0.90 AND every n's Wilson lower bound >= 0.70 AND P - S and P - the best of M / C / N decided (p < 0.005), else
       "아직".
  W5-2 criterion 3 (b) "ten pick": the same rule.
  W5-3 criterion 3 (c) "ten next" per n (report only, as the exam).
  W5-4 criterion 4, number words: "ten box new" -- the same rule with 0.80 (no per-n part); held / far / taught reported;
       STAGE1 counts criterion 4 only after criterion 3 passes (reported either way).
  W5-5 criterion 4, things: "voe ten new" -- the same rule with 0.85.
  W5-6 the shuffle control collapses: P - S on "ten count", "ten pick", "ten box new", "voe ten new": end and life (sign tests).
  W5-7 the body alone (world model silent; D21): every key above with " body" (report, end).
  W5-8 trivial baselines over the heard sentences in the life's session order, each item answered among its candidates:
       M (frequency), C (co-occurrence with the item's words), N (normalised co-occurrence), and Lg (story length -> the word
       heard most often with a story of that length; ties and unseen lengths at random) -- end; N also on "@tenshuf". Lg is
       a trivial baseline only where length does not carry the count: for put-only stories (ten count, ten pick) the length
       IS the number of things (counting the moments is counting), so there Lg is reported but not in the pass rule; for
       box stories (length = a + b, the count a - b) Lg is in the "best baseline" of W5-4 / W5-5.
  W5-9 side effects (report only): P's "contain in", "contain by", "multi", "chain", "never-heard" against the cached
       "@count" lives of the same body (finding 133's, seeds cached): the life means' max |difference|.
What this assumes: the count = the things in relation "ane" with C and at C's cell (finding 133's counting part, unchanged:
a number word for a count is learned from the heard stories, majority); the ten number words are particle-like elements as
MYEOT; the exam's "n things of one kind" become n distinct things drawn from a fixed pool of counting things and the
room's own objects (Voyager's things are individuals);
the exam's scenes become put-in stories (Voyager has no many-thing scene); the four-way choice of (b) is read as the
told word's score after each story (the exam's choice of a scene); the told count of (c) is a moment "[C ane] NUM" never
heard before (as the exam's (c) premise); no counting chant (the exam's exposure has none); 2 / 1 / 1 / 8 / 8 tests per n
or room as above.
EDITION 2 OF THE WORLD (2026-10-14 ~11:50Z, BEFORE ANY W5 RESULT -- the only W5 lives run so far were timing probes, whose
keys were never read): the edition-1 world did not fit a life into the 2-hour job (the P smoke died at the limit; sessions
grew to ~660 s). Profiles showed (1) the pattern memory padded every practice row to the batch's longest sentence -- fixed
exactly in run_schema3.py (life regression 0.0), and (2) the things drawn from EVERY room's objects put ~120 names into
every room's stories. So: the things now come from a fixed world-wide pool of 12 counting things (objects of the
placement rooms, drawn once a world) plus the room's own three, uniformly (still: a large count does not mean "another
kind of name"); and 5 count scenes a room with distinct n (each n heard ~11 times a life) instead of all ten. The tests,
the readings and every pre-registered rule above are unchanged. Projected life ~75 min (8 sessions in 1394 s under load).
EDITION 3 (2026-10-14 ~14:00Z, STILL BEFORE ANY W5 RESULT -- the edition-2 seed-0 smoke also died at the 2-hour limit with
no output; the per-session cost keeps growing with the heard material practised in every session): 3 count scenes a room
with distinct n (each n heard ~7 times a life) and 2 box stories a room (~46 a life). Tests, readings and rules unchanged.
"""
import os
import pickle
import sys

import numpy as np

import run_long as RL
from run_schema import PER_SESSION
from run_count import sgn, sign_p
from run_elements import wilson

BODY = "+rediscover+chain+wm+bind+wmlast+track+count"
CONDS = {"P": "both@ascii@ten" + BODY, "S": "both@ascii@tenshuf" + BODY}
OLD = "both@ascii@count" + BODY
BCACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "baseline_ten_cache")
FORMS = ("count", "next", "box")


def base_life(seed, world="ascii-ten"):
    """M / C / N over the heard sentences in the life's session order and the length table Lg, each item answered among
    its candidates (the ten number words; "ten pick": the told word after each of the four stories; "voe ten": the right
    word against the wrong one)."""
    w = RL.build_world(world, seed)
    rng = np.random.default_rng(70_000 + seed)
    lrng = np.random.default_rng(90_000 + seed)
    V = int(w.V)
    nums = set(int(x) for x in w.ten_words)
    order = [int(c) for phase in w.phases for c in rng.permutation(phase)]
    sessions = [order[i:i + PER_SESSION] for i in range(0, len(order), PER_SESSION)]
    freq, co = np.zeros(V), np.zeros((V, V))
    lg = {}
    seen, out = [], []
    for new in sessions:
        seen += new
        for c in new:
            for _, s, _ in w.categories[c]["heard"]:
                u = sorted({int(t) for t in s})
                freq[u] += 1
                co[np.ix_(u, u)] += 1
                if int(s[-1]) in nums:
                    d = lg.setdefault(len(s) - 1, {})
                    d[int(s[-1])] = d.get(int(s[-1]), 0) + 1
        np.fill_diagonal(co, 0)
        norm = co / np.sqrt(np.outer(freq, freq) + 1e-12)
        items = [(k_, s_, a_) for c in seen for k_, s_, a_ in w.categories[c].get("ten", [])]

        def score(b, s_, cands):
            if b == "Lg":
                d = lg.get(len(s_), {})
                if not d:
                    pick = cands[int(lrng.integers(len(cands)))]
                else:
                    best = [x for x, v in d.items() if v == max(d.values())]
                    pick = best[int(lrng.integers(len(best)))]
                return [float(int(x) == pick) for x in cands]
            cx = [int(x) for x in s_]
            sc = freq if b == "M" else co[cx].sum(0) if b == "C" else norm[cx].sum(0)
            return [float(sc[int(x)]) for x in cands]
        rec = {}
        for b in ("M", "C", "N", "Lg"):
            groups = {}
            for k_, s_, a_ in items:
                kp = k_.split()
                if kp[0] == "ten" and kp[1] in FORMS:
                    v = score(b, s_, a_)
                    r_ = float(v[0] > max(v[1:]))
                    groups.setdefault(f"ten {kp[1]}", []).append(r_)
                    groups.setdefault(f"ten {kp[1]} {kp[2]}", []).append(r_)
                    if kp[1] == "box" and kp[2] in ("held", "far"):
                        groups.setdefault("ten box new", []).append(r_)
                elif kp[0] == "ten" and kp[1] == "pick":
                    word = int(a_[0])
                    v = [score(b, st_, [word] + sorted(nums - {word}))[0] if b == "Lg" else score(b, st_, [word])[0]
                         for st_ in s_]
                    r_ = float(v[0] > max(v[1:]))
                    groups.setdefault("ten pick", []).append(r_)
                    groups.setdefault(f"ten pick {kp[2]}", []).append(r_)
                elif kp[0] == "voe":
                    v = score(b, s_, a_)
                    r_ = float(v[0] > v[1]) + 0.5 * float(v[0] == v[1])
                    groups.setdefault("voe ten", []).append(r_)
                    groups.setdefault(f"voe ten {kp[2]}", []).append(r_)
                    if kp[2] in ("held", "far"):
                        groups.setdefault("voe ten new", []).append(r_)
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


def n_items(seed, prefixes):
    w = RL.build_world("ascii-ten", seed)
    return sum(1 for c in w.categories[1:] for k_, _, _ in c.get("ten", []) if k_.startswith(prefixes))


def p_of(d):
    d = np.asarray(d, dtype=float)
    d = d[~np.isnan(d)]
    return sign_p(int((d > 1e-12).sum()), int((d < -1e-12).sum()))


if __name__ == "__main__":
    n_seeds = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    workers = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    seeds = list(range(n_seeds))
    tasks = [(sd, c) for c in CONDS.values() for sd in seeds]
    worlds = [(sd, w_) for w_ in ("ascii-ten", "ascii-tenshuf") for sd in seeds]
    out = dict(zip(tasks, RL.map_lives(tasks, workers)))      # D57: a dead worker fails loudly; chunksize 1
    base = dict(zip(worlds, RL.map_lives(worlds, workers, fn=base_job, peak=lambda t: 1.0)))
    g = lambda c, k: np.array([[st.get(k, np.nan) for st in out[(sd, CONDS[c])]] for sd in seeds])
    end = lambda c, k: float(np.nanmean(g(c, k)[:, -1]))
    print(f"=== W5 numbers 1-10, put in / take out (criteria 3 and 4, development reading), {n_seeds} seeds ===")
    keys = [f"ten {f}{x}" for f in ("count", "pick", "next") for x in [""] + [f" {n}" for n in range(1, 11)]] + \
        [f"ten box{x}" for x in ("", " new", " held", " far", " taught")] + \
        [f"voe ten{x}" for x in ("", " new", " held", " far", " taught")] + ["ten count all", "ten next all", "ten box all"]
    for k in keys:
        for c in CONDS:
            for tag in ("", " body"):
                v = g(c, k + tag)
                if np.all(np.isnan(v)):
                    continue
                print(f"  {k + tag:24} {c} " + " ".join("  -  " if np.isnan(x) else f"{x:.3f}" for x in np.nanmean(v, 0)))
    bkeys = ["ten count", "ten pick", "ten next", "ten box new", "ten box held", "ten box far", "ten box taught",
             "voe ten new", "voe ten held", "voe ten far", "voe ten taught"]
    bm = {k: {b: np.array([base[(sd, "ascii-ten")][-1].get((b, k), np.nan) for sd in seeds]) for b in ("M", "C", "N", "Lg")}
          for k in bkeys}
    bs = {k: np.array([base[(sd, "ascii-tenshuf")][-1].get(("N", k), np.nan) for sd in seeds]) for k in bkeys}

    def reading(key, line, per_n, cand_base):
        n = sum(n_items(sd, {"ten count": ("ten count ",), "ten pick": ("ten pick ",),
                             "ten box new": ("ten box held", "ten box far"),
                             "voe ten new": ("voe ten held", "voe ten far")}[key]) for sd in seeds)
        m = end("P", key)
        lo = wilson(round(m * n), n)[0]
        ok = lo >= line
        txt = f"pooled {m:.3f}, Wilson lower bound {lo:.3f} (n {n})"
        if per_n:
            n_per = n // 10
            per = {x: end("P", f"{key} {x}") for x in range(1, 11)}
            los = {x: wilson(round(v * n_per), n_per)[0] for x, v in per.items()}
            ok = ok and min(los.values()) >= 0.70
            txt += "; per n " + " ".join(f"{x}:{per[x]:.2f}[{los[x]:.2f}]" for x in range(1, 11))
        dS = g("P", key)[:, -1] - g("S", key)[:, -1]
        best = max(cand_base, key=lambda b: np.nanmean(bm[key][b]))
        dB = g("P", key)[:, -1] - bm[key][best]
        pS, pB = sgn(dS), sgn(dB)
        print(f"  {line_tag[key]} {key} (pass line {line}): {txt}; P - S end {pS}; best baseline {best} "
              f"{np.nanmean(bm[key][best]):.3f}, P - {best} end {pB}")
        return ok, dS, dB

    line_tag = {"ten count": "W5-1", "ten pick": "W5-2", "ten box new": "W5-4", "voe ten new": "W5-5"}
    res = {}
    for key, line, per_n, cb in (("ten count", 0.90, True, ("M", "C", "N")), ("ten pick", 0.90, True, ("M", "C", "N")),
                                 ("ten box new", 0.80, False, ("M", "C", "N", "Lg")),
                                 ("voe ten new", 0.85, False, ("M", "C", "N", "Lg"))):
        res[key] = reading(key, line, per_n, cb)
    for key, (ok, dS, dB) in res.items():
        dec = ok and p_of(dS) < 0.005 and p_of(dB) < 0.005 and np.nanmean(dS) > 0 and np.nanmean(dB) > 0
        print(f"  reading {key}: {'통과' if dec else '아직'} (line met {ok})")
    print("  W5-3 ten next (report), P end: " + " ".join(f"{x}:{end('P', f'ten next {x}'):.2f}" for x in range(1, 10)) +
          f" | pooled {end('P', 'ten next'):.3f}, body {end('P', 'ten next body'):.3f}")
    for key in ("ten count", "ten pick", "ten box new", "voe ten new"):
        d = g("P", key) - g("S", key)
        print(f"  W5-6 {key}, P - S: end {sgn(d[:, -1])} | life {sgn(np.nanmean(d, 1))}")
    for key in ("ten count", "ten pick", "ten next", "ten box new", "voe ten new"):
        print(f"  W5-7 {key} body, P end {end('P', key + ' body'):.3f} (with parts {end('P', key):.3f})")
    for key in bkeys:
        print(f"  W5-8 {key}: P end {end('P', key):.3f}; baselines end: " +
              ", ".join(f"{b} {np.nanmean(bm[key][b]):.3f}" for b in ("M", "C", "N", "Lg")) +
              f", N on @tenshuf {np.nanmean(bs[key]):.3f}")
    cached = [sd for sd in seeds if os.path.exists(os.path.join(RL.CACHE, f"seed{sd}_{OLD}.pkl"))]
    if cached:
        old = {sd: pickle.load(open(os.path.join(RL.CACHE, f"seed{sd}_{OLD}.pkl"), "rb")) for sd in cached}
        for k in ("contain in", "contain by", "multi", "chain", "never-heard"):
            dd = [np.nanmean([st.get(k, np.nan) for st in out[(sd, CONDS["P"])]]) -
                  np.nanmean([st.get(k, np.nan) for st in old[sd]]) for sd in cached]
            print(f"  W5-9 {k}: P - @count life means, mean {np.nanmean(dd):+.3f}, max |diff| {np.nanmax(np.abs(dd)):.3f} "
                  f"(seeds {len(cached)})")
    else:
        print("  W5-9 no cached @count lives for these seeds (report later)")
