"""Track A W2 (2026-10-13; repo STAGE1_VOYAGER.md; shared STAGE1.md edition 2, criterion 2 "new words"): the new-word exposure
curve and its retention. Written and fixed BEFORE the seed-0 smoke. Run first: the world regression (every old glyph world
byte-identical with the option off, 17 worlds x seeds 0-2), the life regression (check_w2.py reg) and the criterion-0 gate on
"@nw" (elements_10_nw.out: PASS, 0.9996, Wilson [0.9984, 0.9999]).

World "@nw" (data_ascii nw=True = the base glyph world "ascii" with one change in the naming moments): in every room each of the
four objects is named with its DRAWN picture -- (picture, name, the place it stands) + 1-3 other pictures of the room -- and is
heard 1 / 2 / 4 / 8 times (which family gets which count: a separate random stream per room); the names are left out of the
noise sentences, so a name is heard exactly that many times in the life. Motion moments as in "ascii".
Test "nw{n}" (the room's list "nw", not in its "test"): the new word in a DIFFERENT sentence -- another place (never the one it
was named at) and fresh background pictures (never its own) -- (a) the name given, its picture asked; (b) the picture given,
its name asked; 3 items each per object (186 per exposure count in a life). Old-style "name{n} same" (the room's "test" list,
as in "ascii": the same place, other background pictures) -- report only.
Keys per session (run_long.life, "@nw"): "nw{n} new" (the rooms that arrived this session = right after exposure), "nw{n} early"
(the rooms of the first half of the life met so far), "nw{n} all".

Conditions (10 seeds first, two-stage rule; 30 if undecided): P = both@ascii@nw+rediscover+chain+wm+wmlast (the glyph world's
parts, as finding 143/144's Voyager B); body = both@ascii@nw+rediscover+chain (D21, body alone).
Baselines (rule 1; no learner; the same session order; all heard sentences): M / C / N as Q5 part 3 (run_baselines3.pick:
candidates = every word but the item's context words, ties -> the lower index, a zero score is wrong) and the shuffle control
(N on the heard words permuted across the session's sentences, rng 80_000 + seed). Cached per seed (baseline_nw_cache/).

Measures per seed: immediate(n) = mean over the sessions of "nw{n} new"; late(n) = "nw{n} early" at the last session;
immediate-early(n) = mean of "nw{n} new" over the first-half sessions (the same rooms' immediate scores); retention(n) =
late(n) / immediate-early(n), as the ratio of the means over seeds with a bootstrap 95% interval (seeds resampled, 10,000, rng 0).
Pre-registered:
  Q1 criterion-2 DEV reading (STAGE1 edition 2: once-heard in a different sentence >= 0.60 and late retention >= 0.80, by the
     LOWER end of the 95% interval): P's immediate(1) (t interval over seeds) and retention(1) (bootstrap) -- a dev reading
     only; the score table (W7, 30 seeds) gives the official line. n = 2 / 4 / 8 the same way (report).
  Q2 the curve: P immediate(8) - immediate(1), sign test (decided if p < 0.005).
  Q3 D21: P - body, immediate(n) and late(n), sign tests (decided if p < 0.005).
  Q4 rule 1: P - the best baseline (highest mean immediate(n) over seeds among M / C / N), immediate(n) and late(n), sign tests
     -- "the learner beats trivial baselines on new words at n" only if p < 0.005; a baseline at or above P means this
     world's new-word test can be solved by counting (to be reported as such, as finding 143's text world).
  Q5 side effects (report only; "@nw" is not "ascii" -- the room streams differ after the first naming, so no paired
     comparison with the base world): never-heard, once all, unseen all, chain of P and body.
What this assumes: a word's meaning = which picture it names (no other property); a naming moment holds 1-3 other pictures (the
base world's design), so one exposure is ambiguous by design (only mutual exclusivity or the like can tell); the "different
sentence" changes the place and the background, not the kind of sentence (no motion sentence with the new word); "right
after" = after the session in which the room arrived (that session's practice included); "late" = the last session, for the
first-half rooms; retention as a ratio of means; 3 test items per direction per object; the baselines count every heard
sentence (Voyager sets a quarter aside for its self-test -- the baselines get slightly MORE data).
"""
import os
import sys
import pickle
from multiprocessing import Pool

import numpy as np
from scipy import stats

import run_long as RL
from run_schema import PER_SESSION
from run_count import sgn
from run_baselines3 import pick

W_ = "+rediscover+chain"
CONDS = {"P": "both@ascii@nw" + W_ + "+wm+wmlast", "body": "both@ascii@nw" + W_}
NS = (1, 2, 4, 8)
TAGS = ("new", "early", "all")
BCACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "baseline_nw_cache")


def base_life(seed, shuffle=False):
    w = RL.build_world("ascii-nw", seed)
    rng = np.random.default_rng(70_000 + seed)
    V = int(w.V)
    order = [int(c) for phase in w.phases for c in rng.permutation(phase)]
    sessions = [order[i:i + PER_SESSION] for i in range(0, len(order), PER_SESSION)]
    early = {c for s_ in sessions[:len(sessions) // 2] for c in s_}
    r2 = np.random.default_rng(80_000 + seed)
    freq, co = np.zeros(V), np.zeros((V, V))
    seen, out = [], []
    for new in sessions:
        seen += new
        block = [[int(t) for t in s] for c in new for _, s, _ in w.categories[c]["heard"]]
        if shuffle:
            flat = r2.permutation([t for s in block for t in s]).tolist()
            nb, i = [], 0
            for s in block:
                nb.append(flat[i:i + len(s)])
                i += len(s)
            block = nb
        for s in block:
            u = sorted(set(s))
            freq[u] += 1
            co[np.ix_(u, u)] += 1
        np.fill_diagonal(co, 0)
        norm = co / np.sqrt(np.outer(freq, freq) + 1e-12)
        scorer = {"M": lambda cx: freq, "C": lambda cx: co[cx].sum(0), "N": lambda cx: norm[cx].sum(0)}
        rec = {}
        for b in (("N",) if shuffle else ("M", "C", "N")):
            for n in NS:
                for tag, rooms in (("new", new), ("early", [c for c in seen if c in early]), ("all", seen)):
                    items = [(s_, ans[0]) for c in rooms for k_, s_, ans in w.categories[c].get("nw", []) if k_ == f"nw{n}"]
                    rec[(b, f"nw{n} {tag}")] = (float(np.mean([pick(scorer[b](cx), cx) == t for cx, t in items]))
                                                if items else np.nan)
        out.append(rec)
    return out


def base_job(args):
    seed, shuffle = args
    path = os.path.join(BCACHE, f"ascii-nw_seed{seed}{'_shuf' if shuffle else ''}.pkl")
    if os.path.exists(path):
        with open(path, "rb") as f:
            return pickle.load(f)
    res = base_life(seed, shuffle)
    os.makedirs(BCACHE, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(res, f)
    return res


def grid(lives, key):
    """seeds x sessions."""
    return np.array([[st.get(key, np.nan) for st in lf] for lf in lives])


def measures(get, n):
    new = get(f"nw{n} new")
    half = new.shape[1] // 2
    return np.nanmean(new, 1), np.nanmean(new[:, :half], 1), get(f"nw{n} early")[:, -1]


def tci(x):
    x = np.asarray(x, dtype=float)
    m, se = x.mean(), x.std(ddof=1) / np.sqrt(len(x))
    h = stats.t.ppf(0.975, len(x) - 1) * se
    return m, m - h, m + h


def ratio_ci(late, imm_e, B=10_000):
    r = np.random.default_rng(0)
    idx = r.integers(0, len(late), (B, len(late)))
    with np.errstate(divide="ignore", invalid="ignore"):
        rat = late[idx].mean(1) / imm_e[idx].mean(1)
    rat = rat[np.isfinite(rat)]
    est = late.mean() / imm_e.mean() if imm_e.mean() > 0 else np.nan
    return est, (np.percentile(rat, 2.5) if len(rat) else np.nan), (np.percentile(rat, 97.5) if len(rat) else np.nan)


if __name__ == "__main__":
    n_seeds = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    workers = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    seeds = list(range(n_seeds))
    tasks = [(sd, c) for c in CONDS.values() for sd in seeds]
    with Pool(workers) as p:
        out = dict(zip(tasks, p.map(RL.one, tasks)))
        bres = dict(zip([(sd, sh) for sd in seeds for sh in (False, True)],
                        p.map(base_job, [(sd, sh) for sd in seeds for sh in (False, True)])))
    lives = {c: [out[(sd, CONDS[c])] for sd in seeds] for c in CONDS}
    getters = {c: (lambda k, c=c: grid(lives[c], k)) for c in CONDS}
    for b in ("M", "C", "N"):
        getters[b] = lambda k, b=b: grid([[{kk: v for (bb, kk), v in st.items() if bb == b} for st in bres[(sd, False)]]
                                          for sd in seeds], k)
    getters["N/shuf"] = lambda k: grid([[{kk: v for (bb, kk), v in st.items()} for st in bres[(sd, True)]]
                                        for sd in seeds], k)
    print(f"=== track A W2: new-word curve and retention (\"@nw\"), {n_seeds} seeds ===")
    print("  immediate = mean over sessions of the new rooms right after exposure [95% t]; early-imm = the same over the "
          "first-half sessions; late = first-half rooms at the end; retention = late / early-imm [bootstrap 95%]")
    res = {}
    for n in NS:
        print(f"  --- heard {n} time(s): different sentence (another place, fresh background) ---")
        for who, get in getters.items():
            imm, imm_e, late = measures(get, n)
            res[(who, n)] = (imm, imm_e, late)
            m, lo, hi = tci(imm)
            r, rlo, rhi = ratio_ci(late, imm_e)
            print(f"  nw{n} {who:6} immediate {m:.3f} [{lo:.3f}, {hi:.3f}] | early-imm {imm_e.mean():.3f} | late "
                  f"{late.mean():.3f} | retention {r:.3f} [{rlo:.3f}, {rhi:.3f}]", flush=True)
    for c in CONDS:
        for n in NS:
            print(f"  nw{n} all {c:4} " + " ".join(f"{v:.3f}" for v in np.nanmean(getters[c](f"nw{n} all"), 0)))
    for c in CONDS:
        for n in NS:
            s_ = getters[c](f"name{n} same")
            print(f"  name{n} same (old-style, same place) {c:4} end {np.nanmean(s_[:, -1]):.3f} | life {np.nanmean(s_):.3f}")
    print("  --- Q1 criterion-2 DEV reading (pass line: immediate lower end >= 0.60 AND retention lower end >= 0.80) ---")
    for n in NS:
        imm, imm_e, late = res[("P", n)]
        m, lo, hi = tci(imm)
        r, rlo, rhi = ratio_ci(late, imm_e)
        ok = lo >= 0.60 and rlo >= 0.80
        print(f"  Q1 n={n}: immediate lower {lo:.3f}, retention lower {rlo:.3f} -> {'meets the line (dev)' if ok else 'not yet'}")
    print(f"  Q2 curve, P immediate(8) - immediate(1): {sgn(res[('P', 8)][0] - res[('P', 1)][0])}")
    for n in NS:
        print(f"  Q3 n={n}, P - body: immediate {sgn(res[('P', n)][0] - res[('body', n)][0])} | late "
              f"{sgn(res[('P', n)][2] - res[('body', n)][2])}")
    for n in NS:
        best = max("MCN", key=lambda b: res[(b, n)][0].mean())
        print(f"  Q4 n={n}, P - best baseline {best}: immediate {sgn(res[('P', n)][0] - res[(best, n)][0])} | late "
              f"{sgn(res[('P', n)][2] - res[(best, n)][2])}")
    for c in CONDS:
        for k in ("never-heard", "once all", "unseen all", "chain"):
            v = getters[c](k)
            print(f"  Q5 {c:4} {k:11} end {np.nanmean(v[:, -1]):.3f} | life {np.nanmean(v):.3f}")
