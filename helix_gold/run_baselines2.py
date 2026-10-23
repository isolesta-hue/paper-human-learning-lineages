"""Q5 part 2 (2026-10-12, review track 3; user approved the literature-check additions): word-learning models and small
learned networks on Voyager's own worlds. Written and fixed BEFORE any result (run_baselines.py = part 1: M, C, N).

Same stream as part 1 (copied, not changed): the heard sentences of each session in the order of a Voyager life (rng
70_000 + seed, the phases' permutations, PER_SESSION categories a session); the same tests after each session (data_v3.
unit_items: hide one unit member, the rest of the test sentence is the context). Candidates = every word; the item's own
context words are excluded; ties -> the lower index; a zero score is wrong (as part 1).

Baselines
  P   Pursuit (Stevens, Gleitman, Trueswell & Yang 2017), word -> word: each word in a heard sentence pursues ONE best
      hypothesis among the other words of that sentence. New word: start on the sentence word with the smallest strongest
      association to any word (mutual exclusivity; ties random), A = gamma. Known word: best h = argmax A(w, .); if h is in
      the sentence A(w,h) += gamma (1 - A(w,h)); else A(w,h) *= (1 - gamma) and a random other sentence word h' gets
      A(w,h') += gamma (1 - A(w,h')). P(h | w) = (A(w,h) + lam) / (sum_h A(w,h) + V lam). Test score(t) = sum over context
      words c of P(t | c). gamma = 0.02, lam = 0.001 (the published values as I recall them; fixed now, not tuned).
  V   propose-but-verify (Trueswell, Medina, Hafri & Gleitman 2013), word -> word: each word holds at most one hypothesis.
      On hearing: recall it with probability a0 = 0.6 (a1 = 0.8 once it was confirmed); recalled and present -> confirmed;
      otherwise a new hypothesis drawn uniformly from the other sentence words (unconfirmed). Test score(t) = number of
      context words whose hypothesis is t. a0, a1 are round numbers chosen now, NOT published fits (reported as such).
  MLP-on / MLP-re   bag of context embeddings (mean, d = 64) -> 128 ReLU -> V logits.
  TF-on / TF-re     2-layer transformer (d = 64, 4 heads, ff 128, learned positions), the context in order + a query token at
                    the end; the query position's output -> V logits.
      Training items: every heard sentence x every position: hide that word, the rest is the context (the test's format).
      Cross-entropy, Adam (one optimizer for the whole life), batch 64, torch seed 90_000 + seed.
      -on (online): each session trains only on that session's new sentences.  -re (replay): on all sentences heard so far.
      Hyperparameters per world and per network x regime from the grid lr {1e-3, 3e-3} x epochs-per-session {10, 40},
      chosen on TUNING seeds 1000-1004 (never evaluation seeds) by the mean never-heard over the life; the choice is printed.
Shuffle control (rule 1; report only): N (part 1, all seeds) and TF-re (first 10 seeds -- cost) on sentences whose words are permuted across the session's
  sentences (rng 80_000 + seed; word frequencies kept, co-occurrence destroyed). Expected near chance; if not, the keys leak
  something other than co-occurrence and the whole baseline reading is suspect.

Pre-registered comparisons (paired by seed; sign test; decided at p < 0.005, else the two-stage rule 30 -> 100):
  B1 glyph world "ascii": Voyager both@ascii+rediscover+chain+wm+wmlast - each of P, V, MLP-on, MLP-re, TF-on, TF-re.
  B2 text world "life32": Voyager both+rediscover - each of the same. Keys: never-heard, once all, unseen all; end and life.
  "Voyager beats the baselines" on a key needs EVERY comparison at p < 0.005 (intersection, so no correction needed).
  Where a baseline is at or above Voyager, that key does not show Voyager's structure is needed (the part-1 reading).
What this assumes: Pursuit and propose-but-verify were made for word -> object mapping across modalities; here the "objects"
are the other words of the same sentence (one sentence = one scene) -- the only reading that fits a text-only stream, and the
same window as C / N. The networks get every heard sentence (Voyager sets a quarter aside for its self-test), so they have
slightly MORE data. Small networks and a 4-point grid are a floor, not the best a network can do: a loss to them is strong
evidence, a win over them is a first step only. CPU, torch 1 thread per process.
Smoke (seed 0, ascii, before the run; hp lr 3e-3 ep 10, not the tuned choice), never-heard every 3rd session:
  P 0.349 .. 0.209 | V 0.167 .. 0.085 | N 0.651 .. 0.258 | MLP-re 0.429 .. 0.634 | TF-re 0.198 .. 0.552 | TF-on 0.198 .. 0.022 |
  N/shuffled 0.016 .. 0.021.  TF-re ep 10 takes ~3 min a life (ep 40 ~4x), hence the shorter TF-re shuffle control.
"""
import sys, os, pickle
import numpy as np
import torch, torch.nn as nn
from multiprocessing import Pool
import data_v3 as W
import run_long as RL
from run_schema import PER_SESSION
from run_count import sgn

KEYS = ("never-heard", "once all", "unseen all")
NETS = ("MLP-on", "MLP-re", "TF-on", "TF-re")
GRID = [(lr, ep) for lr in (1e-3, 3e-3) for ep in (10, 40)]
TUNE = range(1000, 1005)
REFS = {"ascii": "both@ascii+rediscover+chain+wm+wmlast", "life32": "both+rediscover"}


def stream(world, seed, shuffle=False):
    w = RL.build_world(world, seed)
    rng = np.random.default_rng(70_000 + seed)
    V = int(w.V)
    sents = {c: [[int(t) for t in s] for _, s, _ in w.categories[c]["heard"]] for c in range(w.n_cat)}
    tests = {c: W.unit_items(w.categories[c]["test"]) for c in range(w.n_cat)}
    order = [int(c) for phase in w.phases for c in rng.permutation(phase)]
    sessions = [order[i:i + PER_SESSION] for i in range(0, len(order), PER_SESSION)]
    blocks = [[s for c in new for s in sents[c]] for new in sessions]
    if shuffle:
        r2 = np.random.default_rng(80_000 + seed)
        out = []
        for b in blocks:
            flat = r2.permutation([t for s in b for t in s]).tolist()
            lens, i, nb = [len(s) for s in b], 0, []
            for L in lens:
                nb.append(flat[i:i + L]); i += L
            out.append(nb)
        blocks = out
    items_after = []
    seen = []
    for new in sessions:
        seen += new
        ks = {}
        for key in KEYS:
            ks[key] = [([int(x) for x in cx], int(t)) for c in seen for kind, cx, t in tests[c]
                       if key == "never-heard" or (key == "once all" and kind == "once") or (key == "unseen all" and kind == "unseen")]
        items_after.append(ks)
    return V, blocks, items_after


def pick(sc, cx):
    sc = np.asarray(sc, dtype=float).copy(); sc[cx] = -1
    j = int(np.argmax(sc))
    return j if sc[j] > 0 else -1


def score_sessions(V, items_after, scorer_after):
    out = []
    for si, ks in enumerate(items_after):
        sc_fn = scorer_after[si]
        rec = {}
        for key, items in ks.items():
            rec[key] = np.nan if not items else float(np.mean([pick(sc_fn(cx), cx) == t for cx, t in items]))
        out.append(rec)
    return out


def run_pursuit(V, blocks, items_after, seed, gamma=0.02, lam=0.001):
    rng = np.random.default_rng(91_000 + seed)
    A = np.zeros((V, V)); known = np.zeros(V, bool)
    out = []
    for si, b in enumerate(blocks):
        for s in b:
            u = sorted(set(s))
            for w in u:
                others = [x for x in u if x != w]
                if not others:
                    continue
                if not known[w]:
                    strongest = A[:, others].max(0)
                    m = strongest.min(); cand = [o for o, v in zip(others, strongest) if v == m]
                    h = int(rng.choice(cand)); A[w, h] = gamma; known[w] = True
                    continue
                h = int(np.argmax(A[w]))
                if h in others:
                    A[w, h] += gamma * (1 - A[w, h])
                else:
                    A[w, h] *= (1 - gamma)
                    h2 = int(rng.choice(others)); A[w, h2] += gamma * (1 - A[w, h2])
        P = (A + lam) / (A.sum(1, keepdims=True) + V * lam)
        P[~known] = 0
        Pc = P.copy()
        out.append(score_sessions(V, [items_after[si]], [lambda cx, Pc=Pc: Pc[cx].sum(0)])[0])
    return out


def run_pbv(V, blocks, items_after, seed, a0=0.6, a1=0.8):
    rng = np.random.default_rng(92_000 + seed)
    hyp = -np.ones(V, int); conf = np.zeros(V, bool)
    out = []
    for si, b in enumerate(blocks):
        for s in b:
            u = sorted(set(s))
            for w in u:
                others = [x for x in u if x != w]
                if not others:
                    continue
                recalled = hyp[w] >= 0 and rng.random() < (a1 if conf[w] else a0)
                if recalled and hyp[w] in others:
                    conf[w] = True
                else:
                    hyp[w] = int(rng.choice(others)); conf[w] = False
        H = hyp.copy()

        def sc(cx, H=H):
            v = np.zeros(V)
            for c in cx:
                if H[c] >= 0: v[H[c]] += 1
            return v
        out.append(score_sessions(V, [items_after[si]], [sc])[0])
    return out


class MLP(nn.Module):
    def __init__(self, V, d=64):
        super().__init__()
        self.e = nn.EmbeddingBag(V + 2, d, mode="mean", padding_idx=V + 1)
        self.f = nn.Sequential(nn.Linear(d, 128), nn.ReLU(), nn.Linear(128, V))

    def forward(self, x):
        return self.f(self.e(x))


class TF(nn.Module):
    def __init__(self, V, d=64, L=48):
        super().__init__()
        self.V = V
        self.e = nn.Embedding(V + 2, d, padding_idx=V + 1); self.p = nn.Embedding(L, d)
        layer = nn.TransformerEncoderLayer(d, 4, 128, dropout=0.0, batch_first=True)
        self.t = nn.TransformerEncoder(layer, 2, enable_nested_tensor=False)
        self.o = nn.Linear(d, V)

    def forward(self, x):                 # x: context padded with V+1, query token V at its last real position
        pad = x == self.V + 1
        h = self.e(x) + self.p(torch.arange(x.shape[1]))[None]
        h = self.t(h, src_key_padding_mask=pad)
        qi = (~pad).sum(1) - 1
        return self.o(h[torch.arange(x.shape[0]), qi])


def batchify(ctxs, V, query):
    L = max(len(c) for c in ctxs) + (1 if query else 0)
    x = np.full((len(ctxs), max(L, 1)), V + 1, dtype=np.int64)
    for i, c in enumerate(ctxs):
        seq = list(c) + ([V] if query else [])
        x[i, :len(seq)] = seq
    return torch.from_numpy(x)


def run_net(kind, V, blocks, items_after, seed, lr, ep):
    torch.manual_seed(90_000 + seed); torch.set_num_threads(1)
    net = MLP(V) if kind.startswith("MLP") else TF(V)
    query = kind.startswith("TF")
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    rng = np.random.default_rng(93_000 + seed)
    seen_s, out = [], []
    for si, b in enumerate(blocks):
        seen_s += b
        train = b if kind.endswith("-on") else seen_s
        ex = [([y for j, y in enumerate(s) if j != i], s[i]) for s in train for i in range(len(s)) if len(s) > 1]
        net.train()
        for _ in range(ep):
            perm = rng.permutation(len(ex))
            for k in range(0, len(ex), 64):
                bt = [ex[j] for j in perm[k:k + 64]]
                x = batchify([c for c, _ in bt], V, query); y = torch.tensor([t for _, t in bt])
                loss = nn.functional.cross_entropy(net(x), y)
                opt.zero_grad(); loss.backward(); opt.step()
        net.eval()
        rec = {}
        with torch.no_grad():
            for key, items in items_after[si].items():
                if not items:
                    rec[key] = np.nan; continue
                logits = net(batchify([cx for cx, _ in items], V, query)).numpy()
                hits = 0
                for (cx, t), lg in zip(items, logits):
                    lg = lg.copy(); lg[cx] = -np.inf
                    hits += int(int(np.argmax(lg)) == t)
                rec[key] = hits / len(items)
        out.append(rec)
    return out


def job(args):
    world, seed, what, hp, shuffle = args
    V, blocks, items = stream(world, seed, shuffle)
    if what == "P": return run_pursuit(V, blocks, items, seed)
    if what == "V": return run_pbv(V, blocks, items, seed)
    if what == "N":                                     # part-1 N, for the shuffle control only
        freq, co, out = np.zeros(V), np.zeros((V, V)), []
        for si, b in enumerate(blocks):
            for s in b:
                u = sorted(set(s)); freq[u] += 1; co[np.ix_(u, u)] += 1
            np.fill_diagonal(co, 0)
            norm = co / np.sqrt(np.outer(freq, freq) + 1e-12)
            out.append(score_sessions(V, [items[si]], [lambda cx, nm=norm: nm[cx].sum(0)])[0])
        return out
    return run_net(what, V, blocks, items, seed, *hp)


def mean_life(lives, key="never-heard"):
    return float(np.nanmean([[st[key] for st in lf] for lf in lives]))


if __name__ == "__main__":
    n_seeds = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    workers = int(sys.argv[2]) if len(sys.argv) > 2 else 2
    only = sys.argv[3].split(",") if len(sys.argv) > 3 else ["ascii", "life32"]
    pool = Pool(workers)
    for tag, world in (("B1", "ascii"), ("B2", "life32")):
        if world not in only:
            continue
        chosen = {}
        for net in NETS:
            tasks = [(world, sd, net, hp, False) for hp in GRID for sd in TUNE]
            res = pool.map(job, tasks)
            by = {hp: mean_life(res[i * len(TUNE):(i + 1) * len(TUNE)]) for i, hp in enumerate(GRID)}
            chosen[net] = max(GRID, key=lambda hp: by[hp])
            print(f"[{tag}] tune {net}: " + ", ".join(f"lr {hp[0]:g} ep {hp[1]} -> {v:.3f}" for hp, v in by.items())
                  + f" | chosen lr {chosen[net][0]:g} ep {chosen[net][1]}", flush=True)
        seeds = list(range(n_seeds))
        names = ["P", "V"] + list(NETS)
        tasks = [(world, sd, b, chosen.get(b), False) for b in names for sd in seeds]
        tasks += [(world, sd, b, chosen.get(b), True) for b in ("N", "TF-re") for sd in (seeds if b == "N" else seeds[:10])]
        res = pool.map(job, tasks)
        R = {}
        for i, (_, sd, b, _, sh) in enumerate(tasks):
            R.setdefault((b, sh), []).append(res[i])
        V = int(RL.build_world(world, 0).V)
        print(f"=== {tag} world {world}, {n_seeds} seeds; chance 1/V = {1 / V:.4f} ===")
        g = lambda key, b, sh=False: np.array([[st[key] for st in lf] for lf in R[(b, sh)]])
        for k in KEYS:
            for b in names:
                print(f"  {k:12} {b:7} " + " ".join(f"{v:.3f}" for v in np.nanmean(g(k, b), 0)))
            for b in ("N", "TF-re"):
                print(f"  {k:12} {b + '/shuf':10} " + " ".join(f"{v:.3f}" for v in np.nanmean(g(k, b, True), 0)))
        ref = REFS[world]
        have = [sd for sd in seeds if os.path.exists(os.path.join(RL.CACHE, f"seed{sd}_{ref}.pkl"))]
        vy = [pickle.load(open(os.path.join(RL.CACHE, f"seed{sd}_{ref}.pkl"), "rb")) for sd in have]
        for k in KEYS:
            v = np.array([[st.get(k, np.nan) for st in lf] for lf in vy])
            print(f"  {k:12} Voyager {ref} ({len(have)} seeds) " + " ".join(f"{x:.3f}" for x in np.nanmean(v, 0)))
            for b in names:
                bb = g(k, b)[have]
                d = v[:, :bb.shape[1]] - bb
                print(f"  {tag} {k}, Voyager - {b}: end {sgn(d[:, -1])} | life {sgn(np.nanmean(d, 1))}", flush=True)
    pool.close()
