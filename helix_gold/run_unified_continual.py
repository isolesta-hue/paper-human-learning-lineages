"""One vocabulary, two tasks, three learners, three consecutive stages.

Vocabulary: 72 real word names in 6 categories (12 each). Every word also
carries a synthetic "meaning vector" (category center + noise), so a word
the model has never trained on can still be classified by generalization
-- a pure ID/lookup embedding would make that impossible.

Two tasks built from the SAME vocabulary, trained jointly every stage:
  - sentence task (Gemini-style): fixed 4-word sentences made of one
    category's words; one position is masked, the model must name the
    masked word from the rest of the sentence.
  - classification task (Claude-style): which category is this word in.

Three stages, two new categories each (class-incremental):
  stage 1 animal+vehicle, stage 2 food+color, stage 3 emotion+weather.

Three learners on an identical backbone, differing only in protection:
  vanilla -- plain MLP, everything trainable every stage.
  claude  -- identity axes: the first ID_DIM latent axes come from a fixed,
             never-trained branch of the raw input (Module A's frozen core);
             the rest adapt. Plus the ratchet: within each stage keep the
             best-validation snapshot.
  gemini  -- stage 1 trains everything; from stage 2 the whole base
             (projection, encoder, both heads) is frozen and a zero-init
             bottleneck adapter per stage is trained instead. Picking the
             right adapter needs the stage id at test time ("oracle"), which
             the other learners never get, so both routings are reported.
"""
import copy
import math
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.set_num_threads(1)

CATEGORIES = ["animal", "vehicle", "food", "color", "emotion", "weather"]
WORD_NAMES = {
    "animal": ["dog", "cat", "lion", "tiger", "wolf", "fox", "bear", "horse", "eagle", "shark", "rabbit", "deer"],
    "vehicle": ["car", "bus", "train", "truck", "bicycle", "ship", "airplane", "boat", "taxi", "tram", "scooter", "subway"],
    "food": ["bread", "rice", "apple", "cheese", "soup", "pasta", "honey", "pizza", "salad", "noodle", "cake", "egg"],
    "color": ["red", "blue", "green", "yellow", "purple", "orange", "pink", "brown", "black", "white", "gray", "gold"],
    "emotion": ["joy", "anger", "fear", "sadness", "pride", "calm", "hope", "shame", "love", "envy", "guilt", "relief"],
    "weather": ["rain", "snow", "wind", "fog", "storm", "sunshine", "thunder", "hail", "frost", "cloud", "drizzle", "heat"],
}
STAGES = [["animal", "vehicle"], ["food", "color"], ["emotion", "weather"]]
N_TRAIN, N_VAL, N_TEST = 8, 2, 2  # per category: 12 words
SENTENCES_PER_CAT = 10
SENT_LEN = 4

RAW_DIM = 40
HIDDEN = 64
LATENT = 32
ID_DIM = 6  # same 3/16 share of the latent as imitation_core's identity_dim=3
BOTTLENECK = 16
NOISE = 0.2
EPOCHS = 200
CHECK_EVERY = 10
LR = 0.01
METHODS = ["vanilla", "claude", "gemini"]

N_CLASSES = len(CATEGORIES)
V = sum(len(w) for w in WORD_NAMES.values())


def build_data(seed):
    rng = np.random.default_rng(seed)
    centers = rng.uniform(0.2, 0.8, (N_CLASSES, RAW_DIM))
    vecs, cats, names = [], [], []
    split = {}
    wid = 0
    for c, cat in enumerate(CATEGORIES):
        ids = []
        for name in WORD_NAMES[cat]:
            vecs.append(np.clip(centers[c] + rng.normal(0, NOISE, RAW_DIM), 0, 1))
            cats.append(c)
            names.append(name)
            ids.append(wid)
            wid += 1
        ids = list(rng.permutation(ids))
        split[cat] = {"train": ids[:N_TRAIN], "val": ids[N_TRAIN:N_TRAIN + N_VAL],
                      "test": ids[N_TRAIN + N_VAL:]}
        sents = [list(rng.choice(split[cat]["train"], SENT_LEN, replace=False)) for _ in range(SENTENCES_PER_CAT)]
        split[cat]["sentences"] = sents
    return torch.tensor(np.array(vecs), dtype=torch.float32), torch.tensor(cats), names, split


def stage_tensors(vecs, cats, split, stage_cats, part):
    ids = [i for cat in stage_cats for i in split[cat][part]]
    ids = torch.tensor(ids)
    return vecs[ids].unsqueeze(1), cats[ids]


def sentence_tensors(vecs, split, stage_cats):
    """Every sentence appears SENT_LEN times, once with each position masked."""
    xs, targets, positions = [], [], []
    for cat in stage_cats:
        for sent in split[cat]["sentences"]:
            base = vecs[torch.tensor(sent)]
            for p in range(SENT_LEN):
                x = base.clone()
                x[p] = 0.0
                xs.append(x)
                targets.append(int(sent[p]))
                positions.append(p)
    return torch.stack(xs), torch.tensor(targets), torch.tensor(positions)


class Net(nn.Module):
    def __init__(self, method):
        super().__init__()
        self.method = method
        self.proj = nn.Linear(RAW_DIM, HIDDEN)
        free_dim = LATENT - ID_DIM if method == "claude" else LATENT
        self.enc = nn.Linear(2 * HIDDEN, free_dim)
        if method == "claude":
            self.identity = nn.Linear(RAW_DIM, ID_DIM, bias=False)
            self.identity.weight.requires_grad_(False)
        self.recon = nn.Linear(LATENT, V)
        self.cls = nn.Linear(LATENT, N_CLASSES)
        self.adapters = nn.ModuleList()

    def add_adapter(self):
        a = nn.Sequential(nn.Linear(LATENT, BOTTLENECK), nn.GELU(), nn.Linear(BOTTLENECK, LATENT))
        nn.init.zeros_(a[2].weight)
        nn.init.zeros_(a[2].bias)
        self.adapters.append(a)
        return a

    def freeze_base(self):
        for p in self.parameters():
            p.requires_grad_(False)

    def forward(self, x, adapter=None):
        e = torch.tanh(self.proj(x))
        ctx = e.mean(dim=1, keepdim=True).expand_as(e)
        z = torch.tanh(self.enc(torch.cat([e, ctx], dim=-1)))
        if self.method == "claude":
            z = torch.cat([torch.tanh(self.identity(x)), z], dim=-1)
        if adapter is not None:
            z = z + adapter(z)
        return self.recon(z), self.cls(z)


def cls_acc(model, x, y, adapter=None):
    with torch.no_grad():
        _, logits = model(x, adapter)
        return (logits[:, 0].argmax(-1) == y).float().mean().item()


def sent_acc(model, x, t, p, adapter=None):
    with torch.no_grad():
        logits, _ = model(x, adapter)
        picked = logits[torch.arange(len(t)), p]
        return (picked.argmax(-1) == t).float().mean().item()


def curve_stats(values, gain_fraction=0.9, early=3):
    arr = np.maximum.accumulate(np.array(values))
    start, final = arr[0], arr[-1]
    target = start + gain_fraction * (final - start)
    idx = next(i for i, v in enumerate(arr) if v >= target - 1e-9)
    deltas = np.diff(arr)
    return {"checks_to_90": idx, "early_slope": float(np.mean(deltas[:early])) if len(deltas) else 0.0}


def run_method(method, data, seed):
    vecs, cats, names, split = data
    torch.manual_seed(seed)
    model = Net(method)

    per_stage = []  # tensors for every stage, built once
    for stage_cats in STAGES:
        per_stage.append({
            "train": stage_tensors(vecs, cats, split, stage_cats, "train"),
            "val": stage_tensors(vecs, cats, split, stage_cats, "val"),
            "test": stage_tensors(vecs, cats, split, stage_cats, "test"),
            "sent": sentence_tensors(vecs, split, stage_cats),
        })

    stage_adapters = []  # gemini only: adapter used for stage s items (None = base)
    acc_matrix = {k: np.full((len(STAGES), len(STAGES)), np.nan) for k in
                  ["cls_test", "cls_train", "sent", "cls_test_latest", "sent_latest"]}
    curves = []

    for s, d in enumerate(per_stage):
        adapter = None
        if method == "gemini" and s > 0:
            model.freeze_base()
            adapter = model.add_adapter()
            params = list(adapter.parameters())
        else:
            params = [p for p in model.parameters() if p.requires_grad]
        opt = torch.optim.Adam(params, lr=LR)

        xw, yw = d["train"]
        xs, ts, ps = d["sent"]
        best_score, best_state = -1.0, None
        cls_curve, sent_curve = [], []

        for epoch in range(EPOCHS + 1):
            if epoch % CHECK_EVERY == 0:
                cv = cls_acc(model, *d["val"], adapter)
                sv = sent_acc(model, xs, ts, ps, adapter)
                cls_curve.append(cv)
                sent_curve.append(sv)
                if method == "claude" and cv + sv > best_score:
                    best_score, best_state = cv + sv, copy.deepcopy(model.state_dict())
            if epoch == EPOCHS:
                break
            opt.zero_grad()
            _, cl = model(xw, adapter)
            rl, _ = model(xs, adapter)
            loss = F.cross_entropy(cl[:, 0], yw) + F.cross_entropy(rl[torch.arange(len(ts)), ps], ts)
            loss.backward()
            opt.step()

        if method == "claude" and best_state is not None:
            model.load_state_dict(best_state)

        stage_adapters.append(adapter)
        curves.append({"cls": curve_stats(cls_curve), "sent": curve_stats(sent_curve)})

        for j in range(s + 1):
            routed = stage_adapters[j]
            dj = per_stage[j]
            acc_matrix["cls_test"][s, j] = cls_acc(model, *dj["test"], routed)
            acc_matrix["cls_train"][s, j] = cls_acc(model, *dj["train"], routed)
            acc_matrix["sent"][s, j] = sent_acc(model, *dj["sent"], routed)
            acc_matrix["cls_test_latest"][s, j] = cls_acc(model, *dj["test"], adapter)
            acc_matrix["sent_latest"][s, j] = sent_acc(model, *dj["sent"], adapter)

    return {"acc": acc_matrix, "curves": curves}


def sign_p(n_pos, n_neg):
    n = n_pos + n_neg
    if n == 0:
        return float("nan")
    k = min(n_pos, n_neg)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) * 0.5 ** n)


def summarize(results, seeds):
    last = len(STAGES) - 1

    def ms(vals):
        vals = np.array(vals, dtype=float)
        return f"{np.nanmean(vals):.3f}+/-{np.nanstd(vals):.3f}"

    print(f"\n=== {len(seeds)} seeds, {EPOCHS} epochs/stage ===")
    for key, title in [("cls_test", "classification of NEVER-TRAINED words (generalization)"),
                       ("cls_train", "classification of trained words (memory)"),
                       ("sent", "masked-word sentence task")]:
        print(f"\n--- {title}: accuracy on each stage's data, measured after stage 3 ---")
        print(f"{'method':>16} " + " ".join(f"{'stage'+str(j+1):>15}" for j in range(len(STAGES))) + f"{'average':>15}")
        rows = [(m, key) for m in METHODS]
        if key in ("cls_test", "sent"):
            rows.append(("gemini", key + "_latest"))
        for m, k in rows:
            label = m if not k.endswith("_latest") else "gemini(no-oracle)"
            per = [ms([r[m]["acc"][k][last, j] for r in results]) for j in range(len(STAGES))]
            avg = ms([np.nanmean(r[m]["acc"][k][last]) for r in results])
            print(f"{label:>16} " + " ".join(f"{v:>15}" for v in per) + f"{avg:>15}")

    print("\n--- forgetting of stage 1 (acc right after stage 1 minus acc after stage 3) ---")
    for key in ["cls_test", "sent"]:
        for m in METHODS:
            f = [r[m]["acc"][key][0, 0] - r[m]["acc"][key][last, 0] for r in results]
            print(f"{key:>9} {m:>8}: {ms(f)}")

    print("\n--- learning curve per stage (current-stage accuracy while training it) ---")
    for task in ["cls", "sent"]:
        for s in range(len(STAGES)):
            parts = []
            for m in METHODS:
                c90 = ms([r[m]["curves"][s][task]["checks_to_90"] for r in results])
                sl = ms([r[m]["curves"][s][task]["early_slope"] for r in results])
                parts.append(f"{m}: checks_to_90={c90} early_slope={sl}")
            print(f"{task} stage{s+1}  " + "  |  ".join(parts))

    print("\n--- paired comparison vs vanilla, same seed (final average over the 3 stages) ---")
    for key in ["cls_test", "sent"]:
        for m in ["claude", "gemini"]:
            diffs = np.array([np.nanmean(r[m]["acc"][key][last]) - np.nanmean(r["vanilla"]["acc"][key][last])
                              for r in results])
            n_pos, n_neg = int((diffs > 1e-9).sum()), int((diffs < -1e-9).sum())
            print(f"{key:>9} {m:>7}-vanilla: mean={diffs.mean():+.3f}  better/worse={n_pos}/{n_neg}  "
                  f"sign-test p={sign_p(n_pos, n_neg):.3g}")


def main():
    n_seeds = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    seeds = list(range(n_seeds))
    results = []
    for seed in seeds:
        data = build_data(seed)
        results.append({m: run_method(m, data, seed) for m in METHODS})
    summarize(results, seeds)


if __name__ == "__main__":
    main()
