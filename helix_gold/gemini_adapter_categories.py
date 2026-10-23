"""Gemini's adapter architecture, now fed the SAME category structure
Claude's numpy code uses (animal/vehicle/food/color/emotion/weather), so
the two codebases share content built from the same underlying semantic
structure -- without forcing either one to adopt the other's evaluation
method. Per project decision (2026-09-22): share the category structure,
keep each codebase's own native evaluation (Gemini: reconstruction loss/
token accuracy; Claude: classification accuracy) -- they ask different
questions and don't need to be merged into one metric.

Stage design mirrors run_sequential_experiment.py's OVERLAP_CATEGORIES=3
test exactly: categories 0-2 (animal, vehicle, food) are "shared" --
they appear in BOTH stage 1 and stage 2, with DIFFERENT sentences each
time (same category, new specific content, the way Claude's version
redraws noise around the same center). Categories 3-5 (color, emotion,
weather) are "novel" -- introduced for the first time in stage 2, with no
stage-1 exposure at all. One held-out sentence per category (all 6) is
never trained on at all, used only to measure true generalization.

This is a direct extension of gemini_adapter_heldout.py's held-out check,
now organized by category so "shared" vs "novel" can be measured
separately -- the same breakdown that showed, in Claude's numpy adapter
(300 seeds), shared-category accuracy 0.914 vs novel-category accuracy
0.394. Does Gemini's actual transformer+adapter, measured by ITS OWN
metric (held-out token reconstruction accuracy), show the same split?
"""
import torch
import torch.nn as nn
import torch.optim as optim
import copy

CATEGORIES = ["animal", "vehicle", "food", "color", "emotion", "weather"]
SHARED_CATEGORIES = {"animal", "vehicle", "food"}   # appear in stage 1 AND stage 2
NOVEL_CATEGORIES = {"color", "emotion", "weather"}  # appear ONLY in stage 2

# Every category gets exactly 6 STAGE-2 sentences (what the adapter -- the
# only thing that can learn anything new once the base is frozen -- gets
# to train on), so shared and novel categories get an EQUAL adapter
# training budget. Shared categories additionally get 3 STAGE-1 sentences
# (learned by the fully-trainable base, before it's frozen); novel
# categories have no stage-1 sentences at all (introduced only in stage 2).
# A first version split each shared category's 6 total sentences 3/3
# between stage1/stage2, which meant shared categories got only 3 stage-2
# sentences vs novel categories' 6 -- a confound that made "novel"
# generalize BETTER than "shared" (0.711 vs 0.386), backwards from the
# hypothesis, simply because shared categories were adapter-data-starved.
# This version gives every category the same 6-sentence stage-2 budget.
CATEGORY_SENTENCES = {
    "animal": {
        "stage1": [
            "the lion hunts at night",
            "birds fly across the sky",
            "dogs bark loudly outside",
        ],
        "stage2": [
            "fish swim in rivers daily",
            "elephants roam the savanna slowly",
            "wolves howl under moonlight tonight",
            "birds swim across the savanna",
            "dogs roam under the sky",
            "wolves bark loudly at fish",
        ],
        "heldout": [
            "the wolves fly across rivers",
            "elephants howl at night slowly",
            "fish bark loudly under moonlight",
        ],
    },
    "vehicle": {
        "stage1": [
            "cars drive fast on highways",
            "trains carry passengers daily",
            "airplanes fly above clouds",
        ],
        "stage2": [
            "ships sail across oceans slowly",
            "bicycles roll through streets daily",
            "trucks haul heavy cargo fast",
            "cars carry heavy cargo daily",
            "trains roll through busy highways",
            "airplanes haul passengers above clouds",
        ],
        "heldout": [
            "trucks sail through highways daily",
            "ships fly above oceans slowly",
            "bicycles haul heavy cargo fast",
        ],
    },
    "food": {
        "stage1": [
            "bread bakes in hot ovens",
            "soup simmers on the stove",
            "apples grow on tall trees",
        ],
        "stage2": [
            "rice feeds many hungry people",
            "cheese melts over warm pasta",
            "honey drips from golden combs",
            "bread simmers in golden ovens",
            "soup feeds many tall people",
            "apples melt on the stove",
        ],
        "heldout": [
            "cheese melts on warm bread",
            "apples grow on the stove",
            "honey drips from warm bread",
        ],
    },
    "color": {
        "stage1": [],
        "stage2": [
            "red paint covers the wall",
            "blue skies stretch above hills",
            "green leaves cover the forest",
            "yellow flowers bloom in spring",
            "purple grapes hang on vines",
            "orange sunsets fade slowly",
        ],
        "heldout": [
            "yellow leaves fade above hills",
            "purple flowers bloom on vines",
            "orange grapes hang above hills",
        ],
    },
    "emotion": {
        "stage1": [],
        "stage2": [
            "joy fills her happy heart",
            "anger rises during heated arguments",
            "fear grips him in darkness",
            "sadness lingers after quiet goodbyes",
            "pride swells after great victories",
            "calm settles over tired minds",
        ],
        "heldout": [
            "joy rises during quiet victories",
            "fear settles over tired minds",
            "pride fills her happy heart",
        ],
    },
    "weather": {
        "stage1": [],
        "stage2": [
            "rain falls softly on rooftops",
            "storms brew over distant mountains",
            "snow covers frozen winter fields",
            "wind howls through empty streets",
            "fog blankets the quiet valley",
            "sunshine warms the early morning",
        ],
        "heldout": [
            "wind falls over frozen streets",
            "fog covers distant mountains softly",
            "sunshine warms the empty valley",
        ],
    },
}

for cat, d in CATEGORY_SENTENCES.items():
    assert len(d["stage2"]) == 6, f"{cat} must have exactly 6 stage-2 sentences (equal adapter budget), got {len(d['stage2'])}"
    all_words = set(" ".join(d["stage1"] + d["stage2"]).split())
    for h in d["heldout"]:
        unknown = set(h.split()) - all_words
        assert not unknown, f"{cat}'s held-out sentence '{h}' uses never-trained words: {unknown}"
        assert h not in d["stage1"] + d["stage2"], f"{cat}'s held-out sentence '{h}' was literally trained on"

# Stage 1: shared categories' 3 dedicated stage-1 sentences (novel
# categories contribute nothing here -- they don't exist yet).
# Stage 2: EVERY category's 6 dedicated stage-2 sentences -- shared and
# novel categories get an equal adapter-training budget.
stage1_sentences, stage1_cats = [], []
stage2_sentences, stage2_cats = [], []
for cat in CATEGORIES:
    stage1_sentences += CATEGORY_SENTENCES[cat]["stage1"]
    stage1_cats += [cat] * len(CATEGORY_SENTENCES[cat]["stage1"])
    stage2_sentences += CATEGORY_SENTENCES[cat]["stage2"]
    stage2_cats += [cat] * len(CATEGORY_SENTENCES[cat]["stage2"])

all_sentences = stage1_sentences + stage2_sentences + [
    h for d in CATEGORY_SENTENCES.values() for h in d["heldout"]
]
words = sorted(list(set(" ".join(all_sentences).split())))
word2idx = {word: i + 1 for i, word in enumerate(words)}
vocab_size = len(word2idx) + 1
hidden_dim = 256
seq_len = 8


def text_to_tensor(sentence_list, max_len=seq_len):
    tensor_list = []
    for sent in sentence_list:
        tokens = [word2idx.get(w, 0) for w in sent.split()]
        if len(tokens) < max_len:
            tokens += [0] * (max_len - len(tokens))
        else:
            tokens = tokens[:max_len]
        tensor_list.append(tokens)
    return torch.tensor(tensor_list, dtype=torch.long)


class AdapterTelemetrySystem(nn.Module):
    def __init__(self, vocab_size, hidden_dim):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.embedding = nn.Embedding(vocab_size, hidden_dim, padding_idx=0)
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=4, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.reconstructor = nn.Linear(hidden_dim, vocab_size)
        self.adapters = nn.ModuleDict()
        self.active_stage = None

    def freeze_base_network(self):
        for param in self.embedding.parameters():
            param.requires_grad = False
        for param in self.encoder.parameters():
            param.requires_grad = False
        for param in self.reconstructor.parameters():
            param.requires_grad = False

    def add_stage_adapter(self, stage_name, bottleneck_dim=64):
        adapter = nn.Sequential(
            nn.Linear(self.hidden_dim, bottleneck_dim),
            nn.GELU(),
            nn.Linear(bottleneck_dim, self.hidden_dim)
        )
        nn.init.zeros_(adapter[2].weight)
        nn.init.zeros_(adapter[2].bias)
        self.adapters[stage_name] = adapter
        self.active_stage = stage_name

    def forward(self, x):
        emb = self.embedding(x)
        latent = self.encoder(emb)
        if self.active_stage and self.active_stage in self.adapters:
            latent = latent + self.adapters[self.active_stage](latent)
        recon = self.reconstructor(latent)
        return recon, latent


def token_accuracy(recon, target):
    pred = recon.argmax(dim=-1)
    mask = target != 0
    return (pred[mask] == target[mask]).float().mean().item()


print("=" * 85)
print("category-unified vocabulary: same 6 categories as Claude's numpy code")
print(f"shared (stage1+stage2): {sorted(SHARED_CATEGORIES)}   novel (stage2 only): {sorted(NOVEL_CATEGORIES)}")
print(f"vocab_size={vocab_size}  hidden_dim={hidden_dim}  "
      f"stage1 sentences={len(stage1_sentences)}  stage2 sentences={len(stage2_sentences)}")
print("=" * 85)

epochs_per_stage = 150  # more than the original 40 -- more sentences per stage now,
# and 60 wasn't enough for the adapter to even fit its own stage-2 training set well
clean_benchmark_data = text_to_tensor(stage1_sentences)
criterion = nn.CrossEntropyLoss(ignore_index=0)


def run_once(seed):
    torch.manual_seed(seed)
    model_a = AdapterTelemetrySystem(vocab_size, hidden_dim)
    model_b = copy.deepcopy(model_a)
    optimizer_b = optim.Adam(model_b.parameters(), lr=0.005)

    for stage_name, sentences in [("Stage_1", stage1_sentences), ("Stage_2", stage2_sentences)]:
        current_data = text_to_tensor(sentences)

        if stage_name == "Stage_1":
            optimizer_a = optim.Adam(model_a.parameters(), lr=0.005)
        else:
            model_a.freeze_base_network()
            model_a.add_stage_adapter(stage_name)
            optimizer_a = optim.Adam(model_a.adapters[stage_name].parameters(), lr=0.005)

        for epoch in range(1, epochs_per_stage + 1):
            optimizer_a.zero_grad()
            recon_a, _ = model_a(current_data)
            loss_a = criterion(recon_a.view(-1, vocab_size), current_data.view(-1))
            loss_a.backward()
            optimizer_a.step()

            optimizer_b.zero_grad()
            recon_b, _ = model_b(current_data)
            loss_b = criterion(recon_b.view(-1, vocab_size), current_data.view(-1))
            loss_b.backward()
            optimizer_b.step()

    model_a.eval()
    model_b.eval()
    with torch.no_grad():
        current_active = model_a.active_stage
        model_a.active_stage = None
        bench_recon_a, _ = model_a(clean_benchmark_data)
        bench_acc_a = token_accuracy(bench_recon_a, clean_benchmark_data)
        model_a.active_stage = current_active

        bench_recon_b, _ = model_b(clean_benchmark_data)
        bench_acc_b = token_accuracy(bench_recon_b, clean_benchmark_data)

        shared_acc_a, novel_acc_a, shared_acc_b, novel_acc_b = [], [], [], []
        for cat in CATEGORIES:
            heldout = text_to_tensor(CATEGORY_SENTENCES[cat]["heldout"])
            recon_a, _ = model_a(heldout)
            acc_a = token_accuracy(recon_a, heldout)
            recon_b, _ = model_b(heldout)
            acc_b = token_accuracy(recon_b, heldout)
            (shared_acc_a if cat in SHARED_CATEGORIES else novel_acc_a).append(acc_a)
            (shared_acc_b if cat in SHARED_CATEGORIES else novel_acc_b).append(acc_b)

    return {
        "bench_acc_a": bench_acc_a, "bench_acc_b": bench_acc_b,
        "shared_a": sum(shared_acc_a) / len(shared_acc_a), "novel_a": sum(novel_acc_a) / len(novel_acc_a),
        "shared_b": sum(shared_acc_b) / len(shared_acc_b), "novel_b": sum(novel_acc_b) / len(novel_acc_b),
    }


SEEDS = list(range(10))
results = []
for seed in SEEDS:
    r = run_once(seed)
    results.append(r)
    print(f"[seed {seed}] bench_a={r['bench_acc_a']:.3f} bench_b={r['bench_acc_b']:.3f}  "
          f"A: shared={r['shared_a']:.3f} novel={r['novel_a']:.3f}  "
          f"B: shared={r['shared_b']:.3f} novel={r['novel_b']:.3f}")

print("\n" + "=" * 85)
print(f"=== summary over {len(SEEDS)} seeds ===")
print("=" * 85)


def mean_std(key):
    vals = [r[key] for r in results]
    return sum(vals) / len(vals), (sum((v - sum(vals) / len(vals)) ** 2 for v in vals) / len(vals)) ** 0.5


for key, label in [("bench_acc_a", "A benchmark (forgetting check)"), ("bench_acc_b", "B benchmark (forgetting check)"),
                    ("shared_a", "A shared-category heldout"), ("novel_a", "A novel-category heldout"),
                    ("shared_b", "B shared-category heldout"), ("novel_b", "B novel-category heldout")]:
    m, s = mean_std(key)
    print(f"{label:36}: {m:.3f} +/- {s:.3f}")

n_shared_wins = sum(1 for r in results if r["shared_a"] > r["novel_a"])
print(f"\nA: seeds where shared-category heldout_acc > novel-category heldout_acc: {n_shared_wins}/{len(SEEDS)}")
print("Compare directly to Claude's numpy adapter (300 seeds): shared-cat acc=0.914, novel-cat acc=0.394")
