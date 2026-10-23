"""Does Gemini's adapter (Model A) actually GENERALIZE to new stage-2/3
sentences, or does it only reproduce the exact sentences it was trained on?

Gemini's own script (gemini_adapter.py / gemini_adapter_nonoverlap.py)
never checks this: every loss it reports -- including the "benchmark" --
is measured on the EXACT SAME sentences that were just trained on. There
is no train/val split anywhere in that code. Claude's own numpy replica
(run_sequential_experiment.py) DOES hold out a validation split, and that
adapter condition failed badly at the new stage (0.235 accuracy, near
chance for 6 classes) even after the vocabulary-overlap hypothesis was
ruled out by gemini_adapter_nonoverlap.py's result (Model A still fit its
OWN training sentences well with zero vocabulary overlap).

Hypothesis this file tests: the discrepancy is that Gemini's setup only
ever measures memorization of a handful of training sentences, which a
small adapter can do well in a wide (256-dim) latent space regardless of
domain shift ("blessing of dimensionality" -- almost any handful of points
in a high-dim space can be separated by a big enough frozen random
projection). Claude's setup measures GENERALIZATION to held-out val
category-instances the model never saw, which is a fundamentally harder
bar. If Model A reconstructs its own stage-2/3 training sentences well but
fails on a NEW sentence built from the same stage's word pool in a
different combination, that confirms this is memorization, not learning.

Built on gemini_adapter_nonoverlap.py (zero cross-stage vocabulary overlap
already confirmed not to be the deciding factor) -- this adds one held-out
sentence per stage, built from words already in that stage's own trained
vocabulary but in a combination never seen during training.
"""
import torch
import torch.nn as nn
import torch.optim as optim
import copy

stages_corpus = [
    [
        "artificial intelligence is growing fast",
        "machine learning models require data",
        "deep learning changes our world",
        "hello how are you today"
    ],
    [
        "neural networks process information",
        "transformers use attention mechanisms",
        "gradient descent optimizes model weights",
        "tokens get embedded into hidden spaces"
    ],
    [
        "sun rises over eastern hills",
        "ancient history teaches us lessons",
        "oceans cover vast earth surface",
        "mountains stand tall against strong wind"
    ]
]

# One held-out sentence per stage -- same stage's own word pool, recombined
# into an order/pairing never seen during that stage's training.
heldout_sentences = {
    1: "hello growing fast today world",
    2: "attention networks process weights",
    3: "oceans teaches tall eastern wind",
}

all_sentences = [sent for stage in stages_corpus for sent in stage]
words = sorted(list(set(" ".join(all_sentences).split())))
word2idx = {word: i + 1 for i, word in enumerate(words)}
vocab_size = len(word2idx) + 1
hidden_dim = 256
seq_len = 8

# Confirm every held-out word is actually in-vocabulary (built from that
# stage's own trained words) and that no stage shares vocabulary with any
# other -- same guarantee as gemini_adapter_nonoverlap.py.
_stage_word_sets = [set(" ".join(stage).split()) for stage in stages_corpus]
for i in range(len(_stage_word_sets)):
    for j in range(i + 1, len(_stage_word_sets)):
        assert not (_stage_word_sets[i] & _stage_word_sets[j])
for stage_num, sent in heldout_sentences.items():
    unknown = set(sent.split()) - _stage_word_sets[stage_num - 1]
    assert not unknown, f"held-out sentence for stage {stage_num} uses out-of-pool words: {unknown}"
    assert sent not in stages_corpus[stage_num - 1], f"held-out sentence for stage {stage_num} was literally trained on"


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
    mask = target != 0  # ignore padding
    return (pred[mask] == target[mask]).float().mean().item()


print("=" * 85)
print("held-out generalization check -- same word pool, never-trained combination")
print(f"vocab_size={vocab_size}  hidden_dim={hidden_dim}")
print("=" * 85)

model_a = AdapterTelemetrySystem(vocab_size, hidden_dim)
model_b = copy.deepcopy(model_a)
optimizer_b = optim.Adam(model_b.parameters(), lr=0.005)
criterion = nn.CrossEntropyLoss(ignore_index=0)

epochs_per_stage = 40
results = []

for stage in range(1, 4):
    stage_name = f"Stage_{stage}"
    current_data = text_to_tensor(stages_corpus[stage - 1])
    heldout_data = text_to_tensor([heldout_sentences[stage]])

    if stage == 1:
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
        train_recon_a, _ = model_a(current_data)
        train_acc_a = token_accuracy(train_recon_a, current_data)
        heldout_recon_a, _ = model_a(heldout_data)
        heldout_acc_a = token_accuracy(heldout_recon_a, heldout_data)
        heldout_loss_a = criterion(heldout_recon_a.view(-1, vocab_size), heldout_data.view(-1)).item()

        train_recon_b, _ = model_b(current_data)
        train_acc_b = token_accuracy(train_recon_b, current_data)
        heldout_recon_b, _ = model_b(heldout_data)
        heldout_acc_b = token_accuracy(heldout_recon_b, heldout_data)
        heldout_loss_b = criterion(heldout_recon_b.view(-1, vocab_size), heldout_data.view(-1)).item()
    model_a.train()
    model_b.train()

    print(f"\n[{stage_name}] held-out sentence: \"{heldout_sentences[stage]}\"")
    print(f"  A (adapter) : train_token_acc={train_acc_a:.3f}  heldout_token_acc={heldout_acc_a:.3f}  heldout_loss={heldout_loss_a:.4f}")
    print(f"  B (control) : train_token_acc={train_acc_b:.3f}  heldout_token_acc={heldout_acc_b:.3f}  heldout_loss={heldout_loss_b:.4f}")

    results.append({
        "stage": stage_name,
        "train_acc_a": train_acc_a, "heldout_acc_a": heldout_acc_a,
        "train_acc_b": train_acc_b, "heldout_acc_b": heldout_acc_b,
    })

print("\n" + "=" * 85)
print("=== summary: train-set fit vs held-out generalization ===")
print("=" * 85)
for r in results:
    print(f"{r['stage']}: A train={r['train_acc_a']:.3f} -> heldout={r['heldout_acc_a']:.3f}  "
          f"(gap={r['train_acc_a']-r['heldout_acc_a']:+.3f})   "
          f"B train={r['train_acc_b']:.3f} -> heldout={r['heldout_acc_b']:.3f}  "
          f"(gap={r['train_acc_b']-r['heldout_acc_b']:+.3f})")
print("\nIf A's heldout_token_acc craters relative to its train_token_acc (a big positive gap),")
print("while train-set fit stays high, that's memorization, not generalization -- matching")
print("Claude's numpy adapter's near-chance validation accuracy despite good training fit.")
