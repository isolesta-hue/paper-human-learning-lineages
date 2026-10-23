"""Variant of gemini_adapter.py with a DELIBERATELY non-overlapping
vocabulary across stages -- no word (not even function words like "the",
"is", "are") appears in more than one stage's sentences.

Why: Claude's numpy replica of this same adapter idea (run_sequential_
experiment.py's "adapter" condition) got PERFECT forgetting protection
(0.000 forgetting, 300/300 seeds) but nearly total failure to learn stage 2
(accuracy 0.235, near chance for 6 classes) when stage 2's input
distribution was constructed to share NOTHING with stage 1's. Gemini's own
log showed both perfect protection AND good new-task fit (Stage_2 train
loss 0.187, actually lower than the fully-fine-tuned control's 1.478) --
the suspected reason for the difference is that Gemini's stages, as
originally written, share real vocabulary (the word "the" appears in both
Stage_2 and Stage_3's sentences, so its embedding row -- already refined
during whichever stage first used it -- carries usable signal into the
next stage even though the whole embedding table is frozen after stage 1).

This file removes every such overlap (see word lists in the docstring below
each stage) so Gemini's own architecture faces the same "genuinely
disjoint domains" condition that broke Claude's numpy adapter. If Model A
here ALSO loses most of its new-task fit once vocabulary sharing is
removed, that confirms shared structure -- not the adapter mechanism
itself -- was what let the original demo look good on both axes at once.
"""
import torch
import torch.nn as nn
import torch.optim as optim
import copy
import torch.nn.functional as F

# --- 1. 코퍼스 -- 스테이지 간 단어가 단 하나도 겹치지 않도록 재작성 ---
# Stage 1 words: artificial, intelligence, is, growing, fast, machine,
#   learning, models, require, data, deep, changes, our, world, hello,
#   how, are, you, today
# Stage 2 words: neural, networks, process, information, transformers,
#   use, attention, mechanisms, gradient, descent, optimizes, model,
#   weights, tokens, get, embedded, into, hidden, spaces
#   ("the" -> "model", "are" -> "get" so nothing re-uses a Stage-1 word)
# Stage 3 words: sun, rises, over, eastern, hills, ancient, history,
#   teaches, us, lessons, oceans, cover, vast, earth, surface, mountains,
#   stand, tall, against, strong, wind
#   ("the" removed/reworded throughout)
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

all_sentences = [sent for stage in stages_corpus for sent in stage]
words = sorted(list(set(" ".join(all_sentences).split())))
word2idx = {word: i + 1 for i, word in enumerate(words)}
vocab_size = len(word2idx) + 1
hidden_dim = 256
seq_len = 8

# Sanity check this file's whole point: assert no word appears in more
# than one stage's sentence set.
_stage_word_sets = [set(" ".join(stage).split()) for stage in stages_corpus]
for i in range(len(_stage_word_sets)):
    for j in range(i + 1, len(_stage_word_sets)):
        overlap = _stage_word_sets[i] & _stage_word_sets[j]
        assert not overlap, f"Stage {i+1} and Stage {j+1} share words: {overlap}"

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

# --- 2. 어댑터 지원 텔레메트리 신경망 (gemini_adapter.py와 동일) ---
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

def get_grad_norm(model):
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None and p.requires_grad:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    return total_norm ** 0.5

# --- 3. 시뮬레이션 가동 ---
print("="*85)
print("📡 [비중첩 어휘 버전] 스테이지 간 단어 0% 공유 -- 어댑터가 여전히 두 목표를 다 달성하는지 관측")
print(f" - 전체 어휘 사전 크기 (Vocab Size): {vocab_size}")
print(f" - 잠재 벡터 차원 (Latent Dim):     {hidden_dim}")
print("="*85)

model_a = AdapterTelemetrySystem(vocab_size, hidden_dim)
model_b = copy.deepcopy(model_a)

optimizer_b = optim.Adam(model_b.parameters(), lr=0.005)
criterion = nn.CrossEntropyLoss(ignore_index=0)

clean_benchmark_data = text_to_tensor(stages_corpus[0])

model_a.eval()
with torch.no_grad():
    init_bench_recon, baseline_latent = model_a(clean_benchmark_data)
    best_benchmark_loss = criterion(init_bench_recon.view(-1, vocab_size), clean_benchmark_data.view(-1)).item()
model_a.train()

frozen_baseline_latent = baseline_latent.clone().detach()

print(f"\n📌 [초기 관측 기준선 스냅샷]")
print(f" - 순수 벤치마크 기준 Loss : {best_benchmark_loss:.6f}")

total_stages = len(stages_corpus)
epochs_per_stage = 40

stage_final_train_loss_a = {}
stage_final_train_loss_b = {}
stage_bench_loss_a = {}
stage_bench_loss_b = {}

for stage in range(1, total_stages + 1):
    stage_name = f"Stage_{stage}"
    stage_sentences = stages_corpus[stage - 1]
    current_data = text_to_tensor(stage_sentences)

    print("\n" + "="*85)
    print(f"📂 [{stage_name}/{total_stages}] 도메인 학습 개시 (단어 0% 중첩)")
    print("="*85)

    if stage == 1:
        optimizer_a = optim.Adam(model_a.parameters(), lr=0.005)
    else:
        model_a.freeze_base_network()
        model_a.add_stage_adapter(stage_name)
        optimizer_a = optim.Adam(model_a.adapters[stage_name].parameters(), lr=0.005)
        print(f"🔒 [Model A] Base 가중치 동결 완료 ➔ '{stage_name}' 전용 Adapter 활성화")

    model_a_checkpoint = copy.deepcopy(model_a.state_dict())

    for epoch in range(1, epochs_per_stage + 1):
        optimizer_a.zero_grad()
        recon_a, latent_a = model_a(current_data)
        loss_a = criterion(recon_a.view(-1, vocab_size), current_data.view(-1))
        loss_a.backward()
        optimizer_a.step()

        optimizer_b.zero_grad()
        recon_b, latent_b = model_b(current_data)
        loss_b = criterion(recon_b.view(-1, vocab_size), current_data.view(-1))
        loss_b.backward()
        optimizer_b.step()

        if epoch % 10 == 0 or epoch == 1 or epoch == epochs_per_stage:
            print(f" ⏱️ [Epoch {epoch:2d}/{epochs_per_stage}] "
                  f"Model A loss={loss_a.item():.5f}  Model B loss={loss_b.item():.5f}")

    stage_final_train_loss_a[stage_name] = loss_a.item()
    stage_final_train_loss_b[stage_name] = loss_b.item()

    model_a.eval()
    model_b.eval()
    with torch.no_grad():
        current_active = model_a.active_stage
        model_a.active_stage = None
        bench_a, _ = model_a(clean_benchmark_data)
        current_bench_loss_a = criterion(bench_a.view(-1, vocab_size), clean_benchmark_data.view(-1)).item()
        model_a.active_stage = current_active

        bench_b, _ = model_b(clean_benchmark_data)
        current_bench_loss_b = criterion(bench_b.view(-1, vocab_size), clean_benchmark_data.view(-1)).item()
    model_a.train()
    model_b.train()

    stage_bench_loss_a[stage_name] = current_bench_loss_a
    stage_bench_loss_b[stage_name] = current_bench_loss_b

    print(f"\n📊 [{stage_name} 종료]")
    print(f" ├─ [A (Adapter)] 벤치마크 Loss: {best_benchmark_loss:.6f} -> {current_bench_loss_a:.6f}  "
          f"| Stage Train Loss: {loss_a.item():.6f}")
    print(f" └─ [B (Control)]  벤치마크 Loss: {current_bench_loss_b:.6f}  "
          f"| Stage Train Loss: {loss_b.item():.6f}")

    if current_bench_loss_a <= best_benchmark_loss + 1e-4:
        best_benchmark_loss = current_bench_loss_a

print("\n" + "="*85)
print("=== 요약: 비중첩 어휘에서도 어댑터가 두 목표(보호 + 새 과제 학습)를 다 달성하는가? ===")
print("="*85)
for stage_name in stage_final_train_loss_a:
    print(f"{stage_name}: A train_loss={stage_final_train_loss_a[stage_name]:.4f}  "
          f"B train_loss={stage_final_train_loss_b[stage_name]:.4f}  "
          f"(A bench_loss={stage_bench_loss_a[stage_name]:.6f}, B bench_loss={stage_bench_loss_b[stage_name]:.6f})")
print("\n읽는 법: A의 bench_loss가 Stage_1 이후로도 안 변해야 '보호'가 유지된 것.")
print("A의 train_loss가 B보다 크게 나쁘면(특히 어휘 중첩 버전과 비교해서), 원본 데모의")
print("'보호와 학습을 동시에 달성'이 어휘 공유 덕분이었다는 가설이 뒷받침됨.")
