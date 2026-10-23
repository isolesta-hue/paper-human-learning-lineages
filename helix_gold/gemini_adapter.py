"""Gemini's own PyTorch implementation, saved verbatim as the project's
"sub research subject" (per project decision, 2026-09-22): Claude's numpy
codebase (imitation_core.py + run_sequential_experiment.py) is the MAIN
research track; this file and its variants are a SEPARATE track, tracking
an independently-built adapter design that Claude's own "adapter" condition
in run_sequential_experiment.py was modeled after (see finding 18).

This file is intentionally unmodified from what was pasted into the
conversation, so it stays the faithful baseline to diff future edits
against. See gemini_adapter_nonoverlap.py for the variant with deliberately
non-overlapping per-stage vocabulary.
"""
import torch
import torch.nn as nn
import torch.optim as optim
import copy
import torch.nn.functional as F

# --- 1. 코퍼스 및 토크나이저 정의 ---
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
        "gradient descent optimizes the weights",
        "tokens are embedded into hidden spaces"
    ],
    [
        "the sun rises in the east",
        "ancient history teaches us lessons",
        "oceans cover the earth surface",
        "mountains stand tall against the wind"
    ]
]

all_sentences = [sent for stage in stages_corpus for sent in stage]
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

# --- 2. 어댑터 지원 텔레메트리 신경망 ---
class AdapterTelemetrySystem(nn.Module):
    def __init__(self, vocab_size, hidden_dim):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim

        # [Base Network] 기초 지식 뼈대
        self.embedding = nn.Embedding(vocab_size, hidden_dim, padding_idx=0)
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=4, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.reconstructor = nn.Linear(hidden_dim, vocab_size)

        # [Adapters] 스테이지별 우회 가중치
        self.adapters = nn.ModuleDict()
        self.active_stage = None

    def freeze_base_network(self):
        """기초 뼈대 가중치 완전히 동결 (기초 지식 보호)"""
        for param in self.embedding.parameters():
            param.requires_grad = False
        for param in self.encoder.parameters():
            param.requires_grad = False
        for param in self.reconstructor.parameters():
            param.requires_grad = False

    def add_stage_adapter(self, stage_name, bottleneck_dim=64):
        """새로운 도메인을 위한 전용 병목 어댑터 생성"""
        adapter = nn.Sequential(
            nn.Linear(self.hidden_dim, bottleneck_dim),
            nn.GELU(),
            nn.Linear(bottleneck_dim, self.hidden_dim)
        )
        # 초기 항등 사상(Identity Mapping)을 위해 마지막 레이어 0 초기화
        nn.init.zeros_(adapter[2].weight)
        nn.init.zeros_(adapter[2].bias)
        self.adapters[stage_name] = adapter
        self.active_stage = stage_name

    def forward(self, x):
        emb = self.embedding(x)
        latent = self.encoder(emb)

        # 활성화된 어댑터가 있다면 우회 지식 덧붙이기 (Residual Link)
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
print("📡 [어댑터 동결 제어 + 텔레메트리] 듀얼 루프 심층 모니터링 가동")
print(f" - 전체 어휘 사전 크기 (Vocab Size): {vocab_size}")
print(f" - 잠재 벡터 차원 (Latent Dim):     {hidden_dim}")
print("="*85)

model_a = AdapterTelemetrySystem(vocab_size, hidden_dim)  # 어댑터 방어 탑재
model_b = copy.deepcopy(model_a)                         # 순수 계속 학습 대조군

optimizer_b = optim.Adam(model_b.parameters(), lr=0.005)
criterion = nn.CrossEntropyLoss(ignore_index=0)

clean_benchmark_data = text_to_tensor(stages_corpus[0])

# 초기 관측 기준선 산출
model_a.eval()
with torch.no_grad():
    init_bench_recon, baseline_latent = model_a(clean_benchmark_data)
    best_benchmark_loss = criterion(init_bench_recon.view(-1, vocab_size), clean_benchmark_data.view(-1)).item()
model_a.train()

frozen_baseline_latent = baseline_latent.clone().detach()

print(f"\n📌 [초기 관측 기준선 스냅샷]")
print(f" - 순수 벤치마크 기준 Loss : {best_benchmark_loss:.6f}")
print(f" - 잠재 벡터 평균 Norm      : {baseline_latent.norm(dim=-1).mean().item():.4f}")
print(f" - 잠재 벡터 분산 (Variance): {torch.var(baseline_latent, dim=[0, 1]).mean().item():.6f}")

total_stages = len(stages_corpus)
epochs_per_stage = 40

for stage in range(1, total_stages + 1):
    stage_name = f"Stage_{stage}"
    stage_sentences = stages_corpus[stage - 1]
    current_data = text_to_tensor(stage_sentences)

    print("\n" + "="*85)
    print(f"📂 [{stage_name}/{total_stages}] 도메인 학습 개시")
    print(f"   학습 문장 목록:")
    for idx, s in enumerate(stage_sentences, 1):
        print(f"     {idx}. \"{s}\"")
    print("="*85)

    # --- Model A 학습 준비 (어댑터 및 동결 설정) ---
    if stage == 1:
        # Stage 1: Base 전체 학습
        optimizer_a = optim.Adam(model_a.parameters(), lr=0.005)
    else:
        # Stage 2, 3: Base 완전히 얼리고 전용 Adapter만 학습!
        model_a.freeze_base_network()
        model_a.add_stage_adapter(stage_name)
        optimizer_a = optim.Adam(model_a.adapters[stage_name].parameters(), lr=0.005)
        print(f"🔒 [Model A] Base 가중치 동결 완료 ➔ '{stage_name}' 전용 Adapter 활성화")

    model_a_checkpoint = copy.deepcopy(model_a.state_dict())

    for epoch in range(1, epochs_per_stage + 1):
        # --- 1. Model A 학습 및 상상 루프 ---
        optimizer_a.zero_grad()
        recon_a, latent_a = model_a(current_data)
        loss_a = criterion(recon_a.view(-1, vocab_size), current_data.view(-1))
        loss_a.backward()
        grad_norm_a = get_grad_norm(model_a)
        optimizer_a.step()

        # 상상 탐색 루프
        with torch.no_grad():
            noise = torch.randn_like(latent_a) * 0.03
            imag_latent = latent_a + noise
            imag_recon = model_a.reconstructor(imag_latent)
            imag_loss = criterion(imag_recon.view(-1, vocab_size), current_data.view(-1)).item()
            perturbation_dist = noise.norm().item()
            imag_delta = imag_loss - loss_a.item()

        # --- 2. Model B 학습 (대조군: 무제한 계속 학습) ---
        optimizer_b.zero_grad()
        recon_b, latent_b = model_b(current_data)
        loss_b = criterion(recon_b.view(-1, vocab_size), current_data.view(-1))
        loss_b.backward()
        grad_norm_b = get_grad_norm(model_b)
        optimizer_b.step()

        # --- 3. 실시간 고밀도 로그 (매 10 에폭) ---
        if epoch % 10 == 0 or epoch == 1:
            with torch.no_grad():
                var_a = torch.var(latent_a, dim=[0, 1]).mean().item()
                drift_a = torch.norm(latent_a - frozen_baseline_latent).item()
                cos_sim_a = F.cosine_similarity(latent_a.mean(dim=1), frozen_baseline_latent.mean(dim=1), dim=1).mean().item()

            print(f" ⏱️ [Epoch {epoch:2d}/{epochs_per_stage}]")
            print(f"   ├─ [Model A (Adapter)] Train Loss: {loss_a.item():.5f} | Grad Norm: {grad_norm_a:.4f}")
            print(f"   │   ├─ 🧠 잠재 공간 진단: Variance={var_a:.5f} | Drift={drift_a:.3f} | CosSim={cos_sim_a:.4f}")
            print(f"   │   └─ ✨ 상상 탐색 루프: Imag Loss={imag_loss:.5f} (ΔLoss={imag_delta:+.5f}) | Perturb Dist={perturbation_dist:.3f}")
            print(f"   └─ [Model B (Control)] Train Loss: {loss_b.item():.5f} | Grad Norm: {grad_norm_b:.4f}")
            print("   " + "-"*80)

    # --- 4. 스테이지 종료 심사 ---
    model_a.eval()
    model_b.eval()
    with torch.no_grad():
        # A그룹 평가 시 기초 벤치마크 검증을 위해 active_stage 해제 (Base 평가)
        current_active = model_a.active_stage
        model_a.active_stage = None
        bench_a, new_latent_a = model_a(clean_benchmark_data)
        current_bench_loss_a = criterion(bench_a.view(-1, vocab_size), clean_benchmark_data.view(-1)).item()
        model_a.active_stage = current_active  # 원상 복구

        bench_b, new_latent_b = model_b(clean_benchmark_data)
        current_bench_loss_b = criterion(bench_b.view(-1, vocab_size), clean_benchmark_data.view(-1)).item()

        drift_final_a = torch.norm(new_latent_a - frozen_baseline_latent).item()
        drift_final_b = torch.norm(new_latent_b - frozen_baseline_latent).item()
    model_a.train()
    model_b.train()

    print(f"\n📊 [방어 위원회 심층 감사 보고서 ({stage_name} 종료)]")
    print(f" ├─ [A그룹 (Base 동결 + Adapter 흡수)]")
    print(f" │   ├─ 벤치마크 Loss 변화 : {best_benchmark_loss:.6f} ➔ {current_bench_loss_a:.6f}")
    print(f" │   └─ Stage Train Loss   : {loss_a.item():.6f}")
    print(f" ├─ [B그룹 (순수 대조군)]")
    print(f" │   ├─ 벤치마크 Loss 변화 : {current_bench_loss_b:.6f}")
    print(f" │   └─ Stage Train Loss   : {loss_b.item():.6f}")

    # 판정 및 기준선 업데이트
    if current_bench_loss_a <= best_benchmark_loss + 1e-4:
        best_benchmark_loss = current_bench_loss_a
        print(" └─ ⚖️ [판정 결과: ✅ 승인 및 성공적 지식 확장 (Knowledge Extension)]")
        print("      -> 기존 지식 훼손 0%! 성공적으로 새 어댑터 모듈이 결합되었습니다.")
    else:
        print(" └─ ⚖️ [판정 결과: 🚨 강제 롤백 집행 (Rollback)]")
        model_a.load_state_dict(model_a_checkpoint)

print("\n" + "="*85)
print("🏁 [어댑터 듀얼 루프 관측 최종 종료]")
print("="*85)
