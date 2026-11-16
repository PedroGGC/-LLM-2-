# IA2 Campanha 011 (NVIDIA L4 — Roteiro Fase 1: Self-Speculative MTP + Telemetria Real NVML/DVFS + Currículo `512→1024→2048`)

Data: 2026-11-16  
Branch: `llm-ideas` (revisão executada na L4: `08fbf1f`)  
Hardware: `NVIDIA L4` (`23.034 MiB`, faixa de Power Limit `40,00 W`–`72,00 W`), CUDA `13.0`, PyTorch `2.14.1+cu130`, `BF16`, contador elétrico de hardware NVML (`nvmlDeviceGetTotalEnergyConsumption` + amostragem contínua de `Watts` / `°C` / `SM MHz`), dataset TinyStories BPE train-only `vocab=4096` (`/content/stories_bpe_cache/prepared-bpe-v4096-96af1443b0104ead-37be307a`).

---

## 1. Impacto do Currículo `512 → 1024 → 2048` no Campeão `7.45M` (`seed=351`, `3.500` passos)

Com o currículo escalado (`0..1749` em `512×16`, `1750..2624` em `1024×8`, `2625..3499` em `2048×4` e `writes=48` no estágio final, mantendo `8.192` tokens/passo) sobre o `fractal_pcc_mtp_muon_bpe`:

- **Checkpoint SHA-256 (`model.safetensors`):** `54bd2deb6c481cc4a6f4cc3bf2d8ad25f464681d5b430d2ab3f7cca7d78a06c5`
- **Parâmetros:** `7,445,215` (inferência) / `10,590,943` (treino com 2 cabeças auxiliares MTP) · **Estado:** `844,808 B/seq` · **VRAM pico de treino:** `3,922,641,408 B` (`3,65 GiB`)

| Modelo | `val@256` BPB ↓ | `val@512` BPB ↓ | `val@3219` BPB ↓ | Banda `1–256` ↓ | Banda `257–512` ↓ | Banda `513–2048` ↓ | `ΔTrunc` ↑ | `ΔSwap` ↑ | Bind `8k w=32` ↑ | Bind `HARD w=64` (acc / par) ↑ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `transformer_bpe_med` (C010, `15.74M`, `n=3`) | `1.0688` | `1.0402` | `1.0361` | `2.8959` | `2.7106` | `3.2903` | `+0.0217` | `+0.0753` | `1.50%` | `1.79% / 0.00%` |
| `fractal_pcc_mtp_muon_bpe` sem currículo (C010, `n=3`) | `1.0451` | `1.0018` | `0.9978` | `2.7784` | `2.6259` | `3.8526` | `+0.0056` | `+0.0861` | **`100.00%`** | `99.15% / 98.54%` |
| **`fractal_pcc_mtp_muon_bpe` + Currículo (C011, `s351`)** | **`0.9148`** | **`0.9016`** | **`0.8993`** | **`2.5177`** | **`2.3728`** | **`2.4578`** | **`+0.0502`** | **`+0.1310`** | **`100.00%`** (`nll=0.00025`) | **`99.12% / 98.73%`** |

### O que mudou com o currículo
1. **Quebrou `0.900 BPB` (`0.8993 BPB` em `val@3219`, `2.5080 NLL`)** usando apenas `7.45M` parâmetros em inferência (`-0.1386 BPB` abaixo do Transformer de `15.74M` em `val@512`).
2. **Eliminou o gargalo da banda `513–2048`:** caiu de `3.8526` NLL (C010) para **`2.4578` NLL** (`-1.3948` NLL; agora supera até o Transformer `3.2903` por **`-0.8325` NLL**).
3. **Uso causal real do prefixo disparou:** `ΔTrunc` passou de `+0.0056` → **`+0.0502`** e `ΔSwap` de `+0.0861` → **`+0.1310`**.
4. **Custo exposto para a Fase 2:** o treino completo levou `2.386,3s` (`39,8 min`, `45,3 min` totais com evals), com média final de apenas `4.869 tok/s` ao entrar em `2048×4` + 3 recompilações discretas do `torch.compile`. **Acelerar `dplr_chunked` com kernel fundido (Fase 2) passa a ser prioridade antes de escalar para `50M–150M+`.**

---

## 2. Medição Elétrica Real (`NVML`) + `Self-Speculative MTP` + `L2 Pinning` + `DVFS`

Varredura direta no checkpoint treinado (`step-00003500`, `model.safetensors` + `objective.safetensors`), gerando **`1.024` tokens contíguos** a partir de prefixo real de validação (`data.arrays["val"][:256]`) após warmup em todas as configurações:

- **Paridade determinística (`greedy`):** **`100%` idêntica** (`spec_toks == base_toks` em todos os `1.024` tokens × 4 execuções) + suíte `benchmark_decode` em prefixo sintético verificada em paridade (`16,8%` aceitação aleatória -> `1,344.2` vs `1,148.9 tok/s`, `1.17×`).
- **Taxa real de aceitação MTP em texto real de validação:** **`31,1%`** (`1,62 tokens/ciclo` com `mtp_tokens=2` e `mtp_weight=0.25`).

| Power Limit (`nvidia-smi -pl`) | `L2 Pin` (`cudaDeviceSetLimit`) | Baseline `tok/s` | Baseline `Watts` | Baseline `mJ/tok` | **Speculative `tok/s`** | **Speculative `Watts`** | **Speculative `mJ/tok`** | Temp (`°C`) | SM Clock (`MHz`) | Ganho `Tokens/J` vs Baseline `72W` |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `72 W` (100% TDP) | `False` | `1.158,5` | `36,94 W` | `40,66` | **`1.662,8`** (`1,44×`) | `46,52 W` | `31,89` (`-21,6%`) | `59,9–60,0` | `2.040` | `1,28×` |
| `72 W` (100% TDP) | `True` (`7,70 MB`) | `1.157,9` | `47,35 W` | `40,77` | **`1.666,2`** (`1,44×`) | `46,71 W` | `31,85` (`-21,9%`) | `60,5–60,9` | `2.040` | `1,28×` |
| `55 W` (76% TDP) | `True` (`7,70 MB`) | `1.150,7` | `37,24 W` | `39,94` | **`1.646,7`** (`1,42×`) | `45,66 W` | `27,26` (`-33,1%`) | `60,7–61,0` | `2.040` | `1,50×` |
| **`40 W` (56% TDP — mín. L4)** | `True` (`7,70 MB`) | `1.090,4` | `34,94 W` | `35,77` | **`1.562,6`** (`1,35×` vs base 72W!) | `39,74 W` | **`23,45` (`-42,5%`)** | `60,2–60,8` | `1.826 / 1.797` | **`1,74×` (`42,6` vs `24,5 tok/J`)** |

*(Observação honesta sobre `NVML` em bursts curtos de ~`0,6s`: `nvmlDeviceGetTotalEnergyConsumption` é exato em milizoules acumulados, enquanto `watts_mean` instantâneo amostrado a cada `10 ms` pode incluir pequena latência de transição térmica/boost da GPU; por isso `mJ/token` calculado pelo contador acumulado de hardware `E1 - E0` é a métrica mais confiável.)*

### Achados práticos desta etapa (além da teoria do guia)
1. **Self-Speculative MTP + Power Capping funciona de verdade, mas abaixo da projeção teórica ingênua:** em vez dos `70%` / `2,04×` simulados no guia, nossas cabeças `mtp_weight=0.25` na escala `7.45M` atingiram **`31,1%` de aceitação (`1,62 tok/ciclo`)**, gerando **`1,44×` speedup** a `72W` e **`1,35×` speedup com `-42,5%` de energia (`23,45` vs `40,77 mJ/tok`)** ao travar a L4 no piso de `40 W` (`1,74×` mais `Tokens/Joule`). Aumentar a capacidade/peso das cabeças MTP é candidato direto em escala maior.
2. **L2 Weight Pinning em `7.45M` (`cudaDeviceSetLimit`):** embora o driver CUDA tenha aceitado reservar `7,70 MB` (`cudaDeviceSetLimit(0x06) == 0`), a diferença de velocidade (`1.662,8` vs `1.666,2 tok/s`) foi desprezível (~`0,2%`), porque os ~`15 MB` totais do modelo já cabem passivamente dentro dos `48 MB` de L2 da L4.
3. **Bug descoberto no `ia2_benchmark.py` pós-rodada (corrigido em seguida):** `_evaluate_stories_and_decode` buscava `data.val.tokens` em vez de `data.arrays["val"]`, fazendo a avaliação integrada embutida cair no fallback aleatório (`16,8%`) enquanto nossa varredura dedicada leu `data.arrays["val"][:256]` real (`31,1%`). Corrigido para as próximas rodadas.
