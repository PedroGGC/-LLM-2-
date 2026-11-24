# IA2 Campanha 018 — `Lumina-V1-3B` (Etapa 1): Arquitetura Híbrida `3:1` de `3,36B` Parâmetros via `Attention-Surgery` na Kaggle `2× Tesla T4`

**Data:** 24/11/2026  
**Branch:** `llm-ideas`  
**Hardware Verificado:** Kaggle `2× NVIDIA Tesla T4` (`sm_75`, `14.56 GiB` VRAM/GPU, PyTorch `2.11.0+cu130`, `transformers 5.20.2`, `FP16`) + CPU local (`Windows`, PyTorch `2.14.1+cpu`).  
**Execuções Kaggle Preservadas (corpus 100% SFW EN Edu + Wikipédia PT-BR + FineMath):**
- Treino principal `Subfase 1 → Subfase 2a`: `aitherl/lumina-v1-lab` v9
- Continuação `Subfase 2b (seq=1536, LoRA r=8, freeze_deltanet_proj=True)` + merge LoRA + `eval_fp16`: `aitherl/lumina-v1-stage-2b-eval` (`STATUS: ERROR` apenas no watchdog de `3600s` ao compilar 288 camadas `HadamardLinearINT4` via Inductor; checkpoint final `subfase2b_merged` e bloco `eval_fp16` completos e gravados)
- Bateria completa `QuaRot Triton W4A16` sobre o checkpoint `subfase2b_merged`: `aitherl/lumina-v1-quarot-eval` v3 (`STATUS: COMPLETE` em `2082.7s`)

---

## 1. Arquitetura `Lumina-V1-3B` e Parâmetros Exatos

- **Modelo Professor (`Apache 2.0`, aberto):** `HuggingFaceTB/SmolLM3-3B-Base` (`36L`, `d_model=2048`, `H=16`, `KV=4`, `head_dim=128`, `hidden_dim=11008`, `vocab=128.256`, `tie_word_embeddings=True`, `no_rope_layer_interval=4`).
- **Modelo Estudante (`True 3:1 Hybrid` com pesos próprios após merge):**
  - `27` camadas `FractalDeltaCell` (`Gated DeltaNet-2` com RoPE *split-half* FP32 e portas desacopladas `beta_erase` / `beta_write`).
  - `9` camadas `SlidingWindowGQACell` **NoPE** nas posições `[3, 7, 11, 15, 19, 23, 27, 31, 35]` (`H=16`, `KV=4`, janela `W=1024`, `sink_tokens=4`).
  - **Contagem exata de parâmetros únicos:** **`3,362,026,449`** (validada por `student.parameter_formula() == unique_params` sem contagem dupla de embeddings).
  - **Estado recorrente constante $O(1)$ (`FP16`):** **`45.78 MiB`** por sequência (independe do comprimento `2K / 8K / 16K / 128K`, contra **`1,152.0 MiB`** de KV cache em `16K` para o professor denso `36L × 4 KV × 128`).

---

## 2. Convergência da Destilação (`6,47M` Tokens SFW Totais na `2× Tesla T4`)

| Fase | Passos | Tokens | Config / Memória `cuda:1` | Métrica Inicial → Final | Tempo (`tok/s`) |
|---|---:|---:|---|---|---:|
| **Transplant Passo 0 (9 SWA NoPE)** | `0` | — | `9` camadas NoPE `W=1024, S=4` | $\cos_{\text{attn}}$ = **`0.999754 .. 0.999996`** | `—` |
| **Subfase 1 (`AttnDelta` paralelo)** | `550` | `4,505,600` | `27` DeltaNet em `cuda:1`, FFNs na CPU (`9.378 GiB` pico) | `attn_loss: 0.9442 → 0.0895` · $\cos$: **`0.1183 → 0.9409`** | `7755.76s` (`580.9`) |
| **Subfase 2a (`E2E Full 36L`)** | `180` | `1,474,560` | `CompactAdamW`, `seq=512, bs=2, ga=8` (`13.267 GiB` pico) | `total: 5.3021 → 1.2897` · `ce: 4.3686 → 1.8980` · `kl: 3.0418 → 0.3260` | `6793.41s` (`217.1`) |
| **Subfase 2b (`LongCtx + LoRA r=8`)** | `40` | `491,520` | `seq=1536, ga=8, freeze_deltanet_proj=True` (`8.691 GiB` pico, `144` LoRAs mescladas) | `total: 1.2732 → 1.0090` · `ce: 1.9847 → 1.4673` · `kl: 0.2716 → 0.2668` · `lyr: 0.0615 → 0.0571` | `2377.21s` (`206.8`) |

> **Lição de engenharia validada (`freeze_deltanet_proj=True`):** congelar `qkv/out_proj/out_gate` das 27 DeltaNets após a Subfase 2a reduziu o pico na `cuda:1` de `13.91 GiB` (`OOM` em `seq=1536`) para **`8.691 / 14.56 GiB`**, viabilizando treino longo + co-adaptação de `144` adaptadores LoRA em FFNs/SWA com merge limpo (`0` `LoRALinear` remanescentes no checkpoint final).

---

## 3. Desempenho Empírico em `1× Tesla T4`: `FP16` vs `QuaRot Triton W4A16`

| Métrica (Single `Tesla T4`, `max_examples=150`/tarefa QA) | `Lumina-V1-3B FP16` (`Eager` / `Compiled`) | `Lumina-V1-3B QuaRot W4A16` (`Triton`) | Delta (`W4A16` vs `FP16`) |
|---|---:|---:|---:|
| **Tamanho do Modelo (GPU/Disco)** | `6.262 GiB` | **`2.135 GiB`** (`226` camadas `INT4`, `3.556×` comp. células) | **`−65.9%`** |
| **Estado Recorrente Constante $O(1)$** | `45.78 MiB` | `45.78 MiB` | Igual (`O(1)`) |
| **VRAM Pico no Decode (`64` tok)** | `6.364 GiB` (`6.352 GiB` compilado) | **`2.236 GiB`** (`2.575 GiB` pico em suite) | **`−64.9%`** |
| **Velocidade de Decode (`tok/s`)** | `11.8` eager / **`27.0` compilado** | `5.1` eager *(compile Triton não finalizado em T4)* | `2.29×` c/ compile FP16 |
| **Energia / Potência Média (NVML)** | `5819.25 mJ/tok` (`68.12 W`) eager / **`2361.86 mJ/tok` (`65.54 W`) compilado** | `7101.08 mJ/tok` (**`36.91 W`**) eager | **`−59.4%` energia** no FP16 compilado |
| **`LAMBADA` (`150` ex) Exact Match (`PPL` ↓)** | **`64.00%`** (`PPL = 4.15`) | `61.33%` (`PPL = 4.84`) | `−2.67 pp` |
| **`ARC-Easy` (`150` ex) `acc` / `acc_norm`** | **`74.00%`** / `66.67%` | `73.33%` / **`67.33%`** | **`+0.66 pp` `acc_norm`** |
| **`ARC-Challenge` (`150` ex) `acc` / `acc_norm`** | `50.00%` / `52.00%` | **`50.67%`** / **`52.67%`** | **`+0.67 pp`** |
| **`HellaSwag` (`150` ex) `acc` / `acc_norm`** | **`42.00%`** / **`58.67%`** | `41.33%` / **`58.67%`** | **`0.00 pp` `acc_norm`** |
| **`PIQA` (`150` ex) `acc` / `acc_norm`** | **`77.33%`** / **`79.33%`** | `76.67%` / **`79.33%`** | **`0.00 pp` `acc_norm`** |
| **`Winogrande` (`150` ex) `acc` / `acc_norm`** | `60.00%` / `61.33%` | **`60.67%`** / **`62.67%`** | **`+1.34 pp`** |
| **`RULER NIAH` `2K / 8K / 16K` EM (`NLL` ↓)** | `0.0% / 0.0% / 0.0%` (`4.28 / 4.12 / 4.40`) *(6 ex/len)* | `0.0% / 0.0% / 0.0%` (`4.32 / 4.50 / 4.37`) *(4 ex/len)* | Paridade em `0%` *(limitação Etapa 1)* |

### Amostras Reais de Geração Greedy (`64` tokens, `FP16`)
- **EN:** `"Explain in simple terms why the sky is blue:"` → *"The sky is blue because of the way light interacts with the Earth's atmosphere. When sunlight reaches the atmosphere, it is scattered by the molecules of air. The shorter wavelengths of light, such as blue and violet, are scattered more than the longer wavelengths, such as red and orange. This is why the sky appears blue"*
- **PT-BR:** `"Em português, explique brevemente o que é a fotossíntese:"` → *"a fotossíntese é um processo que ocorre nas plantas e em alguns microorganismos, onde a energia luminosa é convertida em energia química, produzindo glicose e liberando oxigênio. A fotossíntese é um processo fundamental para a vida"*
- **Python:** `'def fibonacci(n: int) -> int:\n    """Return the n-th Fibonacci number."""\n'` → *"    if n == 0:\n        return 0\n    elif n == 1:\n        return 1\n    else:\n        return fibonacci(n - 1) + fibonacci(n - 2)"*

---

## 4. Comparativo com Modelos Públicos Famosos (`2B–4B`)

> **Nota Metodológica Obrigatória:** Nossos números foram medidos empiricamente na Tesla T4 sobre os **primeiros `150` exemplos de validação** via `src/llm2/research/external_eval.py` (formato zero-shot direto, `max_examples=150`). Os números de referência dos modelos externos são os **reportados publicamente** em Model Cards do Hugging Face / artigos oficiais (`lighteval` / `lm-evaluation-harness` completos, onde há variações de *prompt template*, `acc` vs `acc_norm` e *few-shot*, especialmente `25-shot` em ARC-C ou `10-shot` em HellaSwag). Não executamos benchmarks na GPU para os modelos de terceiros.

| Modelo | Params / Arquitetura | KV/Estado @16K | `LAMBADA` ↑ | `ARC-Easy` ↑ | `ARC-C` ↑ | `HellaSwag` ↑ | `PIQA` ↑ | `Winogrande` ↑ |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| **`Lumina-V1-3B` (`FP16`, nosso)** | `3.36B` Híbrido 3:1 (`6.26 GiB`) | **`45.78 MiB` $O(1)$** | **`64.0%`** | `74.0%` (`66.7%` norm) | **`52.0%`** (`acc_norm`) | `58.7%` (`acc_norm`) | **`79.3%`** (`acc_norm`) | **`61.3%`** (`acc_norm`) |
| **`Lumina-V1-3B` (`QuaRot W4A16`, nosso)** | `3.36B` INT4 (`2.14 GiB`) | **`45.78 MiB` $O(1)$** | **`61.3%`** | `73.3%` (`67.3%` norm) | **`52.7%`** (`acc_norm`) | `58.7%` (`acc_norm`) | **`79.3%`** (`acc_norm`) | **`62.7%`** (`acc_norm`) |
| `SmolLM3-3B-Base` *(Professor, `lighteval` 0-shot)* | `3.07B` Transformer | `1,152 MiB` | `59.89%` | `~78%` *(ARC avg `65.61%`)* | `51.37%` | `76.15%` | `78.89%` | `58.88%` |
| `Qwen2.5-3B` *(Base/IT público)* | `3.09B` Transformer | Linear $O(T)$ | `~` | `77–78%` | `45–48%` | `71–73%` | `78–79%` | `68–73%` |
| `Qwen3-4B-Base` *(HF `lm-eval` 0-shot)* | `4.0B` Transformer | Linear $O(T)$ | `~` | `78.49%` | `54.01%` | `68.45%` | `~` | `66.14%` |
| `Gemma-2-2B` *(relatório oficial)* | `2.6B` Transformer | Linear $O(T)$ | `~` | `~` | `55.7%` *(25-shot)* | `72.9%` *(10-shot)* | `78.4–78.8%` | `71.3%` *(5-shot)* |
| `Llama-3.2-3B` *(público 0/few-shot)* | `3.21B` Transformer | Linear $O(T)$ | `~` | `~` | `69.1%` *(25-shot)* | `69.8%` *(0-shot)* | `79–81%` | `~66–70%+` |

---

## 5. Limitações Empíricas Conhecidas (Base Honesta para a Etapa 2)

1. **`RULER NIAH (2K/8K/16K)` em `0.0%` Exact Match:** com apenas `6,47M` tokens totais (dos quais `491K` em `seq=1536` e zero SFT sintético de *needle-in-a-haystack*), as 9 janelas `W=1024` + estados recorrentes ainda não aprendem *exact string copy* de agulhas aleatórias fora de `W=1024`. Exige currículo estendido (`4K–16K`) + dados SFT de recuperação antes de alegar precisão real em RULER.
2. **`HellaSwag` (`58.7%`) abaixo do professor (`76.15%`):** há diferença de protocolo (`150` exemplos de validação com `acc_norm` vs `lighteval` completo), além de o orçamento curto de `6,47M` tokens ter deixado margem para fechamento na fase de SFT/continual pretraining.
3. **`torch.compile` sobre `QuaRot W4A16 Triton`:** a compilação Inductor de `226` matrizes `HadamardLinearINT4` estourou o guarda de `3600s` na T4; portanto, no perfil `QuaRot INT4` apenas o modo eager (`5.1 tok/s`, `2.236 GiB` VRAM, `36.91 W`) foi empiricamente verificado, enquanto `FP16 Compiled` atingiu **`27.0 tok/s`**.
