# IA2 Campanha 017 — Auditoria Cirúrgica de `Attention-Surgery` (`RADLADS` 1:1 em `SmolLM2-135M`) & Fase D `MetaQueryBridge` (`Nano Lumina`)

**Data:** 23/11/2026  
**Branch:** `llm-ideas`  
**Hardware Verificado:** CPU local (`Windows`, PyTorch `2.14.1+cpu`) + **Google Colab `NVIDIA Tesla T4` (`sm_75`, `15 GB`, PyTorch `2.14.1+cu130`, `FP16`/`FP32`)** (`runtime` encerrado imediatamente após coleta com `list_runtimes() == {}`).

---

## 1. Os 5 Bugs Raízes Identificados e Corrigidos na Auditoria Pós-V1

A primeira rodada de `Attention-Surgery` em `SmolLM2-135M` (`30L`, `vocab=49.152`, `GQA 3:1`) havia colapsado para `Score = 10.67` (`CE inicial = 13.45`, pior que chute uniforme $\ln(49152) = 10.80$). A auditoria linha a linha provou que a causa principal eram **5 erros concretos de código**, todos corrigidos e validados na `Tesla T4`:

1. **Bug #1 — `_expand_gqa_kv` embaralhava os canais das cabeças $K$ e $V$ em modelos GQA ([`src/llm2/research/distill_convert.py`](../../src/llm2/research/distill_convert.py)):**
   - Aplicar `w.repeat_interleave(3, dim=0)` diretamente na matriz 2D `(192, 576)` repetia linhas escalares (`[c0, c0, c0, c1, c1, c1, ...]`) em vez de blocos de cabeça de `64` dimensões.
   - **Correção:** remodelar para `(n_kv_heads, head_dim, d_in)` antes de `repeat_interleave(rep, dim=0)`, preservando cada cabeça $K$ e $V$ intacta (`erro máximo = 0.000000` vs `0.375218` no código antigo).
2. **Bug #2 — `F.mse_loss(F.normalize(...))` dividia o gradiente de `layer_mse` por `d_model = 576` e explodia em *Attention Sinks* ([`src/llm2/research/distill_convert.py`](../../src/llm2/research/distill_convert.py)):**
   - O `F.mse_loss` sobre vetores unitários dividia a perda por `576` (`0.0017`, apenas `0.017%` do gradiente total). Além disso, na camada `L11` do `SmolLM2-135M` ocorre um *attention sink* com norma `22.901` no token inicial, que explodia o L2 quadrático relativo para `307.12`.
   - **Correção:** `_repr_alignment_loss` combina distância de cosseno $(1 - \cos(s_l, t_l))$ invariante à dimensão com `F.smooth_l1_loss(log(||s_l||), log(||t_l||))`, imune a *outliers* de *attention sink*.
3. **Bug #3 — `out_gate` aleatório sem bias + `head_gamma = 1.0` destruíam a escala de $W_O$ no Passo 0 ([`src/llm2/architectures/fractal_delta.py`](../../src/llm2/architectures/fractal_delta.py), [`src/llm2/research/distill_convert.py`](../../src/llm2/research/distill_convert.py)):**
   - Multiplicar `y` (de RMS unitário `1.0`) por `F.silu(W_rand @ normed)` invertia/zerava metade dos canais antes de $W_O$ (`out_proj`), e na camada `L00` do `SmolLM2-135M` (`||h_in|| = 2.74`, `||T_attn|| = 2.59`) injetava uma atualização `||S_attn|| = 44.08` (**`17×` maior que o professor**).
   - **Correção:** buffer `out_gate_bias = 1.784376` ($\text{SiLU}(1.784376) = 1.0$) + `out_gate.weight` inicializado em `std=1e-4` + `calibrate_surgery_attention_scales` calibrando `cell.head_gamma` pela razão real `||T_attn_l|| / ||S_attn_l||` no primeiro lote. No Passo 0, `L00` passou a ter **`||S_attn|| = 2.56` vs `||T_attn|| = 2.59`** e **`cos_E2E = 0.9685` (`96.85%`)**!
4. **Bug #4 — Diluição do resíduo $h_l^{\text{in}}$ e erro acumulado em 30 camadas vs Subfase 1 `AttnDelta` do `RADLADS` ([`src/llm2/architectures/fractal_delta.py`](../../src/llm2/architectures/fractal_delta.py), [`src/llm2/research/distill_convert.py`](../../src/llm2/research/distill_convert.py)):**
   - Nas camadas intermediárias (`L02..L28`), o fluxo residual `||h_in||` (`63 → 754`) é `~10×` maior que a atualização de atenção `||T_attn||` (`11 → 90`). Supervisionar $h_l^{\text{in}} + \Delta_l$ diluía o erro da atenção em `10×` (na `L20`, `cos(u_s, u_t)` parecia `0.8851` mesmo com `cos(S_attn, T_attn) = -0.0892` negativo!).
   - **Correção:** adicionado `forward_level_attn_delta` (pulando as 30 MLPs `SwiGLU` congeladas) e captura exata de $\Delta_l^{\text{attn\_teacher}} = u_l^{\text{teacher}} - h_l^{\text{in}}$ via pre-hook em `post_attention_layernorm`. Na Subfase 1, todas as 30 camadas alinham diretamente $\Delta_l^{\text{attn\_student}}(h_l^{\text{in}})$ contra $\Delta_l^{\text{attn\_teacher}}(h_l^{\text{in}})$ em paralelo (`~2.5×` mais rápido), seguido pela Subfase 2 ponta-a-ponta (`KL + CE + layer_mse`).
5. **Bug #5 — Contagem de parâmetros `QuaRot INT4` e isolamento de VRAM no Ringue ([`src/llm2/research/ringue_eval.py`](../../src/llm2/research/ringue_eval.py)):**
   - Adicionado `_count_logical_parameters` (contabilizando `2 * packed_weight.numel()` nos buffers `uint8` do `HadamardLinearINT4`, corrigindo `28.9M → 163.2M`) e medição isolada de VRAM (`_model_storage_bytes + decode_delta`).
6. **Bônus de Hardware Turing (`sm_75`, Tesla T4) ([`src/llm2/architectures/triton_dplr.py`](../../src/llm2/architectures/triton_dplr.py)):**
   - Guardião `shared_memory_per_block_optin < 96 KiB` em `can_use_triton_dplr` para evitar `OutOfResources (73.728–90.112 B > 65.536 B)` quando `head_dim = 64` na T4.

---

## 2. Verificação em CPU Local (`12L`, `GQA 3:1`, `H=9, KV=3`)

Antes de subir para a GPU, a correção foi provada em CPU num professor `12L` com a mesma topologia **GQA 3:1 (`H=9, KV=3`)** do `SmolLM2-135M` (`Teacher val_ce = 0.7924`, `val_acc = 83.01%`, chute aleatório `CE = 5.5452`):

| Estado (`12L GQA 3:1`, CPU) | `val_ce` ↓ | `val_kl` ↓ | `val_acc` ↑ | Acordo c/ Professor ↑ | `final_cos` ↑ |
|---|---:|---:|---:|---:|---:|
| **Passo 0 — Código Antigo (Bugs #1 e #3)** | `8.1061` (> acaso!) | `7.3341` | `5.76%` | `6.15%` | `0.1307` |
| **Passo 0 — Código Corrigido (Fixes #1 e #3, 0 passos)** | **`2.6425`** | **`1.9038`** | **`36.43%`** | **`43.46%`** | **`0.6694`** |
| **Após 60 Passos (`RADLADS` 2 Fases: Fixes #1–#4)** | **`0.7751`** | **`0.0806`** | **`83.11%`** | **`91.60%`** | **`0.9597`** |

---

## 3. Resultado Real na `Tesla T4` (`SmolLM2-135M` `30L`, `vocab=49.152`): Antes vs Depois

### 3.1 Convergência das 30 Camadas (`150` passos Subfase 1 `AttnDelta` + `60` passos Subfase 2 `E2E` = `210` passos / `215.040` tokens)
- **Alinhamento direto do Delta de Atenção $\cos(\Delta_l^{\text{student}}, \Delta_l^{\text{teacher}})$ nas 30 camadas (em apenas `40s` / `60` passos de Subfase 1):**
  - `L00`: **`0.9860`** (`||T_attn|| = 2.59` vs `||S_attn|| = 2.56`, `cos_E2E = 0.9939`)
  - `L01`: **`0.9702`** (`||T_attn|| = 27.51` vs `||S_attn|| = 26.93`, `cos_E2E = 0.9799`)
  - `L15`: **`0.8973`** (`cos_E2E = 0.9287`)
  - `L20`: saltou de **`-0.0892` (negativo!) → `+0.7777`** (`cos_E2E = 0.9157`)
  - `L29` (saída final após 30 camadas em série): `cos_E2E` saltou de **`0.0131` (`1.3%`) → `0.8805` (`88.1%`)**!
- **Perda End-to-End (`E2E`) na T4:**
  - Código antigo (`240` passos, `449.8s`): ficou preso em `CE = 6.7623`, `KL = 4.9939`.
  - **Código corrigido (`150` passos Subfase 1 `AttnDelta` [`103.4s`] + `60` passos Subfase 2 `E2E` [`36.4s`] = `139.8s` totais, `3.2×` mais rápido):** `CE` caiu para **`3.2240` (`PPL = 25.1` em `vocab=49.152`!)**, `KL` pós-Subfase 1 caiu para **`1.1256`** (`4.4×` menor), e `layer_mse` caiu para **`0.0556` (`~94.5%` de similaridade cosseno média nas 30 camadas)**.

### 3.2 Placar Oficial "O Ringue" na `Tesla T4` (`Before` vs `After` dos 5 Fixes)

| Modelo (`Tesla T4`, `n=50`/tarefa, eager `30L`) | Quant | Params | VRAM Isolada (GiB) | Estado/KV @16K (MiB) | `tok/s` | `LAMBADA %` (`PPL` ↓) | `ARC-E %` ↑ | `HellaSwag %` ↑ | `RULER 2K / 8K / 16K NLL` ↓ | `Score` ↑ | `Score/GiB` ↑ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|
| **`SmolLM2-135M (Teacher Base)`** | `fp16` | `134.5M` | `0.254` | `360.00` | `31.6` | `34.0%` (`15.5`) | `48.0%` | `48.0%` | `0.17 / 0.28 / —` | **`43.33`** | `170.59` |
| `Lumina-V1-Surgery` *(Antigo, c/ 5 bugs)* | `fp16` | `163.2M` | `3.714*` | `5.3974` | `11.0` | `0.0%` (`34.377`) | `10.0%` | `22.0%` | `3.69 / 29.59 / 38.14` | `10.67` | `2.87*` |
| **`Lumina-V1-Surgery-v2` *(Corrigido)* ** | **`fp16`** | **`163.2M`** | **`0.315`** | **`5.3974` (`66.7×`↓)** | `11.0` | **`2.0%` (`550.2`)** | **`30.0%`** | **`46.0%`** | **`3.05 / 2.85 / 2.78`** | **`26.00` (`2.44×`↑)** | **`82.54`** |
| **`Lumina-V1-Surgery-v2` *(Corrigido)* ** | **`quarot_w4a16`** | **`163.2M`** | **`0.138` (`1.84×`↓)** | **`5.3974` (`66.7×`↓)** | `7.0` | **`2.0%` (`684.2`)** | **`30.0%`** | **`44.0%`** | **`3.55 / 3.36 / 3.27`** | **`25.33`** | **`183.55` (1º!)** |

**Destaques Empíricos da Correção na `Tesla T4`:**
1. **`HellaSwag` saltou de `22.0%` (nível de chute aleatório) para `46.0%` (`FP16`) e `44.0%` (`QuaRot INT4`)**, recuperando **`95.8%` da pontuação do professor `SmolLM2-135M` (`48.0%`)** em apenas `140` segundos de treino (`215K` tokens)!
2. **`ARC-Easy` triplicou de `10.0% → 30.0%`** (tanto em `FP16` quanto em `QuaRot INT4`).
3. **Fim da explosão em contexto longo (`RULER 8K / 16K`):** a NLL de resposta em `16K` caiu de **`38.14 → 2.7791` (`13.7×` menor)**, mantendo estabilidade monotônica de `2K (3.05) → 8K (2.85) → 16K (2.78)` com estado $O(1)$ de apenas **`5.40 MiB`** (`66.7×` menor que o KV-cache de `360 MiB` do professor).
4. **Vitória em Eficiência de Memória (`Score/GiB`):** com a medição isolada de VRAM e `QuaRot Triton W4A16` (`3.555×` compressão nas células), o `Lumina-V1-Surgery-v2` ocupou apenas **`0.138 GiB` (`141 MiB`) de VRAM** e atingiu **`183.55 Score/GiB`**, superando o professor `SmolLM2-135M` (`170.59 Score/GiB` em `0.254 GiB`).

### 3.3 O que Ainda é Limitação Real (Honestidade Científica)
- **`LAMBADA` (`exact_match = 2.0%`, `PPL = 550.2` vs `15.5` do professor) e geração aberta longa:** embora a perplexidade no LAMBADA tenha melhorado **`62.5×`** (`34.377 → 550.2`) e o modelo tenha completado `"The reason recurrent neural networks use constant memory is a good one."` antes de emitir `\n`, acertar a palavra exata final de uma história (`LAMBADA`) exige atenção *exact-copy* de longo alcance que `215K` tokens (`140s` na T4) ainda não consolidam completamente nas `7` janelas híbridas `CausalLocalWindow`.
- **Velocidade `eager` em `30` camadas independentes (`11.0 tok/s`):** diferente do `Lumina-V1-Proto` de `12L` compilado (`248.3 tok/s` na C016), `30` camadas sem compartilhamento executadas com `compile_fractal=False` sofrem overhead de despacho Python (`30` células + `7` janelas por token); para produção em `30L`, é obrigatório habilitar `CompiledDecode` (`CUDAGraph`).

---

## 4. Stage 4 — Fase D `MetaQueryBridge` (`Nano Lumina`) na `Tesla T4` com `SigLIP 2` Real
Validado sobre o backbone `Lumina` (`163.2M`) + `google/siglip2-base-patch16-224` e streams reais (`20` amostras de `HuggingFaceM4/FineVision` + `20` de `aipracticecafe/curated-danbooru-2026`):
- **Congelamento (`freeze_backbone_for_metaquery`):** `8.315.712` parâmetros treináveis no `MetaQueryBridge` vs `259.985.124` congelados.
- **Shapes de saída `DiT` verificados em GPU:** `(1, 64, 2048)` + `(1, 768)` (1 imagem) e `(1, 8, 64, 2048)` + `(1, 768)` (GIF 3D de 8 frames).
- **Medição empírica real dos bytes de estado recorrente na T4:** `state_bytes_1_image = 5.659.628 B == state_bytes_8_gif_frames = 5.659.628 B` (`o1_state_invariant = True`), pico de VRAM `3.957,89 MiB` (`1` img) vs `4.033,21 MiB` (`8` frames, `+75 MiB`), e `grm_segments = 1` e `8`.
- **Treino do conector (`40` passos com alvo substituto `siglip2_fixed_orthogonal_projection`):** perda caiu de **`0.8303 → 0.2189` (`−73.6%`)** com **`backbone_mutated = False`**.
