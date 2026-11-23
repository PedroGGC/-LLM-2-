# IA2 Campanha 017 — Attention-Surgery (`RADLADS` / `Jet-Nemotron` 1:1) & Fase D `MetaQueryBridge` (`Nano Lumina`)

**Data:** 22/11/2026  
**Branch:** `llm-ideas`  
**Hardware Verificado:** CPU local (`Windows`, PyTorch `2.14.1+cpu`) + **Google Colab `NVIDIA Tesla T4` (`sm_75`, `15 GB`, PyTorch `2.14.1+cu130`, `FP16`/`FP32`)** (`runtime` encerrado imediatamente após coleta com `list_runtimes() == {}`).

---

## 1. Mecanismos Implementados

1. **`Attention-Surgery` 1:1 ([`src/llm2/research/distill_convert.py`](../../src/llm2/research/distill_convert.py), [`src/llm2/model.py`](../../src/llm2/model.py), [`src/llm2/architectures/fractal_delta.py`](../../src/llm2/architectures/fractal_delta.py)):**
   - `transplant_teacher_weights`: copia Embeddings/`lm_head` amarrados, `RMSNorm`s, `SwiGLU` FFNs e projeções `QKV`/`Out` (com expansão automática de `GQA` via `_expand_gqa_kv`), inicializando `conv`, `assoc_readout`, `anchor_gate` e `hybrid_windows.out_proj` em baixa magnitude.
   - `apply_surgery_freeze` + `layer_mse_weight`: congela `SwiGLU` FFNs e Embeddings e alinha estados intermediários por camada (`return_layer_states=True`).
2. **Fixes Reais de Hardware Turing (`sm_75`, Tesla T4):**
   - [`src/llm2/architectures/triton_quarot.py`](../../src/llm2/architectures/triton_quarot.py): suporte explícito a Tensor Cores `FP16` (`USE_FP16_DOT`) além de `BF16`.
   - [`src/llm2/architectures/triton_dplr.py`](../../src/llm2/architectures/triton_dplr.py): guardião em `can_use_triton_dplr` checando `shared_memory_per_block_optin` — descobriu-se na T4 que o kernel backward `64×64` requer `73.728–90.112 B` (> limite de `65.536 B` da `T4`, vs `102.400 B` na `L4`), redirecionando automaticamente para o fallback PyTorch sem erro de compilação.
3. **Fase D — `MetaQueryBridge` para o `Nano Lumina` ([`src/llm2/multimodal/metaquery_head.py`](../../src/llm2/multimodal/metaquery_head.py)):**
   - $Q=64$ `MetaQueries` aprendíveis + conector bidirecional Pre-Norm de 2 camadas (`RMSNorm` FP32 + `SDPA` + `SwiGLU`) + fusão de âncoras visuais `<img_end>` do `GRM`.
   - Emite `seq_cond = (B, 64, 2048)` + `pooled_cond = (B, 768)` para imagem estática (`Anima 2B` / `DiT`) e `seq_cond = (B, F, 64, 2048)` para GIFs 3D animados mantendo estado $O(1)$ constante no backbone VLM congelado.

---

## 2. Verificações Executadas

### 2.1 Suítes `pytest` (CPU Local & Tesla T4 CUDA)
- **CPU local:** `python -m pytest -q tests/test_metaquery_bridge.py tests/test_phase_c_distill_ringue.py` → **`7 passed, 2 skipped in 10.04s`** (skips são exclusivamente os 2 testes `@pytest.mark.cuda`). Pós-ajuste de `triton_dplr.py`, revalidada a suíte CPU (`tests/test_metaquery_bridge.py`, `tests/test_phase_c_distill_ringue.py`, `tests/test_fractal_v2_grm.py`).
- **Colab `Tesla T4` (Stage 1 CUDA Gate):** `pytest -q -m cuda tests/test_phase_c_distill_ringue.py tests/test_metaquery_bridge.py` → **`2 passed, 7 deselected in 150.24s`**.

### 2.2 Ablação Pareada Controlada em Escala Pequena (`CPU`, `60` passos, `4L`, `vocab=512`)
Comparando `Random-MLP` (receita C016) vs `Attention-Surgery 1:1` sobre o mesmo professor `TransformerLM` pré-treinado (`89.36s`), avaliando retenção inclusive em metade do vocabulário `[256, 512)` omitida da destilação:

| Braço (`FractalDelta V2`, `4L`, `60` passos) | Params Treináveis / Congelados | Val Full `CE` ↓ | Val Full `KL` ↓ | Val `Top-1 Acc` ↑ | Acordo c/ Professor (Full) ↑ | Retenção em Vocab Não Visto ↑ |
|---|---:|---:|---:|---:|---:|---:|
| **Professor (`TransformerLM` `4L`)** | `1.074.162` | `0.6904` | `0.0000` | `85.09%` | `100.00%` | `100.00%` |
| **A. `Random-MLP` (Receita C016: só `warmstart_embed`)** | `1.074.162 / 0` | `3.6030` | `2.8112` | `21.55%` | `23.86%` | `5.71%` (`KL 2.0770`) |
| **B. `Attention-Surgery 1:1 + FFN Congelado`** | **`418.290 / 655.872`** | **`1.0069`** | **`0.3782`** | **`75.00%`** | **`78.48%`** | **`48.83%` (`KL 0.5786`)** |
| **B + `QuaRot W4A16` (`3.552×` compressão nas células)** | `0 / 1.074.162` | **`1.0031`** | **`0.3766`** | **`75.78%`** | **`79.36%`** | — |
| **C. `Attention-Surgery 1:1 + Refino Conjunto` (FFN livre)** | `1.008.626 / 65.536` | `1.0790` | `0.4524` | `73.89%` | `77.34%` | `47.95%` (`KL 0.5667`) |

*Escopo:* prova de conceito sintética controlada mostrando transferência efetiva de conhecimento FFN com `61.1%` menos parâmetros treináveis; não prova sozinha fluência em linguagem natural nem `vocab=49K`.

---

## 3. Execução Real em GPU (`Tesla T4`): `SmolLM2-135M` (`30L`, `vocab=49.152`) & `MetaQueryBridge` (`SigLIP 2`)

### 3.1 Stage 2 & Stage 3 — `Attention-Surgery 1:1` de `SmolLM2-135M` + `O Ringue` (Eager `compile_fractal=False`)
- **Configuração:** `SmolLM2-135M` (`134.5M`, `30L`, `d=576`, `H=9`, `FFN=1536`, `vocab=49.152`) convertido em 1:1 para `Lumina-V1-Surgery` (`163.15M` params únicos; `55.20M` treináveis / `107.96M` FFNs+Embeddings congelados), `240` passos (`245.760` tokens em `FineWeb-Edu` + SFT sem censura, `449.76s` em 2 passes, pico `8.523 GiB`): `CE 13.4495 → 6.7623`, `KL 12.5034 → 4.9939`, `Total 19.2445 → 8.3980`.

| Modelo (`Tesla T4`, `compile_fractal=False` eager) | Quant | Params | VRAM Sessão (GiB)* | Estado/KV @16K (MiB) | `tok/s` | `mJ/tok` | `LAMBADA %` | `ARC-E %` | `HellaSwag %` | `Score` | `RULER 2K/8K/16K` |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `SmolLM2-135M (Teacher Base)` | `fp16` | `134.5M` | `3.696*` | `360.0` | `31.1` | `1.399.6` | `34.0` | `48.0` | `48.0` | **`43.33`** | `0.17 / 0.28 / —` |
| `Lumina-V1-Surgery (30L 1:1)` | `fp16` | `163.2M` | `3.714*` | **`5.3974`** (`66.7×`↓) | `11.0` | `3.290.6` | `0.0` | `10.0` | `22.0` | **`10.67`** | `3.69 / 29.59 / 38.14` |
| `Lumina-V1-Surgery (30L 1:1)` | `quarot_w4a16` (`3.555×`) | `28.9M` | `3.714*` | **`5.3974`** | `7.3` | `4.713.9` | `0.0` | `16.0` | `34.0` | **`16.67`** | `11.91 / 25.30 / 35.08` |

*\*Nota honesta de medição:* o `VRAM Pico` de `3.696–3.714 GiB` refletiu tensores da mesma sessão que permaneceram alocados do Stage 2 (não é VRAM isolada de inferência). Da mesma forma, `tok/s` caiu para `11.0` (`7.3` em `QuaRot`) porque `evaluate_native_contender` rodou com `compile_fractal=False` (Python eager em `30` camadas sem compartilhamento `Prelude-Core-Coda`, ao contrário dos `248.3 tok/s` compilados da C016 em `12L`).

**Diagnóstico honesto do resultado linguístico na T4 (`Score 10.67` / `16.67`, texto repetitivo `"100000..."`):**
Ao contrário do teste sintético (`4L`, `vocab=512`), a cirurgia 1:1 direta em um modelo Transformer real de `30` camadas e `vocab=49.152` com apenas `245K` tokens **não foi suficiente para preservar ou recuperar fluência linguística**. Os três gargalos técnicos identificados empiricamente foram:
1. O passo 0 já partiu com `CE = 13.45` (pior que acaso $\ln(49152)=10.80$) porque `out_gate` aleatório × `RMSNorm(head_gamma)` alimenta os `o_proj` e FFNs congelados com ativações fora de escala através de 30 camadas em série.
2. `245.760` tokens é uma fração mínima frente aos orçamentos reais de conversão `RADLADS`/`Jet-Nemotron` (centenas de milhões a bilhões de tokens), além de o modo `30L` sem compartilhamento triplicar o custo por token frente ao `Prelude-Core-Coda`.
3. A substituição de `SelfAttention` (`1/√d_h` softmax) por `Gated DeltaNet-2` em 30 camadas profundas exige alinhamento de matriz de atenção ou inicialização residual com zero-gate na saída (`out_gate` aprendendo residual nulo no início) — trabalho técnico de evolução pós-V1.

### 3.2 Stage 4 — Fase D `MetaQueryBridge` (`Nano Lumina`) na `Tesla T4` com `SigLIP 2` Real
Executado diretamente sobre o backbone `Lumina` (`163.2M`) + `google/siglip2-base-patch16-224` e streams reais (`20` amostras de `HuggingFaceM4/FineVision` + `20` de `aipracticecafe/curated-danbooru-2026`):
- **Congelamento (`freeze_backbone_for_metaquery`):** **`8.315.712` parâmetros treináveis** no `MetaQueryBridge` vs **`259.985.124` congelados** (backbone + `SigLIP 2` + `VisionSpatialProjector`).
- **Shapes de saída `DiT` verificados em GPU:** `(1, 64, 2048)` + `(1, 768)` (1 imagem) e `(1, 8, 64, 2048)` + `(1, 768)` (GIF 3D de 8 frames).
- **Medição empírica real dos bytes de estado recorrente:** `state_bytes_1_image = 5.659.628 B == state_bytes_8_gif_frames = 5.659.628 B` (`o1_state_invariant = True`), com pico de VRAM passando apenas de `3.957,89 MiB` (`1` img) para `4.033,21 MiB` (`8` frames, `+75 MiB`), e segmentos `GRM` `1` e `8`.
- **Treino do conector (`40` passos com alvo substituto `siglip2_fixed_orthogonal_projection`):** perda híbrida `MSE + Cosine` caiu de **`0.8303 → 0.2189` (`−73.6%`)** com **`backbone_mutated = False`**. Como o gerador `Anima 2B / DiT` não estava carregado nesta etapa, isso valida o gradiente, as formas e a interface $O(1)$ da ponte — não a geração visual de imagens reais ainda.
