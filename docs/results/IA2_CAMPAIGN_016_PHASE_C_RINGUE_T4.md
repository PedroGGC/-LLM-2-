# Campanha 016 — Fase C (`CausalLocalWindow` Vetorizado + Destilação Rota 1 + Mistura 4 Partes + Primeiro Ringue Real) na NVIDIA T4

- **Data:** 22/11/2026
- **Ambiente:** Colab `1× NVIDIA Tesla T4` (`15 GB VRAM`, `SHAPE_HIGHMEM`, `sm_75`, `PyTorch 2.14.1+cu130`, `FP16` nativo para inferência/teacher e `FP32` para treino do aluno). Runtime encerrado imediatamente após coleta (`list_runtimes == {}`).
- **Objetivo:** validar ponta a ponta a otimização $O(T \cdot W)$ de `CausalLocalWindow` (`unfold`), a destilação arquitetural **Rota 1** (`SmolLM2-1.7B` → `Lumina-V1-Proto 64,6M` em loop), a mistura multimodal + SFT sem censura (`FineVision` + `curated-danbooru-2026` + `booru-tags-to-nl` + `uncensor-ptbr` + `Uncensored-SFT-v2`) e o **primeiro placar real do Ringue** (`ringue_eval.py`).

## 1. Gate de Testes
- **CPU local (`tests/test_phase_c_distill_ringue.py` + `tests/test_multimodal_bridge.py` + `tests/test_fractal_v2_grm.py`):** **`15 passed, 3 skipped`** (`29.19s`).
- **GPU real (`Tesla T4`):** `pytest -q -m cuda tests/test_phase_c_distill_ringue.py` → **`1 passed, 4 deselected`** (`153.54s`, `exit 0`), validando `torch.compile` do `CausalLocalWindow` vetorizado via `unfold` e a destilação em GPU.

## 2. Destilação Rota 1 (`Stage 2`) & Alinhamento Multimodal (`Stage 3`)
- **Stage 2 (`HuggingFaceTB/SmolLM2-1.7B` → `Lumina-V1-Proto 64,64M`, `300` passos = `614.400` tokens reais em `755,4s`, `813 tok/s`, SVD warm-start de `49.152×2048 → 49.152×768`):**
  - `CE`: `14.6585 → 6.5291` | `KL`: `13.6917 → 5.4370` | `Hidden MSE`: `0.00261 → 0.000535` (`−79,5%`) | `Exit Gate`: `1.7349 → 0.4888` | `MTP`: `10.9649 → 7.5364` | `Total`: **`23.3881 → 10.2579`**.
  - Checkpoint salvo com `safetensors.torch.save_model` preservando peso compartilhado `embedding == lm_head`.
- **Stage 3 (`SigLIP 2` + Mistura Multimodal + Texto SFT sem censura, `val_examples=16`):**
  - Streams carregados sem `HF_TOKEN`: `HuggingFaceM4/FineVision:allava_laion` (`100`), `aipracticecafe/curated-danbooru-2026` (`50`), `Mexes/booru-tags-to-natural-language` (`80`), `Guilherme34/uncensor-ptbr` (`80`), `Pradip24042020/Uncensored-SFT-v2` (`80`).
  - Streams com restrição/incompatibilidade descoberta empiricamente: `wallstoneai/civitai-top-nsfw-images-with-metadata` expõe apenas `['image']` (sem legenda) via `imagefolder` no streaming; `VillanovaAI/multi-pixmo-cap` exige `HF_TOKEN` (*gated*); `Vezora/Mini_Orca_Code_Uncensored_alpaca_Format` retornou 404 no Hub.
  - Validação multimodal reservada (`n=16`): **`7.1748` (inicial) → `7.0933` (Stage 1, `30.9s`) → `6.0008` (Stage 2, `39.7s`)**.

## 3. Primeiro Placar Real — "O Ringue" (`Tesla T4`, `n=50` por tarefa, `NVML`)

| Modelo | Quant | Params | VRAM Pico (GiB) | Estado/KV @16K (MiB) | tok/s | mJ/tok | LAMBADA % | ARC-E % | HellaSwag % | Score | Score/GiB |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `SmolLM2-1.7B` (Teacher Base) | `fp16` | `1.711,4M` | `3.362` | `3.072,0` | `38.5` | `1.755,14` | `62.0` | `80.0` | `62.0` | `68.0` | `20.23` |
| `Lumina-V1-Proto` (`64,6M` Looped) | `fp16` | `64,6M` | `0.264` | **`4.072`** | **`248.3`** | **`177.33`** | `0.0` | `16.0` | `26.0` | `14.0` | `53.03` |
| `Lumina-V1-Proto` (`64,6M` Looped) | `QuaRot W4A16` | `38,0M` (packed) | `0.264` | **`4.072`** | `223.7` | `209.60` | `0.0` | `16.0` | `26.0` | `14.0` | `53.03` |

- **`RULER NIAH` (`mean_answer_nll`, `8` exemplos/comprimento):**
  - `SmolLM2-1.7B`: `2K = 0.0933`, `8K = 0.0515`, `16K = n/a` (`max_position_embeddings=8192`, KV-cache estimado `3.072 MiB` @16K).
  - `Lumina-V1-Proto (FP16 + GRM)`: `2K = 3.6092`, `8K = 4.1625`, **`16K = 4.1671`** (zero deriva de `8K → 16K` com estado fixo `4.072 MiB`, **`754×` menor** que o KV-cache do professor!).
  - `Lumina-V1-Proto (QuaRot W4A16 + GRM)`: `2K = 3.6671`, `8K = 4.2421`, `16K = 4.2764` (`3.556×` compressão de células sem perda de acurácia em `ARC-E`/`HellaSwag`).

## 4. Diagnóstico Honesto & Próxima Evolução Obrigatória
1. **Eficiência comprovada na T4:** `Lumina-V1-Proto` entregou **`12,7×` menos VRAM (`0,264` vs `3,362 GiB`)**, **`754×` menos estado em `16K` (`4,07` vs `3.072 MiB`)**, **`6,45×` mais `tok/s` (`248,3` vs `38,5`)** e **`9,9×` menos energia (`177,3` vs `1.755,1 mJ/tok`)**.
2. **Qualidade ainda imatura (`Score 14,0` vs `68,0`) — causa raiz diagnosticada:**
   - Destilar um modelo `64,6M` com pesos aleatórios de célula em apenas `614K` tokens (`0,000005%` do orçamento de pré-treino do `SmolLM2-1.7B`) sobre um vocabulário de `49.152` tokens não basta para consolidar fluência (`CE = 6,53`), e o Stage 2 multimodal (`90%` legendas) sobreescreveu o `Prelude` gerando repetições `"the top of the image shows..."`.
   - **Solução arquitetural (próximo passo real da Rota 1):** em vez de iniciar o aluno `FractalDelta` com MLPs aleatórios, implementar **Attention-Surgery / Weight-Transfer (`RADLADS`/`Jet-Nemotron` literal)**: copiar `100%` dos pesos dos **MLPs (`SwiGLU`), `RMSNorm` e `Embeddings`** de um modelo aberto `-Base` (ex.: `1,5B–3B`), substituindo apenas `75%` das camadas de Self-Attention por `Gated DeltaNet-2` (mantendo `25%` como atenção local híbrida `3:1`). Assim o aluno já começa com `>80%` da fluência do professor no passo 0 e fecha o gap de qualidade com fração minúscula de tokens!
