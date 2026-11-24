# Roteiro Mestre — Família Lumina (`Lumina 1` → `Lumina 2`)

**Objetivo científico central ("LLM 2"):** construir modelos substancialmente mais econômicos em VRAM, RAM, contexto longo e energia que Transformers densos equivalentes, mantendo ou superando a precisão de modelos parelhos/maiores (`Qwen3-VL-4B-Instruct`, `Qwen3-4B-Instruct-2507`, `Qwen3.5-4B` e acima).

---

## 1. Arquitetura Aprovada Empiricamente (`Kaggle 2× T4`, 24/11/2026)

Comprovada em A/B controlado em escala real `4B` (`Qwen/Qwen3-4B-Instruct-2507`, ver `VALIDATION.md`):
- **Backbone híbrido:** `27` camadas `Gated DeltaNet-2` (`QK-Norm`, `out_gate_rank=256`, `assoc_channel=False`) + `9` camadas Globais escolhidas por **`HALO`** (`[0, 1, 6, 7, 12, 23, 24, 34, 35]`).
- **Contexto longo:** Atenção Global sem evicção com **`KV-FP8` + `RoPE/HyPE`** (empatou `100%` com `KV-FP16` no `RULER` com metade da VRAM; `HALO + Global` bateu `3:1 + SWA` por **`+28` a `+40 pp`**).
- **Empacotamento:** **`QuaRot W4A16`** (`act_bits=16`, Triton `block_n=32`) → `~2.5–2.6 GiB` em `4B`.
- **Fora do `Lumina 1` por ora:** `Titans / TTT` (8× mais estado $O(1)$ e instável em `FP16` na T4; reservado para re-teste em `BF16` na Colab `L4 / A100`).

---

## 2. Escada de Tamanhos da Família Lumina (Ajustada ao Hardware Real)

Com **Kaggle `2× T4` (grátis semanal)** + **Colab `L4` (barato em CUs, `24 GB BF16`)** + **Colab `A100` (CUs limitados, `40–80 GB BF16`)**:

| Versão | Escopo | Tamanho(s) | Doador 1:1 / Professor `OPD` | GPU Principal | Viabilidade Real em CUs |
|---|---|---|---|---|---|
| **`Lumina-1-4B`** *(atual)* | **Texto puro SFW** (fundação sólida) | **`4B`** (`~2.5–2.6 GiB` INT4) | Doador: `Qwen3-4B-Instruct-2507`<br>`OPD`: **`Qwen3-14B`** | **Kaggle `2× T4`** *(Stage 1–2)*<br>**Colab `A100`** *(Stage 3 `OPD`)* | **100% no ponto doce** (quase zero CUs) |
| **`Lumina-2-4B`** | **Multimodal** (Texto + Visão `SigLIP 2`) | **`4B`** | Base: `Lumina-1-4B` final<br>Dados visuais: `Qwen3.8-27B` Q4 | **Colab `L4`** (`BF16`) | **100% no ponto doce** (barato em CUs) |
| **`Lumina-2-8B`** | **Multimodal Principal** | **`8B`** (`~4.8 GiB` INT4) | Doador: `Qwen3-8B`<br>`OPD`: **`Qwen3-32B`** | **Colab `A100`** (`BF16`, scripts já zerados no `4B`) | **Totalmente viável** |
| **`Lumina-2-12B / 14B`** *(teto máximo)* | **Multimodal Flagship** *(opcional)* | **`12B` ou `14B`** *(ex.: `Qwen3-14B` ou futuro `12B`)* | Doador: `12B/14B` denso<br>Cirurgia com FFN congelado + LoRA | **Colab `A100 80 GB`** | **Limite máximo** (só executar se houver folga de CUs e após o `8B` validar retorno real) |
| ~~`16B – 20B+`~~ | *Fora de escopo* | — | — | Exigiria multi-GPU / meses de CUs | **Descartado** para 1× A100 Colab |
| **`Nano Lumina`** | Gerador Imagem + GIF acoplado ao `Lumina-2-4B` | Ponte `MetaQueryBridge` + LoRA sobre difusão aberta | Cérebro: `Lumina-2-4B` | **Colab `L4 / A100`** | **Viável** (sem treinar difusão do zero) |

---

## 3. Execução em 3 Fases (do `Lumina-1-4B` ao `Lumina-2`)

### Fase 1 — `Lumina-1-4B` Base Eficiente (`Kaggle 2× T4`)
1. **Stage 1 (`AttnDelta` + `HALO`):** `~800` passos (`~3–4M` tok SFW) até `cos > 0.94`.
2. **Stage 2 (`Off-Policy Top-K KL + CE`):** `~2–3M` tok (`FineWeb-Edu` + `FineMath-4plus` + `Wikipédia PT-BR` + código Python, `seq=1024–1536`) até `rep4 < 0.08` e `RULER 2K–8K` forte.
3. **Empacotamento `QuaRot W4A16`:** validação de `VRAM`, `tok/s`, `mJ/tok` e retenção QA/RULER.

### Fase 2 — Superando o Próprio Doador (`Kaggle 2× T4` → `Colab A100`)
1. **Stage 2.5 (SFT Frontier Offline):** trajetórias verificadas de código/matemática/instruções (`OpenThoughts` / `FineMath-4plus` / PT-BR) com LoRA.
2. **Stage 3 (`OPD` com `Qwen3-14B` na `Colab A100`):** o `Lumina-1-4B` (já fluente) gera seus próprios rollouts e o `Qwen3-14B` (mesmo tokenizador, 3,5× maior) corrige os logits `Top-K`.
3. *(Opcional)* **RLVR:** auto-treino verificado em Python/matemática (só trajetórias que passam nos testes).
4. **Gate de aprovação:** só avançamos para a Fase 3 quando o `Lumina-1-4B` empatar/superar `Qwen3-4B-Instruct-2507` e `Qwen3-VL-4B-Instruct` com clara vantagem em VRAM/energia/contexto longo.

### Fase 3 — `Lumina-2` Multimodal (`4B` → `8B` → teto `12B/14B`)
1. **Dataset Visual (`Colab L4`):** `Qwen3.8-27B-Uncensored` (Q4, `~17 GB`) gerando descrições ricas sobre `VillanovaAI/multi-pixmo-cap` + datasets privados no Drive.
2. **`Lumina-2-4B` (`Colab L4` `BF16`):**
   - Alinhamento congelado (`SigLIP 2` + `MetaQueryBridge` — preserva 100% do texto do `Lumina-1-4B`).
   - SFT Multimodal misto (`70%` visual + `30%` texto puro do `Lumina-1-4B` para evitar esquecimento).
3. **Escala (`Colab A100`):** repetir pipeline já validado para **`Lumina-2-8B`** (e avaliar custo/benefício de um flagship **`12B/14B`** como teto absoluto), seguido do **`Nano Lumina`**.
