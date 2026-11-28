# Stack Tecnológico Completo da Família Lumina

> **Documento vivo** consolidando todas as tecnologias da família **Lumina** (`Lumina-1-4B`, `Lumina 2`, `Nano Lumina` e `lumina-rs`), separadas por maturidade: **Em Produção**, **Protótipo no Repo**, **Camada 1 (Upgrades Cirúrgicos)**, **Camada 2 (Experimentos `E1–E7` para `Lumina 2`)**, **Camada 3 (Apostas Especulativas / `Nano Lumina`)** e **Descartados Empiricamente**.
>
> *Baseado nos resultados reais da Tesla T4 até 27/11/2026 (`HEAD 521b0a2`) e na expedição histórica de matemática e física.*

---

## 0. Em Produção Hoje (`Lumina-1-4B v2` — Validado em GPU Real)

| Tecnologia | Raiz Matemática / Física | Onde Está no Repo | Status & Evidência Real (Tesla T4) |
|---|---|---|---|
| **Híbrido `GQA-DeltaNet (27L)` + `Global HALO (9L)`** | Widrow–Hoff (1960), Kálmán (1960), Mori–Zwanzig (1965), `HALO` (2026) | [`fractal_delta.py`](../src/llm2/architectures/fractal_delta.py), [`state_kernels.py`](../src/llm2/architectures/state_kernels.py) | **Em produção** (`4,071B` params, `32 Q / 8 KV`, estado $O(1)$ nos 75% recorrentes) |
| **Fases Áureas (`_GOLDEN_PHI`)** | Hurwitz (1891), Teoria KAM (1954–1963) | [`fractal_delta.py`](../src/llm2/architectures/fractal_delta.py) (`L20, L953`) | **Em produção** (evita ressonâncias entre escalas) |
| **KV-Cache Estático `INT8` (`mode="fp8"`) + `Q=128` Micro-Tiling** | Quantização simétrica + tiling de streaming (`FlashAttention`) | [`fractal_delta.py`](../src/llm2/architectures/fractal_delta.py) (`L198, L691, L740`) | **Em produção** (`64K` prefill `9,153 GiB`, decode `3,870 GiB`, `scatter_` zero-alloc) |
| **`CUDAGraphs Eager` Zero-Clone** | Captura de grafo estático sem overhead Python | [`decode_optimization.py`](../src/llm2/research/decode_optimization.py) | **Em produção** (`capture 0,56–1,19s`, paridade exata `64/64`) |
| **Perfis `Lumina-Q8 / Q6 / Q5 / Q4` + Kernels Triton Agrupados** | Sylvester/Hadamard (1893), L2 pinning (`evict_last`/`evict_first`) | [`profiles.py`](../src/llm2/architectures/profiles.py), [`triton_quarot.py`](../src/llm2/architectures/triton_quarot.py) | **Em produção** — T4 real: **`Q4 = 31,10 tok/s` (`2.114 mJ/tok`, `2,73 GiB`)**, **`Q8` lossless = `28,02 tok/s` (`2.330 mJ/tok`, `4,26 GiB`)** |
| **`RMSNorm` Fundido em Inferência CUDA** | Acumulação interna FP32 (`F.rms_norm`) | [`model.py`](../src/llm2/model.py) (`L19-L27`) | **Em produção** (erro máx. `0.0` vs FP32→FP16; cortou `copy_` de `1164 → 477/tok`) |
| **`dplr_chunked` GQA em `matmul` Puro** | Solver triangular por blocos sem `einsum` | [`state_kernels.py`](../src/llm2/architectures/state_kernels.py) (`L132-L162`) | **Em produção** (erro `1e-7` vs sequencial, `11,06 ms` em `T=1024`) |
| **Cirurgia 1:1 (`RADLADS`) + `LoRA` + `Top-K KL OPD`** | Kullback–Leibler (1951), Destilação on/off-policy | [`distill_convert.py`](../src/llm2/research/distill_convert.py), [`lumina_3b_pipeline.py`](../src/llm2/research/lumina_3b_pipeline.py) | **Em produção** (Stage 2b `4,92M` tok rodando; Stage 3 `OPD` + `MTP` em 02/12 na A100) |
| **`MuonAdamW` (Newton–Schulz 5)** | Iteração polar / Geometria da Informação (Rao/Amari) | [`optim.py`](../src/llm2/optim.py) (`L11-L84`) | **Implementado** no otimizador base |

---

## 1. Protótipos Já no Repositório (Prontos para Ativar / Evoluir)

| Tecnologia | Raiz Matemática / Física | Arquivo | Estado Atual |
|---|---|---|---|
| **`E8Linear2Bit` (`2,25 bpw`)** | Reticulado $E_8$ (Killing/Cartan/Viazovska 2016) | [`sub4bit.py`](../src/llm2/architectures/sub4bit.py) (`L60`) | Implementado & testado (Onda 2: `1,70 GiB`) |
| **`BitPlaneLinear` (2/3/4-bit progressivo)** | Decomposição por planos de bits | [`sub4bit.py`](../src/llm2/architectures/sub4bit.py) (`L345`) | Implementado (pronto para streaming adaptativo) |
| **`Kintsugi` Mixed Precision** | Seleção por sensibilidade + prior `HALO` | [`sub4bit.py`](../src/llm2/architectures/sub4bit.py) (`L395, L459`) | Implementado (hoje usa curtose escalar; Onda 2b `1,81 GiB` → `ARC-E 64%`) |
| **`EngramTableCPU`** | Piṅgala (séc. III A.C.), Al-Kindī (850), DeepSeek `Engram` (2026) | [`engram_offload.py`](../src/llm2/architectures/engram_offload.py) (`L12`) | Esqueleto pronto em RAM *pinned* (hash bigrama único) |
| **`MPOLinear` & `ReLUSquaredActivation`** | Penrose (1971), Oseledets TT (2009), `ProSparse` | [`wave3_ffn_kv.py`](../src/llm2/architectures/wave3_ffn_kv.py) (`L26, L107`) | Implementado (descartado em PTQ; reservado para treino nativo / `ChainDoRA`) |
| **`SharedRecursiveCore` (`Looped Layers`)** | Banach (1922), Floquet (1883), `Huginn`/`Ouro` | [`latent_recursive.py`](../src/llm2/architectures/latent_recursive.py) (`L25`) | Implementado (sem KV-cache / sem early-exit ainda) |
| **`FractalRecursiveBlock` (V-Cycle Multigrid)** | Fedorenko (1961), Brandt (1977) | [`fractal_recursive.py`](../src/llm2/architectures/fractal_recursive.py) (`L106`) | Implementado (mistura `softmax`, escalas fixas 4/16) |
| **`NeuralContextMemory` & `TitansMemoryCell`** | Amari (1972) / Hopfield (1982), Filopono, `Titans` (2025) | [`ncm.py`](../src/llm2/architectures/ncm.py), [`titans_ttt.py`](../src/llm2/architectures/titans_ttt.py) | Implementado (exige `BF16` em L4/A100; etapa própria pelo `AGENTS.md`) |
| **`MetaQueryBridge` + `VisionSpatialProjector`** | Alinhamento contínuo FP32 MSE+Cosine + `SigLIP 2` `2×2` | [`metaquery_head.py`](../src/llm2/multimodal/metaquery_head.py), [`vision_bridge.py`](../src/llm2/multimodal/vision_bridge.py) | Implementado (pronto para `Lumina 2` multimodal) |

---

## 2. Camada 1 — Upgrades Cirúrgicos Imediatos (Sobre o Código Existente)

Melhorias diretas de 20–80 linhas sobre arquivos que já existem em `src/llm2/`:

| # | Upgrade | Origem Histórica → 2026 | Arquivo Alvo | O Que Resolve |
|---|---|---|---|---|
| **C1** | **Espectro Marchenko–Pastur real (via Lanczos)** | Marchenko–Pastur (1967), Lanczos (1950), RMT (2026) | [`sub4bit.py`](../src/llm2/architectures/sub4bit.py) (`L395`) | Troca curtose heurística pela fração de energia acima da borda $\sigma_+$ → melhora `Lumina-Q6/Q5` e `Kintsugi`. |
| **C2** | **`CKDA` ($\alpha \in [-1,1], \beta \in [0,2]$) + `Stable GDN-2`** | Kaczmarz (1937), Householder (1958), `arXiv:2609.24797`, `2611.34051` | [`state_kernels.py`](../src/llm2/architectures/state_kernels.py), [`fractal_delta.py`](../src/llm2/architectures/fractal_delta.py) | Gera autovalores negativos/complexos em 1 passo real ($NC^1$ / paridade) + gates canal-a-canal com cota $O(\sqrt{T})$ mantendo estado $O(1)$. *(Exige `cumprod(sign(w))` junto ao `cumsum(log\|w\|)`).* |
| **C3** | **Tiling SRAM `(BLOCK_V, BLOCK_K)=64` no scan DPLR** | Tiling *cache-aware* para `sm_75` (`64 KiB` max) | [`triton_dplr.py`](../src/llm2/architectures/triton_dplr.py) (`L251-L252`) | Destrava o kernel Triton de `dplr_chunked` na Tesla T4 com `head_dim=128` (hoje sempre cai em PyTorch puro no prefill `64K`!). |
| **C4** | **`Engram` Canônico + Multi-Head 2/3-Gramas + Prefetch** | Piṅgala (séc. III A.C.), Al-Kindī (850), `arXiv:2601.07372` | [`engram_offload.py`](../src/llm2/architectures/engram_offload.py) (`L52-L67`) | Elimina colisões do hash bigrama simples e sobrepõe DMA PCIe da RAM ao compute das camadas anteriores. |
| **C5** | **Fix `sm_75` + `Polar Express` + `NorMuon` / `Muown`** | Chebyshev (1854), Cayley (1846), `arXiv:2505.16932`, `2510.05491`, `2605.10797` | [`optim.py`](../src/llm2/optim.py) (`L11-L23`) | Remove `grad.bfloat16()` forçado em `sm_75` e substitui coeficientes fixos de NS5 por polinômios minimax ótimos (`Polar Express`). |
| **C6** | **Loads `uint32` + Split-K + Fusão `gate+up` SwiGLU** | Intensidade aritmética / coalescing GPU | [`triton_quarot.py`](../src/llm2/architectures/triton_quarot.py) (`L21-L83`) | Próximo salto sobre os `31,1 tok/s` atuais do `W4A16 Fast` (`rotate_input=False`; reservar borboleta *HadaCore* FWHT apenas para `rotate_input=True`). |

---

## 3. Camada 2 — Experimentos Prioritários para `Lumina 2` (`E1–E7` + Runtime `lumina-rs`)

### Trilha I — Compatíveis com Cirurgia 1:1 (`Lumina-1-4B` → `Lumina-2-4B / 8B / 12B`)

| ID | Experimento | Raiz + Referência | Alvo | Portão de Aprovação |
|---|---|---|---|---|
| **E1** | **`QSA` (micro-blocos 4 tok) + `Fast-TurboQuant KV` (2–3 bit + QJL 1-bit)** | Hadamard + JL + `QSA` (`Qwen3.8`) + `arXiv:2606.21448` | `fractal_delta.py` (9 Globais `HALO`) | Decode `64K` sobe **`≥ 3,5×`** (`> 7,5 tok/s` na T4) mantendo `RULER 4K–64K ≥ 95%` do `KV-INT8`. |
| **E2** | **`CKDA` + `Stable GDN-2` (+ ablação `DeltaProduct n_h=2`)** | Householder (1958) + `arXiv:2609.24797`, `2611.34051`, `2502.10297` | `state_kernels.py` + `fractal_delta.py` | **`≥ +15 pp`** em paridade/$S_5$/`Byte-Binding` + ganho em `RULER 8K` com regressão `< 5%` no decode `t==1`. |
| **E3** | **`ChainDoRA` (TT-LoRA) vs `LoRA r=16/32`** | Tensor-Train (Oseledets) + `arXiv:2609.25058` | `wave3_ffn_kv.py` + pipeline Stage 2/SFT | Empatar `kl_loss` e QA usando **`≤ 20%`** dos parâmetros treináveis (viabiliza `8B–14B` em 1× A100). |
| **E4** | **`LLVQ` (Leech $\Lambda_{24}$ + Spherical GPTQ) + `Kintsugi-MP` (+ QAT/STE curto)** | Leech (1967), Golay (1949), `arXiv:2603.11021` | `sub4bit.py` + `profiles.py` | Superar `E8-240` (`PPL 83,93`) por `≥ 25%` *Hadamard-free* e atingir **`≥ 95–97%` do FP16** em `≤ 1,75 GiB`. |

### Trilha II — Pesquisa de Pré-Treino Nativo (`50M–150M` em Colab L4 `BF16` — `Lumina-Next`)

| ID | Experimento | Raiz + Referência | Alvo | Portão de Aprovação |
|---|---|---|---|---|
| **E5** | **`mHC-lite` vs `Block AttnRes` + `Gated Attention` pós-SDPA + `Canon-B/C/D`** | Birkhoff (1946) + `arXiv:2601.05732`, `2603.15031`, `2505.06708`, `2512.17351` | `fractal_delta.py` | `Δval_nll ≤ -0,03` em iso-FLOPs sem perder `> 5%` no decode. |
| **E6** | **`Polar Express` + `NorMuon` + `Muown` vs `MuonAdamW`** | Chebyshev (1854) + `arXiv:2505.16932`, `2510.05491`, `2605.10797` | `optim.py` | Mesmo `val_nll` com **`≥ 12–15%` menos passos/tokens**. |
| **E7** | **`ReLU²` nativo + `NuMuon` (`BF16`) + `Log-Linear Fenwick DeltaNet` + `2 KV heads`** | Norma Nuclear + Árvore de Fenwick + `arXiv:2603.03597`, `2506.04761` | `wave3_ffn_kv.py` + `optim.py` + `state_kernels.py` | `≥ 80%` zeros exatos na FFN (`Δval_nll ≤ 0,02`), erro `MPOLinear < 0,25` e ganho líquido em recall longo. *(E8 condicional: `ParaTTT` + `Select-TTT` em 2–3 camadas).* |

### Extras Consolidados de `Lumina 2` & Runtime `lumina-rs` (Incluindo Bônus ⭐)

- **Multimodal `VL-JEPA` Incremental:** adicionar regularização anti-colapso / target EMA sobre `compute_metaquery_alignment_loss` já pronta em [`metaquery_head.py`](../src/llm2/multimodal/metaquery_head.py) (`L177-L204`).
- **Grafos de Ramanujan (1913/1988) + `p-RoPE` 25% (espectro diofantino KAM) + `HDC` (`XOR+POPCNT`):** máscaras esparsas ótimas e indexador L2 (`4 MiB` em `1M` tokens) para `QSA`/`LSA`.
- **Multiplicadores de Floquet (1883) + `LOOM` / `MoLT-KV` (`arXiv:2610.01153`, `2611.34923`):** estabilidade e reuso de KV em `Looped Layers`.
- **Lógica Linear (Girard 1987) → `lumina-rs` (Rust *zero-copy*):** alocação estática de `DeltaNet` + `KV-INT8` + `Engram` DRAM + aproximações racionais **Bhāskara I / Padé** para `SiLU`/`exp`.
- ⭐ **Bônus 1 — Gramática de Pāṇini (*Aṣṭādhyāyī*, séc. V A.C.):** decodificação restrita por gramática formal no `lumina-rs` (JSON/código 100% válido sem erro de sintaxe, 0 custo de treino).
- ⭐ **Bônus 2 — Curvas de Peano/Hilbert (Z-Order, 1890–1891):** layout *cache-oblivious* de blocos de peso e KV-cache na memória do `lumina-rs` (ataca diretamente o gargalo de Von Neumann em CPU/GPU).
- **Sunzi CRT / `Ozaki Scheme II` (`arXiv:2609.27831`):** usar **apenas** em GEMMs compute-bound (prefill grande / CPUs `AMX-INT8` / `SVE2`) no `lumina-rs`.

---

## 4. Camada 3 — Apostas Especulativas / Moonshot (`Nano Lumina` / `Lumina 3`)

*(Apenas em sondas baratas `10M–50M` ou quando autorizado por etapa explícita conforme `AGENTS.md`)*

| Tecnologia | Raiz Histórica → 2026 | Aplicação Alvo |
|---|---|---|
| **Cabeças de Atenção Tropical** | Imre Simon (1978, USP), NeurIPS 2025, `arXiv:2601.09775` | Raciocínio combinatório e programação dinâmica exata (acopladas a cabeças softmax). |
| **`BusemannFormer` (Horosférico) & `v-PuNNs` ($p$-ádico)** | Lobachevsky / Hensel (1897), `arXiv:2611.27167`, `2508.01010` | Transposição experimental para ASTs/hierarquias e índices taxonômicos sem distorção >512K. |
| **Osciladores `AKOrN` / `Kuramoto` & Feixes `ONDA`** | Kuramoto (1975), Grothendieck (1957), `arXiv:2606.11585`, `2610.10018` | *Binding* multimodal e comunicação entre escalas sem atenuação em `fractal_recursive.py`. |
| **Projetor Visual `Subspace Clifford 2D`** | Grassmann (1844), Clifford (1878), `CliffordNet` (2026) | Substituir MLP plano em `vision_bridge.py` para preservar orientação/pose em edição de cena. |
| **Cabeça Epistêmica *Saptabhaṅgī* / *Catuṣkoṭi* / Paraconsistente + *Odyssey*** | Jainismo / Nāgārjuna / Newton da Costa (1963), `arXiv:2606.27593` | Distinguir certeza, ambiguidade, contradição entre fontes e abstenção em RAG. |
| ⭐ **Bônus 3 — Aritmética Logarítmica (LNS) & `BitNet b1.58`** | Vīrasena (~816 d.C.), Brusentsov *Setun* (1958), `arXiv:2402.17764` | Inferência *multiplier-free* em CPUs/borda extrema (`Nano Lumina` treinado do zero). |
| **Streaming `BitPlaneLinear` + `exit_gate` sob `Active Inference (AIC)`** | Friston (2006+), `sub4bit.py:345`, `fractal_delta.py:1258` | Tokens fáceis leem só 2 bits + early-exit; tokens difíceis carregam 4 bits + profundidade total. |
| **Horizonte Pós-Backprop & Pós-Von Neumann** | Landauer (1961), Bennett (1973), `PotentialEP` (`arXiv:2610.14409`), `Extropic` (`THRML`) | Camadas reversíveis, *Equilibrium Propagation* e amostragem termodinâmica de baixíssima energia. |

---

## 5. Descartados Empiricamente (Registro Negativo — Não Repetir)

| Técnica | Motivo Verificado da Eliminação |
|---|---|
| **`ParetoQ` 2-bit / Ternário em PTQ puro** | Onda 2 na T4: `PPL 158K–990K` (só funciona com QAT/pré-treino do zero). |
| **`MPO-FFN` pós-hoc (SVD sobre FFN pronto)** | Onda 3 na T4: erro Frobenius `80–90%`, `ARC-E 26–32%`, `34×` mais lento. |
| **`ReLU²` zero-shot sobre pesos treinados em SiLU (`FP16`)** | Onda 3 na T4: overflow/`NaN` em FP16 (só vale em treino nativo `BF16` — `E7`). |
| **`HALO` KV-Sharing / `K=V` drop-in pós-treino** | Onda 3 na T4: similaridade cosseno `≈ 0,00` entre camadas (usar `2 KV heads` nativo). |
| **`Titans/TTT` em `FP16` na Tesla T4** | Deu `NaN` em FP16 com `8×` estado (só testar em `BF16` na L4/A100 com `ParaTTT`). |
| **`Ozaki Scheme II` (CRT) em `lm_head` no decode `batch=1`** | `lm_head` já bate `~250 GB/s` (100% memory-bound); fatias CRT aumentariam o tráfego de VRAM e deixariam mais lento. |
| **Conceitos sem kernel competitivo atual** | Octônios não-associativos, Quipus, Anticítera, *Constructor Theory*, Geometria Não-Comutativa (Connes), `Orch-OR` e sistemas `(M,R)` de Rosen. |
