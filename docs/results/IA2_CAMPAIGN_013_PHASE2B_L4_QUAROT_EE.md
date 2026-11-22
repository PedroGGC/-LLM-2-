# IA2 Campanha 013 (NVIDIA L4 — Roteiro Fase 2B: Validação CUDA/BF16/`torch.compile` de QuaRot Triton INT4, Early-Exit, SegmentMemoryCache, Confidence-Gated MTP e Escala `53,9M`)

Data: 2026-11-22  
Branch: `llm-ideas`  
Hardware: `NVIDIA L4` (`23.034 MiB`, faixa de Power Limit `40,00 W`–`72,00 W`), PyTorch `2.14.1+cu130`, Triton `3.8.0`, medição elétrica via contador hardware NVML (`nvmlDeviceGetTotalEnergyConsumption`).

---

## 1. Suíte CUDA Completa (`pytest -q -m cuda`) e Compatibilidade `BF16` / `torch.compile`

- **Novos módulos (`src/llm2/architectures/triton_quarot.py` + `quarot.py`):** criado kernel `llm2::quarot_int4_linear` (`custom_op` + `register_fake` + `register_autograd` para compatibilidade total com `aot_autograd` e `torch.compile`) com caminhos separados `_quarot_int4_matvec_kernel` (`M=1` em registradores) e `_quarot_int4_matmul_kernel` (`M>1` em Tensor Cores `BF16`).
- **Suíte CUDA na L4 (`61 passed`, zero falhas nos testes CUDA):**
  - Execução inicial (`pytest -q -m cuda tests/test_quarot_early_exit.py`): **`2 passed, 6 deselected em 33,82 s`** (paridade `FP32` / `BF16` / `torch.compile` de `triton_quarot`, `QAT`, `PTQ`, `Early-Exit`, `SegmentMemoryCache` e `Confidence-Gated MTP`).
  - Execução do repositório (ignorando `tests/e2e`, `tests/test_m2_challenger_stress.py`, `tests/test_cycle2_git.py` — nenhum contém `@pytest.mark.cuda` — após incluir `configs/` e `scripts/`): **`45 passed`** (run principal em `554,78 s`) + **`16 passed, 77 deselected em 227,83 s`** nos testes dependentes de `configs/` (`test_cycle3.py`, `test_cycle4_models.py`, `test_cycle5.py`, `test_recall_models.py`) = **`61 passed` no total CUDA**.
  - Regressão local CPU pós-edição (`tests/test_quarot_early_exit.py tests/test_hadamard_stress.py tests/test_fwht_benchmark.py`): **`10 passed, 2 skipped em 4,54 s`**.

---

## 2. Descoberta Principal: Onde o Kernel Triton `INT4` Fundido Supera o `Dense BF16` na L4

### 2.1 Microbenchmark por Camada Linear (modo `eager`)
| Camada (`in → out`, `M`) | `Dense BF16` (µs) | `QuaRot PyTorch` unpack (µs) | `QuaRot Triton W4A16` (µs) | `QuaRot Triton W4A8` (µs) | Triton vs PyTorch unpack |
|---|---:|---:|---:|---:|---:|
| `7.45M_ffn` (`384→640`, `M=1`) | `24,0` | `295,0` | `205,3` | `422,4` | **`1,44×`** |
| `7.45M_ffn` (`384→640`, `M=512`) | `24,5` | `295,4` | `211,5` | `426,7` | **`1,40×`** |
| `55M_ffn` (`1024→3200`, `M=1`) | `23,8` | `316,3` | `213,1` | `425,2` | **`1,48×`** |
| `55M_down` (`3200→1024`, `M=512`) | `57,7` | `315,9` | `268,4` | `438,1` | **`1,18×`** |

*(Em uma camada isolada `eager`, a rotação `apply_hadamard_rotation` em PyTorch antes da `custom_op` adiciona `~180 µs` de lançamento em Python, mantendo o módulo avulso abaixo do cuBLAS direto; com **`torch.compile` + CUDA Graphs** em modelo completo, esse overhead some e o comportamento muda conforme o tamanho dos pesos em relação à L2 de `48 MB` da L4.)*

### 2.2 Escala Real `53,90M` (`53.895.465` params inferência, `d=1024, ff=3200, depth=8, vocab=8192`) — Decode `128` tokens (`Compiled`)
- **Compressão real de pesos:** `102,80 MiB` (`Dense BF16`) → **`40,78 MiB`** totais (`QuaRot INT4` comprimiu `19` matrizes de `90.488.832 B` para `25.454.592 B` = **`3,55×` vs `BF16`**, preservando `embedding is lm_head`).

| Modo (`53,90M`, L4 `BF16` compilado) | Vazão (`tok/s`) | Energia NVML (`mJ/tok`) | VRAM pico | Leitura empírica |
|---|---:|---:|---:|---|
| `Dense BF16 Compiled` | `480,3` | `152,98` | `287,12 MiB` | Baseline (pesos excedem L2 de `48 MB` da L4) |
| `QuaRot PyTorch unpack` (`W4A16`) | `455,4` *(−5,2%)* | `155,83` *(+1,9%)* | `310,54 MiB` | Materializar unpack na VRAM **piora** velocidade, energia e VRAM |
| **`QuaRot Triton W4A16`** | **`666,6` (`+38,8%`)** | **`82,84` (`−45,9%`)** | `287,12 MiB` | **Triton fundido vira o jogo: `+38,8%` tok/s e `−45,9%` mJ/tok** |
| `Early-Exit` (`3,04 / 8` camadas, eager) | `95,3` *(−62,0% camadas)* | `457,09` | `266,61 MiB` | Sem grafo compilado/CUDAGraph, overhead eager supera o ganho de camadas |

- **Treino `53,90M` em `seq=2048 × 2` (`Triton DPLR`):** compile **`148,57 s`**, **`10.065,0 tok/s`** (`406,95 ms/step`), **`6,53 GiB`** VRAM (`loss 13.3657 -> 0.9585` em 10 passos sintéticos).

---

## 3. Impacto em `BPB`, `tok/s`, `VRAM` e `mJ/tok` no Modelo `7,45M` TinyStories BPE (`1.000` passos, `407,1 s`)

### 3.1 Qualidade Real (`val@512` TinyStories BPE `vocab=4096`)
- **`Dense BF16`:** `val@512 = 0,8028 BPB` (`2,2391 NLL`)
- **`QuaRot PTQ W4A16` (`Triton`):** `val@512 = 0,8064 BPB` (`2,2491 NLL`, delta **`+0,0036 BPB`** para **`3,55×`** compressão nas células)
- **`QuaRot PTQ W4A8` (`Triton` + INT8 dinâmico):** `val@512 = 0,8064 BPB` (`2,2491 NLL`, delta **`+0,0036 BPB`**)
- **Early-Exit pareado (`16.384` tokens `val`):** `8/8` camadas = **`0,7810 BPB`** (`2,1781 NLL`) vs `Early-Exit @ 4,0/8` camadas (`c=0.40`) = **`0,8429 BPB`** (`2,3507 NLL`)

### 3.2 Decode `512` Tokens de Validação Real (`NVML`) e Cache (`7,45M`)
| Modo (`decode 512 tokens`) | Vazão (`tok/s`) | Energia NVML (`mJ/tok`) | VRAM pico | Observações |
|---|---:|---:|---:|---|
| `Dense BF16 Eager` | `42,9` | `872,95` | `199,73 MiB` | Referência sem `torch.compile` |
| `AdaptiveEarlyExit` (`5,79/8` camadas, eager) | **`53,4`** *(+24,3% vs eager)* | `742,35` | `200,51 MiB` | `-27,6%` camadas (`59,9%` saídas precoces); precisa buckets compilados |
| `Dense BF16 Compiled` | `1.290,6` | `37,81` | `209,47 MiB` | Baseline compilado (`~15 MB` cabem na L2 de `48 MB`) |
| `QuaRot PyTorch unpack Compiled` | `995,6` | `45,30` | `211,09 MiB` | Pior por desempacotamento em VRAM |
| `QuaRot Triton W4A16 Compiled` | `1.111,6` | `41,77` | `211,09 MiB` | `+11,7%` sobre QuaRot PyTorch; em `7,45M` ainda abaixo do `Dense BF16` residente na L2 |
| `QuaRot Triton W4A8 Compiled` | `1.044,5` | `42,99` | `211,09 MiB` | — |
| `MTP Ungated Compiled (CUDAGraph)` | **`1.838,5`** (`+42,5%`) | **`26,01`** (`-31,2%`) | — | `accept=35,5%` (`1,71 tok/ciclo`); `greedy_exact_match=False` nesta rodada em `.to(bfloat16)` |
| `MTP Confidence-Gated (0.25)` *(eager)* | `59,9` | `661,98` | — | Aceitação saltou **`35,5% → 66,9%`** (`360` rascunhos podados!), mas caiu de `1.838 → 59,9 tok/s` porque o código atual desabilita `CUDAGraph` quando `threshold > 0` |
| `SegmentMemoryCache` (`256` tokens) | Hit **`0,499 ms`** vs Miss `40,42 ms` | — | — | **`81,0×`** speedup em prefix hit na L4 |
