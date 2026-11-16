# IA2 Campanha 012 (NVIDIA L4 — Roteiro Fase 2: Kernel Fused Triton em `state_kernels.py` / `triton_dplr.py`)

Data: 2026-11-16  
Branch: `llm-ideas` (revisões executadas na L4: `972854a` + ajuste `v2` em `state_kernels.py`)  
Hardware: `NVIDIA L4` (`SHAPE_STANDARD`), PyTorch `BF16` + `torch.compile(backend="inductor", dynamic=False)`, Triton `3.8.0`, modelo campeão `fractal_pcc_mtp_muon_bpe:s351` (`vocab=4096`, `8.192` tokens/passo, `2` warmup + `20` passos medidos com `torch.cuda.synchronize()`).

---

## 1. Implementação e Descoberta Arquitetural (`triton_dplr.py` + `state_kernels.py`)

1. **`src/llm2/architectures/triton_dplr.py`:** kernels `_dplr_inter_chunk_fwd_kernel` e `_dplr_inter_chunk_bwd_kernel` com acumulação estrita em `FP32` (`input_precision="ieee"`), registrados como `@torch.library.custom_op("llm2::dplr_inter_chunk_scan_fwd" / "_bwd")` com `register_fake` e `register_autograd`.
2. **Descoberta em GPU real (L4 SRAM):** o backward para a célula `assoc` (`K=96, V=48`, que exigiria `BLOCK_K=128`) excedeu a shared memory por SM da L4 (`OutOfResources: 163840 > 101376 bytes`). Por isso `can_use_triton_dplr` limita o kernel Triton a `max(chunk_size, v_dim, k_dim) <= 64` (cobrindo integralmente as `6` células `level` em `K=V=64`, para `chunk=32` e `chunk=64`).
3. **Eliminação de unroll em `assoc` (`v2`, `state_kernels.py`):** na primeira versão (`972854a`), como `assoc` ficava fora da `custom_op`, o `torch.compile` continuava desenrolando o loop de `n_chunks` para `assoc` (`compile_first_s = 288,28 s` em `seq=2048`). Na `v2`, `dplr_chunked` encaminha todo `use_triton and q.is_cuda` via `dplr_inter_chunk_scan` (`custom_op` usa Triton nos `6` `level` e fallback PyTorch opaco no `assoc`), tornando o tempo de compilação praticamente independente do número de chunks.

---

## 2. Benchmark Pareado em L4 (`fractal_pcc_mtp_muon_bpe:s351`, `8.192` tok/step)

| Versão / Backend | `chunk_size` | `seq × B` | `loss0` (paridade) | Compile 1º passo (`s`) | `ms / step` | Vazão (`tok / s`) | VRAM pico (`GiB`) |
|---|---:|---|---:|---:|---:|---:|---:|
| PyTorch desenrolado (`use_triton=False`, `972854a`) | `32` | `512 × 16` | `10.513080` | `266,32` | **`176,81`** | **`46.331,7`** | `3,61` |
| Triton `972854a` (c/ unroll em `assoc`) | `32` | `512 × 16` | `10.512999` | `179,95` | `228,84` | `35.798,2` | `3,74` |
| Triton `972854a` (c/ unroll em `assoc`) | `64` | `512 × 16` | `10.513071` | `156,52` | `207,28` | `39.522,1` | `3,88` |
| Triton `v2` (sem unroll em `assoc`) | `32` | `512 × 16` | `10.513042` | **`128,09`** (`-51,9%`) | `229,71` | `35.662,0` | `3,75` |
| Triton `v2` (sem unroll em `assoc`) | `64` | `512 × 16` | `10.513079` | **`121,89`** (`-54,2%`) | `206,35` | `39.699,9` | `3,89` |
| **PyTorch desenrolado (`use_triton=False`, `v2`)** | `32` | `2048 × 4` | `10.516771` | **`804,76`** *(13,4 min!)* | `325,21` | `25.190,1` | `3,59` |
| `custom_op` + PyTorch fallback (`v2`, grafo em cache) | `32` | `2048 × 4` | `10.516823` | `7,12` *(cache)* | `382,15` | `21.436,6` | `3,74` |
| Triton `972854a` (c/ unroll em `assoc`) | `32` | `2048 × 4` | `10.516739` | `288,28` | `244,57` | `33.495,8` | `3,74` |
| Triton `v2` (sem unroll em `assoc`) | `32` | `2048 × 4` | `10.516829` | **`118,09`** (**`-85,3%` / `6,81×`**) | `246,96` | `33.170,8` (`+31,7%`) | `3,74` |
| **Triton `v2` (sem unroll em `assoc`)** | **`64`** | **`2048 × 4`** | `10.516880` | **`125,83`** (**`-84,4%` / `6,40×`**) | **`223,42`** | **`36.666,0`** (**`+45,6%`**) | `3,88` |

---

## 3. O que explicamos e ganhamos vs Campanha 011

1. **Causa raiz do gargalo `4.869 tok/s` em C011 identificada:** em `seq=2048 × 4` (`64` chunks desenrolados por camada), o PyTorch + Inductor consumia **`804,76 s` (~13,4 min)** apenas na recompilação quando o currículo transitava para `2048`, além de cair para `25.190 tok/s` no steady-state. Com `Triton v2 + custom_op`, a compilação cai para **`118,09–125,83 s`** (**`-85,3%`**) e o throughput sobe para **`33.171` (`chunk=32`)** e **`36.666 tok/s` (`chunk=64`)** — **`7,5×` acima da média da janela `2048` na C011** e **`+45,6%`** acima do baseline steady-state PyTorch desenrolado (`25.190 tok/s`).
2. **Trade-off honesto em contexto curto (`seq=512 × 16`):** com apenas `16` chunks, o PyTorch desenrolado + Inductor ainda é mais rápido no steady-state (`46,3k` vs `39,7k tok/s` em `chunk=64`), porque nossos kernels Triton inter-chunk usam `input_precision="ieee"` (`FP32` escalar) enquanto o PyTorch sob `autocast(bf16)` aproveita Tensor Cores nas projeções por chunk. Em contexto longo (`seq=2048+`), o ganho de compilação e vazão do Triton domina por larga margem — especialmente combinável no futuro com **`chunk_size=64`** ou TF32/BF16 onde permitido.
3. **Verificações complementares (`v2`):**
   - `pytest -q -m cuda tests/test_state_recurrent.py tests/test_fractal_delta.py tests/test_decode_optimization.py` na L4: **`10 passed, 86 deselected em 20,20 s`** (paridade `forward`, `state` e todos os `7` gradientes em `(64,64,1,513)`, `(48,64,1,512)`, `(48,96,1,512)` e `(64,64,64,256)` + `test_state_cuda_inductor`).
   - E2E smoke `ia2_benchmark --smoke --unified-curriculum --measure-energy --device cuda`: exit `0` em `71,8 s` (`training_loss 7.1705 -> 6.9205`, `val@256 = 5.5646`, `bpb = 2.5929`, `decode_inductor = 1766,6 tok/s`, `Speculative MTP = 1373,6 tok/s`, `1,24×`).
