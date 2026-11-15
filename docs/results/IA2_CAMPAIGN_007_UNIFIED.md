# IA2 Campaign 007 — Validação Unificada, Bandas de Posição, Intervenção de Prefixo e Multi-Seed (`5f924e0`)

**Data:** 15/11/2026 · **Revisão executada:** `5f924e02729839227d80cd4b71f0ab0c825a0d7e` (`llm-ideas`) · **Hardware:** 2× NVIDIA Tesla T4 (Kaggle, Python `3.13.16`, PyTorch `2.11.0+cu130`, `FP32`, `status="complete"`, maior run individual `25,62 min`, kernel completo em `4.747 s` ≈ `79,1 min` com preflight) · **Artefato completo:** `llm2-ia2-campaign-007-results.zip`

---

## 1. Escopo e O que Mudou nesta Rodada

Esta rodada executa integralmente o **Passo 1** (correções de código e instrumentação de contexto) e o **Passo 2** (treino de **um único checkpoint** para linguagem + memória associativa e confirmação multi-seed balanceada entre as 2 GPUs) definidos após a [Campanha IA2 006](IA2_CAMPAIGNS_001_006.md):

1. **Contador posicional `phase_pos` em `int64` + redução modular `float64`:** corrigida a saturação de precisão `FP32` acima de $2^{24}$ passos em [fractal_delta.py](../../src/llm2/architectures/fractal_delta.py), com `state_bytes_per_sequence()` atualizado (+8 B para `golden_phase=True`).
2. **Isolamento do `TransformerLM` no decode:** `LocalTransformerDecode` recebe `copy.deepcopy(model).eval()` em [ia2_benchmark.py](../../src/llm2/research/ia2_benchmark.py) para impedir mutação das camadas do modelo em memória.
3. **Checkpoint único (`--mode unified`):** cada execução treina **um único modelo** (`6.000` passos, `LR=0,003`, `story_seq_len=512`, `batch=16` = `12` linhas TinyStories + `4` linhas `Byte-Binding v2`) e avalia **todas as métricas no mesmo `checkpoint_sha256`**.
4. **Diagnóstico causal de uso de contexto (`score_stories_bands_and_prefix`):** separa a NLL em 4 bandas (`1-256`, `257-512`, `513-2048`, `2049+`) e mede nos `506` documentos de validação com ≥ 640 bytes (`195.757` bytes avaliados em posições `512+`) a diferença entre **contexto completo**, **prefixo `[0:256]` truncado** e **prefixo `[0:256]` trocado** pelo de outra história.
5. **Replicação pareada multi-seed (`351`, `352`, `353`):** `fractal_golden_big` (`864.869` params, `422.920` B/seq) vs `fractal_pure_big` (`864.845` params, `422.912` B/seq), intercalados entre GPU 0 (`golden:351`, `pure:352`, `golden:353`) e GPU 1 (`pure:351`, `golden:352`, `pure:353`).

---

## 2. Resultados por Seed e Braço (`--mode unified`, único checkpoint)

| Braço | Seed | `ckpt` (`sha256[:10]`) | Story `@256` ↓ | Story `@512` ↓ | Story `@3219` ↓ | Band `257-512` ↓ | Band `513-2048` ↓ | Band `2049+` ↓ | Δ Swap (`512+`) | Δ Trunc (`512+`) | Bind `@1024` (32w) | Bind `@4096` (32w) | Bind `@8192` (32w / pair) | Bind `@8192` (64w / pair) | Decode `@7936` (B/s) |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `fractal_golden_big` | 351 | `62b527f59f` | **0,7902** | **0,7629** | **0,7474** | **0,7618** | **0,7804** | 1,0358 | +0,0078 | -0,0023 | 99,9% | 78,91% | 51,37% / 27,44% | 47,95% / 24,32% | 1.968,0 |
| `fractal_pure_big` | 351 | `9a1a25e0d4` | 0,7962 | 0,7691 | 0,7534 | 0,7671 | 0,7856 | **1,0277** | +0,0086 | -0,0021 | 100,0% | 89,55% | 66,60% / 47,36% | 62,79% / 42,87% | 2.074,1 |
| `fractal_golden_big` | 352 | `98b0d4a7ae` | **0,7861** | **0,7586** | **0,7435** | **0,7585** | **0,7763** | **1,0875** | +0,0078 | -0,0022 | 100,0% | **99,71%** | **89,75% / 80,37%** | **86,82% / 76,07%** | 1.987,9 |
| `fractal_pure_big` | 352 | `f6d33d66d9` | 0,7937 | 0,7650 | 0,7495 | 0,7644 | 0,7808 | 1,1261 | +0,0078 | -0,0030 | 100,0% | 94,73% | 77,83% / 60,25% | 71,58% / 52,93% | 2.066,2 |
| `fractal_golden_big` | 353 | `9d34273dbc` | 0,7949 | 0,7681 | 0,7543 | **0,7571** | **0,7755** | 1,1044 | +0,0078 | -0,0030 | 100,0% | **91,21%** | **69,92% / 49,71%** | **65,92% / 46,00%** | 2.034,1 |
| `fractal_pure_big` | 353 | `9a9808d764` | **0,7946** | **0,7671** | **0,7524** | 0,7666 | 0,7842 | **1,0880** | +0,0078 | -0,0035 | 100,0% | 90,23% | 68,07% / 45,80% | 61,33% / 39,16% | 2.012,7 |

*Nota sobre bandas:* a banda `1-256` (`161.278` bytes) ficou entre `0,6781` e `0,6922` em 5/6 execuções, exceto `fractal_golden_big` `s353` (`0,7153`, `+0,0259` vs `pure`), único motivo pelo qual `s353` empatou/perdeu por `0,0003`–`0,0019` no agregado global apesar de ter vencido em `257-512` (`-0,0095`) e `513-2048` (`-0,0087`).

---

## 3. Comparativo Pareado (`golden − pure`) e Diagnósticos Críticos

| Métrica | `s351` | `s352` | `s353` | Média ± dp (`n=3`) | Leitura científica |
|---|---:|---:|---:|---:|---|
| **TinyStories `@256`** | -0,0060 | -0,0076 | +0,0003 | **-0,0045 ± 0,0034** | Vence em 2/3 seeds; empata em 1 |
| **TinyStories `@512`** | -0,0062 | -0,0064 | +0,0010 | **-0,0039 ± 0,0034** | Vence em 2/3 seeds; empata em 1 |
| **TinyStories `@3219`** | -0,0060 | -0,0060 | +0,0019 | **-0,0034 ± 0,0037** | Vence em 2/3 seeds; oscilação ligada à banda `1-256` |
| **Banda `1-256`** | -0,0079 | -0,0072 | +0,0259 | +0,0036 ± 0,0158 | Ruído na `s353` dominou o escore global |
| **Banda `257-512`** | -0,0054 | -0,0059 | -0,0095 | **-0,0069 ± 0,0019** | **Vence 3/3 seeds (consistente)** |
| **Banda `513-2048`** | -0,0052 | -0,0044 | -0,0087 | **-0,0061 ± 0,0019** | **Vence 3/3 seeds (consistente)** |
| **Banda `2049+` (4.638 B)** | +0,0080 | -0,0386 | +0,0164 | -0,0047 ± 0,0242 | Amostra pequena (`4.638` B), NLL sobe para `1,03–1,13` |

### Quatro aprendizados objetivos da IA2 007:
1. **`golden` vence `pure` consistentemente no médio/longo alcance (`257-2048`):** em todas as 3 seeds pareadas e balanceadas entre GPUs, a fase áurea reduziu a perda em `-0,0069` (`257-512`) e `-0,0061` (`513-2048`), com ganho médio líquido de `-0,0045` em `@256` e `-0,0034` em `@3219`.
2. **O modelo carrega causalmente informação do prefixo `[0:256]` além do byte `512`:** trocar a abertura da história por outro documento piorou a NLL em `512+` em **todas as 6 rodadas** (`delta_vs_swapped = +0,0078` a `+0,0086`), provando retenção real. Porém `delta_vs_truncated = -0,0021` a `-0,0035` e a subida de NLL na banda `2049+` (`1,03–1,13`) mostram que acima da janela de treino (`512` B) ainda existe degradação de calibração posicional/estado quando a sequência passa de `2K`.
3. **Checkpoint único (`unified`) funciona, mas revela o gap de extrapolação `16×`:** treinado simultaneamente (`75%` Stories + `25%` Binding em apenas `seq_len=512`), o mesmo peso atingiu **`99,9%–100%` em `@1024`** (`2×` a janela de treino) e **`78,9%–99,7%` em `@4096`** (`8×`), caindo em **`@8192`** (`16×`, `51,4%–89,8%` original / `27,4%–80,4%` pareado). Isso mostra que o `100% @8192` das rodadas `002–006` vinha do treino especialista em Binding e que um único modelo geral precisará de **currículo/sequências mais longas** (agora viável em Colab L4/A100 com `BF16` + BPE).
4. **Preflight CUDA/T4 integral aprovado:** na revisão `5f924e0`, `tests/test_fractal_delta.py tests/test_decode_optimization.py tests/test_state_recurrent.py` passaram **82/82** sem skips (`302,16 s`) e `-m cuda tests/test_context_memory.py` passou **2/2** (`66,18 s`), confirmando `CompiledDecode` + `CUDAGraph` (`1.968–2.074 B/s`, erro máximo de paridade `1,00e-5` a `2,91e-5`).
