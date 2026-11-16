# Validação vigente — 16/11/2026

## Roteiro Fase 2: Kernel Fused Triton em `state_kernels.py` / `triton_dplr.py` (`IA2 Campanha 012`, `972854a` + ajuste `v2` em `state_kernels.py`)

Relatório detalhado em [docs/results/IA2_CAMPAIGN_012_PHASE2_TRITON.md](docs/results/IA2_CAMPAIGN_012_PHASE2_TRITON.md).

Verificações reais executadas:
- **CPU local pós-edição (`v2` em `state_kernels.py`, Windows):**
  - `python -m pytest -q tests/test_state_recurrent.py tests/test_fractal_delta.py tests/test_decode_optimization.py` → **86 passed, 10 skipped em 144,22 s** (os `10 skipped` são exclusivamente os testes `@pytest.mark.cuda`, executados diretamente na L4).
- **Google Colab NVIDIA L4 (`Triton 3.8.0`, PyTorch `BF16` + `torch.compile`, `fractal_pcc_mtp_muon_bpe:s351`):**
  - `pytest -q -m cuda tests/test_state_recurrent.py tests/test_fractal_delta.py tests/test_decode_optimization.py` → **10 passed, 86 deselected em 20,20 s** (`forward`, `state` e todos os `7` gradientes em `(64,64,1,513)`, `(48,64,1,512)`, `(48,96,1,512)`, `(64,64,64,256)` + `test_state_cuda_inductor`).
  - **Benchmark pareado `8.192` tok/step em `seq=2048 × 4`:** PyTorch desenrolado (`use_triton=False`) levou **`804,76 s`** de compilação e **`25.190,1 tok/s`** (`325,21 ms/step`, `3,59 GiB`); **`Triton v2` (`chunk=32`)** caiu para **`118,09 s`** de compile (**`-85,3%` / `6,81×`**) e **`33.170,8 tok/s`** (`+31,7%`); **`Triton v2` (`chunk=64`)** fechou em **`125,83 s`** de compile e **`36.666,0 tok/s`** (`+45,6%` vs PyTorch desenrolado, `1,71×` vs `custom_op` PyTorch `21.436,6 tok/s`, e **`~7,5×` vs janela C011**), mantendo paridade inicial (`loss0 = 10.516880` vs `10.516771`, `3,88 GiB`).
  - **E2E smoke (`ia2_benchmark --smoke --unified-curriculum --measure-energy --device cuda`):** exit `0` em `71,8 s` (`training_loss 7.1705 -> 6.9205`, `val@256 = 5.5646`, `bpb = 2.5929`, `decode_inductor = 1766,6 tok/s`, `Speculative MTP = 1373,6 tok/s`, `1,24×`).
- **Limitações declaradas:** L4 SRAM limitou `can_use_triton_dplr` a `max(chunk_size, v_dim, k_dim) <= 64` (célula `assoc` com `K=96, V=48` usa fallback PyTorch dentro do `custom_op` opaco sem desenrolar em `torch.compile`); em contexto curto `seq=512 × 16`, PyTorch BF16 desenrolado (`46.331,7 tok/s`) ainda vence o kernel inter-chunk FP32 `ieee` (`39.699,9 tok/s` em `chunk=64`). Treino full-step (`3.500` passos) em Corpus real `50M–150M+` fica para a etapa seguinte da Fase 2. Runtime L4 encerrado (`list_runtimes() == {}`).

## Roteiro Fase 1: Self-Speculative MTP + Telemetria Real NVML/DVFS + Currículo `512→1024→2048` (`IA2 Campanha 011`, `08fbf1f` + ajuste pós-rodada em `ia2_benchmark.py`)

Relatório detalhado em [docs/results/IA2_CAMPAIGN_011_PHASE1_ENERGY_CURRICULUM.md](docs/results/IA2_CAMPAIGN_011_PHASE1_ENERGY_CURRICULUM.md).

Verificações reais executadas:
- **CPU local pré-subida (`08fbf1f`, Windows, Python 3.12.2, PyTorch 2.14.1+cpu):**
  - `python -m pytest -q tests/test_decode_optimization.py tests/test_fractal_delta.py tests/test_state_recurrent.py tests/test_training.py tests/test_stories_data.py tests/test_predictive.py`: **109 passed, 9 skipped em 222,44 s** (skips são testes exclusivos CUDA). Após correção pontual em `ia2_benchmark.py` (`data.arrays["val"]`), `python -m pytest -q tests/test_decode_optimization.py`: **16 passed em 40,82 s**.
- **Google Colab NVIDIA L4 (`08fbf1f`, Linux, Python 3.13.16, PyTorch 2.14.1+cu130, CUDA 13.0, `BF16`):**
  - Preflight L4: `pytest -q tests/test_decode_optimization.py tests/test_fractal_delta.py tests/test_predictive.py` → **56 passed em 263,72 s** (`0 failed`, `0 skipped`); `pytest -q -m cuda` → **55 passed, 781 deselected em 695,50 s**.
  - **Currículo `512→1024→2048` (`fractal_pcc_mtp_muon_bpe:s351`, `3.500` passos, `3,65 GiB` VRAM):** `val@256 = 0,9148 BPB`, `val@512 = 0,9016 BPB`, `val@3219 = 0,8993 BPB` (`2,5080 NLL`), banda `513–2048 = 2,4578 NLL` (vs `3,8526` sem currículo na C010 e `3,2903` no Transformer `15,74M`), `ΔTrunc = +0,0502`, `ΔSwap = +0,1310`, `Binding 8k w=32 = 100,00%`, `HARD w=64 = 99,12%` (`98,73%` pareado).
  - **Varredura real NVML + `Self-Speculative MTP` + DVFS + L2 Pin (`1.024` tokens em texto `val`, paridade `100%` exata vs `CompiledDecode`):** aceitação MTP real de **`31,1%`** (`1,62 tok/ciclo`); `72W` (`1.666,2` vs `1.157,9 tok/s`, `1,44×`, `31,85` vs `40,77 mJ/tok`), `55W` (`1.646,7 tok/s`, `27,26 mJ/tok`), **`40W` mínimo da L4 (`1.562,6 tok/s`, `1,35×` acima do baseline `72W`, `23,45 mJ/tok` = `-42,5%` energia / `1,74×` `Tokens/J`)**.
- **Limitações declaradas:** `cudaDeviceSetLimit` para L2 (`7,70 MB`) aceito pelo driver mas sem ganho mensurável (~`0,2%`) porque o modelo de `7,45M` já cabe na L2 de `48 MB` da L4; `watts_mean` instantâneo sofre pequena inércia térmica/boost em janelas de ~`0,6 s` (energia acumulada NVML `mJ/tok` é a medição limpa); o throughput de treino em `2048×4` caiu para `4,87k tok/s` (`45,3 min` totais), motivando a **Fase 2 (Kernel Triton em `state_kernels.py`)**. Runtime L4 encerrado (`list_runtimes() == {}`).

## Passo 3: BPE `vocab=4096` + Prelude-Core-Coda + Multilag + MTP + Muon vs KDA (`IA2 Campanha 010`, `b7c6461`)

Relatório detalhado em [docs/results/IA2_CAMPAIGN_010_COLAB_L4_BPE.md](docs/results/IA2_CAMPAIGN_010_COLAB_L4_BPE.md).

Verificações reais executadas na revisão `b7c6461` (Etapa 3: `MuonAdamW` com ortogonalização Newton-Schulz, `fractal_prelude_coda` / `cell_index`, `fractal_kda`, `fractal_assoc_multilag`, `prepare_bpe` train-only `uint32` com `bits_per_byte` exato por bytes UTF-8 e 8 braços BPE em `ia2_benchmark.py`):
- **CPU local (Windows, Python 3.12.2, PyTorch 2.14.1+cpu):**
  - `python -m pytest -q tests/test_fractal_delta.py tests/test_training.py tests/test_stories_data.py tests/test_predictive.py`: **57 passed, 3 skipped** (os 3 skips são testes exclusivos CUDA, não contados como aprovação GPU). Inclui retomada exata do `MuonAdamW` com exclusão de `embedding`/`lm_head`/`mtp_heads`, equivalência causal/decode/contagem do `PCC+KDA+multilag+MTP`, e isolamento train-only do BPE + cálculo exato de `BPB`.
  - `python -m pytest -q --ignore=tests/e2e`: **778 passed, 56 skipped, 6 failed** (as 6 falhas são pré-existentes de fixtures CRLF/SHA-256 no Windows em `tests/test_state_b_operator.py` e `tests/test_state_readout_models.py`, sem relação com a Etapa 3).
- **Google Colab NVIDIA L4 (Linux, Python 3.13.16, PyTorch 2.14.1+cu130, CUDA 13.0, BF16):**
  - Preflight L4: `pytest -q tests/test_fractal_delta.py tests/test_stories_data.py tests/test_training.py` → **52 passed em 96,86 s**; `pytest -q -m cuda` → **55 passed, 769 deselected em 122,44 s**; smoke `--smoke --tokenizer bpe --vocab-size 4096 --mode unified --precision bf16` (`transformer_bpe_med`, `fractal_pcc_full_bpe`) → **completo em 0,8 min**.
  - Dataset BPE train-only preparado uma única vez (`prepared-bpe-v4096-96af1443b0104ead-9a961867`, `3,987×`–`4,036×` compressão), `3.500` passos unificados em `seq=512`, `BF16`, `binding_hard_writes=64`:
    - **Fase 1 + Fase 2 (`seed=351`):** `transformer_bpe_med` (`15,74M`, `1,0373 BPB`, decode `346,0 tok/s`, binding `1,6%–2,1%`), `fractal_classic_bpe` (`3,60M`, `1,0924 BPB`), `fractal_pcc_bpe` (`7,45M`, `1,0345 BPB`), `fractal_pcc_muon_bpe` (`7,45M`, `1,0028 BPB`, mas queda em `8k` `92,8%` e `HARD` `75,5%`), `fractal_pcc_mtp_muon_bpe` (`7,45M` inf / `10,59M` treino, **`0,9980 BPB`**, `8k=100,0%`, `HARD=99,6%`) e `fractal_pcc_full_bpe` (`+KDA`, `1,0047 BPB`, `10,97 GiB` VRAM, `16,9k tok/s`). KDA descartado por piorar `BPB`, elevar VRAM em `2,70×` e reduzir throughput de treino.
    - **Confirmação multi-seed (`seeds 351, 352, 353`, `status="complete"`):** `fractal_pcc_mtp_muon_bpe` vs `transformer_bpe_med` (`2,11×` maior em inferência): `val@256 = 1,0451 ± 0,0048` vs `1,0688 ± 0,0028` (`-0,0237 BPB`), `val@512 = 1,0018 ± 0,0040` vs `1,0402 ± 0,0025` (`-0,0384 BPB`), `val@3219 = 0,9978 ± 0,0038` vs `1,0361 ± 0,0024` (`-0,0383 BPB`), `ΔSwap = +0,0861` vs `+0,0753`, `Decode Inductor = 1.395,4 ± 6,6 tok/s` vs `341,2 ± 4,5 tok/s` (`4,09×`), estado `844 KB` vs `6,29 MB` (`7,45×` menor), `Binding 8k w=32 = 100,00%` (`99,97%` pareado) vs `1,50%` (`0,07%`), `HARD 8k w=64 = 99,15 ± 0,49%` (`98,54 ± 0,43%`) vs `1,79%` (`0,00%`).
- **Limitações declaradas:** treino fixo em `seq=512` sem currículo longo manteve banda `513–2048` em `3,8526 ± 0,0762` vs `3,2903 ± 0,1598` do Transformer (alvo de combinação com currículo na Etapa 4); orçamento restrito a `3.500` passos em TinyStories `16,8 MB`, não qualifica modelo fundacional `100M–1B`. Runtime L4 encerrado (`list_runtimes() == {}`).

## Passo 2 (Fechamento): `assoc_val_dim=48` + Currículo `512→1024→2048` na L4 (`IA2 Campanha 009`, `bd7bcb9` / `2baae7c`)

Relatório detalhado em [docs/results/IA2_CAMPAIGN_009_COLAB_L4.md](docs/results/IA2_CAMPAIGN_009_COLAB_L4.md).
- **NVIDIA L4 (BF16, `torch.compile(..., dynamic=False)`, seeds `351, 352, 353`, `39,2 min`):** `fractal_golden_classic_v2_big` (`877.157` params, `assoc_val_dim=48`) elevou `Binding @8192 w=32` para **`99,7% ± 0,3%` (`99,5%`)** e `HARD @8192 w=64` para **`95,1% ± 1,2%` (`90,9%`)**, levando banda `2049+` para `0,7221 ± 0,0134`; `pytest -q -m cuda` → **55 passed, 765 deselected em 695,64 s**.

## Passo 1 + Passo 2: Validação Unificada `IA2 Campaign 007` (`5f924e0`)

Relatório detalhado em [docs/results/IA2_CAMPAIGN_007_UNIFIED.md](docs/results/IA2_CAMPAIGN_007_UNIFIED.md); artefato completo preservado em `llm2-ia2-campaign-007-results.zip` na raiz do workspace.

Verificações reais executadas na revisão `5f924e02729839227d80cd4b71f0ab0c825a0d7e` (após correção de `phase_pos` para `int64` + redução modular `float64`, isolamento de `TransformerLM` no decode com `copy.deepcopy`, diagnóstico por bandas/intervenção de prefixo e implementação do `UnifiedStream`/`UnifiedTrainer`):
- **CPU local (Windows, Python 3.12.2, PyTorch 2.14.1+cpu):**
  - `python -m pytest -q tests/test_fractal_delta.py tests/test_state_recurrent.py tests/test_decode_optimization.py`: **76 passed, 6 skipped em 95,22 s** (os 6 skips são testes exclusivos CUDA, não contados como aprovação GPU local). Inclui testes para `phase_pos >= 2**25`, igualdade bitwise dos pesos compartilhados iniciais `pure` vs `golden`, causalidade e retomada estrita do `UnifiedTrainer` com `dropout=0,2` e `grad_accum_steps=2`, e não-mutação de `TransformerLM`.
  - Smoke CPU ponta a ponta `python -m llm2.research.ia2_benchmark --smoke --device cpu --mode unified --seeds 351 352 --lr 0.003 --story-seq-len 512 --binding-hard-writes 64 --arms fractal_golden_big fractal_pure_big`: **concluído em 0,5 min** (`exit code 0`, `status="complete"`).
- **Kaggle 2× NVIDIA Tesla T4 (Linux, Python 3.13.16, PyTorch 2.11.0+cu130, FP32, `79,1 min` totais):**
  - Preflight pytest na T4: `python -m pytest -q tests/test_fractal_delta.py tests/test_decode_optimization.py tests/test_state_recurrent.py` → **82 passed em 302,16 s** (zero skips); `python -m pytest -q -m cuda tests/test_context_memory.py` → **2 passed, 11 deselected em 66,18 s**.
  - Execução completa da **IA2 Campaign 007** (`status="complete"`, `deferred=[]`) em `--mode unified --lr 0.003 --story-seq-len 512 --binding-hard-writes 64` nas 3 seeds pareadas (`351`, `352`, `353`) balanceadas entre GPU 0 e GPU 1:
    - **Comparativo pareado `fractal_golden_big` − `fractal_pure_big` (`n=3`):** TinyStories `@256` **`-0,0045 ± 0,0034`**, `@512` **`-0,0039 ± 0,0034`**, `@3219` **`-0,0034 ± 0,0037`**, banda `257-512` **`-0,0069 ± 0,0019` (3/3)**, banda `513-2048` **`-0,0061 ± 0,0019` (3/3)**, banda `2049+` `-0,0047 ± 0,0242` (NLL entre `1,0277` e `1,1261` em `4.638` B).
    - **Intervenção causal de prefixo distante (`506` docs, `195.757` bytes em `512+`):** `delta_vs_swapped` positivo em **6/6 execuções** (`+0,0078` a `+0,0086`), enquanto `delta_vs_truncated` foi `-0,0021` a `-0,0035`.
    - **Checkpoint único (`--mode unified`) no Byte-Binding:** `99,9%–100,0%` em `@1024`, `78,9%–99,7%` em `@4096`, caindo para `51,4%–89,8%` original (`27,4%–80,4%` pareado) em `@8192` sem currículo acima de `512` B; `CompiledDecode @7936` a `1.968,0`–`2.074,1 B/s` (`cudagraph=True`, erro máximo de paridade `1,00e-5`–`2,91e-5`).
- **Limitações declaradas:** treino ainda em `FP32` e bytes (`257`); `BF16`, tokenização BPE e treino/currículo direto em sequências longas (`4K–8K`) dependem do Passo 3 (Colab L4/A100 em configuração).

## Campanhas IA2 001–006 e confirmação em escala do `fractal_golden` (`dd6797d`)

Relatório consolidado em [docs/results/IA2_CAMPAIGNS_001_006.md](docs/results/IA2_CAMPAIGNS_001_006.md); artefatos brutos preservados em `llm2-ia2-campaign-001-results.zip` a `006` na raiz do workspace.

Verificações reais executadas na revisão `dd6797d62083fdac801557149315c2270b54cb8f`:
- **CPU local (Windows, Python 3.12.2, PyTorch 2.14.1+cpu):**
  - `python -m pytest -q tests/test_state_recurrent.py tests/test_fractal_delta.py`: **60 passed, 6 skipped em 67,14 s** (os 6 skips são testes exclusivos CUDA, não contados como aprovação GPU local).
  - Smoke ponta a ponta `python -m llm2.research.ia2_benchmark --smoke --device cpu --story-seq-len 512 --binding-hard-writes 64 --arms transformer_big fractal_delta_big fractal_pure_big fractal_golden_big`: **concluído em 49,5 s** (`exit code 0`, `status="complete"`), paridade de logits Inductor/eager entre `5,96e-8` e `8,94e-8`.
- **Kaggle 2× NVIDIA Tesla T4 (Linux, Python 3.13.16, PyTorch 2.11.0+cu130, FP32):**
  - Preflight pytest na T4: `python -m pytest -q tests/test_fractal_delta.py tests/test_decode_optimization.py tests/test_state_recurrent.py` → **79 passed em 304,85 s** (zero skips); `python -m pytest -q -m cuda tests/test_context_memory.py` → **2 passed, 11 deselected em 71,30 s**.
  - Execução completa da **IA2 Campaign 006** nas duas T4 com caches Inductor/Triton isolados por GPU (`61,8 min` totais; GPU 0 em `45,3 min`, GPU 1 em `61,8 min`, `deferred_arms=[]`):
    - `transformer_big` (`1.091.328` params): TinyStories `@256=0,7680`, `@512=0,7400`, `@3219=1,4256`; Binding `@8192` `2,5%/0,0%` (`32w`) e `1,1%/0,0%` (`64w`); Decode `@7936` `231,7 B/s` (`parity_max_abs_err=9,54e-6`).
    - `fractal_golden_big` (`864.869` params): TinyStories `@256=0,7707`, `@512=0,7437`, `@3219=0,7282`; Binding `@8192` `100,0%/100,0%` (`32w`) e `99,51%/99,12%` (`64w`); Decode `@7936` `1.861,6 B/s` (`cudagraph=True`, `parity_max_abs_err=1,48e-5`).
    - `fractal_pure_big` (`864.845` params): TinyStories `@256=0,7791`, `@512=0,7517`, `@3219=0,7359`; Binding `@8192` `100,0%/100,0%` (`32w`) e `99,80%/99,61%` (`64w`); Decode `@7936` `1.926,2 B/s` (`cudagraph=True`, `parity_max_abs_err=9,54e-6`).
    - `fractal_delta_big` (`861.225` params): TinyStories `@256=0,7822`, `@512=0,7543`, `@3219=0,7375`; Binding `@8192` `100,0%/100,0%` (`32w`) e `100,0%/99,90%` (`64w`); Decode `@7936` `2.474,8 B/s` (`cudagraph=True`, `parity_max_abs_err=2,00e-5`).
- **Limitações declaradas:** todos os treinos das campanhas IA2 001–006 usaram precisão `FP32` e vocabulário byte-level (`257`); o suporte a `BF16` nativo em L4/A100 e tokenização BPE no `fractal_golden` ainda será validado na próxima etapa.

## Compilação do passo incremental — antes da 0036

**26 passed, 2 deselected em 20,92 s** em Python 3.12.2 e PyTorch CPU.
`python -m pytest -q tests/test_decode_optimization.py tests/test_cached_decode.py tests/test_decode_rebuild.py tests/test_context_memory.py -m 'not cuda'`.
A máscara local em blocos coincide com a máscara densa independente.
Paridade e armazenamento foram conferidos antes e depois da volta do
cache circular. Um backend de captura executou os grafos em CPU e
verificou um único grafo para cada adaptador entre comprimentos. Views
com dimensões unitárias foram normalizadas para evitar especializações
de stride. A suíte também verifica perfis, a referência mais rápida e
o critério de 2× com todas as variantes.

Checagem adicional dos quatro safetensors verificados da 0035 em CPU
com prefixos 255/256/1.024/7.936 e 256 bytes de continuação forçada.
Um grafo por checkpoint e erro máximo absoluto de logits 9,06e-5.
O teste focado de bootstrap passou nos dois modos, reconstrução e
otimização. A compilação Inductor e os tempos reais na T4 serão
verificados na execução da campanha. [Protocolo](docs/DECODE_OPTIMIZATION_RUN.md).

## Campanha 0035 — geração na T4 e controle em CPU

Quatro reconstruções de 6.000 passos nas duas T4, com 8 min 03 s antes da
exportação. 12 perfis de geração com 257 posições conferidas por perfil.
Paridade top-1 completa e diferença máxima de logits 8,20e-5.
Ambos os ZIPs passaram em CRC, hashes e equivalência dos manifestos.
Verificados os quatro pesos, os quatro estados como bytes, a proveniência
Git, 80 métricas de treino e as medianas dos perfis.

O diagnóstico CPU carregou os safetensors verificados e percorreu os
524.508 alvos de desenvolvimento em cada condição. O forward completo
reproduziu as perdas da T4 dentro de 3e-7. A máscara local de 256 posições
por camada preservou os logits das primeiras 256 posições em todos os
documentos e reduziu a NLL do Transformer de 2,024229 para 1,006197.
O recorrente obteve 0,978279. O desempenho do cache local será medido
em outra rodada. [Auditoria da 0035](docs/results/CAMPAIGN_0035.md).

## Reconstrução e geração — antes da 0035

A tentativa 0034 terminou na busca pelos checkpoints. A célula exportou
o relatório de falha. O bootstrap da 0035 usa `bootstrap.json` para os
metadados temporários, conforme a guarda de limpeza.

**24 passed, 2 deselected em 21,50 s** em Python 3.12.2 e PyTorch CPU.
`python -m pytest -q tests/test_decode_rebuild.py tests/test_cached_decode.py tests/test_context_memory.py tests/test_context_training.py -m 'not cuda'`.
Integração local com quatro treinos curtos, arquivos safetensors novos,
verificação de configuração e proveniência, carregamento dos novos hashes
para inferência e validação dos dois ZIPs. O ensaio de deadline preservou
o checkpoint completo disponível. O teste simulado do bootstrap verificou
o dispatch, os downloads e o nome aceito pela guarda de limpeza.
Os testes CUDA foram excluídos da execução local. O treino de 6.000 passos
e as medições da 0035 ocorrerão nas T4. [Operação](docs/CACHED_DECODE_RUN.md).

Célula 13 conferida contra o bootstrap Git `94b5aa83d2aff64dd068e74e7f6fbeedc6baa5f0`.
O teste simulado confirmou a seleção da reconstrução, o teto de 3.600 s e
o callback de download. O teste focado do bootstrap passou após configurar
o ambiente cuBLAS.

## Geração incremental — antes da 0034

**17 passed, 2 deselected em 6,43 s**, Python 3.12.2 e PyTorch CPU.
`python -m pytest -q tests/test_cached_decode.py tests/test_context_memory.py -m 'not cuda'`.
Equivalência de logits com vários comprimentos e posições de RoPE, reset,
capacidade de cache, estado recorrente de tamanho fixo, leitura de pesos
safetensors com hashes, rejeição de divergência e relatório de geração
com modelos de teste. O loader de inferência leu somente metadados e pesos.

Um smoke com checkout destacado na revisão original `db507b5` conferiu
a fonte e exportou um ZIP válido de falha quando os pesos estavam ausentes.
O teste usou pesos artificiais nas verificações locais de inferência.
Os quatro pesos reais serão conferidos pelo Kaggle antes de construir os
modelos. A paridade e a velocidade na T4 serão medidas na 0034.
[Protocolo](docs/CACHED_DECODE_RUN.md).

Célula 12 conferida contra o blob Git do bootstrap em
`3f71accb5413a6efe9afdffeb692cb018d250f6c`. Teste simulado verificou modo desativado,
rejeição de payload alterado, limite de 1.800 s e callback do download.

## Campanha 0033 — T4 e auditoria

16 treinos de 6.000 passos concluídos nas duas T4, seeds 341/342, com 35 min
34 s registrados antes da exportação. Preflight registrado pelo Kaggle com
**19 testes CPU aprovados (19,93 s)** e **2 testes CUDA/Inductor aprovados
(73,52 s)**. Versões PyTorch 2.11.0+cu130 e Python 3.13.16.

Auditoria local com Python 3.12.2 e Torch CPU. O ZIP passou na verificação de
CRC, caminhos e 412 membros únicos. Foram confrontados 409 hashes do Git,
154 arquivos do snapshot, manifesto e tokenizer com o preparo local, 16
configurações e metadados de checkpoints, 320 métricas, 64 avaliações e 96
gerações. Recalculados os marcos de desempenho, os tempos medianos e a
energia por byte. Os pesos finais estão no ZIP separado de checkpoints.

Streaming em chunks e sequência inteira diferiram por até 3,59e-5 nos
probes dos candidatos. Duas passagens independentes obtiveram menor NLL
que o Transformer de 149 mil parâmetros nas duas seeds. O perfil T8192
mede forward eager em bytes aleatórios. A qualidade foi avaliada em T256.
[Relatório da 0033](docs/results/CAMPAIGN_0033.md).

## Memória contextual — antes da campanha 0033

[Protocolo](docs/CONTEXT_MEMORY_RUN.md), [plano prévio](docs/PLAN_CONTEXT_MEMORY_IMPLEMENTATION.md).
Windows/Python3.12.2/PyTorch2.14.1+cpu, `PYTHONPATH=src`. Sem CUDA local. A nova
família não foi qualificada na T4 ainda; os dois testes CUDA/Inductor serão
executados no Kaggle antes dos treinos, com dimensões reais e contexto256.

**39 passed, 4 deselected, 98,97s, exit0**:
`python -m pytest -q tests/test_context_memory.py tests/test_context_training.py tests/test_context_operations.py tests/test_stories_models.py tests/test_stories_training.py tests/test_stories_bootstrap.py tests/test_stories_operations.py -m 'not cuda'`.
Inclui kernel sequencial versus blocos, forward/backward/estados, decaimento
extremo, causalidade nas fronteiras, streaming em chunks irregulares, isolamento
de documentos, contribuição do ramo amplo, contagem sem pesos duplicados e
forward inicial idêntico entre compartilhado/independente com dropout desligado.
Estados finais não retêm o armazenamento de toda a sequência via views.
Memorização/finitude, retomada exata com dropout0,2/acumulação2, piloto restaurado.

Integrações incluídas nessa suíte: campanha nova de oito workers CPU reais e
campanha legada de quatro. Duas taxas de calibração, novos treinos principais,
textos, diagnósticos, perfis, último checkpoint porjob, um snapshot e verificação
dos dois ZIPs finais. Reuso sem retreino; protocolo antigo ignorado pela nova
receita; config e fonte continuam verificadas. Falha/deadline/disco exportam
parcial. Marcos de velocidade bloqueados quando falta seed, horizonte ou qualidade.
Dispatch/relógio/seeds/Torch/downloads exercitados nos dois perfis do bootstrap.

Regressão comum: **36 passed, 3 deselected, 109,16s, exit0**:
`python -m pytest -q tests/test_stories_data.py tests/test_binding_models.py tests/test_training.py tests/test_research_configs.py tests/test_stories_delivery.py -m 'not cuda'`.
Inclui dados/divisões, defaults anteriores, treino/checkpoints e célula0032 preservada.
O warning legado de optimizer.step vem do wrapper do teste de benchmark; a ordem
real permanece optimizer antes do scheduler. Deselected são os testes CUDA;
nenhum skip foi contado como aprovação.

Qualificação adicional CPU com dados TinyStories reais preparados: quatro modelos,
seed11033, 200passos, B8/T256, taxa0,001, warmup25/scheduler200, sem compilação,
apenas treino e `cal_val`. A perda caiu de 5,56–5,58 para:

| Modelo | NLL final de calibração |
|---|---:|
| Transformer86k | 2,521143 |
| Contextual simples | 2,204664 |
| Duas passagens compartilhadas | 2,233852 |
| Hierarquia compartilhada | 2,215107 |

É teste de aprendizado/finitude, não ranking científico, calibração da campanha
ou alegação de velocidade. Todas as variantes/taxas previstas permanecem na matriz;
esses pesos não serão continuados. Dados: SHA portátil
`d61615e79496f14ef473cc10ad11e5deac6d2bda89be7046d738498ff16ba849`.
Registro local com configurações, fontes e contadores em
`artifacts/development/context-cpu-qualification-20261111/qualification.json`.
Nenhum teste reservado foi avaliado. CUDA/energia/eficiência em2h continuam pendentes.

SHA-256 desse registro: `d70575a44e19bd925a0061e7abbf4b92f8d8d675f5f118f97b6efe6721f513ed`.
Após tornar os marcos visíveis no HTML e bloquear candidatos com horizonte
incompleto: **2 passed, 4 deselected, 45,08s, exit0**,
`python -m pytest -q tests/test_context_operations.py -k 'resource_milestone or full_cpu'`.
Exercita novamente publicação, oito workers e reuso; não repete a qualificação
de 200 passos nem transforma o smoke CPU em resultado de campanha.

Células 10 e 11 e seus perfis de bootstrap verificados com
`python -m pytest -q tests/test_stories_delivery.py tests/test_stories_bootstrap.py`.
**11 passed em 1,21s**. A célula 11 usa a revisão
`db507b5258615e94cab7620dc5816fb31c36d8c7` e confere o SHA-256 do bootstrap
`dc53834fb3b7c6f10f81914abc8e33105e34a008274e55232b75ec415326d815`.
Testes cobrem o payload alterado, a revisão exata, relógio, seeds, teto de 2h,
dispatch e callback dos botões de download.
Após publicar em `experiment/fractal-lrm`, a URL fixada do bootstrap retornou
HTTP 200 e 4.687 bytes com o SHA-256 exato da célula.


## Campanha 0032 — T4 e auditoria dos resultados

[Relatório](docs/results/CAMPAIGN_0032.md): status complete, oito treinos novos,
seeds331/332, 6.000 passos; 1.271,61s antes de exportar. ZIP2.527.514bytes/212membros,
CRC válido e caminhos seguros, sem pesos/pickle/corpus binário. 388 hashes do
bootstrap contra Git6f469fa; snapshot150arquivos (107fontes/dependências e43configs).
Configurações resolvidas, protocolo/manifesto portátil, parâmetros por meta/fórmula,
embedding ligado, workers, metadados de checkpoints, resumos, 160registros finitos,
32avaliações completas e48gerações por bytes originais reconciliados. Dados locais
verificados têm o mesmo manifesto; nenhum teste reservado avaliado.

Preflight no Kaggle: **CPU13passed/2deselected,18,57s; CUDA2passed/5deselected,
32,40s**, GPU0. Todos os braços treinados/perfilados nas duas T4 com alternância
porseed; Python3.13.16/Torch2.11.0+cu130. Forward/gradientes do caminho bloqueado
qualificados nas formas desses testes; não todos os backends/precisões.
Sem traceback/OOM ou fallback registrado. Cache máximo amostrado167.970.718bytes,
sem violação da guarda, removido ao final. Metadados finais apenas: tensores da
campanha não fornecidos, não rehashados nem carregados localmente.

NLL média maior/pequeno/simples/pirâmide0,802563/1,128058/1,334464/1,329944.
Nenhuma comparação com as referências qualifica ganho linguístico. Blocos versus
referência dos mesmos pesos em8K: -71,91–72,23%VRAM incremental,1,53–1,66xmais
rápido, erro máximo de logits4,73e-5–1,31e-4. Isso é forward eager, não linguagem
8K, geração com cache ou custo para qualidade igual.

Auditoria executada por leitura/asserções sem treino ou inferência dos pesos finais.
Probe adicional CPU com pesos aleatórios/seed10032/dropout0: dois prefixos com
mesma contagem de triplas e sufixo comum dão diferença1,49e-7 nos logits finais
de ambos os candidatos. Ilustra a perda de ordem do agregado local; não é prova
de causa única da perda treinada. Nenhum código de treino foi alterado nesta auditoria.

## TinyStories — validação local anterior à campanha 0032

[Plano prévio](docs/PLAN_TINYSTORIES_IMPLEMENTATION.md),
[protocolo e limitações](docs/TINYSTORIES_RUN.md). Windows, Python 3.12.2,
PyTorch 2.14.1+cpu, `PYTHONPATH=src`. Sem CUDA local; os dois testes CUDA novos
não estavam aprovados nessa entrega. A qualificação real posterior está acima.

Execução bloqueada versus referência: forward/backward e estados, streaming,
causalidade em fronteiras e fases 0/1/3, pesos ligados e contagens. Preparação:
prefixos limitados e SHA fixados, registros completos, deduplicação entre divisões,
manifesto portátil, documentos isolados e caudas/padding/BOS contabilizados.
Teste reservado bloqueado no avaliador. Treinamento: finitude/memorização,
retomada exata com dropout 0,2 e acumulação 2, restauração dos pilotos e próxima
atualização, duas taxas train-only e inicialização principal nova.

Suíte ampliada final: **48 passed, 6 deselected, 81,24 s, exit 0**:
`python -m pytest -q tests/test_stories_models.py tests/test_stories_data.py tests/test_stories_training.py tests/test_stories_operations.py tests/test_stories_bootstrap.py tests/test_binding_models.py tests/test_training.py tests/test_research_configs.py -m 'not cuda' -k 'not full_cpu'`.
Inclui contagens/configs, rotação física, helpers fixados, relógio/Torch/dispatch,
falha/deadline/disco com exportação parcial e preservação de arquivo alheio.
O warning do wrapper de optimizer.step é do teste legado de equivalência do
benchmark; não altera a ordem real dos updates. Deselected inclui CUDA e a
integração completa exercitada separadamente abaixo.

Integração final de quatro workers CPU reais: **1 passed, 5 deselected, 22,80 s,
exit 0**, `python -m pytest -q tests/test_stories_operations.py -k full_cpu`.
Inclui calibração das duas taxas, novos treinos principais, checkpoints finais,
gerações, barreira/perfis, curvas PNG, verificação dos dois ZIPs e segunda
campanha que reutiliza os quatro jobs completos sem novo treino. As configurações
resolvidas correspondem à taxa realmente selecionada. Checkpoints intermediários
e estados descartáveis de calibração não permanecem como pesos duplicados.

Entrega da célula 10: **4 passed, 0,11 s, exit 0**,
`python -m pytest -q tests/test_stories_delivery.py`. SHA literal conferido contra
o bootstrap do Git `6f469fa6b80f2109b92c8b7be4a052c8c2ffaddd`; célula desativada
sem rede, payload alterado bloqueado, dispatch/seeds/relógio e callback de download
verificados por simulação. Nenhum treino foi iniciado na conta Kaggle nesta entrega.
Após publicação, download HTTP real do bootstrap fixado retornou 4.391 bytes com
SHA-256 exatamente igual ao da célula; links locais da entrega e diff conferidos.

Dados reais: download HTTP parcial total de 34 MiB na revisão fixada; seleção
produziu manifesto `d61615e79496f14ef473cc10ad11e5deac6d2bda89be7046d738498ff16ba849`,
16.777.421 bytes de treino/18.876 histórias e divisões distintas de calibração,
desenvolvimento e teste reservado. Uma segunda preparação em diretório novo,
partindo dos dois prefixos brutos locais, reproduziu exatamente esse SHA.
Contagens de três bytes: NLL 1,699787, bpb
2,452275 sobre todos os 524.508 bytes de desenvolvimento; 68.959.268 bytes de
estado. É referência estatística com orçamento próprio, não rede equivalente.

Smoke adicional em dados reais: 200 passos CPU, seed 11001, B8/T256, taxa 0,0003,
sem compilação e apenas treino/subdivisão de calibração. Memória simples:
NLL 5,59619 para 2,73839; Transformer pequeno: 5,54350 para 3,18167. Serve para
detectar falha de aprendizado/finitude e preparo, não ranking da campanha,
confirmação, custo GPU ou resultado do teste reservado. Esses pesos não serão
continuados nos oito treinos principais. Não houve treino de 6.000 passos local.

## Campanha0031 — qualificação real T4 e auditoria

[Relatório](docs/results/CAMPAIGN_0031.md): oito treinos novos/3000passos, duas
seeds311/312, statuscomplete. ZIP954.594bytes/129membros, CRC íntegro, sem caminhos
externos/duplicatas/pesos/pickle.366hashes de bootstrap e144arquivos de snapshot
conferidos contra Gitbcd7780 (101fontes/43configs); protocolo/gerador/configs,
parâmetros meta/fórmula, embedding ligado, workers/metadados/relatórios/CSV,
96avaliações e160registros de gradiente reconciliados. Regeneradas8192consultas
originais/alteradas para conferir labels; nenhum treino/inferência neural local.
Pesos reais ausentes: hashes declarados conferidos, não os tensores finais.

Preflight Kaggle **CPU13passed/2deselected,33,40s; CUDA2passed/13deselected,33,22s**,
GPU0. Os oito treinos/perfis usaram ambas asT4com atribuição alternada porseed.
Nova família qualificada emCUDA/Inductor nesse ambiente e formas testadas;
isso não aprova todos os backends/precisões/formas. Py3.13.16/Torch2.11.0+cu130.
Sem traceback/OOM; cache amostrado máximo155.209.556bytes, guarda sem violação.

Candidatos100%nas duas seeds, incluindo contrafactuais/capacidade/alinhamento.
Baseline não qualificou; o relatório corretamente mantémfalseas aceitações de
eficiência/compactação a referência qualificada. Sem confirmação de linguagem,
cinco seeds ou custo igualado. Código/treino não foram alterados nesta auditoria.

## Família associativa de transições — entrega posterior à0030

[Plano prévio](docs/PLAN_ASSOCIATIVE_TRANSITION_IMPLEMENTATION.md),
[operação](docs/ASSOCIATIVE_BINDING_RUN.md). Python3.12.2/Torch2.14.1+cpu, Windows,
`PYTHONPATH=src`. Nova arquitetura separada; modelos/protocolos antigos preservados.

- Kernel aditivo vetorizado versus referência sequencial, forward e gradientes;
  causalidade em fronteiras1/3/4/5/7/8/31/32; streaming e gradientes em chunks;
  contagens por fórmula e enumeração; estado real sem buffers de prefixo presos;
  embedding ligado; mesmo forward inicial entre os controles correspondentes.
- Generalização aprendida para novos mapas e chaves trocadas, nos três braços
  neurais novos. Teste de memorização de batch fixo não é a única qualificação.
- Retomada exata com dropout0,2/acumulação2 e próxima atualização bit a bit;
  sampler preserva fases4/8/16/32. Benchmark restaura pesos/otimizador/scheduler/
  RNG/sampler/contadores, sem alterar o horizonte ou guardar updates de piloto.
- Primeira rodada encontrou telemetria que não aceitava o inteiro da fase e
  benchmark que atingia o limite do Trainer. Corrigidos: metadados escalares
  excluídos dos bytes de tensores; updates explícitos de medição e restauração,
  sem ampliar max_steps. Rodada focada9passed/3deselected,5,36s.
- Suíte ampliada final: **85passed,12deselected,97,86s,exit0**. Comando:
  `python -m pytest -q tests/test_binding_models.py tests/test_binding_operations.py tests/test_recall_models.py tests/test_recall_bootstrap.py tests/test_state_recurrent.py tests/test_training.py tests/test_research_configs.py -m 'not cuda' -k 'not full_cpu'`.
  Inclui regressões de modelos/Trainer/configurações; CUDA não executado localmente.
- Integração real de quatro workers CPU, treino4passos, checkpoint/aposentadoria,
  curvas, contrafactuais, barreira de perfis, ZIPs e reuso nativo sem novo treino:
  primeira execução **1passed/4deselected,22,26s**. Após ajustes finais de telemetria
  e sondas, operações+bootstrap: **8passed,17,89s,exit0**, incluindo essa integração,
  falha de worker, espaço insuficiente, deadline/exportação parcial, rotação de
  GPUs, dispatch/relógio/Torch e botões. Nenhum arquivo alheio removido.

Última execução conjunta da nova família/operação/bootstrap, incluindo equivalência
do update medido com o Trainer: **21passed,2deselected,74,29s,exit0**.
Comando:`python -m pytest -q tests/test_binding_models.py tests/test_binding_operations.py tests/test_binding_bootstrap.py -m 'not cuda'`.
Os dois deselected são CUDA. O único warning vem do wrapper de optimizer.step
usado no teste de equivalência, não de mudança da ordem de updates na operação.
Entrega da célula09: **4passed,0,11s,exit0**; SHA literal contra Gitbcd7780,
execução desativada sem rede, payload alterado bloqueado antes deexec e despacho/
relógio/seeds/callback de download simulados. Após publicação, downloadHTTP real
conferiu4370bytes e SHA-256 exato da célula09. Links locais e diff conferidos;
nenhuma execução na conta Kaggle foi feita por esta entrega.

Qualificação de desenvolvimento adicional: seed816/1000passos reais no núcleo
86.785parâmetros, CPU/FP32/dropout0,05/contexto256/B16/LR0,001. Avaliações novas de
32sequências/512consultas por condição: original e par contrafactual100%em
256/1024/4096/8192com32associações,1024com64e1027com prefixo3. Registro e limites:
[qualificação](docs/results/transition_development_qualification.json). Medição
explícita apenas dos pesos com a versão equivalente de prefixo exclusivo;
fontes de treino/medição separadas, sem carregar pickle ou alegar retomada entre
fontes. Não houve controle Transformer nem medição GPU/energia/linguagem local.

Os dois testes novos CUDA/Inductor eram obrigatórios no Kaggle e foram aprovados
na0031acima. A qualificação CPU isolada não sustenta ganho de velocidade nem capacidade
geral de uma LLM. Histórico0030 e documentação abaixo permanecem como evidência
anterior, não resultado dessa nova campanha.

## Campanha0030 — auditoria da execução parcial T4

[Relatório](docs/results/CAMPAIGN_0030.md): ZIP1.076.272bytes/196membros, CRC íntegro,
sem caminhos externos/duplicatas/pesos/pickle. Conferidos345hashes do bootstrap e
138arquivos de snapshot contra Gitdc2888a,95fontes e43configs,24registros de worker,
identidade do protocolo/gerador, configs, contagens meta/fórmula, embedding ligado,
hash inicial igual nos dois bancos concluídos,40avaliações e80registros de gradiente.
Quatro treinos/seed271/3000passos completos; dez pendentes. Metadados reconciliados;
tensores finais ausentes, sem rehash de pesos/estado nem forward/treino científico
local. Testes Kaggle: **CPU14passed/2deselected,15,88s; CUDA2passed/14deselected,
78,10s**, GPU0. Duas T4 utilizadas nos treinos. Não é aprovação de qualidade.

Auditoria local reproduzível em `artifacts/audits/campaign_0030/audit.py`:
regeneração dos exemplos/labels/distâncias, oráculo simbólico100% nos quatro
comprimentos, preditor de frequência sem chave e alcance dos padrões calculados.
Esses controles não são modelos aprendidos nem novos resultados de linguagem.
Finitude/gradientes presentes não demonstram associação; o teste histórico de
memorização de batch fixo não qualificava generalização para novos pares.
Recomendações de rever mistura local/inicialização e reduzir recompilação não
foram implementadas nesta auditoria. Código de treino e célula fixada preservados.

## Memória consultável/Fibonacci — implementação e limites reais

[Plano prévio](docs/PLAN_MEMORY_RECALL_IMPLEMENTATION.md),
[operação](docs/MEMORY_RECALL_RUN.md). Python3.12.2/Torch2.14.1+cpu, Windows,
`PYTHONPATH=src`. Nenhum treino científico de recuperação na T4 feito localmente.

- Modelos/gerador/índices/causalidade/gradientes/contagens/finitude/memorização/
  retomada: primeira rodada corrigida11passed/2deselected,7,71s. Depois incluídos
  streaming com banco cheio, referência sequencial e contagem de custo de validação.
- Suíte focada e regressões de estados, controles0029, Trainer e configurações:
  **85passed,12deselected,74,07s,exit0**, excluindo integração longa e CUDA.
  Comando:`python -m pytest -q tests/test_recall_models.py tests/test_recall_operations.py tests/test_recall_bootstrap.py tests/test_state_recurrent.py tests/test_state_readout_models.py tests/test_training.py tests/test_research_configs.py -m 'not cuda' -k 'not full_cpu_engineering'`.
- Integração real CPU:`python -m pytest -q tests/test_recall_operations.py -k full_cpu`:
  **1passed,6deselected,245,35s,exit0**. Sete treinosT32/4passos,14pilotos frios/
  aquecidos, piloto concorrente, sete avaliações/perfis posteriores isolados,
 14avaliações de comprimento e14sondas de capacidade. ZIPs reais conferidos por
  hashes/manifestos/cobertura de checkpoints. Só último completo por job, fonte
  única, sem after-NN/pesos no ZIP de resultados.
- A integração também executa mais duas campanhas: histórico completo reutilizado
  sem treinar e avaliação de treino concluído recuperada após interrupção simulada,
  preservando pesos/contador. Banco completo e três padrões têm mesmos parâmetros
  iniciais. Guarda de fonte/dados/configuração/ambiente permanece ativa.
- Falhas injetadas de worker, deadline e espaço insuficiente exportam pares ZIP
  parciais válidos, preservando arquivos alheios. Admissão mantém pares, horizontes
  e schedulers. Caminho procedural agora absoluto e roundtrip YAML idêntico, após
  a integração detectar resolução relativa inconsistente. Manifesto identifica
  fingerprint procedural para verificar o par de arquivos; não enfraqueceu a
  verificação existente. Campo de progresso/modelo/GPU também conferido.
- Retomada compara próxima atualização bit a bit com dropout0,2 e acumulação2,
  incluindo pesos, otimizador, scheduler e sampler NumPy. Mudança de padrão é
  incompatível. Testes de distribuição não recebem os rótulos como input.
- Bootstrap simulado confere relógio desde célula, preservação do Torch, incremento
  de número reservado, seeds e downloads. Helpers reais conferidos contra Git1c8ce2a.
  O bootstrap não repete download/preparo/qualificação de WikiText.

Último ajuste apenas de contabilização: validação do Transformer acumula seu
custo forward estimado, em vez de deixar zero. Teste específico acrescentado;
não altera forward, loss, otimização ou gerador. Estimativas de backbones de
estado continuam null. Testes novos focados foram reexecutados após o ajuste;
o resultado exato dessa execução consta na seção de entrega abaixo.

CUDA/Inductor/energia reais desta arquitetura permanecem **não verificados
localmente**. Há dois testes CUDA obrigatórios no Kaggle antes dos pilotos;
marcados como deselected nos comandos CPU, nunca aprovados por suposição.
Perfis serão de forward batch1/T1024, não geração/KV ou comparação iso-compute.
Dois seeds e uma tarefa sintética não estabelecem superioridade geral como LLM.
Nenhum resultado dos smokes de4passos é evidência de qualidade ou velocidade.

Entrega da memória: após a correção da contabilização de validação, suíte nova
focada reexecutada: **24passed,3deselected,11,22s,exit0**. Comando:
`python -m pytest -q tests/test_recall_models.py tests/test_recall_operations.py tests/test_recall_bootstrap.py -m 'not cuda' -k 'not full_cpu_engineering'`.
Os3deselected são os2CUDA e a integração longa já executada separadamente.
Sintaxe dos módulos novos e links locais conferidos; `git diff --check` limpo.
Célula08 fixada emdc2888a: **4passed,0,15s,exit0**. Hash literal conferido contra
Git, execução desativada sem rede, bootstrap alterado bloqueado e despacho/relógio/
callback de download simulados. Nenhuma execução na conta Kaggle foi feita localmente.

## Campanha0029 — evidência recebida do Kaggle

[Auditoria](docs/results/CAMPAIGN_0029.md): ZIP íntegro,234membros/4.303.546bytes.
Conferidos325hashes do checkout contra Git77b2058,132arquivos do snapshot, protocolo,
configurações, parâmetros únicos/estados por meta/fórmula, cinco inicializações
novas e igualdade dos dois hashes iniciais da hierarquia. Cinco treinos completos
6000passos/seed251, dez avaliações completas e dez de subconjunto; métricas/perdas/
posições/relatório reconciliados. Bins não fornecidos; hashes declarados iguais ao
corpus fixado, tokenizer rehashado. Pesos reais não fornecidos: somente metadados,
hashes e retenção final conferidos, sem carregar pickle ou realizar avaliação local.

Preflights reais registrados no bundle: CPU11passed/2deselected,15,59s; CUDA
2passed/11deselected,84,57s naGPU0. Ambos os slots T4 exercitaram treinos e pilotos
FP32/Inductor. Avisos de depreciação, sem traceback/OOM. Teste reservado intacto.
Perfis finais isolados30amostras/uma GPU por braço; energia NVML de placa em30s,
host/compilação excluídos. Sem geração, recall, crossover de placas, IC entre seeds
ou custo igualado. Inferioridade local da hierarquia não prova falha geral fractal.

Auditoria local reexecutável em `artifacts/audits/campaign_0029/audit.py`, sem
extração de arquivos nem avaliação de modelos. Exit0; gerou JSON/CSV e figura
inspecionada visualmente. Nenhuma fórmula de treino ou configuração foi alterada
nesta auditoria; testes de implementação já aprovados não foram repetidos.

## Campanha0029 — implementação dos controles de profundidade/leitura

[Plano prévio](docs/PLAN_STATE_READOUT.md), [operação](docs/STATE_READOUT_RUN.md).
Python local3.12.2/Torch2.14.1+cpu, Windows, `PYTHONPATH=src`. Fonte nova,
nenhum treino científico iniciado localmente e nenhum checkpoint antigo migrado.

- Modelos novos, primeira rodada CPU:11passed/2deselected,10,32s.
- Operação/retenção/estimativas/histórico:6passed/1deselected,2,56s.
- Integração real:1passed/6deselected,116,56s; a captura do teste em Windows
  emitiu aviso de decodingCP1252. Corrigida a captura paraUTF-8 explícito.
- Regressões antigas + controles/operador/bootstrap novos:
  **77passed,8deselected,132,19s,exit0**, sem o aviso de captura. Comando:
  `python -m pytest -q tests/test_state_recurrent.py tests/test_state_study.py tests/test_state_readout_models.py tests/test_state_readout_operations.py tests/test_state_readout_bootstrap.py -m 'not cuda'`.
- Retomada também exercitada diretamente no ReadoutTrainer/DocumentBatchStream,
  dropout ativo e acumulação2: **2passed,11deselected,4,53s**, próxima atualização,
  pesos/otimizador/scheduler/sampler iguais à execução ininterrupta. Leitura
  incompatível bloqueada. A nova validação por fase não altera RNG/treino.
- Preflight portátil: fixture histórica7203bytes conferida contra Git1c8ce2a e
  referência0027. Não depende de objeto histórico num clone shallow. CPU após
  essa adaptação: **11passed,2deselected,6,48s**; acesso Git histórico simulado como
  proibido nos testes de equivalência do default.
- Célula07 fixada em77b2058: **4passed,0,13s,exit0**. SHA literal verificado contra
  Git; payload alterado bloqueado, execução desativada sem rede e despacho/download
  simulados. Não executa conta Kaggle local.

Integração executa cinco treinos CPU/T17/4passos, dez pilotos frios/aquecidos,
piloto concorrente, cinco perfis isolados e exportação real. Verifica20avaliações,
contagens por fase, inicialização igual nas hierarquias, apenas checkpoint final,
metadados intermediários preservados, par ZIP com cinco estados completos e uma
fonte canônica. Segunda execução reutiliza os cinco jobs completos sem treinar.
É smoke de engenharia, não resultado de linguagem ou velocidade.

Falhas injetadas de worker e falta de espaço exportam par parcial íntegro e
preservam os dados originais. Testes de admissão mantêm pares/horizontes; fonte,
dados, configuração ou versões divergentes no histórico são bloqueados. ZIPs
antigos não são varridos. Recibo de erro de preflight sobrevive à limpeza do
checkout. Helpers de clone/limpeza conferidos contra bytes Git fixados.

Defaults/forward/contagens de parâmetros conferem com a implementação histórica.
Novos caminhos testados em causalidade, fronteiras, streaming forward+gradientes,
finitude, memorização, embedding compartilhado e isolamento. Fonte nova usa versão2
nos controles novos, sem enfraquecer contratos de checkpoint. Limpeza atua apenas
em intermediários redundantes da execução nova e caches próprios de lane inativa.

Limitações: nenhum CUDA/BF16/Inductor/A100 novo localmente. Os8deselected acima
são testes CUDA, não aprovação GPU. A célula executará dois preflights CUDA novos
naGPU0 e pilotos reais nas duas T4. Não repetir qualificação das três famílias nem
reescrever relatório0025. Perfis em uma GPU por variante não confirmam eficiência
nas duas GPUs. Suite completa fora desse conjunto não repetida. Uma seed e LR
fixa são triagem; confirmação continua exigindo seeds novas/custo igualado.

## Auditoria da campanha0028

[Relatório](docs/results/CAMPAIGN_0028.md),
[dados conferidos](docs/results/campaign_0028_audit.json). Somente análise/documentação.

- ZIP419.492bytes,14membros, CRC íntegro, sem duplicação/caminhos externos ou
  payloads de pesos/pickle/datasets. SHA registrado; dois snapshots verificados.
- 80fontes/43configurações científicas contra Git1c8ce2a;5arquivos operacionais
  contra Git55dec8e. Referência idêntica à entregue. Manifesto/fingerprint igual0027.
- Quatro originais reproduzidos; erro máximo2,6091e-9. Onze condições têm as mesmas
  1087janelas/136batches/1.113.088alvos. Zero optimizer-steps/teste reservado.
- Perdas/deltas/bits por byte e totais por fase/quarto da janela recalculados.
  Somatórios, RMS/cossenos e contagens consistentes entre JSONs/CSV/campanha.
- Logs lidos integralmente: sem traceback/OOM/timeout; máscaras0/1 comprovam
  uso das duas GPUs nos quatro workers. Dados/pesos nativos reutilizados.
- Auditoria local exit0, sem carregar/avaliar modelos. Figura gerada com apenas
  valores auditados e verificada visualmente. Suite de treino não repetida, pois
  nenhum modelo, contrato ou treinamento foi modificado.
- Análise do índice original, conferido por SHA:2244casos de offsets0–3 e
  comprimentos1–33 verificam que ler o grupo recém-fechado usaria apenas bytes
  observados. É análise matemática, não teste de uma implementação futura.

Uma seed e intervenções pós-treino, sem IC entre seeds/ablação treinada. Contraste
por posição não demonstra causa nem ganho de uma mudança de leitura. Runtime
observado não foi exportado como inventário separado; o código fixado compara
versões com a referência. ZIP sem tensores para rehash direto; durações/VRAM são
diagnósticos concorrentes, sem alegação de eficiência de treino/geração. Relógio
antes de exportação final; sem inventário remoto completo ou recibo de limpeza.

## Diagnóstico congelado da0027 — implementação e entrega

[Escopo/riscos/aceitação prévios](docs/PLAN_STATE_DIAGNOSTIC.md),
[instruções](docs/STATE_DIAGNOSTIC_RUN.md). Modelo/treino científicos originais
permanecem1c8ce2a; nova operação é executada como arquivo separado.

- `python -m pytest -q tests/test_state_diagnostic.py tests/test_state_diagnostic_bootstrap.py`:
  **18passed,35,36s,exit0**, no Python3.12.2/Torch2.14.1+cpu local, `PYTHONPATH=src`.
- Célula06 fixada em55dec8e: **4passed,0,22s,exit0**. SHA literal conferido
  contra o objeto Git; execução desativada não acessa rede, payload alterado é
  rejeitado e despacho/download são simulados. Não executa conta Kaggle local.
- Integração extrai Git archive1c8ce2a para diretório novo, fabrica quatro pesos
  CPU/T17 e dados por documento, depois executa o novo operador contra esse pacote
  original em subprocessos. Quatro jobs/11condições, zero optimizer-steps, NLL
  original reproduzida e um ZIP sem pesos/pickle. Isso é smoke de engenharia, não
  resultado de linguagem ou equivalência CUDA dos checkpoints reais.
- Falha de reprodução injetada exporta ZIP parcial/JSON não reproduzido e mantém
  pesos originais. Testes cobrem causalidade das intervenções, reconstrução da
  hierarquia, perdas/contagens por posição, RNG/modos/pesos inalterados, hooks
  removidos em falha, deadline/cobertura, hash/config/fontes/versões e caminhos.
- Restauração inclui somente safetensors/metadados, sem desserializar pickle.
  Limpeza retira somente os dois arquivos próprios, sem exclusão recursiva;
  origem/backup e alvos externos preservados. Links/ancestrais vinculados bloqueados.
- Bootstrap simulado verifica pesos antes de instalar, mantém `PYTHONPATH` antigo,
  recusa fontes divergentes e passa o orçamento incluindo setup. Download embutido
  reutiliza o helper científico já fixado. Não há compilação ou benchmark novo.
- Backup real localizado:92.880.997bytes,23membros, CRC íntegro, SHA-256
  `2b4724f97c012876334d0776fb89ee50d564a17566328563603c630f9bebc076`.
  Os quatro hashes de pesos/metadados reais conferem com a referência. Cabeçalhos
  safetensors enumeram2.174.232/2.174.232/1.136.652/2.164.992parâmetros únicos.
  Cópias temporárias dessa conferência foram removidas; ZIP original preservado.

Limitações: nenhuma avaliação científica dos pesos reais localmente, nenhum teste
CUDA novo, nenhuma velocidade T4 alegada. A execução Kaggle deverá repetir a NLL
original em5e-5 antes de interpretar intervenções. Suite completa não repetida;
nenhuma alteração de matemática, treino ou contratos de retomada.

## Auditoria da campanha 0027

[Relatório](docs/results/CAMPAIGN_0027.md),
[conferência estruturada](docs/results/campaign_0027_audit.json).
Somente análise/documentação; sem alteração de modelo, worker, contrato ou células.

- ZIP12.342.536bytes,363membros, CRC válido, sem duplicação/caminhos externos.
- 80fontes +43configurações contra Git1c8ce2a;30snapshots com123arquivos cada;
  quatro arquivos e snapshot operacionais contra Git6396959. Qualificação0025
  preservada byte a byte; mapa operacional igual em bootstrap/campanha/relatório.
- Protocol ID recalculado; quatro configurações validadas, parâmetros por fórmula
  e enumeração meta conferidos. Quatro treinos6000passos reiniciados do zero;
  49.152.000alvos por braço; metadados3000/6000, métricas e contadores consistentes.
- Oito avaliações completas e oito em subconjunto; NLL/bits por byte/perplexidade
  conferidos entre métricas/workers/relatório. Oito tentativas LR; seleção recalculada
  e divisão interna1408/175/175documentos verificada disjunta dos oficiais.
- Oito perfis isolados,30amostras positivas/finitas cada: medianas/médias e energia
  por alvo recalculadas; contraste/decisão refeitos por `state_report.analyze`.
- Logs Kaggle: CPU51passed/6deselected32,17s; CUDA2passed/40deselected72,63s
  naGPU0. Duas T4 executaram treino/perfis. Sem traceback/OOM/erro de compilação
  nos logs lidos. Nenhum modelo consultou teste reservado.
- Auditoria local executada com sucesso (exit0), sem carregar pickle/pesos,
  treinar ou avaliar modelos. Figura original conferida visualmente; conecta só
  avaliações completas. Uma seed241; texto genérico “Three seeds” não é evidência
  de três réplicas. Pesos e binários preparados ausentes: apenas hashes declarados
  e consistência de metadados conferidos, sem rehash de tensores/dados ausentes.

Nenhum teste GPU local novo. Suíte de treino não repetida por alteração exclusiva
de análise/documentação. Sem IC entre seeds, medição de geração/recuperação ou
custo igualado. Duração registrada antes da exportação final; lista de snapshots
intermediários não prova sua remoção remota posterior.

## Recuperação do cache após falha0026

[Plano prévio](docs/PLAN_STATE_B_CACHE_RECOVERY.md),
[conferência real do corpus](docs/results/STATE_B_CACHE_RECOVERY.json).
O log fornecido contém CPU51passed/6deselected (36,15s) e CUDA2passed/40deselected
(78,88s); falhou no checksum de shard existente, antes dos pilotos/treinos.
Não houve NLL nem checkpoint. Sem arquivo atual do Kaggle, causa/hash observado
da corrupção não são conhecidos. A guarda de integridade funcionou.

- Novo preflight apenas operacional; worker científico e hashes continuam1c8ce2a.
- Testes de cache/bootstrap:19passed/1deselected,5,10s; após testar falha anterior
  a instalação/CUDA:20passed/1deselected,4,67s; versão com recibo persistente:
  **21passed,1deselected,4,73s**,exit0.
- Regressões de importação WikiText/contrato: **3passed,25deselected,0,56s**,exit0.
- Células por revisão/hash após fixar6396959: **10passed,0,49s**,exit0;
  fonte conferida contra Git e despacho simulado, sem iniciar conta Kaggle.
- Quatro shards reais do backup local foram lidos e extraídos em diretório novo
  de artifacts, sem modificar o backup. Todos conferem com SHA originais.
- Corpus preparado novamente a partir desses shards; `verify_dataset` aceitou
  os quatro hashes originais (train/val/test/tokenizer), sem alteração de contrato.
  Fingerprint:f10ae5a31fb066205e9da284726f5f0b756cf2e2e68bd7cb15caf0ca17018c81.
- Download real do shard de validação fixado:657.209bytes, SHA original conferido;
  esse teste de transporte não avalia um modelo no conjunto de validação/teste.
- Fixtures provam preservação do legado inválido, cópia independente de válidos,
  rejeição de HTML/hash incorreto, falha de rede, limpeza de temporário em erro
  normal, publicação concorrente sem sobrescrita, links, limite de transferência
  e parser AST que não executa o código de origem. Recibo de falha sobrevive à
  exclusão do checkout. Preflight que falha impede instalação e despacho do worker.

Não repetido treino/integrador de5min: esta alteração não muda matemática,
otimizador, sampler ou passos; modifica transporte/verificação prévia de arquivos.
Nenhuma nova execução GPU local. Os testes CUDA acima são os do log do usuário,
não aprovação da nova rotina de cache no Kaggle. Suite completa não repetida.

## Entrega da etapa B — ciência qualificada e operação separadas

Plano prévio: [escopo/riscos/aceitação](docs/PLAN_STATE_B_DELIVERY.md).
[Operação](docs/STATE_B_RUN.md). A fonte científica permanece fixada em1c8ce2a;
novos arquivos operacionais não substituem o validador científico nem seus hashes.
Python local3.12.2, Torch2.14.1+cpu, Windows, instalação direta, `PYTHONPATH=src`.

| Verificação executada | Resultado |
|---|---|
| Primeiros testes operacionais, sem integração | 9passed,1deselected,5,09s |
| Operação B + núcleo/protocolo/utilitários, com integração real | **76passed,6skipped,306,46s**,exit0 |
| Após acrescentar somente campos/print de previsão e seeds restantes | **10passed,1deselected,5,19s**,exit0; integração não repetida após esse acréscimo de metadados |
| Células B/A/limpeza por revisão e hash, após entrega fixada | **10passed,0,55s**,exit0; validação contra objetos Git e despacho simulado sem conta Kaggle |

A integração extrai o Git archive da revisão científica1c8ce2a e executa o novo
operador como arquivo em subprocesso com `PYTHONPATH` do worker antigo. Executou
quatro jobs mLSTM/Transformer em duas lanes CPU, oito tentativas de taxa, oito
pilotos isolados, piloto concorrente e oito perfis finais. Todos os checkpoints
e a campanha possuem exatamente os hashes da fonte científica qualificada da0025.
Foi verificado `operator-source.zip`, identificação operacional no relatório,
curvas, par ZIP definitivo e ausência dos ZIPs temporários após cobertura validada.
Esse smoke tem4passos/T17; não é resultado de linguagem nem evidência de desempenho.

Testes focados cobrem: família mantida entre comandos, troca de variante,
impossibilidade de limpar GPU0 em uso, preservação de GPU1 e arquivos científicos,
liberação de lock em falha, reserva de pós-perfis incluindo compilação fria,
rejeição de tuning incompatível, importação da fonte errada e fonte de bootstrap
alterada. O bootstrap simulado instala somente o worker científico e passa seu
`PYTHONPATH`; SHA do relatório original da0025 e fontes Git foram conferidos.
Os testes de núcleo/protocolo incluem causalidade, forward/backward, finitude,
memorização, compartilhamento e retomada com dropout/acumulação.

Limitações: nenhuma nova execução CUDA/Inductor local. A0025 qualifica a regra
recorrente na T4, não toda a hierarquia nem esta operação B. Os preflights originais
na T4 continuam obrigatórios. Manter cache não garante cache hit; warmup, tempo
de comando e tempo total são medidos. Pilotos/estimativas podem recusar um bloco;
não há garantia de conclusão em2h. Sem alteração de horizonte, scheduler ou dataset.
Suíte completa não repetida; verificações focadas e integração reais acima.

## Auditoria da campanha 0025 — 31/10/2026

[Relatório](docs/results/CAMPAIGN_0025.md),
[dados estruturados](docs/results/campaign_0025_audit.json). Sem alteração de código,
pesos, treinamento ou contratos científicos; somente análise/documentação.

- ZIP75.152bytes,21membros, CRC íntegro; SHA-256 registrado.
- 123 hashes de fontes/configurações conferem com objetos Git da revisão1c8ce2a;
  fonte entre campanha/seleção coincide. Seleção recalculada: `mlstm`.
- Quatro pilotos reais CUDA/Inductor aprovados no Kaggle (RWKV e mLSTM, duas T4):
  forward/gradientes, finitude, causalidade e streaming registrados. Duas execuções
  Gated Delta foram encerradas pelo supervisor em600,42s, sem falsa aprovação.
- 30amostras por piloto aprovado: valores finitos/positivos e medianas conferidas.
  Logs lidos integralmente; sem timeout absorvido/traceback/codegen error/fallback.
- Testes CPU registrados no Kaggle:51passed,6deselected,36,53s. Isso não equivale
  à execução de toda a suíte CUDA de hierarquia, que permanece pendente para B.
- Cache: pico amostrado927,95MiB,4,696GiB acumulados removidos, limite não excedido.
  Checkout shallow3,39MiB; política de remoção declarada, sem recibo posterior no ZIP.
- Tempo registrado55,17min. Nenhum treino científico, NLL ou teste reservado.

Não foram repetidos testes locais de treinamento: não houve alteração de código.
Nenhuma nova GPU foi usada localmente. A fonte científica qualificada permanece
fixada e não foi modificada para registrar esta auditoria.

## Campanha 0024, download e armazenamento — 31/10/2026

Plano prévio: [auditoria/armazenamento](docs/PLAN_CAMPAIGN_0024_STORAGE.md).
[Conclusão científica](docs/results/CAMPAIGN_0024.md). Alterações somente na branch
experimental. Python local3.12.2/Torch2.14.1+cpu; comandos com `PYTHONPATH=src`.

- ZIP recebido: CRC íntegro,22membros,87.636bytes; SHA-256 registrado na auditoria.
  Os120 hashes de fontes/configurações conferem com o commit científico registrado.
- Seis logs T4 auditados: todos registram o alarme absorvido; primeiros passos
 707–968s excedem600s. Tempos de mLSTM são evidência operacional provisória.
  Nenhum resultado de qualidade ou ganho fractal foi inferido.
- Primeira rodada focada/regressão:125passed,9skipped,2failed,98,88s. As falhas
  estavam em fixtures novas (tamanho de `b'cache'` e sessão menor que a reserva);
  corrigidas, sem enfraquecer verificações do produto.
- Núcleo, protocolo, armazenamento/download e integração real em dois processos:
  **64passed,6skipped,242,46s**,exit0. Inclui causalidade, forward/backward,
  finitude, memorização, retomada com dropout/acumulação, triagem8tentativas,
  quatro jobs, perfis e par de ZIPs finais sem acúmulo dos intermediários.
- Regressões de Campaign/Session/bootstrap/ciclos2–4 e utilitários novos:
  **115passed,9skipped,92,23s**,exit0, após o teste de download em erro parcial e
  a distinção entre deadline global e timeout de candidato. Sem CUDA local.
- O supervisor mata um worker real que envolve o trabalho em `except BaseException`;
  o próximo candidato consegue iniciar. Seleção rejeita registros sem supervisor,
  com tempo excedido e contratos antigos. Falha de candidato não aprova seu caso.
- Download: bytes recuperados do HTML são idênticos ao arquivo; partes reconstroem
  exatamente o original, com hashes, sem cópias em disco. Callback de download em
  falha parcial do worker é testado por simulação; navegador novo ainda não testado
  localmente. Usuário confirmou o método base64 no Kaggle na mensagem desta entrega.
- Limpeza: preserva pesos e ZIPs; bloqueia workers, arquivos ignorados desconhecidos,
  checkout modificado e alvos externos; links internos não apagam destinos externos.
  Git shallow com fetch de SHA exato validado contra repositório local de fixture.
- Limite de cache testado com arquivos pequenos: aborta sem excluir ciência.
  Não é um limite instantâneo de bytes: amostra5s, podendo ultrapassar entre amostras.
- Após proteger também commits locais não publicados: utilitários finais
  **14passed,4,28s**,exit0. Integração completa não foi repetida após essa guarda
  adicional de limpeza; matemática/treino permanecem inalterados.
- Células de entrega fixadas na revisão1c8ce2a: **6passed,0,25s**,exit0;
  hashes conferidos contra objetos Git, execução simulada e rejeição de fonte
  alterada. Não executam limpeza real nem acessam a conta Kaggle nesses testes.

Nenhum CUDA/Inductor novo executado nesta máquina. A0024 valida somente a execução
antiga e revela sua falha de orçamento; não é aprovação do código novo. O encerramento
de grupos POSIX precisa de verificação no Kaggle; a máquina de testes é Windows.
Suíte completa não repetida nesta alteração; integração e regressões focadas são
registradas acima. Horizontes de treinamento e matemática dos backbones não mudaram.

## Arquitetura recorrente/hierárquica inspirada em fractais — 31/10/2026

Plano prévio: [plano aprovado](docs/PLAN_RECURRENT_FRACTAL.md). Código somente no
checkout `experiment/fractal-lrm`. Python local explícito3.12.2, Torch2.14.1+cpu,
Windows; nenhum venv, nenhum treino científico, nenhum teste T4 novo executado aqui.

| Verificação realmente executada | Resultado |
|---|---|
| Primeira implementação das regras/backbones | 33passed,21,31s |
| Regras, amostragem por documento, budgets e seleção | 44passed,6skipped,41,57s; depois inicialização pareada44passed,6skipped,22,05s |
| Primeira integração real em dois processos CPU | 1passed,173,49s; triagem8tentativas,4jobs,8perfis e exportação |
| `python -m pytest -q`, suíte completa | **504passed,46skipped,553,39s**,exit0, antes do ajuste final de storage compacto/curvas |
| Após compactar caches e exportar curvas: três arquivos de testes novos | **51passed,6skipped,297,06s**,exit0; inclui integração completa e remoção dos ZIPs temporários verificados |
| Após isolar falhas de compilação por candidato: núcleo e protocolo | **51passed,6skipped,25,37s**,exit0; falha de um candidato não impede os outros em fixture explicitamente simulada |
| Contexto real1024, batch1, três regras compartilhadas | Forward/backward finitos, logits(1,1024,257), parâmetros1.136.652/1.132.800/1.136.652; CPU, sem ranking de velocidade |
| Parâmetros de produção por fórmula/enumeração em meta | Controles independentes2.174.232(GDN/mLSTM),2.166.528(RWKV), Transformer2.164.992; sharing menor sem alargar |
| Figura de curvas | Inspecionada visualmente; somente validações completas ligadas; rótulo ENGINEERING SMOKE renderizado após ajuste textual final |
| Links Markdown locais / whitespace | 99destinos válidos,0quebrados; git diff --check sem erros |
| Célula curta de entrega, depois de fixar fonte Git | **4passed,0,23s**; hash confere com objeto Git, modo inativo sem rede, rejeição de fonte divergente e dispatch simulado |

As46/6marcações puladas não representam suporte CUDA. Testes novos cobrem referências
sequenciais independentes contra chunks (outputs/estado/gradientes), causalidade nas
fronteiras, streaming, memória FP32, finitude, memorização, safetensors ligado, fórmula
sem duplicação, igualdade inicial dos controles, e próxima atualização retomada
exatamente igual com dropout0,2/acumulação2. Cache compacto tem teste de storage real,
evitando views que retenham todo o prefixo. Fixtures não são resultados científicos.

Nova avaliação usa janelas contidas em documentos, exclui EOS da byte-NLL e descarta
caudas incompletas igualmente. Portanto não comparar diretamente NLL com política
packed histórica. Tuning usa somente documentos do treino original, com conferência
de hashes; teste oficial não é avaliado. Fonte/dataset/versões não foram relaxados.

Limites explícitos: são três regras de memória num shell comum, não reproduções
dos modelos completos dos artigos. Seleção CUDA e aceleração ainda dependem da T4.
FLOPs dos scans são null; fórmula Transformer antiga não foi aplicada a eles.
Energia GPU quando disponível não representa host/campanha inteira. Geração com
cache comparável fica null/pendente. MC/TTT/ACT/DEQ/precisão reduzida/TinyStories/
escala7–10M não foram implementados como mecanismos ativos: são etapas condicionais.

Suíte completa não foi repetida após os pequenos ajustes finais citados; verificações
focadas/integradas acima correspondem às alterações. Não inferir validação GPU de
testes CPU, fixtures de plataforma ou logs de campanhas antigas.

## Auditoria 0023 e controle/retenção ciclo 5 — 31/10/2026

Plano prévio: [auditoria/retenção/ciclo 5](docs/PLAN_CAMPAIGN_0023_RETENTION_CYCLE5.md).
[Resultados 0023](docs/results/CAMPAIGN_0023.md), [operação nova](docs/CYCLE5_RUN.md).
Auditor executado com `PYTHONPATH=src` e Python explícito3.12.2. A primeira chamada
sem PYTHONPATH selecionou a instalação editável antiga e não encontrou o módulo;
foi corrigida a seleção do checkout, sem alterar verificação de dados ou código.

| Verificação real | Resultado |
|---|---|
| Dois ZIPs locais recebidos | CRC íntegro, membros únicos; results31.823.295bytes / checkpoints464.552.147bytes |
| Fonte e snapshots 0023 | 232 arquivos contra Git6a6af1a; 44snapshots /4.576membros consistentes |
| NLL reconstruída | 72 principais +24probes; 4.469janelas /1.144.064alvos cada; erro máximo7,01692e−9 |
| Estado final | 36 hashes safetensors/estado de18jobs conferidos por streaming, sem pickle |
| Diagnósticos | 18arquivos,7.119floats finitos; não prova de raciocínio/eficiência |
| Novo validador de bundles no material real | Cobertura integral dos18jobs; SHA dos definitivos idêntico às cópias locais |
| Primeira seleção de storage/profundidade/operação | 36passed,1skipped,11deselected,19,69s |
| Integrações reais CPU em dois processos (C5novo e C4compatível) | 2passed,47,18s; metade/final e formato legado cobertos |
| `python -m pytest -q`, executável explícito3.12.2 | **455passed,40skipped**,437,41s,exit0 |
| Após endurecer apenas a entrada Git do auditor | 1passed,0,44s; metadado ZIP com opção Git rejeitado antes de qualquer comando |
| Links Markdown e whitespace | 145 destinos locais válidos; `git diff --check` sem erros nos dois checkouts |

Suíte completa inclui causalidade, forward/backward finito, memorização e retomada
exata CPU/math com dropout0,2/acumulação2 dos quatro braços K1/K2/K4/Transformer;
parâmetros/embedding ligado/fórmula e horizontes-FLOPs pré-fixados. Inclui dois
processos reais por sessão de engenharia, publicação aos50% em jobs, remoção só
após definitivos íntegros, preservação em deadline, conteúdo único, hashes errados,
ZIP quebrado, estado incompleto, worker, symlink, confinamento e proteção dos finais.
Células ativas testadas com fontes/bootstrap simulados, números/hashes/relógio
conferidos; inativas sem rede. Backup anterior fora do Git por novo ignore explícito.

Os40skips não validam CUDA nem comportamento específico de Linux.
Este ambiente tem Torch2.14.1+cpu/Windows, sem medição GPU nova. Os logs recebidos
da0023 registram64CPU/Linux e6CUDA/Inductor aprovados em Torch2.11.0+cu128, Python
3.13.15, duasT4. Evidência remota anterior não valida CUDA dos novos controles;
o próximo preflight executará os quatro braços no hardware real.

Não houve treino científico nem limpeza remota pelo agente. Somente fixtures
temporárias foram removidas; os dois arquivos originais da0023 permanecem locais.
Manifesto/tokenizador e hashes declarados do dataset foram conferidos, sem rehash
dos bins ausentes nos bundles. Fonte de treino0023 permanece congelada em6a6af1a;
nova operação tem protocolo/revisão próprios e não relaxa retomada científica.

## Reconciliação do inventário na substituição — 30/10/2026

O usuário recebeu bloqueio após cerca de 60 s; a mensagem antiga não identificava
as diferenças. Não foi inspecionado o inventário remoto e não se atribuiu uma
causa específica. O comparador exigia também igualdade de nomes e mtime,
bloqueando adições e timestamps. [Plano prévio](docs/PLAN_BACKUP_RECONCILIATION.md).

Agora lista categorias/caminhos, admite adições e compara identidade por
tipo/tamanho/SHA (ou destino do link). Payloads selecionados antigos que estão
ausentes/diferentes são recuperados para pasta exclusiva, sem sobrescrita, com
SHA do destino conferido. ZIP antigo só sai após preservação, espaço, verificação
de CRC/SHA, ausência de workers e rechecagem do conteúdo corrente. Inventário e
mapa antigos ficam junto das versões recuperadas, incluídos no backup novo.

Com o mesmo Python 3.12.2 local, sem alterações no treino:

```text
python -m pytest -q tests/test_kaggle_reset.py tests/test_backup_delivery.py tests/test_cycle4_operations.py -k "not real_two_process and not weights_profile and not early_stop"
53 passed, 1 skipped, 6 deselected in 4.24s
```

Inclui adições/timestamps, ausência/alteração com preservação de ambas as versões,
arquivo substituído por link externo sem seguir o destino, espaço de recuperação,
falha de escrita/hash de destino, mudança concorrente, falha posterior de compressão
e repetição com versões antigas ainda presentes. CRC inválido detectado durante a
leitura SHA, sem segunda passagem `testzip`. Conferido no código da stdlib local
que `ZipExtFile` verifica CRC ao alcançar EOF. Reuso/limpeza ignoram somente mtime;
bytes/caminhos novos ou alterados ainda bloqueiam e mostram diferenças.

Skip: processo POSIX; seis integrações com treino/probes/exportação não selecionadas.
Somente fixtures locais foram removidas. Sem execução remota, limpeza de dados do
usuário ou medição de tempo/tamanho no Kaggle. A causa concreta aparecerá no diff
impresso pelo script atualizado. Segunda célula científica permanece em `6a6af1a`.

## Compressão e substituição do backup — 30/10/2026

Plano: [PLAN_BACKUP_COMPRESSION.md](docs/PLAN_BACKUP_COMPRESSION.md). O usuário
relatou ZIP de quase 8 GB e download sem resposta. A inspeção confirmou ZIP_STORED
na primeira entrega; isso não diagnostica sozinho a causa do download. O padrão
passou a ZIP DEFLATE nível 9, com heartbeat e link relativo. Substituição exige
nome exato, ZIP íntegro, originais iguais ao inventário, workers ausentes e espaço
previsto suficiente. Nenhum original é excluído nessa etapa. A limpeza completa
continua exigindo o SHA-256 calculado na cópia local do novo backup.

Verificação real, Python 3.12.2/Windows, mesma instalação local:

```text
python -m pytest -q tests/test_kaggle_reset.py tests/test_backup_delivery.py tests/test_cycle4_operations.py -k "not real_two_process and not weights_profile and not early_stop"
42 passed, 1 skipped, 6 deselected in 2.07s
```

O skip é POSIX; seis testes de integração com treino/probes/exportação ficaram
fora desta seleção, pois não foram alterados. Casos novos cobrem layout legado
sem campo de compressão; compressão efetiva de bytes repetitivos; DEFLATE/LZMA
com CRC/hash e limpeza; exclusão somente do ZIP autorizado; reuso sem aninhamento;
recusa de originais alterados/ausentes, worker, falta de espaço, ZIP corrompido,
links/diretórios/caminhos externos; falha de compressão mantendo originais e
removendo a saída temporária. A célula ativa foi exercitada com bootstrap
simulado, conferindo nome explícito e nível 9; limpar não chama recompressão.

Nenhum backup remoto foi lido, apagado ou comprimido por este agente. Tamanho e
duração reais do ZIP do usuário não foram medidos. O ensaio com dados repetitivos
é um teste de compressão/integridade, não previsão da taxa de compressão de pesos.
Não houve alteração de arquitetura, hiperparâmetros, protocolo ou treino. A entrega
de correção modifica somente o primeiro entrypoint; o segundo original permanece
fixado em `6a6af1a`, sem substituir a entrega anterior.

## Implementação do ciclo 4 — 30/10/2026

Código na branch `experiment/fractal-lrm`, em checkout isolado dentro de
`artifacts/worktrees/fractal-diagnostic-impl-20261030/` da base comum. Python local
3.12.2 (`.pyenv/pyenv-win/versions/3.12.2/python.exe`), Torch 2.14.1+cpu, Windows;
sem venv novo. A resolução explícita do executável foi necessária porque o shim
`python` não tinha versão global/local selecionada neste ambiente.

Plano registrado antes das mudanças: [PLAN_CYCLE4.md](docs/PLAN_CYCLE4.md).
Operação/limites: [CYCLE4_RUN.md](docs/CYCLE4_RUN.md). Nenhuma campanha científica
foi iniciada localmente e nenhum working remoto foi limpo.

Verificações reais até a conclusão da suíte geral:

| Verificação | Resultado observado |
|---|---|
| Primeira seleção nova, antes da integração | 47 passed, 7 deselected |
| Dois processos reais CPU, quatro checkpoints e avaliações por job | 1 passed; dois treinos/8 avaliações, ZIPs incrementais de um job e bundle final de dois |
| Seleção nova expandida | 53 passed, 1 failed, 1 skipped, 6 deselected; a falha era um teste que tentava modo não compartilhado, já proibido pelo contrato histórico; corrigido o teste, sem relaxar o contrato |
| `python -m pytest -q` (executável explícito acima) | 392 passed, 1 failed, 36 skipped, 428,90 s; única falha: sandbox bloqueou `sh.exe` no teste legado de clone Git local, `NtCreateDirectoryObject / 0xC0000022` |
| Repetição isolada do teste de clone fora do sandbox | 1 passed, 1,75 s; repositórios temporários locais, sem alteração do teste/contrato |
| Seleção final de modelos/operação/backup/JSON, após reserva de pós-treino e proteção contra corrida de deadline | 62 passed, 1 skipped (POSIX), 6 deselected (CUDA), 57,91 s |
| Operação após travar a publicação concorrente das fases | 22 passed, 1 skipped (POSIX), 39,95 s; inclui dois processos reais e 40 fases publicadas sob lock |
| Enumeração/fórmula dos seis modelos de produção, em meta | 18 configurações consistentes; embeddings ligados contados uma vez; totais no guia |
| Links Markdown locais verificados | 122, sem destinos ausentes |
| ZIP original da 0022 | SHA-256 preservado: `460b16daec5895deb87633ed1cd1d4eab40c3e5f64f9c23f22a1f9892a7346ee` |

As aprovações CPU incluem causalidade nos limites dos patches, remoção física dos
ramos e chamadas do núcleo, padrão `None` versus `[4,16]` com RNG/forward/backward
iguais, gradientes finitos, memorização dos seis braços, checkpointing e retomada
estrita CPU/SDPA math. Esta última exercita dropout 0,2 e acumulação 2, compara os
contadores carregados e a próxima loss/modelo/objetivo/AdamW/scheduler/sampler/RNG;
tempos de parede não são exigidos idênticos após nova atualização.

Backup testado com arquivos temporários: último checkpoint íntegro, retenção
conservadora de grupos/formatos desconhecidos, CRC/SHA, reuso, hash local divergente,
alteração do inventário, worker ativo, espaço insuficiente e links externos sem
leitura de metadados ligados ou exclusão dos destinos. Não houve carga de pickle
nesse fluxo. A limpeza real ocorreu apenas nas fixtures temporárias.

Operação testada com duas threads e dois processos independentes, publicação
serializada, falha/cancelamento, admissão por prazo/espaço, exportação parcial,
células inertes sem rede e célula ativa com bootstrap simulado (número do recibo,
relógio novo e limites). Perfil CPU com pesos finais e probe K1/2/4/8 preservaram
todos os hashes dos arquivos do checkpoint científico.

Os 36 skips da suíte geral incluem testes CUDA e um teste de grupos de processos
POSIX. Não representam aprovação de T4/Inductor/NVML nem do deadline em Linux.
O preflight Kaggle executará os testes focados CPU/Linux e os seis braços CUDA
FP32/Inductor com backward/checkpoint/retomada. Memorização/retomada estrita são
evidências CPU; desempenho/energia/compilação reais dependem do hardware remoto.
Os testes de integração CPU usam modelos/dados minúsculos; pilotos e pós-processamento
completos da matriz de 18 treinos não foram executados nesse smoke.

As seções abaixo preservam resultados e condições anteriores.

## Organização do repositório

Checkout: `experiment/fractal-lrm`. Python 3.12.2, PyTorch 2.14.1+cpu, Windows, CUDA indisponível.
O código do checkout foi selecionado explicitamente por `PYTHONPATH=src`.

| Verificação executada | Resultado |
|---|---|
| `python -m pytest -q` neste checkout | **184 passed, 0 failed, 21 skipped**, exit code 0 |
| Empacotamento e notebooks | Cobertos pela suíte: ZIP íntegro, exclusões, recusa de sobrescrita, sintaxe e execução padrão sem rede/treino |
| Regras de ignore | 18 casos positivos/negativos aprovados por checkout; nenhum rastreado indevidamente ignorado |
| Preservação de evidências/notas locais | 1.285 arquivos, tamanho e SHA-256 preservados na base comum |
| Links relativos locais Markdown | Sem destinos ausentes na documentação verificada |
| `git diff --check` | Sem erros |
| Código/modelos/configs | Nenhuma alteração em `src/`, YAMLs científicos ou `pyproject.toml` |

Os 21 skips são testes CUDA; não foram tratados como aprovação. A suíte inclui
contrato de compilação com backend eager, que não comprova funcionamento de Inductor/CUDA.
As suítes dos dois checkouts rodaram em paralelo; seus tempos não são um benchmark.

Também foram executados na base comum `inspect --meta` (98.688 parâmetros, enumeração
igual à fórmula), planejamento de `experiment` sem `--execute` e ajuda dos comandos
operacionais. No experimento, o comando de planejamento do guia Kaggle retornou
`status=not_started`, sem treino/download. O primeiro exemplo sem `--input`/`--data`
falhou e foi corrigido; a impressão do erro expôs a limitação CP1252 documentada no guia.

Não foram executados novamente o exemplo de treino do guia, o notebook habilitado na
nuvem nem testes em GPU. Os testes de regressão usam dados e treinos temporários de
engenharia; nenhuma nova campanha científica foi iniciada. Nenhum commit/push foi feito.

## Evidência científica anterior

A [campanha 0015](docs/results/CAMPAIGN_0015.md) contém medições anteriores na T4
FP32/Inductor, com seu próprio commit e ambiente. Isso não valida A100/BF16 nem a
execução dos notebooks modificados nesta limpeza. Os dados curados foram extraídos
do ZIP recebido; não são resultados produzidos pelos testes CPU acima.

- [Auditoria de limpeza e limites](docs/CLEANUP_AUDIT.md).
- [Guia operacional](docs/OPERATIONS.md).
- [Histórico de validações preservado](docs/archive/VALIDATION_HISTORY.md).

Manifesto de movimentação, checks e resumos de execução ficam na base comum em
`artifacts/cleanup-2026-10-28/`, ignorados pelo Git. Resumos não substituem logs brutos.

## Auditoria da campanha 0016 — 29/10/2026

[Resultado e conformidade do ciclo 1](docs/results/CAMPAIGN_0016.md). CRC íntegro; 820 hashes em 20 snapshots e 41 arquivos contra Git conferidos; 60 avaliações completas reconstruídas com erro máximo 1,68583e−8. Binários de dados reconstruídos do corpus incluído reproduzem os hashes da 0015.

No checkout experimental, `python -m pytest -q tests/test_research_controls.py tests/test_document_data.py`: **28 passed**, CPU. Não houve nova execução CUDA nem treino científico. Logs do Kaggle registram 212 testes CPU e 17 CUDA FP32 aprovados; são resultados recebidos, distintos da checagem local.

Decisão: contexto sem RMS no ciclo 2. O runner recebido ainda não implementa teto global de sessão de 4h; perfis foram posteriores e treinos concorrentes. A implementação completa dos próximos ciclos continua pendente.

## Preparação do ciclo 2 — 29/10/2026

Implementação em `experiment/fractal-lrm`: matriz comunicação × FiLM, controles Transformer/LRM, seeds 81/82/83, contexto sem RMS, T256/B32/K2/2000 passos, FP32/Inductor. Novo executor serial com piloto prévio por variante/GPU, verificação da inicialização compartilhada, origem de dados, orçamento global de 4h e reserva de 300 s. Pacote fonte + bootstrap verificado por SHA-256 dispensam publicar commit.

Validações locais no checkout experimental (Python 3.12.2, Torch 2.14.1+cpu):

| Comando/check | Resultado real |
|---|---|
| `python -m pytest -q` | **237 passed, 0 failed, 25 skipped** |
| `python -m pytest -q tests/test_cycle2.py tests/test_cycle2_bootstrap.py tests/test_cycle2_release.py -m "not cuda"` após finalizar empacotador/mapeamento de GPU | **27 passed**, 4 testes CUDA fora da seleção |
| Planejamento do CLI sem `--execute` | 18 jobs, configurações/contagens verificadas, sem criar execução |
| Simulação ponta a ponta CPU | **complete**, seis braços, um bloco de seed, seis pilotos, 12 avaliações completas, relatório fatorial e ZIP CRC íntegro |
| Exclusões do ZIP de resultados | Sem pesos/estado de otimizador/binários de datasets |

Os testes abrangem causalidade, backward finito, memorização, retomada estrita com dropout/acumulação nos quatro braços, compilação AOT-eager CPU mantendo saídas dos ramos grossos, isolamento entre ramos, inicialização e contagens, watchdog/interrupção, exportação exclusiva, rejeição de ZIP inseguro/fonte alterada e notebook padrão inerte. O pequeno ajuste final de metadados de GPU foi coberto pelo segundo comando; não altera cálculo do modelo.

A simulação usou explicitamente `--engineering-smoke --skip-tests`: quatro passos, modelo pequeno, corpus sintético e CPU. A suíte foi executada separadamente. Esses resultados verificam o fluxo, não qualidade/eficiência científica. Os 25 skips são CUDA, inclusive quatro novos testes FP32/Inductor do ciclo 2; nenhum foi relatado como aprovado. A validação T4/Inductor do novo pacote ocorrerá no script entregue, antes da campanha real.

Logs, relatório da simulação e resumos ficam em `artifacts/cycle2-preparation-20261029/` no checkout experimental. O pacote novo contém arquivos de fonte e documentação, incluindo o presente registro. Sem commit/push ou execução científica nesta preparação. [Instruções do ciclo 2](docs/CYCLE2_RUN.md).

## Triagem das ideias externas — 29/10/2026

[Parecer sobre os 17 itens do catálogo do Argon](docs/REVIEW_TECHNOLOGIES_ARGON.md),
com fontes primárias e critérios de experimentação. Alterações desta revisão apenas
documentais; nenhum teste de modelo, benchmark ou treino novo. Checks reais: 28 links
locais válidos nos três documentos revisados em cada checkout; cópias do parecer
idênticas; `git diff --check` aprovado em ambos. SHA-256 do ZIP fonte e do launcher
do ciclo 2 permanecem iguais aos entregues. A revisão nova fica fora desse snapshot
imutável. Nenhum mecanismo adicional implementado e nenhum commit/push realizado.

## Célula curta por Git e teto de duas horas — 29/10/2026

A pedido do usuário, a entrega vigente usa commit completo do GitHub e dispensa
Input ZIP. Padrão de 7.200 s no bootstrap, CLI e orçamento da sessão; download e
checkout contam no relógio, com reserva de 300 s. Mantidos seis braços, seeds
81/82/83, 2.000 passos, scheduler e hiperparâmetros. Arquivos antigos em `dist/`
foram preservados; o launcher curto é gerado em novo diretório.

Validações reais em Python 3.12.2/Torch 2.14.1+cpu:

- `python -m pytest -q tests/test_cycle2_git.py tests/test_cycle2_bootstrap.py tests/test_cycle2_release.py`: **22 passed**.
- `rtk proxy python -m pytest -q -o faulthandler_timeout=90`: **250 passed, 25 skipped**, 298,10 s. A primeira chamada sem saída de progresso foi interrompida; não conta como aprovada. A repetição apenas acrescentou diagnóstico de timeout/saída direta, sem excluir testes.
- Clone Git local real: checkout da revisão anterior ao HEAD da fixture, conteúdo e hashes conferidos. Célula ativa baixa o helper pelo mesmo SHA; notebook desativado não faz ações externas.
- Setup simulado propaga o mesmo teto de 7.200 s e o tempo já gasto ao executor. Planejamento do CLI: 18 jobs, seeds 81/82/83 e 2.000 passos, sem execução.
- 47 links locais válidos nos guias/planos conferidos em cada checkout; `git diff --check` aprovado. ZIP e launcher anteriores conservam seus hashes.

Os 25 skips são de CUDA; não são suporte GPU verificado nesta máquina. Nenhum
treino científico novo foi executado. O bootstrap fará os testes T4/Inductor antes
da campanha. A entrega Git é publicada somente na branch experimental; documentação
espelhada na base comum não acrescenta código FLRM a essa branch.

## Auditoria da campanha 0017 — 29/10/2026

[Relatório do ciclo 2](docs/results/CAMPAIGN_0017.md). ZIP de 17.388.696 bytes,
743 entradas, CRC íntegro e sem duplicatas; SHA-256
`8883f8213cb6eea11bd0890bd13dfef23c93310db5ee9b01793c3d41357c852f`.
Commit executado `60219eafbceea09066dd05206adaa9f4ce865f96`, limpo no registro.

Verificações locais desta auditoria: 178 hashes do bootstrap contra todo o commit;
46 arquivos de fonte contra Git; 1.380 hashes em 30 snapshots; 72 avaliações completas
reconstruídas de 437 janelas/111.872 alvos cada, erro máximo 1,92880818e−8; metadados
dos checkpoints/configs; inicialização pareada pelos registros; enumeração em meta
e fórmula de parâmetros dos seis modelos; dataset/manifesto idênticos à 0016 e
binários reconstruídos do corpus incluído. Script independente terminou sem erros.

Logs recebidos do Kaggle: **250 testes CPU aprovados**, 25 fora da seleção, 384,17 s;
**21 CUDA FP32 aprovados**, 254 fora da seleção, 153,66 s. Incluem os quatro braços
compilados do ciclo 2. São resultados no hardware real recebido, não testes CUDA
executados nesta máquina local. Treinos/pilotos em duas T4, serial/alternado;
protocolo FP32/Inductor, 18 treinos completos, 2.000 passos e três blocos de seed.

Tempo global registrado de 5.562,96 s antes de relatório/exportação, dentro do teto
de 7.200 s. Fonte, configs e pesos existentes preservados; nenhum treino, alteração
de código científico ou commit/push nesta auditoria. Os pesos/AdamW não estão no ZIP:
os hashes de pesos foram comparados entre registros, sem rehash dos arquivos ausentes.
Estatísticas exploratórias, três seeds no mesmo corpus, sem custo igualado. FiLM
fica opcional, comunicação é candidata à confirmação; superioridade não demonstrada.

Auditoria/script, dados intermediários e figura em
`artifacts/audits/campaign-0017-2026-10-29/` no checkout comum. Checks documentais:
links locais, cópias dos cinco artefatos curados e `git diff --check` conferidos.

## Preparação do ciclo 3 — 29/10/2026

Plano registrado em [docs/PLAN_CYCLE3.md](docs/PLAN_CYCLE3.md); operação em
[docs/CYCLE3_RUN.md](docs/CYCLE3_RUN.md). Sem mecanismos novos nem mudança de
backbone/loss/otimizador. Treinos e comparação científica na T4 continuam pendentes.

Verificações locais reais em Python 3.12.2 / Torch 2.14.1+cpu / PyArrow 23.0.0:

- Download dos quatro shards WikiText fixados, SHA-256 integral conferido; preparo
  real de 1.758/60/62 artigos e hashes tokenizados congelados. Detectada uma cópia
  treino/teste, excluída explicitamente só do treino antes da seleção. As outras
  442 duplicatas internas ao treino são deduplicadas. Nada foi avaliado no teste.
  [Proveniência curta](docs/results/CYCLE3_PREPARATION.json).
- Testes direcionados de ciclo 3, bootstrap anterior e dados: **49 passed, 4 skipped**.
  Testes adicionais de relógio de quatro horas e validação antes de download:
  **6 passed**. Os quatro skips são CUDA indisponível localmente.
- Suíte completa `python -m pytest -q -o faulthandler_timeout=90`: **274 passed,
  29 skipped**. Os 29 skips são de CUDA; não equivalem a testes GPU aprovados.
- Causalidade, forward/backward finitos, memorização e retomada da próxima atualização
  com dropout e acumulação CPU FP32/SDPA math, nas quatro famílias selecionadas,
  incluindo Transformer de dois blocos. Equivalência exata entre avaliação agrupada
  e CLI; publicação atômica do manifesto sob falha; rejeição de sobreposição desconhecida,
  conflitos de agregação, seeds duplicadas e falsa confirmação com dados incompletos.
- Simulação ponta a ponta CPU: **seis treinos pequenos / 24 avaliações completas**,
  quatro perfis, piloto de duração de validação, relatório/CSV/curvas e dois ZIPs.
  Status `complete`, `engineering_smoke=true`, `confirmation_eligible=false`,
  `test_evaluated=false`. CRC dos dois ZIPs e hashes de todos os seis checkpoints
  preservados no backup conferidos. Não é benchmark nem resultado científico.
  Evidência ignorada em `artifacts/cycle3-engineering-20261029/verification.json`.
- 57 links locais válidos nos guias/planos em cada checkout; `git diff --check` limpo.
  Dados, ZIPs e releases antigos preservados; fonte experimental fica fora da base comum.

O pareamento usa FLOPs estimados, com exclusões explícitas; não equivale a tempo
ou número de parâmetros igualados. Warmup inclui compilação/aquecimento; tempo
exclusivo de compilação e contador de recompilações não foram instrumentados.
O launcher executará os testes T4/Inductor e piloto no hardware real antes do treino.

## Correção de serialização Windows/Linux no ciclo 3 — 29/10/2026

A tentativa Kaggle da revisão `855c8d2` falhou após preparar o dataset e antes de
testes/treinos, no contrato de hashes. Causa reproduzida: `tokenizer.json` de 43 bytes
com CRLF no Windows tinha SHA-256 `a9be2688d42639c5915aa9222fcd9a389acbe6483c47ce170f3b0405e7644667`;
o JSON equivalente de 39 bytes com LF em Linux tem SHA-256
`f88f84a387d00bb8027db376d50fff07c86c4ee137f1678351521fefe0f8d252`.
O freeze local inicial foi incorreto para o Kaggle. Não era erro da configuração
usada pelo usuário nem um resultado de treinamento.

Correção: `write_json` agora grava UTF-8/LF explícito; protocolo exige o hash canônico,
mantendo os três hashes binários e demais contratos. Uma falha informa campo,
valor esperado e observado. Datasets existentes não foram migrados/sobrescritos;
retomada continua exigindo a implementação e o ambiente originais.

Verificações locais reais:

- Antes da correção, testes de portabilidade reproduziram **2 falhas e 1 aprovação**;
  depois, primeira rodada direcionada de JSON/dados/ciclo 3: **43 passed**, CUDA
  fora dessa seleção. Novos testes simulam defaults LF/CRLF independentemente do host.
- Suíte completa `python -m pytest -q -o faulthandler_timeout=90`: **280 passed,
  29 skipped**, exit code 0. O diagnóstico de 90 s imprimiu uma pilha durante um teste
  demorado; a suíte prosseguiu até aprovação. Skips são CUDA indisponível localmente.
- Corpus real preparado novamente em `artifacts/cycle3-portability-20261029/`,
  sem sobrescrever a preparação anterior. **Todos os hashes do protocolo passaram**;
  os três `.bin` foram comparados byte a byte com os anteriores e são idênticos.
  Tokens: 33.570.713 treino, 1.144.308 validação, 1.287.718 teste. Sem avaliar modelos.
- Leitura de dataset legado CRLF preserva bytes e fingerprint; adulteração continua
  rejeitada. Contrato do ciclo 3 rejeita tokenizer CRLF mesmo com semântica equivalente
  e identifica exatamente `files.tokenizer.json` no diagnóstico.
- Entrega de recuperação usa número 19, célula/notebook consistentes e pasta nova;
  a tentativa 0018, datasets, cache e releases anteriores são preservados. A duração
  máxima continua 14.400 s por execução. 45 links locais conferidos em cada checkout.

Plano/risco anterior às alterações: [PLAN_CYCLE3_PORTABILITY_FIX.md](docs/PLAN_CYCLE3_PORTABILITY_FIX.md).
CUDA não está disponível localmente; a correção de serialização não constitui
validação de GPU. O launcher mantém os testes T4/Inductor antes do treino.

## Auditoria 0019 e operação em lotes de uma hora — 29/10/2026

[Relatório](docs/results/CAMPAIGN_0019.md) e [plano anterior às mudanças](docs/PLAN_CYCLE3_SHORT_SESSIONS.md).
ZIP recebido de SHA-256 `e609269cf31f8d55d6afbbda9ef56b9b4c799fbd28f7921fd1f16479aa14887f`,
14.002.802 bytes, 442 entradas, CRC íntegro e sem nomes duplicados. **12 treinos
completos / 48 avaliações**, seeds 101/102; parada registrada por previsão de orçamento
aos 9.645,32 s. Não há erro de rede registrado como causa da parada nesse arquivo.

Conferidos 197 hashes contra o commit científico, 20 snapshots com 1.880 hashes,
configurações/contadores/FLOPs estimados, enumeração/fórmula de parâmetros e hashes
de checkpoint entre registros. NLL reconstruída de 4.469 janelas / 1.144.064 alvos
por avaliação; erro máximo 4,37404424e−9. Pesos não estão no ZIP recebido: isso não
constitui rehash dos tensores ausentes. Logs reais do Kaggle: **280 CPU / 25 CUDA
FP32 aprovados**. Resultado continua inconclusivo com duas seeds e teste reservado.

O novo operador apenas organiza os lotes: workers congelados exatamente em
`8798f6b84714d2c6671d970685984270078a982c`, com hashes científicos originais. Operador
e snapshot têm proveniência separada. Sessões de até 3.600 s, até dois treinos novos,
validação remota focada declarada e checkpoints/avaliações completos preservados.
Não há alegação de aceleração de kernel ou redução automática do custo de cinco seeds.

Verificações locais reais:

- Testes do operador/filas/agregação/deadline/exports/bootstrap: **16 passed**.
  Cobrem deduplicação de cópias, rejeição de conflitos de pesos/dataset/código/configs,
  seleção do par primário 103 sem repetir os 12 treinos e propagação de um único relógio.
- Suíte completa local `python -m pytest -q -o faulthandler_timeout=90`: **296 passed,
  29 skipped**, exit code 0. Skips são CUDA indisponível. Houve impressão diagnóstica
  de pilha aos 90 s durante um teste demorado; ele terminou e a suíte foi aprovada.
- Simulação ponta a ponta CPU com **checkout real do worker congelado**, dois treinos
  pequenos / oito avaliações completas. Status completo de lote, engenharia explícita,
  sem confirmação científica. Os hashes de todas as fontes e configs científicas nos
  resultados são idênticos aos da 0019. Três pares de ZIPs (após 1/2 treinos e final)
  tiveram CRC e hashes dos checkpoints conferidos, sem sobrescrita.
  Evidência: `artifacts/cycle3-short-engineering-20261029/verification.json`.
- Heartbeat operacional, perfis limitados aos modelos/slots selecionados, fila
  cumulativa e contrastes por pares preservam o requisito original de cinco blocos
  completos para a decisão confirmatória.
- 64 links locais conferidos em cada checkout, `git diff --check` sem erros; o ZIP
  original foi rehashado e preservado. Base comum recebe documentação, sem fonte FLRM.

A primeira chamada da suíte completa nesta alteração foi interrompida para ajustar
o ambiente do novo bootstrap antes da checagem de compatibilidade; não conta como
aprovada. `CUBLAS_WORKSPACE_CONFIG` segue a política do Trainer antes da comparação;
o checkout congelado exige bytes LF. CUDA do operador novo permanece pendente da
execução T4 real, ainda que o código científico já tenha sido validado na 0019.

## Descoberta de ZIPs inválidos — 30/10/2026

O bootstrap `42d9179` propagava `BadZipFile` ao encontrar um candidato ilegível na
busca automática de histórico. O traceback recebido não identifica o caminho do ZIP;
não foi atribuído o problema a um arquivo específico sem evidência. Reprodução
com a API original confirmou a falha antes da correção.

Correção limitada ao operador/entrega: caminhos e motivos são registrados;
histórico legado ilegível pode ser contornado pela auditoria até 0019. Histórico
recente ilegível só é recuperado quando outro snapshot da mesma tentativa comprova
todo o lote e seus hashes de checkpoints; caso contrário, há erro explícito antes
de treinar. Descoberta precede instalação/reserva de número. Nenhum candidato é
apagado/modificado e a verificação científica posterior permanece estrita.

Verificações reais:

- `python -m pytest -q tests/test_history_discovery.py tests/test_cycle3_short.py tests/test_cycle2_bootstrap.py tests/test_cycle2_git.py tests/test_cycle3.py -k 'not cuda'`:
  **77 passed**. Seleção CPU, sem alegação de CUDA aprovado.
- Cobertura: ZIP vazio, texto, truncado, erro CRC, JSON/metadata inválidos; mistura
  com históricos válidos; outro protocolo; snapshot completo versus parcial;
  histórico cumulativo sem linhas do lote; números de tentativa e artefatos exclusivos.
- Ensaio com cópia do ZIP real 0019 e um candidato 0018 inválido: os 12 resultados
  foram preservados/deduplicados e a fila continuou no par primário 103. Candidato
  inválido e ZIP original intactos. Registro em
  `artifacts/history-discovery-fix-20261030/verification.json`.
- 29 links locais conferidos em cada checkout e `git diff --check` limpo. Entrega
  corrigida usa 0021 porque 0020 pode ter sido reservado pela tentativa anterior.

Não houve mudança de modelo, treino, avaliação ou do worker congelado `8798f6b`.
A suíte completa de 296 testes registrada acima não foi repetida para esta correção
de descoberta; foram executadas as regressões relevantes indicadas. Plano prévio:
[PLAN_HISTORY_DISCOVERY_FIX.md](docs/PLAN_HISTORY_DISCOVERY_FIX.md).

## Snapshot 0021 e manutenção do working — 30/10/2026

[Auditoria](docs/results/CAMPAIGN_0021_PARTIAL.md) e
[plano anterior à implementação](docs/PLAN_KAGGLE_STORAGE_RECOVERY.md).
Resultados after-01 SHA-256 `9043115ab066614e5a567f334a32d9be5ed75e5d90793536d9a52112f2bee223`;
checkpoint ZIP SHA-256 `4bd407d901ea4af8a11f3c12b346a5d308fc69b13c6e3c243540e1ffd6ca49b5`.
Ambos com CRC íntegro, sem entradas duplicadas; pesos e estado opaco conferidos por
hash, pickle não carregado. FLRM 103 concluiu 8.000 passos, NLL 1,320334273 e quatro
avaliações completas. Erro máximo de reconstrução de NLL 4,05051504e−9.

Header safetensors: 2.165.772 elementos de parâmetros únicos mais buffer inteiro
`training_step=8000`. Alias do embedding compartilhado preservado e contado uma vez.
Snapshot do operador comparado integralmente ao Git usando LF explícito. Identidade
científica igual à 0019. O controle estava pendente no snapshot; estado atual remoto
não foi recebido. Histórico agregado tem 13 treinos completos.

Logs reais T4 recebidos: **12 CPU aprovados em 16,39 s**, **2 CUDA FP32 aprovados em
56,95 s** na seleção focada; fases externas totalizaram 88,04 s. O primeiro treino
e exportação incremental aconteceram de fato, apesar da ausência de saída visível.
Isso não comprova o término do segundo job ou do lote.

Nova manutenção usa somente Python padrão: inventário, detecção de workers, proteção
de dados/checkpoints, remoção de checkouts Git limpos reproduzíveis e próximo número
livre. Não foi executada no Kaggle nesta conversa; tamanho real e ganho de espaço
dependem da execução remota. O script não encerra processos nem apaga campanhas.

Verificações locais reais:

- `python -m pytest -q tests/test_kaggle_maintenance.py tests/test_history_discovery.py tests/test_cycle3_short.py tests/test_cycle2_bootstrap.py tests/test_cycle2_git.py tests/test_cycle3.py -k 'not cuda'`: **91 passed**. Não é a suíte completa nem teste CUDA local.
- Remoção real de cache Git de fixture, incluindo readonly do Windows, com alvos
  resolvidos e limitados à pasta temporária. Dados, ZIPs, checkpoints, reservas e
  metadados preservados. Cache alterado, com payload ou remoto diferente é recusado.
- Worker ativo bloqueia limpeza; checkpoint parcial/conclusão não exportada bloqueia
  nova rodada. A entrega exige histórico do FLRM 103 verificado para não repeti-lo.
- Célula guardada usa número livre calculado e até um job; início/baixamento têm
  mensagens imediatas. Notebook desativado é inerte. Relógio inclui a manutenção.
- Simulação com o snapshot real reconhece a conclusão do FLRM e número 22; adicionar
  um checkpoint parcial fictício do controle bloqueou reinício. Evidência em
  `artifacts/kaggle-recovery-maintenance-20261030/verification.json`.
- 30 links locais conferidos em cada checkout e `git diff --check` limpo. Os ZIPs
  originais foram preservados. Sem alteração do worker congelado `8798f6b`.

## Caches antigos mostrados no Kaggle — 30/10/2026

O launcher cria checkouts `llm2-<revision[:12]>`. A manutenção passou a reconhecer
esse padrão exato, exigindo HEAD correspondente ao nome e todas as proteções de
Git/conteúdo/caminhos anteriores. Workers executados por esses caminhos também
bloqueiam a limpeza. Pastas `llm2-campaign-*`, incluindo nomes hexadecimais, continuam
protegidas. As imagens não permitem medir o espaço recuperável.

Verificações locais reais:

- `python -m pytest -q tests/test_kaggle_maintenance.py`: **27 passed**.
- `python -m pytest -q tests/test_kaggle_maintenance.py tests/test_history_discovery.py tests/test_cycle3_short.py tests/test_cycle2_bootstrap.py tests/test_cycle2_git.py tests/test_cycle3.py -k 'not cuda'`: **104 passed**.
- Remoção real de checkout legado limpo em fixture temporária; campanhas dos dois
  padrões, ZIP e diretórios de metadados preservados. Alterações, arquivos extras,
  payload ignorado/rastreado, remoto diferente e HEAD divergente recusados.
- Worktree legado com common-dir externo recusado; argumentos de workers antigos
  reconhecidos sem confundir nomes de campanhas. Notebook desativado e guarda da
  continuação permanecem cobertos pelas regressões.

Esses checks são CPU/arquivos locais, não a suíte completa nem execução CUDA ou
limpeza no Kaggle. Nenhuma alteração de treino, dataset, horizontes ou worker.

## Entrega 0022 — par primário seed 104, 30/10/2026

A saída de manutenção recebida informa ambos os jobs da 0021 completos/exportados,
sem workers ativos ou avisos. Os ZIPs finais ainda não foram auditados localmente.
Entrega nova usa o operador já publicado `0066131`, worker congelado `8798f6b`,
seed 104, até dois jobs e teto de 3.600 s a partir da nova célula. Não altera fontes
do operador, treino, horizontes ou protocolo. Preservar o histórico mais recente da
0021 no working/Input para agregação automática.

Verificação da entrega em `dist/cycle3-seed104-1h-20261030/verification.json`:
hashes do manifesto conferidos; bootstrap remoto igual ao código publicado; célula
ativa exercitada com bootstrap simulado, confirmando parâmetros e relógio novo;
notebook desativado sem rede. A seleção retorna `anchor_flrm-seed-104` e
`flops_transformer-seed-104`; jobs concluídos são pulados. Não houve treino local,
novo teste CUDA ou repetição da suíte: somente configuração e verificação da entrega.

## Auditoria dos resultados 0022 — 30/10/2026

[Relatório](docs/results/CAMPAIGN_0022.md) e evidência estruturada/curvas/pares
versionados em `docs/results/campaign_0022_*`. Plano anterior e auditor local em
`artifacts/audits/campaign-0022-2026-10-30/`. ZIP original SHA-256
`460b16daec5895deb87633ed1cd1d4eab40c3e5f64f9c23f22a1f9892a7346ee`,
4.908.174 bytes, 138 entradas únicas, CRC íntegro; preservado.

Verificações reais: 197 hashes do worker `8798f6b` e 213 do operador `0066131`
contra Git; snapshot integral do operador e quatro snapshots científicos (376
membros). Configurações, manifesto/tokenizer, identidade, runtime agregado, passos,
contadores, fórmulas/enumeração de parâmetros e hashes de pesos entre registros
consistentes. Oito NLLs reconstruídas de 4.469 janelas / 1.144.064 alvos cada,
erro máximo 4,03674250e−9. Métricas registradas finitas. Histórico antigo conferido
contra a evidência curada 0019 e o snapshot 0021 disponível; quatro pares recalculados.

Logs reais do Kaggle: **12 CPU aprovados em 17,16 s** e **2 CUDA FP32 em 59,24 s**,
seleção focada. Auditoria local executada com sucesso após corrigir a leitura da
chave do tokenizer no próprio auditor (`files/tokenizer.json`). Não houve alteração
do dataset nem da validação científica para aceitar o arquivo. Sem treino local,
nova execução pytest ou alegação de CUDA local. Figura científica inspecionada.

Sessão completa aos 1.925,28 s antes da exportação final; 16 treinos cumulativos,
quatro pares primários, somente dois blocos de seis braços. NLL 104 FLRM 1,318235045
e Transformer 1,312205419. Diferença média n=4 +0,001844508, IC95% exploratório
[−0,006697720; +0,010386737]. Com duas vitórias em quatro pares, o requisito positivo
pré-fixado de quatro vitórias em cinco já não pode ser satisfeito. Recomendação de
revisão antes de mais treinos; protocolo e resultado inconclusivo preservados.

Pesos/estado de treino não recebidos neste ZIP e não rehashados/carregados. Os logs
brutos do Transformer 103 também não estão neste arquivo; sua linha herdada foi
conferida na agregação, sem reconstrução local daquela NLL. Teste reservado intacto.
Compilação e aquecimento continuam sem medição isolada. Nenhuma fonte de treino mudou.
Conferidos 86 links locais em cada checkout; artefatos do relatório espelhados,
ZIP original rehashado e preservado, `git diff --check` limpo.

---

## Campanha 008 — Colab NVIDIA L4 (`BF16` + Estado `FP32`, `fractal_golden_classic_big` vs `fractal_golden_big`, 15/11/2026)

- **Ambiente real verificado via Colab MCP (`https://colaboratory.googleapis.com/mcp`):** `1× NVIDIA L4` (`22,03 GiB`), Python `3.13.16`, PyTorch `2.14.1+cu130`, CUDA `13.0`, cuDNN `92800`, commit `3b178e0c10fe4f1c5508f532114942be2c22c620` (`llm-ideas`).
- **Testes reais executados:**
  - Local CPU: `python -m pytest -q tests/test_fractal_delta.py` -> **`29 passed in 54.59s`** (`0` pulados).
  - Colab GPU L4: `python -m pytest -q tests/test_fractal_delta.py` -> **`29 passed in 56.59s`** (`0` pulados); `python -m pytest -q -m cuda` -> **`55 passed in 131.83s`** (`0` pulados).
- **Resultados medidos (`--mode unified --precision bf16 --story-seq-len 512 --binding-hard-writes 64 --lr 0.003 --seeds 351`):**
  - `fractal_golden_big:s351` (`864.869` params, `422.920` B/seq, `635,6s`, `1,72 GB` VRAM):
    - TinyStories: `val@256 = 0.7895`, `val@512 = 0.7622`, `val@3219 = 0.7465`.
    - Bandas: `1–256 = 0.6813`, `257–512 = 0.7628`, `513–2048 = 0.7796`, `2049+ = 1.0276`.
    - Intervenção de prefixo: `Δ Trunc = -0.0021`, `Δ Swap = +0.0082`.
    - Decode Inductor (CUDA Graph): `2.286,2 B/s` (`parity_max_abs_err = 1,24e-5`).
    - Byte-Binding-v2: `@256 = 99,8% (99,8% paired)`, `@1024 = 99,8% (99,8% paired)`, `@4096 = 83,5% (72,4% paired)`, `@8192 (w=32) = 60,9% (36,8% paired)`, `@8192 (w=64) = 56,0% (34,0% paired)`.
  - `fractal_golden_classic_big:s351` (`864.869` params, `422.920` B/seq, `642,6s`, `1,73 GB` VRAM, `fractal_leaky_budget=True`, `fractal_assoc_delta=True`):
    - TinyStories: `val@256 = 0.7913`, `val@512 = 0.7646`, **`val@3219 = 0.7420`**.
    - Bandas: `1–256 = 0.6868`, `257–512 = 0.7640`, **`513–2048 = 0.7694`**, **`2049+ = 0.7261`** (`-0,3015` NLL vs baseline).
    - Intervenção de prefixo: **`Δ Trunc = +0.0053`** (positivo), `Δ Swap = +0.0079`.
    - Decode Inductor (CUDA Graph): `2.179,5 B/s` (`parity_max_abs_err = 2,86e-5`).
    - Byte-Binding-v2: `@256 = 99,8% (99,7% paired)`, **`@1024 = 100,0% (100,0% paired)`**, **`@4096 = 99,4% (98,2% paired)`**, **`@8192 (w=32) = 95,1% (90,4% paired)`**, **`@8192 (w=64) = 78,4% (67,9% paired)`**.
- **Limitações registradas:** comparação em `BF16` na L4 realizada até aqui com 1 semente pareada (`seed=351`); leve trade-off de `+0,0018` em `val@256` em troca da eliminação da deriva em `2049+` (`1.0276 -> 0.7261`) e do salto de `+53,6` pp em `paired_accuracy` em `@8192 (w=32)`. Runtime L4 encerrado (`delete_runtime` + `list_runtimes == {}`) imediatamente após a coleta. Relatório completo em [IA2_CAMPAIGN_008_COLAB_L4.md](docs/results/IA2_CAMPAIGN_008_COLAB_L4.md).

### Campanha 009 — Fechamento do Passo 2 (`fractal_golden_classic_v2_big` + `--unified-curriculum`, Colab NVIDIA L4)
- **Alterações validadas (`e1d4cd2` → `bd7bcb9`):**
  - `fractal_assoc_val_dim` configurável (`ResearchConfig`, `FractalDeltaMemory`), novo braço `fractal_golden_classic_v2_big` (`assoc_val_dim=48`, `877,157` params, `432,136` B/seq), currículo progressivo `--unified-curriculum` (`512×16 → 1024×8 @ 3000 → 2048×4 @ 4500`, `writes=48`, `8,192` tok/step), e especialização `torch.compile(..., dynamic=False)` em `Trainer` para eliminar falha de autotune Triton simbólico no PyTorch `2.14.1+cu130` durante transições de shape (`CUDA_ERROR_ILLEGAL_ADDRESS` observada no passo `3001` antes do fix).
- **Testes unitários e CUDA executados:**
  - Local CPU (`tests/test_fractal_delta.py` em `e1d4cd2` e `bd7bcb9`): `exit 0` (`30 passed` na execução `e1d4cd2` e reconfirmado após fix em `bd7bcb9`).
  - Local CPU (`tests/test_training.py` em `bd7bcb9`): `13 passed, 1 skipped` (1 teste CUDA pulado em ambiente sem GPU local).
  - Colab `NVIDIA L4` (`pytest -q -m cuda` em `bd7bcb9`): **`55 passed, 765 deselected`** em `695.64s` (`0` falhas, `0` pulados dentro do recorte CUDA).
- **Campanha pareada 3 seeds na L4 (`bf16`, `--mode unified`, `--unified-curriculum`, `--lr 0.003`, seeds `351, 352, 353`, `2354.5s`):**
  - `fractal_golden_classic_v2_big`: TinyStories `val@256 = 0.9447 ± 0.0041`, `val@512 = 0.9122 ± 0.0041`, `val@3219 = 0.8823 ± 0.0043`, banda `2049+ = 0.7221 ± 0.0134`, `ΔTrunc = +0.0110 ± 0.0003`, `ΔSwap = +0.0120 ± 0.0003`, `decode_inductor = 2,215.2 ± 27.3 B/s`; Byte-Binding-v2 `8192 w=32 = 99.7% ± 0.3%` (`99.5% ± 0.4% paired`), **`HARD 8192 w=64 = 95.1% ± 1.2%` (`90.9% ± 1.4% paired`)**, peak VRAM `1.73 GB`.
  - Ver relatório completo em [docs/results/IA2_CAMPAIGN_009_COLAB_L4.md](docs/results/IA2_CAMPAIGN_009_COLAB_L4.md).
