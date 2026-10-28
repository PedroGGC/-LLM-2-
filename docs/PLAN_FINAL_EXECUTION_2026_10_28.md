# Plano Técnico — Primeira Entrega (Etapas 1 e 2 de `PLANO_FINAL_EXECUCAO_FLRM_2026-10-28.md`)

Este documento registra no controle de versão o plano, os riscos e os critérios de aceitação para a execução das **Etapas 1 e 2** do consenso técnico de 28/10/2026.

## 1. Escopo da Primeira Entrega

1. **Correções obrigatórias de documentação (`VALIDATION.md`):**
   - Registrar nota de revisão datada (`28/10/2026`) esclarecendo:
     - O FLRM avaliado utiliza **1 único `DecoderBlock` compartilhado** (`research.core_layers=1`); `model.n_layers=6` nos YAMLs base não define a profundidade do backbone fractal ($K=2$ significa duas passagens na escala fina, não doze).
     - A Campanha 0008 reproduziu numericamente a loss da 0003 na seed 42 ($\Delta \approx 1,1 \times 10^{-8}$), o que não demonstra identidade bit a bit dos pesos finais nem superioridade geral da arquitetura.
     - Nas seeds 42/43/44 (`0009 + 0010`), as losses médias eager são: Transformer pequeno **1,396428302** ($s = 0,002709898$, $\text{SE} = 0,001564560$), FLRM sem NCM **1,397842610** ($s = 0,001172627$, $\text{SE} = 0,000677016$), FLRM com NCM-16 **1,397658978** ($s = 0,001092997$, $\text{SE} = 0,000631042$). A Campanha 0011 repete as seeds 43/44 e não acrescenta inicializações independentes.
     - Médias aritméticas de throughput de treino nas 3 seeds (`0009 + 0010`): Transformer **153.342 alvos/s**, FLRM legado **53.087 alvos/s**, FLRM com NCM-16 **52.315 alvos/s**.
     - O NCM-16 reduziu a NLL em média por **−0,000184 por byte** (**−0,0184% em perplexidade**) nas três seeds observadas, sendo candidato de ablação sem benefício geral garantido.
     - O Inductor acelerou o treino dos dois modelos na 0012, mas a proximidade entre razões de tempo e FLOPs não prova que o gargalo restante seja exclusivamente aritmético; o benchmark de geração da 0012 executou inferência eager sem KV cache.
     - Magnitudes de estado/contexto e gradientes devem ser medidas empiricamente (Etapa 1); normalizar contexto ou condicionar o núcleo por $k$ são hipóteses arquiteturais para a Etapa 4, não correções de bugs já demonstrados.
2. **Etapa 1 — Telemetria de Magnitudes e Gradientes (`src/llm2/research/diagnostics.py`, `src/llm2/cli.py`, `src/llm2/research/campaign.py`):**
   - Coletar por escala e iteração: RMS de `state`, `state_norm(state)`, contexto bruto/condicionado, `W_ctx(c)`, soma de entrada do `DecoderBlock`, saídas de `attn_norm` e `ffn_norm`, razões explícitas de magnitude e RMS/norma L2 dos gradientes nas ativações recorrentes retidas, além da norma acumulada dos gradientes dos parâmetros compartilhados.
   - Separar posições válidas ($T_2 = \lceil T/16 \rceil$) e padding ($T_1 - T_2$) de $s_2$ no modo `compressed_grouped`.
   - Registrar `null` com `null_reason` para saídas não utilizadas ou estados iniciais constantes sem gradiente.
   - Preservar integralmente pesos, `.grad` prévios, modos `.training`, RNGs, contadores e logits de referência.
3. **Etapa 2 — Eficiência (`src/llm2/config.py`, `src/llm2/architectures/latent_recursive.py`, `src/llm2/architectures/fractal_recursive.py`, `src/llm2/research/profiling.py`, `src/llm2/research/campaign.py`, `scripts/kaggle_notebook.py`):**
   - Flag explícito `research.cache_context_projection: bool = False` (reúso restrito a um `forward`, sem `.detach()`, sem alterar parâmetros e sem ativar `fused_film`).
   - Aliases explícitos sem JEPA e com $K$ fixo:
     - `G0`: `flrm_compressed_grouped_no_jepa_k2` (`compressed_grouped`, `cache=False`, `328,28 GFLOPs/passo`)
     - `N0`: `flrm_compressed_native_no_jepa_k2` (`compressed_native`, `cache=False`, `291,82 GFLOPs/passo`)
     - `G1`: `flrm_compressed_grouped_cached_no_jepa_k2` (`compressed_grouped`, `cache=True`, `317,41 GFLOPs/passo`)
     - `N1`: `flrm_compressed_native_cached_no_jepa_k2` (`compressed_native`, `cache=True`, `282,31 GFLOPs/passo`)
     - `lrm_no_jepa_k1` e `lrm_no_jepa_k2` (para a futura Etapa 3)
   - Perfilamento de `G0`, `N0`, `G1` e `T` (`baseline_small`) em `eager` e `inductor` (FP32) na mesma GPU física (GPU 0), com 10 warmups, 30 passos cronometrados sem profiler, 1 passo instrumentado e 3 repetições com ordem alternada entre repetições.

## 2. Riscos e Critérios de Aceitação

- Preservação estrita de causalidade, RoPE FP32, RMSNorm FP32, loss FP32, dropout zero em avaliação e retomada bit a bit em CPU FP32 dentro de cada variante.
- Suíte completa `python -m pytest -q` aprovada localmente antes da publicação do commit para execução no Kaggle.

## 3. Revisão Pós-Campanha 0013 e Plano da Segunda Entrega (Correções de Telemetria/Estimador + Etapa 3)

Após auditoria técnica dos resultados da Campanha 0013 (`llm2-campaign-0013-results.zip`, SHA256 `52fa6b8f41b5c7a5990658fb0c783435914260f82c2247381f7fafd82961c2a4`), ficam estabelecidos os seguintes ajustes e a especificação da **Etapa 3**:

### 3.1 Correções do Observador (`diagnostics.py`), Estimador (`fractal_recursive.py`) e Relatório
1. **Bug de atribuição de contexto na 1ª chamada em `G1` e `N1` (`src/llm2/research/diagnostics.py`):**
   - Quando `cache_context_projection=True`, as projeções de `c0, c1, c2` ocorrem antes do laço recorrente, deixando a projeção residual de `c2` em `call_state["ctx_proj_out"]`. Na 1ª chamada do núcleo ($s_0, k=1$), como `context_projection` não é invocada novamente, o hook pós-chamada recuperava a projeção residual de $s_2$ ($\text{RMS} = 0,002614$) em vez do argumento `ctx_arg` de $s_0$ ($\text{RMS} = 0,007842$).
   - **Correção:** registrar um `register_forward_pre_hook(..., with_kwargs=True)` no núcleo (`core_pre`) que limpa `call_state` no início de cada chamada e captura `context_is_projected`. Quando `context_is_projected=True`, selecionar explicitamente o tensor de argumento `ctx_arg` consumido por aquela chamada. Capturar separadamente os tensores de contexto projetados antes do laço (`preloop_cached_projections`).
2. **Escopo explícito dos gradientes de contexto (`src/llm2/research/diagnostics.py`):**
   - Adicionar campo `"scope"` em todos os dicionários de `_grad_summary` (`"per_call"`, `"accumulated_across_iterations"` ou `"accumulated_across_scales_and_iterations"`).
   - Em `N1` (todas as escalas) e `G1` ($s_0$), `ctx_arg` é o próprio tensor compartilhado entre iterações (`scope="accumulated_across_iterations"`). Em `G1` ($s_1, s_2$), `grouped_c` é concatenado a cada iteração (`scope="per_call"`), enquanto os tensores pré-laço `c1, c2` acumulam gradientes entre iterações. Registrar explicitamente `projected_context` (com seu `scope` real) e `cached_projected_context` (tensor pré-laço reutilizado com `scope="accumulated_across_iterations"` quando o cache está ativo).
3. **Correção do estimador analítico de GFLOPs para `G1` (`src/llm2/architectures/fractal_recursive.py`):**
   - Com `cache_context_projection=True`, `project_context` é executado antes do right-padding de $s_2$, operando sobre comprimentos `lengths = (s, s1, s2)` (336 posições por sequência em $T=256$), e não `full_core_lengths = (s, s1, s1)` (384 posições).
   - Em $B=32, T=256, K=2$, o custo analítico estimado de `G1` passa de `317,410247 GFLOPs` para **316,051292 GFLOPs** (−1,358954 GFLOPs/passo, −3,7256% vs `G0`). Os valores de `G0` (`328,281883`), `N0` (`291,816604`) e `N1` (`282,303922`) permanecem inalterados.
4. **Correções de metadados e interpretações científicas (`VALIDATION.md` e relatório da 0013):**
   - Ambiente real da 0013: **Python 3.13.15, PyTorch 2.11.0+cu128, CUDA 12.8, cuDNN 91900, Tesla T4**.
   - Escopo da telemetria: lote `B=1`, primeira janela de validação (`validation_first_window_b1`, 256 alvos, dropout desligado), apenas na inicialização (passo 0). A confirmação pós-treino (passos 500 e 2.000) pertence à execução da Etapa 3.
   - Qualificação dos gradientes e magnitudes: a mudança do gradiente L2 de `core_input_sum` entre $k=1$ (`4,9138`) e $k=2$ (`0,02258`) acompanha a mudança de escala da entrada (`0,007842` $\to$ `0,993286`) sob a geometria da `RMSNorm` (cujo produto $\text{RMS}(\text{entrada}) \times \text{L2}(\text{gradiente})$ varia apenas $\approx 1,72\times$: `0,038535` vs `0,022426`), não demonstrando colapso funcional de $217\times$ nem indiferença aos tokens. Razões de RMS não constituem decomposição aditiva de variância.
   - Linha do Transformer na matriz da Etapa 3: `baseline_small` tem **1 camada (`model.n_layers=1`) e `H=1280`** (**2.164.224** parâmetros), e não `L=6`.

### 3.2 Etapa 3 — Protocolo de Inicialização Comum Pareada (`paired_common_init`) e Matriz $6 \times 3$
1. **Protocolo `research.paired_common_init: bool = False` (`src/llm2/config.py`, `src/llm2/research/pairing.py`, `src/llm2/research/factory.py`, `src/llm2/trainer.py`, `src/llm2/research/campaign.py`):**
   - Quando `paired_common_init=True` em modelos `recursive` ou `fractal` com pesos compartilhados:
     - Constrói uma referência canônica `LatentRecursiveModel` a partir do estado de RNG pós-`setup_runtime(cfg.train)` e captura o estado de RNG imediatamente posterior (`post_ref_rng`).
     - Para `FractalLatentModel`, inicializa os módulos exclusivos (`block.conditioning`, `block.interaction`, `block.comp4`, `block.comp16`, `block.ncm`) em escopo determinístico isolado (`fork_rng` semeado por `cfg.train.seed + 10007`) e copia explicitamente (`copy_` sem compartilhar `data_ptr`) todos os parâmetros e buffers compartilháveis da referência (`embedding.weight`, `norm.weight` e a totalidade de `core.*` $\to$ `block.cores.0.*`: `state_norm.weight`, `context_projection.weight`, `blocks.*.attn_norm.weight`, `blocks.*.attn.qkv.weight`, `blocks.*.attn.proj.weight`, buffers RoPE `cos/sin`, `blocks.*.ffn_norm.weight`, `blocks.*.ffn.gate.weight`, `blocks.*.ffn.up.weight`, `blocks.*.ffn.down.weight` e `gate`), preservando `lm_head.weight is embedding.weight` internamente.
     - Restaura `post_ref_rng` ao final de `build_backbone`, garantindo que tanto o estado de RNG de treino quanto o `RandomBatchStream` (`seed + 1`) partam do mesmo estado em todas as variantes recorrentes da mesma seed.
     - Registra hashes SHA-256 dos tensores comuns (`common_parameter_hashes`) e exclusivos (`exclusive_parameter_hashes`) na proveniência da execução e no diagnóstico.
2. **Matriz Científica da Etapa 3 (5 controles + candidato N1 × 3 seeds = 18 treinos de 2.000 passos em Inductor FP32):**
   - `lrm_no_jepa_k1` ($K=1, H=1024$, 1 escala, `paired_common_init=True`, **2.071.681** params, **108,74 GFLOPs/passo**)
   - `lrm_no_jepa_k2` ($K=2, H=1024$, 1 escala, `paired_common_init=True`, **2.071.681** params, **212,62 GFLOPs/passo**)
   - `flrm_compressed_grouped_no_jepa_k1` ($K=1, H=1024$, 3 escalas agrupadas, `paired_common_init=True`, **2.168.076** params, **166,57 GFLOPs/passo**)
   - `flrm_compressed_grouped_no_jepa_k2` (`G0`, $K=2, H=1024$, 3 escalas agrupadas, `paired_common_init=True`, **2.168.076** params, **328,28 GFLOPs/passo**)
   - `flrm_compressed_native_cached_no_jepa_k2` (`N1`, $K=2, H=1024$, 3 escalas nativas + cache, `paired_common_init=True`, **2.168.076** params, **282,30 GFLOPs/passo**)
   - `baseline_small` ($K=1, L=1, H=1280$, Transformer inicialização padrão na mesma seed, **2.164.224** params, **115,98 GFLOPs/passo**)
3. **Avaliações e Telemetria Pós-Treino na Etapa 3:**
   - Avaliação completa (437 janelas de validação) nos checkpoints de **500, 1.000, 1.500 e 2.000 passos**.
   - Telemetria corrigida de magnitudes e gradientes (`diagnose`) na inicialização (passo 0) e nos checkpoints de **500 e 2.000 passos** (com lote de janelas fixas de validação identificadas).
