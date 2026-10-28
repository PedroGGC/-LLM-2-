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
