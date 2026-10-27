# Plano — Campanha 0007: FLRM Simétrico Fused+NCM vs Ciclo-V Invertido (Macro-First) + Otimização Nativa de Upsample

## 1. Objetivo

1. **`flrm_fused_ncm_k2` (FLRM Simétrico $K=2$ com Fusão + NCM-16)**:
   - Executa as 2 iterações completas sobre todas as escalas ($s_0, s_1, s_2$), mas com a **Fusão Vetorizada de Kernels** (`fused_film=True`), a **Otimização Algébrica de Upsample/Compressão** e o **NCM-16** (`ncm_slots=16`).
   - Meta: bater o recorde absoluto de loss de validação da Campanha 0003 ($< 1,397467$) e medir a economia real de tempo da fusão sem poda de escala.
2. **`flrm_inv_vcycle_ncm_k2` (FLRM Ciclo-V Invertido / "Macro-Primeiro" + Fusão + NCM-16)**:
   - **Iteração $k=0$ (Abstração Macro)**: Executa o `SharedFractalCore` + NCM **exclusivamente nas escalas macro comprimidas** ($s_1, s_2$, apenas $64 + 16 = 80$ posições em $T=256$), enquanto $v_0 = s_0$ aguarda em zero. Ao final de $k=0$, o `CrossScaleInteraction` injeta o resumo macro causal diretamente em `next_s0`.
   - **Iteração $k=1$ (Síntese Fina Contextualizada — Passo Final)**: Com $s_0$ já enriquecido pelo raciocínio macro e pelos protótipos NCM da iteração anterior, executa o `SharedFractalCore` **em todas as escalas** ($s_0, s_1, s_2$), garantindo que a escala fina $s_0$ passe por Atenção + SwiGLU na última iteração antes de `hidden = norm(s0)`.
   - Meta: Resolver a baixa atualização relativa de $s_0$ observada no Ciclo-V padrão ($10,5\%$), recuperando a loss competitiva ($< 1,398$) com o mesmo custo reduzido de FLOPs (~225 GFLOPs/passo, ~220 s para 2.000 passos).
3. **Otimização Algébrica de `CrossScaleInteraction.forward_native` (Eliminação de Upsample + Recompressão Redundante)**:
   - Como $\text{compress}_C(\text{upsample}_C(v, T))$ depende apenas da média em blocos de $C$ ou $4C$ posições de um sinal constante por blocos de tamanho $C$, evitamos materializar e recomprimir tensores de comprimento $T=256$ para as saídas macro:
     - $\text{compress}_4(u_1) = \text{shift}_1(v_1, \text{prior}_4)$ diretamente em $T_1 = 64$ posições!
     - $\text{compress}_{16}(u_2) = \text{shift}_1(v_2, \text{prior}_{16})$ diretamente em $T_2 = 16$ posições!
   - Isso reduz a compressão empilhada na escala $T/4$ de 3 sequências de comprimento $T=256$ para apenas **1 sequência** ($v_0$), cortando 2/3 do custo de pooling e reduzindo lançamentos de kernels e tráfego de memória em todos os modos `fused_film=True`.

## 2. Riscos e Invariantes

- **Invariância numérica estrita**: O atalho algébrico para $\text{compress}_4(\text{upsample}_4(v_1, T))$ e $\text{compress}_{16}(\text{upsample}_{16}(v_2, T))$ deve produzir tensores numericamente idênticos ao caminho explícito `compress(upsample_causal(v, T))` em qualquer comprimento $T \ge 1$ (incluindo comprimentos não divisíveis por 4 ou 16, como $T=1, 15, 17, 31, 33, 65, 256$).
- **Causalidade de prefixo**: Nenhuma posição $t$ pode ler posições $> t$ em nenhum modo (`v_cycle`, `inv_vcycle`, `fused_film`, `ncm_slots`).
- **Compatibilidade de Checkpoints e Configurações**: `ResearchConfig` adiciona `inv_vcycle: bool = False` (mutuamente exclusivo com `v_cycle=True`). Todos os modos anteriores permanecem inalterados.

## 3. Critérios de Aceitação

- `python -m pytest -q` 100% aprovado localmente.
- Testes dedicados cobrindo equivalência exata do atalho de compressão/upsample em comprimentos arbitrários, formas de execução do `inv_vcycle` (`[(4, 9), (2, 33), (4, 9)]` para $T=33$), causalidade estrita, memorização e retomada exata bit a bit (`resume-equivalence`).
