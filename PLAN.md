# Plano vigente — aperfeiçoamento e escala do `fractal_golden`

Atualizado em 15/11/2026. A fase de triagem e comparação de arquiteturas foi **concluída** nas campanhas **IA2 001 a 006** ([relatório consolidado e filtro tecnológico](docs/results/IA2_CAMPAIGNS_001_006.md)). O modelo **`fractal_golden`** (`FractalDeltaMemory` com 6 níveis compartilhados, `fractal_value_residual=True`, `fractal_golden_phase=True`, canal associativo lag-1 e `dplr_chunked` otimizado) venceu todas as variantes fractais em modelagem de linguagem e extrapolação (`0,7282` vs `1,4256` NLL em `3.219` bytes frente ao `transformer_big` 26% maior), mantendo `100%` (`32w`) / `99,12%` (`64w`) em `Byte-Binding @8192` e decodificação `8,0×` mais rápida (`1.861,6 B/s`).

**Nova prioridade operacional:** encerrar comparações horizontais entre famílias e concentrar o desenvolvimento na melhoria direta do `fractal_golden`:
1. **Engenharia e escala (Colab L4/A100):** validação de `BF16` com acumuladores FP32 no scan DPLR, tokenizador BPE (`vocab=8192+`), descompartilhamento seletivo (`share_scales` parcial/por blocos) e redução de VRAM de treino.
2. **Contexto e objetivos:** currículo de comprimento de sequência e Multi-Token Prediction (MTP) sobre a base existente.
3. **Esforço adaptativo (médio prazo):** profundidade variável ($K$ dinâmico / RINS) seguida de parada aprendida (ACT/PonderNet).
4. **Condicionais a gargalo de escala:** controle espectral/nGPT (apenas se houver instabilidade em $L \ge 12$ / `BF16`) e Kronecker/Monarch/quantização pós-treino nas camadas densas em escalas $\ge 30\text{M}$ parâmetros.

## Registro anterior — 31/10/2026 (recorrência, hierarquia e contribuição fractal)

O [plano aprovado](docs/PLAN_RECURRENT_FRACTAL.md) substitui
a prioridade anterior: selecionar uma regra recorrente viável, comparar núcleo
simples/hierarquia independente/hierarquia compartilhada/Transformer externo.
Fractal permanece hipótese central de organização. MC/GRM e TTT são candidatos,
não componentes obrigatórios. Base sólida não exige vitória do FLRM antigo.

Implementação/limites: [operação](docs/RECURRENT_RUN.md),
[fichas](docs/RECURRENT_CANDIDATES.md), [validação real](VALIDATION.md).
[Etapa A concluída na0025](docs/results/CAMPAIGN_0025.md): mLSTM qualificado nas duas
T4, RWKV também aprovado; Gated Delta excedeu o orçamento de compilação. Próxima
etapa científica: quatro controles de hierarquia/sharing com mLSTM, começando por
um bloco de seed. [Operação B](docs/STATE_B_RUN.md) conserva a fonte qualificada e
limpa caches por GPU ao trocar variante; triagem agrupada reutiliza compilação.
Sem novo treino científico de linguagem; não repetir qualificação já aprovada.

## Registro anterior — supersedido como próxima operação

Atualizado em 30/10/2026. O projeto usa Transformer como baseline e como bloco do núcleo
recorrente. Esta etapa valida componentes antes da confirmação independente.

1. Organização concluída, conforme a [auditoria de limpeza](docs/CLEANUP_AUDIT.md).
2. Ciclo 1 executado e auditado: [campanha 0016](docs/results/CAMPAIGN_0016.md).
   Manter contexto sem RMS; o ajuste piorou a qualidade nas três seeds.
3. Ciclo 2 executado e auditado: [campanha 0017](docs/results/CAMPAIGN_0017.md).
   Manter comunicação como candidata; FiLM opcional, com candidato simplificado
   desligado e pacote completo preservado como referência. Transformer ainda lidera.
4. Ciclo 3 parcialmente executado: [0022](docs/results/CAMPAIGN_0022.md) preserva
   16 treinos e quatro pares primários. FLRM − Transformer médio +0,001845 NLL,
   IC exploratório cruzando zero; duas vitórias por arquitetura. Somente dois
   blocos de seis braços estão completos. A regra positiva de quatro vitórias em
   cinco seeds já não é alcançável neste ensaio. Recomendação: pausar novos treinos
   deste protocolo e revisar a hipótese com os controles existentes; resultado
   permanece inconclusivo. Novas tecnologias/A100 continuam condicionais.

5. Ciclo 4 executado/auditado: [0023](docs/results/CAMPAIGN_0023.md). Dezoito treinos,
   72 avaliações principais e24probes. Completo−fino+0,000881NLL: nenhum ganho
   confirmado das escalas. Somente fino cumpre critério de simplificação (~24%
   menos tempo). Transformer2blocos vence todos neste regime; K não treinado piora.
6. [Ciclo 5](docs/CYCLE5_RUN.md): 12controles, K1/2/4 treinados do zero e Transformer,
   seeds221/222/223. Parâmetros próximos e FLOPs estimados igualados; alvos diferentes.
   Meta2h/teto3h. Sem tecnologia nova. Um par de ZIPs temporários aos6/12jobs,
   removido após verificar os dois definitivos. Limpeza da0023 só remove ZIPs
   after-NN redundantes, protegendo pasta/pesos e definitivos. Confirmação posterior
   com cinco novas seeds depende de sinal favorável; não repetir matriz de escalas.

Implementação científica continua em `experiment/fractal-lrm`. Tecnologias futuras
são condicionais; resultados novos não autorizam relaxar proveniência/retomada.
A auditoria 0017 não iniciou nova campanha nem alterou código de treinamento.

Plano original da etapa 1: [registro histórico](docs/archive/PLAN_STAGE1.md).
Plano V2 de 25/10: [registro histórico](docs/archive/RESEARCH_V2_PLAN.md).
Resultados de verificações executadas: [VALIDATION.md](VALIDATION.md).
