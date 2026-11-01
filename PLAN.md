# Plano vigente — recorrência, hierarquia e contribuição fractal

Atualizado em31/10/2026. O [plano aprovado](docs/PLAN_RECURRENT_FRACTAL.md) substitui
a prioridade anterior: selecionar uma regra recorrente viável, comparar núcleo
simples/hierarquia independente/hierarquia compartilhada/Transformer externo.
Fractal permanece hipótese central de organização. MC/GRM e TTT são candidatos,
não componentes obrigatórios. Base sólida não exige vitória do FLRM antigo.

Implementação/limites: [operação](docs/RECURRENT_RUN.md),
[fichas](docs/RECURRENT_CANDIDATES.md), [validação real](VALIDATION.md).
[Etapa A concluída na0025](docs/results/CAMPAIGN_0025.md): mLSTM qualificado nas duas
T4, RWKV também aprovado; Gated Delta excedeu o orçamento de compilação. Próxima
etapa científica: quatro controles de hierarquia/sharing com mLSTM, começando por
um bloco de seed. Revisar retenção de cache entre workers antes do script B.
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
