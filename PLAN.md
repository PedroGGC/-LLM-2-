# Plano vigente

Atualizado em 29/10/2026. O projeto usa Transformer como baseline e como bloco do núcleo
recorrente. Esta etapa valida componentes antes da confirmação independente.

1. Organização concluída, conforme a [auditoria de limpeza](docs/CLEANUP_AUDIT.md).
2. Ciclo 1 executado e auditado: [campanha 0016](docs/results/CAMPAIGN_0016.md).
   Manter contexto sem RMS; o ajuste piorou a qualidade nas três seeds.
3. Ciclo 2 executado e auditado: [campanha 0017](docs/results/CAMPAIGN_0017.md).
   Manter comunicação como candidata; FiLM opcional, com candidato simplificado
   desligado e pacote completo preservado como referência. Transformer ainda lidera.
4. Ciclo 3 parcialmente executado e auditado: [campanha 0019](docs/results/CAMPAIGN_0019.md),
   12 treinos completos e duas seeds. Próxima etapa: [lote de até uma hora](docs/CYCLE3_SHORT_RUN.md),
   no máximo dois treinos novos e código científico original preservado. Avaliar a
   seed 103 antes de decidir quanto orçamento adicional a confirmação justifica.
   Nenhuma nova tecnologia nem promoção automática à A100.

Implementação científica continua em `experiment/fractal-lrm`. Tecnologias futuras
são condicionais; resultados novos não autorizam relaxar proveniência/retomada.
A auditoria 0017 não iniciou nova campanha nem alterou código de treinamento.

Plano original da etapa 1: [registro histórico](docs/archive/PLAN_STAGE1.md).
Plano V2 de 25/10: [registro histórico](docs/archive/RESEARCH_V2_PLAN.md).
Resultados de verificações executadas: [VALIDATION.md](VALIDATION.md).
