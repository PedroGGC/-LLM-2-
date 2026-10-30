# Plano vigente

Atualizado em 29/10/2026. O projeto usa Transformer como baseline e como bloco do núcleo
recorrente. Esta etapa valida componentes antes da confirmação independente.

1. Organização concluída, conforme a [auditoria de limpeza](docs/CLEANUP_AUDIT.md).
2. Ciclo 1 executado e auditado: [campanha 0016](docs/results/CAMPAIGN_0016.md).
   Manter contexto sem RMS; o ajuste piorou a qualidade nas três seeds.
3. Ciclo 2 executado e auditado: [campanha 0017](docs/results/CAMPAIGN_0017.md).
   Manter comunicação como candidata; FiLM opcional, com candidato simplificado
   desligado e pacote completo preservado como referência. Transformer ainda lidera.
4. Ciclo 3 parcialmente executado: 12 treinos da 0019 e o FLRM 103 completo na
   [0021](docs/results/CAMPAIGN_0021_PARTIAL.md), total 13. Conferir o estado vivo
   e liberar somente caches de código com a [manutenção preparada](docs/KAGGLE_WORKING_RECOVERY.md).
   Recuperação com até um job pendente, novo número e teto de uma hora. Reavaliar
   investimento após completar o comparador 103; novas tecnologias/A100 condicionais.

Implementação científica continua em `experiment/fractal-lrm`. Tecnologias futuras
são condicionais; resultados novos não autorizam relaxar proveniência/retomada.
A auditoria 0017 não iniciou nova campanha nem alterou código de treinamento.

Plano original da etapa 1: [registro histórico](docs/archive/PLAN_STAGE1.md).
Plano V2 de 25/10: [registro histórico](docs/archive/RESEARCH_V2_PLAN.md).
Resultados de verificações executadas: [VALIDATION.md](VALIDATION.md).
