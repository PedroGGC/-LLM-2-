# Plano vigente

Atualizado em 29/10/2026. O projeto usa Transformer como baseline e como bloco do núcleo
recorrente. Esta etapa organiza o repositório antes de decidir novos experimentos.

1. Organização concluída, conforme a [auditoria de limpeza](docs/CLEANUP_AUDIT.md).
2. Ciclo 1 executado e auditado: [campanha 0016](docs/results/CAMPAIGN_0016.md).
   Manter contexto sem RMS; o ajuste piorou a qualidade nas três seeds.
3. Executor do ciclo 2 preparado no checkout experimental: piloto prévio, limite
   global de duas horas e matriz comunicação × FiLM. Executar a [célula curta no Kaggle](docs/CYCLE2_RUN.md)
   e auditar os resultados antes de escolher variantes.
4. Confirmar candidatos com dados independentes e custo igualado antes de A100.

Implementação científica continua em `experiment/fractal-lrm`. Tecnologias futuras
são condicionais; resultados novos não autorizam relaxar proveniência/retomada.
A auditoria de 29/10 não iniciou nova campanha nem alterou código de treinamento.

Plano original da etapa 1: [registro histórico](docs/archive/PLAN_STAGE1.md).
Plano V2 de 25/10: [registro histórico](docs/archive/RESEARCH_V2_PLAN.md).
Resultados de verificações executadas: [VALIDATION.md](VALIDATION.md).
