# Plano vigente

Atualizado em 28/10/2026. O projeto usa Transformer como baseline e como bloco do núcleo
recorrente. Esta etapa organiza o repositório antes de decidir novos experimentos.

1. Organização, documentação e política de artefatos concluídas conforme
   [auditoria de limpeza](docs/CLEANUP_AUDIT.md).
2. Usar o [estado atual](docs/PROJECT_STATUS.md) e a
   [campanha 0015 corrigida](docs/results/CAMPAIGN_0015.md) como base da discussão.
3. Definir uma hipótese, controles, orçamento e critério de continuidade antes de
   implementar a próxima ablação. As opções estão no [roadmap](docs/RESEARCH_ROADMAP.md).

Nenhum novo mecanismo, treino científico, mudança de precisão ou ampliação de escala
faz parte desta limpeza. Preservar Transformer, contratos de checkpoints, causalidade
e separação da branch fractal. Recursos opcionais existentes não serão removidos só
porque estão desligados na campanha atual.

Plano original da etapa 1: [registro histórico](docs/archive/PLAN_STAGE1.md).
Plano V2 de 25/10: [registro histórico](docs/archive/RESEARCH_V2_PLAN.md).
Resultados de verificações executadas: [VALIDATION.md](VALIDATION.md).
