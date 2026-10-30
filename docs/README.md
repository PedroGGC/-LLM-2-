# Documentação

Atualizado em 29/10/2026. Comece pelos documentos vigentes abaixo. A implementação
fractal pertence à branch `experiment/fractal-lrm`; a base comum não contém esse módulo.

| Documento | Finalidade |
|---|---|
| [Estado do projeto](PROJECT_STATUS.md) | Arquitetura real, tecnologias ativas/opcionais e limites |
| [Campanha 0019](results/CAMPAIGN_0019.md) | Dois blocos completos, custos, NLL e auditoria dos resultados preservados |
| [Snapshot da 0021](results/CAMPAIGN_0021_PARTIAL.md) | FLRM 103 concluído, pesos conferidos e histórico com 13 treinos |
| [Espaço e continuação no Kaggle](KAGGLE_WORKING_RECOVERY.md) | Limpeza limitada a código reproduzível e guarda de processos/checkpoints |
| [Próxima sessão: até uma hora](CYCLE3_SHORT_RUN.md) | Até dois treinos novos, código científico congelado e exportação incremental |
| [Correção da descoberta de ZIPs](PLAN_HISTORY_DISCOVERY_FIX.md) | Recuperação explícita de históricos antigos inválidos e proteção de lotes recentes |
| [Campanha 0017](results/CAMPAIGN_0017.md) | Ciclo 2 completo: comunicação × FiLM, custos e decisão de confirmação |
| [Executar ciclo 3](CYCLE3_RUN.md) | WikiText-103, cinco seeds, alvos/FLOPs estimados, teto de quatro horas e dois ZIPs |
| [Plano de implementação do ciclo 3](PLAN_CYCLE3.md) | Escopo, riscos e critérios registrados antes das mudanças |
| [Correção Windows/Linux](PLAN_CYCLE3_PORTABILITY_FIX.md) | Serialização LF canônica, integridade estrita e recuperação da tentativa 0018 |
| [Campanha 0016](results/CAMPAIGN_0016.md) | Auditoria do ciclo 1, efeito adverso da normalização e decisão para o ciclo 2 |
| [Campanha 0015](results/CAMPAIGN_0015.md) | Referência anterior e correções dos relatórios históricos |
| [Roadmap](RESEARCH_ROADMAP.md) | Perguntas em aberto, sem aprovação automática de tecnologias |
| [Triagem das ideias do Argon](REVIEW_TECHNOLOGIES_ARGON.md) | Análise dos 17 itens, correções e critérios para experiências posteriores |
| [Operação](OPERATIONS.md) | Instalação, dados, treino, retomada, avaliação e exportação |
| [Executar ciclo 2](CYCLE2_RUN.md) | Célula curta por commit fixado, matriz fatorial, piloto e teto global de duas horas |
| [Kaggle T4](KAGGLE_T4_CAMPAIGN.md) | Planejar/selecionar uma campanha com commit fixado |
| [LRM e objetivos](ARCHITECTURE_LRM.md) | Núcleo Transformer recorrente, JEPA/MTP opcionais |
| [FLRM](ARCHITECTURE_FLRM.md) | Escalas, modos comprimidos, cache e limites experimentais |
| [Currículo](CURRICULUM.md) | Harness opcional e avaliação procedural |
| [Catálogo de ablações](EXPERIMENT_MATRIX.md) | Configurações smoke existentes; não é a próxima campanha |
| [Modelo de relatório](RESULTS_TEMPLATE.md) | Como registrar medições e incerteza |
| [Política do repositório](REPOSITORY_POLICY.md) | O que versionar, ignorar e preservar |
| [Auditoria de limpeza](CLEANUP_AUDIT.md) | Escopo, riscos, mudanças e checks da reorganização |
| [Referências](REFERENCES.md) | Inspirações e diferenças da implementação |
| [Validação vigente](../VALIDATION.md) | Testes realmente executados e limitações |

[Arquivo histórico](archive/README.md): planos executados, narrativa conceitual antiga
e registros de validação acumulados. Não usar esses documentos como roteiro atual.
