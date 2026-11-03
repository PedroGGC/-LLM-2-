# Documentação

Atualizado em 03/11/2026. Comece pelos documentos vigentes abaixo. A implementação
fractal pertence à branch `experiment/fractal-lrm`; a base comum não contém esse módulo.

| Documento | Finalidade |
|---|---|
| [Plano aprovado: recorrência/fractal](PLAN_RECURRENT_FRACTAL.md) | Hipóteses, candidatos, riscos, critérios e etapas condicionais |
| [Campanha0029: profundidade/leitura](STATE_READOUT_RUN.md) | Próximo treino:cinco braços/seed251/6000passos, teto2h, cache e ZIPs limitados |
| [Operação vigente: seleção e hierarquia](RECURRENT_RUN.md) | Etapas A/B, duas sessões distintas, meta2h/teto3h |
| [Etapa B: primeiro bloco mLSTM/fractal](STATE_B_RUN.md) | Quatro braços seed241, ciência qualificada congelada e cache por GPU |
| [Fichas dos candidatos](RECURRENT_CANDIDATES.md) | Fontes fixadas, equações e adaptações; não reproduções dos modelos completos |
| [Campanha 0027](results/CAMPAIGN_0027.md) | Primeiro bloco de hierarquia/sharing; Transformer vence, diagnóstico recomendado antes de novas seeds |
| [Campanha 0028](results/CAMPAIGN_0028.md) | Diagnóstico reproduzido; amplo contribui, sinal de defasagem recomenda controle treinado |
| [Diagnóstico dos pesos0027](STATE_DIAGNOSTIC_RUN.md) | Célula da0028 preservada para reprodução;11condições, sem treino ou novos checkpoints |
| [Campanha 0025](results/CAMPAIGN_0025.md) | mLSTM qualificado; etapa A concluída, próximo controle de hierarquia/sharing |
| [Campanha 0024](results/CAMPAIGN_0024.md) | mLSTM candidato operacional; falha na guarda de compilação exige requalificar |
| [Correção e armazenamento](PLAN_CAMPAIGN_0024_STORAGE.md) | Download embutido, temporários limitados, limpeza preservando resultados/pesos |
| [Campanha 0023](results/CAMPAIGN_0023.md) | 18 treinos auditados; fino simplifica, Transformer vence neste regime |
| [Operação histórica: ciclo 5](CYCLE5_RUN.md) | Preservada, suspensa como próxima prioridade |
| [Plano prévio da auditoria/retenção/ciclo 5](PLAN_CAMPAIGN_0023_RETENTION_CYCLE5.md) | Escopo, riscos e critérios da próxima entrega |
| [Dois scripts: backup e ciclo 4](CYCLE4_RUN.md) | Backup externo antes de limpeza completa; ablação de escalas, meta 2h/teto 3h |
| [Plano prévio do ciclo 4](PLAN_CYCLE4.md) | Escopo aprovado, riscos e critérios de aceitação |
| [Estado do projeto](PROJECT_STATUS.md) | Arquitetura real, tecnologias ativas/opcionais e limites |
| [Campanha 0022](results/CAMPAIGN_0022.md) | Lote de 32 min, quatro pares primários, auditoria e recomendação de revisão |
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
