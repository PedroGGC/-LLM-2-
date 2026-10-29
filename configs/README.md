# Configurações

Os caminhos internos de cada YAML são relativos ao próprio arquivo. Configurações
existentes foram preservadas: renomeá-las ou reformatá-las sem necessidade interfere
na proveniência das execuções antigas.

| Grupo | Uso |
|---|---|
| `cpu_smoke.yaml`, `profiles/cpu_minimal.yaml` | Sanidade de engenharia CPU |
| `experiments/e*.yaml`, `c*.yaml`, `combined.yaml` | Catálogo de ablações V2 e currículo, não próximos treinos aprovados |
| `experiments/f*.yaml`, `c5/c6` | Somente no checkout experimental |
| `kaggle_t4*.yaml` | Perfis base/históricos; conferir aliases e overrides do runner |
| `profiles/a100_*.yaml`, `transformer_34m_a100.yaml` | Perfis opcionais, sem validação A100 nesta campanha |
| `profiles/future_352m.yaml` | Planejamento, treino bloqueado por `planning_only` |
| `planning/a100_budget.yaml` | Cenário histórico de orçamento; não saldo/tarifa atuais |

A campanha 0015 resolve suas configurações por `campaign_config`, não por um YAML N1
dedicado. No experimental, o alias N1 é `flrm_compressed_native_cached_no_jepa_k2`;
declarar também compile/pareamento/batch/seeds. O runner pode substituir campos do
preset base. Sempre inspecionar a configuração resolvida e registrar os overrides.

Leia [operação](../docs/OPERATIONS.md), [Kaggle](../docs/KAGGLE_T4_CAMPAIGN.md) e
[estado atual](../docs/PROJECT_STATUS.md). Manter configurações opcionais não significa
que seus mecanismos melhoraram qualidade ou estejam ativos na campanha atual.
