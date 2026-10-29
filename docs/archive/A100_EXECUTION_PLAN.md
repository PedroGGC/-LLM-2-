> **Registro histórico — não é o plano vigente.** Arquivado em 28/10/2026.
> O [estado atual](../PROJECT_STATUS.md), os [resultados corrigidos da 0015](../results/CAMPAIGN_0015.md)
> e a [validação vigente](../../VALIDATION.md) prevalecem sobre descrições e conclusões antigas.

# Preparação A100 40 GB — sem consumo automático

Nenhuma A100 foi usada nesta entrega. BF16 CUDA, SDPA fused, Inductor e VRAM real continuam
pendentes. As configurações não garantem caber em 40 GB; começar pelo smoke CUDA explicitamente.

## Planejamento de créditos

Entradas fornecidas pelo usuário: 200 créditos e aproximadamente 5,30 créditos/hora. Com
reserva de 20%, ficam 160 créditos utilizáveis, equivalentes a **30,19 h estimadas**, deixando
40 de reserva. Uma sessão planejada de 30 min custa ~2,65 créditos nesse cenário; cabem
60 sessões inteiras nesse orçamento matemático. Isso não é previsão de tarifa constante.

```powershell
python -m llm2 plan-budget --config configs/planning/a100_budget.yaml
```

O watchdog verifica tempo **entre optimizer-steps**. `max_seconds` conta tempo cumulativo
observado do processo, inclusive inicialização/sondagens/validação até os pontos registrados;
o I/O do último checkpoint e atividades externas não entram integralmente no estado salvo.
`shutdown_reserve_seconds` antecipa a parada, mas uma etapa, avaliação, compilação, cópia Drive,
download ou outro comando pode ultrapassá-la. Ele não encerra o runtime Colab e não limita
cobrança exatamente. Setup/idle/notebook também gastam recursos. Monitore o runtime e encerre-o.

Limites adicionais: `max_tokens` aplica-se aos tokens densos de input (inclui padding de tarefas);
`max_train_flops` inclui treino e sondagens que influenciam o currículo. Validação científica é
contabilizada separadamente. Cada próxima etapa/sondagem deve caber antes de começar. A saída
registra `stop_reason` e salva só em fronteiras completas. Um erro no meio do backward exige
o checkpoint anterior. Limites podem ser ampliados explicitamente ao retomar, sem alterar
arquitetura/sampler/objetivo/scheduler; o novo YAML resolvido é registrado.

## Perfis, sem iniciar modelos grandes

| YAML em configs/profiles | Uso |
|---|---|
| cpu_minimal.yaml | Teste local mínimo FP32 |
| cuda_smoke.yaml | Sanidade CUDA/BF16 mínima |
| a100_7m.yaml | Faixa de 5M–10M, primeira campanha |
| a100_34m.yaml | Referência 33.990.144 |
| a100_77m.yaml | Faixa de 50M–100M, somente após perfis menores |
| future_352m.yaml | Faixa 300M–500M, planning_only bloqueia treino |

As contagens vêm de `python -m llm2 inspect --config CAMINHO --meta`; meta evita alocar pesos
e não valida consumo de ativações, compilação ou otimização. O teste verifica as faixas e a
contagem exata da referência. Mudar arquitetura pode mudar a contagem; sempre calcular novamente.

## Ordem recomendada de sessão

1. Preparar dados/tokenizer em CPU antes da sessão e preservar os artefatos com hashes no Drive.
2. Criar ZIP do checkout desejado: `python scripts/package_colab.py`. Para FLRM, usar o worktree
   experimental; não misturar módulos de commits diferentes manualmente.
3. Abrir notebook existente `notebooks/colab_a100.ipynb`, escolher A100 e conferir versões.
4. Selecionar YAML/arquitetura/dataset, novo RUN_NAME, limites e semente. A inspeção/preflight
   não inicia treino. Smoke CUDA e treinamento têm switches explícitos desligados por padrão.
5. Fazer um teste curto, medir VRAM/tempo real e reavaliar batch/contexto. `compile` permanece
   opt-in; cache depende de ambiente e o warmup deve ser registrado, não presumido gratuito.
6. Persistir checkpoints no Drive, verificar checksums, retomar no mesmo ambiente e exportar
   logs/configs/relatórios. O ZIP de resultados não duplica pesos/AdamW; eles ficam na execução.
7. Encerrar o runtime manualmente. Reservar créditos para falhas, avaliação e outras sementes.

Antes de escalar, repetir forward/backward/causalidade/overfit/retomada em GPU, comparar BF16
com FP32 e validar Inductor com dropout/EMA/checkpointing e configurações realmente usadas.
CPU/eager é o contrato de igualdade já exercitado; não presumir identidade entre kernels.
