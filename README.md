# LLM2-Lab

Laboratório Python/PyTorch para comparar um **Transformer causal** com variantes que
reutilizam seus blocos por recorrência e entre escalas. Inclui preparo de dados,
treinamento, avaliação, checkpoints, diagnóstico e campanhas reproduzíveis.

**Já usamos Transformer dentro do LRM/FLRM.** A pergunta de pesquisa é se outra forma
de organizar a computação melhora qualidade por custo. Não há superioridade geral
demonstrada, garantia de raciocínio, ausência de colapso ou ganho por tecnologia.

## Qual checkout usar

Este checkout contém a pesquisa fractal: **`experiment/fractal-lrm`**.

| Checkout / branch | Conteúdo |
|---|---|
| `LLM 2` / `feature/llm2-research-v2` | Transformer, LRM, objetivos opcionais e infraestrutura comum |
| `LLM 2-fractal-lrm` / `experiment/fractal-lrm` | FLRM, compressão causal, cache, pareamento e variantes experimentais |
| `master` | Baseline histórico da etapa 1 |

A implementação fractal permanece isolada. O identificador `fractal` em uma configuração
não instala esse código na branch comum. Confira `git branch --show-current` antes de trabalhar.

## Estado atual — 29/10/2026

A campanha 0016 executou o ciclo 1: cinco variantes, seeds 71/72/73 e 2.000 passos,
TinyShakespeare byte-level, T=256 e T4 FP32/Inductor. Médias de NLL completa:

| Modelo | NLL ↓ |
|---|---:|
| Transformer pequeno | **1,394580** |
| FLRM N1 K=2 | 1,400059 |
| LRM K=2 | 1,403886 |
| FLRM N1 K=2 + contexto RMS | 1,443017 |
| LRM K=2 + contexto RMS | 1,444624 |

A normalização piorou a qualidade nas três seeds em ambas as famílias. O próximo
ciclo deve manter `context_normalization: none` e separar comunicação multiescala
e FiLM. Transformer continua melhor; custo igualado e outro corpus permanecem
pendentes. Treinos foram concorrentes e perfis isolados posteriores; o runner
recebido ainda não implementava um teto global de sessão.

JEPA/MTP/currículo/NCM e ciclos assimétricos ficaram desligados. Tecnologias futuras
continuam condicionais. As extensões científicas permanecem no checkout experimental.

Veja [auditoria 0016](docs/results/CAMPAIGN_0016.md),
[estado e decisões](docs/PROJECT_STATUS.md), [referência 0015](docs/results/CAMPAIGN_0015.md)
e [próximas perguntas](docs/RESEARCH_ROADMAP.md).

## Próximo teste — ciclo 2

O novo executor específico está em `experiment/fractal-lrm`: seis variantes × seeds
81/82/83, contexto sem RMS, piloto prévio, trabalhos seriais e orçamento global de
duas horas. Use a célula curta com commit fixado do GitHub, sem Input ZIP de fontes;
o bootstrap da 0016 não inclui esse executor. Instruções e limites: [guia do ciclo 2](docs/CYCLE2_RUN.md).

## Começar localmente

Python >=3.10, usando o mesmo interpretador para instalar e executar. Não é necessário
criar venv nem substituir uma instalação PyTorch compatível existente.

```powershell
python -m pip install -e ".[dev,research]"
$env:PYTHONPATH = (Resolve-Path ./src).Path
python -c "import llm2; print(llm2.__file__)"
python -m pytest -q
python -m llm2 inspect --config configs/cpu_smoke.yaml --meta
```

`PYTHONPATH` seleciona o código deste checkout na sessão — importante quando há duas
worktrees e uma instalação editável. A suíte usa dados temporários; inspeção em meta
não inicia treino. Não é preciso baixar um dataset para esses comandos.

O [guia operacional](docs/OPERATIONS.md) explica preparação de dados, treino individual,
retomada, avaliação e exportação. O [guia Kaggle](docs/KAGGLE_T4_CAMPAIGN.md) distingue
planejamento de execução. Configs smoke e perfis de hardware são exemplos, não uma
fila de experimentos aprovados.

## Organização

```text
src/llm2/       backbones, objetivos, Trainer, dados, avaliação e ferramentas
tests/         correção e regressões CPU/CUDA condicionais
configs/       configurações e catálogo de ablações
scripts/       execução, empacotamento e exportação
notebooks/     notebooks sem outputs de execução
docs/          documentação vigente; results/ contém evidência curada
docs/archive/  planos e registros históricos, explicitamente não vigentes
data/          datasets locais — ignorado
runs/          treinos e checkpoints locais — ignorado
artifacts/     ZIPs, auditorias e notas locais — ignorado
```

Leia o [índice da documentação](docs/README.md), a
[política de versionamento](docs/REPOSITORY_POLICY.md) e o
[registro de validação](VALIDATION.md). A auditoria de limpeza está em
[docs/CLEANUP_AUDIT.md](docs/CLEANUP_AUDIT.md).

Resultados grandes e checkpoints não entram no Git. Relatórios pequenos e dados
curados podem ser versionados em `docs/results/`. O ZIP de análise das campanhas
**não contém pesos/estado do otimizador e não permite retomar treino**.

Datasets e execuções existentes não são sobrescritos. Retomada estrita exige o código,
configuração e ambiente originais; organização do repositório não autoriza relaxar
essas verificações. Regras científicas e de contribuição: [AGENTS.md](AGENTS.md).
