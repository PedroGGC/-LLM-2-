# LLM2-Lab

Laboratório Python/PyTorch para comparar um **Transformer causal** com variantes que
reutilizam seus blocos por recorrência e entre escalas. Inclui preparo de dados,
treinamento, avaliação, checkpoints, diagnóstico e campanhas reproduzíveis.

**Já usamos Transformer dentro do LRM/FLRM.** A pergunta de pesquisa é se outra forma
de organizar a computação melhora qualidade por custo. Não há superioridade geral
demonstrada, garantia de raciocínio, ausência de colapso ou ganho por tecnologia.

## Qual checkout usar

Este checkout é a base comum: **`feature/llm2-research-v2`**.

| Checkout / branch | Conteúdo |
|---|---|
| `LLM 2` / `feature/llm2-research-v2` | Transformer, LRM, objetivos opcionais e infraestrutura comum |
| `LLM 2-fractal-lrm` / `experiment/fractal-lrm` | FLRM, compressão causal, cache, pareamento e variantes experimentais |
| `master` | Baseline histórico da etapa 1 |

A implementação fractal permanece isolada. O identificador `fractal` em uma configuração
não instala esse código na branch comum. Confira `git branch --show-current` antes de trabalhar.

## Estado atual — 28/10/2026

A campanha 0015 comparou seis variantes, três seeds e 2.000 passos em TinyShakespeare
byte-level, T=256, T4 FP32/Inductor. Médias da avaliação completa:

| Referência | NLL ↓ | Tempo acumulado de treino ↓ |
|---|---:|---:|
| Transformer pequeno | **1,394923** | **96,24 s** |
| FLRM N1, nativo com cache, K=2 | 1,399268 | 220,93 s |
| FLRM G0, agrupado, K=2 | 1,401717 | 261,61 s |

O Transformer venceu as três seeds. N1 melhorou custo de treino e loss frente a G0,
mas gerou mais devagar no benchmark eager sem KV cache. Recorrência e multiescala
continuam hipóteses úteis para investigação, não ganhos universais estabelecidos.

JEPA, MTP e currículo são recursos opcionais; estão desligados nessa campanha. NCM
de protótipos e ciclos assimétricos existem somente no experimento e também ficaram
desligados. Fibonacci/φ-ALiBi, ACT/DEQ e Kronecker/Monarch são possibilidades futuras.

Veja [estado e decisões](docs/PROJECT_STATUS.md),
[resultados e errata da campanha 0015](docs/results/CAMPAIGN_0015.md) e
[próximas perguntas](docs/RESEARCH_ROADMAP.md).

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
