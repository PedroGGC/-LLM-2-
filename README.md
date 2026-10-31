# LLM2-Lab

Laboratório Python/PyTorch para investigar uma arquitetura própria **inspirada em
fractais**, com hierarquia, recorrência e compartilhamento entre escalas. O Transformer
causal é a referência externa. Inclui preparo de dados,
treinamento, avaliação, checkpoints, diagnóstico e campanhas reproduzíveis.

O LRM/FLRM histórico reutiliza Transformer; os novos backbones `state_recurrent` e
`state_hierarchy` usam regras de memória recorrentes, sem DecoderBlock/SDPA. A
pergunta é se outra organização melhora qualidade por custo. Não há superioridade geral
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

## Estado atual — 31/10/2026

A [campanha 0023](docs/results/CAMPAIGN_0023.md) concluiu e auditou 18 treinos,
72 avaliações principais e 24 probes de profundidade em aproximadamente duas horas.
Pesos e estados finais foram recebidos e seus hashes conferidos sem pickle.

| Médias de três seeds, 6.000 passos e alvos iguais | NLL ↓ | ms/step isolado, T4 0 / 1 ↓ |
|---|---:|---:|
| Transformer, dois blocos | **1,285904** | **44,98 / 45,58** |
| FLRM somente fino | 1,348510 | 69,98 / 73,81 |
| FLRM completo | 1,349391 | 91,83 / 97,11 |
| LRM K=2 com cache de contexto | 1,349904 | 66,60 / 68,33 |

As escalas não cumpriram o ganho pré-fixado. Somente fino cumpre o critério de
**simplificação candidata**: perda máxima 0,00138 NLL e cerca de 24% menos tempo nas
duas GPUs. Não demonstra equivalência formal ou vantagem fractal. O Transformer
lidera entre as configurações testadas. Aumentar K sem treinar naquela profundidade
piorou a NLL; esse probe não avalia treinamento em profundidade variável.

## Próxima etapa — seleção recorrente e contribuição fractal

A [0024](docs/results/CAMPAIGN_0024.md) mediu mLSTM em94,67/96,48ms por passo,
o melhor entre as regras testadas, mas os seis pilotos ultrapassaram a guarda
de600s: a exceção foi absorvida pelo compilador. A próxima célula requalifica com
supervisor externo e formas estáticas. Não é evidência de qualidade de linguagem.
Download embutido e limpeza de caches/checkouts descartáveis passam a ser padrão;
[limpeza do working](scripts/04_limpar_temporarios_kaggle.py) preserva arquivos científicos.

A [nova etapa](docs/RECURRENT_RUN.md) começa selecionando uma implementação recorrente
na T4, sem escolher previamente Gated Delta/RWKV/mLSTM. São regras num shell comum,
**não reproduções dos modelos publicados**; [fichas e adaptações](docs/RECURRENT_CANDIDATES.md).
CPU verifica matemática e integração, sem escolher vencedor CUDA.

Depois: recorrente simples, hierarquia independente, hierarquia compartilhada e
Transformer2blocos; T1024/B8, 6000passos, seeds241–243. As hierarquias têm as mesmas
operações; sharing reduz pesos, não estados nem chamadas. Meta2h/teto3h, divisão
em blocos de seed sem encurtar horizontes. Nenhum GRM/TTT/ACT/JEPA é obrigatório.

O [plano aprovado](docs/PLAN_RECURRENT_FRACTAL.md) e o [roadmap](docs/RESEARCH_ROADMAP.md)
mantêm fractal explícito e tecnologias candidatas. Ciclo5 fica preservado como operação
histórica, suspenso como próxima prioridade. FLOPs dos novos scans são null até
estimador específico; não usar a fórmula de atenção antiga. Sem vitória nova medida.
Histórico: [0022](docs/results/CAMPAIGN_0022.md), [0019](docs/results/CAMPAIGN_0019.md),
[0017](docs/results/CAMPAIGN_0017.md) e [0016](docs/results/CAMPAIGN_0016.md).

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
