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

## Estado atual — 30/10/2026

A [campanha 0022](docs/results/CAMPAIGN_0022.md) concluiu o par primário da seed 104
em cerca de **32 minutos**, mantendo oito avaliações completas. O histórico cumulativo
preserva **16 treinos** e quatro pares FLRM × Transformer (101–104); somente as seeds
101/102 têm os seis braços completos. Há 14 jobs pendentes na matriz original.

| FLOPs estimados semelhantes — médias de quatro seeds | NLL ↓ |
|---|---:|
| Transformer pequeno | **1,312215** |
| FLRM com comunicação, sem FiLM | 1,314060 |

Diferença FLRM − Transformer: **+0,001845**, IC95% pareado exploratório
[−0,006698; +0,010387]. Duas vitórias para cada arquitetura. Ainda não há vantagem
confirmada nem equivalência demonstrada. Os resultados dos controles LRM e Transformer
de dois blocos continuam limitados às duas seeds da [0019](docs/results/CAMPAIGN_0019.md).
[Ciclo 2](docs/results/CAMPAIGN_0017.md) e [ciclo 1](docs/results/CAMPAIGN_0016.md)
permanecem como evidência anterior. Na 0022, 12 testes CPU e 2 CUDA FP32 passaram
no Kaggle; pesos não constam no ZIP de resultados recebido.

## Próxima etapa recomendada

Revisar a hipótese e os controles antes de novos treinos deste protocolo. A regra
pré-fixada para vantagem FLRM exige vitória em quatro das cinco seeds; após duas
vitórias em quatro pares, uma quinta permitiria no máximo três. A análise permanece
exploratória/inconclusiva; isso não demonstra inviabilidade geral de outras arquiteturas
ou regimes. A100 e novas tecnologias continuam condicionais.

O [executor de lotes curtos](docs/CYCLE3_SHORT_RUN.md) preserva o teto de uma hora,
até dois treinos novos e exportação incremental. O código científico permanece no
commit da 0019, com proveniência separada do operador. A confirmação original exige
cinco blocos completos; quatro pares disponíveis não equivalem a quatro blocos.
A [manutenção do Kaggle](docs/KAGGLE_WORKING_RECOVERY.md) preserva campanhas e backups.

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
