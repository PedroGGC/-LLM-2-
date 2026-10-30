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

A [campanha 0019](docs/results/CAMPAIGN_0019.md) preservou **12 treinos completos e
48 avaliações completas** em WikiText-103, seeds 101/102. O executor registrou parada
controlada aos 2h40min45s por previsão de orçamento; a queda de rede não aparece
como causa de parada no arquivo recebido. Restam as seeds 103/104/105.

| FLOPs estimados semelhantes — médias de duas seeds | NLL ↓ |
|---|---:|
| Transformer de dois blocos | **1,196534** |
| LRM K=2 | 1,278126 |
| FLRM com comunicação, sem FiLM | 1,308835 |
| Transformer pequeno | 1,311478 |

O ganho FLRM frente ao Transformer pequeno é 0,002643, abaixo do limiar operacional,
com intervalo amplo cruzando zero. LRM e Transformer de dois blocos tiveram menor
NLL que o FLRM nas duas seeds. Duas seeds ainda não confirmam a hipótese; parâmetros
não foram igualados. [Ciclo 2](docs/results/CAMPAIGN_0017.md) e
[ciclo 1](docs/results/CAMPAIGN_0016.md) permanecem como evidência anterior.

## Próxima etapa

Por solicitação do usuário, próximas sessões têm **teto total de uma hora**, com
**até dois treinos novos**, validação focada e exportação após cada conclusão.
A próxima compara FLRM e Transformer pequeno com FLOPs estimados iguais na seed 103.
Os 12 resultados completos serão reaproveitados. Horizontes, quatro avaliações
completas, curvas e diagnósticos são preservados.

O operador novo usa workers no mesmo commit científico da 0019 e registra sua própria
proveniência separadamente. A confirmação continua exigindo cinco blocos completos;
um lote completo não encerra toda a pesquisa. A100 e novas tecnologias continuam
condicionais. [Guia vigente: ciclo 3 em lotes curtos](docs/CYCLE3_SHORT_RUN.md).

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
