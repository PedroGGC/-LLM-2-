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

## Estado atual — 09/11/2026

Nova operação autorizada e implementada: [memória de transições e pirâmide](docs/ASSOCIATIVE_BINDING_RUN.md).
Núcleo associativo de86.785pesos, pirâmide compartilhada86.979 e independente111.652,
contra2.164.992do Transformer externo. Não usa o núcleo mLSTM nem DecoderBlock.
Consulta/escrita com endereço aprendido, conteúdo separado e atualização linear.
Fractal é testado pela reutilização do mesmo operador entre ritmos de memória.

Qualificação CPU: núcleo único, seed816/1000passos,100%em novos mapas de32pares
até8K,64pares em1K e fronteira deslocada, incluindo perguntas com chaves trocadas.
Isso é capacidade sintética sobre alfabeto finito; não é comparação com Transformer,
eficiência naT4 ou resultado de linguagem. A campanha nova comparará os quatro
braços/seeds311–312/3000passos nas duas T4, teto2h, com workers persistentes.
[Célula curta para Kaggle](scripts/09_testar_associacao_e_piramide_kaggle.py).

A [campanha0030](docs/results/CAMPAIGN_0030.md) concluiu 4/14 treinos, seed271,
em1h11min16s registrados antes da exportação. T1024: Transformer6,88% de acerto,
recorrente1,32%, banco completo1,42%, Fibonacci1,56%; acaso entre64valores1,56%.
Nenhum resolveu recuperação associativa. Banco completo também falha; não atribuir
o resultado somente ao acesso Fibonacci. Modelos com banco têm40,65% menos pesos,
mas forward eager cerca de19–20× mais lento neste perfil, sem vantagem de eficiência.

Após0030, mudamos de família: em vez do cache de resumos médios, a nova operação
qualifica escrita/leitura de associações e sua organização em ritmos. O mLSTM
histórico continua preservado, sem alterar sua inicialização retroativamente.
Workers persistentes substituem os pilotos repetidos. Hipótese fractal e demais
frentes preservadas. [Operação0030 para reprodução](docs/MEMORY_RECALL_RUN.md).

A [campanha0029](docs/results/CAMPAIGN_0029.md) concluiu cinco treinos de6000passos,
seed251, nas duas T4, em1h10min33s registrados antes da exportação. NLL final:
Transformer1,271962; recorrente de duas células1,315631; hierarquia imediata1,427357;
hierarquia seguinte1,436200; fino de uma célula1,474430. Leitura imediata ajuda
0,008843, mas a hierarquia ainda não supera os controles. Amplo ajuda frente ao
fino treinado, com mais parâmetros; a dependência da posição no grupo permanece.

Após0029 propusemos investigar compressão. A autorização de08/11prioriza agora
memória/recuperação e comparação explícita de padrões de acesso, incluindo Fibonacci.
Compressão permanece hipótese posterior. Hipótese fractal preservada; nenhuma
vantagem confirmada. [Operação0029 preservada](docs/STATE_READOUT_RUN.md).

A [campanha0028](docs/results/CAMPAIGN_0028.md) concluiu11avaliações congeladas,
sem treino, em109,24s registrados; ZIP410KiB. Reproduziu as quatro perdas da0027
com erro máximo2,61e-9. Reduzir/silenciar o ramo amplo piorou ambas as hierarquias.
Há sinal descritivo de perda maior na quarta posição dos grupos; o código adia
para o byte seguinte a leitura do grupo recém-completado. Isso motivou o controle
treinado de leitura e profundidade fina da0029, sem acrescentar mecanismos.

A [campanha 0027](docs/results/CAMPAIGN_0027.md) concluiu o primeiro bloco da etapa B:
quatro treinos de6000passos, seed241, T1024/B8, mLSTM e Transformer. NLL final:
Transformer1,273494; recorrente simples1,313647; hierarquia independente1,431568;
compartilhada1,462067. Fontes/métricas conferidas; teste reservado não consultado.
Só uma seed: resultado exploratório, sem confirmação estatística ou vantagem fractal.

A hierarquia economizou cerca de32% por passo frente ao simples, com forte perda de
qualidade. Sharing reduziu47,72% dos parâmetros frente à hierarquia independente,
mas piorou0,030498NLL e economizou apenas3–4% de tempo. Transformer liderou também
tempo isolado e VRAM de treino. Geração/recuperação ainda não medidas.

Recomendação: diagnosticar compressão, perda de profundidade fina e mistura do ramo
amplo antes de investir nas seeds242/243. Protocolo original preservado; as seeds
restantes continuam pendentes. Hipótese fractal e candidatos futuros permanecem.
Sessão registrada1h55min38s antes da exportação final, cache máximo amostrado1,124GiB;
nenhuma violação da guarda de armazenamento registrada.

A [operação de diagnóstico](docs/STATE_DIAGNOSTIC_RUN.md) permanece disponível
para reprodução da0028. Duas T4/eager, sem cache de compilação ou novo treino.
Não é indicação de repeti-la automaticamente. Preserve a pasta0027 ou seu backup
de checkpoints; `results.zip` sozinho não contém os pesos necessários.

## Evidência histórica — FLRM com atenção, contexto256

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

## Plataforma recorrente e operação da etapa B

A [0025](docs/results/CAMPAIGN_0025.md) concluiu a qualificação corrigida: mLSTM
aprovado nas duas T4,93,43/89,99ms por passo,746,92MiB de pico de VRAM. RWKV
também aprovado; Gated Delta atingiu o timeout externo. Duração55,17min e cache
amostrado abaixo de928MiB;4,70GiB acumulados de cache removidos. A falha da
[0024](docs/results/CAMPAIGN_0024.md) foi corrigida. Não é ganho de linguagem/fractal.
Download embutido e limpeza de caches/checkouts descartáveis passam a ser padrão;
[limpeza do working](scripts/04_limpar_temporarios_kaggle.py) preserva arquivos científicos.

A [etapa B](docs/STATE_B_RUN.md) executou os controles com mLSTM selecionado
por medições T4. Nova operação mantém uma família de cache por GPU e preserva o
worker científico qualificado. Primeiro lote: quatro braços da seed241.
Tentativa0026 parou antes dos treinos por checksum do cache bruto. A célula05
atualizada verifica/recupera esse cache antes dos testes CUDA e começa em0027,
preservando os hashes e a ciência qualificada.
São regras num shell comum,
**não reproduções dos modelos publicados**; [fichas e adaptações](docs/RECURRENT_CANDIDATES.md).
CPU verifica matemática e integração; a escolha operacional foi feita nas duas T4.

Matriz registrada: recorrente simples, hierarquia independente, hierarquia compartilhada e
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
