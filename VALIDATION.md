# Validação vigente — 28/10/2026

## Organização do repositório

Checkout: `experiment/fractal-lrm`. Python 3.12.2, PyTorch 2.14.1+cpu, Windows, CUDA indisponível.
O código do checkout foi selecionado explicitamente por `PYTHONPATH=src`.

| Verificação executada | Resultado |
|---|---|
| `python -m pytest -q` neste checkout | **184 passed, 0 failed, 21 skipped**, exit code 0 |
| Empacotamento e notebooks | Cobertos pela suíte: ZIP íntegro, exclusões, recusa de sobrescrita, sintaxe e execução padrão sem rede/treino |
| Regras de ignore | 18 casos positivos/negativos aprovados por checkout; nenhum rastreado indevidamente ignorado |
| Preservação de evidências/notas locais | 1.285 arquivos, tamanho e SHA-256 preservados na base comum |
| Links relativos locais Markdown | Sem destinos ausentes na documentação verificada |
| `git diff --check` | Sem erros |
| Código/modelos/configs | Nenhuma alteração em `src/`, YAMLs científicos ou `pyproject.toml` |

Os 21 skips são testes CUDA; não foram tratados como aprovação. A suíte inclui
contrato de compilação com backend eager, que não comprova funcionamento de Inductor/CUDA.
As suítes dos dois checkouts rodaram em paralelo; seus tempos não são um benchmark.

Também foram executados na base comum `inspect --meta` (98.688 parâmetros, enumeração
igual à fórmula), planejamento de `experiment` sem `--execute` e ajuda dos comandos
operacionais. No experimento, o comando de planejamento do guia Kaggle retornou
`status=not_started`, sem treino/download. O primeiro exemplo sem `--input`/`--data`
falhou e foi corrigido; a impressão do erro expôs a limitação CP1252 documentada no guia.

Não foram executados novamente o exemplo de treino do guia, o notebook habilitado na
nuvem nem testes em GPU. Os testes de regressão usam dados e treinos temporários de
engenharia; nenhuma nova campanha científica foi iniciada. Nenhum commit/push foi feito.

## Evidência científica anterior

A [campanha 0015](docs/results/CAMPAIGN_0015.md) contém medições anteriores na T4
FP32/Inductor, com seu próprio commit e ambiente. Isso não valida A100/BF16 nem a
execução dos notebooks modificados nesta limpeza. Os dados curados foram extraídos
do ZIP recebido; não são resultados produzidos pelos testes CPU acima.

- [Auditoria de limpeza e limites](docs/CLEANUP_AUDIT.md).
- [Guia operacional](docs/OPERATIONS.md).
- [Histórico de validações preservado](docs/archive/VALIDATION_HISTORY.md).

Manifesto de movimentação, checks e resumos de execução ficam na base comum em
`artifacts/cleanup-2026-10-28/`, ignorados pelo Git. Resumos não substituem logs brutos.

## Auditoria da campanha 0016 — 29/10/2026

[Resultado e conformidade do ciclo 1](docs/results/CAMPAIGN_0016.md). CRC íntegro; 820 hashes em 20 snapshots e 41 arquivos contra Git conferidos; 60 avaliações completas reconstruídas com erro máximo 1,68583e−8. Binários de dados reconstruídos do corpus incluído reproduzem os hashes da 0015.

No checkout experimental, `python -m pytest -q tests/test_research_controls.py tests/test_document_data.py`: **28 passed**, CPU. Não houve nova execução CUDA nem treino científico. Logs do Kaggle registram 212 testes CPU e 17 CUDA FP32 aprovados; são resultados recebidos, distintos da checagem local.

Decisão: contexto sem RMS no ciclo 2. O runner recebido ainda não implementa teto global de sessão de 4h; perfis foram posteriores e treinos concorrentes. A implementação completa dos próximos ciclos continua pendente.

## Preparação do ciclo 2 — 29/10/2026

Implementação em `experiment/fractal-lrm`: matriz comunicação × FiLM, controles Transformer/LRM, seeds 81/82/83, contexto sem RMS, T256/B32/K2/2000 passos, FP32/Inductor. Novo executor serial com piloto prévio por variante/GPU, verificação da inicialização compartilhada, origem de dados, orçamento global de 4h e reserva de 300 s. Pacote fonte + bootstrap verificado por SHA-256 dispensam publicar commit.

Validações locais no checkout experimental (Python 3.12.2, Torch 2.14.1+cpu):

| Comando/check | Resultado real |
|---|---|
| `python -m pytest -q` | **237 passed, 0 failed, 25 skipped** |
| `python -m pytest -q tests/test_cycle2.py tests/test_cycle2_bootstrap.py tests/test_cycle2_release.py -m "not cuda"` após finalizar empacotador/mapeamento de GPU | **27 passed**, 4 testes CUDA fora da seleção |
| Planejamento do CLI sem `--execute` | 18 jobs, configurações/contagens verificadas, sem criar execução |
| Simulação ponta a ponta CPU | **complete**, seis braços, um bloco de seed, seis pilotos, 12 avaliações completas, relatório fatorial e ZIP CRC íntegro |
| Exclusões do ZIP de resultados | Sem pesos/estado de otimizador/binários de datasets |

Os testes abrangem causalidade, backward finito, memorização, retomada estrita com dropout/acumulação nos quatro braços, compilação AOT-eager CPU mantendo saídas dos ramos grossos, isolamento entre ramos, inicialização e contagens, watchdog/interrupção, exportação exclusiva, rejeição de ZIP inseguro/fonte alterada e notebook padrão inerte. O pequeno ajuste final de metadados de GPU foi coberto pelo segundo comando; não altera cálculo do modelo.

A simulação usou explicitamente `--engineering-smoke --skip-tests`: quatro passos, modelo pequeno, corpus sintético e CPU. A suíte foi executada separadamente. Esses resultados verificam o fluxo, não qualidade/eficiência científica. Os 25 skips são CUDA, inclusive quatro novos testes FP32/Inductor do ciclo 2; nenhum foi relatado como aprovado. A validação T4/Inductor do novo pacote ocorrerá no script entregue, antes da campanha real.

Logs, relatório da simulação e resumos ficam em `artifacts/cycle2-preparation-20261029/` no checkout experimental. O pacote novo contém arquivos de fonte e documentação, incluindo o presente registro. Sem commit/push ou execução científica nesta preparação. [Instruções do ciclo 2](docs/CYCLE2_RUN.md).

## Triagem das ideias externas — 29/10/2026

[Parecer sobre os 17 itens do catálogo do Argon](docs/REVIEW_TECHNOLOGIES_ARGON.md),
com fontes primárias e critérios de experimentação. Alterações desta revisão apenas
documentais; nenhum teste de modelo, benchmark ou treino novo. Checks reais: 28 links
locais válidos nos três documentos revisados em cada checkout; cópias do parecer
idênticas; `git diff --check` aprovado em ambos. SHA-256 do ZIP fonte e do launcher
do ciclo 2 permanecem iguais aos entregues. A revisão nova fica fora desse snapshot
imutável. Nenhum mecanismo adicional implementado e nenhum commit/push realizado.

## Célula curta por Git e teto de duas horas — 29/10/2026

A pedido do usuário, a entrega vigente usa commit completo do GitHub e dispensa
Input ZIP. Padrão de 7.200 s no bootstrap, CLI e orçamento da sessão; download e
checkout contam no relógio, com reserva de 300 s. Mantidos seis braços, seeds
81/82/83, 2.000 passos, scheduler e hiperparâmetros. Arquivos antigos em `dist/`
foram preservados; o launcher curto é gerado em novo diretório.

Validações reais em Python 3.12.2/Torch 2.14.1+cpu:

- `python -m pytest -q tests/test_cycle2_git.py tests/test_cycle2_bootstrap.py tests/test_cycle2_release.py`: **22 passed**.
- `rtk proxy python -m pytest -q -o faulthandler_timeout=90`: **250 passed, 25 skipped**, 298,10 s. A primeira chamada sem saída de progresso foi interrompida; não conta como aprovada. A repetição apenas acrescentou diagnóstico de timeout/saída direta, sem excluir testes.
- Clone Git local real: checkout da revisão anterior ao HEAD da fixture, conteúdo e hashes conferidos. Célula ativa baixa o helper pelo mesmo SHA; notebook desativado não faz ações externas.
- Setup simulado propaga o mesmo teto de 7.200 s e o tempo já gasto ao executor. Planejamento do CLI: 18 jobs, seeds 81/82/83 e 2.000 passos, sem execução.
- 47 links locais válidos nos guias/planos conferidos em cada checkout; `git diff --check` aprovado. ZIP e launcher anteriores conservam seus hashes.

Os 25 skips são de CUDA; não são suporte GPU verificado nesta máquina. Nenhum
treino científico novo foi executado. O bootstrap fará os testes T4/Inductor antes
da campanha. A entrega Git é publicada somente na branch experimental; documentação
espelhada na base comum não acrescenta código FLRM a essa branch.
