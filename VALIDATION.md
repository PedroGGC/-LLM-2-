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

## Auditoria da campanha 0017 — 29/10/2026

[Relatório do ciclo 2](docs/results/CAMPAIGN_0017.md). ZIP de 17.388.696 bytes,
743 entradas, CRC íntegro e sem duplicatas; SHA-256
`8883f8213cb6eea11bd0890bd13dfef23c93310db5ee9b01793c3d41357c852f`.
Commit executado `60219eafbceea09066dd05206adaa9f4ce865f96`, limpo no registro.

Verificações locais desta auditoria: 178 hashes do bootstrap contra todo o commit;
46 arquivos de fonte contra Git; 1.380 hashes em 30 snapshots; 72 avaliações completas
reconstruídas de 437 janelas/111.872 alvos cada, erro máximo 1,92880818e−8; metadados
dos checkpoints/configs; inicialização pareada pelos registros; enumeração em meta
e fórmula de parâmetros dos seis modelos; dataset/manifesto idênticos à 0016 e
binários reconstruídos do corpus incluído. Script independente terminou sem erros.

Logs recebidos do Kaggle: **250 testes CPU aprovados**, 25 fora da seleção, 384,17 s;
**21 CUDA FP32 aprovados**, 254 fora da seleção, 153,66 s. Incluem os quatro braços
compilados do ciclo 2. São resultados no hardware real recebido, não testes CUDA
executados nesta máquina local. Treinos/pilotos em duas T4, serial/alternado;
protocolo FP32/Inductor, 18 treinos completos, 2.000 passos e três blocos de seed.

Tempo global registrado de 5.562,96 s antes de relatório/exportação, dentro do teto
de 7.200 s. Fonte, configs e pesos existentes preservados; nenhum treino, alteração
de código científico ou commit/push nesta auditoria. Os pesos/AdamW não estão no ZIP:
os hashes de pesos foram comparados entre registros, sem rehash dos arquivos ausentes.
Estatísticas exploratórias, três seeds no mesmo corpus, sem custo igualado. FiLM
fica opcional, comunicação é candidata à confirmação; superioridade não demonstrada.

Auditoria/script, dados intermediários e figura em
`artifacts/audits/campaign-0017-2026-10-29/` no checkout comum. Checks documentais:
links locais, cópias dos cinco artefatos curados e `git diff --check` conferidos.

## Preparação do ciclo 3 — 29/10/2026

Plano registrado em [docs/PLAN_CYCLE3.md](docs/PLAN_CYCLE3.md); operação em
[docs/CYCLE3_RUN.md](docs/CYCLE3_RUN.md). Sem mecanismos novos nem mudança de
backbone/loss/otimizador. Treinos e comparação científica na T4 continuam pendentes.

Verificações locais reais em Python 3.12.2 / Torch 2.14.1+cpu / PyArrow 23.0.0:

- Download dos quatro shards WikiText fixados, SHA-256 integral conferido; preparo
  real de 1.758/60/62 artigos e hashes tokenizados congelados. Detectada uma cópia
  treino/teste, excluída explicitamente só do treino antes da seleção. As outras
  442 duplicatas internas ao treino são deduplicadas. Nada foi avaliado no teste.
  [Proveniência curta](docs/results/CYCLE3_PREPARATION.json).
- Testes direcionados de ciclo 3, bootstrap anterior e dados: **49 passed, 4 skipped**.
  Testes adicionais de relógio de quatro horas e validação antes de download:
  **6 passed**. Os quatro skips são CUDA indisponível localmente.
- Suíte completa `python -m pytest -q -o faulthandler_timeout=90`: **274 passed,
  29 skipped**. Os 29 skips são de CUDA; não equivalem a testes GPU aprovados.
- Causalidade, forward/backward finitos, memorização e retomada da próxima atualização
  com dropout e acumulação CPU FP32/SDPA math, nas quatro famílias selecionadas,
  incluindo Transformer de dois blocos. Equivalência exata entre avaliação agrupada
  e CLI; publicação atômica do manifesto sob falha; rejeição de sobreposição desconhecida,
  conflitos de agregação, seeds duplicadas e falsa confirmação com dados incompletos.
- Simulação ponta a ponta CPU: **seis treinos pequenos / 24 avaliações completas**,
  quatro perfis, piloto de duração de validação, relatório/CSV/curvas e dois ZIPs.
  Status `complete`, `engineering_smoke=true`, `confirmation_eligible=false`,
  `test_evaluated=false`. CRC dos dois ZIPs e hashes de todos os seis checkpoints
  preservados no backup conferidos. Não é benchmark nem resultado científico.
  Evidência ignorada em `artifacts/cycle3-engineering-20261029/verification.json`.
- 57 links locais válidos nos guias/planos em cada checkout; `git diff --check` limpo.
  Dados, ZIPs e releases antigos preservados; fonte experimental fica fora da base comum.

O pareamento usa FLOPs estimados, com exclusões explícitas; não equivale a tempo
ou número de parâmetros igualados. Warmup inclui compilação/aquecimento; tempo
exclusivo de compilação e contador de recompilações não foram instrumentados.
O launcher executará os testes T4/Inductor e piloto no hardware real antes do treino.

## Correção de serialização Windows/Linux no ciclo 3 — 29/10/2026

A tentativa Kaggle da revisão `855c8d2` falhou após preparar o dataset e antes de
testes/treinos, no contrato de hashes. Causa reproduzida: `tokenizer.json` de 43 bytes
com CRLF no Windows tinha SHA-256 `a9be2688d42639c5915aa9222fcd9a389acbe6483c47ce170f3b0405e7644667`;
o JSON equivalente de 39 bytes com LF em Linux tem SHA-256
`f88f84a387d00bb8027db376d50fff07c86c4ee137f1678351521fefe0f8d252`.
O freeze local inicial foi incorreto para o Kaggle. Não era erro da configuração
usada pelo usuário nem um resultado de treinamento.

Correção: `write_json` agora grava UTF-8/LF explícito; protocolo exige o hash canônico,
mantendo os três hashes binários e demais contratos. Uma falha informa campo,
valor esperado e observado. Datasets existentes não foram migrados/sobrescritos;
retomada continua exigindo a implementação e o ambiente originais.

Verificações locais reais:

- Antes da correção, testes de portabilidade reproduziram **2 falhas e 1 aprovação**;
  depois, primeira rodada direcionada de JSON/dados/ciclo 3: **43 passed**, CUDA
  fora dessa seleção. Novos testes simulam defaults LF/CRLF independentemente do host.
- Suíte completa `python -m pytest -q -o faulthandler_timeout=90`: **280 passed,
  29 skipped**, exit code 0. O diagnóstico de 90 s imprimiu uma pilha durante um teste
  demorado; a suíte prosseguiu até aprovação. Skips são CUDA indisponível localmente.
- Corpus real preparado novamente em `artifacts/cycle3-portability-20261029/`,
  sem sobrescrever a preparação anterior. **Todos os hashes do protocolo passaram**;
  os três `.bin` foram comparados byte a byte com os anteriores e são idênticos.
  Tokens: 33.570.713 treino, 1.144.308 validação, 1.287.718 teste. Sem avaliar modelos.
- Leitura de dataset legado CRLF preserva bytes e fingerprint; adulteração continua
  rejeitada. Contrato do ciclo 3 rejeita tokenizer CRLF mesmo com semântica equivalente
  e identifica exatamente `files.tokenizer.json` no diagnóstico.
- Entrega de recuperação usa número 19, célula/notebook consistentes e pasta nova;
  a tentativa 0018, datasets, cache e releases anteriores são preservados. A duração
  máxima continua 14.400 s por execução. 45 links locais conferidos em cada checkout.

Plano/risco anterior às alterações: [PLAN_CYCLE3_PORTABILITY_FIX.md](docs/PLAN_CYCLE3_PORTABILITY_FIX.md).
CUDA não está disponível localmente; a correção de serialização não constitui
validação de GPU. O launcher mantém os testes T4/Inductor antes do treino.

## Auditoria 0019 e operação em lotes de uma hora — 29/10/2026

[Relatório](docs/results/CAMPAIGN_0019.md) e [plano anterior às mudanças](docs/PLAN_CYCLE3_SHORT_SESSIONS.md).
ZIP recebido de SHA-256 `e609269cf31f8d55d6afbbda9ef56b9b4c799fbd28f7921fd1f16479aa14887f`,
14.002.802 bytes, 442 entradas, CRC íntegro e sem nomes duplicados. **12 treinos
completos / 48 avaliações**, seeds 101/102; parada registrada por previsão de orçamento
aos 9.645,32 s. Não há erro de rede registrado como causa da parada nesse arquivo.

Conferidos 197 hashes contra o commit científico, 20 snapshots com 1.880 hashes,
configurações/contadores/FLOPs estimados, enumeração/fórmula de parâmetros e hashes
de checkpoint entre registros. NLL reconstruída de 4.469 janelas / 1.144.064 alvos
por avaliação; erro máximo 4,37404424e−9. Pesos não estão no ZIP recebido: isso não
constitui rehash dos tensores ausentes. Logs reais do Kaggle: **280 CPU / 25 CUDA
FP32 aprovados**. Resultado continua inconclusivo com duas seeds e teste reservado.

O novo operador apenas organiza os lotes: workers congelados exatamente em
`8798f6b84714d2c6671d970685984270078a982c`, com hashes científicos originais. Operador
e snapshot têm proveniência separada. Sessões de até 3.600 s, até dois treinos novos,
validação remota focada declarada e checkpoints/avaliações completos preservados.
Não há alegação de aceleração de kernel ou redução automática do custo de cinco seeds.

Verificações locais reais:

- Testes do operador/filas/agregação/deadline/exports/bootstrap: **16 passed**.
  Cobrem deduplicação de cópias, rejeição de conflitos de pesos/dataset/código/configs,
  seleção do par primário 103 sem repetir os 12 treinos e propagação de um único relógio.
- Suíte completa local `python -m pytest -q -o faulthandler_timeout=90`: **296 passed,
  29 skipped**, exit code 0. Skips são CUDA indisponível. Houve impressão diagnóstica
  de pilha aos 90 s durante um teste demorado; ele terminou e a suíte foi aprovada.
- Simulação ponta a ponta CPU com **checkout real do worker congelado**, dois treinos
  pequenos / oito avaliações completas. Status completo de lote, engenharia explícita,
  sem confirmação científica. Os hashes de todas as fontes e configs científicas nos
  resultados são idênticos aos da 0019. Três pares de ZIPs (após 1/2 treinos e final)
  tiveram CRC e hashes dos checkpoints conferidos, sem sobrescrita.
  Evidência: `artifacts/cycle3-short-engineering-20261029/verification.json`.
- Heartbeat operacional, perfis limitados aos modelos/slots selecionados, fila
  cumulativa e contrastes por pares preservam o requisito original de cinco blocos
  completos para a decisão confirmatória.
- 64 links locais conferidos em cada checkout, `git diff --check` sem erros; o ZIP
  original foi rehashado e preservado. Base comum recebe documentação, sem fonte FLRM.

A primeira chamada da suíte completa nesta alteração foi interrompida para ajustar
o ambiente do novo bootstrap antes da checagem de compatibilidade; não conta como
aprovada. `CUBLAS_WORKSPACE_CONFIG` segue a política do Trainer antes da comparação;
o checkout congelado exige bytes LF. CUDA do operador novo permanece pendente da
execução T4 real, ainda que o código científico já tenha sido validado na 0019.
