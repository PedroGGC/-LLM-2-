# Validação vigente — 10/11/2026

## Campanha 0032 — T4 e auditoria dos resultados

[Relatório](docs/results/CAMPAIGN_0032.md): status complete, oito treinos novos,
seeds331/332, 6.000 passos; 1.271,61s antes de exportar. ZIP2.527.514bytes/212membros,
CRC válido e caminhos seguros, sem pesos/pickle/corpus binário. 388 hashes do
bootstrap contra Git6f469fa; snapshot150arquivos (107fontes/dependências e43configs).
Configurações resolvidas, protocolo/manifesto portátil, parâmetros por meta/fórmula,
embedding ligado, workers, metadados de checkpoints, resumos, 160registros finitos,
32avaliações completas e48gerações por bytes originais reconciliados. Dados locais
verificados têm o mesmo manifesto; nenhum teste reservado avaliado.

Preflight no Kaggle: **CPU13passed/2deselected,18,57s; CUDA2passed/5deselected,
32,40s**, GPU0. Todos os braços treinados/perfilados nas duas T4 com alternância
porseed; Python3.13.16/Torch2.11.0+cu130. Forward/gradientes do caminho bloqueado
qualificados nas formas desses testes; não todos os backends/precisões.
Sem traceback/OOM ou fallback registrado. Cache máximo amostrado167.970.718bytes,
sem violação da guarda, removido ao final. Metadados finais apenas: tensores da
campanha não fornecidos, não rehashados nem carregados localmente.

NLL média maior/pequeno/simples/pirâmide0,802563/1,128058/1,334464/1,329944.
Nenhuma comparação com as referências qualifica ganho linguístico. Blocos versus
referência dos mesmos pesos em8K: -71,91–72,23%VRAM incremental,1,53–1,66xmais
rápido, erro máximo de logits4,73e-5–1,31e-4. Isso é forward eager, não linguagem
8K, geração com cache ou custo para qualidade igual.

Auditoria executada por leitura/asserções sem treino ou inferência dos pesos finais.
Probe adicional CPU com pesos aleatórios/seed10032/dropout0: dois prefixos com
mesma contagem de triplas e sufixo comum dão diferença1,49e-7 nos logits finais
de ambos os candidatos. Ilustra a perda de ordem do agregado local; não é prova
de causa única da perda treinada. Nenhum código de treino foi alterado nesta auditoria.

## TinyStories — validação local anterior à campanha 0032

[Plano prévio](docs/PLAN_TINYSTORIES_IMPLEMENTATION.md),
[protocolo e limitações](docs/TINYSTORIES_RUN.md). Windows, Python 3.12.2,
PyTorch 2.14.1+cpu, `PYTHONPATH=src`. Sem CUDA local; os dois testes CUDA novos
não estavam aprovados nessa entrega. A qualificação real posterior está acima.

Execução bloqueada versus referência: forward/backward e estados, streaming,
causalidade em fronteiras e fases 0/1/3, pesos ligados e contagens. Preparação:
prefixos limitados e SHA fixados, registros completos, deduplicação entre divisões,
manifesto portátil, documentos isolados e caudas/padding/BOS contabilizados.
Teste reservado bloqueado no avaliador. Treinamento: finitude/memorização,
retomada exata com dropout 0,2 e acumulação 2, restauração dos pilotos e próxima
atualização, duas taxas train-only e inicialização principal nova.

Suíte ampliada final: **48 passed, 6 deselected, 81,24 s, exit 0**:
`python -m pytest -q tests/test_stories_models.py tests/test_stories_data.py tests/test_stories_training.py tests/test_stories_operations.py tests/test_stories_bootstrap.py tests/test_binding_models.py tests/test_training.py tests/test_research_configs.py -m 'not cuda' -k 'not full_cpu'`.
Inclui contagens/configs, rotação física, helpers fixados, relógio/Torch/dispatch,
falha/deadline/disco com exportação parcial e preservação de arquivo alheio.
O warning do wrapper de optimizer.step é do teste legado de equivalência do
benchmark; não altera a ordem real dos updates. Deselected inclui CUDA e a
integração completa exercitada separadamente abaixo.

Integração final de quatro workers CPU reais: **1 passed, 5 deselected, 22,80 s,
exit 0**, `python -m pytest -q tests/test_stories_operations.py -k full_cpu`.
Inclui calibração das duas taxas, novos treinos principais, checkpoints finais,
gerações, barreira/perfis, curvas PNG, verificação dos dois ZIPs e segunda
campanha que reutiliza os quatro jobs completos sem novo treino. As configurações
resolvidas correspondem à taxa realmente selecionada. Checkpoints intermediários
e estados descartáveis de calibração não permanecem como pesos duplicados.

Entrega da célula 10: **4 passed, 0,11 s, exit 0**,
`python -m pytest -q tests/test_stories_delivery.py`. SHA literal conferido contra
o bootstrap do Git `6f469fa6b80f2109b92c8b7be4a052c8c2ffaddd`; célula desativada
sem rede, payload alterado bloqueado, dispatch/seeds/relógio e callback de download
verificados por simulação. Nenhum treino foi iniciado na conta Kaggle nesta entrega.
Após publicação, download HTTP real do bootstrap fixado retornou 4.391 bytes com
SHA-256 exatamente igual ao da célula; links locais da entrega e diff conferidos.

Dados reais: download HTTP parcial total de 34 MiB na revisão fixada; seleção
produziu manifesto `d61615e79496f14ef473cc10ad11e5deac6d2bda89be7046d738498ff16ba849`,
16.777.421 bytes de treino/18.876 histórias e divisões distintas de calibração,
desenvolvimento e teste reservado. Uma segunda preparação em diretório novo,
partindo dos dois prefixos brutos locais, reproduziu exatamente esse SHA.
Contagens de três bytes: NLL 1,699787, bpb
2,452275 sobre todos os 524.508 bytes de desenvolvimento; 68.959.268 bytes de
estado. É referência estatística com orçamento próprio, não rede equivalente.

Smoke adicional em dados reais: 200 passos CPU, seed 11001, B8/T256, taxa 0,0003,
sem compilação e apenas treino/subdivisão de calibração. Memória simples:
NLL 5,59619 para 2,73839; Transformer pequeno: 5,54350 para 3,18167. Serve para
detectar falha de aprendizado/finitude e preparo, não ranking da campanha,
confirmação, custo GPU ou resultado do teste reservado. Esses pesos não serão
continuados nos oito treinos principais. Não houve treino de 6.000 passos local.

## Campanha0031 — qualificação real T4 e auditoria

[Relatório](docs/results/CAMPAIGN_0031.md): oito treinos novos/3000passos, duas
seeds311/312, statuscomplete. ZIP954.594bytes/129membros, CRC íntegro, sem caminhos
externos/duplicatas/pesos/pickle.366hashes de bootstrap e144arquivos de snapshot
conferidos contra Gitbcd7780 (101fontes/43configs); protocolo/gerador/configs,
parâmetros meta/fórmula, embedding ligado, workers/metadados/relatórios/CSV,
96avaliações e160registros de gradiente reconciliados. Regeneradas8192consultas
originais/alteradas para conferir labels; nenhum treino/inferência neural local.
Pesos reais ausentes: hashes declarados conferidos, não os tensores finais.

Preflight Kaggle **CPU13passed/2deselected,33,40s; CUDA2passed/13deselected,33,22s**,
GPU0. Os oito treinos/perfis usaram ambas asT4com atribuição alternada porseed.
Nova família qualificada emCUDA/Inductor nesse ambiente e formas testadas;
isso não aprova todos os backends/precisões/formas. Py3.13.16/Torch2.11.0+cu130.
Sem traceback/OOM; cache amostrado máximo155.209.556bytes, guarda sem violação.

Candidatos100%nas duas seeds, incluindo contrafactuais/capacidade/alinhamento.
Baseline não qualificou; o relatório corretamente mantémfalseas aceitações de
eficiência/compactação a referência qualificada. Sem confirmação de linguagem,
cinco seeds ou custo igualado. Código/treino não foram alterados nesta auditoria.

## Família associativa de transições — entrega posterior à0030

[Plano prévio](docs/PLAN_ASSOCIATIVE_TRANSITION_IMPLEMENTATION.md),
[operação](docs/ASSOCIATIVE_BINDING_RUN.md). Python3.12.2/Torch2.14.1+cpu, Windows,
`PYTHONPATH=src`. Nova arquitetura separada; modelos/protocolos antigos preservados.

- Kernel aditivo vetorizado versus referência sequencial, forward e gradientes;
  causalidade em fronteiras1/3/4/5/7/8/31/32; streaming e gradientes em chunks;
  contagens por fórmula e enumeração; estado real sem buffers de prefixo presos;
  embedding ligado; mesmo forward inicial entre os controles correspondentes.
- Generalização aprendida para novos mapas e chaves trocadas, nos três braços
  neurais novos. Teste de memorização de batch fixo não é a única qualificação.
- Retomada exata com dropout0,2/acumulação2 e próxima atualização bit a bit;
  sampler preserva fases4/8/16/32. Benchmark restaura pesos/otimizador/scheduler/
  RNG/sampler/contadores, sem alterar o horizonte ou guardar updates de piloto.
- Primeira rodada encontrou telemetria que não aceitava o inteiro da fase e
  benchmark que atingia o limite do Trainer. Corrigidos: metadados escalares
  excluídos dos bytes de tensores; updates explícitos de medição e restauração,
  sem ampliar max_steps. Rodada focada9passed/3deselected,5,36s.
- Suíte ampliada final: **85passed,12deselected,97,86s,exit0**. Comando:
  `python -m pytest -q tests/test_binding_models.py tests/test_binding_operations.py tests/test_recall_models.py tests/test_recall_bootstrap.py tests/test_state_recurrent.py tests/test_training.py tests/test_research_configs.py -m 'not cuda' -k 'not full_cpu'`.
  Inclui regressões de modelos/Trainer/configurações; CUDA não executado localmente.
- Integração real de quatro workers CPU, treino4passos, checkpoint/aposentadoria,
  curvas, contrafactuais, barreira de perfis, ZIPs e reuso nativo sem novo treino:
  primeira execução **1passed/4deselected,22,26s**. Após ajustes finais de telemetria
  e sondas, operações+bootstrap: **8passed,17,89s,exit0**, incluindo essa integração,
  falha de worker, espaço insuficiente, deadline/exportação parcial, rotação de
  GPUs, dispatch/relógio/Torch e botões. Nenhum arquivo alheio removido.

Última execução conjunta da nova família/operação/bootstrap, incluindo equivalência
do update medido com o Trainer: **21passed,2deselected,74,29s,exit0**.
Comando:`python -m pytest -q tests/test_binding_models.py tests/test_binding_operations.py tests/test_binding_bootstrap.py -m 'not cuda'`.
Os dois deselected são CUDA. O único warning vem do wrapper de optimizer.step
usado no teste de equivalência, não de mudança da ordem de updates na operação.
Entrega da célula09: **4passed,0,11s,exit0**; SHA literal contra Gitbcd7780,
execução desativada sem rede, payload alterado bloqueado antes deexec e despacho/
relógio/seeds/callback de download simulados. Após publicação, downloadHTTP real
conferiu4370bytes e SHA-256 exato da célula09. Links locais e diff conferidos;
nenhuma execução na conta Kaggle foi feita por esta entrega.

Qualificação de desenvolvimento adicional: seed816/1000passos reais no núcleo
86.785parâmetros, CPU/FP32/dropout0,05/contexto256/B16/LR0,001. Avaliações novas de
32sequências/512consultas por condição: original e par contrafactual100%em
256/1024/4096/8192com32associações,1024com64e1027com prefixo3. Registro e limites:
[qualificação](docs/results/transition_development_qualification.json). Medição
explícita apenas dos pesos com a versão equivalente de prefixo exclusivo;
fontes de treino/medição separadas, sem carregar pickle ou alegar retomada entre
fontes. Não houve controle Transformer nem medição GPU/energia/linguagem local.

Os dois testes novos CUDA/Inductor eram obrigatórios no Kaggle e foram aprovados
na0031acima. A qualificação CPU isolada não sustenta ganho de velocidade nem capacidade
geral de uma LLM. Histórico0030 e documentação abaixo permanecem como evidência
anterior, não resultado dessa nova campanha.

## Campanha0030 — auditoria da execução parcial T4

[Relatório](docs/results/CAMPAIGN_0030.md): ZIP1.076.272bytes/196membros, CRC íntegro,
sem caminhos externos/duplicatas/pesos/pickle. Conferidos345hashes do bootstrap e
138arquivos de snapshot contra Gitdc2888a,95fontes e43configs,24registros de worker,
identidade do protocolo/gerador, configs, contagens meta/fórmula, embedding ligado,
hash inicial igual nos dois bancos concluídos,40avaliações e80registros de gradiente.
Quatro treinos/seed271/3000passos completos; dez pendentes. Metadados reconciliados;
tensores finais ausentes, sem rehash de pesos/estado nem forward/treino científico
local. Testes Kaggle: **CPU14passed/2deselected,15,88s; CUDA2passed/14deselected,
78,10s**, GPU0. Duas T4 utilizadas nos treinos. Não é aprovação de qualidade.

Auditoria local reproduzível em `artifacts/audits/campaign_0030/audit.py`:
regeneração dos exemplos/labels/distâncias, oráculo simbólico100% nos quatro
comprimentos, preditor de frequência sem chave e alcance dos padrões calculados.
Esses controles não são modelos aprendidos nem novos resultados de linguagem.
Finitude/gradientes presentes não demonstram associação; o teste histórico de
memorização de batch fixo não qualificava generalização para novos pares.
Recomendações de rever mistura local/inicialização e reduzir recompilação não
foram implementadas nesta auditoria. Código de treino e célula fixada preservados.

## Memória consultável/Fibonacci — implementação e limites reais

[Plano prévio](docs/PLAN_MEMORY_RECALL_IMPLEMENTATION.md),
[operação](docs/MEMORY_RECALL_RUN.md). Python3.12.2/Torch2.14.1+cpu, Windows,
`PYTHONPATH=src`. Nenhum treino científico de recuperação na T4 feito localmente.

- Modelos/gerador/índices/causalidade/gradientes/contagens/finitude/memorização/
  retomada: primeira rodada corrigida11passed/2deselected,7,71s. Depois incluídos
  streaming com banco cheio, referência sequencial e contagem de custo de validação.
- Suíte focada e regressões de estados, controles0029, Trainer e configurações:
  **85passed,12deselected,74,07s,exit0**, excluindo integração longa e CUDA.
  Comando:`python -m pytest -q tests/test_recall_models.py tests/test_recall_operations.py tests/test_recall_bootstrap.py tests/test_state_recurrent.py tests/test_state_readout_models.py tests/test_training.py tests/test_research_configs.py -m 'not cuda' -k 'not full_cpu_engineering'`.
- Integração real CPU:`python -m pytest -q tests/test_recall_operations.py -k full_cpu`:
  **1passed,6deselected,245,35s,exit0**. Sete treinosT32/4passos,14pilotos frios/
  aquecidos, piloto concorrente, sete avaliações/perfis posteriores isolados,
 14avaliações de comprimento e14sondas de capacidade. ZIPs reais conferidos por
  hashes/manifestos/cobertura de checkpoints. Só último completo por job, fonte
  única, sem after-NN/pesos no ZIP de resultados.
- A integração também executa mais duas campanhas: histórico completo reutilizado
  sem treinar e avaliação de treino concluído recuperada após interrupção simulada,
  preservando pesos/contador. Banco completo e três padrões têm mesmos parâmetros
  iniciais. Guarda de fonte/dados/configuração/ambiente permanece ativa.
- Falhas injetadas de worker, deadline e espaço insuficiente exportam pares ZIP
  parciais válidos, preservando arquivos alheios. Admissão mantém pares, horizontes
  e schedulers. Caminho procedural agora absoluto e roundtrip YAML idêntico, após
  a integração detectar resolução relativa inconsistente. Manifesto identifica
  fingerprint procedural para verificar o par de arquivos; não enfraqueceu a
  verificação existente. Campo de progresso/modelo/GPU também conferido.
- Retomada compara próxima atualização bit a bit com dropout0,2 e acumulação2,
  incluindo pesos, otimizador, scheduler e sampler NumPy. Mudança de padrão é
  incompatível. Testes de distribuição não recebem os rótulos como input.
- Bootstrap simulado confere relógio desde célula, preservação do Torch, incremento
  de número reservado, seeds e downloads. Helpers reais conferidos contra Git1c8ce2a.
  O bootstrap não repete download/preparo/qualificação de WikiText.

Último ajuste apenas de contabilização: validação do Transformer acumula seu
custo forward estimado, em vez de deixar zero. Teste específico acrescentado;
não altera forward, loss, otimização ou gerador. Estimativas de backbones de
estado continuam null. Testes novos focados foram reexecutados após o ajuste;
o resultado exato dessa execução consta na seção de entrega abaixo.

CUDA/Inductor/energia reais desta arquitetura permanecem **não verificados
localmente**. Há dois testes CUDA obrigatórios no Kaggle antes dos pilotos;
marcados como deselected nos comandos CPU, nunca aprovados por suposição.
Perfis serão de forward batch1/T1024, não geração/KV ou comparação iso-compute.
Dois seeds e uma tarefa sintética não estabelecem superioridade geral como LLM.
Nenhum resultado dos smokes de4passos é evidência de qualidade ou velocidade.

Entrega da memória: após a correção da contabilização de validação, suíte nova
focada reexecutada: **24passed,3deselected,11,22s,exit0**. Comando:
`python -m pytest -q tests/test_recall_models.py tests/test_recall_operations.py tests/test_recall_bootstrap.py -m 'not cuda' -k 'not full_cpu_engineering'`.
Os3deselected são os2CUDA e a integração longa já executada separadamente.
Sintaxe dos módulos novos e links locais conferidos; `git diff --check` limpo.
Célula08 fixada emdc2888a: **4passed,0,15s,exit0**. Hash literal conferido contra
Git, execução desativada sem rede, bootstrap alterado bloqueado e despacho/relógio/
callback de download simulados. Nenhuma execução na conta Kaggle foi feita localmente.

## Campanha0029 — evidência recebida do Kaggle

[Auditoria](docs/results/CAMPAIGN_0029.md): ZIP íntegro,234membros/4.303.546bytes.
Conferidos325hashes do checkout contra Git77b2058,132arquivos do snapshot, protocolo,
configurações, parâmetros únicos/estados por meta/fórmula, cinco inicializações
novas e igualdade dos dois hashes iniciais da hierarquia. Cinco treinos completos
6000passos/seed251, dez avaliações completas e dez de subconjunto; métricas/perdas/
posições/relatório reconciliados. Bins não fornecidos; hashes declarados iguais ao
corpus fixado, tokenizer rehashado. Pesos reais não fornecidos: somente metadados,
hashes e retenção final conferidos, sem carregar pickle ou realizar avaliação local.

Preflights reais registrados no bundle: CPU11passed/2deselected,15,59s; CUDA
2passed/11deselected,84,57s naGPU0. Ambos os slots T4 exercitaram treinos e pilotos
FP32/Inductor. Avisos de depreciação, sem traceback/OOM. Teste reservado intacto.
Perfis finais isolados30amostras/uma GPU por braço; energia NVML de placa em30s,
host/compilação excluídos. Sem geração, recall, crossover de placas, IC entre seeds
ou custo igualado. Inferioridade local da hierarquia não prova falha geral fractal.

Auditoria local reexecutável em `artifacts/audits/campaign_0029/audit.py`, sem
extração de arquivos nem avaliação de modelos. Exit0; gerou JSON/CSV e figura
inspecionada visualmente. Nenhuma fórmula de treino ou configuração foi alterada
nesta auditoria; testes de implementação já aprovados não foram repetidos.

## Campanha0029 — implementação dos controles de profundidade/leitura

[Plano prévio](docs/PLAN_STATE_READOUT.md), [operação](docs/STATE_READOUT_RUN.md).
Python local3.12.2/Torch2.14.1+cpu, Windows, `PYTHONPATH=src`. Fonte nova,
nenhum treino científico iniciado localmente e nenhum checkpoint antigo migrado.

- Modelos novos, primeira rodada CPU:11passed/2deselected,10,32s.
- Operação/retenção/estimativas/histórico:6passed/1deselected,2,56s.
- Integração real:1passed/6deselected,116,56s; a captura do teste em Windows
  emitiu aviso de decodingCP1252. Corrigida a captura paraUTF-8 explícito.
- Regressões antigas + controles/operador/bootstrap novos:
  **77passed,8deselected,132,19s,exit0**, sem o aviso de captura. Comando:
  `python -m pytest -q tests/test_state_recurrent.py tests/test_state_study.py tests/test_state_readout_models.py tests/test_state_readout_operations.py tests/test_state_readout_bootstrap.py -m 'not cuda'`.
- Retomada também exercitada diretamente no ReadoutTrainer/DocumentBatchStream,
  dropout ativo e acumulação2: **2passed,11deselected,4,53s**, próxima atualização,
  pesos/otimizador/scheduler/sampler iguais à execução ininterrupta. Leitura
  incompatível bloqueada. A nova validação por fase não altera RNG/treino.
- Preflight portátil: fixture histórica7203bytes conferida contra Git1c8ce2a e
  referência0027. Não depende de objeto histórico num clone shallow. CPU após
  essa adaptação: **11passed,2deselected,6,48s**; acesso Git histórico simulado como
  proibido nos testes de equivalência do default.
- Célula07 fixada em77b2058: **4passed,0,13s,exit0**. SHA literal verificado contra
  Git; payload alterado bloqueado, execução desativada sem rede e despacho/download
  simulados. Não executa conta Kaggle local.

Integração executa cinco treinos CPU/T17/4passos, dez pilotos frios/aquecidos,
piloto concorrente, cinco perfis isolados e exportação real. Verifica20avaliações,
contagens por fase, inicialização igual nas hierarquias, apenas checkpoint final,
metadados intermediários preservados, par ZIP com cinco estados completos e uma
fonte canônica. Segunda execução reutiliza os cinco jobs completos sem treinar.
É smoke de engenharia, não resultado de linguagem ou velocidade.

Falhas injetadas de worker e falta de espaço exportam par parcial íntegro e
preservam os dados originais. Testes de admissão mantêm pares/horizontes; fonte,
dados, configuração ou versões divergentes no histórico são bloqueados. ZIPs
antigos não são varridos. Recibo de erro de preflight sobrevive à limpeza do
checkout. Helpers de clone/limpeza conferidos contra bytes Git fixados.

Defaults/forward/contagens de parâmetros conferem com a implementação histórica.
Novos caminhos testados em causalidade, fronteiras, streaming forward+gradientes,
finitude, memorização, embedding compartilhado e isolamento. Fonte nova usa versão2
nos controles novos, sem enfraquecer contratos de checkpoint. Limpeza atua apenas
em intermediários redundantes da execução nova e caches próprios de lane inativa.

Limitações: nenhum CUDA/BF16/Inductor/A100 novo localmente. Os8deselected acima
são testes CUDA, não aprovação GPU. A célula executará dois preflights CUDA novos
naGPU0 e pilotos reais nas duas T4. Não repetir qualificação das três famílias nem
reescrever relatório0025. Perfis em uma GPU por variante não confirmam eficiência
nas duas GPUs. Suite completa fora desse conjunto não repetida. Uma seed e LR
fixa são triagem; confirmação continua exigindo seeds novas/custo igualado.

## Auditoria da campanha0028

[Relatório](docs/results/CAMPAIGN_0028.md),
[dados conferidos](docs/results/campaign_0028_audit.json). Somente análise/documentação.

- ZIP419.492bytes,14membros, CRC íntegro, sem duplicação/caminhos externos ou
  payloads de pesos/pickle/datasets. SHA registrado; dois snapshots verificados.
- 80fontes/43configurações científicas contra Git1c8ce2a;5arquivos operacionais
  contra Git55dec8e. Referência idêntica à entregue. Manifesto/fingerprint igual0027.
- Quatro originais reproduzidos; erro máximo2,6091e-9. Onze condições têm as mesmas
  1087janelas/136batches/1.113.088alvos. Zero optimizer-steps/teste reservado.
- Perdas/deltas/bits por byte e totais por fase/quarto da janela recalculados.
  Somatórios, RMS/cossenos e contagens consistentes entre JSONs/CSV/campanha.
- Logs lidos integralmente: sem traceback/OOM/timeout; máscaras0/1 comprovam
  uso das duas GPUs nos quatro workers. Dados/pesos nativos reutilizados.
- Auditoria local exit0, sem carregar/avaliar modelos. Figura gerada com apenas
  valores auditados e verificada visualmente. Suite de treino não repetida, pois
  nenhum modelo, contrato ou treinamento foi modificado.
- Análise do índice original, conferido por SHA:2244casos de offsets0–3 e
  comprimentos1–33 verificam que ler o grupo recém-fechado usaria apenas bytes
  observados. É análise matemática, não teste de uma implementação futura.

Uma seed e intervenções pós-treino, sem IC entre seeds/ablação treinada. Contraste
por posição não demonstra causa nem ganho de uma mudança de leitura. Runtime
observado não foi exportado como inventário separado; o código fixado compara
versões com a referência. ZIP sem tensores para rehash direto; durações/VRAM são
diagnósticos concorrentes, sem alegação de eficiência de treino/geração. Relógio
antes de exportação final; sem inventário remoto completo ou recibo de limpeza.

## Diagnóstico congelado da0027 — implementação e entrega

[Escopo/riscos/aceitação prévios](docs/PLAN_STATE_DIAGNOSTIC.md),
[instruções](docs/STATE_DIAGNOSTIC_RUN.md). Modelo/treino científicos originais
permanecem1c8ce2a; nova operação é executada como arquivo separado.

- `python -m pytest -q tests/test_state_diagnostic.py tests/test_state_diagnostic_bootstrap.py`:
  **18passed,35,36s,exit0**, no Python3.12.2/Torch2.14.1+cpu local, `PYTHONPATH=src`.
- Célula06 fixada em55dec8e: **4passed,0,22s,exit0**. SHA literal conferido
  contra o objeto Git; execução desativada não acessa rede, payload alterado é
  rejeitado e despacho/download são simulados. Não executa conta Kaggle local.
- Integração extrai Git archive1c8ce2a para diretório novo, fabrica quatro pesos
  CPU/T17 e dados por documento, depois executa o novo operador contra esse pacote
  original em subprocessos. Quatro jobs/11condições, zero optimizer-steps, NLL
  original reproduzida e um ZIP sem pesos/pickle. Isso é smoke de engenharia, não
  resultado de linguagem ou equivalência CUDA dos checkpoints reais.
- Falha de reprodução injetada exporta ZIP parcial/JSON não reproduzido e mantém
  pesos originais. Testes cobrem causalidade das intervenções, reconstrução da
  hierarquia, perdas/contagens por posição, RNG/modos/pesos inalterados, hooks
  removidos em falha, deadline/cobertura, hash/config/fontes/versões e caminhos.
- Restauração inclui somente safetensors/metadados, sem desserializar pickle.
  Limpeza retira somente os dois arquivos próprios, sem exclusão recursiva;
  origem/backup e alvos externos preservados. Links/ancestrais vinculados bloqueados.
- Bootstrap simulado verifica pesos antes de instalar, mantém `PYTHONPATH` antigo,
  recusa fontes divergentes e passa o orçamento incluindo setup. Download embutido
  reutiliza o helper científico já fixado. Não há compilação ou benchmark novo.
- Backup real localizado:92.880.997bytes,23membros, CRC íntegro, SHA-256
  `2b4724f97c012876334d0776fb89ee50d564a17566328563603c630f9bebc076`.
  Os quatro hashes de pesos/metadados reais conferem com a referência. Cabeçalhos
  safetensors enumeram2.174.232/2.174.232/1.136.652/2.164.992parâmetros únicos.
  Cópias temporárias dessa conferência foram removidas; ZIP original preservado.

Limitações: nenhuma avaliação científica dos pesos reais localmente, nenhum teste
CUDA novo, nenhuma velocidade T4 alegada. A execução Kaggle deverá repetir a NLL
original em5e-5 antes de interpretar intervenções. Suite completa não repetida;
nenhuma alteração de matemática, treino ou contratos de retomada.

## Auditoria da campanha 0027

[Relatório](docs/results/CAMPAIGN_0027.md),
[conferência estruturada](docs/results/campaign_0027_audit.json).
Somente análise/documentação; sem alteração de modelo, worker, contrato ou células.

- ZIP12.342.536bytes,363membros, CRC válido, sem duplicação/caminhos externos.
- 80fontes +43configurações contra Git1c8ce2a;30snapshots com123arquivos cada;
  quatro arquivos e snapshot operacionais contra Git6396959. Qualificação0025
  preservada byte a byte; mapa operacional igual em bootstrap/campanha/relatório.
- Protocol ID recalculado; quatro configurações validadas, parâmetros por fórmula
  e enumeração meta conferidos. Quatro treinos6000passos reiniciados do zero;
  49.152.000alvos por braço; metadados3000/6000, métricas e contadores consistentes.
- Oito avaliações completas e oito em subconjunto; NLL/bits por byte/perplexidade
  conferidos entre métricas/workers/relatório. Oito tentativas LR; seleção recalculada
  e divisão interna1408/175/175documentos verificada disjunta dos oficiais.
- Oito perfis isolados,30amostras positivas/finitas cada: medianas/médias e energia
  por alvo recalculadas; contraste/decisão refeitos por `state_report.analyze`.
- Logs Kaggle: CPU51passed/6deselected32,17s; CUDA2passed/40deselected72,63s
  naGPU0. Duas T4 executaram treino/perfis. Sem traceback/OOM/erro de compilação
  nos logs lidos. Nenhum modelo consultou teste reservado.
- Auditoria local executada com sucesso (exit0), sem carregar pickle/pesos,
  treinar ou avaliar modelos. Figura original conferida visualmente; conecta só
  avaliações completas. Uma seed241; texto genérico “Three seeds” não é evidência
  de três réplicas. Pesos e binários preparados ausentes: apenas hashes declarados
  e consistência de metadados conferidos, sem rehash de tensores/dados ausentes.

Nenhum teste GPU local novo. Suíte de treino não repetida por alteração exclusiva
de análise/documentação. Sem IC entre seeds, medição de geração/recuperação ou
custo igualado. Duração registrada antes da exportação final; lista de snapshots
intermediários não prova sua remoção remota posterior.

## Recuperação do cache após falha0026

[Plano prévio](docs/PLAN_STATE_B_CACHE_RECOVERY.md),
[conferência real do corpus](docs/results/STATE_B_CACHE_RECOVERY.json).
O log fornecido contém CPU51passed/6deselected (36,15s) e CUDA2passed/40deselected
(78,88s); falhou no checksum de shard existente, antes dos pilotos/treinos.
Não houve NLL nem checkpoint. Sem arquivo atual do Kaggle, causa/hash observado
da corrupção não são conhecidos. A guarda de integridade funcionou.

- Novo preflight apenas operacional; worker científico e hashes continuam1c8ce2a.
- Testes de cache/bootstrap:19passed/1deselected,5,10s; após testar falha anterior
  a instalação/CUDA:20passed/1deselected,4,67s; versão com recibo persistente:
  **21passed,1deselected,4,73s**,exit0.
- Regressões de importação WikiText/contrato: **3passed,25deselected,0,56s**,exit0.
- Células por revisão/hash após fixar6396959: **10passed,0,49s**,exit0;
  fonte conferida contra Git e despacho simulado, sem iniciar conta Kaggle.
- Quatro shards reais do backup local foram lidos e extraídos em diretório novo
  de artifacts, sem modificar o backup. Todos conferem com SHA originais.
- Corpus preparado novamente a partir desses shards; `verify_dataset` aceitou
  os quatro hashes originais (train/val/test/tokenizer), sem alteração de contrato.
  Fingerprint:f10ae5a31fb066205e9da284726f5f0b756cf2e2e68bd7cb15caf0ca17018c81.
- Download real do shard de validação fixado:657.209bytes, SHA original conferido;
  esse teste de transporte não avalia um modelo no conjunto de validação/teste.
- Fixtures provam preservação do legado inválido, cópia independente de válidos,
  rejeição de HTML/hash incorreto, falha de rede, limpeza de temporário em erro
  normal, publicação concorrente sem sobrescrita, links, limite de transferência
  e parser AST que não executa o código de origem. Recibo de falha sobrevive à
  exclusão do checkout. Preflight que falha impede instalação e despacho do worker.

Não repetido treino/integrador de5min: esta alteração não muda matemática,
otimizador, sampler ou passos; modifica transporte/verificação prévia de arquivos.
Nenhuma nova execução GPU local. Os testes CUDA acima são os do log do usuário,
não aprovação da nova rotina de cache no Kaggle. Suite completa não repetida.

## Entrega da etapa B — ciência qualificada e operação separadas

Plano prévio: [escopo/riscos/aceitação](docs/PLAN_STATE_B_DELIVERY.md).
[Operação](docs/STATE_B_RUN.md). A fonte científica permanece fixada em1c8ce2a;
novos arquivos operacionais não substituem o validador científico nem seus hashes.
Python local3.12.2, Torch2.14.1+cpu, Windows, instalação direta, `PYTHONPATH=src`.

| Verificação executada | Resultado |
|---|---|
| Primeiros testes operacionais, sem integração | 9passed,1deselected,5,09s |
| Operação B + núcleo/protocolo/utilitários, com integração real | **76passed,6skipped,306,46s**,exit0 |
| Após acrescentar somente campos/print de previsão e seeds restantes | **10passed,1deselected,5,19s**,exit0; integração não repetida após esse acréscimo de metadados |
| Células B/A/limpeza por revisão e hash, após entrega fixada | **10passed,0,55s**,exit0; validação contra objetos Git e despacho simulado sem conta Kaggle |

A integração extrai o Git archive da revisão científica1c8ce2a e executa o novo
operador como arquivo em subprocesso com `PYTHONPATH` do worker antigo. Executou
quatro jobs mLSTM/Transformer em duas lanes CPU, oito tentativas de taxa, oito
pilotos isolados, piloto concorrente e oito perfis finais. Todos os checkpoints
e a campanha possuem exatamente os hashes da fonte científica qualificada da0025.
Foi verificado `operator-source.zip`, identificação operacional no relatório,
curvas, par ZIP definitivo e ausência dos ZIPs temporários após cobertura validada.
Esse smoke tem4passos/T17; não é resultado de linguagem nem evidência de desempenho.

Testes focados cobrem: família mantida entre comandos, troca de variante,
impossibilidade de limpar GPU0 em uso, preservação de GPU1 e arquivos científicos,
liberação de lock em falha, reserva de pós-perfis incluindo compilação fria,
rejeição de tuning incompatível, importação da fonte errada e fonte de bootstrap
alterada. O bootstrap simulado instala somente o worker científico e passa seu
`PYTHONPATH`; SHA do relatório original da0025 e fontes Git foram conferidos.
Os testes de núcleo/protocolo incluem causalidade, forward/backward, finitude,
memorização, compartilhamento e retomada com dropout/acumulação.

Limitações: nenhuma nova execução CUDA/Inductor local. A0025 qualifica a regra
recorrente na T4, não toda a hierarquia nem esta operação B. Os preflights originais
na T4 continuam obrigatórios. Manter cache não garante cache hit; warmup, tempo
de comando e tempo total são medidos. Pilotos/estimativas podem recusar um bloco;
não há garantia de conclusão em2h. Sem alteração de horizonte, scheduler ou dataset.
Suíte completa não repetida; verificações focadas e integração reais acima.

## Auditoria da campanha 0025 — 31/10/2026

[Relatório](docs/results/CAMPAIGN_0025.md),
[dados estruturados](docs/results/campaign_0025_audit.json). Sem alteração de código,
pesos, treinamento ou contratos científicos; somente análise/documentação.

- ZIP75.152bytes,21membros, CRC íntegro; SHA-256 registrado.
- 123 hashes de fontes/configurações conferem com objetos Git da revisão1c8ce2a;
  fonte entre campanha/seleção coincide. Seleção recalculada: `mlstm`.
- Quatro pilotos reais CUDA/Inductor aprovados no Kaggle (RWKV e mLSTM, duas T4):
  forward/gradientes, finitude, causalidade e streaming registrados. Duas execuções
  Gated Delta foram encerradas pelo supervisor em600,42s, sem falsa aprovação.
- 30amostras por piloto aprovado: valores finitos/positivos e medianas conferidas.
  Logs lidos integralmente; sem timeout absorvido/traceback/codegen error/fallback.
- Testes CPU registrados no Kaggle:51passed,6deselected,36,53s. Isso não equivale
  à execução de toda a suíte CUDA de hierarquia, que permanece pendente para B.
- Cache: pico amostrado927,95MiB,4,696GiB acumulados removidos, limite não excedido.
  Checkout shallow3,39MiB; política de remoção declarada, sem recibo posterior no ZIP.
- Tempo registrado55,17min. Nenhum treino científico, NLL ou teste reservado.

Não foram repetidos testes locais de treinamento: não houve alteração de código.
Nenhuma nova GPU foi usada localmente. A fonte científica qualificada permanece
fixada e não foi modificada para registrar esta auditoria.

## Campanha 0024, download e armazenamento — 31/10/2026

Plano prévio: [auditoria/armazenamento](docs/PLAN_CAMPAIGN_0024_STORAGE.md).
[Conclusão científica](docs/results/CAMPAIGN_0024.md). Alterações somente na branch
experimental. Python local3.12.2/Torch2.14.1+cpu; comandos com `PYTHONPATH=src`.

- ZIP recebido: CRC íntegro,22membros,87.636bytes; SHA-256 registrado na auditoria.
  Os120 hashes de fontes/configurações conferem com o commit científico registrado.
- Seis logs T4 auditados: todos registram o alarme absorvido; primeiros passos
 707–968s excedem600s. Tempos de mLSTM são evidência operacional provisória.
  Nenhum resultado de qualidade ou ganho fractal foi inferido.
- Primeira rodada focada/regressão:125passed,9skipped,2failed,98,88s. As falhas
  estavam em fixtures novas (tamanho de `b'cache'` e sessão menor que a reserva);
  corrigidas, sem enfraquecer verificações do produto.
- Núcleo, protocolo, armazenamento/download e integração real em dois processos:
  **64passed,6skipped,242,46s**,exit0. Inclui causalidade, forward/backward,
  finitude, memorização, retomada com dropout/acumulação, triagem8tentativas,
  quatro jobs, perfis e par de ZIPs finais sem acúmulo dos intermediários.
- Regressões de Campaign/Session/bootstrap/ciclos2–4 e utilitários novos:
  **115passed,9skipped,92,23s**,exit0, após o teste de download em erro parcial e
  a distinção entre deadline global e timeout de candidato. Sem CUDA local.
- O supervisor mata um worker real que envolve o trabalho em `except BaseException`;
  o próximo candidato consegue iniciar. Seleção rejeita registros sem supervisor,
  com tempo excedido e contratos antigos. Falha de candidato não aprova seu caso.
- Download: bytes recuperados do HTML são idênticos ao arquivo; partes reconstroem
  exatamente o original, com hashes, sem cópias em disco. Callback de download em
  falha parcial do worker é testado por simulação; navegador novo ainda não testado
  localmente. Usuário confirmou o método base64 no Kaggle na mensagem desta entrega.
- Limpeza: preserva pesos e ZIPs; bloqueia workers, arquivos ignorados desconhecidos,
  checkout modificado e alvos externos; links internos não apagam destinos externos.
  Git shallow com fetch de SHA exato validado contra repositório local de fixture.
- Limite de cache testado com arquivos pequenos: aborta sem excluir ciência.
  Não é um limite instantâneo de bytes: amostra5s, podendo ultrapassar entre amostras.
- Após proteger também commits locais não publicados: utilitários finais
  **14passed,4,28s**,exit0. Integração completa não foi repetida após essa guarda
  adicional de limpeza; matemática/treino permanecem inalterados.
- Células de entrega fixadas na revisão1c8ce2a: **6passed,0,25s**,exit0;
  hashes conferidos contra objetos Git, execução simulada e rejeição de fonte
  alterada. Não executam limpeza real nem acessam a conta Kaggle nesses testes.

Nenhum CUDA/Inductor novo executado nesta máquina. A0024 valida somente a execução
antiga e revela sua falha de orçamento; não é aprovação do código novo. O encerramento
de grupos POSIX precisa de verificação no Kaggle; a máquina de testes é Windows.
Suíte completa não repetida nesta alteração; integração e regressões focadas são
registradas acima. Horizontes de treinamento e matemática dos backbones não mudaram.

## Arquitetura recorrente/hierárquica inspirada em fractais — 31/10/2026

Plano prévio: [plano aprovado](docs/PLAN_RECURRENT_FRACTAL.md). Código somente no
checkout `experiment/fractal-lrm`. Python local explícito3.12.2, Torch2.14.1+cpu,
Windows; nenhum venv, nenhum treino científico, nenhum teste T4 novo executado aqui.

| Verificação realmente executada | Resultado |
|---|---|
| Primeira implementação das regras/backbones | 33passed,21,31s |
| Regras, amostragem por documento, budgets e seleção | 44passed,6skipped,41,57s; depois inicialização pareada44passed,6skipped,22,05s |
| Primeira integração real em dois processos CPU | 1passed,173,49s; triagem8tentativas,4jobs,8perfis e exportação |
| `python -m pytest -q`, suíte completa | **504passed,46skipped,553,39s**,exit0, antes do ajuste final de storage compacto/curvas |
| Após compactar caches e exportar curvas: três arquivos de testes novos | **51passed,6skipped,297,06s**,exit0; inclui integração completa e remoção dos ZIPs temporários verificados |
| Após isolar falhas de compilação por candidato: núcleo e protocolo | **51passed,6skipped,25,37s**,exit0; falha de um candidato não impede os outros em fixture explicitamente simulada |
| Contexto real1024, batch1, três regras compartilhadas | Forward/backward finitos, logits(1,1024,257), parâmetros1.136.652/1.132.800/1.136.652; CPU, sem ranking de velocidade |
| Parâmetros de produção por fórmula/enumeração em meta | Controles independentes2.174.232(GDN/mLSTM),2.166.528(RWKV), Transformer2.164.992; sharing menor sem alargar |
| Figura de curvas | Inspecionada visualmente; somente validações completas ligadas; rótulo ENGINEERING SMOKE renderizado após ajuste textual final |
| Links Markdown locais / whitespace | 99destinos válidos,0quebrados; git diff --check sem erros |
| Célula curta de entrega, depois de fixar fonte Git | **4passed,0,23s**; hash confere com objeto Git, modo inativo sem rede, rejeição de fonte divergente e dispatch simulado |

As46/6marcações puladas não representam suporte CUDA. Testes novos cobrem referências
sequenciais independentes contra chunks (outputs/estado/gradientes), causalidade nas
fronteiras, streaming, memória FP32, finitude, memorização, safetensors ligado, fórmula
sem duplicação, igualdade inicial dos controles, e próxima atualização retomada
exatamente igual com dropout0,2/acumulação2. Cache compacto tem teste de storage real,
evitando views que retenham todo o prefixo. Fixtures não são resultados científicos.

Nova avaliação usa janelas contidas em documentos, exclui EOS da byte-NLL e descarta
caudas incompletas igualmente. Portanto não comparar diretamente NLL com política
packed histórica. Tuning usa somente documentos do treino original, com conferência
de hashes; teste oficial não é avaliado. Fonte/dataset/versões não foram relaxados.

Limites explícitos: são três regras de memória num shell comum, não reproduções
dos modelos completos dos artigos. Seleção CUDA e aceleração ainda dependem da T4.
FLOPs dos scans são null; fórmula Transformer antiga não foi aplicada a eles.
Energia GPU quando disponível não representa host/campanha inteira. Geração com
cache comparável fica null/pendente. MC/TTT/ACT/DEQ/precisão reduzida/TinyStories/
escala7–10M não foram implementados como mecanismos ativos: são etapas condicionais.

Suíte completa não foi repetida após os pequenos ajustes finais citados; verificações
focadas/integradas acima correspondem às alterações. Não inferir validação GPU de
testes CPU, fixtures de plataforma ou logs de campanhas antigas.

## Auditoria 0023 e controle/retenção ciclo 5 — 31/10/2026

Plano prévio: [auditoria/retenção/ciclo 5](docs/PLAN_CAMPAIGN_0023_RETENTION_CYCLE5.md).
[Resultados 0023](docs/results/CAMPAIGN_0023.md), [operação nova](docs/CYCLE5_RUN.md).
Auditor executado com `PYTHONPATH=src` e Python explícito3.12.2. A primeira chamada
sem PYTHONPATH selecionou a instalação editável antiga e não encontrou o módulo;
foi corrigida a seleção do checkout, sem alterar verificação de dados ou código.

| Verificação real | Resultado |
|---|---|
| Dois ZIPs locais recebidos | CRC íntegro, membros únicos; results31.823.295bytes / checkpoints464.552.147bytes |
| Fonte e snapshots 0023 | 232 arquivos contra Git6a6af1a; 44snapshots /4.576membros consistentes |
| NLL reconstruída | 72 principais +24probes; 4.469janelas /1.144.064alvos cada; erro máximo7,01692e−9 |
| Estado final | 36 hashes safetensors/estado de18jobs conferidos por streaming, sem pickle |
| Diagnósticos | 18arquivos,7.119floats finitos; não prova de raciocínio/eficiência |
| Novo validador de bundles no material real | Cobertura integral dos18jobs; SHA dos definitivos idêntico às cópias locais |
| Primeira seleção de storage/profundidade/operação | 36passed,1skipped,11deselected,19,69s |
| Integrações reais CPU em dois processos (C5novo e C4compatível) | 2passed,47,18s; metade/final e formato legado cobertos |
| `python -m pytest -q`, executável explícito3.12.2 | **455passed,40skipped**,437,41s,exit0 |
| Após endurecer apenas a entrada Git do auditor | 1passed,0,44s; metadado ZIP com opção Git rejeitado antes de qualquer comando |
| Links Markdown e whitespace | 145 destinos locais válidos; `git diff --check` sem erros nos dois checkouts |

Suíte completa inclui causalidade, forward/backward finito, memorização e retomada
exata CPU/math com dropout0,2/acumulação2 dos quatro braços K1/K2/K4/Transformer;
parâmetros/embedding ligado/fórmula e horizontes-FLOPs pré-fixados. Inclui dois
processos reais por sessão de engenharia, publicação aos50% em jobs, remoção só
após definitivos íntegros, preservação em deadline, conteúdo único, hashes errados,
ZIP quebrado, estado incompleto, worker, symlink, confinamento e proteção dos finais.
Células ativas testadas com fontes/bootstrap simulados, números/hashes/relógio
conferidos; inativas sem rede. Backup anterior fora do Git por novo ignore explícito.

Os40skips não validam CUDA nem comportamento específico de Linux.
Este ambiente tem Torch2.14.1+cpu/Windows, sem medição GPU nova. Os logs recebidos
da0023 registram64CPU/Linux e6CUDA/Inductor aprovados em Torch2.11.0+cu128, Python
3.13.15, duasT4. Evidência remota anterior não valida CUDA dos novos controles;
o próximo preflight executará os quatro braços no hardware real.

Não houve treino científico nem limpeza remota pelo agente. Somente fixtures
temporárias foram removidas; os dois arquivos originais da0023 permanecem locais.
Manifesto/tokenizador e hashes declarados do dataset foram conferidos, sem rehash
dos bins ausentes nos bundles. Fonte de treino0023 permanece congelada em6a6af1a;
nova operação tem protocolo/revisão próprios e não relaxa retomada científica.

## Reconciliação do inventário na substituição — 30/10/2026

O usuário recebeu bloqueio após cerca de 60 s; a mensagem antiga não identificava
as diferenças. Não foi inspecionado o inventário remoto e não se atribuiu uma
causa específica. O comparador exigia também igualdade de nomes e mtime,
bloqueando adições e timestamps. [Plano prévio](docs/PLAN_BACKUP_RECONCILIATION.md).

Agora lista categorias/caminhos, admite adições e compara identidade por
tipo/tamanho/SHA (ou destino do link). Payloads selecionados antigos que estão
ausentes/diferentes são recuperados para pasta exclusiva, sem sobrescrita, com
SHA do destino conferido. ZIP antigo só sai após preservação, espaço, verificação
de CRC/SHA, ausência de workers e rechecagem do conteúdo corrente. Inventário e
mapa antigos ficam junto das versões recuperadas, incluídos no backup novo.

Com o mesmo Python 3.12.2 local, sem alterações no treino:

```text
python -m pytest -q tests/test_kaggle_reset.py tests/test_backup_delivery.py tests/test_cycle4_operations.py -k "not real_two_process and not weights_profile and not early_stop"
53 passed, 1 skipped, 6 deselected in 4.24s
```

Inclui adições/timestamps, ausência/alteração com preservação de ambas as versões,
arquivo substituído por link externo sem seguir o destino, espaço de recuperação,
falha de escrita/hash de destino, mudança concorrente, falha posterior de compressão
e repetição com versões antigas ainda presentes. CRC inválido detectado durante a
leitura SHA, sem segunda passagem `testzip`. Conferido no código da stdlib local
que `ZipExtFile` verifica CRC ao alcançar EOF. Reuso/limpeza ignoram somente mtime;
bytes/caminhos novos ou alterados ainda bloqueiam e mostram diferenças.

Skip: processo POSIX; seis integrações com treino/probes/exportação não selecionadas.
Somente fixtures locais foram removidas. Sem execução remota, limpeza de dados do
usuário ou medição de tempo/tamanho no Kaggle. A causa concreta aparecerá no diff
impresso pelo script atualizado. Segunda célula científica permanece em `6a6af1a`.

## Compressão e substituição do backup — 30/10/2026

Plano: [PLAN_BACKUP_COMPRESSION.md](docs/PLAN_BACKUP_COMPRESSION.md). O usuário
relatou ZIP de quase 8 GB e download sem resposta. A inspeção confirmou ZIP_STORED
na primeira entrega; isso não diagnostica sozinho a causa do download. O padrão
passou a ZIP DEFLATE nível 9, com heartbeat e link relativo. Substituição exige
nome exato, ZIP íntegro, originais iguais ao inventário, workers ausentes e espaço
previsto suficiente. Nenhum original é excluído nessa etapa. A limpeza completa
continua exigindo o SHA-256 calculado na cópia local do novo backup.

Verificação real, Python 3.12.2/Windows, mesma instalação local:

```text
python -m pytest -q tests/test_kaggle_reset.py tests/test_backup_delivery.py tests/test_cycle4_operations.py -k "not real_two_process and not weights_profile and not early_stop"
42 passed, 1 skipped, 6 deselected in 2.07s
```

O skip é POSIX; seis testes de integração com treino/probes/exportação ficaram
fora desta seleção, pois não foram alterados. Casos novos cobrem layout legado
sem campo de compressão; compressão efetiva de bytes repetitivos; DEFLATE/LZMA
com CRC/hash e limpeza; exclusão somente do ZIP autorizado; reuso sem aninhamento;
recusa de originais alterados/ausentes, worker, falta de espaço, ZIP corrompido,
links/diretórios/caminhos externos; falha de compressão mantendo originais e
removendo a saída temporária. A célula ativa foi exercitada com bootstrap
simulado, conferindo nome explícito e nível 9; limpar não chama recompressão.

Nenhum backup remoto foi lido, apagado ou comprimido por este agente. Tamanho e
duração reais do ZIP do usuário não foram medidos. O ensaio com dados repetitivos
é um teste de compressão/integridade, não previsão da taxa de compressão de pesos.
Não houve alteração de arquitetura, hiperparâmetros, protocolo ou treino. A entrega
de correção modifica somente o primeiro entrypoint; o segundo original permanece
fixado em `6a6af1a`, sem substituir a entrega anterior.

## Implementação do ciclo 4 — 30/10/2026

Código na branch `experiment/fractal-lrm`, em checkout isolado dentro de
`artifacts/worktrees/fractal-diagnostic-impl-20261030/` da base comum. Python local
3.12.2 (`.pyenv/pyenv-win/versions/3.12.2/python.exe`), Torch 2.14.1+cpu, Windows;
sem venv novo. A resolução explícita do executável foi necessária porque o shim
`python` não tinha versão global/local selecionada neste ambiente.

Plano registrado antes das mudanças: [PLAN_CYCLE4.md](docs/PLAN_CYCLE4.md).
Operação/limites: [CYCLE4_RUN.md](docs/CYCLE4_RUN.md). Nenhuma campanha científica
foi iniciada localmente e nenhum working remoto foi limpo.

Verificações reais até a conclusão da suíte geral:

| Verificação | Resultado observado |
|---|---|
| Primeira seleção nova, antes da integração | 47 passed, 7 deselected |
| Dois processos reais CPU, quatro checkpoints e avaliações por job | 1 passed; dois treinos/8 avaliações, ZIPs incrementais de um job e bundle final de dois |
| Seleção nova expandida | 53 passed, 1 failed, 1 skipped, 6 deselected; a falha era um teste que tentava modo não compartilhado, já proibido pelo contrato histórico; corrigido o teste, sem relaxar o contrato |
| `python -m pytest -q` (executável explícito acima) | 392 passed, 1 failed, 36 skipped, 428,90 s; única falha: sandbox bloqueou `sh.exe` no teste legado de clone Git local, `NtCreateDirectoryObject / 0xC0000022` |
| Repetição isolada do teste de clone fora do sandbox | 1 passed, 1,75 s; repositórios temporários locais, sem alteração do teste/contrato |
| Seleção final de modelos/operação/backup/JSON, após reserva de pós-treino e proteção contra corrida de deadline | 62 passed, 1 skipped (POSIX), 6 deselected (CUDA), 57,91 s |
| Operação após travar a publicação concorrente das fases | 22 passed, 1 skipped (POSIX), 39,95 s; inclui dois processos reais e 40 fases publicadas sob lock |
| Enumeração/fórmula dos seis modelos de produção, em meta | 18 configurações consistentes; embeddings ligados contados uma vez; totais no guia |
| Links Markdown locais verificados | 122, sem destinos ausentes |
| ZIP original da 0022 | SHA-256 preservado: `460b16daec5895deb87633ed1cd1d4eab40c3e5f64f9c23f22a1f9892a7346ee` |

As aprovações CPU incluem causalidade nos limites dos patches, remoção física dos
ramos e chamadas do núcleo, padrão `None` versus `[4,16]` com RNG/forward/backward
iguais, gradientes finitos, memorização dos seis braços, checkpointing e retomada
estrita CPU/SDPA math. Esta última exercita dropout 0,2 e acumulação 2, compara os
contadores carregados e a próxima loss/modelo/objetivo/AdamW/scheduler/sampler/RNG;
tempos de parede não são exigidos idênticos após nova atualização.

Backup testado com arquivos temporários: último checkpoint íntegro, retenção
conservadora de grupos/formatos desconhecidos, CRC/SHA, reuso, hash local divergente,
alteração do inventário, worker ativo, espaço insuficiente e links externos sem
leitura de metadados ligados ou exclusão dos destinos. Não houve carga de pickle
nesse fluxo. A limpeza real ocorreu apenas nas fixtures temporárias.

Operação testada com duas threads e dois processos independentes, publicação
serializada, falha/cancelamento, admissão por prazo/espaço, exportação parcial,
células inertes sem rede e célula ativa com bootstrap simulado (número do recibo,
relógio novo e limites). Perfil CPU com pesos finais e probe K1/2/4/8 preservaram
todos os hashes dos arquivos do checkpoint científico.

Os 36 skips da suíte geral incluem testes CUDA e um teste de grupos de processos
POSIX. Não representam aprovação de T4/Inductor/NVML nem do deadline em Linux.
O preflight Kaggle executará os testes focados CPU/Linux e os seis braços CUDA
FP32/Inductor com backward/checkpoint/retomada. Memorização/retomada estrita são
evidências CPU; desempenho/energia/compilação reais dependem do hardware remoto.
Os testes de integração CPU usam modelos/dados minúsculos; pilotos e pós-processamento
completos da matriz de 18 treinos não foram executados nesse smoke.

As seções abaixo preservam resultados e condições anteriores.

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

## Descoberta de ZIPs inválidos — 30/10/2026

O bootstrap `42d9179` propagava `BadZipFile` ao encontrar um candidato ilegível na
busca automática de histórico. O traceback recebido não identifica o caminho do ZIP;
não foi atribuído o problema a um arquivo específico sem evidência. Reprodução
com a API original confirmou a falha antes da correção.

Correção limitada ao operador/entrega: caminhos e motivos são registrados;
histórico legado ilegível pode ser contornado pela auditoria até 0019. Histórico
recente ilegível só é recuperado quando outro snapshot da mesma tentativa comprova
todo o lote e seus hashes de checkpoints; caso contrário, há erro explícito antes
de treinar. Descoberta precede instalação/reserva de número. Nenhum candidato é
apagado/modificado e a verificação científica posterior permanece estrita.

Verificações reais:

- `python -m pytest -q tests/test_history_discovery.py tests/test_cycle3_short.py tests/test_cycle2_bootstrap.py tests/test_cycle2_git.py tests/test_cycle3.py -k 'not cuda'`:
  **77 passed**. Seleção CPU, sem alegação de CUDA aprovado.
- Cobertura: ZIP vazio, texto, truncado, erro CRC, JSON/metadata inválidos; mistura
  com históricos válidos; outro protocolo; snapshot completo versus parcial;
  histórico cumulativo sem linhas do lote; números de tentativa e artefatos exclusivos.
- Ensaio com cópia do ZIP real 0019 e um candidato 0018 inválido: os 12 resultados
  foram preservados/deduplicados e a fila continuou no par primário 103. Candidato
  inválido e ZIP original intactos. Registro em
  `artifacts/history-discovery-fix-20261030/verification.json`.
- 29 links locais conferidos em cada checkout e `git diff --check` limpo. Entrega
  corrigida usa 0021 porque 0020 pode ter sido reservado pela tentativa anterior.

Não houve mudança de modelo, treino, avaliação ou do worker congelado `8798f6b`.
A suíte completa de 296 testes registrada acima não foi repetida para esta correção
de descoberta; foram executadas as regressões relevantes indicadas. Plano prévio:
[PLAN_HISTORY_DISCOVERY_FIX.md](docs/PLAN_HISTORY_DISCOVERY_FIX.md).

## Snapshot 0021 e manutenção do working — 30/10/2026

[Auditoria](docs/results/CAMPAIGN_0021_PARTIAL.md) e
[plano anterior à implementação](docs/PLAN_KAGGLE_STORAGE_RECOVERY.md).
Resultados after-01 SHA-256 `9043115ab066614e5a567f334a32d9be5ed75e5d90793536d9a52112f2bee223`;
checkpoint ZIP SHA-256 `4bd407d901ea4af8a11f3c12b346a5d308fc69b13c6e3c243540e1ffd6ca49b5`.
Ambos com CRC íntegro, sem entradas duplicadas; pesos e estado opaco conferidos por
hash, pickle não carregado. FLRM 103 concluiu 8.000 passos, NLL 1,320334273 e quatro
avaliações completas. Erro máximo de reconstrução de NLL 4,05051504e−9.

Header safetensors: 2.165.772 elementos de parâmetros únicos mais buffer inteiro
`training_step=8000`. Alias do embedding compartilhado preservado e contado uma vez.
Snapshot do operador comparado integralmente ao Git usando LF explícito. Identidade
científica igual à 0019. O controle estava pendente no snapshot; estado atual remoto
não foi recebido. Histórico agregado tem 13 treinos completos.

Logs reais T4 recebidos: **12 CPU aprovados em 16,39 s**, **2 CUDA FP32 aprovados em
56,95 s** na seleção focada; fases externas totalizaram 88,04 s. O primeiro treino
e exportação incremental aconteceram de fato, apesar da ausência de saída visível.
Isso não comprova o término do segundo job ou do lote.

Nova manutenção usa somente Python padrão: inventário, detecção de workers, proteção
de dados/checkpoints, remoção de checkouts Git limpos reproduzíveis e próximo número
livre. Não foi executada no Kaggle nesta conversa; tamanho real e ganho de espaço
dependem da execução remota. O script não encerra processos nem apaga campanhas.

Verificações locais reais:

- `python -m pytest -q tests/test_kaggle_maintenance.py tests/test_history_discovery.py tests/test_cycle3_short.py tests/test_cycle2_bootstrap.py tests/test_cycle2_git.py tests/test_cycle3.py -k 'not cuda'`: **91 passed**. Não é a suíte completa nem teste CUDA local.
- Remoção real de cache Git de fixture, incluindo readonly do Windows, com alvos
  resolvidos e limitados à pasta temporária. Dados, ZIPs, checkpoints, reservas e
  metadados preservados. Cache alterado, com payload ou remoto diferente é recusado.
- Worker ativo bloqueia limpeza; checkpoint parcial/conclusão não exportada bloqueia
  nova rodada. A entrega exige histórico do FLRM 103 verificado para não repeti-lo.
- Célula guardada usa número livre calculado e até um job; início/baixamento têm
  mensagens imediatas. Notebook desativado é inerte. Relógio inclui a manutenção.
- Simulação com o snapshot real reconhece a conclusão do FLRM e número 22; adicionar
  um checkpoint parcial fictício do controle bloqueou reinício. Evidência em
  `artifacts/kaggle-recovery-maintenance-20261030/verification.json`.
- 30 links locais conferidos em cada checkout e `git diff --check` limpo. Os ZIPs
  originais foram preservados. Sem alteração do worker congelado `8798f6b`.

## Caches antigos mostrados no Kaggle — 30/10/2026

O launcher cria checkouts `llm2-<revision[:12]>`. A manutenção passou a reconhecer
esse padrão exato, exigindo HEAD correspondente ao nome e todas as proteções de
Git/conteúdo/caminhos anteriores. Workers executados por esses caminhos também
bloqueiam a limpeza. Pastas `llm2-campaign-*`, incluindo nomes hexadecimais, continuam
protegidas. As imagens não permitem medir o espaço recuperável.

Verificações locais reais:

- `python -m pytest -q tests/test_kaggle_maintenance.py`: **27 passed**.
- `python -m pytest -q tests/test_kaggle_maintenance.py tests/test_history_discovery.py tests/test_cycle3_short.py tests/test_cycle2_bootstrap.py tests/test_cycle2_git.py tests/test_cycle3.py -k 'not cuda'`: **104 passed**.
- Remoção real de checkout legado limpo em fixture temporária; campanhas dos dois
  padrões, ZIP e diretórios de metadados preservados. Alterações, arquivos extras,
  payload ignorado/rastreado, remoto diferente e HEAD divergente recusados.
- Worktree legado com common-dir externo recusado; argumentos de workers antigos
  reconhecidos sem confundir nomes de campanhas. Notebook desativado e guarda da
  continuação permanecem cobertos pelas regressões.

Esses checks são CPU/arquivos locais, não a suíte completa nem execução CUDA ou
limpeza no Kaggle. Nenhuma alteração de treino, dataset, horizontes ou worker.

## Entrega 0022 — par primário seed 104, 30/10/2026

A saída de manutenção recebida informa ambos os jobs da 0021 completos/exportados,
sem workers ativos ou avisos. Os ZIPs finais ainda não foram auditados localmente.
Entrega nova usa o operador já publicado `0066131`, worker congelado `8798f6b`,
seed 104, até dois jobs e teto de 3.600 s a partir da nova célula. Não altera fontes
do operador, treino, horizontes ou protocolo. Preservar o histórico mais recente da
0021 no working/Input para agregação automática.

Verificação da entrega em `dist/cycle3-seed104-1h-20261030/verification.json`:
hashes do manifesto conferidos; bootstrap remoto igual ao código publicado; célula
ativa exercitada com bootstrap simulado, confirmando parâmetros e relógio novo;
notebook desativado sem rede. A seleção retorna `anchor_flrm-seed-104` e
`flops_transformer-seed-104`; jobs concluídos são pulados. Não houve treino local,
novo teste CUDA ou repetição da suíte: somente configuração e verificação da entrega.

## Auditoria dos resultados 0022 — 30/10/2026

[Relatório](docs/results/CAMPAIGN_0022.md) e evidência estruturada/curvas/pares
versionados em `docs/results/campaign_0022_*`. Plano anterior e auditor local em
`artifacts/audits/campaign-0022-2026-10-30/`. ZIP original SHA-256
`460b16daec5895deb87633ed1cd1d4eab40c3e5f64f9c23f22a1f9892a7346ee`,
4.908.174 bytes, 138 entradas únicas, CRC íntegro; preservado.

Verificações reais: 197 hashes do worker `8798f6b` e 213 do operador `0066131`
contra Git; snapshot integral do operador e quatro snapshots científicos (376
membros). Configurações, manifesto/tokenizer, identidade, runtime agregado, passos,
contadores, fórmulas/enumeração de parâmetros e hashes de pesos entre registros
consistentes. Oito NLLs reconstruídas de 4.469 janelas / 1.144.064 alvos cada,
erro máximo 4,03674250e−9. Métricas registradas finitas. Histórico antigo conferido
contra a evidência curada 0019 e o snapshot 0021 disponível; quatro pares recalculados.

Logs reais do Kaggle: **12 CPU aprovados em 17,16 s** e **2 CUDA FP32 em 59,24 s**,
seleção focada. Auditoria local executada com sucesso após corrigir a leitura da
chave do tokenizer no próprio auditor (`files/tokenizer.json`). Não houve alteração
do dataset nem da validação científica para aceitar o arquivo. Sem treino local,
nova execução pytest ou alegação de CUDA local. Figura científica inspecionada.

Sessão completa aos 1.925,28 s antes da exportação final; 16 treinos cumulativos,
quatro pares primários, somente dois blocos de seis braços. NLL 104 FLRM 1,318235045
e Transformer 1,312205419. Diferença média n=4 +0,001844508, IC95% exploratório
[−0,006697720; +0,010386737]. Com duas vitórias em quatro pares, o requisito positivo
pré-fixado de quatro vitórias em cinco já não pode ser satisfeito. Recomendação de
revisão antes de mais treinos; protocolo e resultado inconclusivo preservados.

Pesos/estado de treino não recebidos neste ZIP e não rehashados/carregados. Os logs
brutos do Transformer 103 também não estão neste arquivo; sua linha herdada foi
conferida na agregação, sem reconstrução local daquela NLL. Teste reservado intacto.
Compilação e aquecimento continuam sem medição isolada. Nenhuma fonte de treino mudou.
Conferidos 86 links locais em cada checkout; artefatos do relatório espelhados,
ZIP original rehashado e preservado, `git diff --check` limpo.
