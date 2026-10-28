# Validação executada — etapa 1

## V2 — registro incremental em 25/10/2026

- Auditoria antes de alterações: **29 passed, 1 skipped**, 22,80 s, CPU.
- Infraestrutura comum após hooks/factories/contabilidade: suíte original **29 passed,
  1 skipped**, 18,01 s; nenhuma atualização de PyTorch ou criação de venv.
- Recursão/JEPA/LRM: `pytest tests/test_recursive.py tests/test_predictive.py -q`:
  **18 passed**, 6,90 s. Inclui causalidade adversarial, profundidades 1/2/4, comparação
  compartilhado/não compartilhado, BPTT/checkpointing/truncamento, memorização, geração,
  teacher congelado/EMA, alinhamento MTP e retomada exata com progressão latente.
- Os resultados V1 abaixo são preservados como histórico. CUDA/A100 seguem não validados.
- Fractal isolado em `experiment/fractal-lrm`: **7 passed**, 5,92 s, CPU. F1–F4,
  pooling causal de referência, memorização, custos por recursão e checkpoint com JEPA.
- Harness curricular/orçamentos: **10 passed**, 12,82 s. DAG, verificadores, splits,
  exploração/dificuldade, respostas completas, mistura e retomada em Transformer/LRM.
- Integração principal após currículo: **57 passed, 1 skipped**, 37,91 s.
- Ferramentas de benchmark/profiler/relatório: **4 passed**, 5,33 s; traces reais e SVGs.
- Configs/perfis e notebook: **7 passed**, 6,49 s. O primeiro teste meta encontrou leitura
  indevida de Tensor.item(); corrigido para representar explicitamente a etapa inicial 0.
- Suíte principal ampliada: **66 passed, 3 skipped**, 59,75 s. Os três skips são testes
  CUDA/BF16 (baseline, Transformer+auxiliares, LRM+auxiliares); nenhum crédito GPU foi gasto.
- Perfis contados por código, sem alocar o futuro modelo: 352.359.936 parâmetros no perfil
  planning_only; a referência continua exatamente 33.990.144. Estimativas não são benchmarks.

### Fechamento integrado em 26/10/2026

- Branch comum: **66 passed, 3 skipped**, 68,16 s; log `runs/v2-main-final-pytest.log`.
- Worktree fractal: **74 passed, 4 skipped**, 73,48 s; log
  `../LLM 2-fractal-lrm/runs/v2-fractal-final-pytest.log`. Inclui os sete YAMLs F0–F4/C5/C6,
  retomada FLRM+JEPA+currículo adaptativo e causalidade nas fronteiras das janelas.
- Todos os skips são CUDA/BF16; nenhuma execução CUDA, Inductor ou A100 foi feita.
- CLI em processos separados: E0, E3, combined e F3 chegaram a quatro optimizer-steps.
  C6 parou corretamente no passo 2 por `learning_flop_budget`, antes de exceder 2 bilhões
  de FLOPs estimados de aprendizado (treino mais probes), preservando checkpoint completo.
- E3 retomado 2+2 versus quatro passos ininterruptos: igualdade **exata** dos tensores do
  modelo/objetivo/teacher, otimizador, scheduler, sampler, RNG e contadores sem temporização.
  Artefato `runs/v2-20261026/resume-equivalence.json`. Dropout 0,1 ativo; acumulação 1 nesse
  smoke externo, e acumulação 2 nos testes estritos da suíte. Safetensors pode mudar a ordem
  das chaves JSON de metadados; igualdade de estado não requer bytes serializados idênticos.
- Pesos V1 avaliados na CLI V2: checkpoint CPU etapa 20, um batch, validation loss
  **4,0527744293**. Não houve migração silenciosa nem retomada V1 usando código V2.
- Benchmarks de inferência e avaliações por respostas geradas executados para os cinco
  smokes; relatórios JSON/CSV/Markdown/SVG exportados. Exact match no test foi 0 nos cinco
  casos. Modelos tão pequenos e com tão poucas atualizações não validam as hipóteses.
- Resultados, comandos e limitações: [RESULTS_CPU_SMOKE.md](docs/RESULTS_CPU_SMOKE.md).
  Datasets e execuções da etapa 1 permanecem preservados; somente CPU local foi utilizada.
- Após os ajustes finais de documentação/notebook: **3 passed**, 2,98 s, nas verificações de
  notebook e configs; `compileall` e `git diff --check` sem erros.

Data: 22/10/2026 (America/Sao_Paulo). Os resultados abaixo são testes de infraestrutura,
não benchmarks de qualidade ou comparação de desempenho entre arquiteturas.

## Ambiente local

- Windows, Python 3.12.2, instalado diretamente em
  `C:\Users\Lux\.pyenv\pyenv-win\versions\3.12.2\python.exe`.
- Verificado: `sys.prefix == sys.base_prefix`; nenhum ambiente virtual foi criado.
- PyTorch **2.14.1+cpu**, pytest **9.1.1**, numpy **2.5.4**, PyYAML **6.0.3**,
  tokenizers **0.22.2**, safetensors **0.8.0**, TensorBoard **2.21.0**.
- Instalação do pacote em modo editável no mesmo Python, com `pip install -e ".[dev]"`.
- CUDA indisponível nessa instalação CPU. Não houve execução em A100, BF16 CUDA ou Inductor.

## Testes por etapa

| Etapa | Comando | Resultado |
| --- | --- | --- |
| Backbone | `python -m pytest tests/test_model.py -q` | 6 aprovados |
| Backbone + dados | `python -m pytest tests/test_data.py tests/test_model.py -q` | 12 aprovados |
| Treino/checkpoint inicial | `python -m pytest tests/test_training.py -q` | 7 aprovados, 1 pulado por falta de CUDA |
| Integração e regressões, suíte final | `python -m pytest -q` | **29 aprovados, 1 pulado**, 31,88 s |

A suíte final inclui:

- Shapes, comparação SDPA com referência explícita mascarada e causalidade por perturbação
  dos tokens futuros; dropout desabilitado em avaliação.
- RoPE preserva norma e posição zero; RMSNorm com entradas grandes e zero; gradientes finitos.
- Forward/backward também no **modelo completo de 33.990.144 parâmetros**, em sequência curta
  CPU, além dos modelos mínimos usados na maioria dos testes.
- Memorização de um minibatch pequeno em 100 atualizações, loss final <0,05 e <1/20 da inicial.
- Dados byte/BPE reproduzíveis, BPE treinado apenas no split de treino, round-trip UTF-8
  incluindo caracteres não latinos, emoji e grafia literal de token especial.
- Deduplicação antes do split, hashes/tamanhos/IDs, rejeição de corrupção, validação sequencial.
- Preservação exata do amostrador; avaliação não altera RNGs, modos ou estado do sampler.
- Retomada preserva pesos, objetivo, AdamW, scheduler, RNGs Python/NumPy/Torch e contadores.
  A próxima atualização coincide exatamente, com dropout e acumulação ativos.
- Equivalência numérica da acumulação com batch efetivo equivalente; clipping e contadores.
- Backbone alternativo simples e objetivo parametrizado, com restauração de parâmetros/buffers
  e estimativa de FLOPs nula para uma arquitetura desconhecida.
- Rejeição de configuração, dataset ou ambiente incompatível; rejeição de checkpoint no meio
  de uma atualização que falhou; preservação de logs de tentativas abandonadas.
- CLI preparar/treinar/avaliar/gerar/retomar; saída Unicode mesmo sob `PYTHONIOENCODING=ascii`.
- `torch.compile` com backend eager (Dynamo); notebook Colab em JSON válido e Python parseável.

## Verificação em processos separados

Executados os comandos documentados com `configs/cpu_smoke.yaml`, corpus sintético de
128 documentos e amostragem byte. O modelo CPU de demonstração tem **98.688 parâmetros**.
Os artefatos de demonstração estão ignorados pelo Git, em `data/` e `runs/`.

1. Preparação byte: 203.568 tokens de treino e 21.048 de validação.
2. Treino até a etapa 10 e retomada em outro processo até a etapa 20, em `runs/cpu-smoke/`.
3. Treino ininterrupto de 20 etapas, em `runs/cpu-continuous/`.
4. Comparação dos dois checkpoints finais: pesos safetensors, estado do objetivo, AdamW,
   scheduler, sampler, RNGs, etapa, tokens, sequências e FLOPs estimados **exatamente iguais**.
   Os tempos medidos foram excluídos dessa igualdade. Resultado persistido em
   `runs/integration-verification.json`.
5. Os eventos TensorBoard foram lidos com EventAccumulator: **20 pontos de training loss**
   e **4 de validation loss** na execução retomada.
6. Avaliação completa: **21.024 tokens**, **165 batches**, loss **4,022365317859969** nats/token.
7. Geração textual executada com sucesso em outro processo, incluindo prompt acentuado;
   o resultado após apenas 20 passos é texto sem qualidade linguística demonstrada.

Ao final do treino de demonstração: 2.560 tokens alvos, 80 sequências, training loss
4,0673301219940186, validation loss periódica 4,014386475086212 (512 tokens) e
1.636.761.600 FLOPs **estimados**. A diferença para a avaliação completa decorre da amostra
periódica limitada a quatro batches. VRAM e tempo GPU são `null` em CPU. Tempo de treino e
tokens/s estão nos logs; execuções de verificação tiveram outros processos concorrentes
e não formam um benchmark de throughput.

Também foi preparado o BPE de demonstração: **553 tokens efetivos**, com 8.192 solicitados.
Isso confirma o tratamento de corpus pequeno sem falsear a dimensão do modelo de referência.
`inspect` para a configuração 34M calculou **33.990.144 parâmetros** por enumeração e fórmula.
Com contexto 512, sua estimativa analítica é **118.883.352.576 FLOPs de treino por sequência**;
não é medição de kernels, nem ganho de desempenho.

## Correções durante a verificação

O teste inicial de hashes de documentos foi ajustado para respeitar a política de `strip`
do preparo. A execução externa de geração encontrou a codificação legada do stdout do
Windows; a CLI passou a configurar UTF-8 e recebeu regressão em subprocesso ASCII. A suíte
final e os dois treinos de comparação foram repetidos após a correção. Artefatos anteriores
foram preservados em diretórios `*-before-console-fix`, com snapshots do código correspondente.

## Dependências globais e limites

`python -m pip check` encontrou conflitos **preexistentes** em pacotes externos ao projeto:

- `aider-chat 0.86.2` requer numpy 1.26.4; o Python local já tinha numpy 2.5.4.
- `scipy 1.15.3` requer numpy <2.5; o Python local já tinha numpy 2.5.4.
- `virtualenv 21.2.0` requer filelock >=3.24.2,<4; já estava instalado filelock 3.20.3.

Esses pacotes não foram alterados. As dependências declaradas do LLM2-Lab são satisfeitas
e todos os testes CPU acima passaram; o ambiente global, como um todo, não tem `pip check`
limpo. O freeze de cada execução registra também os pacotes externos instalados.

O notebook e o ZIP Colab foram preparados para verificação real na A100. O teste CUDA/BF16
é condicional e foi pulado; Inductor, seleção de kernel SDPA e pico de 40 GB não foram
validados localmente. A CI CPU foi configurada, mas ainda não executada no GitHub.
Renames atômicos locais e checksums não são uma garantia de durabilidade em Drive/FUSE.

## Repetir a validação

```powershell
python -m pytest -q
python -m llm2 inspect --config configs/transformer_34m_a100.yaml
python scripts/package_colab.py
```

Use novos diretórios para repetir a preparação e o treino, ou retome checkpoints com o
mesmo código/ambiente. O README documenta o protocolo, as métricas e as fórmulas.

## P01–P06 — preparação da campanha Kaggle, 26/10/2026

- Branch comum: `python -m pytest -q -r a`: **78 passed, 5 skipped**, 119,54 s, Python local 3.12.2/Torch 2.14.1+cpu. Log local ignorado pelo Git: `runs/p01-p06-common-pytest.log`.
- Testes direcionados de campanha/CLI/objetivos/perfil: **17 passed, 2 skipped**, 51,04 s; `runs/p01-p06-targeted.log`. Inclui campanha CPU em subprocessos, falha explícita, exportação sem sobrescrita, avaliação por janela, teacher hidden-only, alinhamento JEPA, acumulação CE por tokens sem reponderar auxiliares e retomada exata com dropout/acumulação.
- Adicionados testes CUDA **FP32** para Transformer/LRM, separados dos testes BF16 existentes. Os cinco skips são CUDA; não houve GPU local, medição T4, energia ou Inductor. O bootstrap executará esses checks no Kaggle.
- Perfil agora separa warmup, amostras sem profiler e passo instrumentado. Seus FLOPs suportados e agregados de memória não são medições universais.
- Fonte pequena tinyshakespeare fixada em char-rnn 6f9487a6fe5b420b7ca9afb0d7c078e37c1d1b4e: URL conferida HTTP 200; nenhum corpus grande foi baixado e não houve treino científico local.
- Não foi afrouxada a retomada de checkpoints antigos. A origem dos dados, configurações, código e pesos permanece verificada; a correção dos objetivos abre execuções novas.
- Uso no navegador: docs/KAGGLE_T4_CAMPAIGN.md. O código FLRM continua isolado na branch experimental; sua validação será registrada separadamente.

### Fechamento P01–P06

- Suíte comum final: **80 passed, 5 skipped**, 123,86 s; `runs/p01-p06-common-final-pytest.log`. Após revisão: **18 passed, 2 skipped**, 39,86 s nos testes de campanha, objetivos/CLI e ferramentas.
- Suíte fractal final: **94 passed, 8 skipped**, 143,84 s; `../LLM 2-fractal-lrm/runs/p01-p06-fractal-final-pytest.log`. Direcionados fractal: **14 passed, 3 skipped**, 14,73 s.
- Paridade sequencial/agrupado validada em CPU FP32 com SDPA math e dropout controlado; comparação de gradientes/parâmetros, causalidade, memorização e retomada exata dentro de cada modo com dropout ativo, acumulação 2, checkpointing, JEPA, MTP e progressão latente.
- Campanha completa simulada em CPU, em processos separados: Transformer, LRM, FLRM reference e FLRM batched; dois optimizer-steps por modelo, corpus demo preservado, contexto 32. Todos os quatro chegaram a `complete`. Artefatos locais em `../LLM 2-fractal-lrm/runs/p01-p06-integration/` e ZIP de análise correspondente.
- Cada avaliação completa processou **21.024 alvos em 657 janelas**. Diferença entre média reconstruída dos JSONLs e média agregada <= 5,57e-8. Houve geração, diagnóstico e perfis dos dois modos FLRM, com warmup 1, duas amostras sem profiler e um passo instrumentado.
- O ZIP de integração passou em `ZipFile.testzip()`, com 112 arquivos e 1.339.174 bytes. Não contém tensores/checkpoints retomáveis. Essa execução e as suítes concorrentes são verificações de engenharia, não benchmarks de throughput/qualidade CPU.
- Corrigidos na revisão o default de inferência para respeitar FP32 do checkpoint, mapeamento de GPUs sob CUDA_VISIBLE_DEVICES e rótulos de avaliação completa/parcial. O runner T4 seleciona testes CUDA FP32; não exige BF16 nativo/emulado.
- `compileall`, parse do bootstrap e `git diff --check` passaram. Não houve execução CUDA/T4/BF16/Inductor, energia ou confirmação de ganho de velocidade/VRAM. O próximo resultado empírico de GPU virá do Kaggle pelo navegador.

### Esclarecimento de dimensão e recursos — 26/10/2026

- Corrigido o rótulo histórico incorreto de ~10,7M em `kaggle_t4_1h_fractal.yaml`: esse perfil usa 2.167.308 parâmetros no backbone FLRM. `research.core_layers: 1` agora está explícito nas duas configs Kaggle antigas; comentários esclarecem que `model.n_layers` não define a profundidade desse núcleo e que T4x2 não unifica VRAM.
- Comparação das duas configurações antigas/novas via `ExperimentConfig.from_dict(...).to_dict()` retornou igualdade completa. Registro local: `runs/flrm-config-clarification.json`. Nenhuma mudança de configuração resolvida ou de código de treino; não houve novo treino nem repetição da suíte para essa clarificação.
- Enumeração de parâmetros em dispositivo `meta`, sem treino ou alocação GPU, conferida com `fractal_parameter_formula`: núcleo 1/iteração 1 = **2.167.308**; núcleo 1/iterações 2 = **2.167.308**; núcleo 2/iteração 1 = **3.937.548**; núcleo 6/iteração 1 = **11.018.508**. Alterar apenas `model.n_layers` de 6 para 12 preservou 2.167.308 no FLRM. São contagens estruturais, não medições de qualidade/velocidade dessas variantes.

### Notebook amigável e revisão de checkpoints — 26/10/2026

- Plano/riscos/aceitação em `docs/KAGGLE_FRIENDLY_PLAN.md`. Mudanças de apresentação: etapas numeradas, GPU física/modelo/seed, progresso a cada 100 passos com ETA aproximado, loss parcial identificada, geração visível, direções/unidades das métricas e resumo final em português. Não houve alteração de modelos, objetivos, otimizador ou matemática do treino.
- Novo modo `avaliar_salvos` executa apenas avaliação, usando o checkout original depois de conferir hashes de código, dados e pesos. Mantém os dados/checkpoints antigos, exporta outra pasta/ZIP e só reutiliza avaliação completa depois de verificar identidade, escopo e NLL reconstruída das janelas. Ausência de pesos ou incompatibilidade interrompe; não há fallback para treino.
- Direcionados finais: **24 passed**, 33,06 s, sob `CUDA_VISIBLE_DEVICES=0`; `runs/kaggle-friendly-targeted-final.log`. Incluem avaliação CPU real de dois checkpoints em subprocessos, reutilização verificada sem subprocessos/treino, imutabilidade dos pesos, rejeição de cache com hash errado, dados incompatíveis, pesos ausentes e fonte alterada. A primeira rodada teve três falhas de leitura CP1252 em testes Windows; leituras UTF-8 explícitas corrigiram os testes.
- Suíte comum completa: **99 passed, 5 skipped**, 130,23 s; `runs/kaggle-friendly-full-pytest.log`. Suíte fractal completa: **113 passed, 8 skipped**, 140,40 s; `../LLM 2-fractal-lrm/runs/kaggle-friendly-full-pytest.log`. Todos os skips locais são CUDA indisponível.
- Integração adicional CPU com código arquivado do commit original `341421a3464fea090d69dc10f897a574ce1428f9`: campanha mínima Transformer/FLRM (um passo por modelo), seguida de conferência dos 34 hashes fonte e reutilização das avaliações pelo novo executor. Status `complete`; logs em `runs/kaggle-friendly-original-source-check/`. São verificações de engenharia, não resultados científicos de qualidade/performance.
- `compileall`, parse do bootstrap e `git diff --check` passaram. O novo fluxo de revisão e apresentação ainda não foi executado em T4 localmente ou no Kaggle; as medições empíricas da campanha anterior permanecem identificadas por seu commit original. Não foi carregado pickle pelo modo de revisão.

### Correção da máscara de GPU herdada — 26/10/2026

- A execução Kaggle fixada em `10b05f554947efa1171d298e774608a25f20e0a7` detectou duas T4, mas parou na suíte CPU antes de qualquer treino: **1 failed, 93 passed, 8 deselected**. O teste de configuração sem máscara herdou `CUDA_VISIBLE_DEVICES=0` do subprocesso e pediu o índice lógico 1. Avisos de matplotlib/pyparsing não causaram a falha.
- Reprodução CPU local com a mesma variável: **1 failed**, 0,69 s; `runs/kaggle-visible-gpu-repro.log`. O teste agora remove explicitamente a máscara ao verificar o cenário sem máscara. O helper valida índice lógico e disponibilidade, preserva UUID/MIG e nunca escapa de uma máscara vazia, desativada ou insuficiente por fallback.
- Testes de campanha sob `CUDA_VISIBLE_DEVICES=0`: **20 passed**, 15,63 s; `runs/kaggle-visible-gpu-targeted.log`. Incluem mapeamento de máscara unitária/múltipla/UUID/MIG, índices inválidos, CUDA desativado, preservação de ambiente e a campanha CPU em subprocessos.
- Suíte comum completa sob a mesma máscara: **95 passed, 5 skipped**, 111,08 s; `runs/kaggle-visible-gpu-full-pytest.log`.
- Suíte fractal completa sob a mesma máscara: **109 passed, 8 skipped**, 124,72 s; `../LLM 2-fractal-lrm/runs/kaggle-visible-gpu-full-pytest.log`.
- Todos os skips locais continuam sendo CUDA indisponível. `compileall` dos arquivos alterados e `git diff --check` passaram. A correção não modifica modelos, objetivos, treino ou retomada; os testes CUDA FP32 permanecem habilitados na próxima campanha Kaggle. Ainda não há resultado de treino ou benchmark T4 desta entrega.

### Evidência Kaggle recebida — campanha 8f86ba9cc48b

- O usuário executou no Kaggle o commit `341421a3464fea090d69dc10f897a574ce1428f9` e forneceu o ZIP de análise. SHA256: `e7aa76610a2e9023a20f909f45cc38b76e1f8209fb04d108932d02ef1d34fe14`; 129 arquivos, CRC íntegro. Dados e ZIP bruto ficam fora do Git. Esta seção registra evidência externa recebida, não uma execução GPU local.
- Logs: suíte CPU **109 passed, 8 deselected**, 109,80 s; CUDA FP32 **4 passed, 113 deselected**, 16,53 s. Os quatro testes CUDA verificam atualização finita, save e continuação em Transformer/LRM/FLRM sequencial/agrupado. Não verificam retomada bit a bit em CUDA, BF16 ou Inductor.
- Os quatro treinos terminaram em 2.000 optimizer-steps/16.384.000 alvos. Transformer e FLRM sequencial usaram uma T4; LRM e FLRM agrupado usaram a outra, conforme máscaras e UUIDs distintos. Ambiente: Python 3.13.15, Torch 2.11.0+cu128/CUDA 12.8; FP32/eager.
- Snapshots/hashes fonte conferem com o objeto Git fixado. Corpus bruto e manifest permitiram reconstruir contagens e SHA256 dos arrays de treino/validação; splits documentais disjuntos. Dataset SHA256 `beed0146113d17994cd94ddd246e8cabecf6c5eceb005c394526884df66a3ef7`. Cada avaliação final cobre 437 janelas/111.872 alvos; reconstrução da loss pelos JSONLs difere menos de 4e-9 do agregado. Restam 59 alvos sem janela completa.
- Loss LM completa: Transformer **1,36585736**; LRM **1,40629617**; FLRM sequencial **1,40810997**; FLRM agrupado **1,40727011**. Tamanhos e JEPA diferem; uma seed não estabelece superioridade arquitetural. LRM ficou ligeiramente melhor que os FLRMs no conjunto completo, invertendo a ordenação da avaliação periódica parcial.
- Throughput médio de treino (alvos/s): **33.769 / 59.357 / 39.457 / 38.298**, na mesma ordem. VRAM pico alocada: **683,73 / 337,10 / 449,57 / 541,12 MiB**. Agrupado vs sequencial: throughput -2,94%, VRAM +20,36%; treinos em GPUs distintas e com jobs concorrentes não isolam efeitos do agrupamento.
- Perfil isolado na mesma T4, três warmups/cinco amostras sem profiler: sequencial **41.530,53** e agrupado **43.327,50 alvos/s**, +4,33%. Chamadas cudaLaunchKernel por passo instrumentado: 3.884 vs 1.936. Atenção eficiente SDPA observada. Essa janela curta precisa de repetição; a meta de 60 mil alvos/s não foi demonstrada.
- Geração batch 1, 64 tokens, contexto até 256, sem KV cache, mesma T4: Transformer **124,10**; LRM **232,51**; FLRM sequencial **149,32**; agrupado **315,58 tokens/s**. Razão agrupado/sequencial **2,11×**, observação curta sem replicações, não extrapolável a contexto longo.
- FLRM usou uma iteração e o LRM duas chamadas recursivas. Os diagnósticos FLRM mostram escalas distintas nesta amostra, mas não provam especialização semântica ou benefício causal. Esta campanha não testa Fibonacci, DEQ, Kronecker, contexto longo ou qualidade de raciocínio.
- O ZIP exclui pesos e estado do optimizer/RNG. A análise verificou consistência dos hashes declarados, sem reexecutar checkpoints. Os tensores precisam ser preservados no Kaggle para avaliar checkpoints intermediários e para retomada. O Transformer tem sinal de sobreajuste na validação parcial após o passo 1.000; a avaliação completa dos intermediários permanece pendente.

### Revisão completa de checkpoints recebida — 9d8a6efd9be9

- Evidência fornecida pelo usuário após executar o modo `avaliar_salvos` no Kaggle. ZIP com 67 arquivos, CRC íntegro, SHA256 `f7ed81da42dc2f4b63c4e80109413fb733328c450fe50a637b5b9a23b9b34e53`. Executor `c9bbb034c81d263c8800aa3635de0ad862aa6a16`; código efetivo das avaliações preservado em `341421a3464fea090d69dc10f897a574ce1428f9`. Nenhum treino novo foi disparado.
- As 16 avaliações terminaram: 12 novas na GPU e quatro de 2.000 passos reutilizadas, byte a byte idênticas às originais. Todos os metadados dos checkpoints coincidem com os do bundle anterior. Snapshots de executor/avaliador conferem com os respectivos objetos Git; dataset/tokenizer/fingerprint e versões de Python, dependências, CUDA e cuDNN continuam iguais. A nova sessão tem outras T4 físicas; não foram feitas novas medições de performance.
- Cada resultado usa as mesmas 437 janelas/111.872 alvos, contexto 256, FP32. A maior diferença entre NLL reconstruída dos JSONLs e loss agregada foi 1,11e-8. Loss **menor = melhor** previsão.

| Modelo | 500 passos | 1.000 passos | 1.500 passos | 2.000 passos | Melhor passo salvo |
|---|---:|---:|---:|---:|---:|
| Transformer | 1,37577653 | 1,30951297 | 1,32755842 | 1,36585736 | 1.000 |
| LRM | 1,56969399 | 1,46794103 | 1,42364914 | 1,40629617 | 2.000 |
| FLRM sequencial | 1,56378443 | 1,46553219 | 1,42437732 | 1,40810997 | 2.000 |
| FLRM agrupado | 1,56380001 | 1,46433329 | 1,42343905 | 1,40727011 | 2.000 |

- A piora do Transformer depois do passo 1.000, juntamente à redução da loss de treino já registrada, é consistente com sobreajuste. Seu checkpoint final tem perplexidade 5,80% maior (pior) que o melhor checkpoint salvo. Comparando os melhores salvos, a perplexidade do FLRM agrupado é 10,27% maior (pior) que a do Transformer; tamanho/objetivo diferem e a seleção usa validação, não teste independente.
- LRM e FLRM melhoraram em cada checkpoint salvo, mas isso não garante melhoria com mais passos. LRM continua ligeiramente melhor em loss final; uma seed não demonstra superioridade estatística ou benefício de qualidade das escalas fractais. Permanecem prioritários controle Transformer pequeno, objetivos pareados, múltiplas seeds e ablação de uma/duas iterações.
- Esta revisão fornece validação empírica do fluxo de avaliação em T4, não de novo treino, BF16, Inductor ou alteração arquitetural. Os ZIPs não contêm os tensores: a análise local verificou identidades declaradas e evidências de execução, sem carregar modelos/pickle. Dados brutos e análise detalhada permanecem fora do Git.

### Campanha comprimida recebida — 312eca785d57, 27/10/2026

- Evidência fornecida pelo usuário, commit `a668af73fde01eccfcffb3f6d36d9b022747b7fc`; ZIP com 85 arquivos, CRC íntegro, SHA256 `721fc05c1d524c2c2253d6f182f4756af0eb368f2ed3b987ffb54bb1073f3803`. Status complete nos três treinos, 2.000 passos/16.384.000 alvos, FP32, seed 42, contexto 256. Próximas comparações solicitadas pelo usuário focam FLRM × Transformer, preservando LRM histórico.
- Dataset fingerprint original preservado. Snapshots conferem com a fonte Git publicada e os mapas de hashes de treino/checkpoint/campanha são iguais. Avaliação completa usa 437 janelas/111.872 alvos; erro máximo de reconstrução da NLL <8,3e-9. Pesos não estão no ZIP; identidades declaradas e evidências foram verificadas localmente, sem carregar checkpoints.
- Transformer pequeno: 2.164.224 parâmetros backbone/total, loss **1,40049777**. FLRM comprimido K=2: 2.168.076 backbone/4.631.064 total de treino, loss **1,39778180**. Loss menor = melhor previsão; diferença 0,002716, redução de perplexidade de **0,271%**, ainda uma seed. LRM K=2: loss 1,40629622, mantido como registro histórico.
- Transformer vs FLRM: throughput médio **155.757,77 vs 34.073,75 alvos/s** (maior = melhor); passos de treino **105,19 vs 480,84 s** (menor = melhor); VRAM pico alocada **202,16 vs 443,81 MiB** (menor = menor consumo). FLRM demandou 4,57× tempo e 2,20× VRAM. Tamanhos do backbone são próximos, mas K/JEPA/teacher/custo diferem.
- Geração curta na mesma GPU, batch 1, 64 tokens novos, prompt 14, warmup 8, sem KV cache: **468,82 vs 78,44 tokens/s** (maior = melhor), FLRM 5,98× mais lento. Não houve perfil isolado nesta campanha (`profile=False`) nem demonstração de ganho em contexto longo.
- Logs Kaggle: **114 passed, 8 deselected** CPU, 148,85 s; **4 passed, 118 deselected** CUDA FP32, 16,67 s. Os quatro casos CUDA existentes não incluem o modo comprimido. O treino é evidência de execução comprimida na T4, mas seu contrato específico de memorização/retomada exata CPU e continuação CUDA ainda deve ser ampliado.
- Probes locais somente CPU, modelo pequeno novo, dropout zero, K=2, comprimento 33: causalidade sob mutação futura e equivalência de prefixo passaram em 32 cortes. Confirmadas seis chamadas de núcleo com comprimentos 33/9/3 e seis projeções de interação com comprimento completo 33. Isso não constitui benchmark nem teste dos pesos treinados.
- Pontos para revisão futura, sem alteração de treino nesta análise: `profile_runs` não seleciona `flrm_compressed`; o estimador de interação conta uma projeção em T, mas o forward executa três; CKA trunca contagens sem alinhar âncoras temporais; fatores 4/16 são fixos enquanto `research.scales` permanece [1,4,0].
- A rota de upsample causal seguida de compress desloca um bloco: com prior zero, [10,20,30,40] vira [0,10,20,30]. Isso requer explicitar a semântica do estado recorrente, não prova vazamento nem invalida automaticamente o resultado. Otimização equivalente e mudança dessa dinâmica devem ser experimentos separados.
- Foco recomendado: perfil/testes/estimativas do modo comprimido; interação em resoluções nativas; controles JEPA/K com seeds adicionais e orçamento explícito; cache de geração e resumos aprendidos como etapas posteriores. Recursão fixa/JEPA EMA já existem; DEQ/ACT/NCM/Fibonacci/fatoração de pesos não foram implementados por esta análise.

### Etapa 1 — otimizações do FLRM comprimido, 27/10/2026

- Plano prévio, riscos e aceitação em `docs/PLAN_COMPRESSED_OPTIMIZATION.md`.
  Trabalho somente na branch experimental. O controle `compressed` conserva seu
  forward; dois modos explícitos adicionam projeção após pooling e agrupamento
  opcional apenas das escalas curtas. Mesmo conjunto de pesos e contagem de
  parâmetros. Nenhuma transferência de pesos ou relaxamento de checkpoint.
- `python -m pytest -q`: **132 passed, 12 skipped**, Python 3.12.2 /
  Torch 2.14.1+cpu. Os skips são CUDA indisponível, não aprovações de GPU.
  `compileall` dos fontes/scripts/testes alterados e `git diff --check` passaram.
- Testes novos: equivalência de logits/hidden/gradientes CPU FP32/math SDPA com
  mistura/FiLM/priors não triviais, K=1/2, comprimentos 1/4/15/16/17/31/32/33/65;
  invariância futura exata e equivalência de prefixo com tolerância em 32 cortes.
  Retomada exata antes e depois da próxima atualização nas três variantes com
  dropout .2, acumulação 2, EMA e checkpointing ativos; compara modelo, objetivo,
  otimizador, scheduler, sampler, RNG e contadores determinísticos. Tempos e picos
  de memória são observações, não invariantes determinísticos. Memorização de
  minibatch reduz loss para menos de 1/15 da inicial em cada modo.
- A primeira rodada direcionada teve 13 aprovações/3 falhas/3 skips: o teste de
  formas solicitava 33 posições com max_seq_len=32. Corrigido o fixture para 33,
  sem alterar o limite do modelo. Rodada direcionada seguinte: 37 aprovações,
  3 skips. A suíte completa acima inclui também diagnósticos e novo teste CUDA
  de equivalência. Os quatro casos CUDA novos serão executados pelo seletor
  `cuda and fp32` do notebook; ainda não foram executados em hardware real.
- Estimativa v3 corrige as três projeções no controle e contabiliza padding do
  agrupamento. Fórmula, convenções e exclusões em `docs/ARCHITECTURE_FLRM.md`.
  Não comparar totais v2/v3 como ganho medido. CKA entre escalas comprimidas agora
  é null por falta de alinhamento temporal; não atribuir especialização a esse
  diagnóstico. Nenhuma mudança na loss/otimização do Trainer.
- Campanha/perfis: todas as variantes solicitadas selecionadas, uma seed por
  modelo, mesma GPU e sem concorrência nos perfis. Notebook expõe warmup e amostras
  (10/30 por padrão); console mostra alvos/s isolados. Teste do seletor inclui
  Transformer e as três variantes comprimidas e evita repetir seeds.
- A célula de triagem mantém K=2 e seed 42, comparando Transformer pequeno,
  comprimido controle, projeção nativa e projeção nativa com agrupamento. LRM sai
  dos modelos padrão. Ganhos de velocidade/VRAM e manutenção da qualidade após
  2.000 passos continuam pendentes da execução Kaggle T4. Não foi medido ganho GPU
  nem implementado cache de geração, adaptação de K ou nova arquitetura nesta etapa.
- Integração CPU descartável `runs/stage1_smoke.py`: quatro variantes em processos
  separados, D=16/H=32, contexto 17, dropout .1, acumulação 2, K=2, um optimizer-step;
  avaliação completa de 1.003 alvos por modelo, benchmark/geração/diagnósticos,
  quatro perfis reais (1 warmup/2 amostras/1 trace) e ZIP concluídos. Log:
  `runs/stage1-smoke-final.log`; campanha `runs/stage1-smoke-f65c58cff2/campaign`.
  Diretórios temporários e pesos ficam ignorados pelo Git. Não usar os tempos
  desse smoke como benchmark nem a loss de um passo como resultado científico.
- `python -m pytest -q -m cuda`: **12 skipped, 132 deselected**, 0,35 s, log
  `runs/stage1-cuda-unavailable.log`; confirma somente a coleta dos testes CUDA,
  sem execução local. Reforço posterior da comparação de contadores determinísticos
  verificado novamente na suíte direcionada do modo comprimido: **17 passed,
  4 skipped**, 18,10 s, log `runs/stage1-compressed-tests.log`.

### Evidência Kaggle — otimizações comprimidas, campanha 8b8ac432c0ea

- ZIP fornecido pelo usuário em 27/10/2026, 152 arquivos, 10.015.046 bytes, CRC
  íntegro, SHA256 `206559f7f4792166ae5018639cc5606f529a185aa439e614ee8b5dd63257e4ce`.
  Executou `0a819dd64c65a633d8b53055121bc0c3aa5e9db5`, checkout limpo. Snapshots,
  mapas de fontes e configurações conferem byte a byte com Git. Análise local
  dos artefatos, sem carregar tensores ou executar checkpoints recebidos.
- Quatro treinos completos de 2.000 passos/16.384.000 alvos, seed 42, contexto
  256, FP32/eager. Mesmo dataset fingerprint original `beed0146113d17994cd94ddd246e8cabecf6c5eceb005c394526884df66a3ef7`.
  Avaliação completa em 437 janelas/111.872 alvos; NLL reconstruída dos JSONLs
  concorda dentro de 1e-6. Identidades declaradas dos pesos coincidem entre
  treinamento/checkpoint/avaliação/benchmark; o ZIP não contém esses tensores.
- CPU no Kaggle: **132 passed, 12 deselected**, 158,55 s. CUDA FP32:
  **8 passed, 136 deselected**, 22,97 s, incluindo equivalência das otimizações
  comprimidas e save/continuação nos três modos. Esta evidência supera a limitação
  anterior de GPU ainda pendente para esses testes, somente neste ambiente
  FP32/eager (Python 3.13.15, Torch 2.11.0+cu128, CUDA 12.8, Tesla T4). Não valida
  BF16/Inductor nem retomada bit a bit em CUDA.
- Dois UUIDs T4 distintos confirmam treinos nas GPUs 0/1: Transformer/controle
  iniciaram juntos, seguidos de nativo/agrupado nas respectivas filas. Perfis,
  avaliação e geração sequenciais na mesma GPU 0; UUID dos perfis idêntico.
  Cada perfil tem 10 warmups, 30 amostras sem profiler e um passo separado
  instrumentado, a partir de inicialização nova, sem alterar os modelos finais.

| Modelo | Loss completa ↓ | Treino alvos/s ↑ | VRAM MiB ↓ | Geração tok/s ↑ | Perfil isolado alvos/s ↑ |
|---|---:|---:|---:|---:|---:|
| Transformer pequeno | 1,40050188 | 152.324 | 202,16 | 515,17 | 169.427 |
| FLRM comprimido controle | 1,39778178 | 34.236 | 443,81 | 82,03 | 36.738 |
| FLRM nativo | 1,39778174 | 34.937 | 433,80 | 76,95 | 37.434 |
| FLRM nativo + agrupado | 1,39707953 | 36.000 | 470,00 | 94,13 | 38.769 |

- Loss menor = melhor; velocidade maior = melhor; VRAM menor = menor consumo
  para a mesma tarefa. Passos de treino: 107,56 / 478,57 / 468,96 / 455,11 s,
  respectivamente, sem preparação/testes/validação/perfis. Transformer K=1 e sem
  JEPA; todos os FLRMs K=2 e JEPA .1/teacher EMA, 2.168.076 backbone/4.631.064
  parâmetros totais de treino versus 2.164.224 backbone/total no Transformer.
- Contra controle, nativo: throughput isolado +1,90%, VRAM −2,25%, geração
  −6,20%; loss difere apenas 3,41e-8. Agrupado: throughput isolado +5,53%, geração
  +14,75%, VRAM +5,90%. Chamadas cudaLaunchKernel instrumentadas: 7.420 controle,
  7.520 nativo, 5.988 agrupado (−19,30% vs controle). Ganhos observados modestos,
  30 passos consecutivos não são replicações independentes; sem meta de 60k atingida.
- Agrupado vs Transformer: perplexidade −0,34165%, mas passos de treino 4,23× mais
  demorados, VRAM 2,32× e geração 5,47× mais lenta. Uma seed e objetivos/custos
  diferentes não demonstram superioridade arquitetural ou eficiência global.
  A pequena melhora de loss do agrupado vs controle também não isola capacidade:
  agrupamento altera RNG/dropout .05 e arredondamento. No parcial final a ordem
  das variantes é diferente da validação completa; priorizar a completa.
- Geração é um benchmark curto, batch 1, prompt 14, 64 tokens novos, warmup 8,
  sem KV cache. Seleção de checkpoint completo intermediário e replicações de
  timing/qualidade continuam pendentes. O próximo controle recomendado é o mesmo
  FLRM agrupado com/sem JEPA; `flrm_no_jepa` existente usa o modelo não comprimido
  e não serve para essa ablação. Nenhuma implementação de treino alterada nesta
  análise; catálogo Argon continua como ideias futuras.

### Preparação da ablação JEPA agrupada — 27/10/2026

- Pedido do usuário: preparar o script Transformer pequeno × mesmo FLRM agrupado
  com/sem JEPA, para execução posterior no Kaggle. Plano prévio, riscos e
  critérios em `docs/PLAN_JEPA_ABLATION.md`. Novo identificador explícito
  `flrm_compressed_grouped_no_jepa`; arquitetura `compressed_grouped`, jepa_weight=0.
  O existente `flrm_compressed_grouped` preserva JEPA .1. Sem alterações no
  backbone, Trainer, algoritmos de checkpoint ou fontes dos dados.
- Defaults/célula selecionam sem JEPA, com JEPA, Transformer, seed 42, 2.000
  passos, K=2 nos FLRMs. Filas existentes: sem JEPA e depois Transformer na GPU 0;
  com JEPA na GPU 1. Espera-se melhor distribuição do trabalho, sem garantir
  durações. Perfis/avaliações/geração continuam sequenciais na mesma GPU.
- Teste do par confere configs iguais exceto jepa_weight/output_dir, pesos iniciais
  do backbone idênticos e mesmo primeiro batch. Sem JEPA constrói
  CausalLanguageModeling, sem target/predictor; com JEPA constrói PredictiveObjective.
  Sampler tem RNG próprio, mas a inicialização do preditor muda RNG de dropout;
  não prometer mesmos sorteios de dropout durante as duas trajetórias.
- Enumeração local do perfil completo D=384/H=1024/vocab=257, conferindo fórmula
  dos backbones fractais: sem JEPA **2.168.076 backbone/total**; com JEPA
  **2.168.076 backbone/4.631.064 total**; Transformer pequeno **2.164.224
  backbone/total**. Log `runs/jepa-ablation-parameters.log`, sem treino desse perfil.
- Direcionados: **43 passed, 7 skipped**, 27,85 s,
  `runs/jepa-ablation-targeted-tests.log`. Inclui retomada CPU FP32/math SDPA
  com dropout, acumulação e checkpointing ativos, agora também sem JEPA nos
  três modos comprimidos; compara próxima atualização e estados determinísticos.
  Memorização e causalidade do mesmo backbone continuam na suíte.
- `python -m pytest -q`: **137 passed, 15 skipped**, 114,72 s,
  `runs/jepa-ablation-full-tests.log`, Python 3.12.2/Torch 2.14.1+cpu. Os skips são
  CUDA indisponível. Novos casos CUDA FP32 de save/continuação sem JEPA estão
  coletados, mas não executados localmente. Evidência Kaggle anterior valida
  os modos com JEPA, não a nova combinação sem JEPA. A próxima célula executa
  o seletor CUDA FP32 antes dos treinos reais.
- Integração CPU real de três processos de treino e processos separados de
  avaliação/geração/diagnóstico/perfis, D=16/H=32, contexto 17, dropout .1,
  acumulação 2, K=2, um optimizer-step: **complete**. Os três modelos avaliaram
  1.003 alvos cada e geraram três perfis com 1 warmup/2 amostras/1 trace, relatório
  e ZIP. Log `runs/jepa-ablation-smoke.log`; artefatos descartáveis em
  `runs/jepa-ablation-smoke-1bf1771d1a/campaign`. Sem benchmark de GPU ou resultado
  de qualidade científica desta ablação antes da execução do usuário.
- Célula publicada usa novos diretórios e não retoma arquivos antigos. K=2,
  objetivos puros sem MTP/currículo e mesmo orçamento de alvos. Comparar qualidade
  pela loss LM completa (menor = melhor); velocidade maior = melhor; VRAM menor =
  menor consumo. Uma seed é triagem, não evidência de superioridade consistente.

### Saídas numeradas — 27/10/2026

- Pedido: retirar sufixos aleatórios dos ZIPs. Plano em
  `docs/PLAN_NUMBERED_OUTPUTS.md`. Novo helper stdlib em
  `src/llm2/research/naming.py`, incluído em snapshots/hashes. Bootstrap e defaults
  CLI usam `llm2-campaign-0001`/`llm2-review-0001`; ZIP herda o número da pasta.
  Os nomes existentes e os diretórios do código identificados por SHA são preservados.
- Números vêm de pastas, ZIPs e reservas locais; bootstrap também consulta outputs
  anteriores anexados em `/kaggle/input`. `run_number` positivo permite escolha
  explícita. Sessão vazia sem histórico reinicia em 0001: não há contador remoto
  global. Revisões têm sequência separada; falhas podem consumir números.
- Reservas por mkdir exclusivo fora da pasta da campanha: sem pré-criar o output
  nem permitir reutilização/overwrite. Testes cobrem 24 reservas concorrentes,
  arquivos ZIP isolados preservados, input de execução interrompida, outputs
  anexados, número manual ocupado/inválido e carregamento do helper via runpy
  para evitar import antigo no kernel do notebook. Campanha real de metadados/ZIP
  confirma compatibilidade com criação exclusiva do output.
- Direcionados: **37 passed**, 23,95 s, incluindo testes de campanha/revisão,
  `runs/numbered-output-tests.log`. Suíte `python -m pytest -q`: **147 passed,
  15 skipped**, 87,34 s, `runs/numbered-output-full-tests.log`. Skips são CUDA
  indisponível localmente. Compileall e git diff --check passaram.
- Nenhuma alteração no backbone, objetivos, Trainer, dados ou checkpointing.
  Nenhum novo resultado GPU ou de qualidade; a ablação JEPA continua pendente
  da execução Kaggle do usuário. `--output` explícito mantém seu comportamento.

### Evidência Kaggle — ablação JEPA, campanha 0001

- ZIP recebido do usuário, `llm2-campaign-0001-results.zip`, 6.481.624 bytes,
  118 entradas, CRC íntegro, SHA256
  `f1dbcf713a1a569d84e056efc5de05887f7524cd85a47e965e20cbc1ac0349ba`.
  Executou `f16765f6e350b343de782f848ee91ed32b258796`, checkout limpo. Código,
  configurações versionadas e snapshots conferem byte a byte com os objetos Git.
- Os três treinos terminaram em 2.000 passos/16.384.000 alvos, seed 42, contexto
  256, FP32/eager. Configs dos dois FLRMs diferem somente em jepa_weight e
  diretório de saída: mesmo `compressed_grouped`, K=2. Sem JEPA tem 2.168.076
  parâmetros backbone/total, zero congelados; com JEPA tem mesmo backbone,
  4.631.064 total/2.168.076 congelados. Transformer pequeno: 2.164.224 total, K=1.
- Dataset fingerprint original `beed0146113d17994cd94ddd246e8cabecf6c5eceb005c394526884df66a3ef7`;
  437 janelas completas/111.872 alvos iguais nas avaliações. Loss reconstruída
  dos JSONLs: erro máximo 3,98e-9. Identidades declaradas dos pesos são iguais
  entre checkpoint, avaliação e benchmark; tensores ausentes do ZIP e não
  carregados localmente. Auditoria teve 205 verificações sobre os artefatos.
- Logs Kaggle: CPU **147 passed, 15 deselected**, 200,74 s; CUDA FP32 **11 passed,
  151 deselected**, 32,17 s. Incluem os novos casos de save/continuação comprimida
  sem JEPA, anteriormente pendentes. Ambiente Python 3.13.15/Torch 2.11.0+cu128,
  CUDA 12.8, T4, FP32/eager. Não valida BF16/Inductor nem retomada bit a bit CUDA.
- Dois UUIDs T4 distintos: os FLRMs começaram juntos nas GPUs 0/1; Transformer
  treinou depois na GPU 0. Perfis/geração/avaliação foram sequenciais na GPU 0,
  com UUID idêntico nos três perfis. Nome numérico e ZIP correspondente funcionaram.

| Modelo | Loss LM completa ↓ | Tempo de passos de treino s ↓ | Treino alvos/s ↑ | VRAM MiB ↓ | Geração tok/s ↑ | Perfil isolado alvos/s ↑ |
|---|---:|---:|---:|---:|---:|---:|
| Transformer pequeno | 1,40050306 | 104,99 | 156.055 | 202,16 | 470,78 | 166.224 |
| FLRM agrupado sem JEPA | 1,39746688 | 314,09 | 52.163 | 408,27 | 90,18 | 55.226 |
| FLRM agrupado com JEPA | 1,39707956 | 445,15 | 36.805 | 470,00 | 88,79 | 39.447 |

- Loss/tempo menores = melhores; velocidade maior = melhor; VRAM menor = menor
  consumo para a mesma tarefa. Sem JEPA: throughput de treino +41,73%, isolado
  +40,00%, tempo −29,44%, VRAM −13,13% e FLOPs estimados −27,13% frente a com JEPA.
  Com JEPA: loss completa ligeiramente melhor, diferença 0,00038732, redução
  de perplexidade de apenas 0,03872%. Uma seed, com consumo de RNG/dropout distinto
  na construção do objetivo, não estabelece melhoria consistente ou dano geral.
- Console: loss total de treino com JEPA 1,454003 inclui LM 1,393200 e auxiliar
  ponderado; sem JEPA 1,388974 é só LM. Esses totais não medem o mesmo objetivo.
  Na validação parcial, sem JEPA ganhou (1,367769 vs 1,369742); o conjunto completo
  inverte a pequena diferença. Só o checkpoint final foi avaliado completamente.
- Perfis: 10 warmups/30 amostras consecutivas sem profiler/um passo instrumentado,
  inicialização nova; não replicações independentes. Geração: observação curta,
  prompt 14/64 novos tokens/warmup 8/batch 1/sem cache. Variação de 1,56% na
  geração não demonstra aceleração causada pela retirada do teacher: ele já
  está ausente na inferência em ambos os modelos.
- Sem JEPA vs Transformer: perplexidade −0,30316%, mas tempo de treino 2,99×,
  VRAM 2,02× e geração 5,22× mais lenta. Ainda há custo de eficiência relevante.
  Recomendação prática: candidato principal sem JEPA, preservar a opção; depois
  parear K=1/K=2 na mesma arquitetura comprimida e repetir seeds 42/43/44. Esta
  análise não alterou modelos, algoritmos ou script nem iniciou novo treino local.

### Preparação da ablação K=1/K=2 sem JEPA — 27/10/2026

- Pedido: preparar o próximo teste depois da campanha 0001. Plano prévio e
  aceitação em `docs/PLAN_K_ABLATION.md`. Aliases explícitos
  `flrm_compressed_grouped_no_jepa_k1` e `_k2` fixam iterations=1/2, arquitetura
  `compressed_grouped`, jepa_weight=0 e nenhuma outra perda auxiliar. K do alias
  prevalece sobre --iterations global; aliases antigos mantêm o controle global.
- Defaults/célula: K1, K2, Transformer pequeno, seed 42, 2.000 passos. GPU 0
  recebe K1 e depois Transformer, GPU 1 recebe K2; filas paralelas preservadas.
  Manifesto registra o K resolvido da configuração, não o argumento global.
  Transformer declara uma passagem; sua profundidade permanece em n_layers.
  Perfis/geração/avaliação sequenciais na mesma GPU; labels e perfis distinguem K.
- Novo teste pareado valida configs iguais exceto iterations/output_dir, mesmo
  state_dict inicial e primeiro batch, mesmo total de parâmetros e ausência de
  auxiliares. Também confere override do alias mesmo com iterations global=7,
  preservação dos aliases antigos e estimativa de operações maior para K=2.
- Enumeração CPU do perfil completo D=384/H=1024/vocab=257: **2.168.076 backbone/
  total** em cada FLRM, conferindo fórmula e pesos iniciais idênticos. Transformer
  pequeno: **2.164.224 backbone/total**. Estimativa de treino por sequência:
  5.205.196.800 / 10.258.808.832 / 3.624.468.480 FLOPs para K1/K2/Transformer,
  respectivamente. São estimativas com convenções/exclusões existentes, não
  medições de velocidade. Log `runs/k-ablation-parameters.log`, sem treino desse perfil.
- Direcionados: **46 passed, 8 skipped**, 30,56 s,
  `runs/k-ablation-targeted-tests.log`. Acrescentados memorização K1 e retomada
  exata CPU FP32/math SDPA K1 agrupado sem JEPA, dropout/acumulação/checkpointing
  ativos, comparando próximo optimizer-step e todos os estados determinísticos.
  Causalidade/saídas/gradientes K1/K2 seguem na suíte existente.
- `python -m pytest -q`: **150 passed, 16 skipped**, 97,88 s,
  `runs/k-ablation-full-tests.log`, Python 3.12.2/Torch 2.14.1+cpu. Skips são
  CUDA indisponível. Novo caso CUDA FP32 K1 save/continuação está coletado, ainda
  não executado localmente; será selecionado pelo notebook antes do treino.
  Evidência anterior da campanha 0001 cobre K2, não valida empiricamente K1.
- Integração CPU real: três treinos minúsculos em processos separados, D=16/H=32,
  contexto 17, dropout .1, acumulação 2, um optimizer-step; avaliação completa de
  1.003 alvos/modelo, geração/diagnósticos, três perfis com 1 warmup/2 amostras/1
  trace, relatório e ZIP **complete**. Manifesto e summaries confirmam K=1/2/1;
  nenhum auxiliar e contagem total igual ao backbone. Artefatos em
  `runs/k-ablation-smoke-4821197d9c/llm2-campaign-0002`, log `runs/k-ablation-smoke.log`.
  Nenhuma medição de ganho GPU ou qualidade final antes da execução Kaggle.
- Compileall e git diff --check passaram. Nenhuma alteração no algoritmo do
  backbone/Trainer, dados ou verificação de checkpoints. Cada modelo começa do
  zero; run_number=2 preserva a campanha 0001. Igual orçamento de alvos não é
  igual computação; K altera consumo de dropout/RNG. Uma seed é triagem.

### Campanha 0002 recebida — repetição JEPA, não ablação de K

- O ZIP entregue como novos resultados 0002 executou **f16765f6e350b343de782f848ee91ed32b258796**,
  não o commit 2c0eb18 preparado para K1/K2. Configs e summaries mostram FLRM
  sem JEPA K=2, FLRM com JEPA K=2 e Transformer pequeno com uma passagem.
  O número do ZIP não identifica o experimento; não relatar resultado K=1 aqui.
- 118 entradas, 6.481.621 bytes, CRC íntegro, SHA256
  `58ef199573f75ab0ee04400829e10abefed285a5e5386d81f452671ae79d3f69`.
  205 verificações da análise: snapshots/configs conferem com Git; dataset
  original e mesmas 437 janelas/111.872 alvos; identidades declaradas dos pesos
  coincidem. Tensores ausentes do ZIP, não carregados localmente.
- Três treinos completos 2.000 passos/16.384.000 alvos, seed 42, FP32/eager.
  CPU **147 passed, 15 deselected**, 168,80 s; CUDA FP32 **11 passed,
  151 deselected**, 29,64 s, correspondentes ao código anterior. O teste CUDA K1
  novo não é evidência fornecida por esta campanha.
- Loss completa (menor = melhor): sem JEPA K2 **1,39746688**, com JEPA K2
  **1,39707955**, Transformer **1,40050012**. Treino alvos/s (maior = melhor):
  **51.305 / 37.381 / 156.691**; VRAM alocada MiB (menor consumo):
  **408,27 / 470,00 / 202,16**; geração tok/s **92,58 / 92,80 / 478,19**.
  Tempos de passos de treino: **319,35 / 438,30 / 104,56 s**.
- Perfis isolados na mesma T4/GPU 0, 10 warmups/30 amostras: sem JEPA
  **53.504,30**, com JEPA **38.040,93**, Transformer **159.872,10 alvos/s**.
  Desligar JEPA deu +40,65% no perfil, −13,13% VRAM e −27,14% tempo de treino.
  JEPA preservou vantagem minúscula de 0,03873% na perplexidade completa.
- Comparação independente com 0001: mesmos hashes de fonte, dataset, pacotes
  e configs funcionais (excluídos somente caminhos); mesma seed. Deltas de loss:
  sem JEPA −5,73e-9, com JEPA −1,09e-8, Transformer −2,94e-6. Confirma repetibilidade
  dessa seed, não é replicação com seed independente nem teste K1/K2.
- Reenviada célula completa da revisão 2c0eb188f1c648938605f28bb959da34143b214d,
  com aliases `_no_jepa_k1`/`_no_jepa_k2`, referência Transformer e run_number=3
  para preservar 0002. Código já foi implementado/testado; nenhuma mudança de
  treino nesta análise, somente registro das evidências e classificação correta.

### Evidência Kaggle — campanha 0003, K=1/K=2 sem JEPA

- Fonte correta e limpa `2c0eb188f1c648938605f28bb959da34143b214d`; ZIP com 118
  entradas, 5.495.752 bytes, CRC íntegro, SHA256
  `c874b942aa8e010e9b8017b29a7c36eec69a09a32f5a2e21991e27e5872f3482`.
  Configs, manifesto e summaries concordam em K=1/2/1. Esta campanha contém
  efetivamente a ablação solicitada, diferente da repetição 0002.
- Três treinos completos de 2.000 passos/16.384.000 alvos, seed 42, contexto 256,
  FP32/eager. FLRMs compressed_grouped sem JEPA/MTP/auxiliares, latent_steps=0,
  configs iguais exceto iterations/output_dir. Ambos 2.168.076 backbone/total,
  zero congelados; Transformer pequeno 2.164.224 backbone/total. Não há teacher.
- Dataset original `beed0146113d17994cd94ddd246e8cabecf6c5eceb005c394526884df66a3ef7`;
  corpus bruto e splits documentais conferidos. Mesmas 437 janelas/111.872 alvos
  completos avaliados; NLL reconstruída dos JSONLs com erro máximo 9,18e-9.
  Código/configs/snapshots conferem byte a byte com Git. Identidades declaradas
  dos pesos coincidem entre checkpoint/avaliação/benchmark; tensores ausentes,
  não carregados ou reexecutados localmente. Auditoria realizou 212 verificações.
- CPU **150 passed, 16 deselected**, 161,49 s; CUDA FP32 **12 passed,
  154 deselected**, 31,78 s. Novo save/continuação K1 agrupado sem JEPA agora
  tem evidência real na T4. Ambiente Python 3.13.15/Torch 2.11.0+cu128/CUDA 12.8,
  FP32/eager; não estender a BF16/Inductor ou retomada bit a bit CUDA.
- T4 distintas nos treinos K1/K2; Transformer depois de K1 na GPU 0. Todos os
  perfis, avaliações e geração foram sequenciais na mesma GPU 0, com UUID igual
  nos perfis. 10 warmups, 30 amostras consecutivas sem profiler e trace separado.

| Modelo | Loss completa ↓ | Perplexidade ↓ | Treino s ↓ | Treino alvos/s ↑ | VRAM MiB ↓ | Geração tok/s ↑ | Perfil isolado alvos/s ↑ |
|---|---:|---:|---:|---:|---:|---:|---:|
| Transformer pequeno | 1,40049951 | 4,05722608 | 105,25 | 155.660 | 202,16 | 494,50 | 164.255 |
| FLRM K1 sem JEPA | 1,40755594 | 4,08595688 | 164,21 | 99.777 | 259,71 | 169,48 | 105.472 |
| FLRM K2 sem JEPA | 1,39746689 | 4,04494069 | 315,79 | 51.882 | 408,27 | 91,83 | 54.507 |

- Loss/perplexidade/tempo menores = melhores; velocidades maiores = melhores;
  VRAM menor = menor consumo na mesma tarefa, allocator PyTorch, não total da
  placa. K2 melhora 0,01008905 em loss/1,00383% perplexidade contra K1, com quase
  duas vezes o custo de treino. K1: +93,50% throughput isolado, −48,00% tempo,
  −36,39% VRAM e +84,55% geração contra K2. Não é vantagem gratuita de recursão.
- Contra Transformer, K1 tem perplexidade +0,70814%, tempo 1,56×, VRAM 1,28×,
  geração 2,92× mais lenta; é dominado nos critérios medidos deste ensaio. K2 tem
  perplexidade −0,30280%, mas tempo 3,00×, VRAM 2,02× e geração 5,38× mais lenta.
  Uma seed/igual quantidade de alvos não estabelecem vantagem geral por orçamento
  de tempo/FLOPs. Os três melhores parciais foram no passo 2000; intermediários
  não têm avaliação completa fornecida. K muda RNG/dropout e trajetória.
- Trace por passo instrumentado: 782 launches Transformer, 2.095 K1, 4.183 K2.
  Isso motiva, sem provar ganho, próximo perfil de microbatch/acumulação 8×4,
  16×2 e 32×1 mantendo 32 sequências por optimizer-step. Não implementado nem
  medido nesta análise; requer verificar amostragem/gradientes/retomada/qualidade.
  Depois, replicar candidatos com seeds 42/43/44 e orçamentos de tempo pareados.
- Geração é observação curta batch 1, prompt 14, 64 novos tokens, warmup 8,
  sem KV cache. Otimização do lote de treino não resolve diretamente a geração.
  Nenhuma mudança de modelos/Trainer/código de treino nesta análise, só registro
  da evidência. K2 sem JEPA permanece candidato de qualidade; Transformer é a
  referência de eficiência, sem alegar uma vitória global do FLRM.

## Campanha 0004 — Estudo de microbatch e acumulação (8×4, 16×2, 32×1), 27/10/2026

- Plano formal: [PLAN_BATCH_PROFILING.md](docs/PLAN_BATCH_PROFILING.md).
- Implementação de matriz e overrides de lotes: `campaign_config` aceita `batch_size` e `grad_accum_steps`. A CLI da campanha aceita `--batch-size`, `--grad-accum-steps`, `--profile-batches` e `--batch-matrix`. O bootstrap `scripts/kaggle_notebook.py` aceita `mode="perfilar_lotes"`.
- Invariantes de lote efetivo e amostragem: preservado lote efetivo de 32 sequências (8.192 alvos supervisionados em contexto 256) em todos os formatos pareados (`8×4`, `16×2` e `32×1`). Confirmada a invariância estrita da amostragem em `RandomBatchStream`: os sorteios de números aleatórios pelo gerador CPU determinístico geram a sequência idêntica de janelas por optimizer-step nos três formatos.
- Deduplicação de perfis ajustada: `profile_runs` agora deduplica pela tupla `(model, batch_size, grad_accum_steps)`, permitindo medir com precisão diferentes formatos do mesmo modelo na mesma campanha sem descartes indesejados.
- Testes locais CPU: suíte completa `python -m pytest -q`: **155 passed, 16 skipped** em 125,16 s. Testes direcionados em `tests/test_batch_profiling.py` cobriram invariância do sampler, validação de argumentos, formatação da tabela de resultados e um teste de fumaça (smoke test) de integração em CPU.
- Skips locais são exclusivamente testes CUDA indisponíveis no ambiente local sem GPU. O speedup empírico real e o pico de VRAM nas Tesla T4 serão medidos na execução da Campanha 0004 no Kaggle.

### Evidência Kaggle recebida — campanha 0004 (Matriz de Lotes 8×4, 16×2, 32×1), 27/10/2026

- ZIP recebido do usuário (`llm2-campaign-0004-results.zip`), 58 entradas, CRC íntegro, SHA256 `4131be102a55c10599ba1a759a5ae88e86394492585639149d76162ecd23c0a5`. Executou o commit `95fd6bf2100abe6626b04e1c6fe8a03fea9a4840` na branch `experiment/fractal-lrm` em checkout limpo.
- Dataset fingerprint original `beed0146113d17994cd94ddd246e8cabecf6c5eceb005c394526884df66a3ef7`. Testes na máquina Kaggle antes dos perfis: CPU **155 passed, 16 deselected** (164,87 s); CUDA FP32 **12 passed, 159 deselected** (31,59 s). Ambiente: Python 3.13.15, Torch 2.11.0+cu128, CUDA 12.8, Tesla T4 (GPU 0 isolada).
- Nove perfis executados sequencialmente na mesma Tesla T4 (10 warmups, 30 amostras cronometradas sem profiler + 1 passo instrumentado com trace), mantendo o lote efetivo fixo em 32 sequências (8.192 alvos supervisionados por passo em $T=256$):

| Modelo | Formato (`B×A`) | Throughput (`alvos/s` ↑) | Tempo/passo (`ms` ↓) | VRAM alocada (`MiB` ↓) | `cudaLaunchKernel` / passo ↓ | GFLOPs analíticos / passo |
|---|:---:|---:|---:|---:|---:|---:|
| Transformer pequeno | `8×4` | 158.782 | 51,6 ms | 202,2 MiB | 817 | 115,98 |
| Transformer pequeno | `16×2` | 179.930 | 45,5 ms | 298,4 MiB | 457 | 115,98 |
| Transformer pequeno | `32×1` | 188.981 | 43,3 ms | 491,9 MiB | 269 | 115,98 |
| FLRM agrupado sem JEPA ($K=1$) | `8×4` | 106.040 | 77,3 ms | 259,7 MiB | 2.130 | 166,57 |
| FLRM agrupado sem JEPA ($K=1$) | `16×2` | 118.299 | 69,2 ms | 412,3 MiB | 1.150 | 166,57 |
| FLRM agrupado sem JEPA ($K=1$) | `32×1` | 124.732 | 65,7 ms | 717,1 MiB | 642 | 166,57 |
| FLRM agrupado sem JEPA ($K=2$) | `8×4` | 54.027 | 151,6 ms | 408,3 MiB | 4.218 | 328,28 |
| FLRM agrupado sem JEPA ($K=2$) | `16×2` | 59.983 | 136,6 ms | 700,4 MiB | 2.242 | 328,28 |
| FLRM agrupado sem JEPA ($K=2$) | `32×1` | 62.456 | 131,2 ms | 1.285,1 MiB | 1.162 | 328,28 |

- Conclusões empíricas da Campanha 0004:
  - Mudar de `8×4` para `32×1` reduziu os lançamentos de kernel CUDA em **~3,6×** (de 4.218 para 1.162 no FLRM $K=2$), mas acelerou o throughput do FLRM $K=2$ em apenas **+15,6%** (131,2 ms vs 151,6 ms), mantendo a distância para o Transformer em **3,03×** (131,2 ms vs 43,3 ms).
  - Isso demonstra que no formato `32×1` o gargalo dominante do FLRM $K=2$ é o **volume real de operações na GPU** (328,28 GFLOPs vs 115,98 GFLOPs do Transformer, 2,83× mais FLOPs por reexecutar o `SharedFractalCore` na escala fina $s_0$ na segunda iteração) combinado ao custo de banda de memória de **1.162 kernels** fragmentados (`312` chamadas `aten::mul` consumindo 22,5 ms de GPU e `89` chamadas `aten::mm` consumindo 57,0 ms).
  - Esta campanha mediu exclusivamente velocidade, VRAM e perfil de operadores; não executou treino de 2.000 passos nem avaliou loss de validação.

## Campanha 0005 — FLRM Ciclo-V Assimétrico + Fusão Vetorizada + NCM Causal, 27/10/2026

- Plano formal, riscos e critérios de aceitação: [PLAN_CAMPAIGN_0005.md](docs/PLAN_CAMPAIGN_0005.md).
- Mudanças implementadas na branch `experiment/fractal-lrm`:
  1. **Recursão Assimétrica Ciclo-V (`v_cycle=True` em `ResearchConfig` / `FractalRecursiveBlock`)**:
     - Na iteração $k=0$ (`step_index == 0`), o `SharedFractalCore` executa sobre a escala fina $s_0$ ($T=256$) e sobre o par agrupado de escalas macro comprimidas $(s_1, s_2)$ ($T/4=64$ e $T/16=16$).
     - Nas iterações $k \ge 1$ (`step_index > 0`), o modelo preserva $v_0 = s_0$ e executa o `SharedFractalCore` **exclusivamente nas escalas macro comprimidas** ($s_1, s_2$), propagando o refinamento semântico de volta à escala fina $s_0$ via `upsample_causal` + `CrossScaleInteraction.forward_native`.
     - O bloco continua retornando a 3-tupla `(next_s0, next_s1, next_s2)` em todas as iterações, mantendo compatibilidade total com `diagnose_model`.
  2. **Fusão e Vetorização de Kernels (`fused_film=True`)**:
     - Pré-condicionamento FiLM estático fora do laço de iterações $k$ via `ScaleConditioning.forward_scales` com `torch.addcmul`, eliminando recomputações redundantes de `tanh` sobre o contexto estático.
     - Empacotamento vetorizado em `CrossScaleInteraction.forward_native`: compressão empilhada (`compress_stacked`), mistura $3 \times 3$ em lote único via `torch.einsum`, concatenação temporal das 3 escalas para executar `self.projection` em **1 única chamada `aten::mm`** por iteração (em vez de 3) e aplicação do gate residual via `torch.addcmul`.
  3. **Banco de Protótipos Causais NCM (`src/llm2/architectures/ncm.py`, `ncm_slots=16`)**:
     - Módulo `NeuralContextMemory` com matriz aprendível `self.memory` $\in \mathbb{R}^{M \times D}$ ($16 \times 384 = 6.144$ parâmetros, $+0,28\%$), aplicado de forma estritamente ponto a ponto (causal) nas escalas macro comprimidas ($s_1, s_2$).
  4. **Contabilidade Analítica de Parâmetros e FLOPs**:
     - `fractal_parameter_formula` atualizada com `+ research.ncm_slots * d` (`flrm_vcycle_k2`: **2.168.076** parâmetros; `flrm_vcycle_ncm_k2`: **2.174.220** parâmetros, conferidos por enumeração e fórmula).
     - `FractalLatentModel.analytical_cost` atualizado para descontar a escala fina nas iterações $k \ge 1$ quando `v_cycle=True` e somar os FLOPs exatos do NCM quando `ncm_slots > 0`. No lote `32×1` ($T=256$), o custo analítico por optimizer-step cai de **328,28 GFLOPs** (`flrm_compressed_grouped_no_jepa_k2`) para **224,40 GFLOPs** (`flrm_vcycle_k2`, **−31,6%**) e **225,61 GFLOPs** (`flrm_vcycle_ncm_k2`, **−31,3%**).
- Verificação em CPU com `torch.profiler` ($T=256$):
  - O total de operadores despachados por passo caiu de **5.320** (`flrm_k2_legacy`) para **4.144** (`flrm_vcycle_k2`, **−1.176 ops**) e **4.551** (`flrm_vcycle_ncm_k2`).
  - As chamadas `aten::mul` caíram de **302** para **184** (**−39,1%**), e as multiplicações de matriz (`mm`/`bmm`) caíram de **120** para **90** (**−25,0%**).
- Testes executados localmente (CPU, Python 3.12.2, Torch 2.14.1+cpu):
  - `python -m pytest tests/test_vcycle_fractal.py -q`: **12 passed, 2 skipped** (testes de contagem e fórmula de parâmetros, equivalência numérica forward/backward da fusão nativa, redução de formas e FLOPs analíticos, causalidade estrita sob mutação futura e truncamento de prefixo, retomada exata bit a bit com dropout/acumulação/checkpointing ativos, memorização com gradiente finito nos slots NCM e diagnósticos).
  - Suíte completa `python -m pytest -q`: **167 passed, 18 skipped**. Todos os 18 skips são testes marcados com `@pytest.mark.cuda` (GPU ausente localmente; serão executados na etapa 2 da célula Kaggle nas Tesla T4).
  - Limitações: a redução real de `cudaLaunchKernel` e do tempo por passo em ms na Tesla T4 será verificada empiricamente na triagem rápida (`mode="perfilar_lotes"`, `batch_matrix=["32x1"]`) da Campanha 0005 no Kaggle.

### Evidências Kaggle recebidas — Campanhas 0005 (Triagem 32×1) e 0006 (Treino 2.000 passos), 27/10/2026

- **Campanha 0005 (`llm2-campaign-0005-results.zip`)**: SHA256 `c4b581d091ef7f21a8b911b1fb9d520425b513b72a862f370577506711b701fa` (52 entradas, commit `24545f19f977725653f3578c9d3b3d836784fb24`).
  - Testes no Kaggle: CPU **167 passed, 18 deselected** (231,66 s); CUDA FP32 **14 passed, 171 deselected** (39,73 s).
  - Perfil isolado em lote `32×1` na Tesla T4 (GPU 0):
    - `baseline_small`: **191.417 alvos/s**, **42,8 ms/passo**, **491,9 MiB** VRAM, **269** `cudaLaunchKernel`, **115,98 GFLOPs**.
    - `flrm_compressed_grouped_no_jepa_k2` (Legado): **63.284 alvos/s**, **129,4 ms/passo**, **1.285,1 MiB** VRAM, **1.162** `cudaLaunchKernel`, **328,28 GFLOPs**.
    - `flrm_vcycle_k2`: **78.308 alvos/s** (**+23,7%**), **104,6 ms/passo** (**−24,8 ms**), **1.071,0 MiB** VRAM (**−214,1 MiB**), **882** `cudaLaunchKernel` (**−280**), **224,40 GFLOPs** (**−31,6%**).
    - `flrm_vcycle_ncm_k2`: **76.253 alvos/s** (**+20,5%**), **107,4 ms/passo** (**−22,0 ms**), **1.096,3 MiB** VRAM (**−188,8 MiB**), **961** `cudaLaunchKernel` (**−201**), **225,61 GFLOPs** (**−31,3%**).
- **Campanha 0006 (`llm2-campaign-0006-results.zip`)**: SHA256 `ef252090ea6c70cca40ca0bcf615b79b667df1903689718324654779615ec886` (118 entradas, mesmo commit `24545f19f977725653f3578c9d3b3d836784fb24`, 2.000 passos / 16.384.000 alvos em lote `32×1`, seed 42):
  - `baseline_small`: Loss completa **1,39921737**, treino **169.663 alvos/s** (96,57 s), VRAM **491,93 MiB**, geração **415,83 tok/s**.
  - `flrm_vcycle_k2`: Loss completa **1,40676885**, treino **72.893 alvos/s** (224,77 s, **−28,8%** vs 315,79 s da Campanha 0003), VRAM **1.070,97 MiB**, geração **112,56 tok/s**.
  - `flrm_vcycle_ncm_k2`: Loss completa **1,40612024**, treino **71.348 alvos/s** (229,64 s), VRAM **1.096,29 MiB**, geração **105,76 tok/s**.
  - Diagnóstico científico: o NCM-16 melhorou a loss do Ciclo-V (`1,406120` vs `1,406769`), mas pular o `SharedFractalCore` na escala fina $s_0$ na última iteração ($k=1$) limitou a atualização relativa de $s_0$ a apenas `0,1051` (10,5%), contra `4,88` e `5,51` nas escalas macro, impedindo que a escala fina sintetizasse o contexto macro antes da cabeça de linguagem (`lm_head`).

## Campanha 0007 — FLRM Simétrico Fused+NCM vs Ciclo-V Invertido (Macro-First) + Atalho Algébrico Nativo, 27/10/2026

- Plano formal, riscos e critérios de aceitação: [PLAN_CAMPAIGN_0007.md](docs/PLAN_CAMPAIGN_0007.md).
- Mudanças implementadas na branch `experiment/fractal-lrm`:
  1. **Ciclo-V Invertido / "Macro-Primeiro" (`inv_vcycle=True` em `ResearchConfig` / `FractalRecursiveBlock`)**:
     - Na iteração $k=0$ (`step_index == 0`), o modelo salta a escala fina ($v_0 = s_0 = 0$) e roda o `SharedFractalCore` + NCM **exclusivamente nas escalas macro comprimidas** ($s_1, s_2$, apenas $64 + 16 = 80$ posições em $T=256$), injetando o contexto macro causal em `next_s0` via `CrossScaleInteraction.forward_native`.
     - Na iteração $k=1$ (`step_index > 0`), o `SharedFractalCore` roda em **todas as escalas** ($s_0, s_1, s_2$), garantindo que a escala fina $s_0$ processe Atenção + SwiGLU na última iteração já alimentada pela abstração macro de $k=0$, mantendo exatamente o mesmo orçamento reduzido de FLOPs do Ciclo-V (~225,61 GFLOPs/passo em `32×1`).
  2. **FLRM Simétrico com Fusão + NCM-16 (`flrm_fused_ncm_k2`)**:
     - Combina a recursão simétrica completa ($K=2$ em todas as escalas) com `fused_film=True` e `ncm_slots=16` para testar o teto de qualidade em loss e medir o ganho puro da fusão vetorizada sem poda de escala.
  3. **Atalho Algébrico Exato em `CrossScaleInteraction.forward_native` (`raw_macro`)**:
     - Em vez de recomprimir em comprimento $T=256$ os tensores `u1 = comp4.upsample_causal(v1, T)` e `u2 = comp16.upsample_causal(v2, T)` nas escalas $T/4$ e $T/16$, utiliza as identidades exatas $\text{comp}_4(\text{upsample}_4(v_1, T)) = \text{shift}_4(v_1)$, $\text{comp}_4(\text{upsample}_{16}(v_2, T)) = \text{repeat\_interleave}(\text{shift}_{16}(v_2), 4)[:T_1]$ e $\text{comp}_{16}(\text{upsample}_{16}(v_2, T)) = \text{shift}_{16}(v_2)$ diretamente nas resoluções comprimidas nativas ($T_1=64, T_2=16$), eliminando 3 compressões redundantes de comprimento $T=256$ por iteração com equivalência numérica exata.
- Testes executados localmente (CPU, Python 3.12.2, Torch 2.14.1+cpu):
  - `python -m pytest tests/test_vcycle_fractal.py -q`: **13 passed, 2 skipped** (incluindo equivalência numérica exata do atalho `raw_macro` em comprimentos arbitrários `[1, 4, 15, 16, 17, 31, 32, 33, 65]`, formas `[(4, 9), (2, 33), (4, 9)]` do `inv_vcycle`, causalidade estrita e retomada exata bit a bit).
  - Suíte completa `python -m pytest -q`: **168 passed, 18 skipped**. Todos os 18 skips são testes `@pytest.mark.cuda`.

### Evidência Kaggle recebida — Campanha 0007 (`flrm_fused_ncm_k2`, `flrm_inv_vcycle_ncm_k2`, `baseline_small` em `32×1`), 27/10/2026

- ZIP recebido (`llm2-campaign-0007-results.zip`), SHA256 `59176855081d7469b9b06f07a75ee217d9b66a1e40bc521a1ceb39bf19f6ef63` (118 entradas, commit `e8c261e2569aa8e4e82efd034f5ff7b9c73e5e84`).
- Testes no Kaggle: CPU **168 passed, 18 deselected** (187,74 s); CUDA FP32 **14 passed, 172 deselected** (31,56 s).
- Resultados em 2.000 passos / 16.384.000 alvos supervisionados, lote `32×1`, seed 42, validação completa (437 janelas / 111.872 alvos):
  - `baseline_small`: Loss completa **1,39921776**, treino **152.344 alvos/s** (107,55 s), perfil isolado **188.015 alvos/s** (43,6 ms/passo, 269 `cudaLaunchKernel`), VRAM **491,93 MiB**, geração **398,17 tok/s**.
  - `flrm_fused_ncm_k2`: Loss completa **1,40102090**, treino **48.002 alvos/s** (341,32 s), perfil isolado **52.931 alvos/s** (154,8 ms/passo, 1.078 `cudaLaunchKernel`), VRAM **1.425,32 MiB**, geração **64,95 tok/s**.
  - `flrm_inv_vcycle_ncm_k2`: Loss completa **1,54726840**, treino **71.462 alvos/s** (229,27 s), perfil isolado **79.313 alvos/s** (103,3 ms/passo, 899 `cudaLaunchKernel`), VRAM **1.122,82 MiB**, geração **108,82 tok/s**.
- Diagnóstico pós-Campanha 0007:
  1. `flrm_inv_vcycle_ncm_k2` degradou severamente (`1,547268`) porque desligar o `SharedFractalCore` na escala fina $s_0$ em $k=0$ reduz a profundidade efetiva de atenção/MLP em resolução $T=256$ de $12$ para $6$ camadas.
  2. A comparação entre `flrm_compressed_grouped_no_jepa_k2` na `0003` (`1,397467` em `8×4`) e `flrm_fused_ncm_k2` na `0007` (`1,401021` em `32×1`) confundiu simultaneamente formato do lote (`8×4` vs `32×1`), fusão (`fused_film`), NCM (`ncm_slots=16`) e um deslocamento de RNG nos pesos iniciais comuns causado pela chamada de `nn.init.normal_` dentro de `NeuralContextMemory.__init__` antes de `self.apply(TransformerLM._init_weights)`.
  3. Além disso, o empacotamento `stacked_all` + `torch.einsum("ij,jbtd->ibtd", weights, stacked_all)` em `CrossScaleInteraction.forward_native` calculava todas as 3 combinações de escala para todas as 336 posições ($1.008$ posições-escala) e descartava $2/3$ do tensor resultante, explicando o aumento de VRAM para `1.425,32 MiB` e a queda de throughput para `48.002 alvos/s`.

## Campanha 0008 — Correção de Empacotamento, Isolamento de RNG do NCM e Matriz Fatorial 2×2, 27/10/2026

- Mudanças implementadas na branch `experiment/fractal-lrm`:
  1. **Eliminação do cálculo descartado de $2/3$ em `CrossScaleInteraction.forward_native`**:
     - Substituído `stacked_all = torch.cat((s_fine, c1_3, c2_3), dim=2)` + `torch.einsum("ij,jbtd->ibtd", weights, stacked_all)` por três contrações diretas por linha (`m0 = torch.einsum("j,jbtd->btd", weights[0], s_fine)`, `m1 = ...`, `m2 = ...`), concatenando apenas os tensores efetivamente utilizados (`packed_mixed = torch.cat((m0, m1, m2), dim=1)`) antes da projeção linear única `self.projection(packed_mixed)`.
     - Elimina $2/3$ das multiplicações da mistura entre escalas e $\sim 94,5\text{ MiB}$ de alocações intermediárias por invocação em $B=32, T=256$, preservando equivalência numérica estrita (`rtol=3e-5, atol=2e-6`) com o caminho não fundido.
  2. **Isolamento de RNG na inicialização do `NeuralContextMemory`**:
     - `FractalRecursiveBlock` instancia `NeuralContextMemory(..., init_parameters=False)` sem consumir o RNG global antes de `self.apply(TransformerLM._init_weights)` em `FractalLatentModel.__init__`, executando `self.block.ncm.reset_parameters()` imediatamente após `self.apply`.
     - Com isso, variantes com `ncm_slots=0` e `ncm_slots=16` possuem pesos iniciais **100% idênticos bit a bit (`torch.equal`)** em todos os parâmetros comuns (`embedding`, `cores`, `conditioning`, `interaction`, `comp4`, `comp16`, `norm`, `lm_head`) sob a mesma seed.
  3. **Matriz Fatorial $2 \times 2$ Completa (`Fusão × NCM`) e suporte a `--compile`**:
     - Adicionados os aliases `flrm_fused_k2` (`fused_film=True, ncm_slots=0`) e `flrm_ncm_k2` (`fused_film=False, ncm_slots=16`) em `campaign.py` e `presentation.py`, completando o desenho fatorial ortogonal ao lado de `flrm_compressed_grouped_no_jepa_k2` (`False, 0`) e `flrm_fused_ncm_k2` (`True, 16`).
     - Adicionado argumento opcional `--compile` (`compile=False` por padrão) em `campaign.py` e `scripts/kaggle_notebook.py`.
- Testes executados localmente (CPU, Python 3.12.2, Torch 2.14.1+cpu):
  - `python -m pytest -q`: **169 passed, 18 skipped** em 131,99 s (incluindo `test_ncm_preserves_bit_identical_common_initial_weights`, `test_fused_film_and_native_numerical_equivalence` e `test_campaign_configs_for_vcycle_models`). Todos os 18 skips são testes `@pytest.mark.cuda`.

### Evidência Kaggle recebida — Campanha 0008 (Opção 1: Reprodução do Legado em 8×4), 27/10/2026

- ZIP recebido (`llm2-campaign-0008-results.zip`), SHA256 `b77b7e2a7d470268222347ade53214423a8ce5d01dbd33fdc36d6df7c099314b` (83 entradas, commit `7bcab2b93c0e7ffc9800e343162f1f747c0a88c5`).
- Testes no Kaggle: CUDA FP32 **14 passed, 173 deselected** (54,98 s).
- Resultados em 2.000 passos / 16.384.000 alvos supervisionados em lote `8×4`, seed 42, validação completa (437 janelas / 111.872 alvos):
  - `baseline_small`: Loss completa **1,40050305** (vs `1,40049951` na Campanha 0003, delta `+3,5e-6`), treino **160.828 alvos/s** (101,87 s), perfil isolado **163.539 alvos/s** (50,09 ms/passo), VRAM **202,16 MiB**, geração **507,7 tok/s**.
  - `flrm_compressed_grouped_no_jepa_k2`: Loss completa **1,39746688** (vs `1,39746689` na Campanha 0003, delta `-1,1e-8`, reprodução exata), treino **52.955 alvos/s** (309,39 s), perfil isolado **53.870 alvos/s** (152,07 ms/passo), VRAM **408,27 MiB**, geração **90,3 tok/s**.
- Conclusão: a vantagem numérica de loss do FLRM simétrico $K=2$ sobre o Transformer pequeno em lote `8×4` (`1,397467` vs `1,400503`, delta de **−0,303% em perplexidade**) foi reproduzida na revisão atual com diferença de métrica de `-1,1e-8` (reprodução numérica da métrica; os hashes de `model.safetensors` diferem entre os commits `2c0eb188` da 0003 e `7bcab2b9` da 0008 devido à evolução do código). O modelo levou **3,037× mais tempo** que o Transformer (`309,39 s` vs `101,87 s`). A Opção 2 no lote `32×1` isolará o formato do lote e a ablação fatorial 2×2 (Fusão × NCM).

### Evidência Kaggle recebida — Campanha 0009 (Opção 2: Isolamento 32×1 e Ablação Fatorial 2×2 Fusão × NCM), 28/10/2026

- ZIP recebido (`llm2-campaign-0009-results.zip`), SHA256 `4e056e2744f00a70470bdf1cfac7e250fa9bad37f1c708dd47d502c59ed461b0` (185 entradas, commit `04f101d535f108b15b1a0e25d1c1bbe56dc610df`).
- Testes no Kaggle: CUDA FP32 **14 passed, 173 deselected**.
- Resultados em 2.000 passos / 16.384.000 alvos supervisionados em lote `32×1`, seed 42, validação completa (437 janelas / 111.872 alvos):
  - `flrm_compressed_grouped_no_jepa_k2` (Variante A — sem fusão, sem NCM): Loss completa **1,39673830**, treino **53.330 alvos/s** (307,22 s), perfil isolado **57.830 alvos/s** (141,66 ms/passo, 1.162 `cudaLaunchKernel`), VRAM **1.285,14 MiB**, geração **91,39 tok/s**.
  - `flrm_fused_k2` (Variante B — com fusão corrigida, sem NCM): Loss completa **1,39673829** (delta `-1,5e-8` vs A), treino **52.384 alvos/s** (312,77 s), perfil isolado **56.490 alvos/s** (145,02 ms/passo, 1.078 `cudaLaunchKernel`), VRAM **1.321,13 MiB**, geração **102,41 tok/s**.
  - `flrm_ncm_k2` (Variante C — sem fusão, NCM-16 com RNG isolado): Loss completa **1,39668728**, treino **52.156 alvos/s** (314,14 s), perfil isolado **54.994 alvos/s** (148,96 ms/passo, 1.241 `cudaLaunchKernel`), VRAM **1.310,21 MiB**, geração **86,27 tok/s**.
  - `flrm_fused_ncm_k2` (Variante D — com fusão corrigida + NCM-16 com RNG isolado): Loss completa **1,39668727** (delta `-1,5e-8` vs C), treino **49.377 alvos/s** (331,81 s), perfil isolado **52.979 alvos/s** (154,63 ms/passo, 1.157 `cudaLaunchKernel`), VRAM **1.347,20 MiB**, geração **94,17 tok/s**.
  - `baseline_small` (Transformer pequeno): Loss completa **1,39921744**, treino **154.456 alvos/s** (106,08 s), perfil isolado **158.597 alvos/s** (51,65 ms/passo, 269 `cudaLaunchKernel`), VRAM **491,93 MiB**, geração **494,68 tok/s**.
- Conclusões da ablação fatorial 2×2:
  1. O isolamento de RNG em `NeuralContextMemory` resolveu integralmente a regressão observada na Campanha 0007 (`1,401021` $\to$ `1,396687`), provando que a perda de qualidade anterior vinha exclusivamente do deslocamento da inicialização aleatória dos pesos do tronco compartilhado.
  2. No formato `32×1`, tanto o FLRM $K=2$ (`1,397467` $\to$ `1,396738`) quanto o Transformer (`1,400503` $\to$ `1,399217`) melhoraram em relação ao `8×4`, mantendo uma vantagem de `-0,002479` nats (`-0,248%` em perplexidade) para a Variante A na seed 42.
  3. A fusão manual (`fused_film=True`) é matematicamente equivalente ao caminho legado (`delta ~ 1,5e-8`) e acelera a geração batch-1 em `+12%`, mas em modo eager no lote 32 adiciona cópias de concatenação que custam ~3,3 ms/passo e ~36 MiB de VRAM no treino.
  4. Como uma simples mudança de inicialização aleatória dos pesos do tronco (na 0007) deslocou a loss em `+0,004334` nats, a replicação com seeds `43` e `44` em `32×1` é obrigatória antes de atribuir significância estatística aos `-0,002479` nats (FLRM vs Transformer) ou `-0,000051` nats (NCM vs sem NCM).

### Evidência Kaggle recebida — Campanhas 0010 e 0011 (Passo 1: Replicação Multi-Seed 43 e 44 em 32×1), 28/10/2026

- ZIPs recebidos: `llm2-campaign-0010-results.zip` (SHA256 `07d2cc08...`) e `llm2-campaign-0011-results.zip` (SHA256 `5f367749...`), ambos executando commit `2d7eb45674bc4178fdd99517d158ac937aae0828`. A concordância de loss entre a 0010 e a 0011 é da ordem de $10^{-8}$.
- Consolidação das 3 seeds (`42` da 0009 + `43, 44` da 0010) em lote `32×1` eager:
  - `baseline_small`: Seed 42 = `1,39921744`, Seed 43 = `1,39380524`, Seed 44 = `1,39626223`. **Média: 1,39642830** ($s = 0,00270990$, $\text{SE} = 0,00156456$); throughput médio nas 3 seeds: **153.342 alvos/s** (`[154.456; 156.539; 149.032]`); geração média eager sem KV cache: **499,8 tok/s**.
  - `flrm_ncm_k2` (Variante C): Seed 42 = `1,39668728`, Seed 43 = `1,39884250`, Seed 44 = `1,39744715`. **Média: 1,39765898** ($s = 0,00109300$, $\text{SE} = 0,00063104$); throughput médio nas 3 seeds: **52.315 alvos/s** (`[52.156; 52.019; 52.770]`); geração média eager sem KV cache: **84,1 tok/s**.
  - `flrm_compressed_grouped_no_jepa_k2` (Variante A): Seed 42 = `1,39673830`, Seed 43 = `1,39907332`, Seed 44 = `1,39771621`. **Média: 1,39784261** ($s = 0,00117263$, $\text{SE} = 0,00067702$); throughput médio nas 3 seeds: **53.087 alvos/s** (`[53.330; 55.312; 50.618]`); geração média eager sem KV cache: **91,9 tok/s**.
- Conclusão estatística e qualificação metodológica:
  1. O NCM-16 apresentou direção consistente nas 3 seeds frente à Variante A (diferença pareada média de `-0,000184 ± 0,000116` NLL/byte, $\text{SE} = 0,000067$), mas seu efeito absoluto é minúsculo frente à distância para o Transformer e adiciona `+6.144` parâmetros e leve custo de tempo/VRAM, permanecendo classificado como candidato de ablação opcional.
  2. Na média das 3 seeds, o Transformer pequeno apresentou menor loss (`1,396428` vs `1,397843`), além de treinar ~2,89× mais rápido. A vantagem do FLRM observada na seed 42 isolada não se repetiu nas seeds 43 e 44.

### Evidência Kaggle recebida — Campanha 0012 (Passo 2: Efeito do torch.compile/Inductor em 32×1, seed 42), 28/10/2026

- ZIP recebido (`llm2-campaign-0012-results.zip`), SHA256 `e1fb6d1da25be6a36d21d4a6a761ca71df6d5c032dab531a52f8a8600a698cb6` (82 entradas, commit `2d7eb45674bc4178fdd99517d158ac937aae0828`).
- Resultados com `compile: True` (`backend: inductor`) vs modo eager (Campanha 0009):
  - `flrm_compressed_grouped_no_jepa_k2`: Loss completa **1,397512** (vs `1,396738` eager), perfil isolado **72.359 alvos/s** / **113,21 ms/passo** (+25,1% vs `57.830 alvos/s`), VRAM alocada **1.011,42 MiB** (−273,7 MiB vs `1.285,1 MiB`), `cudaLaunchKernel` cortado de **1.162 para 290** (−75,0%), treino total **280,33 s** (vs `307,22 s`).
  - `baseline_small`: Loss completa **1,398501** (vs `1,399217` eager), perfil isolado **204.606 alvos/s** / **40,04 ms/passo** (+29,0% vs `158.597 alvos/s`), VRAM alocada **418,90 MiB** (−73,0 MiB vs `491,9 MiB`), `cudaLaunchKernel` cortado de **269 para 136** (−49,4%), treino total **93,93 s** (vs `106,08 s`).
- Conclusão técnica e qualificação do compilador:
  - O Inductor reduziu os lançamentos de kernel CUDA (290 < 350 no FLRM) e acelerou ambos os modelos. A razão de tempo por passo nos perfis isolados ($113,21 / 40,04 = 2,827\times$) alinhou-se à razão de FLOPs analíticos estimados ($328,28 / 115,98 = 2,830\times$), indicando que, após reduzir o despacho de kernels, o custo relativo do passo é fortemente governado pelo trabalho aritmético total (sem implicar ausência absoluta de overheads de memória ou padding).
  - Nota sobre geração na 0012: o comando `benchmark` executa inferência eager sem KV cache em processo separado; variações em `tok/s` (`92,2` vs `91,4` no FLRM; `528,4` vs `494,7` no Transformer) refletem ruído entre sessões, e não compilação na geração.
  - Nota de arquitetura: nos perfis Kaggle do FLRM (`configs/kaggle_t4_1h_fractal.yaml`), a profundidade de cada chamada de `SharedFractalCore` é definida por `research.core_layers = 1` (1 `DecoderBlock` compartilhado), enquanto `model.n_layers = 6` aplica-se ao `TransformerLM`. As hipóteses de desbalanço entre `RMSNorm(state)` e `W_ctx(context)` e de condicionamento por iteração são tratadas como questões empíricas para as Etapas 1, 3 e 4 conforme `docs/PLAN_FINAL_EXECUTION_2026_10_28.md`.

## Plano de Execução Consenso (28/10/2026) — Etapas 1 e 2: Telemetria de Magnitudes/Gradientes, `compressed_native` sem JEPA e `cache_context_projection`

- Plano formal, riscos e critérios de aceitação: [PLAN_FINAL_EXECUTION_2026_10_28.md](docs/PLAN_FINAL_EXECUTION_2026_10_28.md).
- Mudanças implementadas na branch `experiment/fractal-lrm`:
  1. **Etapa 1 — Telemetria Não Invasiva de Magnitudes e Gradientes por Escala e Iteração (`src/llm2/research/diagnostics.py`, `src/llm2/cli.py`)**:
     - `diagnose_model` agora registra `batch_metadata` (`batch_source`, `batch_size`, `seq_len`, `evaluated_tokens`, `eval_mode_dropout_disabled=True`) e `stage1_telemetry` contendo `ratio_definitions`, `iterations` (RMS por escala $s_0, s_1, s_2$ e por iteração $k=1 \dots K$ para `state_in`, `state_norm`, `raw_context`, `conditioned_context`, `projected_context`, `core_input_sum`, `attn_norm_out`, `ffn_norm_out`, `core_branch_delta`, `core_output`, `post_interaction_state`), razões explícitas (`state_norm_over_projected_context`, `projected_context_over_core_input_sum`, `state_norm_over_core_input_sum`, `core_branch_delta_over_state_in` apenas para $k \ge 2$, `core_branch_delta_over_core_output`, `post_interaction_delta_over_core_output`), `padding_telemetry` separado para as 48 posições de right-padding de $s_2$ no modo `compressed_grouped` (excluídas do RMS válido de $s_2$) e `gradients` (RMS e norma L2 de gradientes em relação às ativações retidas e parâmetros compartilhados acumulados, registrando `null` com `null_reason` onde o gradiente não flui, como `state_in` constante zero em $k=1$ e `post_interaction_state` de $s_1, s_2$ na última iteração).
     - `llm2 diagnose` aceita `--checkpoint` ou `--config` (para diagnosticar o modelo na inicialização aleatória na primeira janela de validação de comprimento $T=256$).
  2. **Etapa 2 — Cache Estático da Projeção de Contexto (`cache_context_projection`) e Aliases `compressed_native` sem JEPA**:
     - Adicionado `cache_context_projection: bool = False` em `ResearchConfig`, suportado em `SharedRecursiveCore`, `LatentRecursiveModel` e `FractalLatentModel`. Pré-computa $W_{\text{ctx}}(c)$ uma única vez por sequência antes do laço recorrente $k=1 \dots K$, preservando exatamente os logits, estados ocultos, gradientes de parâmetros e contagem de parâmetros (**2.168.076** em $D=384, H=1024, V=257$).
     - Contabilidade analítica em `FractalLatentModel.analytical_cost`: em $B=32, T=256, K=2$, o custo estimado por optimizer-step passa de **328,281883 GFLOPs** (`G0`: `flrm_compressed_grouped_no_jepa_k2`) para **317,410247 GFLOPs** (`G1`: `flrm_compressed_grouped_cached_no_jepa_k2`, **−3,3117%**), **291,816604 GFLOPs** (`N0`: `flrm_compressed_native_no_jepa_k2`, **−11,1079%** vs `G0`) e **282,303922 GFLOPs** (`N1`: `flrm_compressed_native_cached_no_jepa_k2`, **−14,0056%** vs `G0`).
     - Adicionados os aliases `flrm_compressed_native_no_jepa`, `flrm_compressed_native_no_jepa_k1`, `flrm_compressed_native_no_jepa_k2` (`N0`), `flrm_compressed_grouped_cached_no_jepa_k2` (`G1`), `flrm_compressed_native_cached_no_jepa_k2` (`N1`), `lrm_no_jepa_k1` e `lrm_no_jepa_k2` em `campaign.py` e `presentation.py`.
  3. **Perfilamento Multi-Backend com Ordem Alternada (`profiling.py`, `campaign.py`, `scripts/kaggle_notebook.py`)**:
     - `profile_step` registra `warmup_timing` (incluindo o tempo do 1º passo de compilação/aquecimento e o total de warmup) separadamente de `timing` (média, mediana, desvio-padrão, mínimo e máximo do passo medido) e `cuda_launch_kernel_calls`.
     - `run_batch_profile` suporta `--profile-backends` (`eager`, `inductor`), `--profile-repeats` (alternando a ordem dos modelos a cada repetição na GPU 0) e `--diagnose-init`.
- Testes executados localmente (CPU, Python 3.12.2, Torch 2.14.1+cpu):
  - `python -m pytest tests/test_etapa1_etapa2_execution_plan.py -q`: **8 passed, 3 skipped** (zero efeitos colaterais em pesos/gradientes/modos/RNG, separação do padding de $s_2$, razões e gradientes nulos documentados, equivalência numérica forward/backward de `cache_context_projection` em comprimentos divisíveis e não divisíveis `[1, 4, 15, 16, 17, 31, 32, 33, 65]`, causalidade estrita de prefixo, invariância de parâmetros e GFLOPs analíticos, retomada exata CPU FP32 com dropout e acumulação ativos para `N0`, `G1` e `N1`, e execução de `run_batch_profile` com ordem alternada e diagnóstico de inicialização).
  - Suíte completa `python -m pytest -q`: **177 passed, 21 skipped** em 188,32 s. Todos os 21 skips são testes `@pytest.mark.cuda` (GPU indisponível localmente; serão executados no Kaggle nas Tesla T4).
  - Limitações: os números empíricos de telemetria em $T=256$ e o comparativo de velocidade/VRAM entre `G0`, `G1`, `N0`, `N1` e `baseline_small` sob `eager` e `inductor` na Tesla T4 dependem da execução das células das Etapas 1 e 2 no Kaggle. As Etapas 3 e 4 permanecem aguardando esses resultados.
