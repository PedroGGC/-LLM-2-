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
