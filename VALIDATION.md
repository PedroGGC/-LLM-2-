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
