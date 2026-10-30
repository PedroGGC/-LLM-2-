# Guia operacional

Operação vigente para a próxima sessão: [backup/limpeza e diagnóstico do ciclo 4](CYCLE4_RUN.md).
As instruções de ciclos anteriores continuam para reprodução. Não executar esses
treinos como complemento automático do ciclo 4.

## Interpretador e checkout

Usar Python >=3.10 e instalação direta. Não substituir automaticamente o Torch existente.
O arquivo `requirements/cpu-tested.txt` registra uma combinação CPU testada, não um lock
universal nem requisitos para uma GPU. Instale uma wheel apropriada ao hardware se Torch
ainda não estiver disponível.

```powershell
python -m pip install -e ".[dev,research]"
$env:PYTHONPATH = (Resolve-Path ./src).Path
python -c "import sys, llm2, torch; print(sys.executable); print(llm2.__file__); print(torch.__version__)"
python -m pytest -q
```

No Windows com terminal CP1252, mensagens com símbolos Unicode do runner podem
falhar ao imprimir. Use `$env:PYTHONUTF8 = "1"` antes de iniciar o Python nesse caso.
É uma limitação de apresentação existente, não uma validação do treino.

`PYTHONPATH` vale para a sessão. Se já havia outro valor, guarde-o e restaure-o ao terminar.
Use `python -m pytest -q -m cuda` apenas quando houver GPU compatível; skips não validam CUDA.

## Inspeção sem treino

```powershell
python -m llm2 inspect --config configs/cpu_smoke.yaml --meta
python -m llm2 inspect --config configs/experiments/e2_recursive.yaml --meta
python -m llm2 experiment --config configs/experiments/combined.yaml
python scripts/run_kaggle.py --help
```

`experiment` e `profile` precisam de `--execute`. `train` executa diretamente o treino
solicitado. A CLI de campanha também exige `--execute`; ver [Kaggle](KAGGLE_T4_CAMPAIGN.md).

## Exemplo CPU individual

Este exemplo inicia um smoke test de engenharia, não uma campanha científica. Execute
somente quando desejar esse teste e use nomes novos se os destinos já existirem.

```powershell
python scripts/make_demo_corpus.py --output data/raw/demo-cleanup-example.txt
python -m llm2 prepare --input data/raw/demo-cleanup-example.txt --output data/processed-demo-example --tokenizer byte --seed 42
python -m llm2 train --config configs/cpu_smoke.yaml --data data/processed-demo-example --output-dir runs/demo-example --until-step 10
python -m llm2 train --config configs/cpu_smoke.yaml --data data/processed-demo-example --output-dir runs/demo-example --resume runs/demo-example/checkpoints/step-00000010
python -m llm2 evaluate --checkpoint runs/demo-example/checkpoints/step-00000020
python -m llm2 generate --checkpoint runs/demo-example/checkpoints/step-00000020 --prompt "O laboratório" --seed 42 --max-new-tokens 40
```

Caminhos escritos em YAML são relativos ao YAML; argumentos CLI são relativos ao diretório
atual. O corpus sintético não mede competência linguística. Preparação e execução recusam
sobrescrita. Não reutilizar o exemplo de retomada se já houver checkpoints posteriores.

## Dados e checkpoints

Preparo: documentos separados por linhas vazias, deduplicação exata antes do split,
BPE treinado somente no treino ou tokenizador byte (256 bytes + EOS). Dados preparados:
`manifest.json`, `tokenizer.json`, `train.bin`, `val.bin`. Verificações de hashes e IDs
permanecem obrigatórias. Documentos são concatenados com EOS; atenção pode cruzar fronteiras.

O sampler usa janelas com reposição e RNG próprio. Tokens processados contam alvos com
repetição; não são um contador de exemplos únicos. Validação completa usa todas as janelas
completas e descarta a cauda. A avaliação periódica de treino pode cobrir só `eval_batches`.

Checkpoints são publicados após optimizer-step completo, scheduler e `zero_grad`:
`model.safetensors`, `training_state.pt`, `metadata.json` e, quando há estado auxiliar,
`objective.safetensors`. Preservam objetivo, otimizador,
scheduler, RNGs, sampler e contadores. Estado PyTorch usa pickle: carregar só fontes confiáveis.
Avaliação/geração usam metadata e safetensors, sem desserializar o estado de treino.

Retomada estrita exige código, configuração semântica, dataset e ambiente compatíveis.
Não ampliar silenciosamente o horizonte do scheduler nem remover verificações de hash.
Transferência apenas de pesos é outro experimento. Preserve snapshots e dependências originais.

## Métricas e exportação

```powershell
python -m llm2 benchmark --checkpoint runs/demo-example/checkpoints/step-00000020 --output runs/demo-example/benchmark.json
python -m llm2 report --runs runs/demo-example --output runs/demo-example-report
python scripts/export_results.py --run runs/demo-example --output artifacts/campaigns/demo-example-results.zip
python scripts/package_colab.py --output dist/llm2-source-example.zip
```

Os ZIPs de análise não contêm pesos/AdamW/datasets binários. Guardar a execução completa
se for necessário retomar. O pacote de fonte contém código, configs, testes, documentação
e notebooks; não deve incorporar caches, outputs locais ou credenciais.

Loss LM é média de cross-entropy em nats por alvo, distinta da loss total com auxiliares.
Tempo de treino soma etapas sincronizadas e exclui avaliação/log/checkpoint; primeiras
etapas podem incluir compilação. VRAM mede allocator PyTorch, não toda a placa. CPU usa
`null` para tempo GPU/VRAM. Throughput depende do protocolo. Geração atual é sem KV cache.

FLOPs são estimativas de multiplicações densas: FMA=2, backward≈2×forward diferenciável;
normas, RoPE, softmax, elementwise, loss, otimizador e compilação ficam excluídos. Usar o
estimador da arquitetura, não aplicar a fórmula Transformer a qualquer backbone.

## Hardware e extensões

A campanha 0015 validou seu protocolo T4 FP32/Inductor. Isso não valida A100/BF16 ou FP16.
Os perfis A100 e `notebooks/colab_a100.ipynb` continuam opções que exigem validação real.
O perfil 34M tem 33.990.144 parâmetros; o 352M é planejamento com bloqueio de treino.
T4x2 são duas placas independentes, sem VRAM unificada pelo runner.

O contrato de extensão separa Backbone, AttentionKernel, Objective, otimização, Trainer
e avaliação. Factories inicializam depois da semente. O objetivo controla forward e
declara alvos. `accumulation_weighting=tokens` pondera LM por alvos válidos; auxiliares
mantêm sua semântica própria. O modo legado `microbatch` continua explícito.

Arquitetura selecionada pela factory; FLRM só no checkout experimental. Novo objetivo
ou kernel exige causalidade, finitude, memorização e retomada, não apenas shapes corretos.
