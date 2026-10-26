# LLM2-Lab — infraestrutura e pesquisa V2

Infraestrutura Python/PyTorch para experimentos controlados com modelos de linguagem.
Inclui um Transformer decoder-only inicializado do zero, pipeline offline de dados,
treino, validação, geração, TensorBoard e checkpoints completos. O plano e os critérios
de aceitação estão em [PLAN.md](PLAN.md); resultados executados estão em [VALIDATION.md](VALIDATION.md).

O baseline da etapa 1 foi preservado. A V2 adiciona recursão causal compartilhada, objetivo
preditivo inspirado em JEPA, refinamento contínuo, MTP opcional e currículo procedural externo.
FLRM fica isolado em `experiment/fractal-lrm`, no worktree vizinho `LLM 2-fractal-lrm`.
São hipóteses experimentais próprias, não reproduções integrais de TRM/HRM/Coconut/DeepSeek.
Não há evidência de superioridade ou redução de ordens de grandeza; NCM permanece futuro.

## Começar na V2

Instalação direta no Python local: `python -m pip install -e ".[dev,research]"`.
O PyTorch existente não precisa ser trocado. Os YAMLs e comandos V1 continuam válidos para
novas execuções; resultados e datasets antigos não foram sobrescritos.

```powershell
python -m pytest -q
python -m llm2 inspect --config configs/experiments/e3_lrm.yaml --meta
python -m llm2 experiment --config configs/experiments/combined.yaml
python -m llm2 train --config configs/experiments/e3_lrm.yaml --data data/processed-byte --output-dir runs/my-lrm --until-step 2
python -m llm2 train --config configs/experiments/e3_lrm.yaml --data data/processed-byte --output-dir runs/my-lrm --resume runs/my-lrm/checkpoints/step-00000002
python -m llm2 evaluate --checkpoint runs/my-lrm/checkpoints/step-00000004
python -m llm2 benchmark --checkpoint runs/my-lrm/checkpoints/step-00000004 --output runs/my-lrm/benchmark.json
python -m llm2 reasoning --checkpoint runs/my-lrm/checkpoints/step-00000004 --examples 1 --max-new-tokens 8 --output runs/my-lrm/reasoning.json
python -m llm2 report --runs runs/my-lrm --output runs/my-lrm-report
python -m llm2 plan-budget --config configs/planning/a100_budget.yaml
python -m llm2 profile --config configs/experiments/e3_lrm.yaml --output runs/my-profile
```

`experiment` e `profile` só executam com `--execute`. `train` é uma solicitação explícita
de treino de **uma** configuração. Nada inicia campanhas E/F/C automaticamente. As configs
de experimentos são smoke tests CPU de quatro passos; os perfis A100 são configurações para
campanhas futuras. `--seed` permite repetir um experimento com outras sementes. Use diretórios
novos; o exemplo de retomada requer que o checkpoint 2 já exista e não haja checkpoint posterior
no mesmo destino (ou escolha outro `--output-dir` para derivar uma execução).

Documentos da V2:

- [Auditoria e plano](docs/RESEARCH_V2_PLAN.md), [roadmap e limites](docs/RESEARCH_ROADMAP.md).
- [LRM/JEPA/MTP](docs/ARCHITECTURE_LRM.md) e [FLRM](docs/ARCHITECTURE_FLRM.md).
- [Currículo e isolamento dos dados](docs/CURRICULUM.md).
- [Matriz de experimentos](docs/EXPERIMENT_MATRIX.md), [A100/créditos](docs/A100_EXECUTION_PLAN.md).
- [Modelo de relatório](docs/RESULTS_TEMPLATE.md), [referências](docs/REFERENCES.md).
- [Resultados CPU executados e reprodução](docs/RESULTS_CPU_SMOKE.md).

Configurações antigas não ganham uma seção research implicitamente. Pesos V1 safetensors
continuam legíveis para avaliação; retomada exata V1 exige os bytes de código do respectivo
`source.zip` e o ambiente registrado. A V2 não desativa o bloqueio por hash nem migra o estado
silenciosamente. O commit inicial preserva a etapa 1; extrair o snapshot original em outra pasta
e usar seu `src` via `PYTHONPATH` preserva também os finais de linha exigidos pelo hash.
Não sobrescreva o checkout atual para recuperar código histórico.

## Instalação local, sem ambiente virtual

Requer Python >=3.10. A instalação realizada neste computador usa Python 3.12.2 diretamente.
Para testes CPU em uma instalação nova:

```powershell
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e ".[dev]"
python -m pytest -q
```

As versões diretas verificadas estão em `requirements/cpu-tested.txt`. Para reproduzi-las:

```powershell
python -m pip install -r requirements/cpu-tested.txt
python -m pip install -e ".[dev]"
```

Use o mesmo `python` para instalar e executar. `python -c "import sys; print(sys.executable)"`
identifica o interpretador. A CLI é acessível por `python -m llm2`, mesmo quando o diretório
de scripts não está no PATH. O arquivo CPU de versões não deve ser usado no Colab.

## Fluxo CPU reproduzível

Na raiz do projeto:

```powershell
python scripts/make_demo_corpus.py
python -m llm2 prepare --input data/raw/demo.txt --output data/processed-byte --tokenizer byte --seed 42
python -m llm2 inspect --config configs/cpu_smoke.yaml
python -m llm2 train --config configs/cpu_smoke.yaml --until-step 10
python -m llm2 train --config configs/cpu_smoke.yaml --resume runs/cpu-smoke/checkpoints/step-00000010
python -m llm2 evaluate --checkpoint runs/cpu-smoke/checkpoints/step-00000020
python -m llm2 generate --checkpoint runs/cpu-smoke/checkpoints/step-00000020 --prompt "O laboratório" --seed 42 --max-new-tokens 40
python -m tensorboard.main --logdir runs/cpu-smoke/tensorboard
```

O corpus é sintético e serve para testar a engenharia. Vinte atualizações não constituem
um modelo de linguagem útil. Comandos de preparação e treino recusam sobrescrever dados ou
execuções. Use outro caminho para um novo experimento. Todos os caminhos dentro do YAML são
relativos ao próprio YAML; caminhos passados na CLI são relativos ao diretório atual.

## Modelo de referência e CUDA

`configs/transformer_34m_a100.yaml` define:

| Item | Valor |
| --- | ---: |
| Vocabulário alocado | 8.192 |
| Camadas | 9 |
| Dimensão do modelo | 512 |
| Cabeças / dimensão por cabeça | 8 / 64 |
| Dimensão interna SwiGLU | 1.472 |
| Contexto | 512 |
| Parâmetros únicos calculados | **33.990.144** |
| Microbatch / acumulação | 8 / 4 |
| Tokens alvos por optimizer-step | 16.384 |

Pre-norm RMSNorm; RoPE em queries e keys, pares de dimensões intercaladas; SwiGLU;
projeções sem bias; embeddings de entrada/saída compartilhados; residual dropout configurável.
Pesos normais com desvio 0,02 e projeções residuais escaladas por `1/sqrt(2*n_layers)`.
O otimizador AdamW aplica weight decay apenas a tensores com duas ou mais dimensões,
contando o parâmetro compartilhado uma vez. O scheduler usa warmup linear e decaimento cosseno.

Para um corpus real, forneça arquivos UTF-8 próprios; linhas vazias separam documentos:

```bash
python -m llm2 prepare --input "data/raw/*.txt" --output data/processed-bpe --tokenizer bpe --vocab-size 8192 --seed 42
python -m llm2 inspect --config configs/transformer_34m_a100.yaml
python -m llm2 train --config configs/transformer_34m_a100.yaml
```

Na A100 use uma instalação PyTorch **CUDA**, não a wheel CPU. O treino mantém parâmetros
e AdamW em FP32 e usa autocast BF16; RMSNorm, rotação RoPE e cross-entropy usam FP32
nas operações sensíveis. BF16 não usa GradScaler. Solicitar CUDA/BF16 sem suporte causa erro.

SDPA escolhe kernels conforme hardware, versão, dtype e forma. Avaliação passa dropout zero,
como exige a [API oficial do PyTorch](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention).
`train.deterministic: true` força SDPA math e algoritmos determinísticos, desabilita TF32 e
configura `CUBLAS_WORKSPACE_CONFIG`. A configuração A100 permite seleção automática de kernels
com `deterministic: false`; isso não equivale a uma promessa de igualdade bit a bit.
O PyTorch [não garante reprodução entre versões e plataformas](https://docs.pytorch.org/docs/stable/notes/randomness.html).

`train.compile: true` ativa `torch.compile`; o padrão é false. O teste local exercita
captura Dynamo com backend eager. Inductor/CUDA e o consumo de 40 GB precisam ser verificados
na A100. Não foram medidos ganhos nem escolhido um batch por benchmark de hardware.

## Google Colab A100

1. Execute `python scripts/package_colab.py --output dist/llm2-lab-v2-colab.zip`.
   Arquivos existentes são preservados; escolha outro nome para um novo pacote.
2. Abra `notebooks/colab_a100.ipynb` no Colab e selecione runtime GPU A100.
3. Execute as células: elas verificam A100/BF16, recebem o ZIP, preservam o Torch CUDA
   instalado, instalam o pacote, executam os testes e demonstram treino e retomada.
4. O notebook oferece persistência no Drive. Ajuste corpus, `RUN_NAME`, `MAX_STEPS`,
   `COMPILE` e `DETERMINISTIC` antes de iniciar seu experimento.

O notebook não foi executado em uma A100 nesta entrega; sua estrutura e código Python foram
validados localmente. A [disponibilidade dos recursos do Colab varia](https://research.google.com/colaboratory/faq.html).
O filesystem local do runtime é temporário; preserve dados preparados, código/config e
diretório de execução. Checksums verificam cópias de checkpoints; atomicidade de rename em
disco local não garante durabilidade contra falhas físicas ou desconexões do Drive/FUSE.

## Dados e tokenização

`prepare` lê UTF-8 em memória por documento, remove espaços nas bordas, deduplica documentos
idênticos e aplica um split determinístico por SHA-256 ordenado + shuffle com RNG privado.
Fração de validação padrão: 10%, com ao menos um documento em cada split. O BPE byte-level
é treinado **somente no treino**, sem normalização ou dependência de tokenizador remoto.
O modo byte tem 256 bytes + EOS. Literais com a grafia do token especial são tratados como texto.

Artefatos: `manifest.json`, `tokenizer.json`, `train.bin`, `val.bin`. O manifesto registra
hashes dos arquivos e documentos, fontes, semente, versões do preparo, vocabulário efetivo
e política de concatenação. O carregamento verifica hashes, tamanhos, IDs e isolamento dos
documentos. Um corpus pequeno pode gerar BPE com menos de 8.192 símbolos; isso é registrado,
mantendo a matriz de 8.192 entradas e a contagem do modelo. Geração restringe a amostragem
ao vocabulário efetivo.

Documentos são concatenados com EOS; a atenção pode atravessar suas fronteiras. O amostrador
escolhe janelas contíguas uniformes **com reposição** usando generator próprio, sem workers
ou prefetch. Não há conceito implícito de época. Cada janela de `S+1` tokens produz `S` inputs
e `S` targets deslocados; não há padding. Cada split precisa de ao menos `S+1` tokens.
Deduplicação é exata, não uma detecção de similaridade ou de contaminação semântica.

Validação usa janelas sequenciais fixas e média ponderada pelos tokens, sem alterar o RNG
ou o amostrador de treino. O treino avalia até `eval_batches` batches; a CLI `evaluate`
avalia todos os blocos completos por padrão e reporta quantos tokens/batches avaliou.
A cauda incompleta é descartada. Tokens de treino incluem repetições e EOS; validação
não incrementa o contador de tokens treinados.

## Checkpoints e retomada

Cada diretório `checkpoints/step-NNNNNNNN/` contém:

- `model.safetensors`: pesos, usando `save_model`/`load_model` para preservar o
  [compartilhamento de tensores](https://huggingface.co/docs/safetensors/torch_shared_tensors).
- `training_state.pt`: objetivo (parâmetros/buffers), AdamW, scheduler, RNGs Python/NumPy/Torch/
  CUDA, estado do sampler e contadores de tokens, FLOPs estimados, tempo e memória.
- `metadata.json`: configuração resolvida, classes, hashes, versões, código e método de FLOPs.

O checkpoint ocorre depois de uma atualização completa, scheduler e limpeza de gradientes.
Uma atualização com erro não pode ser salva; retome o último checkpoint completo. O diretório
é publicado por rename após escrever todos os arquivos, sem sobrescrita. `latest.json` aponta
para o último checkpoint. Não há restauração do meio de um microbatch.

O arquivo de estado PyTorch usa pickle: **retome apenas checkpoints confiáveis**.
Avaliação e geração pela CLI carregam metadata e safetensors, sem desserializar o estado pickle.

Retomada exige o mesmo conteúdo de dados, código do pacote, configuração semântica e ambiente
registrado. É permitido alterar `max_steps` (até `schedule_steps`), diretório de saída,
localização dos mesmos dados e intervalos de logging/validação/checkpoint. O horizonte e
hiperparâmetros do scheduler são imutáveis. `--max-steps` não estica o schedule silenciosamente.
Para derivar de um checkpoint antigo quando há checkpoints posteriores, use `--output-dir`
novo. Logs de tentativas posteriores incompletas são preservados em arquivos separados ao
retomar; o JSONL ativo e o TensorBoard são reabertos no ponto apropriado.

Equivalência estrita foi testada em CPU FP32/eager/SDPA math, com dropout e acumulação ativos:
a próxima atualização produz loss, pesos, otimizador, scheduler, sampler e RNGs idênticos.
Tempos e throughput são medições e naturalmente não coincidem entre execuções. Migrar de
CPU para CUDA, trocar versões ou trocar backend constitui outro ambiente; a retomada estrita
recusa essas mudanças. Para outra arquitetura ou transferência apenas de pesos, use uma nova
execução pela API Python e documente o protocolo.

## Métricas e proveniência

Cada execução mantém `run.json`, YAML resolvido, `source.zip`, `environment.freeze.txt`,
`metrics.jsonl`, `summary.json`, eventos em `tensorboard/` e checkpoints. Retomadas têm
metadados próprios. O snapshot contém `src/`, `pyproject.toml` e configs; os hashes de código,
commit/diff/status do Git e versões identificam a implementação mesmo sem um commit inicial.
O freeze registra o ambiente instalado, não é um lock portável nem remove conflitos prévios
de pacotes globais. Dados e artefatos grandes não entram no Git por padrão.

| Métrica | Significado |
| --- | --- |
| `model_parameters_calculated`, `total_parameters_calculated` | Enumeração única de parâmetros do modelo e do conjunto modelo+objetivo |
| `training_loss`, `validation_loss` | Cross-entropy média por token alvo, em nats |
| `tokens_processed`, `sequences_processed` | Contagens acumuladas de treino; incluem repetições |
| `training_flops_estimate` | Estimativa analítica acumulada, não profiler |
| `step_tokens_per_second`, `tokens_per_second` | Tokens por tempo de treino da etapa / acumulado |
| `train_wall_seconds` | Tempo sincronizado das etapas: dados, transferência, forward/backward, AdamW e scheduler; exclui validação/log/checkpoint |
| `gpu_time_seconds` | Soma dos intervalos por eventos CUDA nas etapas de treino; inclui intervalos ociosos do stream, não é soma de kernels |
| `peak_vram_allocated_bytes`, `peak_vram_reserved_bytes` | Máximos do allocator PyTorch no processo, incluindo treino e validação; não VRAM total via NVML |
| `lr`, `grad_norm` | LR usada na atualização e norma antes do clipping |

CPU registra tempo GPU/VRAM como `null`. Tempos incluem warmup/compile da etapa; não há descarte
de warmup implícito. Use um processo dedicado para comparações de memória. A função de
validação reporta seus tokens separadamente; FLOPs/tempo de validação e geração não entram
nos contadores de treinamento.

Para `D=d_model`, `H=hidden_dim`, `L=n_layers`, `V=vocab_size`, `S=seq_len`:

```text
parâmetros únicos = V*D + L*(4*D² + 3*D*H + 2*D) + D
FLOPs forward por sequência ≈ L*(8*S*D² + 6*S*D*H + 4*S²*D) + 2*S*D*V
FLOPs treino por sequência ≈ 3 * FLOPs forward
```

Uma multiplicação-acumulação vale 2 FLOPs. O fator 3 estima forward+backward. A projeção
de vocabulário custa operações mesmo com pesos compartilhados. A atenção é contada como
matriz quadrada densa; kernels causais podem executar menos trabalho. Excluem-se normalização,
RoPE, ativações, softmax, lookup, dropout, loss, otimizador e compilação. Backbones/objetivos
desconhecidos recebem `null` até fornecerem estimador próprio. Não há alegação de ganhos
de desempenho ou de qualidade; essas fórmulas não substituem benchmarks.

## Organização e futuras extensões

```text
src/llm2/
  interfaces.py    Backbone, ModelOutput, AttentionKernel, Objective, ObjectiveOutput
  model.py         Transformer, RMSNorm, RoPE, SwiGLU e SDPA causal
  objectives.py    loss autoregressiva independente
  optim.py         AdamW e scheduler
  data.py          tokenização, manifesto, memmap e sampler
  trainer.py       treinamento em um dispositivo, acumulação e orquestração
  evaluation.py    validação sem alteração de RNG/modos
  generation.py    amostragem causal simples
  checkpoint.py    persistência e contrato de retomada
  metrics.py       contagens, estimativa de FLOPs, JSONL e TensorBoard
  runtime.py       sementes, precisão, ambiente e snapshots
  config.py        dataclasses/YAML estrito
  cli.py           comandos offline
```

`Trainer(config, model_factory=..., objective_factory=..., flops_estimator=...)` permite
substituir os componentes. Factories são chamadas depois de fixar a semente; com instâncias
pré-construídas, a inicialização aleatória é responsabilidade do chamador. O objetivo controla
o forward e retorna loss escalar e contagem de tokens; seus parâmetros também são otimizados
e salvos. A acumulação faz média dos microbatches de tamanho fixo. Um objetivo futuro com
normalização diferente deve definir explicitamente como agrega perdas e supervisiona tokens.

`ModelOutput` permite logits opcionais, hidden states e estado opaco. O baseline rejeita
estado recorrente e o Trainer não propaga estado entre janelas. Uma extensão futura deve
definir esse protocolo no objetivo/backbone antes de usá-lo. `attention_factory` no Transformer
injeta um `AttentionKernel` por camada; o kernel é responsável por sua causalidade. Coloque
extensões em `src/llm2/` para que sejam incluídas nos hashes e snapshots. A CLI de inferência
reconhece o baseline e os backbones registrados da V2; modelos externos usam os contratos Python.

Geração é para inspeção, com greedy (`temperature=0`) ou amostragem/top-k e generator local.
Sem KV cache: a janela é recalculada e posições RoPE reiniciam ao truncar o contexto.
Há testes CPU em CI e testes condicionais CUDA/BF16. DDP, FP16/scaler, streaming de corpus
maior que RAM, NCM completo e treinamento de 300M–500M ficam fora desta entrega.
