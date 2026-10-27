# FLRM — branch experimental isolada

Código: `experiment/fractal-lrm`, worktree `../LLM 2-fractal-lrm`. A branch principal conhece
o identificador `fractal`, mas falha explicitamente se o módulo isolado não estiver presente.
Não há placeholder que imite um modelo funcional na branch comum.

Hipótese: operações compartilhadas e condicionadas entre contextos de diferentes alcances
podem favorecer reutilização composicional. Não atribuimos automaticamente palavras/conceitos
às escalas, não alegamos dimensão fractal matemática, nem reproduzimos FractalNet.

## Componentes implementados no experimento

- `FractalScaleEncoder`: escala local (janela 1), intermediária (janela 4) e global causal
  (0 = prefixo completo). Todas as grades retêm T posições; não há compressão de sequência.
- `SharedFractalCore`: reutiliza a operação comum de refinamento causal; uma instância por
  escala em F1 ou uma instância compartilhada em F2/F3/F4.
- `ScaleConditioning`: FiLM com gamma limitado por tanh e beta, inicializados em zero.
- `CrossScaleInteraction`: mistura aprendida entre escalas **na mesma posição**, projeção
  compartilhada e gates. Nenhum pooling usa a sequência inteira para atualizar o passado.
- `FractalRecursiveBlock`: core condicionado seguido da interação; repete-se com pesos
  compartilhados na profundidade. Estado por escala zerado a cada forward.
- `FractalLatentModel`: embeddings/cabeça compartilhados, normalização, histórico opcional,
  truncamento/checkpointing e objetivos/factories comuns.

Agregação usa somas prefixadas em FP32 e diferenças para janelas passadas. Logits do prefixo
devem ser invariantes sob mudanças futuras, incluindo posições na fronteira de uma janela.
A geração continua autoregressiva e recalcula a janela, sem KV cache nem memória entre documentos.

F1–F4 mantêm o número de escalas e as operações por escala explícitos. Compartilhar pesos reduz
parâmetros, mas não o número de chamadas. Cada iteração custa o equivalente a três processamentos do core e as
projeções de interação. FiLM/pooling/mistura elementwise são exclusões documentadas da estimativa.
`fractal_parameter_formula` confere parâmetros únicos, incluindo interação/condicionamento.

Os sete testes iniciais passaram em CPU, incluindo memorização, pooling de referência,
causalidade e retomada com teacher EMA/progressão/checkpointing. A suíte integrada do worktree
e seus resultados finais estão em VALIDATION.md. F0–F4 e C5/C6 são configs executáveis
individualmente nessa branch. Nenhuma campanha ou ganho foi demonstrado.

## Executar no worktree isolado

Uma instalação editável aponta para um único checkout. Para usar o código experimental
sem trocar a instalação local da branch comum, selecione `src` no processo:

```powershell
Set-Location '../LLM 2-fractal-lrm'
$previousPythonPath = $env:PYTHONPATH
try {
    $env:PYTHONPATH = (Resolve-Path ./src).Path
    python -m pytest -q
    python -m llm2 inspect --config configs/experiments/f3_conditioned.yaml --meta
    python -m llm2 train --config configs/experiments/c6_flrm_adaptive.yaml --data '../LLM 2/data/processed-byte' --output-dir runs/my-flrm
} finally {
    $env:PYTHONPATH = $previousPythonPath
}
```

Esses comandos de treino executam somente a configuração indicada; a suíte usa dados
temporários e não depende do corpus preparado na branch comum.

Futuro: escalas realmente comprimidas com máscaras corretas, NCM discreto, memória entre
escalas, DeltaNet/Mamba, DEQ, parada aprendida, profundidade dinâmica e kernels Triton.
Nenhum desses componentes foi implementado nesta entrega.

## Execução agrupada P01–P06

`research.scale_execution: reference` preserva a execução sequencial. A opção `batched`, somente com `share_scales: true`, empilha estados/contextos em S×B, aplica uma chamada do núcleo compartilhado e vetoriza FiLM/interação. Os parâmetros e o número analítico de operações são iguais; ordem de redução e RNG/dropout podem mudar. A soma de mistura permanece FP32 sob autocast. O pooling continua uma soma prefixada por encoder; não foi trocado por projeções que mudariam o modelo.

`return_logits=False` fornece hidden-only para teacher/diagnóstico e o estimador omite a cabeça correspondente. Não é KV cache nem memória entre segmentos.

Causalidade, saídas/gradientes CPU FP32 com SDPA math, memorização e retomada exata com dropout/acumulação são verificados por `tests/test_fractal_grouped.py`. CUDA FP32 tem testes separados; T4 será medida no Kaggle. Nenhum speedup ou queda de VRAM do modo agrupado foi presumido. A célula e a campanha estão em KAGGLE_T4_CAMPAIGN.md.
