# FLRM — branch experimental isolada

Código: `experiment/fractal-lrm`, worktree `../LLM 2-fractal-lrm`. A branch principal conhece
o identificador `fractal`, mas falha explicitamente se o módulo isolado não estiver presente.
Não há placeholder que imite um modelo funcional na branch comum.

Hipótese: operações compartilhadas e condicionadas entre contextos de diferentes alcances
podem favorecer reutilização composicional. Não atribuimos automaticamente palavras/conceitos
às escalas, não alegamos dimensão fractal matemática, nem reproduzimos FractalNet.

## Componentes implementados no experimento

- `FractalScaleEncoder`: escala local (janela 1), intermediária (janela 4) e global causal
  (0 = prefixo completo). Nos modos `reference`/`batched`, todas retêm T posições.
  Os modos `compressed*` usam o compressor de blocos descrito abaixo.
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
individualmente nessa branch. Nesse registro inicial ainda não havia campanha GPU;
os resultados posteriores e suas limitações estão em VALIDATION.md.

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

Futuro: NCM discreto, memória entre
escalas, DeltaNet/Mamba, DEQ, parada aprendida, profundidade dinâmica e kernels Triton.
Nenhum desses componentes foi implementado nesta entrega.

## Compressão real e otimizações de execução — etapa 1, 27/10/2026

`compressed`, introduzido em `a668af73`, usa grades T, ceil(T/4), ceil(T/16),
pooling por média e dois priors aprendidos. Cada leitura grossa na grade fina é
atrasada um bloco: o bloco ainda incompleto não pode influenciar seus tokens.
O estado interno conserva a mesma dinâmica compress(upsample(...)) do controle.
A escala fina mantém atenção densa; a complexidade total continua quadrática.
`research.scales` deve ser [1,4,0] nesses modos por compatibilidade com o schema
existente; os fatores efetivos são fixos 1/4/16, e 0 não indica prefixo global aqui.

Há três modos comprimidos com o mesmo state_dict e **2.168.076** parâmetros de
backbone no perfil D=384/H=1024/vocab=257/core_layers=1:

| scale_execution / modelo da campanha | Núcleos por iteração | Comprimentos das projeções da interação |
|---|---|---|
| compressed / flrm_compressed | T, ceil(T/4), ceil(T/16) separados | T, T, T |
| compressed_native / flrm_compressed_native | mesmos núcleos separados | T, ceil(T/4), ceil(T/16) |
| compressed_grouped / flrm_compressed_grouped | T separado, duas escalas curtas agrupadas em 2B × ceil(T/4) | T, ceil(T/4), ceil(T/16) |

Se U_j são as leituras causais na grade fina, a saída grossa original é
P_i(U_i + g_i W Σ_j a_ij U_j). Como média P_i e projeção W são lineares, a
interação nativa calcula P_i(U_i) + g_i W Σ_j a_ij P_i(U_j). Isso inclui as caudas
parciais e gradientes dos priors. Não muda a janela causal nem adiciona pesos.
Arredondamento FP32 pode diferir; não se promete trajetória de treino idêntica.

Agrupar somente as escalas curtas reduz três chamadas de núcleo para duas, mas
faz a escala de fator 16 processar padding à direita até o comprimento da de
fator 4. Consultas válidas não leem posições futuras; saídas de padding são
descartadas. O consumo de RNG/dropout muda. É uma ablação experimental até medir
tempo/VRAM na T4, não um modo presumidamente mais rápido.

O estimador v3 soma por comprimento n do núcleo:
`core_layers × (8 n D² + 6 n D H + 4 n² D) + 2 n D²`.
A interação acrescenta `2 D² × (3T)` no controle, ou
`2 D² × (T + ceil(T/4) + ceil(T/16))` nos modos nativos. O modo agrupado contabiliza
os dois núcleos curtos no comprimento com padding. Logits acrescentam `2 T D V`.
FMA conta duas operações; atenção usa a convenção quadrada densa. Backward é
aproximado por duas vezes o forward diferenciável; teacher sem gradiente e
recomputação são contabilizados à parte. Pooling, mistura elementar, FiLM,
normalizações, RoPE, softmax, loss e optimizer são excluídos. A versão anterior
subcontava as projeções do controle comprimido; não comparar totais v2 e v3
como evidência de ganho computacional medido.

Testes em `test_compressed_optimization.py`: saídas/gradientes CPU FP32/math SDPA
com os mesmos pesos e tolerâncias, K=1/2, limites 4/16 e comprimentos não múltiplos;
causalidade em todos os cortes de 33 tokens; contagem única, memorização, retomada
exata dentro de cada modo com dropout/EMA/checkpointing/acumulação ativos.
CUDA FP32 tem equivalência e save/continue próprios para rodar no Kaggle.

CKA entre escalas comprimidas fica null até haver alinhamento temporal adequado.
Ranks brutos com diferentes números de amostras não são comparáveis diretamente.
Geração continua recalculando o prefixo, sem KV cache. Checkpoints antigos devem
ser avaliados com sua fonte original; nenhuma checagem de compatibilidade foi relaxada.

## Execução agrupada P01–P06

`research.scale_execution: reference` preserva a execução sequencial. A opção `batched`, somente com `share_scales: true`, empilha estados/contextos em S×B, aplica uma chamada do núcleo compartilhado e vetoriza FiLM/interação. Os parâmetros e o número analítico de operações são iguais; ordem de redução e RNG/dropout podem mudar. A soma de mistura permanece FP32 sob autocast. O pooling continua uma soma prefixada por encoder; não foi trocado por projeções que mudariam o modelo.

`return_logits=False` fornece hidden-only para teacher/diagnóstico e o estimador omite a cabeça correspondente. Não é KV cache nem memória entre segmentos.

Causalidade, saídas/gradientes CPU FP32 com SDPA math, memorização e retomada exata com dropout/acumulação são verificados por `tests/test_fractal_grouped.py`. CUDA FP32 tem testes separados. A campanha Kaggle 8f86ba9cc48b mediu os dois modos e a revisão 9d8a6efd9be9 avaliou checkpoints intermediários; resultados, hashes e limites estão em VALIDATION.md. O ganho depende da tarefa, e o agrupamento aumentou VRAM no perfil medido. A célula e a campanha estão em KAGGLE_T4_CAMPAIGN.md.

## O que determina os 2,17 milhões de parâmetros

Esse número é o tamanho estrutural do backbone, não uma quantidade de conhecimento,
uma geração antiga de pesos ou um limite que o treino precise superar. O treino muda
os valores dos pesos; não aumenta sua quantidade nesta implementação. A contagem
inclui embeddings compartilhados uma vez e não inclui o teacher/auxiliares de treino.

O perfil d_model=384, hidden_dim=1024, vocabulário byte de 257 símbolos, três escalas
compartilhadas e core_layers=1 tem exatamente 2.167.308 parâmetros. Distribuição:
98.688 embedding, 1.918.464 núcleo, 2.304 FiLM, 147.468 interação, 384 normalização.
O comentário histórico de ~10,7M na configuração fractal estava incorreto.

`model.n_layers` controla o Transformer convencional. Para FLRM/LRM, a profundidade
do núcleo usa `research.core_layers`. Repetições usam `iterations + latent_steps`;
com pesos compartilhados, repetir o núcleo aumenta trabalho e pode aumentar memória
de ativações, mas não cria novos parâmetros.

| Camadas no núcleo | Iterações | Parâmetros backbone calculados |
|---:|---:|---:|
| 1 | 1 | 2.167.308 |
| 1 | 2 | 2.167.308 |
| 2 | 1 | 3.937.548 |
| 6 | 1 | 11.018.508 |

Contagens conferidas por enumeração em meta e fórmula; as variantes maiores não
foram treinadas/medidas nessa verificação. Mais camadas/iterações não garantem loss
menor (melhor qualidade); precisam de comparação. Menos parâmetros significa modelo
menor, não automaticamente mais inteligente ou mais eficiente para a mesma qualidade.

A campanha atual fixa core_layers=1 e iterations=1 para FLRM em seu construtor de
configuração, mesmo que outro preset-base contenha outros valores. Uma próxima matriz
de profundidade precisa expor e registrar essa escolha explicitamente no executor.

O FLRM atual é um núcleo de Transformer compartilhado entre três representações de
contexto (token, média causal de janela 4 e média do prefixo), condicionado por FiLM
e combinado por interação aprendida. Todas as escalas conservam T posições; não há
compressão real da sequência nem evidência de propriedade fractal matemática. Com
uma iteração, só a escala 0 é lida para a saída e não há segunda rodada de feedback
entre escalas. Isso limita a hipótese testada, sem invalidar causalidade ou o código.

Próxima investigação recomendada: controles de tamanho/objetivo, múltiplas seeds,
comparação de uma/duas iterações e ablação das escalas antes de redesenho maior. Preservar
V1 e seus checkpoints como referência. Resultados em corpus byte pequeno, frases
geradas e indicadores numéricos de não colapso não comprovam gramática geral,
raciocínio ou a tese inteira do projeto. Fibonacci/DEQ/compressão exigem experimentos
próprios; sua motivação matemática não substitui evidência de utilidade no modelo.
