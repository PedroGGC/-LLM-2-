# FLRM — arquitetura experimental atual

Código somente em `experiment/fractal-lrm`, checkout `LLM 2-fractal-lrm`. Na branch
comum a factory rejeita a seleção fractal sem o módulo experimental. Estado científico:
[PROJECT_STATUS.md](PROJECT_STATUS.md). Resultados: [campanha 0015](results/CAMPAIGN_0015.md).

## Transformer reutilizado em várias escalas

`FractalLatentModel` usa embedding/cabeça compartilhados e `FractalRecursiveBlock`.
`SharedFractalCore` herda o núcleo LRM: estado normalizado + projeção de contexto,
`DecoderBlock`s causais, atualização residual com gate vetorial. A profundidade do
núcleo é `research.core_layers`, não `model.n_layers`.

FiLM condiciona o contexto por escala. Uma matriz de mistura, projeção e gates combinam
as escalas causalmente. O estado começa em zero a cada forward; somente a escala fina
alimenta a cabeça de linguagem. Não há memória entre documentos nem KV cache.

O termo fractal designa esta hipótese multiescala; não demonstra dimensão fractal,
especialização semântica, raciocínio ou superioridade sobre Transformer.

## Modos de execução

| Modo | Resoluções do núcleo | Interação |
|---|---|---|
| `reference` | Todas com T posições; token, janela causal 4, prefixo causal | Sequencial |
| `batched` | Mesmas T posições nas três escalas | Núcleo e mistura agrupados |
| `compressed` | T, ceil(T/4), ceil(T/16) | Projeções da interação na grade fina |
| `compressed_native` | Mesmas resoluções comprimidas, chamadas separadas | Pooling antes das projeções grossas |
| `compressed_grouped` | Fina separada; escalas curtas agrupadas com padding | Interação nativa |

Nos modos comprimidos, `scales=[1,4,0]` é exigido pelo schema, mas os fatores reais são
1/4/16. O zero não significa atenção global nesse caminho. Para T=256, comprimentos
válidos são **256/64/16**, e o modo agrupado preenche 16→64. Os priors e atraso de um
bloco nas leituras grossas preservam a causalidade, inclusive em fronteiras parciais.
A escala fina usa atenção densa; compressão não torna a complexidade total subquadrática.

Agrupar reduz chamadas do núcleo, mas processa padding. Reordenar projeção e média
preserva a álgebra apropriada; arredondamento e dropout podem alterar trajetórias.
Equivalência numérica controlada não implica igualdade bit a bit entre modos de treino.

## Controle atual e recursos opcionais

- **G0**: `flrm_compressed_grouped_no_jepa_k2`, agrupado, sem cache.
- **N1**: `flrm_compressed_native_cached_no_jepa_k2`, nativo, reutiliza `W_context(c)`
  dentro do forward. O grafo é preservado; gradientes somam entre reutilizações.
- `paired_common_init` pareia parâmetros comuns LRM/FLRM por seed e isola inicialização
  dos exclusivos. O Transformer convencional não participa desse pareamento de tensores.
- `fused_film` habilita operações vetorizadas/`addcmul`; não garante um único kernel ou speedup.
- `v_cycle` evita a passagem fina após a primeira; `inv_vcycle` evita a primeira passagem
  fina. Mudam o modelo e tiveram perdas de qualidade; desligados na 0015.
- `ncm_slots` cria protótipos aprendidos consultados nas escalas grossas. Não há escrita
  contextual persistente. Cache de projeção e NCM são incompatíveis no validador atual.

Na 0015, N1/G0 usaram K=2, LM pura, `fused_film=false`, NCM=0 e ambos os ciclos desligados.
O cache de contexto não é cache de keys/values da geração. A implementação de treino
compilado não compila automaticamente o caminho de benchmark de geração.

## Parâmetros, estimativas e testes

D=384, FFN=1024, vocab=257, core_layers=1: LRM tem 2.017.536 parâmetros comuns;
FLRM comprimido adiciona 150.540, totalizando **2.168.076** únicos. A variante alinhada
sem os dois priors de compressão tem 2.167.308. Repetir K com pesos compartilhados
não acrescenta parâmetros, mas custa operações e ativações.

O estimador conta formas realmente executadas, inclusive padding no núcleo agrupado,
projeções, cache e NCM quando ativos. FMA=2, backward≈2×forward diferenciável; normas,
RoPE, softmax, elementwise, loss, otimizador e compilação são exclusões. FLOPs são
estimativas, não medições exatas. Não misturar revisões do estimador silenciosamente.

Testes exercitam causalidade, equivalência controlada de saídas/gradientes, contagem,
memorização e retomada. CKA entre escalas comprimidas permanece nulo sem alinhamento
adequado. Diagnósticos em quatro janelas não provam generalização; gates são vetores
cuja média estável não significa congelamento. Ver [validação](../VALIDATION.md).
