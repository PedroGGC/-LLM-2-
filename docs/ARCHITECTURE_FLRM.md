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
parâmetros, mas não o número de chamadas. Cada iteração custa três chamadas do core e as
projeções de interação. FiLM/pooling/mistura elementwise são exclusões documentadas da estimativa.
`fractal_parameter_formula` confere parâmetros únicos, incluindo interação/condicionamento.

Os sete testes iniciais passaram em CPU, incluindo memorização, pooling de referência,
causalidade e retomada com teacher EMA/progressão/checkpointing. A suíte integrada do worktree
e seus resultados finais estão em VALIDATION.md. F0–F4 e C5/C6 são configs executáveis
individualmente nessa branch. Nenhuma campanha ou ganho foi demonstrado.

Futuro: escalas realmente comprimidas com máscaras corretas, NCM discreto, memória entre
escalas, DeltaNet/Mamba, DEQ, parada aprendida, profundidade dinâmica e kernels Triton.
Nenhum desses componentes foi implementado nesta entrega.
