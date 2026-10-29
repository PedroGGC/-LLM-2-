# Estado científico e técnico — 29/10/2026

## Já existe Transformer dentro do modelo

`TransformerLM` empilha `DecoderBlock`s distintos. `LatentRecursiveModel` reutiliza
um núcleo com esses mesmos blocos. `FractalLatentModel`, na branch experimental,
também reutiliza o núcleo entre representações em diferentes resoluções.

```text
Transformer: embedding → blocos causais distintos → norma → cabeça de linguagem
LRM:         embedding → contexto fixo + estado refinado K vezes → norma → cabeça
FLRM:        contexto em escalas → núcleo recorrente + interação causal → escala fina → cabeça
```

Os blocos contêm SDPA causal, RoPE, RMSNorm e SwiGLU. A saída textual continua
autoregressiva. Estado contínuo não equivale a raciocínio demonstrado. O experimento
investiga organização, compartilhamento e orçamento de computação, não a substituição
completa da atenção por uma tecnologia já superior.

No perfil da campanha 0015, `core_layers=1`, D=384, FFN=1024 e K=1/2. O campo
`model.n_layers=6` não cria seis blocos no LRM/FLRM; governa apenas o Transformer.
O Transformer pequeno usa um bloco e FFN=1280 para aproximar o total de parâmetros.

## Disponibilidade e uso

| Recurso | Implementação | Uso na 0015 |
|---|---|---|
| Transformer e LRM causal, BPTT, controle untied | Base comum e experimental | Transformer e LRM K=1/2 |
| FLRM, FiLM, compressão causal por 4/16 | Experimental | Ativos no FLRM |
| Execução agrupada/nativa e cache da projeção de contexto | Experimental | G0 agrupado; N1 nativo com cache |
| Inicialização comum pareada e telemetria treinada | Experimental | Ativas; pareamento de tensores cobre LRM/FLRM |
| JEPA auxiliar com teacher/EMA | Ambas | Desligado |
| MTP, supervisão intermediária, refinamentos extras e currículo | Ambas | Desligados |
| NCM de 16 protótipos, fusão manual, Ciclo-V/invertido | Experimental | Desligados |
| ACT/DEQ, Fibonacci/φ-ALiBi, Kronecker/Monarch | Hipóteses, sem implementação desses algoritmos | Ausentes |
| Memória persistente entre documentos, KV cache, DDP/MoE | Não implementados | Ausentes |

Recursos implementados não são recomendações automáticas. NCM atual consulta parâmetros
globais de protótipos; não grava o conteúdo de documentos. O validador experimental
proíbe `cache_context_projection` junto com `ncm_slots>0`.

## Evidência e decisão atual

A [campanha 0016](results/CAMPAIGN_0016.md) é a referência mais recente. Executou o
ciclo 1 no commit `d173bdea0d9b7b497fcf108dc1bb7ddc99148c62`, com cinco variantes,
seeds 71/72/73 e 2.000 passos. Conteúdo e splits dos dados coincidem com a 0015;
o hash do manifesto mudou somente pela proveniência do código de preparo.

Normalizar contexto após a projeção aumentou a NLL em 0,040738 no LRM e 0,042958
no FLRM, com regressão nas três seeds. As normas ficaram equilibradas sem benefício
na qualidade. Decisão: manter `context_normalization: none` para o ciclo 2.

Transformer venceu as três seeds. N1 melhorou NLL frente ao LRM, mas usa mais
parâmetros/operações e cache diferente. Não há demonstração de superioridade fractal
nem comparação confirmatória com custo igualado, corpus independente e cinco seeds.

A infraestrutura experimental contém `context_normalization` e `cross_scale_exchange`,
preparo documental com splits explícitos e importador WikiText. Isso não significa
que os ciclos posteriores já foram executados ou toda a infraestrutura concluída.
O runner recebido ainda precisa de piloto prévio e teto global de sessão; treinos
0016 foram concorrentes, com perfis isolados posteriores. A100 permanece condicional.

Próxima etapa: separar comunicação multiescala e FiLM conforme o
[roadmap](RESEARCH_ROADMAP.md). Todas as execuções tiveram melhor checkpoint no passo
final; convergência e generalidade continuam não demonstradas.

## Próximo teste preparado

A [célula curta do ciclo 2](CYCLE2_RUN.md) usa commit fixado de `experiment/fractal-lrm`,
com piloto prévio, execução serial e teto global de duas horas, incluindo preparação.
O bootstrap faz checkout por Git e não exige ZIP de fontes como Input. Testes CPU e
simulação do fluxo passaram; a validação CUDA/Inductor será executada pelo bootstrap
na T4. O runner usado na 0016 permanece histórico e não deve ser usado para esse ciclo.
