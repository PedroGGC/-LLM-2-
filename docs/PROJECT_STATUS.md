# Estado científico e técnico — 28/10/2026

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

A [campanha 0015](results/CAMPAIGN_0015.md) é a referência científica mais recente:
três seeds, um corpus/split byte-level, contexto 256 e treino FP32/Inductor na T4.
Transformer teve menor NLL nas três seeds. N1 melhorou custo e loss frente a G0,
mas teve pior throughput de geração eager. Nenhum desses resultados prova capacidade
geral, convergência final ou eficiência em outra escala/hardware.

Recorrência ajudou após os mesmos alvos, usando mais computação. O pacote multiescala
teve sinal favorável dentro da família recorrente, sem isolar cada componente nem
igualar completamente parâmetros/FLOPs. Três seeds não eliminam incerteza e seleção
por uso repetido da mesma validação.

O gate não congelou. O desbalanceamento RMS entre estado/contexto existe, mas sua causa
e impacto na loss não foram isolados. As hipóteses de normalização ou caminho residual
devem ser testadas separadamente, caso escolhidas para a próxima etapa.

Nesta etapa apenas a organização está autorizada. A proposta de nova ablação continua
no [roadmap](RESEARCH_ROADMAP.md), aguardando definição de hipótese e protocolo.
