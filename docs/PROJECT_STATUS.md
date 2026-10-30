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

A [campanha 0017](results/CAMPAIGN_0017.md) concluiu a ablação comunicação × FiLM no
commit `60219eafbceea09066dd05206adaa9f4ce865f96`: 18 treinos, seeds 81/82/83 e
2.000 passos, com 72 avaliações completas. Fonte/configurações e dados conferidos;
o dataset é idêntico à 0016. Tempo global registrado: 1h32min43s antes do relatório
e exportação final, dentro de 7.200 s.

Transformer: NLL média 1,394574; FLRM completo: 1,398282; FLRM com comunicação e
sem FiLM: 1,398978; LRM: 1,402445. O Transformer venceu o completo em duas seeds e
tem menor custo medido. O completo melhorou frente ao LRM nas três seeds, mas
essa comparação também envolve mais parâmetros/operações e cache diferente.

Ativar comunicação melhorou 0,003331 NLL com FiLM e 0,002498 sem FiLM, nas três
seeds de cada contraste. Os intervalos exploratórios ainda cruzam zero. FiLM com
comunicação ganhou 0,000695, abaixo da margem operacional de 0,002; sem comunicação
não há ganho consistente. A conclusão é sinal específico favorável, ainda sem
confirmação com custo igualado, corpus novo e cinco seeds.

Decisão documental: preservar o pacote completo como referência e selecionar
provisoriamente comunicação ligada/FiLM desligado como candidato mais simples à
confirmação. Contexto continua sem RMS conforme a 0016. Os controles sem comunicação
continuam apenas de atribuição: preservam projeção/gate e forward dos ramos grossos,
mas esses ramos não contribuem para a loss fina. Não são controles de custo igualado.

Logs do Kaggle registram 250 testes CPU e 21 CUDA FP32 aprovados, incluindo os quatro
braços compilados do ciclo 2. Treinos/pilotos usaram as duas T4 em série, com ordem
rotacionada e alternância; avaliações usam a primeira GPU. Isso valida o protocolo
executado no ambiente registrado, não BF16/A100 ou qualquer configuração futura.

Todas as execuções tiveram o melhor checkpoint avaliado no passo final; convergência
não foi demonstrada. Os pesos não estão no ZIP de análise: hashes de checkpoints e
inicialização foram conferidos nos registros, sem rehash de pesos ausentes.

## Evidência mais recente: campanha 0019

[Relatório](results/CAMPAIGN_0019.md): 12 treinos e 48 avaliações completos em
WikiText-103, seeds 101/102. CPU 280 e CUDA FP32 25 testes aprovados no Kaggle.
Sessão terminou `partial_budget` aos 2h40min45s; o próximo bloco inteiro não cabia
na previsão conservadora. O ZIP não registra falha de rede como causa da parada.

Com FLOPs estimados semelhantes: Transformer de dois blocos 1,196534 NLL; LRM
1,278126; FLRM 1,308835; Transformer pequeno 1,311478. Parâmetros não igualados.
O sinal FLRM frente ao Transformer pequeno é pequeno/inconclusivo; LRM e controle
mais profundo venceram o FLRM nas duas seeds. Ainda não há vantagem fractal confirmada.

## Atualização: snapshot 0021

[FLRM 103 concluído](results/CAMPAIGN_0021_PARTIAL.md): 8.000 passos, quatro avaliações,
NLL 1,320334273. Pesos/estado conferidos por hash; histórico com 13 treinos. O controle
103 estava pendente no snapshot. O erro de número ocupado pertence a uma nova
tentativa e preserva a pasta existente; não comprova falha do treino original.

## Próxima etapa

[Operar em lotes de até uma hora](CYCLE3_SHORT_RUN.md), até dois treinos novos,
começando pelo par primário da seed 103. Reaproveitar os 12 completos e preservar
quatro avaliações completas, curvas e diagnósticos. Código científico original
congelado em checkout próprio; operador com proveniência separada. Testes remotos
focados nas famílias presentes, com opção de suíte completa, sem declarar cobertura
integral onde só houve seleção. Snapshots são salvos após cada conclusão.

A100, profundidade variável, compressão adaptativa e demais tecnologias permanecem
condicionais. Uma sessão curta não demonstra aceleração de kernel nem substitui a
confirmação de cinco seeds. Reavaliar o custo da investigação após a próxima seed.
