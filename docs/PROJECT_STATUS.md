# Estado científico e técnico — 08/11/2026

Implementação autorizada: [bancada de recuperação/memória](MEMORY_RECALL_RUN.md),
sete braços/duas seeds,3000passos fixos, duas T4 e teto2h. Banco completo como
controle de capacidade, consultas uniforme/base2/Fibonacci com orçamento esparso
igual. Pesos compartilhados entre níveis e estados separados. Gerador procedural
sem corpus; nenhum resultado de linguagem novo. Triagem poderá ocupar mais de
uma sessão; piloto admite pares inteiros. Controle de latência é forward completo,
sem alegação de geração/KV. Execução científica T4 ainda pendente.

A [campanha0029](results/CAMPAIGN_0029.md) concluiu cinco braços/seed251/6000passos
nas duas T4, em1h10min33s antes da exportação. NLL Transformer1,271962, recorrente
1,315631, hierarquia imediata1,427357, seguinte1,436200, fino1,474430. Leitura
imediata melhora0,008843; amplo ajuda frente ao fino, sem superar duas células por
byte. A dependência de posição permanece. Não há vantagem fractal confirmada nem
novo controle de sharing. CPU11passed eCUDA2passed no preflight da nova fonte.
Após essa auditoria propusemos compressão média/último estado/aprendida; em08/11
a prioridade autorizada passou à bancada funcional de memória. Defaults
históricos e qualificação0025 preservados; teste reservado intacto. Pesos reais
da0029não acompanham o ZIP fornecido, apenas seus metadados/hashes declarados.

## Direção vigente: arquitetura própria inspirada em fractais

O [plano aprovado](PLAN_RECURRENT_FRACTAL.md) substitui a prioridade do ciclo5.
Novos backbones: `state_recurrent` e `state_hierarchy`, regras Gated Delta/RWKV-7/
mLSTM num shell comum; não são modelos completos dos artigos. Hierarquia com
operações independentes versus núcleo compartilhado testa contribuição fractal.
Transformer permanece referência externa. A [0025](results/CAMPAIGN_0025.md)
qualificou mLSTM e RWKV nas duas T4 com guarda externa; Gated Delta excedeu600s.
mLSTM selecionado:93,43/89,99ms por passo,746,92MiB de VRAM na qualificação. Etapa A concluída.
A [operação B](STATE_B_RUN.md) mantém a ciência
qualificada e implementa cache por variante/GPU; primeiro lote da seed241.
MC/GRM/TTT/ACT permanecem candidatos separados.

Tentativa0026: CPU e dois testes CUDA aprovados, mas cache bruto de WikiText
divergiu antes do treino. Novo preflight recupera cópias verificadas em cache
separado, antes de gastar compilação CUDA; ciência e protocolo preservados.
Ver [operação](RECURRENT_RUN.md) e [fichas/adaptações](RECURRENT_CANDIDATES.md).

A [0027](results/CAMPAIGN_0027.md) concluiu quatro treinos/seed241 e6000passos:
NLL Transformer1,273494, simples1,313647, hierarquia independente1,431568 e
compartilhada1,462067. Fontes, configurações, oito avaliações completas e oito de
subconjunto conferidas. Nenhum teste reservado; uma seed, sem IC entre seeds.
Hierarquia economiza32% por passo frente ao simples, mas perde0,117921NLL.
Sharing economiza47,72% dos pesos, perde0,030498NLL e reduz só3–4% do tempo.
Transformer vence qualidade, tempo isolado e VRAM de treino, sem medição de geração.

Próximo passo recomendado: diagnosticar o ramo amplo antes das seeds242/243.
Controle simples tem segunda célula por byte; a hierarquia a substitui por
processamento comprimido. Compressão, profundidade fina e mistura não foram
isoladas. Manter hipótese fractal; não promover esta implementação nem adicionar
GRM/TTT/ACT automaticamente. Protocolo permanece congelado, seeds restantes pendentes.
Relógio1h55min38s antes da exportação final; cache máximo amostrado1,124GiB,
guarda sem violação. GPU/Inductor desta operação exercitados nas duas T4 na0027;
pytest de preflight CUDA rodou apenas naGPU0. Tensores finais não acompanhavam
o ZIP de resultados.

Entrega de diagnóstico: [operação pós-treino](STATE_DIAGNOSTIC_RUN.md),11leituras de
quatro pesos da241, sem treino. Nova operação separada usa a fonte científica
original; inspeção de componentes e intervenções de leitura, sem mudar backbone.
O backup de checkpoints foi localizado localmente depois da auditoria: quatro
hashes reais/safetensors e contagens únicas conferidos. Sem avaliação científica
local desses pesos, pois o ambiente local difere do Kaggle.

A [0028](results/CAMPAIGN_0028.md) executou o diagnóstico nas duas T4:11condições,
zero optimizer-steps, erro máximo de reprodução2,61e-9,109,24s antes da exportação.
ZIP419.492bytes. Silenciar o amplo piora0,193690/0,152317NLL; ganho0,5 também piora.
RMS amplo/fino0,522/0,479; atenuação global não melhorou nos valores testados.
Quarta posição tem perda0,032765/0,020614acima da média das demais nas hierarquias,
sem padrão semelhante nos controles. A leitura atual só usa o novo resumo no byte
seguinte ao fechamento. Isso sugere controle de defasagem, não comprova sua causa.
Próximo ensaio recomendado: núcleo fino treinado sozinho e comparação isolada do
momento de leitura, com controles atuais. Sem alteração de código/treino nesta
auditoria, sem confirmação de vantagem fractal ou consulta ao teste reservado.

## Evidência histórica e próxima operação anterior

Evidência mais recente: [0023](results/CAMPAIGN_0023.md), 18 treinos e 96 avaliações
completas auditadas (72 principais +24 probes), pesos finais conferidos. Não houve
vantagem das escalas; somente fino é candidato de simplificação (~24% menos tempo,
perda máxima0,00138). Transformer2blocos: NLL1,285904, contra1,349391 do completo,
com menor custo observado. Aumentar K após treinoK2 piorou. Operação então preparada:
[ciclo 5](CYCLE5_RUN.md), profundidades treinadasK1/2/4 a FLOPs estimados iguais;
sem tecnologia nova. As seções históricas abaixo identificam seus regimes.

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

## Evidência anterior: campanha 0019

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

## Evidência mais recente: campanha 0022

[Relatório auditado](results/CAMPAIGN_0022.md): dois treinos novos, oito avaliações
completas, 32min05s registrados, 12 CPU e 2 CUDA FP32 aprovados no Kaggle. Histórico
com 16 treinos e quatro pares primários (101–104). NLL seed 104: FLRM 1,318235,
Transformer 1,312205. Média das diferenças pareadas +0,001845, IC95% exploratório
[−0,006698; +0,010387]; duas vitórias para cada arquitetura. Equivalência não demonstrada.

A regra pré-fixada exige FLRM melhor em quatro de cinco seeds para confirmar vantagem.
Com duas vitórias em quatro pares, uma quinta seed permitiria no máximo três.
Há somente dois blocos completos de seis braços; o resultado permanece inconclusivo.

## Próxima etapa recomendada

Pausar novos treinos deste protocolo e revisar a hipótese usando os controles já
medidos. A fila registra 14 jobs pendentes, sem obrigação de executá-los para
encerrar esta análise exploratória. Uma nova experiência exige pergunta e critérios
próprios. A auditoria não muda o protocolo original nem inicia treino.

A100, profundidade variável, compressão adaptativa e demais tecnologias permanecem
condicionais. Uma sessão curta não demonstra aceleração de kernel nem substitui a
confirmação prevista. [Lotes curtos](CYCLE3_SHORT_RUN.md) continuam disponíveis como
infraestrutura, com preservação de horizontes, avaliações e snapshots.
