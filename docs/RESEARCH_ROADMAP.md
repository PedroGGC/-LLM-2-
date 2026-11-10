# Próximas perguntas de pesquisa

## Roteiro vigente — arquitetura própria inspirada em fractais

10/11/2026: [0031concluída](results/CAMPAIGN_0031.md), oito treinos/duas seeds.
Três candidatos100%na recuperação/contrafactuais até8K e64pares. Referência
Transformer não qualificou. Sharing economiza sem perda observada frente à
pirâmide independente; memória única já resolve tudo. Priorizar agora linguagem
e referência bem qualificada, mantendo pirâmide compartilhada como controle.
Ganhos observados de tempo/energia em8K têm escopo de forward, eVRAM8K é maior
na implementação nova. Nenhuma vitória geral ou confirmação iso-qualidade.
Não adicionar nova tecnologia à tarefa saturada antes desse marco.

Entrega posterior09/11: [nova família de transições/pirâmide](ASSOCIATIVE_BINDING_RUN.md),
não dependente do mLSTM. Memória associativa linear com endereço/conteúdo separados,
convolução causal e operadores compartilhados/independentes entre ritmos. Testes
locais de novos mapas/contrafactuais passam; CPU único86.785pesos/1000passos resolveu
as seis sondas de desenvolvimento. Comparação T4 com Transformer, linguagem e
eficiência continuam pendentes. Campanha nova: quatro braços/duas seeds, teto2h;
pesos rápidos são agora uma frente ativa, sem alegar implementar TTT/GRM completos.

09/11/2026: [campanha0030 parcial](results/CAMPAIGN_0030.md),4/14treinos/seed271.
Não houve recuperação funcional no banco completo ou Fibonacci; a referência
Transformer também está longe de resolver a tarefa. Recomenda-se rever formação
local de associações, escrita/leitura e gates da adaptação mLSTM, qualificando
generalização em tarefa curta antes de repetir a matriz. Banco funcional precede
ranking de espaçamentos. Reduzir recompilações/pilotos antes de nova sessão. Nada
disso era implementado na auditoria; a entrega posterior acima implementa a nova
família e operação. Preservar protocolos e hipótese fractal.

08/11/2026: [bancada de memória/recuperação](MEMORY_RECALL_RUN.md) autorizada e
implementada; sete braços, seeds271/272, dois ritmos compartilhados, banco completo
e acessos uniforme/base2/Fibonacci. Acerto/custo em tarefa procedural antes de
linguagem e escala; não demonstramos ainda retenção insuficiente. Compressão,
esforço variável e alternativas do plano permanecem. Teto2h, matriz divisível.

Histórico04/11/2026: [campanha0029 concluída](results/CAMPAIGN_0029.md), cinco controles,
seed251/6000passos. Leitura imediata melhora0,008843NLL, mas hierarquia continua
inferior ao simples/Transformer; fino treinado também perde. Amplo ajuda frente
ao fino; dependência de posição persiste. Próxima proposta: comparar compressão
média, último estado e compressão aprendida, com leitura imediata fixa, antes de
novo sharing ou memórias/adaptação. Não implementado; exige protocolo próprio.
Não repetir automaticamente as matrizes históricas nem promover uma seed a vitória.

O [plano aprovado](PLAN_RECURRENT_FRACTAL.md) substitui as prioridades históricas
abaixo. Transformer é referência externa; o FLRM antigo permanece reproduzível,
sem novas repetições previstas. Ciclo5 suspenso como próxima operação.

A [0027](results/CAMPAIGN_0027.md) concluiu somente seed241 da etapaB. Hierarquia e
sharing perderam qualidade; Transformer venceu também tempo isolado e VRAM de treino.
Antes de repetir as seeds restantes, recomenda-se diagnosticar compressão, mistura
e perda da segunda transformação fina. Não confirma fracasso geral do fractal nem
justifica adicionar memória/adaptação sem identificar a limitação. Matriz original
preservada; etapas condicionais abaixo continuam no catálogo.

O [diagnóstico0028](results/CAMPAIGN_0028.md) foi concluído: reduzir/remover o ramo
amplo piorou as duas hierarquias. Priorizar controle treinado de profundidade fina
e do momento causal de leitura do grupo fechado; sinal de defasagem na quarta
posição é descritivo. Compressão aprendida, sharing, memória e esforço adaptativo
mantêm suas condições, sem introdução automática. Diagnóstico não é ablação treinada.

| Frente | Prioridade e pergunta |
|---|---|
| Núcleo recorrente | Etapa A concluída na0025: mLSTM selecionado nas duas T4; RWKV aprovado, Gated Delta fora do orçamento de compilação |
| Fractal explícito | Mesma hierarquia com núcleo compartilhado versus operações independentes; separar hierarquia de auto-similaridade |
| Hierarquia e compressão | Caminhos rápido/lento; compressão causal fixa, depois aprendida/adaptativa se justificada; BLT/Cantor como referências |
| Profundidade variável | Treinar K1/2/4; RINS como organização, ACT/PonderNet somente após benefício de esforço adicional |
| Memória consultável | MC/GRM é candidato com orçamento de bytes e custo de consulta; não obrigatório |
| Memória adaptativa | TTT/fast weights é outra possibilidade com escrita, esquecimento e reset |
| Geração/engenharia | KV cache e estados recorrentes corretos; FractalLLM posterior; AMP/compilação medidos |

Base sólida é correção/reprodução/custo conhecido; não exige vitória prévia do FLRM.
As [fichas](RECURRENT_CANDIDATES.md) distinguem referências implementadas no laboratório
das arquiteturas completas dos artigos. [Operação vigente](RECURRENT_RUN.md).

Condicionais: DEQ/equilíbrio/solver; controle espectral/ortogonalidade/nGPT por
instabilidade; Kronecker/Monarch/MERA/hypernetworks por gargalo medido. Preservados
desligados: FiLM, JEPA, NCM, RMS de contexto, Ciclo-V, MTP e currículo. Baixa
prioridade: Fibonacci/φ/lattices, KAN e variantes, Liquid/CfC/LFM2 e L-systems.
FractalNet/FANN/fractal nodes são referências. Difusão/flow/árvore/NCA/inferência
ativa são linhas próprias futuras. Quantização/destilação complementam depois.

## Registro histórico até a auditoria 0023 — não é a fila vigente

Atualizado em 31/10/2026, após a auditoria 0023.
Este documento lista possibilidades; não dispara
treinos nem autoriza incorporar todas as tecnologias. Estado factual em
[PROJECT_STATUS.md](PROJECT_STATUS.md); etapa atual em [PLAN.md](../PLAN.md).

## Roteiro consolidado após o diagnóstico

O [ciclo 4 / 0023](results/CAMPAIGN_0023.md) não demonstrou ganho das escalas.
Somente fino cumpriu o critério exploratório de simplificação; o Transformer de
dois blocos liderou qualidade e custo. Probes de K não treinado pioraram. Antes
de profundidade variável/RINS, o [ciclo 5](CYCLE5_RUN.md) treina K1/K2/K4 separados
com FLOPs estimados iguais e referência Transformer. São controles, sem novo
mecanismo. Não repetir a matriz de escalas ou escalar A100 agora. Se esta forma
de recorrência continuar sem benefício, interromper investimento nela neste regime
e escolher uma frente independente justificada pelo objetivo. As prioridades
abaixo continuam condicionais, sem promessa de incorporação automática.

| Ordem | Tecnologia | Teste e condição |
|---|---|---|
| 1 | **Profundidade variável / RINS** | Distribuição explícita de profundidades no treino e K1/2/4/8; comparar localização da recorrência. O probe pós-treino do ciclo 4 mede sensibilidade, sem treinar essa política. |
| 2 | **Compressão aprendida/adaptativa** | Comparar compressão causal por conteúdo com fatores fixos, contabilizando fronteiras/roteamento; recuperação e perda de informação. BLT/Cantor como referências. |
| 3 | **Gated DeltaNet** | Primeiro baseline sequencial independente. Medir retenção/recuperação e custo em contexto longo antes de hibridizar com FLRM. RWKV-7/mLSTM continuam alternativas. |
| 4 | **Memory Caching / GRM** | Memória externa por segmentos com orçamento explícito em bytes e política de retenção; controles de escrita, leitura, isolamento entre documentos e esquecimento. |
| 5 | **TTT / fast weights** | Estado adaptativo e custo de atualização; exigir ganho de retenção que justifique treino/inferência. Experimento separado de GRM. |
| Engenharia | **KV cache → FractalLLM** | Primeiro cache correto no Transformer com equivalência à geração sem cache; depois dependências por escala/iteração e especulação. Não atribuir geração paralela à recursão atual. |
| Posterior | **Quantização e destilação** | Complementos após selecionar arquitetura/precisão de referência. Medir qualidade, RAM/VRAM, tempo e energia no hardware real. |

Referências primárias: [RINS](https://arxiv.org/abs/2502.07503),
[BLT](https://arxiv.org/abs/2412.09871), [Cantor](https://doi.org/10.1109/MC.2025.3624444),
[Gated DeltaNet](https://research.nvidia.com/publication/2025-04_gated-delta-networks-improving-mamba2-delta-rule),
[TTT](https://arxiv.org/abs/2407.04620) e [FractalLLM](https://aclanthology.org/2025.findings-emnlp.1286/).
Resultados dessas famílias não garantem ganhos na nossa escala.

**Memory Caching é um mecanismo concreto, não uma garantia de memória gratuita.**
O [artigo original](https://arxiv.org/html/2602.24281v1) separa RM, GRM, Soup e SSC.
Uma adaptação precisaria definir representação dos segmentos, consulta, gate,
atualização e persistência. O FLRM atual recorre em profundidade sobre uma janela;
não mantém automaticamente estado sequencial entre documentos. Cache que cresce
com a sequência ocupa memória crescente. Consultar todos os segmentos pode
reintroduzir custo quadrático no comprimento para tamanho de segmento fixo;
medir tokens retidos, bytes e trabalho de leitura, incluindo estados auxiliares.

O objetivo de aproximar qualidade de modelos maiores com menos parâmetros é uma
hipótese de longo prazo. Contar todos os pesos, estados, cache e computação;
"10M equivale a 100M/1B" não é critério demonstrado por este laboratório. Não
promover mecanismos apenas por K, contexto, nome fractal ou inspiração física.

## Evidência anterior e interpretação

A [campanha 0017](results/CAMPAIGN_0017.md) concluiu comunicação × FiLM: há sinal
favorável à comunicação nas três seeds, ainda com intervalo exploratório cruzando
zero. FiLM acrescenta um benefício pequeno quando a comunicação está ativa; o
Transformer mantém a melhor média e menor custo observado.

Preservar o completo como referência e levar comunicação ligada/FiLM desligado como
candidato simplificado para confirmação. Manter contexto sem RMS, K=2 e mecanismos
auxiliares desligados. A decisão é provisória no regime testado, não uma declaração
de equivalência universal ou inutilidade de FiLM.

O [ciclo 3](results/CAMPAIGN_0019.md) já tem dois blocos completos. A vantagem FLRM
frente ao Transformer pequeno com FLOPs estimados iguais é pequena e inconclusiva;
o LRM e o Transformer de dois blocos tiveram NLL menor nas duas seeds.
As checagens primárias chegaram à seed 104 na [0022](results/CAMPAIGN_0022.md),
com duas vitórias por arquitetura em quatro pares. O [lote de até uma hora](CYCLE3_SHORT_RUN.md)
continua como reprodução histórica. O diagnóstico aprovado agora separa as escalas,
com protocolo próprio; não altera a confirmação antiga nem sua regra de decisão.

O [executor do ciclo 2](CYCLE2_RUN.md) foi usado com sucesso na 0017 e permanece como
reprodução. Antes de implementar tecnologias adicionais, confirmar o sinal em dados
novos. A contribuição individual das escalas por 4/16 continua uma possível ablação
posterior, condicionada à utilidade do pacote e ao orçamento.

Depois da base sólida, manter profundidade variável (RINS como uma referência de
organização da recorrência) e compressão adaptativa (BLT e ideias hierárquicas como
referências) em experimentos separados. KV cache é prioridade de engenharia para
geração; FractalLLM exige comparação posterior com cache. TTT e um baseline como
RWKV/mLSTM/Gated DeltaNet permanecem candidatos de contexto/memória; escolher um
antes de combiná-lo com FLRM. Nenhuma dessas ideias integra a configuração atual.

## Antes de aumentar a complexidade

Comparar qualidade com alvos iguais e com tempo/FLOPs estimados comparáveis. Avaliar
outra distribuição e tarefas verificáveis se o objetivo for raciocínio. Distinguir treino
compilado de geração eager. Melhor checkpoint no último passo não demonstra convergência.

| Possibilidade | Evidência necessária antes de priorizar |
|---|---|
| NCM | Ganho reproduzível que justifique custo; ablação com cache desligado em ambos, pois o NCM atual é incompatível com cache |
| JEPA/MTP/currículo | Objetivo e tarefa onde o benefício auxiliar possa ser medido |
| Ciclo-V | Compensar a perda de qualidade observada num protocolo controlado |
| ACT | Utilidade da profundidade variável e economia real da política de parada |
| DEQ/phantom gradients | Estabilidade/equilíbrio, custo e correção do gradiente |
| Fibonacci/φ-ALiBi | Gargalo de contexto, cobertura causal e benchmark de backend |
| Kronecker/Monarch | Perfil justificando fatoração e comparação de qualidade/custo |
| BF16/A100 ou FP16/T4 | Implementação e validação no hardware real; FP16 requer scaler |
| Ortogonalidade/controle espectral | Instabilidade medida em profundidades maiores, com comparação de custo/qualidade |
| Atenção local/global e FlexAttention | Atenção como gargalo medido e backend correto/eficiente no hardware real |
| KAN/ChebyKAN/Wav-KAN, Liquid/CfC/LFM | Baixa prioridade para texto; separar os conceitos e exigir evidência na tarefa |
| L-systems/hypernetworks | Adiados; contabilizar geração, armazenamento temporário e aplicação de pesos |
| Difusão/flow matching/geração em árvore | Linha própria de objetivo/decodificação; não emerge automaticamente do FLRM |
| NCA/reação-difusão/Regra 110 | Linha separada de baixa prioridade para linguagem |
| Energia livre/inferência ativa | Exige tarefa e modelo de agente definidos; não substitui a loss atual |
| Feynman/holografia/MERA | Analogias sem garantias; MERA exigiria estrutura tensorial concreta e avaliação |
| FANN/FractalNet/fractal nodes | Referências de desenho; resultados em outras tarefas não estabelecem prioridade em linguagem |

Fibonacci, frequências áureas e lattices são hipóteses distintas de baixa prioridade;
nenhuma oferece contexto infinito ou recuperação garantida. Fusão manual depende
de gargalo no perfil e equivalência; Inductor permanece a via principal, sem novo
caminho TorchScript. JEPA fica preservado e desligado, sem descarte universal.
NCM atual consulta protótipos aprendidos e não grava documentos. ACT e DEQ seguem
condições separadas: parada adaptativa e equilíbrio/solver, respectivamente.

Sem promoção automática ao atingir K ou T específicos; sem promessa de capacidade de
um modelo maior com menos pesos. Planos antigos estão no [arquivo](archive/README.md).

## Triagem de tecnologias externas

A [revisão das ideias do Argon](REVIEW_TECHNOLOGIES_ARGON.md) analisa os 17 itens do
catálogo recebido em 29/10/2026. Preserva o ciclo 2 e prioriza, depois da base sólida,
profundidade variável e compressão adaptativa em experimentos separados. TTT e um
baseline temporal como RWKV-7 ou mLSTM entram como candidatos posteriores para
memória/contexto longo; não como dependências ou configurações ativas. As promessas
de capacidade equivalente, contexto infinito e cancelamento semântico por fases não
são premissas do projeto. Nenhum mecanismo adicional foi implementado nesta revisão.
