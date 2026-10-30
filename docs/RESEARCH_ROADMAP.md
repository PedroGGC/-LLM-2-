# Próximas perguntas de pesquisa

Atualizado em 30/10/2026, após a 0022 e aprovação do diagnóstico do ciclo 4.
Este documento lista possibilidades; não dispara
treinos nem autoriza incorporar todas as tecnologias. Estado factual em
[PROJECT_STATUS.md](PROJECT_STATUS.md); etapa atual em [PLAN.md](../PLAN.md).

## Roteiro consolidado após o diagnóstico

O trabalho atual é o [ciclo 4](CYCLE4_RUN.md): contribuição das escalas 4/16,
controle somente fino, LRM cached e Transformer de dois blocos. O histórico 0022
não confirmou vantagem fractal. As prioridades abaixo são experimentos posteriores
separados, sujeitos aos resultados e a orçamentos próprios; não entram nesta rodada.

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
