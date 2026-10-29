# Próximas perguntas de pesquisa

Atualizado após a campanha 0017. Este documento lista possibilidades; não dispara
treinos nem autoriza incorporar todas as tecnologias. Estado factual em
[PROJECT_STATUS.md](PROJECT_STATUS.md); etapa atual em [PLAN.md](../PLAN.md).

## Prioridade após o ciclo 2

A [campanha 0017](results/CAMPAIGN_0017.md) concluiu comunicação × FiLM: há sinal
favorável à comunicação nas três seeds, ainda com intervalo exploratório cruzando
zero. FiLM acrescenta um benefício pequeno quando a comunicação está ativa; o
Transformer mantém a melhor média e menor custo observado.

Preservar o completo como referência e levar comunicação ligada/FiLM desligado como
candidato simplificado para confirmação. Manter contexto sem RMS, K=2 e mecanismos
auxiliares desligados. A decisão é provisória no regime testado, não uma declaração
de equivalência universal ou inutilidade de FiLM.

Executar o [ciclo 3](CYCLE3_RUN.md): WikiText com documentos completos e splits
reservados, configuração congelada, cinco seeds novas e ensaios separados de alvos
iguais e FLOPs estimados igualados. Executor e hashes do dataset estão preparados;
o piloto e a ciência na T4 ainda serão executados. Cada sessão terá no máximo quatro
horas; repartir o protocolo se necessário, sem mudar horizontes para ocupar o relógio.

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
| NCM | Ganho reproduzível que justifique custo e compatibilidade explícita com cache |
| JEPA/MTP/currículo | Objetivo e tarefa onde o benefício auxiliar possa ser medido |
| Ciclo-V | Compensar a perda de qualidade observada num protocolo controlado |
| ACT | Utilidade da profundidade variável e economia real da política de parada |
| DEQ/phantom gradients | Estabilidade/equilíbrio, custo e correção do gradiente |
| Fibonacci/φ-ALiBi | Gargalo de contexto, cobertura causal e benchmark de backend |
| Kronecker/Monarch | Perfil justificando fatoração e comparação de qualidade/custo |
| BF16/A100 ou FP16/T4 | Implementação e validação no hardware real; FP16 requer scaler |

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
