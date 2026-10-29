# Próximas perguntas de pesquisa

Atualizado após a campanha 0016. Este documento lista possibilidades; não dispara
treinos nem autoriza incorporar todas as tecnologias. Estado factual em
[PROJECT_STATUS.md](PROJECT_STATUS.md); etapa atual em [PLAN.md](../PLAN.md).

## Prioridade imediata após o ciclo 1

A [campanha 0016](results/CAMPAIGN_0016.md) rejeitou, neste regime, a normalização RMS
do contexto: piora nas três seeds de LRM e FLRM. Preservar a opção como ablação
negativa e manter `context_normalization: none` no próximo ciclo.

O ciclo 2 deve comparar comunicação entre escalas ligada/desligada × FiLM
ligado/desligado, mantendo referências Transformer/LRM. Sem comunicação, preservar
transformações próprias dos ramos; não desligar todos os ramos como atalho.

O [novo executor do ciclo 2](CYCLE2_RUN.md) implementa piloto prévio, limite global
de duas horas e execução isolada/alternada; sua validação T4 ocorrerá no Kaggle. A confirmação posterior permanece condicionada à seleção
congelada, WikiText com documentos/splits reservados, cinco seeds novas e comparação
com custo igualado. Não ampliar os schedulers dos checkpoints antigos.

Normalização de contexto é um teste negativo específico. Residual, condicionamento
de iteração e novas tecnologias continuam hipóteses; não entram automaticamente na
matriz do ciclo 2. A infraestrutura completa dessas etapas ainda está pendente.

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
