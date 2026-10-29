# Próximas perguntas de pesquisa

Atualizado após a campanha 0015. Este documento lista possibilidades; não dispara
treinos nem autoriza incorporar todas as tecnologias. Estado factual em
[PROJECT_STATUS.md](PROJECT_STATUS.md); etapa atual em [PLAN.md](../PLAN.md).

## Prioridade imediata

Organizar o projeto e decidir a próxima hipótese com o usuário. Preservar Transformer
como controle e N1 como referência de investigação FLRM. Recursos opcionais existentes
não devem ser removidos por uma campanha inconclusiva.

## Candidatos à próxima ablação — escolher um

- Normalizar somente a projeção de contexto antes da soma com o estado normalizado.
  Hipótese: balanceamento facilita a aprendizagem. Risco: amplificar contexto pouco
  informativo. Melhorar RMS sem melhorar qualidade/custo não basta.
- Alterar isoladamente o caminho inicial/residual do embedding. Exige controles de FFN,
  inicialização e parâmetros antes de atribuir toda a diferença K=1 a esse mecanismo.
- Condicionamento explícito por índice de iteração. O estado já informa parcialmente
  a fase; não presumir incapacidade do núcleo de distinguir iterações.

Definir antes controles, seeds, horizonte/scheduler, avaliação e critério de continuidade.
Uma matriz possível compara Transformer, N1 e uma intervenção; controles adicionais LRM
são necessários para atribuir o mecanismo. Confirmar com seeds novas e conjunto reservado.

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
