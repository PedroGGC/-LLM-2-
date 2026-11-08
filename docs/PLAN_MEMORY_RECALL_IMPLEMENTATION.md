# Memória consultável — plano de implementação autorizado

08/11/2026. Implementar a proposta de primeira vantagem verificável. Fonte nova,
nenhuma migração de checkpoints históricos. Próxima campanha começa em0030.

Sete braços: Transformer2blocos, recorrente2células, hierarquia compartilhada,
banco completo e bancos esparsos uniforme/base2/Fibonacci. O banco completo é um
controle adicional necessário: distingue falha da memória de perda por seleção
esparsa. Ele consulta mais posições; não tem orçamento de leitura igual aos três
esparsos. Mesmos pesos iniciais nos quatro bancos de cada seed. Não é MC/GRM
integral: guardamos vetores contextualizados, não matrizes de estado.

Tarefa sintética byte-level de recuperação associativa:32escritas e16consultas
por sequência, chaves únicas e associações aleatórias por exemplo, posições
aleatórias. Registros de4bytes; respostas de1byte são rótulos externos, nunca
inseridas no input das consultas. Supervisionar apenas consultas; contar queries
separadamente dos bytes lidos. Distratores e formatos são idênticos para todos.
Não é linguagem natural, atualização de associações ou reprodução oficial de MQAR.

TreinoT1024/B8, FP32/Inductor, dropout0,05,3000passos fixos, seeds271/272.
LR0,0004 é receita inicial comum, não ajuste ótimo. Validação de desenvolvimento
em750/1500/2250/3000; checkpoints1500/3000. Não parar por acurácia. Gerador/custos
fixados por hash. Avaliação final cega à seleção em1024/2048/4096/8192,128exemplos
por comprimento, batch1. Avaliação longa e perfis isolados em processo novo após
o par de treinos, protegidos pelo deadline. Sondas de capacidade e interferência
por quantidade de associações usam as mesmas regras e seeds reservadas.

Padrões esparsos:16consultas, banco de no máximo2048resumos (mesmo teto no treino
e avaliação). Frações de gaps uniforme/base2/Fibonacci são reescaladas para o
histórico disponível. Offset_i=1+i+floor(frac_i*(n-k)), k=min(16,n), garantindo
mesmo número e alcance sem duplicações. Isso é espaçamento inspirado em Fibonacci,
não offsets Fibonacci literais. Prefixos com menos de16grupos consultam todos.

Até2h desde a célula, duas T4, reserva300s. Pilotos de todos os braços antes de
admitir pares inteiros com margem35%; horizonte/scheduler imutáveis. Matriz
incompleta é parcial, não confirmação. Reaproveitar somente jobs nativos completos
de protocolo/fonte/ambiente idênticos, verificando arquivos reais e metadados.
Sem escanear ZIPs antigos. Prioridade: controles simples/banco completo; depois
comparação esparsa e controle de hierarquia. Perfis pós-treino isolados: rotação
física entre seeds. Piloto concorrente estima contenção. Não realizar benchmarks
concorrentes nem alegar aceleração de geração/KV.

Relatório: acurácia exata%, perda de consulta, buckets de distância e exemplos,
curvas, bytes de pesos/estados, RAM/VRAM e latência/energia de forward completo
semelhante a prefill. Contar custo de leitura/formação de banco. Energia opcional
NVML, null quando indisponível; excluir host e identificar escopo. Sem FLOPs
fictícios ou benchmark de geração sem KV na referência. Metas95%/margem1p.p./2×
são alvos de triagem, não previsões. Dois seeds não confirmam superioridade geral.

Um par final de ZIPs; fonte uma vez, metadados/JSON/CSV/relatório HTML legível.
Sem after-NN. Remover apenas intermediários redundantes após verificar final,
limpar caches próprios de lanes inativas, preservar checkpoints completos em falha.
2GiB de reserva livre e cache global amostrado<=2GiB. Dados são procedurais, sem
download de WikiText ou criação de milhares de arquivos de dataset.

Antes da entrega: testes de causalidade, streaming/gradientes/fronteiras/isolamento,
contagens/formula/tie, índices e orçamento, influência da memória, finitude,
memorização, retomada exata com dropout/acumulação, falhas/espaço/deadline/exportação,
histórico incompatível e bootstrap/célula fixados. Integração CPU real. CUDA/
Inductor só poderão ser aprovados no hardware Kaggle; registrar limite real.
