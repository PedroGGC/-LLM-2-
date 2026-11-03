# Campanha0029 — profundidade fina e leitura causal, plano prévio

03/11/2026. Implementação autorizada após0028. Perguntas: custo/qualidade de uma
célula fina treinada sozinha; efeito de consultar o grupo no próprio fechamento.
Fractal permanece hipótese. Não adicionar GRM/TTT/ACT/FiLM ou outros mecanismos.

Cinco braços, seed nova251: Transformer2blocos; mLSTM2células; mLSTM1célula;
hierarquia independente com leitura seguinte atual; mesma hierarquia com leitura
imediata. T1024/B8, WikiText preparado fixado, byte-level, FP32/Inductor,
dropout0,05,6000passos e taxa0,0004. Reutilizar receita inicial da0027 como triagem,
sem afirmar ajuste ótimo para os dois novos controles. Treinar todos do zero.
Validação completa3000/6000; subconjunto fixo1500/4500; teste reservado intacto.

Adicionar `state_readout: next|immediate`, padrão next, somente hierarquia pode
usar immediate. Permitir1célula apenas em state_recurrent; dimensões não aumentam.
Manter defaults históricos e versão1 nesses casos; novas semânticas usam versão2.
Estados separados por nível; estado de1célula contém1memória. Fórmula/enumeração de
parâmetros e bytes de estado acompanham a seleção. Sem migração de checkpoint.

Contrastes principais: immediate−next;1célula−2células. Hierarquias têm mesmas
operações/dimensões/parametrização e inicialização; só índice de leitura difere.
Comparação com1célula envolve menos parâmetros/operações; não é iso-compute.
Critérios exploratórios anteriores preservados, sem promoção estatística com
uma seed: sinal de qualidade≥0,005NLL; eficiência exige perda≤0,002 e economia≥10%
em benchmark comparável. Cinco seeds novas e custo igualado continuam confirmação.

Sessão máxima7200s incluindo setup, testes, dados, pilotos, treino e avaliações;
reserva300s. Usar duas T4 em jobs independentes. Pilotos isolados dos cinco braços,
incluindo custo frio e cache aquecido; piloto concorrente estima contenção. Prever
treinos, validações, arranque e perfis isolados com35% de margem. Admitir pares
inteiros (contraste de leitura primeiro, profundidade depois, Transformer por
último). Se não couber, dividir jobs entre sessões preservando horizonte/scheduler;
exportar pendências explícitas e não chamar matriz parcial de confirmação.

Uma família de cache por GPU, no máximo2GiB globais amostrados; apagar somente
cache de lane inativa ao mudar família. Treinar por pares e perfilar isoladamente
após ambos encerrarem, na GPU de treino com cache aquecido. Cada braço é medido
em uma GPU nesta triagem; resultados devem identificar o aparelho. Não alegar
economia confirmada nas duas GPUs; avaliação cruzada fica para candidato.

Um ZIP de resultados e um de últimos checkpoints, publicados ao final ou falha.
Sem after-NN/ZIP acumulativo. Salvar checkpoint3000 e6000; após verificar o6000,
remover apenas o intermediário redundante da execução nova. Preservar o último
completo em falha/deadline. Reutilizar dados preparados quando válidos, sem cópia
dos bins. Dados/ZIPs/pesos históricos ficam preservados. Downloads embutidos.

Validar: defaults históricos; causa/fronteiras/streaming forward+gradientes; pesos
e operações das hierarquias pareados; ausência real da segunda célula no fino;
fórmula/tie de embedding; forward/backward/finitude; memorização; safetensors e
retomada próxima atualização com dropout/acumulação. Configurações/semânticas
incompatíveis devem ser recusadas. Testar orçamento, jobs parciais, concorrência,
retenção/publicação, falhas/deadline/espaço e bootstrap fixado. Integração CPU;
GPU/Inductor real será preflight obrigatório no Kaggle. Registrar resultados
efetivos e limitações emVALIDATION.md antes de entregar a célula publicada.
