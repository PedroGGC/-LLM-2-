# Memória, fractal e Fibonacci — campanha a partir da0030

Entrega08/11/2026. A nova bancada foi autorizada depois da0029; não repete sua
matriz e não baixa WikiText. [Plano de implementação](PLAN_MEMORY_RECALL_IMPLEMENTATION.md).
O objetivo é procurar vantagem verificável numa capacidade antes de escalar.

Célula curta: [08_testar_memoria_e_fibonacci_kaggle.py](../scripts/08_testar_memoria_e_fibonacci_kaggle.py).
Revisão fixada:`dc2888a876aa0f3718a27114ea1d3c637fadb25f`.
SHA-256 do bootstrap:`c50fb24dfff81b00b41dcc6074c5bca102f83cb110f79785349e1902e27de21c`.
Ativar Internet e GPU T4x2 antes de rodar. O número avança se0030estiver reservado.

## O que será comparado

| Braço | Parâmetros únicos | Implementação |
|---|---:|---|
| transformer | 2.164.992 | Transformer causal,2blocos |
| recurrent | 2.174.232 | Duas células recorrentes por byte |
| hierarchy | 1.136.652 | Dois ritmos, uma regra mLSTM compartilhada |
| memory_all | 1.284.878 | Hierarquia compartilhada e consulta de todos os resumos disponíveis |
| memory_uniform | 1.284.878 | Mesmo banco,16leituras com espaçamento uniforme |
| memory_binary | 1.284.878 | Mesmo banco, gaps inspirados em potências de dois |
| memory_fibonacci | 1.284.878 | Mesmo banco, gaps inspirados em Fibonacci |

O quarto braço é controle adicional à proposta inicial de seis: evita confundir
falha do banco com informações descartadas pelos padrões esparsos. Sua quantidade
de leitura é maior; não anunciar equivalência de computação entre denso/esparso.
Os quatro bancos iniciam com pesos idênticos por seed. Compartilhar pesos mantém
estados diferentes por nível; não confundir1,285Mpesos com bytes totais de memória.

Banco de vetores contextualizados de grupos causais completos, máximo2048slots,
dimensão original384; projeções de consulta/chave/valor com96dimensões. A leitura
é condicionada ao estado fino e uma porta aprendida mistura seu resultado. Núcleo
recorrente, sem DecoderBlock ou autoatenção global de tokens. Não é reprodução
integral de MC/GRM, xLSTM ou do artigo de atenção Fibonacci.

Fibonacci/base2/uniforme são frações de gaps reescaladas ao histórico disponível,
com o mesmo número de posições distintas e os mesmos extremos. Não são offsets
Fibonacci literais. A regra exata está no protocolo/código e nos testes. Nenhum
modelo recebe informação verdadeira do gerador para escolher posições ou escrever
o banco; só inputs, perda e gradientes. Objetivo causal original reutilizado.

## Tarefa e dados

Recuperação associativa procedural, um byte de resposta:32registros de chave/valor
no primeiro meio e16consultas no segundo meio, com posições aleatórias/distratores.
Registros de4bytes.64chaves possíveis,64valores; chaves únicas por sequência e
valores aleatórios por exemplo. Nas consultas, respostas são rótulos separados,
não entram no input. Essa estrutura/alinhamento são conhecidos de todos os modelos.
Não é teste em linguagem natural nem reprodução oficial de MQAR.

Se o banco guardar/responder melhor a essas associações com custo menor, haverá
um resultado de recuperação. Isso não estabelece competência de uma LLM geral.
TinyStories, linguagem WikiText e confronto posterior comTransformer8M continuam
próximas confirmações condicionadas ao sinal, fora desta rodada.

## Protocolo

- Seeds271/272;3000passos fixos, T1024/B8, FP32/Inductor, dropout0,05 e LR0,0004.
  São384.000consultas supervisionadas e24.576.000bytes de input por treino; leitura
  do contexto não é contada como alvo supervisionado.14treinos no total.
- Mesmos exemplos/ordem por seed para todos os braços. Pilotos não continuam no
  treino principal. Scheduler não muda para preencher tempo ou porque alguém vence.
- Desenvolvimento fixo em750/1500/2250/3000;64sequências por avaliação, batch1.
  Checkpoints1500/3000. Depois de verificar3000, remover apenas1500redundante.
- Avaliações finais:128sequências por comprimento1024/2048/4096/8192, mesmas seeds
  reservadas;32escritas/16consultas. Capacidade:8/64escritas emT1024,64sequências.
  Distâncias aleatórias, sem favorecer os offsets de um padrão. Manter teto do banco.
- Essas sondas finais poderão orientar a campanha seguinte; são exploração,
  não um teste definitivo independente de toda seleção futura.

## Tempo, continuidade e arquivos

Até7200s desde a célula, incluindo clone/pip/testes/compilação/pilotos/avaliação e
reserva300s. Duas T4 em treinos independentes, rotação física entre seeds. Pilotos
isolados frio/aquecido de cada braço e piloto concorrente antes da admissão de
pares inteiros com35% de margem. Se a matriz não couber, concluir parte e registrar
pendências; não reduzir horizontes. O script não promete14treinos completos em2h.

O número começa em0030 e avança automaticamente se já usado. Ao repetir a mesma
célula no mesmo working, reutiliza somente treinos nativos concluídos com mesma
fonte/protocolo/configuração/runtime, verificando checkpoints reais. Se o treino
terminou e a avaliação final foi interrompida, repete a avaliação, sem repetir
os passos de treino. Jobs interrompidos antes do horizonte mantêm checkpoints,
mas não são retomados automaticamente por esta operação.

Um ZIP final de resultados e um de últimos checkpoints completos, sem after-NN.
O resultado contém fonte canônica uma vez, configs, logs, curvas, exemplos e
`report/report.html`, `report/scores.csv`, `report/report.json`. Checkpoints ficam
fora do Git. Zips são publicados também em falha/deadline quando houver orçamento.
Histórico reaproveitado aponta para os pesos originais, não os duplica no novo ZIP.

Cache por GPU/família, limpeza apenas em lane inativa, limite global amostrado2GiB
e reserva livre2GiB. Verificação inicial inclui pesos transitórios e ZIP final.
Não apaga campanhas/dados/ZIPs históricos do usuário. Download embutido no notebook,
sem URLs de proxy; arquivos grandes podem ser baixados em partes por `LLM2_BAIXAR`.

## Como interpretar

Resultados em acurácia exata%, perda por consulta, buckets de distância e exemplos.
Perfis isolados após treino: forward completo batch1/T1024, mediana30amostras,
VRAM, RAM, estado retornado e energia de placa quando disponível. Perfis não são
latência de geração/KV nem custo de atingir acurácia comum. Não aplicar o tempo
T1024 às linhasT8192. Energia exclui host/compilação; indisponibilidade é null.

Metas prévias95%acerto/margem1p.p./fator2em recurso são objetivos, não resultados.
Comparar custo somente junto à qualidade e no mesmo regime. Dois seeds são
triagem, sem confirmação geral. Não declarar sucesso se a referência/avaliação
estiver ausente ou falhar em algum comprimento. Backend automático é política,
não identificação observada de kernel. FLOPs recorrentes continuam null.

Testes locais/limitações constam em [VALIDATION](../VALIDATION.md). CUDA e Inductor
reais são preflight obrigatório no Kaggle, antes dos treinos principais.
