# Memória associativa de transições — implementação autorizada em 09/11/2026

## Pergunta e mudança de direção

A0030 não qualificou associação em exemplos novos. O usuário autorizou mudanças
amplas para buscar vantagens claras. Preservar a ciência histórica; adicionar uma
família independente, sem substituir defaults ou carregar seus checkpoints.

Nova hipótese: separar endereço e conteúdo, com viés causal de transição entre
posições vizinhas, permite reter associações sem um Transformer ou resumos médios.
A memória será matriz de pesos rápidos aditivos, com endereço aprendido
compartilhado entre consulta atual e escrita da posição anterior, conteúdo e
porta aprendidos. Não haverá parser de marcadores/rótulos no modelo.
É uma variante de memória associativa/atenção linear, não reivindicação de
descoberta de uma regra inédita nem reprodução de mLSTM/TTT/GRM.

## Controles arquiteturais

1. Transformer de dois blocos histórico, como referência externa.
2. Memória única de transições, dimensões menores e custo de estado declarado.
3. Pirâmide de dois ritmos, usando a mesma escrita/leitura aprendida nos níveis.
4. Mesma pirâmide com operadores independentes, inicialmente copiados.

A pirâmide separa prefixo de grupos completos e grupo fino corrente de quatro
posições, usando operadores matriciais que preservam associações. Gate por
consulta combina as leituras. Não duplicar cada escrita nos dois níveis.
Na inicialização com pesos copiados e gates neutros, controles de sharing terão
forward igual. Menos parâmetros é compactação; eficiência exige qualidade/custo.

Treinamento vetorizado por chunks: interações causais dentro do chunk e prefixo
de atualizações matriciais entre chunks, sem materializar matriz de estado por
token ou atenção densa global. Streaming carrega matriz, endereço anterior e
contexto local; reset explícito entre documentos. FP32, RMS acumulado em FP32.

## Tarefa e campanha

Novo protocolo/identidade. Pares de chaves e valores distintos por sequência;
novos mapas a cada exemplo, valores nunca entram como respostas nas consultas.
Fases fixadas antes de treinar:4/8/16/32associações em contexto256, mesmas fases
para todos. Shapes de treino constantes para compilar uma vez por worker.
Seeds novas; snapshots, contadores e sampler preservam a fase exata em retomada.
Avaliação inclui acerto original, perguntas com chaves trocadas e respostas
correspondentes, capacidade, comprimentos maiores e controle simbólico separado.
O viés de transição favorece pares adjacentes; declarar essa especialização e
não promover a vitória na tarefa a superioridade geral de linguagem.

Primeiro qualificar generalização local em exemplos novos. Se falhar, corrigir
antes da entrega. No Kaggle, cada worker persistente faz aquecimento/medição,
treino e avaliação sem recriar seu modelo/compilador entre etapas. Dois jobs,
um porT4, com teto global2h, reserva5min e previsão conservadora. Sem bateria
fria de14pilotos. Matriz divisível; não reduzir horizonte/scheduler por tempo.

## Validação e aceitação

- Equivalência sequencial/chunks e gradientes, causalidade, fronteiras4,
  compartilhamento/contagem, finitude, memorização e generalização.
- Retomada exata CPU com dropout e acumulação, incluindo RNG/sampler/fase.
- CUDA/Inductor verificados no Kaggle; não simular suporte local.
- Comparar acerto e custo de forward sob a mesma política; declarar estado,
  energia de placa, treino/compilação separados. Sem comparar geração sem KV.
- Resultados iniciais antes do treino, para distinguir viés estrutural de
  aprendizado. Objetivos:95%recuperação e fator2em recurso, não garantias.
- Oráculo simbólico só valida dados. Modelo não recebe chaves/valores verdadeiros
  extraídos nem máscaras de escrita do gerador. Testes contrafactuais verificam
  dependência da chave. Confirmação posterior exige seeds/dados novos e linguagem.
- Um par ZIP final/falha; último checkpoint porjob; caches temporários limitados;
  não remover dados antigos. Download embutido e célula curta fixada porGit/SHA.

Riscos: interferência de memória aditiva, viés da defasagem1, colisões de endereço,
estado crescendo em magnitude, custo da pirâmide e compilação. A arquitetura
deve aceitar byte streams arbitrários, mas a campanha ainda é recuperação
sintética, não prova de uma LLM melhor. Nenhum ganho será inventado para cumprir
o desejo de fatores grandes.
