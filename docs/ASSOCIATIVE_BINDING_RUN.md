# Memória de transições e pirâmide — campanha a partir da0031

Entrega09/11/2026. [Plano prévio](PLAN_ASSOCIATIVE_TRANSITION_IMPLEMENTATION.md).
A autorização de mudanças amplas levou a uma família nova, independente da
adaptação mLSTM. Os backbones e protocolos antigos continuam reproduzíveis por
suas revisões. Não repetimos a matriz da0030 nem migramos seus pesos.

## O que mudou

Endereço e conteúdo são separados. A consulta do byte atual usa um endereço
aprendido; a escrita do conteúdo atual usa o endereço da posição anterior.
Portas de escrita e codificação local causal são aprendidas. Esse viés explícito
de transição é adequado a associações adjacentes; não é uma capacidade geral de
binding já demonstrada. Nenhum tokenID, marcador, máscara de escrita ou rótulo é
interpretado pelo backbone. O modelo aceita streams arbitrários de tokens.

O estado é uma matriz aditiva de pesos rápidos. A execução agrupa atualizações
matriciais com prefixo cumulativo exclusivo, somando somente chunks passados e
leituras causais locais. Custo em comprimento é linear com dimensões/grupo fixos;
o estado persistente tem tamanho fixo. Ativações de treino ainda crescem comT.
Memória fixa pode sofrer interferência; não é retenção ilimitada.

É pesquisa em memória associativa/atenção linear, não mLSTM completo nem uma nova
lei matemática. Referências: [atenção linear](https://arxiv.org/abs/2006.16236v3)
e [pesos rápidos](https://arxiv.org/abs/2102.11174v3). Não implementa o algoritmo
completo de TTT, GRM, delta rule ou aprendizado interno por gradiente.

| Braço | Parâmetros únicos | Estado tensorial/seq | Pergunta |
|---|---:|---:|---|
| Transformer, dois blocos | 2.164.992 | null | Referência externa |
| Transições, memória única | 86.785 | 50.304bytes | O novo mecanismo recupera associações? |
| Pirâmide compartilhada | 86.979 | 50.304bytes | Particionar a memória em ritmos ajuda? |
| Pirâmide independente | 111.652 | 99.456bytes | Compartilhar os operadores prejudica ou ajuda? |

Os bytes são tensores do estado retornado, sem overhead Python/ativações/pesos.
Transformer sem KV neste forward retorna null, não zero. As pirâmides separam
grupo corrente de quatro posições e prefixo de grupos concluídos. Cada escrita
participa uma vez. Gate dependente da consulta combina as leituras. A versão
compartilhada reutiliza as mesmas projeções/endereço/conteúdo/leitura nos níveis;
a independente usa cópias que podem divergir. Com cópias iguais e gates neutros,
o forward inicial coincide. Isso testa uma organização inspirada em fractais;
não prova propriedades de um fractal matemático.

Há cerca de25× menos parâmetros no núcleo único que no Transformer desta matriz.
Esse é um fator de compactação. Só haverá candidato de eficiência se qualidade,
latência, memória e energia medidas também sustentarem a comparação.

## Qualificação local: ganho funcional, com limites

O [registro de desenvolvimento](results/transition_development_qualification.json)
contém uma qualificação CPU, seed816,1000passos, FP32, sem compilação, núcleo único
de86.785pesos, contexto256/B16/dropout0,05/LR0,001, currículo fixo4/8/16/32pares.
Não é a campanha Kaggle; seu scheduler de1000passos difere do principal3000.

Após o treino, seis condições novas, cada uma com32sequências/512consultas e
suas intervenções, obtiveram **100% original e100% nos pares contrafactuais**:
T256/1024/4096/8192 com32associações, T1024com64 e T1027com prefixo deslocado3bytes.
Os pesos foram reavaliados explicitamente com a versão equivalente de prefixo
exclusivo e novas seeds de medição, sem carregar otimizador ou alegar retomada
exata entre fontes diferentes. Hash de pesos e ambas as fontes estão registrados.
Não houve comparação com Transformer nessa qualificação, medição de GPU ou teste
de linguagem. São mapas novos sobre o mesmo alfabeto finito64chaves/64valores,
não símbolos inéditos, textos livres ou tarefas de raciocínio.

## Protocolo Kaggle

- Quatro braços × seeds311/312 = oito treinos novos,3000passos cada. Duas T4,
  um worker por placa, distribuição física alternada entre seeds.
- Contexto256, batch16, FP32/Inductor, dropout0,05, LRcomum0,001, warmup50,
  weight decay0,01. Scheduler/horizonte não mudam por resultado ou tempo restante.
- Shapes constantes. Currículo fixo de escritas:4nos passos1–200,8em201–400,
  16em401–600,32em601–3000.16consultas por sequência; novos mapas a cada exemplo.
  Cada treino tem768.000alvos supervisionados e12.288.000bytes de input, repetidos.
- Chaves e valores distintos por sequência. Resposta só como rótulo na posição
  da chave consultada, nunca inserida na consulta. Todos veem o mesmo fluxo porseed.
  Comparação a inputs/alvos iguais, não parâmetros, tempo ou FLOPs iguais.
- Avaliação fixa de64sequências a cada750passos. Além do acerto normal, trocar
  as chaves das perguntas exige as respostas correspondentes. A métrica pareada
  só aceita a consulta quando ambos os mapas de pergunta/resposta estão corretos.
- Avaliação inicial antes de qualquer treino, inclusive4/32associações. Registra
  quanto vem da estrutura/inicialização e quanto foi aprendido.
- Sondas finais:64sequências por comprimento256/1024/4096/8192,32associações.
  Capacidade64emT1024 e deslocamento de fronteira com3bytes de prefixo emT1027.
  Comprimentos acima256 são extrapolação. Não há treino no teste.
- Checkpoints1500/3000; remover apenas1500depois de verificar o final. Manter
  metadados do intermediário. Um ZIP final de resultados e um de pesos/estado.
  Não há backups cumulativos `after-NN` nem corpus baixado.

## Operação e medições

Célula curta: [09_testar_associacao_e_piramide_kaggle.py](../scripts/09_testar_associacao_e_piramide_kaggle.py).
Revisão fixada:`bcd7780e8d929ce938c69a8eaf8167241518e744`.
SHA-256 do bootstrap:`0bee532c03cb07974a7649be32ddbe6a9715c64f73db289880590de5f736f2f9`.
Ativar Internet
e GPU T4x2. Número mínimo0031, incremento automático se já reservado. Teto7200s
desde a célula, com300s reservados para exportação, incluindo setup/testes.

Um worker persistente faz aquecimento e12updates de medição, restaura exatamente
pesos/otimizador/scheduler/RNG/sampler/contadores e continua no mesmo compilador.
Esses updates não entram no horizonte. Não alteramos nem contornamos o limite de
passos do Trainer: o benchmark possui sua chamada explícita de medição, seguida
de restauração. O teste CPU confere estado e próxima atualização bit a bit.

Depois do piloto, cada job publica a previsão com35% de margem e reserva de
avaliação. Só inicia3000passos se couber no tempo disponível; caso contrário
fica adiado. Não promete os oito treinos em2h. Após cada par, o restante é
admitido com o relógio corrente. Não há bateria de14pilotos nem recriação do
modelo entre treino e perfis. Caches por GPU persistem durante a sessão, com
limite global2GiB; reserva livre2GiB; remoção somente do cache próprio ao final.

Os workers terminam treino/avaliação antes de liberar perfis, um por vez, através
de uma barreira. Há mediana30updates de treino restaurados e forward completo
eager medido separadamente em **cada comprimento real**, com30amostras e janela
de energia de placa quando disponível. Não são geração com KV ou custo para
qualidade comum. Compilação/piloto/treino/avaliação têm campos próprios. Medições
indisponíveis são null; FLOPs dos novos modelos permanecem null.

Diagnóstico fora do benchmark: gradientes por parâmetro, portas de escrita por
papel na tarefa e exemplos de respostas originais/trocadas. O parser existe
apenas no gerador/avaliador/diagnóstico, nunca no modelo. Estatísticas por
sequência permitem avaliar correlação entre consultas.

Treinos nativos completos com mesma fonte/protocolo/config/runtime são verificados
e reutilizados. Checkpoints interrompidos são preservados; a operação não retoma
automaticamente passos/avaliações incompletos. Históricos completos não duplicam
pesos no novo ZIP: preservar os backups originais. Download embutido no notebook,
com `LLM2_BAIXAR` disponível para repetir o botão/baixar partes grandes.

## Decisão que queremos obter

Meta95% de recuperação,90% de pares contrafactuais corretos. Se o candidato
mantiver qualidade dentro de1p.p. do Transformer, registrar compactação e fatores
de tempo/energia separadamente. Fator2em latência só conta como candidato de
eficiência quando a qualidade também passa. As duas seeds são triagem; não
confirmação geral. Não transformar referência ruim em vitória de uma LLM.

Se funcionar naT4, o próximo marco é linguagem (TinyStories/WikiText), com
dados/orçamento próprios e confirmação independente. O mecanismo aditivo ainda
não oferece sobrescrita seletiva de fatos repetidos, memória vitalícia, esforço
adaptativo ou recuperação de documentos arbitrários. Fibonacci, TTT/GRM e demais
frentes permanecem no plano, com testes posteriores próprios.
