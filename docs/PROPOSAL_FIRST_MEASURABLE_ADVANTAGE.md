# Proposta — primeira vantagem verificável sobre o Transformer

07/11/2026. Proposta original preservada. Em08/11a implementação foi autorizada;
ver [operação vigente](MEMORY_RECALL_RUN.md), que acrescenta o controle de banco completo.
Não altera retroativamente protocolos/resultados. Depende de implementação,
validação e piloto antes de fixar configurações e horizonte de treino.

## Objetivo

Procurar uma vantagem de arquitetura em uma capacidade definida: recuperar
informações de um contexto com distratores, usando menos recursos para atingir
uma acurácia comum. Depois verificar se o mecanismo ajuda em linguagem.

Competir em uma tarefa não demonstra equivalência geral com modelos maiores.
O marco de 1.000.000× permanece uma ambição sem evidência ou previsão de escala.
Primeiro procurar ganhos de 2× em um recurso explicitamente medido. Menos pesos
isoladamente significa compactação, não redução equivalente de energia ou memória.

## Protótipo candidato

Reutilizar a regra recorrente já validada, sem DecoderBlock ou autoatenção global
de tokens. A implementação local é uma regra mLSTM num shell comum, não xLSTM
completo. Reutilizar inicialização/RMSNorm/SwiGLU não cria um Transformer.

Organização inspirada em fractais: dois níveis causais, ritmo fino/amplo, mesma
regra compatível compartilhada, estados distintos por nível. Acrescentar um banco
de resumos latentes de grupos concluídos, consultado pelo estado corrente.
Manter compressão/formação dos resumos fixa entre padrões nesta primeira triagem.
O banco deve ser escrito e lido por mecanismos aprendidos; sem parser que receba
chaves/valores verdadeiros do gerador. O protótipo usa resumos vetoriais, não uma
reprodução integral de MC/GRM com snapshots de matrizes de memória.

Consultar apenas um subconjunto causal dos resumos. Políticas candidatas:
distâncias uniformes, geométricas de base dois e Fibonacci. Mesmos bytes de banco,
número de consultas, janela local e alcance máximo. Fixar antes do treino a regra
de escala/arredondamento/deduplicação dos offsets e cobertura das bordas; não
comparar padrões com números diferentes de vizinhos ou maior alcance oculto.
Comparar custo do índice, armazenamento, formação de resumos e leitura completos.
Não presumir complexidade linear, aceleração CUDA ou vantagem específica de phi.

## Primeira competição

Seis braços de triagem, dimensões/contagens resolvidas antes do treino:

1. Transformer de aproximadamente2M parâmetros, referência externa.
2. Recorrente simples de aproximadamente2M, controle já existente.
3. Hierarquia compartilhada sem o novo banco, controle da organização candidata.
4. Mesma hierarquia com banco e consultas uniformes.
5. Mesmo banco com distâncias baseadas em potências de dois.
6. Mesmo banco com distâncias de Fibonacci.

Não inflar a largura do compartilhado para esconder sua compactação. Os três
braços com banco terão parâmetros/operações de leitura equivalentes; o efeito do
banco frente à ausência dele é comparação de pacotes com seus custos declarados.
Essa matriz não isola novamente o efeito de sharing: ele poderá ter controle
independente depois se o pacote candidato mostrar benefício.

Treinar do zero em associações aleatórias por exemplo, com consultas múltiplas,
valores novos e distratores. Reservar exemplos/seeds do gerador para avaliação.
Distâncias aleatórias, sem concentrar respostas nos offsets de um dos padrões.
Distribuição de comprimentos e orçamento de exemplos iguais entre braços.
Treino inicial em1024posições; sondas em1024/2048/4096/8192, separando interpolação
e extrapolação. A cobertura efetiva da memória em cada comprimento deve ser
documentada; aumentar banco na avaliação é um experimento separado.

Subteste separado de atualização/esquecimento para medir interferência; não mudar
a semântica das respostas em silêncio. Dataset inspirado em MQAR, com desvios
de tokenização/gerador explicitados; não anunciar reprodução de benchmark oficial.

Dois seeds de triagem, exploratórios. Mesmos exemplos, horizonte fixo, schedulers,
precisão validada e esforço de ajuste simétrico. Não treinar o candidato até
convergir e interromper a referência antes. Testes de integridade e piloto de
duração antecedem a definição final do horizonte. Matriz pode ocupar sessões
distintas; nunca reduzir a dificuldade para produzir um resultado favorável.

## Resultado legível e critérios

Relatório com acurácia de resposta exata em%, acurácia por distância/interferência,
exemplos concretos corretos/incorretos, tempo para qualidade fixada, latência por
consulta, pico de memória, parâmetros e bytes totais de estados/caches.
Energia de placa por execução/resposta correta quando NVML permitir; host e
compilação separados. Valores indisponíveis recebem null. Intervalos por exemplo
não substituem confirmação entre seeds.

Primeiro alvo proposto: pelo menos95% de respostas exatas, margem de até1p.p.
frente à referência, com pelo menos2× menos latência ou energia no regime medido.
É uma meta futura, não previsão. Candidatos fora desse alvo ainda serão reportados,
sem redefinir o critério depois. Mais acertos por custo equivalente também podem
justificar um novo protocolo de confirmação, sem chamar este alvo de atingido.

O Transformer deve usar caminho eficiente disponível no hardware e mesma política
de precisão validada. Medir prefill e decodificação separadamente; comparação de
geração autorregressiva requer KV cache correto na referência. Perfil atual do
projeto não implementa ainda esse benchmark. Não usar recomputação de todo o
prefixo no Transformer para anunciar aceleração da recorrência.

## Ponte para uma LLM e modelos maiores

Se houver sinal, testar o candidato em modelagem de linguagem no corpus preparado,
com avaliação reservada e comparação justa. TinyStories pode complementar exemplos
de continuação de texto, sem substituir a avaliação de linguagem já existente.
Resultados de recuperação não serão apresentados como capacidade linguística geral.

Comparação posterior de compactação: candidato aproximadamente1–2M contra
Transformer aproximadamente8M, ambos treinados adequadamente na mesma tarefa.
Não consumir agora o orçamento com essa matriz maior antes do sinal de triagem.
Reportar separadamente alvos iguais e custo igualado. Confirmar candidato com
cinco seeds novas antes de alegar vantagem reproduzível.

## Orçamento, validação e encerramento

Sessões de até2h nas duas T4, incluindo setup/testes/compilação/avaliação/exportação.
Piloto com margem35% e reserva de exportação; dividir jobs inteiros se necessário.
Não prometer que todas as variantes e comprimentos caberão numa única sessão.
Um par final de ZIPs, sem snapshots cumulativos. Preservar último checkpoint
completo; contabilizar cache e bytes de memória adicionais no working/VRAM.

Validar causalidade, igualdade sequencial/chunks, fronteiras e leitura somente de
grupos fechados, isolamento entre documentos, ausência de informações de rótulos,
influência dos caminhos, forward/backward/finitude e retomada com dropout e
acumulação. Padrões devem passar por contagens/alcance/duplicações e controles de
posição. Parâmetros compartilhados não são contados duas vezes; estados são.

Uma triagem por proposta, confirmação longa apenas se houver sinal. Empate entre
Fibonacci e potências de dois significa ausência de vantagem específica de
Fibonacci. Falha da memória vetorial não encerra MC/GRM, TTT ou a hipótese fractal;
encerra ou redireciona essa implementação conforme o diagnóstico.

## Referências

- [Zoology/MQAR](https://arxiv.org/abs/2312.04927): recuperação associativa como
  capacidade discriminante entre arquiteturas; não garante vitória da recorrência.
- [Memory Caching](https://arxiv.org/html/2602.24281v1): inspiração para armazenar
  representações anteriores; implementar banco vetorial não reproduz o artigo.
- [Fibonacci sparse attention, preprint2026](https://arxiv.org/abs/2606.28560):
  precedente de espaçamento, também com controle base dois e resultados negativos.
