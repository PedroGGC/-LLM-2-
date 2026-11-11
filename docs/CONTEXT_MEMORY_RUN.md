# Memória contextual, profundidade e hierarquia — campanha a partir de 0033

Implementação autorizada em 10/11/2026, entrega em 11/11. Branch
`experiment/fractal-lrm`. [Plano mestre](PLAN_RECURRENT_FRACTAL.md),
[plano de implementação](PLAN_CONTEXT_MEMORY_IMPLEMENTATION.md) e
[verificações reais](../VALIDATION.md).

A execução da 0033 concluiu os 16 treinos nas duas T4. A auditoria registra
qualidade, custo e a decisão para o próximo ensaio de geração.
[Resultados da 0033](results/CAMPAIGN_0033.md).

## O que mudou e por quê

A 0032 demonstrou aprendizado de linguagem, mas o Transformer de 86.400 pesos
venceu os candidatos de aproximadamente 87 mil. O agregado anterior conserva
transições locais e pode perder a ordem entre trechos: permutações com a mesma
contagem de triplas podem produzir a mesma consulta final. É uma limitação
estrutural verificável, não uma explicação exclusiva para todos os resultados.

A nova família usa memória matricial com esquecimento dependente do conteúdo:

```
S_t = a_t S_(t-1) + b_t k_t v_t^T
r_t = q_t S_t
```

Cada cabeça tem uma porta escalar `a` entre 0 e 1 e uma porta de escrita `b`.
Consultas, chaves, valores e portas recebem representações de uma convolução
causal de quatro posições. Q/K e acumulação são FP32. Leitura normalizada por
cabeça, porta de saída, residual e SwiGLU completam a célula. As portas iniciam
com escalas de esquecimento de 8/32/128/512 passos e podem aprender outros ritmos.
As passagens seguintes recebem representações contextualizadas pela anterior.

[RetNet](https://arxiv.org/abs/2307.08621) e
[GLA](https://arxiv.org/abs/2312.06635) fundamentam a investigação de retenção,
portas e formas paralelas/recorrentes. Esta implementação é uma adaptação com
decay escalar por cabeça; não reproduz as arquiteturas ou kernels completos.
Não implementa a atualização corretiva de
[Gated DeltaNet](https://arxiv.org/abs/2412.06464), nem o aprendizado interno de
[TTT](https://arxiv.org/abs/2407.04620). Essas hipóteses permanecem disponíveis.

Não há DecoderBlock, SDPA ou atenção softmax global nos candidatos. A matemática
pertence à família de retenção/atenção linear: reutilizar seus princípios não
estabelece novidade científica nem vantagem sobre Transformer.

## Onde está a hipótese fractal

Na hierarquia, o caminho fino processa cada byte. Uma média de quatro estados
finos **já contextualizados** alimenta o caminho amplo somente quando o grupo
termina. O grupo completo pode ajudar a prever o próximo byte, jamais um byte
anterior. Grupos incompletos são carregados no estado para o próximo chunk.

O mesmo operador processa ambos os níveis na versão compartilhada. A versão
independente tem as mesmas dimensões, operações e comunicação; começa com cópias
iguais dos pesos e pode se especializar. Cada nível/passagem mantém seu próprio
estado, inclusive quando os pesos são compartilhados. O teste distingue
hierarquia, especialização e economia de pesos. Não pretende provar propriedades
de um fractal matemático ou reproduzir literalmente o cérebro.

## Matriz congelada

| Variante | Parâmetros | Pergunta |
|---|---:|---|
| `transformer_matched` | 86.400 | Referência externa próxima dos modelos compartilhados |
| `context_single` | 85.848 | Uma passagem da nova memória |
| `transformer_medium` | 148.960 | Referência externa próxima dos modelos independentes |
| `context_shared2` | 85.848 | Duas passagens finas com o mesmo operador |
| `context_untied2` | 151.056 | Duas passagens finas independentes |
| `hierarchy_shared` | 85.928 | Fino + amplo/4 com operador compartilhado |
| `hierarchy_untied` | 151.136 | Mesma hierarquia com operadores independentes |
| `transformer` | 2.164.992 | Referência maior preservada |

Contagens por enumeração; fórmula independente para a nova família e embedding
ligado à saída. Candidatos D80/FFN144, quatro cabeças, 16 dimensões de chave por
cabeça. Não aumentar a largura do compartilhado para esconder a compactação.

Dezesseis treinos novos: seeds **341/342**, **6.000 passos**, TinyStories preparado
da 0032, byte-level, contexto 256, batch 32, FP32/Inductor, dropout 0,05.
Calibração simétrica de duas taxas, 0,0003 e 0,001, 400 passos por taxa, seed 941.
Escolha pela subdivisão de calibração extraída do treino; novo estado inicial em
cada treino principal. Nenhum peso da 0032 ou da calibração é transferido.

Avaliação de todos os bytes da divisão de desenvolvimento em 1.500/3.000/4.500/
6.000; caudas incluídas, BOS/padding excluídos, documentos isolados. Teste reservado
não consultado. O corpus é uma seleção determinística de prefixos limitados de
TinyStories, não o corpus inteiro; detalhes em [dados da 0032](TINYSTORIES_RUN.md).
Três prompts fixos, greedy e amostragem, 256 bytes novos: **96 textos** se todos
os treinos terminarem. Todos serão exportados, inclusive resultados ruins.

O pacote novo testa uma mudança de família. O contraste com a 0032 é histórico;
não isola causalmente cada nova porta/normalização/convolução. Dentro da rodada,
as ablações isolam profundidade, compartilhamento e organização hierárquica.

## Execução e resultados

Cole a [célula para Kaggle](../scripts/11_testar_memoria_contextual_kaggle.py)
com revisão e SHA fixados em um notebook com Internet e GPU T4 x2.
Um job por GPU, alternância entre seeds e **teto de duas horas**
desde o início da célula. Não é duração obrigatória. Pilotos com margem de 35%
decidem a admissão de cada par; horizontes/schedulers não são encurtados para
caber. Se faltar tempo, exportar parcial e identificar os jobs pendentes.

Usa workers persistentes, restauração exata do piloto, barreira para benchmarks
isolados e reserva de cinco minutos para exportação. O corpus preparado pode ser
reutilizado após verificação de hashes. Reuso de jobs exige protocolo, dados,
configuração, fonte e ambiente iguais. Campanhas antigas não satisfazem o novo
protocolo. Não é necessário apagar o working antes da célula.

Somente um par de ZIPs finais: resultados e checkpoints, com botões de download
embutidos no notebook. Mantém o último checkpoint completo de cada job; estados
de calibração são descartados depois de preservar seus metadados. Cache privado
de compilação limitado a 2 GiB e removido ao final; reserva mínima de 2 GiB livres.
Não cria arquivos cumulativos `after-NN`. A numeração avança se 0033 já existir.

O relatório traz NLL/bits por byte, curvas por alvos/tempo, todos os textos,
contagens de pesos, tempo isolado de atualização, perfis de forward, RAM/VRAM,
estado retornado e energia da placa quando disponível. FLOPs da família nova
permanecem `null` até existir estimador validado; ausência não significa custo zero.
Forward é eager sobre bytes aleatórios, não decodificação com cache. A geração
qualitativa recompõe a mesma janela limitada para todos os modelos.

A implementação usa produtos locais 32x32 e prefix scan afim entre blocos.
O scan por duplicação tem trabalho O(N log N) para N blocos, não treinamento
estritamente linear. A inferência recorrente guarda estado de tamanho fixo;
streaming é testado contra o forward completo. Testes de correção em CPU não
substituem medição de velocidade, VRAM e compilação na T4.

## Decisão depois da rodada

Um ganho de qualidade candidato exige pelo menos 0,005 NLL médio, favorável nas
duas seeds. Cada marco **2x/4x/8x/20x/...** exige referência nomeada, horizonte
completo, NLL final até 0,002 pior e fator de tempo isolado atingido nas duas
seeds. O relatório não promove automaticamente menos pesos a mais eficiência.
São sinais exploratórios de custo por atualização com qualidade final equivalente;
tempo total para atingir uma qualidade fixa e confirmação independente vêm depois.
O objetivo de 1.000.000x é aspiracional, sem evidência ou promessa de viabilidade.

- Compartilhado bom e independente similar: confirmar compactação/qualidade por custo.
- Profundidade fina útil e hierarquia fraca: testar compressão aprendida ou o local
  de compartilhamento, sem forçar a pirâmide.
- Memória insuficiente: atualização corretiva, MC/GRM e TTT permanecem alternativas.
- Benefício de mais passos internos: treinar distribuição de profundidades antes
  de ACT/PonderNet, separando refinamento de nova escrita da mesma entrada.
- Todos inferiores em qualidade e custo: encerrar a receita, registrar diagnóstico
  e selecionar outra família; não repetir apenas para ocupar a sessão.

O catálogo do plano mestre permanece: Fibonacci é um padrão de acesso possível,
não um requisito; BLT/Cantor, compressão, memória consultável, TTT, profundidade
variável e engenharia de geração continuam com perguntas próprias. Nenhuma
tecnologia é promovida só por fazer parte da ambição do projeto.
