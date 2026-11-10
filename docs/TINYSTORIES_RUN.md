# Linguagem após a campanha 0031

Implementação de 10/11/2026, branch `experiment/fractal-lrm`. A próxima campanha
começa na numeração 0032. A 0031 resolveu associação sintética nos candidatos,
mas sua referência Transformer não qualificou; isso não demonstra superioridade
geral. Agora mediremos aprendizado de linguagem e publicaremos os textos gerados.

## Comparação congelada

| Variante | Parâmetros | Pergunta |
|---|---:|---|
| Memória de transições simples | 86.785 | O núcleo associativo aprende linguagem? |
| Pirâmide com operador compartilhado | 86.979 | Separar ritmos de memória ajuda além do núcleo simples? |
| Transformer pequeno | 86.400 | Como se compara com orçamento próximo de pesos? |
| Transformer de dois blocos histórico | 2.164.992 | Qual é a distância para a referência maior? |

Seeds 331/332, oito treinos principais de 6.000 passos fixos, contexto 256,
batch 32, byte-level, FP32/Inductor e dropout 0,05. As janelas sorteadas são
idênticas entre braços de cada seed; alvos de padding são mascarados. São até
49.152.000 bytes supervisionados por treino, com repetição. Documentos curtos
podem reduzir esse total; a campanha verifica a igualdade real entre modelos.
Não é comparação com computação igualada. Menos pesos será compactação, não
equivalência automática de capacidade.

Antes dos treinos principais, cada modelo recebe duas tentativas independentes
de 400 passos, taxas 0,0003 e 0,001 e seed 931. Escolha pela NLL em `cal_val`,
derivada exclusivamente do treino; menor taxa desempata. Os dois treinos principais
reinicializam pesos/otimizador/scheduler e usam a mesma taxa selecionada.
Preservam-se os caches de compilação no processo, não os pesos da calibração.
É uma busca curta pré-fixada, sem garantia de taxa ótima em 6.000 passos.

## Dados pequenos e verificáveis

Fonte: [TinyStories oficial](https://huggingface.co/datasets/roneneldan/TinyStories),
revisão `f54c09fd23315a6f9c86f9dc80f725de7d8f9c64`.
[O artigo](https://arxiv.org/abs/2305.07759) descreve histórias sintéticas em inglês;
esta rodada não representa a diversidade de uma LLM geral.

Baixam-se somente os primeiros 32 MiB do arquivo de treino e 2 MiB do arquivo
de validação, com tamanho e SHA-256 fixados. O download é limitado mesmo se o
servidor responder sem respeitar Range. Descarta-se o último registro incompleto,
mantêm-se histórias completas e remove-se duplicação exata entre divisões.
Seleção por hash dentro desses prefixos: não é amostra aleatória do corpus inteiro.

| Divisão | Origem | Bytes | Histórias |
|---|---|---:|---:|
| Treino principal | Treino oficial | 16.777.421 | 18.876 |
| Calibração | Treino oficial, excluído do treino principal | 525.115 | 604 |
| Desenvolvimento | Validação oficial | 524.508 | 630 |
| Teste reservado | Validação oficial, separado do desenvolvimento | 524.973 | 656 |

O corpus não fornece teste oficial. Nosso teste é uma divisão reservada derivada
da validação oficial; ele só recebe verificações de integridade/deduplicação nesta
rodada, sem perda ou seleção por seus conteúdos. Manifesto portátil:
`d61615e79496f14ef473cc10ad11e5deac6d2bda89be7046d738498ff16ba849`.

BOS 256, vocabulário 257 e loss somente nos bytes reais. Nenhuma janela cruza
documentos. Desenvolvimento completo em 1.500/3.000/4.500/6.000 passos, percorrendo
todos os bytes selecionados, incluindo caudas; cada janela reinicia o estado.
Não se prediz EOS nem se conta BOS/padding na NLL.

## O que receberemos

- NLL, bits por byte e curvas por bytes supervisionados e tempo de treino.
- Taxa selecionada, resultados da calibração, fontes/configurações/hashes e
  contadores de cada seed. Os custos de calibração, compilação, piloto e treino
  permanecem distinguíveis nos registros; curvas de treino excluem os anteriores.
- Três prompts fixos por modelo/seed, greedy e amostragem com seed fixa,
  256 bytes novos, temperatura 0,8/top-40. Todos os exemplos são exportados,
  com bytes originais em hexadecimal para identificar UTF-8 inválido.
- Referência train-only de contagens de três bytes com suavização fixa: um
  diagnóstico de estatística local, com seus próprios 68.959.268 bytes de estado,
  sem comparação de orçamento neural ou eficiência com os quatro modelos.
- Benchmark isolado de updates, forward eager em 256/1K/8K, VRAM/RAM, estado e
  energia de placa quando disponível. Forward não é geração com cache.

Geração usa o mesmo limite de contexto 256 e recomputação sem cache para todos.
É uma inspeção qualitativa, sem escolher os melhores exemplos ou atribuir nota de
raciocínio. Um modelo pode melhorar a perda e ainda produzir textos ruins.

## Mudança na execução da memória

`associative_execution=blocked` agrupa prefixos matriciais em blocos de 32,
preservando os grupos lógicos de quatro e os mesmos pesos. O modo `reference`
permanece padrão histórico. Forward, gradientes e streaming são equivalentes
com tolerância FP32, sem migração automática de checkpoints da 0031.

A redução de intermediários é uma hipótese de engenharia. A T4 medirá ambos
os caminhos nos mesmos pesos e dados, registrando erro e pico incremental.
Não prometemos economia de VRAM ou aceleração antes dessa medição.

## Kaggle, duração e arquivos

Ativar Internet e GPU T4 x2. Executar a célula curta por revisão Git e SHA.
O teto de duas horas inclui bootstrap, dados, testes, calibração, compilação,
treinos, perfis e reserva de cinco minutos para exportação. Cada job mede um
piloto restaurado e exige previsão com margem de 35% para iniciar o horizonte
inteiro. Oito treinos são a matriz, não promessa de conclusão no prazo.

Dois jobs independentes usam as duas T4, com atribuição alternada entre seeds.
Cada worker calibra e treina no mesmo processo; uma barreira isola os perfis
finais depois que ambos terminam treino/geração. GPU concorrente pode afetar
tempos de treino, que ficam separados dos benchmarks isolados.

Somente um `-results.zip` e um `-checkpoints.zip` finais por campanha. Mantém-se
o último checkpoint completo de cada treino principal; estados de calibração
são removidos após validação, conservando seus metadados. Não há ZIP após cada
job. Caches próprios de compilação têm limite de 2 GiB e são removidos ao final;
pesos/fontes/dados antigos não são apagados. Dados preparados e prefixos ocupam
cerca de 51 MiB em cache, sem baixar os arquivos completos de vários gigabytes.

Resultados têm botões embutidos no notebook. O callback `LLM2_BAIXAR(caminho)`
permite recriar um botão. Guardas preservam Torch instalado e bloqueiam outro
treino ativo. Número já ocupado avança automaticamente. Falha/deadline exporta
evidência parcial e checkpoints completos. Reuso automático cobre somente jobs
completos no mesmo código/dados/ambiente/protocolo, com hashes verificados;
retomada automática de jobs incompletos não é oferecida nesta célula.

## Decisão seguinte

Sinal exploratório de qualidade: pelo menos 0,005 NLL menor. Sinal de eficiência:
perda até 0,002 maior com pelo menos 10% menos tempo por update isolado. Cada
comparação será mostrada por seed e contra as duas referências, sem promover
uma seed a confirmação. Se nenhum recurso melhorar, registrar isso e revisar
escrita/esquecimento/composição antes de escalar.

Fractal permanece explícito na pirâmide compartilhada, mas esta rodada não
repete o controle independente já saturado em associação. Se surgir ganho
linguístico da hierarquia, esse controle volta antes de atribuí-lo ao sharing.
Profundidade variável/RINS, compressão adaptativa, GRM, TTT e demais tecnologias
continuam no [plano mestre](PLAN_RECURRENT_FRACTAL.md), sem adoção automática.

Validação CPU está em [VALIDATION.md](../VALIDATION.md). O caminho bloqueado novo
e sua compilação CUDA precisam passar no preflight da T4; resultados da 0031
não aprovam automaticamente a nova execução nem o aprendizado de linguagem.
