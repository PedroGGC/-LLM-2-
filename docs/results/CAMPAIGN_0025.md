# Campanha 0025 — mLSTM qualificado nas duas T4

Auditoria em31/10/2026, horário de Brasília. Sessão registrada às00:40UTC de01/11
(21:40de31/10 em Brasília). Fonte científica congelada:
`1c8ce2abb8f47d3167454b3503598854b7756dad`.

**Decisão: encerrar a etapa A e selecionar a regra mLSTM no shell comum para a
etapa B.** Não repetir a qualificação sem mudança relevante de fonte/ambiente.
Isso é uma escolha operacional, não evidência de qualidade de linguagem ou de
vantagem fractal. Nenhum corpus foi treinado nesta campanha.

## Integridade e contrato

ZIP:75.152bytes,21membros,194.845bytes descompactados, CRC íntegro, sem pickle.
SHA-256:`5d827ac94d387fadb69b144ab0d12b32aea03f0986bcc12c587bad5547e3044b`.
[Auditoria estruturada](campaign_0025_audit.json).

123 hashes de fontes/configurações conferem com os objetos Git da revisão
registrada. Fontes de `platforms.json` e `campaign.json` coincidem; Git limpo.
Seleção recalculada pelo seletor do projeto: `mlstm`. Contrato de qualificação v2,
formas estáticas, supervisor externo. Ambiente: Python3.13.15,
Torch2.11.0+cu128, CUDA12.8, cuDNN91900, duas TeslaT4 de15.636.037.632bytes cada.
Não executar `validate_report` como aprovação CUDA no ambiente CPU local:
o worker voltará a exigir o ambiente exato antes da etapa B.

## Resultados operacionais

T1024/B8, FP32/Inductor, 10warmups +30passos medidos por candidato aprovado.
São regras de memória implementadas num shell comum, não reproduções integrais
de RWKV-7, xLSTM ou Gated DeltaNet.

| Regra | Situação T4 0 / 1 | Mediana ms/step 0 / 1 | Pico VRAM (MiB) | Estado por sequência (KiB) |
|---|---|---:|---:|---:|
| Gated Delta | Timeout / timeout | não disponível | não disponível | não medido no piloto |
| RWKV-7 | Aprovado / aprovado | 364,987 / 364,461 | 3.816,859 | 192,000 |
| mLSTM | Aprovado / aprovado | **93,428 / 89,988** | **746,915** | 195,047 |

O timeout foi aplicado pelo supervisor em600,42s nas duas execuções Gated Delta.
Ele não foi absorvido pelo compilador nem convertido em aprovação. Isso rejeita
esta implementação dentro do orçamento definido, sem concluir inferioridade
matemática, de qualidade ou da arquitetura publicada.

RWKV e mLSTM passaram forward/gradientes contra referência sequencial, causalidade,
streaming e finitude. Erro máximo de forward:1,13e-6 e2,50e-6, respectivamente.
As30 amostras positivas/finitas e suas medianas foram conferidas. Nenhum log contém
o timeout absorvido da0024, traceback, erro de codegen ou aviso de fallback.
O aviso de SMs insuficientes para max-autotune permanece, sem impedir a aprovação.
Testes CPU executados no Kaggle:51passed,6deselected,36,53s.

A média das medianas é91,708ms para mLSTM e364,724ms para RWKV: aproximadamente
3,98vezes a taxa de passos, com80,4% menos pico de VRAM neste microbenchmark.
O estado mLSTM é ligeiramente maior. Não extrapolar para desempenho em corpus,
geração, arquiteturas publicadas ou confronto com Transformer.

## Duração, compilação e armazenamento

- Sessão:3.309,90s, aproximadamente55,17min, contra92,93min na0024.
  Protocolos de compilação diferem: essa redução é operacional, não ganho de modelo.
- Primeiro passo mLSTM:406,87/405,56s; warmup407,74/406,40s. A compilação/arranque
  ainda domina a duração. Esses valores incluem o primeiro passo; não são medição
  isolada do tempo puro de compilação.
- RWKV: primeiro passo472,57/477,20s; subprocessos completos548,00/553,21s.
- mLSTM: subprocessos completos471,00/469,89s, dentro do limite600s.
- Cache máximo amostrado:973.024.602bytes (927,95MiB), abaixo do limite2GiB.
  `storage_budget.exceeded` é null. Amostragem de5s não garante pico instantâneo.
- Foram removidos5.042.199.342bytes (4,696GiB) ao longo dos seis pilotos; os caches
  de cada candidato foram descartados e a pasta final de cache ficou vazia.
  Total acumulado removido não é pico simultâneo de disco.
- Checkout shallow registrado:3.556.024bytes (3,39MiB). O ZIP registra a política
  de remoção após sair do worker, não um recibo posterior que prove a exclusão
  efetiva desse checkout. Não alegar essa verificação pelo ZIP.
- Resultados finais:73,39KiB. Sem checkpoints, treino científico ou teste reservado.

## Continuidade

Etapa B: recorrente simples, hierarquia independente, hierarquia compartilhada e
Transformer2blocos, usando mLSTM como regra dos três braços recorrentes. T1024/B8,
6000passos, LR0,0002/0,0004, avaliações previstas e seeds241/242/243 preservadas.
Recomenda-se começar pelo bloco completo da seed241 e decidir admissão por piloto,
com meta2h/teto3h, sem truncar horizontes.

Antes de lançar o script B, revisar a retenção de cache entre seus workers:
o runner existente limpa por candidato na etapa A e no fim da sessão, mas a etapa B
acumula caches de várias variantes/GPUs durante pilotos/treinos. O limite2GiB
protege contra crescimento, porém pode abortar a campanha se não houver descarte
entre workers inativos. É uma pendência de engenharia, não motivo para repetir
as seis qualificações nem para alterar pesos/objetivos.

Manter o worker científico fixado na revisão qualificada, com proveniência
explícita para qualquer orquestração adicional. Não reescrever os hashes do
relatório nem enfraquecer verificações para aceitar outra fonte. Nenhum checkpoint
foi criado na0025. Novas tecnologias continuam condicionais à análise da etapa B.

Não comparar diretamente estes tempos com Transformer45ms da0023: contexto,
regime e avaliação são diferentes, mesmo quando o total de alvos do batch coincide.
