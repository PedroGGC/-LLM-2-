# Campanha 0019 — dois blocos completos, 29/10/2026

ZIP recebido: 14.002.802 bytes, 442 entradas sem duplicatas, CRC íntegro; SHA-256
`e609269cf31f8d55d6afbbda9ef56b9b4c799fbd28f7921fd1f16479aa14887f`.
Revisão científica `8798f6b84714d2c6671d970685984270078a982c`. WikiText-103 raw,
byte-level, T256/B32, FP32/Inductor. Teste reservado não foi consultado.

## O que terminou

**12 treinos completos, 48 avaliações completas, seeds 101/102.** Restam 103/104/105.
O encerramento registrado é `partial_budget`, aos 9.645,32 s (2h40min45s): com cerca
de 74 min de trabalho disponíveis, a previsão conservadora recusou iniciar o próximo
bloco inteiro de seis modelos. O arquivo não registra falha de rede como causa da
parada. Uma interrupção na conexão do usuário pode ter ocorrido, mas não está
demonstrada nos logs do executor. Os dois blocos concluídos estão preservados.

| Fase | Minutos |
|---|---:|
| Treinamento dos 12 modelos | 137,37 |
| Testes de código CPU | 7,01 |
| Testes CUDA FP32 | 4,04 |
| Perfis de treinamento | 3,63 |
| Pilotos de validação | 1,02 |
| Avaliação completa, geração e diagnóstico | 6,48 |
| Dados/configurações | 0,29 |

Os treinos foram seriais. O custo dominante foi a quantidade de treinos e seus
horizontes, não a avaliação. Remover avaliação completa economizaria pouco e perderia
informação importante. Somente reduzir testes de software também não resolveria o total.

## Qualidade observada

Médias de NLL nas duas seeds; menor é melhor. Horizonte final pré-fixado, sem seleção
do melhor checkpoint. Os FLOPs continuam estimativas, com exclusões já documentadas.

| Braço | Passos por seed | NLL média | Parâmetros |
|---|---:|---:|---:|
| Transformer, alvos iguais | 8.000 | 1,349322 | 2.164.224 |
| LRM, alvos iguais | 8.000 | 1,310299 | 2.017.536 |
| FLRM, âncora dos dois ensaios | 8.000 | 1,308835 | 2.165.772 |
| Transformer, FLOPs estimados igualados | 19.472 | 1,311478 | 2.164.224 |
| LRM, FLOPs estimados igualados | 10.620 | 1,278126 | 2.017.536 |
| Transformer de dois blocos, FLOPs estimados igualados | 11.396 | 1,196534 | 3.639.552 |

O FLRM melhora apenas 0,001464 frente ao LRM com alvos iguais. Frente ao Transformer
pequeno com FLOPs estimados iguais, melhora 0,002643; IC t pareado exploratório de
95% [−0,024148; +0,018862] para FLRM menos Transformer. Ainda inconclusivo e abaixo
do ganho operacional de 0,005. LRM e Transformer de dois blocos vencem o FLRM nas
duas seeds com o orçamento estimado igualado: diferenças médias +0,030709 e +0,112301.
O controle de dois blocos tem mais pesos; esse ensaio não iguala parâmetros.

Há informação útil para despriorizar expansão de complexidade e conferir mais uma
seed antes de decidir quanto investir na confirmação. Não há vantagem fractal
confirmada, equivalência demonstrada nem licença para escolher pelo teste reservado.

## Auditoria e próxima execução

Conferidos 197 hashes do bootstrap contra Git; 20 snapshots com 1.880 hashes;
configurações, contadores e fórmulas/enumeracão de parâmetros; hashes de pesos entre
registros; reconstrução das 48 NLLs a partir de 4.469 janelas / 1.144.064 alvos cada.
Maior erro absoluto: 4,37404424e−9. O ZIP não contém pesos: não foi possível rehashar
os tensores ausentes. Logs reais: **280 CPU e 25 CUDA FP32 aprovados** no Kaggle.

Evidência curada para reaproveitamento sem repetição:
[campaign_0019_ledger.json](campaign_0019_ledger.json). Auditoria independente local:
`artifacts/audits/campaign-0019-2026-10-29/audit.py` no checkout experimental.

Por solicitação do usuário, próximas sessões têm **teto total de uma hora**, até
dois treinos novos, testes focados e snapshots após cada conclusão. A próxima é o
par primário na seed 103, preservando o código científico original e todas as quatro
avaliações completas. [Operação dos lotes curtos](../CYCLE3_SHORT_RUN.md).
