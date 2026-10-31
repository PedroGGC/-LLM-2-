# Campanha 0024 — seleção operacional com falha na guarda de duração

31/10/2026. Fonte científica `ffc6fb832c756771f92bb3212294813e7109e10f`.
ZIP entregue: 87.636 bytes (85,58 KiB); CRC íntegro, 22 membros, 349.690 bytes
descompactados. SHA-256:
`04266ec69663e89eafe736b4ef06f7212ccdf3618c5d911cba7fde637c0ac201`.
[Auditoria estruturada](campaign_0024_audit.json).
Os120 hashes declarados de fontes/configurações conferem com os objetos Git da
revisão registrada; nenhum arquivo incompatível foi encontrado nessa conferência.

## O que foi medido

Etapa A, três regras de memória no shell comum, duas Tesla T4. T1024/B8,
FP32/Inductor, 10 passos de warmup + 30 de medição por piloto. Sem treino
científico em corpus, sem seeds de confirmação, sem NLL e sem checkpoints.
Não são reproduções integrais dos modelos publicados.

| Regra | Passo T4 0 (ms) | Passo T4 1 (ms) | Primeiro passo 0/1 (s) | Pico VRAM (MiB) | Estado/seq (bytes) |
|---|---:|---:|---:|---:|---:|
| Gated Delta | 271,855 | 268,974 | 964,8 / 968,1 | 2.987,4 | 196.608 |
| RWKV-7 | 373,852 | 371,781 | 834,4 / 835,6 | 3.816,9 | 196.608 |
| mLSTM | 94,667 | 96,484 | 707,2 / 753,4 | 746,9 | 199.728 |

Tempo de sessão registrado: 5.575,63 s (92,93 minutos). Testes CPU na T4:
51 passed, 6 deselected; checagens sequencial/chunks, gradientes, causalidade e
streaming foram declaradas aprovadas nos seis registros, erro máximo de forward
entre 9,54e-7 e 2,24e-6. Ambiente: Python3.13.15, Torch2.11.0+cu128, CUDA12.8,
cuDNN91900. Energia não disponível.

## Falha encontrada na auditoria

Todos os seis logs contêm `candidate exceeded 600 seconds` durante a compilação
do backward. A exceção do SIGALRM foi absorvida pelo caminho de compilação AOT;
o processo continuou e publicou `status: passed`. Todos os primeiros passos
excedem600s. Logo, o limite planejado não foi respeitado.

Há recompilação ao passar da checagem B1/T65 para B8/T1024 e generalização
dinâmica. O log não permite afirmar se a versão final usou fallback eager ou
recompilou com sucesso: preservar essa distinção. A documentação de
[`torch.compile`](https://docs.pytorch.org/docs/2.11/generated/torch.compile.html)
explica a generalização automática com `dynamic=None` e a especialização com
`dynamic=False`.

**Decisão:** mLSTM é o candidato operacional mais promissor pelos tempos medidos,
mas a seleção automática da0024 não será aceita como qualificação limpa. Repetir
a etapa A com timeout externo, formas estáticas e registro v2. Não iniciar os
12 treinos nem atribuir vantagem de linguagem/fractal a esses números.

## Armazenamento

O ZIP tem85,58KiB e não contém pesos. Os aproximadamente6GB relatados no working
não podem ser atribuídos ao ZIP. Os logs confirmam arquivos gerados dentro de
`llm2-campaign-0024/compiler-cache/gpu-*`; também havia checkout Git completo.
A divisão exata de bytes não foi inventariada: `disk_footprint_bytes: null`.

A nova limpeza mede antes/depois e remove somente caches reconhecidos e checkouts
limpos. Preserva campanha, relatórios, ZIPs, dados e pesos. Novas sessões registram
cache, aplicam limite monitorado de2GiB (amostragem5s), reserva livre de2GiB,
limpam após candidatos e ao finalizar, buscam somente a revisão Git necessária
e descartam o checkout limpo. Cache pode ultrapassar momentaneamente o limiar
entre amostras; o limiar provoca encerramento/exportação, não exclusão de treino.

O download padrão passa a ser um botão embutido no notebook, já confirmado pelo
usuário nesta campanha. Arquivos grandes usam partes de até24MiB por solicitação,
sem criar cópias das partes no disco remoto. SHA-256 permite conferir a reunião.
