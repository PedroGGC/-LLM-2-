# Campanha 0015 — referência corrigida

Atualizado em 28/10/2026 após recálculo independente. Este documento substitui, para
interpretação atual, as afirmações conflitantes do relatório recebido e da seção 0015
do [histórico de validação](../archive/VALIDATION_HISTORY.md). Artefatos originais preservados.

## Proveniência e escopo

- ZIP `llm2-campaign-0015-results.zip`, 10.757.180 bytes, 664 entradas.
- SHA-256: `8afb1969ea131ad3f724f642642c45a1d01a60eaad50c252f5a21bf18b83b7e8`.
- Commit executado: `aa51b7d9dc541b00b84f587075f757078ecca262`.
- Python 3.13.15, PyTorch 2.11.0+cu128, CUDA 12.8; duas T4, FP32/Inductor.
- TinyShakespeare byte-level, vocab=257, contexto=256, batch=32, acumulação=1.
- Seeds 42/43/44, 2.000 passos, 16.384.000 alvos com repetição por execução.
- Avaliação de 437 janelas completas/111.872 alvos nos passos 500/1000/1500/2000.

CRC íntegro; 18 snapshots e 702 hashes de arquivos conferidos. Os 39 arquivos de fonte
também correspondem ao commit Git. As 72 avaliações foram reconstruídas de NLL por
janela, com erro máximo de 1,9153e−8. Manifesto registra 995.268 tokens de treino e
111.932 de validação; 59 alvos de cauda não formam nova janela de avaliação.

Hashes registrados dos parâmetros comuns são iguais entre os cinco modelos recorrentes
por seed; exclusivos são iguais entre os três FLRM. Os pesos não estão no ZIP, portanto
esta verificação é de consistência dos registros, não rehash dos tensores originais.
O Transformer não participa do pareamento de tensores LRM/FLRM.

Dados pequenos versionáveis: [execuções individuais](campaign_0015_runs.csv) e
[proveniência](campaign_0015_provenance.json). ZIP e auditoria completa ficam localmente
em `artifacts/campaigns/` e `artifacts/audits/auditoria-2026-10-28/reavaliacao-0015/`
no checkout comum, conforme a [política de arquivos](../REPOSITORY_POLICY.md).

## Resultados

Média ± desvio padrão amostral entre três seeds. Tempo soma etapas de treino e pode
incluir compilação inicial; não inclui toda a preparação/avaliação da campanha.

| Modelo | NLL completa ↓ | Treino, s ↓ | VRAM alocada média, MiB ↓ | Geração média, tok/s ↑ |
|---|---:|---:|---:|---:|
| Transformer pequeno | **1,394923 ± 0,003351** | **96,24 ± 3,71** | 418,90 | 468,54 |
| N1: FLRM nativo + cache K=2 | 1,399268 ± 0,001725 | 220,93 ± 16,70 | 919,93 | 74,71 |
| G0: FLRM agrupado K=2 | 1,401717 ± 0,000666 | 261,61 ± 14,83 | 1.011,42 | 89,87 |
| LRM K=2 | 1,404358 ± 0,002483 | 165,34 ± 2,13 | 648,41 | 256,74 |
| FLRM agrupado K=1 | 1,409668 ± 0,002820 | 126,59 ± 10,33 | 591,66 | 160,97 |
| LRM K=1 | 1,411963 ± 0,004404 | 84,40 ± 4,75 | 413,75 | 418,60 |

Geração: eager, batch 1, 64 novos tokens, sem KV cache. Não é inferência compilada.
N1 reduziu o tempo médio de treino em 15,55% e NLL em 0,002449 frente a G0, mas perdeu
16,87% de throughput de geração. Contra Transformer, N1 teve +0,004346 de NLL, custo
de treino 2,30× e VRAM alocada 2,20×. Transformer venceu as três seeds.

FLRM tem 2.168.076 parâmetros; Transformer, 2.164.224; LRM, 2.017.536. Contagem única
dos embeddings compartilhados. Estimativas por passo: 282,30 GFLOPs N1, 328,28 G0,
115,98 Transformer. Não são FLOPs medidos: FMA=2, backward≈2×forward, exclusões para
normas, RoPE, softmax, elementwise, loss, otimizador e parte das operações de escalas.

| Contraste A−B de NLL | Média | Seeds favoráveis a A |
|---|---:|---:|
| LRM K=2 − LRM K=1 | −0,007605 | 3/3 |
| FLRM G0 K=2 − FLRM agrupado K=1 | −0,007951 | 3/3 |
| FLRM agrupado K=1 − LRM K=1 | −0,002296 | 3/3 |
| FLRM G0 − LRM K=2 | −0,002641 | 2/3 |
| FLRM N1 − LRM K=2 | −0,005090 | 3/3 |

Recorrência ajudou com mais computação após os mesmos alvos. Multiescala teve sinal
favorável, sem isolar FiLM/mistura/compressão nem todos os efeitos de parâmetros/cache.
Três seeds de um corpus não demonstram generalidade. Intervalo t exploratório de 95%
para N1−G0 é [−0,007887; +0,002989], e para N1−Transformer, [−0,006323; +0,015015].
São intervalos frágeis com dois graus de liberdade, sem correção por seleção de modelos.

## Errata técnica

1. **Gate não congelado:** N1/seed 42 começa uniforme em 0,1192029; termina entre
   0,1142630 e 0,1219159, média 0,1190752, gradiente L2 0,0196727 no diagnóstico final.
   Média estável não elimina aprendizado por canal; invariância à escala escalar da
   RMSNorm não significa invariância a gates vetoriais.
2. **Resoluções corretas:** 256/64/16, não 256/64/32; padding agrupado é 16→64.
3. **Dispersões corretas:** N1 é 220,93±16,70 s, não ±0,97; G0 é 261,61±14,83 s,
   não ±1,80. Throughput N1: 74.430±5.400 alvos/s, não ±326. Algumas linhas individuais
   do relatório divergem; o CSV vinculado contém valores diretamente do ZIP.
4. **Hashes exclusivos transcritos errados:** corretos por seed abaixo. Hashes comuns
   conferem; o arquivo exportado se chama `campaign.json`.
5. **Causa do déficit não isolada:** LRM K=1 e baseline diferem em FFN, inicialização,
   normalizações, gate e caminho residual. Não atribuir integralmente +0,017041 de NLL
   a `state_0=0` ou à projeção de contexto.
6. **RMS é diagnóstico:** razão estado normalizado/contexto de N1 após treino é cerca
   de 25,62×/37,43×/45,37×. A norma interna da atenção retorna RMS próximo de 0,996.
   Isso motiva hipótese de balanceamento, não prova apagamento de tokens ou falha de gradiente.
7. **Dropout e execução:** diferenças de padding, sorteios e arredondamento permitem
   trajetórias distintas. Não provam que eliminar sorteios em padding causou melhora de loss.
8. **Sem dominância em todos os critérios:** N1 gera mais devagar que G0 no benchmark disponível.

| Seed | SHA-256 registrado dos parâmetros exclusivos FLRM |
|---|---|
| 42 | `2e1968703ce48ae9d54a9ebc354dfdbace1a9006a059b57ac599e0695b360fbd` |
| 43 | `ddc063f062c2d736a1ee17e3b9ea9799001fac638b5ffba9c344cbf3e31cca73` |
| 44 | `579a97a0d10d3b667930dc3ec2817f9cf0fb5727b7844bc0187d52a4408f9478` |

## Histórico e limites que afetam a decisão

- 0001/0002: JEPA ganhou cerca de 0,000387 de NLL em seed 42, custando aproximadamente
  41,7% mais tempo de treino. A 0002 repete a seed; não justifica exclusão universal do JEPA.
- 0005/0006: Ciclo-V economizou trabalho, mas perdeu qualidade. 0007: ciclo invertido
  teve forte regressão. Não promover assimetria como otimização equivalente.
- 0009+0010: NCM melhorou aproximadamente 0,000184 de NLL nas três seeds, com pequeno
  custo de throughput; 0011 é repetição das seeds 43/44. Protótipos não são memória persistente.
- 0012/0013: Inductor e execução nativa/cache melhoraram desempenho. Telemetria de cache
  teve bug corrigido antes da 0015. Quantidade de kernels isolada não explica velocidade.

Todas as execuções tiveram melhor avaliação no passo final, sem demonstrar convergência.
N1 treinou sistematicamente na GPU 0; G0/baseline na 1, com concorrência no host.
Novos ensaios de custo devem alternar a atribuição ou executar isoladamente.
Validação muito reutilizada, split por parágrafos e diagnóstico em quatro janelas
limitam as conclusões. Faltam outro corpus/tarefa e comparação com orçamento de tempo/FLOPs igualado.

Logs recebidos: CPU 180 passed/21 deselected; CUDA FP32 17 passed/184 deselected.
São verificações históricas no Kaggle, não testes locais novos. Não validam BF16/A100.
O ZIP não contém pesos/estado para retomar treinamento.
