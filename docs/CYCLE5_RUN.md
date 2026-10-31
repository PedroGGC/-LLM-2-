# Ciclo 5 — profundidades treinadas e ZIPs com retenção limitada

Operação vigente em 31/10/2026, após a [auditoria 0023](results/CAMPAIGN_0023.md).
Plano prévio: [auditoria, retenção e controle](PLAN_CAMPAIGN_0023_RETENTION_CYCLE5.md).
Código em `experiment/fractal-lrm`; o ciclo 4 continua reproduzível na revisão 6a6af1a.

## Primeiro script: limpar somente duplicatas da 0023

Cole `01_limpar_zips_redundantes_0023.py` numa célula. Os hashes das cópias locais
de results e checkpoints já foram calculados e auditados, e vêm preenchidos.
O script baixa fontes por revisão/hash e confere os definitivos remotos. Valida
CRC/SHA, identidade e os 18 estados completos, sem pickle; cada incremental precisa
estar coberto pelos arquivos definitivos. Conteúdo único ou ZIP corrompido bloqueia
antes da exclusão; nomes/links/caminhos externos e workers ativos também bloqueiam.

Remove **somente** `llm2-campaign-0023-after-NN-results.zip` e
`llm2-campaign-0023-after-NN-checkpoints.zip` redundantes. Preserva os dois ZIPs
definitivos, pasta da campanha, checkpoints originais, datasets e checkouts.
Não cria outro backup do working nem exige repetir o download de gigabytes.
Guarda recibo pequeno com espaço liberado. Repetir é seguro se nada mais restar.

## Segundo script: próximo controle

Cole `02_rodar_controle_profundidade.py` noutra célula. Internet e duas T4; revisão
Git completa e hash de bootstrap fixados. Começa pela campanha24 e avança caso
esse número esteja reservado. Preserva o Torch instalado, valida dados e testa
CPU/Linux e os quatro braços CUDA antes dos pilotos. Não depende de variável do
kernel anterior. Relógio novo: meta de 7.200 s, teto de 10.800 s incluindo setup, testes,
compilação, treino, avaliações, perfis e exportação. Piloto deve admitir a matriz
com margem de 35%, reserva de 20 min pós-treino e 5 min exportação, mantendo2 GiB livres.
Os horizontes/schedulers são fixados antes da execução e não mudam para ocupar tempo.

| Braço | K treinado | Passos fixos | Alvos de treino | Parâmetros |
|---|---:|---:|---:|---:|
| Fino K=1 | 1 | 11.372 | 93.159.424 | 2.164.994 |
| Fino K=2 | 2 | 6.000 | 49.152.000 | 2.164.994 |
| Fino K=4 | 4 | 3.084 | 25.264.128 | 2.164.994 |
| Transformer de dois blocos | — | 10.496 | 85.983.232 | 2.164.992 |

Seeds221/222/223, total **12 treinos e 48 avaliações completas principais**.
Orçamento por job: o fino K=2/6000 da 0023, aproximadamente 1,3192×10¹⁵ FLOPs
estimados. Multiplicação-acumulação=2 operações, backward≈2×forward; exclusões
do estimador permanecem registradas. Arredondamento para múltiplo de quatro gera
diferença máxima~0,032%. Não é tempo/energia igualado nem alvos iguais; a diferença
de exposição aos dados é parte da comparação de qualidade por computação.

Mesmos WikiText103 preparados, byte-level, T=256, batch efetivo 32, FP32/Inductor,
dropout 0,05, D=384; FFN=1024 no fino, FFN=384 em dois blocos do Transformer. Fino
preserva projeção/gate próprios e remove as duas escalas comprimidas. Nenhuma
implementação de ACT/DEQ, profundidade variável, memória ou tecnologia nova.

Avaliações completas em 25/50/75/100% do horizonte de cada braço, com scheduler
respectivo. Dois jobs independentes, um por T4, distribuição rotacionada. Pilotos
isolados em ambas GPUs e concorrente. Pós-treino: perfis dos quatro braços nas
duas GPUs, energia se utilizável, diagnósticos da seed221 e probes K=1/2/4/8 dos
três finos dessa seed. Os 12 probes completos adicionais não integram as 48
avaliações principais. Sem teste reservado. Tarefas e métricas sem suporte usam null.

Comparação primária: fino K=2−K=1; K=4 secundária e Transformer como referência.
Ganho exploratório candidato: média≤−0,005NLL e três seeds favoráveis. Reportar
intervalo pareado e limitações; não chamar três seeds de confirmação. Eventual
sinal precisa cinco novas seeds. Se falhar, não concluir que toda recorrência é
inviável: o resultado se aplica a esta atualização de estado/arquitetura/regime.

## Como os arquivos serão controlados

Completou **6 de12 jobs** (treino e validação completos): gera um único par
`-midway-results.zip` / `-midway-checkpoints.zip`, incluindo todos os concluídos.
Isso mede progresso em jobs, não metade do tempo. Não gera ZIP após cada treino.

Ao final gera os dois definitivos. Depois de validar integridade/cobertura dos
definitivos, remove os dois `midway`. **Restam somente dois ZIPs finais** desta
campanha. Em interrupção/falha mantém a recuperação da metade e exporta parcial;
até quatro ZIPs podem ficar nesse caso. Falha de publicação/validação não apaga a
recuperação anterior. Falha apenas de exclusão deixa os temporários remanescentes
e emite aviso. Recibo em `export-retention.json` na pasta da campanha.

Os checkpoints por etapa, logs e dados de trabalho continuam na pasta científica;
esta política limita cópias ZIP, preservando os originais. Não sobrescrever
execuções/arquivos já publicados. Baixar e guardar results e checkpoints definitivos.

Roteiro após esse controle: profundidade variável/RINS se houver justificativa;
compressão adaptativa com tarefa causal/recuperação; Gated DeltaNet como primeiro
baseline sequencial; Memory Caching/GRM com orçamento; TTT/fast weights; KV cache
e depois FractalLLM. Quantização/destilação complementam depois. Decisões completas
e demais hipóteses no [roadmap](RESEARCH_ROADMAP.md), sem promoção automática.
