# Campanha 0004 — Estudo de microbatch e acumulação (8×4, 16×2, 32×1), 27/10/2026

## Plano

- Permitir overrides de `batch_size` e `grad_accum_steps` na configuração da campanha (`campaign_config`), nos argumentos da CLI (`--batch-size`, `--grad-accum-steps`) e no dicionário `LLM2_SETTINGS` do bootstrap do Kaggle.
- Implementar modo de estudo e perfil de lotes (`mode: "perfilar_lotes"` no notebook e `--profile-batches` na CLI), testando a matriz `8×4`, `16×2` e `32×1` para os modelos selecionados (`baseline_small`, `flrm_compressed_grouped_no_jepa_k1` e `flrm_compressed_grouped_no_jepa_k2`).
- Garantir a invariante de lote efetivo constante: `batch_size * grad_accum_steps == 32` sequências (8.192 alvos supervisionados por optimizer-step com `seq_len=256`).
- No perfil isolado: 10 passos de aquecimento (warmup) + 30 passos medidos sem profiler + 1 passo instrumentado com trace, em GPU física isolada (GPU 0), sem concorrência de processo na GPU 1.
- Adaptar a chave de deduplicação de perfis em `profile_runs` para `(modelo, batch_size, grad_accum_steps)`, permitindo perfilar múltiplos formatos do mesmo modelo sem descarte indevido.
- Exibir tabela comparativa em português no console com colunas: Modelo, Formato (microbatch × acumulação), Throughput (alvos/s), Tempo por passo (ms) e VRAM alocada (MiB).
- Suportar também `mode: "treinar"` com `batch_size` e `grad_accum_steps` parametrizados, caso se deseje executar o treino completo de 2.000 passos na melhor configuração identificada.
- Exportar artefatos em `llm2-campaign-0004-results.zip` sem sobrescrever campanhas anteriores.

## Riscos e limites

- Amostragem de dados: o `RandomBatchStream` usa gerador CPU determinístico. Confirmou-se que 4 sorteios de 8, 2 sorteios de 16 e 1 sorteio de 32 a partir da mesma semente produzem a sequência idêntica de janelas por atualização.
- Memória: microbatch 32 dobra a VRAM necessária para ativações frente a 16 e quadruplica frente a 8. Como o modelo aloca <500 MiB em placa de 15 GiB, o risco de OOM é baixo, mas qualquer estouro deve ser capturado como falha explícita.
- Ponderação de gradientes: o Trainer já utiliza ponderação proporcional de alvos válidos para a perda LM (`accumulation_weighting="tokens"`), que é matematicamente equivalente entre microbatch único e microbatches acumulados.
- Perfis rápidos de 30 passos não medem convergência nem loss final; seu papel é selecionar o formato de melhor taxa de transferência e saturação de hardware antes de investir tempo em treinos completos.
- Localmente há apenas CPU. Testes de integração locais validam contratos de interface e finitude; o benchmark empírico de throughput e memória é restrito à execução nas T4 no Kaggle.

## Aceitação

- Overrides funcionais em `campaign_config`, CLI de campanha e `kaggle_notebook.py`.
- Validação estrita de argumentos: inteiros positivos e conservação do lote efetivo em 32 sequências.
- Perfis de múltiplos lotes executam sequencialmente e registram métricas distintas.
- Testes locais aprovados (`python -m pytest -q`), cobrindo amostragem, acumulação e integração de perfis.
- Commit publicado na branch `experiment/fractal-lrm` e célula Kaggle entregue com `run_number: 4`.
