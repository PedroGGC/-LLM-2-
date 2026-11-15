# Plano Técnico da Etapa 3 — BPE (`vocab=4096`), *Prelude–Core–Coda* (Huginn), *KDA*, *MTP* e *Muon*

Data: 2026-11-15  
Branch base: `llm-ideas` (pós-Campanha 009 `2baae7c`)  
GPU alvo: Google Colab `NVIDIA L4` (`24 GB`, `BF16`)

---

## 1. Motivação (Baseada na Campanha 009)

A Campanha 009 fechou os gargalos A e B no domínio de bytes (`877k` parâmetros):
- `Binding @8192 (w=32)`: `99.7% ± 0.3%` (`99.5% ± 0.4%` pareado).
- `Binding HARD @8192 (w=64)`: `95.1% ± 1.2%` (`90.9% ± 1.4%` pareado).
- Banda `2049+`: `0.7221 ± 0.0134`, com `ΔTrunc = +0.0110` e `ΔSwap = +0.0120`.

Porém expôs dois limites práticos de continuar apenas em bytes `<1M`:
1. **Janela semântica curta:** `512` bytes equivalem a apenas ~100 palavras. Com BPE (`vocab=4096`), a mesma janela cobre `~4×` mais texto real (`~2.000` bytes).
2. **Competição de capacidade em bloco único compartilhado:** em `877k` params, aumentar `assoc_val_dim` (`24→48`) + deslocar passos finais para `1024/2048` com `writes=48` subiu `val@512` (`0.7646 → 0.9122`). Separar papéis com **Prelude → Core Recorrente → Coda** (Geiping et al. 2025) e ampliar o modelo para a faixa `7M–15M` elimina a sobrecarga sobre as camadas recorrentes.

---

## 2. Escopo e Subetapas

### Subetapa 3A — Pipeline BPE, Métrica `BPB` Exata, Associativo Multi-Lag e Fix de `--smoke`
1. **`src/llm2/research/stories_data.py` (`prepare_bpe`, `StoriesBPEData`):**
   - Reutiliza exatamente `select(train_raw, valid_raw)` dos prefixos pinados do TinyStories (`train`, `cal_val`, `val`, `test` deduplicados e isolados por documento).
   - Treina `TextTokenizer.train(train_docs, vocab_size=4096)` **exclusivamente sobre `train`** (proibido tocar `cal_val`/`val`/`test`, conforme `AGENTS.md`).
   - Armazena `<split>.bin` (`<u4`) + `raw_utf8_bytes` exatos por documento e split (`schema="stories-doc-bpe-v1"`).
   - Avaliação em `stories_worker.py` e `ia2_benchmark.py` calcula `bits_per_byte (BPB) = sum_nll_nats / (eval_raw_bytes * ln 2)`, diretamente comparável entre `byte` (`vocab=257`) e `BPE` (`vocab=4096`).
2. **Canal Associativo Multi-Lag (`fractal_assoc_multilag`):**
   - Em `FractalDeltaMemory._run_assoc`, permite misturar `shifted` (`t-1`) e `conv_out` (`[t-2, t-1, t]` causal) por meio de gate escalar aprendido de 2 pesos quando `fractal_assoc_multilag=True`, preparando o endereçamento para entidades BPE multi-token sem aumentar o estado por sequência.
3. **Fix de `--smoke` em `ia2_benchmark.py`:**
   - Passar `max_batches=2 if smoke else None` para `score_stories`, evitando varrer toda a validação em testes rápidos.

### Subetapa 3B — *Prelude → Core → Coda*, *KDA*, *MTP* e *Muon*
1. **Topologia *Prelude → Core Recorrente → Coda* (`fractal_prelude_coda`):**
   - Em `FractalDeltaMemory` (`depth >= 3`, `share_scales=True`), instancia **3** células únicas (`Prelude=cells[0]`, `Core=cells[1]` iterado `depth-2` vezes com `FractalLevelAdapter` por nível, `Coda=cells[2]`).
   - Preserva `state_bytes_per_sequence()`, causalidade, `parameter_formula()` e `FractalDecode.step()`.
2. ***Kimi Delta Attention (KDA)* (`fractal_kda`):**
   - Projeção low-rank em `FractalDeltaCell` produzindo decaimento diagonal por canal (`log_w` com `w_dim == head_dim`), usando diretamente o ramo já existente em `state_kernels.py` (`dplr_chunked` / `dplr_sequential` em FP32).
3. ***MTP* (`mtp_tokens`, `mtp_weight` em `fractal_delta`):**
   - Libera `mtp_tokens`/`mtp_weight` em `config.py` para `fractal_delta` (`supports_hidden_only=True`), reutilizando integralmente `PredictiveObjective` e `accounting.py` já existentes.
4. **Otimizador *Muon* (`src/llm2/optim.py`, `OptimizerConfig.kind = "adamw" | "muon"`):**
   - `MuonAdamW`: aplica Nesterov + 5 iterações Newton-Schulz sobre pesos 2D ocultos (`ndim == 2`, excluindo `embedding`/`lm_head` compartilhados) e AdamW para embeddings e parâmetros 1D, com `state_dict`/`load_state_dict` exatos compatíveis com `Trainer` e `LambdaLR`.

### Subetapa 3C — Ablações e Benchmark Pareado na GPU L4 (`BF16`)
Escala alvo `~7M` parâmetros no `fractal_pcc_bpe` (`d=384, depth=8 [1 Prelude + 6 Core + 1 Coda], heads=6, hidden=1024, vocab=4096`) vs baseline denso `transformer_bpe_med` (`~14M` params, mesma largura/profundidade `8L`):
1. **Smoke CPU + L4 (`--smoke`):** validar todos os novos braços, invariantes `parameter_formula == count_parameters`, causalidade, `chunked == sequential`, `decode parity` e `exact resume`.
2. **Ablação controlada na L4 (`seed=351`, `bpe vocab=4096`, `BF16`, `--mode unified`):**
   - `transformer_bpe_med` (baseline 8L Transformer denso)
   - `fractal_pcc_bpe` (Prelude–Core–Coda, AdamW, `w_dim=1`)
   - `fractal_pcc_kda_bpe` (+ KDA `w_dim=head_dim`, `chunk_size=32`)
   - `fractal_pcc_mtp_bpe` (+ MTP `2 tokens, weight=0.25`)
   - `fractal_pcc_muon_bpe` (+ Muon nas matrizes 2D)
3. **Confirmação multi-seed (`351, 352, 353`)** da melhor combinação vs `transformer_bpe_med`.

---

## 3. Riscos e Mitigações

| Risco | Mitigação |
|---|---|
| **Consumo de VRAM / lentidão do KDA 5D (`B, H, C, cs, cs, Dk`) em `chunk_size=64`:** | Usar `chunk_size=32` para braços KDA (`4×` menos memória intermediária) e medir throughput + VRAM na L4 antes da multi-seed. |
| **Desalinhamento do Muon com `grad_clip` / `LambdaLR` / `resume`:** | Implementar `MuonAdamW` como subclasse padrão de `torch.optim.Optimizer`, testada com `test_exact_resume_next_step_and_full_state`. |
| **Degradação de `BPB` pela mistura de binding (lição da C009):** | Usar mistura equilibrada (`UNIFIED_WRITES` até `w=32`, sem empurrar 50% do orçamento para `2048+w=48` sem calibração prévia) e acompanhar `val BPB` passo a passo. |
| **Vazamento de validação no BPE:** | Treinar BPE apenas sobre `train` após deduplicação global por hash e verificar checksum + ausência de sobreposição no `StoriesBPEData.__init__`. |

---

## 4. Critérios de Aceitação da Etapa 3

1. **Testes automáticos (`0` falhas):**
   - Novos testes unitários cobrindo BPE sem vazamento + `BPB`, `Prelude–Core–Coda`, `KDA` (`chunked == sequential`), `MTP` em `fractal_delta`, `MuonAdamW` exact resume e multi-lag associativo.
   - `pytest -q` (CPU) e `pytest -q -m cuda` (Colab L4) verdes após a implementação.
2. **Eficiência e qualidade em BPE (`vocab=4096`) na L4:**
   - Braço vencedor `fractal_pcc_*_bpe` (`~7M` params) empata ou supera o `transformer_bpe_med` (`~14M` params, ~2× mais parâmetros) em `BPB @3219` e nas bandas longas (`513-2048`, `2049+`), mantendo decode `O(1)` e `Binding @8192 (w=32) ≥ 95%`.
3. **Documentação:** `VALIDATION.md` e relatório `docs/results/IA2_CAMPAIGN_010_COLAB_L4_BPE.md` atualizados com todos os resultados reais e limitações antes de encerrar o runtime L4.
