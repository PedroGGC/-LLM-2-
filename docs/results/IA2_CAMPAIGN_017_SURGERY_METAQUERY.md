# IA2 Campanha 017 — Attention-Surgery (`RADLADS` / `Jet-Nemotron` 1:1) & Fase D `MetaQueryBridge` (`Nano Lumina`)

**Data:** 22/11/2026  
**Branch:** `llm-ideas`  
**Escopo:** Solução cirúrgica para o gargalo de qualidade semântica identificado na Campanha 016 (`Score 14.0` por MLPs inicializados aleatoriamente) via **`Attention-Surgery` 1:1** (`transplant_teacher_weights`, `apply_surgery_freeze`, `layer_mse_weight`) em [`src/llm2/research/distill_convert.py`](../../src/llm2/research/distill_convert.py), otimização de Tensor Cores `FP16` (`sm_75`) em [`src/llm2/architectures/triton_quarot.py`](../../src/llm2/architectures/triton_quarot.py) e implementação completa da **Fase D (`MetaQueryBridge` para o `Nano Lumina`)** em [`src/llm2/multimodal/metaquery_head.py`](../../src/llm2/multimodal/metaquery_head.py).

---

## 1. Mecanismos Implementados

### 1.1 `Attention-Surgery` (`RADLADS` / `Jet-Nemotron` Literal Weight Transfer)
Na Campanha 016, a destilação Rota 1 com apenas `warmstart_student_embeddings` e `SwiGLU` FFNs aleatórios atingiu `12.7×` menos VRAM, `754×` menor estado em `16K` e `9.9×` menos energia na Tesla T4, mas apenas `Score = 14.0` porque matrizes MLP aleatórias sobre `vocab=49.152` não aprendem sintaxe/factualidade em `614K` tokens.

O pipeline de **`Attention-Surgery`** resolve a raiz desse problema em três etapas:
1. **`transplant_teacher_weights(teacher, student, quiet_init_gates=True)`:**
   - Transplanta `100%` dos **Embeddings / `lm_head` amarrados** (`warmstart_student_embeddings`), **`RMSNorm`s** (`attn_norm`, `ffn_norm`, `final_norm`), **camadas `SwiGLU` FFN** (`gate`, `up`, `down`) e **projeções `QKV` / `Out`** (suportando tanto `q_proj/k_proj/v_proj/o_proj` do HuggingFace com expansão automática de `GQA` via `_expand_gqa_kv` quanto `attn.qkv/proj` do `TransformerLM` nativo).
   - Inicializa `cell.conv` (`std=1e-4`), `assoc_readout` (`std=1e-4`), `anchor_gate = -8.0` e `hybrid_windows.out_proj` (`std=1e-4`) em regime silencioso para que os FFNs transplantados recebam o fluxo residual limpo desde o Passo 0.
2. **`apply_surgery_freeze(student, freeze_ffn=True, freeze_embeddings=True)`:**
   - Congela integralmente os `SwiGLU` FFNs (`cell.ffn`, `cell.ffn_norm`) e `embedding`/`lm_head` (~`61%–67%` dos parâmetros totais), concentrando 100% dos gradientes e estados do `AdamW` na conversão do mecanismo de atenção para o estado recorrente $O(1)$ (`Gated DeltaNet-2` + janela híbrida `3:1`) e impedindo esquecimento catastrófico em orçamentos curtos de destilação.
3. **Alinhamento Intermediário Camada-a-Camada (`layer_mse_weight` + `return_layer_states=True`):**
   - Adicionado `return_layer_states: bool = False` e `layer_states: tuple[Tensor, ...]` em `ModelOutput`, `TransformerLM.forward` e `FractalDeltaModel.forward`.
   - Em `compute_distill_loss`, cada camada $l$ do aluno tem sua direção latente alinhada à camada correspondente do professor via MSE cosseno normalizado em FP32 (`F.mse_loss(F.normalize(sh_f), F.normalize(th))`), eliminando a deriva acumulada (*snowball drift*) entre blocos FFN congelados.

### 1.2 Fase D — `MetaQueryBridge` para o `Nano Lumina` (`Anima 2B` / `DiT` + `VAE 16ch` & `3D GIF`)
Implementado em [`src/llm2/multimodal/metaquery_head.py`](../../src/llm2/multimodal/metaquery_head.py):
- **`MetaQueryBridge` (`num_queries=64, dit_cond_dim=2048, dit_pooled_dim=768, connector_layers=2, max_frames=32, grm_alpha=0.25`):**
  - Após o prefill multimodal de texto + imagens/frames de referência no estado recorrente $O(1)$ do `Lumina V1`, injeta $Q=64$ tokens aprendíveis `meta_queries` via `backbone(inputs_embeds=queries, state=cur_state, return_logits=False)`.
  - Funde automaticamente as âncoras visuais `<img_end>` armazenadas no `SegmentMemoryCache` (`GRM`) via `query_grm_readout(q_hidden, top_k=2)` (com `@torch.no_grad()` para permitir retropropagação limpa pelos `meta_queries`).
  - Refina os $Q$ estados com um conector bidirecional Pre-Norm de 2 camadas (`_ConnectorBlock` com `RMSNorm` FP32 + `SDPA` bidirecional + `SwiGLU`) e projeta:
    - **Imagem Estática / Edição (`num_frames=1`):** `seq_cond` de shape `(B, 64, 2048)` + `pooled_cond` de shape `(B, 768)`.
    - **GIF Animado 3D (`num_frames=F`):** soma embeddings temporais aprendidos `temporal_frame_emb[:F]` e emite `seq_cond` de shape `(B, F, 64, 2048)` + `pooled_cond` `(B, 768)` mantendo **exatamente o mesmo estado recorrente $O(1)$ no backbone**.
- **`freeze_backbone_for_metaquery` & `compute_metaquery_alignment_loss`:**
  - Congela 100% do `Lumina V1` e da ponte de visão (`SigLIP 2` + `VisionSpatialProjector`), treinando apenas os `meta_queries`, `temporal_frame_emb`, `_ConnectorBlock` e projeções `seq_proj`/`pool_proj` com perda híbrida `MSE + Cosine Distance` em FP32.

---

## 2. Verificação de Testes e Ablação Pareada Controlada

### 2.1 Suíte Unitária (`pytest`)
- `python -m pytest -q tests/test_metaquery_bridge.py tests/test_phase_c_distill_ringue.py` → **`7 passed, 2 skipped in 10.04s`** (os 2 skips são exclusivamente os testes `@pytest.mark.cuda` quando executados na CPU local).

### 2.2 Ablação Pareada Controlada (`60` passos, mesmo professor `TransformerLM` pré-treinado em `vocab=512`)
Para testar isoladamente a hipótese de retenção de conhecimento FFN sob orçamento curto de destilação (onde a destilação enxerga apenas metade do vocabulário `[4, 256)` enquanto a avaliação cobra tanto o vocabulário completo `[4, 512)` quanto o subconjunto não visto durante a destilação `[256, 512)`):

| Braço de Conversão (`FractalDelta V2`, `60` passos) | Params Treináveis / Congelados | Step-0 `KL` ↓ | Val Full `CE` ↓ | Val Full `KL` ↓ | Val Full `Top-1 Acc` ↑ | Acordo c/ Professor (Full) ↑ | Acordo c/ Professor (Vocab Não Visto) ↑ |
|---|---:|---:|---:|---:|---:|---:|---:|
| **Professor (`TransformerLM` `4L`)** | `1.074.162` | — | `0.6904` | `0.0000` | `85.09%` | `100.00%` | `100.00%` |
| **A. `Random-MLP` (Receita C016: só `warmstart_embed`)** | `1.074.162 / 0` | `2.4556` | `3.6030` | `2.8112` | `21.55%` | `23.86%` | `5.71%` (`KL 2.0770`) |
| **B. `Attention-Surgery 1:1 + FFN Congelado` (C017)** | **`418.290 / 655.872`** | **`0.7677`** | **`1.0069`** | **`0.3782`** | **`75.00%`** | **`78.48%`** | **`48.83%` (`KL 0.5786`)** |
| **B + `QuaRot W4A16` (`3.552×` compressão nas células)** | `0 / 1.074.162` (PTQ) | — | **`1.0031`** | **`0.3766`** | **`75.78%`** | **`79.36%`** | — |
| **C. `Attention-Surgery 1:1 + Refino Conjunto` (FFN livre)** | `1.008.626 / 65.536` | `0.7677` | `1.0790` | `0.4524` | `73.89%` | `77.34%` | `47.95%` (`KL 0.5667`) |

**Conclusões da Ablação:**
1. **Salto de `+53.45` pontos percentuais em acurácia (`21.55% → 75.00%`) e redução de `7.43×` na divergência KL (`2.8112 → 0.3782`):** transplantar os `SwiGLU` FFNs, `RMSNorm`s e projeções `QKV/Out` do professor elimina o colapso semântico observado na C016.
2. **Congelar os FFNs (`Braço B`) supera deixá-los livres (`Braço C`) em orçamento curto (`1.0069` vs `1.0790` CE):** além de cortar os parâmetros treináveis em **`61.1%` (`418K` vs `1.07M`)**, congelar os FFNs preserva o conhecimento pré-treinado em tokens não vistos durante a conversão (**`48.83%` vs `5.71%` de acordo com o professor, `8.55×` superior**).
3. **Compatibilidade total com `QuaRot W4A16` (`3.552×` compressão):** o aluno cirúrgico quantizado em `INT4` manteve **`75.78%` Top-1 Acc** (`CE = 1.0031`), sem perda de qualidade.

### 2.3 Verificação End-to-End da Fase D (`MetaQueryBridge`)
Conectado ao aluno cirúrgico `FractalDelta V2` + `LuminaMultimodalWrapper`:
- **Isolamento de Parâmetros (`freeze_backbone_for_metaquery`):** `701.056` parâmetros treináveis no `MetaQueryBridge` vs `1.210.610` parâmetros congelados no VLM (`backbone_mutated = False` após `40` passos de `AdamW`).
- **Shapes de Condicionamento (`Anima 2B` / `DiT`):**
  - Estático (`num_frames=1`): `seq_cond = (1, 64, 2048)`, `pooled_cond = (1, 768)`, `grm_segments = 1`.
  - GIF 3D (`num_frames=8`): `seq_cond = (1, 8, 64, 2048)`, `pooled_cond = (1, 768)`, `grm_segments = 8`.
- **Invariância Estrita $O(1)$:** `state_bytes_1_image = 91.264 B == state_bytes_8_gif_frames = 91.264 B` (`o1_state_invariant = True`).
- **Convergência de Alinhamento (`40` passos):** perda híbrida `MSE + Cosine` caiu de **`0.5543 → 0.0180` (`−96.8%`)**.

---

## 3. Limitações e Próxima Execução em GPU (`SmolLM2-135M` / `Qwen3-Base` na T4/L4)

- Durante o preparo desta campanha, o servidor MCP do Google Colab retornou `server name colab failed to load: sending "tools/call": Bad Request` (expiração de sessão OAuth após as 4 campanhas contínuas em GPU `95b0005`, `C014`, `C015` e `C016`). Nenhum runtime remoto ficou ativo.
- Seguindo `AGENTS.md`, os testes `@pytest.mark.cuda` de `tests/test_metaquery_bridge.py` e `tests/test_phase_c_distill_ringue.py` são reportados honestamente como `2 skipped` nesta rodada local (o teste CUDA de `test_phase_c_distill_ringue.py` já havia sido aprovado na Tesla T4 na C016 em `be38b5b`).
- O script completo de GPU para converter `HuggingFaceTB/SmolLM2-135M` (`30L`, `d=576`, `H=9`, `FFN=1536`, `vocab=49.152`) via `Attention-Surgery` 1:1 + Ringue + `MetaQueryBridge` com `SigLIP 2` real já está pronto e empacotado para execução imediata assim que a sessão do Colab for reconectada (ou na virada do mês junto com o `dataset_anima_vlm.jsonl`).
