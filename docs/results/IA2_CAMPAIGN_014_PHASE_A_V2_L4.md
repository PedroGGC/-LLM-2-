# Campanha 014 — Fase A (`FractalDelta V2`: `Gated DeltaNet-2` + Híbrido `3:1` + `Exit Gate` + `GRM` Real) na NVIDIA L4

- **Data:** 22/11/2026
- **Ambiente:** Colab `1× NVIDIA L4` (`PyTorch 2.14.1+cu130`), runtime encerrado imediatamente após coleta (`list_runtimes == {}`).
- **Dataset:** `FineWeb-Edu` (`16,80 MB` treino / `3.470` docs; `533 KB` val / `148` docs; BPE `vocab=4096` treinado apenas no split de treino após split por documento + deduplicação SHA-256).
- **Protocolo pareado:** mesma semente (`351`), mesmo orçamento (`1.500` passos = `6.144.000` tokens reais), mesmo currículo (`512×8 → 1024×4 @ 750 → 2048×2 @ 1200`), mesmo otimizador (`Muon`, `lr=0.002`), `intermediate_weight=0.20`, `mtp_tokens=2`, `mtp_weight=0.25`.

## 1. Gate de Testes
- **CPU local:**
  - `tests/test_fractal_v2_grm.py` + `tests/test_quarot_early_exit.py` + `tests/test_phase3_upgrades.py` + `tests/test_fractal_delta.py::test_unified_curriculum_and_custom_assoc_val_dim_exact_resume`: **`18 passed, 4 skipped`** (`35.95s`).
  - `tests/test_training.py` + `tests/test_decode_optimization.py`: **`30 passed, 1 skipped`** (`53.43s`).
  - Pós-fix do `CausalLocalWindow` (`F.embedding`): `tests/test_fractal_v2_grm.py`: **`7 passed, 1 skipped`** (`11.16s`).
- **GPU real (`NVIDIA L4`):**
  - `pytest -q -m cuda tests/test_fractal_v2_grm.py`: **`1 passed, 7 deselected`** (`293.67s`, `exit 0`), cobrindo `BF16` + `torch.compile` + `Triton DPLR` + `gradient_checkpointing` + `QuaRot Triton W4A16` + `CUDAGraph` Early-Exit e MTP no núcleo `V2` completo.
  - Bug latente corrigido durante o gate: no PyTorch `2.14.1+cu130`, `bias[:, (w - 1) - dist.clamp(0, w - 1)]` em `CausalLocalWindow` causava `InductorError` (`ValueError: The argument '7 - ((0))' is not comparable`); substituído por `F.embedding(rel_idx, bias.transpose(0, 1)).permute(2, 0, 1)` com equivalência matemática exata.

## 2. Comparativo Pareado `V1` vs `V2` em `FineWeb-Edu` (`1.500` passos)

| Métrica | `V1` (`fractal_pcc_mtp_muon`) | `V2` (`Gated DeltaNet-2` + Híbrido `3:1` + `Exit Gate`) | Delta (`V2 − V1`) |
| :--- | :---: | :---: | :---: |
| Parâmetros únicos (inferência) | `7.445.215` | `8.043.682` | `+598.467` (`+8,0%`) |
| Estado fixo $O(1)$ na VRAM | `844.808 B` (`825 KiB`) | `1.038.848 B` (`1.014 KiB`) | `+194.040 B` (2 janelas `w=64`) |
| Tempo de treino (incl. compile 3 shapes) | `635,9s` (`9.661,9 tok/s`) | `1.142,8s` (`5.376,1 tok/s`) | `+506,9s` (compile do loop de janela) |
| Pico VRAM no treino | `2,56 GiB` | `2,68 GiB` | `+0,12 GiB` |
| **`val@512` BPB (Dense BF16)** | `1,7297` | **`1,7257`** | **`−0,0040 BPB`** |
| **`val@1024` BPB (Dense BF16)** | `1,7135` | **`1,7097`** | **`−0,0038 BPB`** |
| **`val@2048` BPB (Dense BF16)** | `1,7062` | **`1,7025`** | **`−0,0037 BPB`** |
| **`val@1024` BPB (`QuaRot Triton W4A16`, `3,55×`)** | `1,7170` (`+0,0035`) | **`1,7132` (`+0,0035`)** | **`−0,0038 BPB`** |
| **`RULER NIAH 2K` NLL (Raw / GRM cego)** | `5,3320` / `5,1315` | `5,4349` / **`5,0719`** | **`−0,2601` NLL** (`V1 Raw → V2+GRM`) |
| **`RULER NIAH 8K` NLL (Raw / GRM cego)** | `5,9785` / `5,9247` | `5,7885` / **`5,3210`** | **`−0,6575` NLL** (`V1 Raw → V2+GRM`) |
| **`RULER NIAH 16K` NLL (Raw / GRM cego)** | `6,6527` / `6,5494` | `6,4439` / **`6,0366`** | **`−0,6161` NLL** (`V1 Raw → V2+GRM`) |
| Decode `Dense BF16` (`256` tok) | `1.140,6 tok/s` (`55,10 mJ/tok`) | `942,0 tok/s` (`52,91 mJ/tok`) | −17,4% tok/s (2 janelas locais) |
| Decode `MTP CUDAGraph` ungated (`256` tok) | `1.561,6 tok/s` (`33,32 mJ/tok`, `1,49 tok/ciclo`) | **`1.227,2 tok/s` (`33,44 mJ/tok`, `1,38 tok/ciclo`)** | **`+30,3%` tok/s** e **`−36,8%` mJ/tok** vs V2 Dense |
| Decode `Exit Gate` aprendido (`t=0,50`, `CUDAGraph`) | n/a | **`53,3%` early-exit (`5,87/8L`, `26,7%` camadas salvas)**, `590,9 tok/s`, `54,9 W` | Calibrou (`match=67,3%`), pronta para escala `≥ 500M` |

## 3. Limitações Honestas & Ações Futuras
1. `exact_match` em `RULER NIAH` permaneceu `0%` em ambos os braços (modelos de `8M` com apenas `6,14M` tokens de pré-treino web não dominam QA de passkey zero-shot; a superioridade de retenção foi medida via `mean_answer_nll`).
2. `CausalLocalWindow.forward` ainda itera em chunks de `64` em Python, quase dobrando o tempo de compilação Inductor em `seq=1024/2048` (`1.143s` vs `636s`). Antes de escalar para `150M–3B`, substituir o loop de janela por uma chamada única de `SDPA`/FlexAttention com máscara banded.
3. Em `8M` (`~1 ms/tok`), o sync host do `Exit Gate` em 2 `CUDAGraphs` ainda anula o speedup de parede (`590,9` vs `942,0 tok/s`), mas já economiza `26,7%` das camadas e derruba a potência média (`64,8 W → 54,9 W`), tendendo a virar speedup líquido em `≥ 500M–3B`.
