# IA2 Campanha 010 (Google Colab NVIDIA L4 — Etapa 3: BPE `vocab=4096` + Prelude-Core-Coda + Multilag + MTP + Muon vs KDA)

Data: 2026-11-15  
Branch: `llm-ideas` (commit executado na L4: `b7c6461`)  
Hardware/Ambiente: `NVIDIA L4` (`23.66 GB`, `22.03 GiB`), PyTorch `2.14.1+cu130`, CUDA `13.0`, cuDNN `92800`, Python `3.13.16`, `tokenizers 0.23.3`, `safetensors 0.8.0`, precisão `BF16`, `torch.compile(..., dynamic=False)`, modo `unified`, `--tokenizer bpe --vocab-size 4096 --story-seq-len 512 --binding-hard-writes 64 --steps 3500`, calibração automática em `{0.0003, 0.001, 0.003}`, seeds `351, 352, 353`.

Dataset BPE train-only fixado na L4 (`/content/stories_bpe_cache/prepared-bpe-v4096-96af1443b0104ead-9a961867`):
- `train`: `16,777,421` bytes (`18,876` docs) -> `4,207,777` tokens BPE (`3.987×`)
- `val`: `524,508` bytes (`630` docs) -> `130,355` tokens BPE (`4.024×`)
- `cal_val`: `525,115` bytes (`604` docs) -> `131,126` tokens BPE (`4.005×`)
- `test` (reservado, não tocado): `524,973` bytes (`656` docs) -> `130,076` tokens BPE (`4.036×`)

*(Nota de unidade: em BPE, `val_*_nll` representa NLL por token BPE, enquanto `val_*_bpb` normaliza a soma exata de bits pelo número real de bytes UTF-8 da partição de texto. No benchmark de geração `decode.inductor`, `new_bytes=256` corresponde a `256` passos de token BPE; portanto a taxa relatada equivale a `tokens/s`.)*

---

## 1. Fase 1 + Fase 2: Estudo de Ablação Completo (`seed=351`, `3.500` passos)

Objetivo: isolar a contribuição de **Prelude-Core-Coda (`PCC`) + Multilag**, **Muon**, **Multi-Token Prediction (`MTP`, 4 horizontes)** e **Kimi Delta Attention (`KDA`)**, comparando com o baseline denso **`transformer_bpe_med`** (`15.74M`, `2.11×` maior em inferência que os candidatos `7.45M`) e o recorrente clássico de 1 célula compartilhada **`fractal_classic_bpe`** (`3.60M`).

| Arm (`seed=351`) | Otim. | Params (Inf / Total) | Estado (`B/seq`) | `LR` | VRAM Pico | Wall (`s`) | `val@256` BPB | `val@512` BPB | `val@3219` BPB (`NLL`) | Decode Inductor (`tok/s`) | Bind `8k w=32` (`acc / par`) | Bind `HARD 8k w=64` (`acc / par`) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `transformer_bpe_med` | AdamW | `15,735,168` | `6,291,464` | `0.001` | `1.72 GiB` | `172.6` | `1.0660` | `1.0373` | `1.0333` (`2.8820`) | `346.0` | `1.56% / 0.00%` | `2.05% / 0.00%` |
| `fractal_classic_bpe` | AdamW | `3,595,591` | `844,808` | `0.003` | `3.41 GiB` | `584.8` | `1.1316` | `1.0924` | `1.0888` (`3.0365`) | `1,395.9` | `99.61% / 99.22%` | `91.50% / 85.06%` |
| `fractal_pcc_bpe` | AdamW | `7,445,215` | `844,808` | `0.003` | `3.61 GiB` | `593.6` | `1.0745` | `1.0345` | `1.0333` (`2.8820`) | `1,385.8` | `99.71% / 99.22%` | `91.70% / 84.96%` |
| `fractal_pcc_muon_bpe` | Muon | `7,445,215` | `844,808` | `0.003` | `3.60 GiB` | `652.1` | `1.0432` | `1.0028` | `0.9995` (`2.7876`) | **`1,417.4`** | `92.77% / 87.21%` | `75.49% / 64.26%` |
| **`fractal_pcc_mtp_muon_bpe`** | **Muon** | **`7,445,215` / `10,590,943`** | **`844,808`** | **`0.003`** | **`4.06 GiB`** | `703.7` | **`1.0397`** | **`0.9980`** | **`0.9943` (`2.7732`)** | `1,388.6` | **`100.00% / 100.00%`** | **`99.61% / 99.02%`** |
| `fractal_pcc_full_bpe` (`+KDA`) | Muon | `7,476,301` / `10,622,029` | `844,808` | `0.003` | `10.97 GiB` | `1,195.5` | `1.0483` | `1.0047` | `1.0011` (`2.7920`) | `1,360.1` | `100.00% / 100.00%` | `99.32% / 98.93%` |

### Achados causais de cada bloco na ablação (`seed=351`)
1. **Prelude-Core-Coda + Multilag (`fractal_classic_bpe` → `fractal_pcc_bpe`):** reduziu `val@512 BPB` em **`-0.0579` BPB** (`1.0924` → `1.0345`), igualando/superando o Transformer de `2.11×` mais parâmetros (`1.0373`) com estado `7.45×` menor e decode `4.0×` mais rápido.
2. **Muon (`fractal_pcc_bpe` → `fractal_pcc_muon_bpe`):** reduziu `val@512 BPB` em mais **`-0.0317` BPB** (`1.0345` → `1.0028`), rompendo `1.000` em `val_long BPB` (`0.9995`), porém degradou a memória associativa extrema em `8k`/`HARD` (`99.71% → 92.77%` e `91.70% → 75.49%`).
3. **MTP + Muon (`fractal_pcc_muon_bpe` → `fractal_pcc_mtp_muon_bpe`):** **corrigiu integralmente a queda de memória do Muon** (`8k w=32`: `92.77% → 100.00%`; `HARD w=64`: `75.49% → 99.61%` acurácia / `99.02%` pareada) **e** melhorou ainda mais a modelagem de linguagem (`val@512`: **`0.9980` BPB**, `val@3219`: **`0.9943` BPB**). Como as 3 cabeças extras de `MTP` são descartadas após o treino, o custo de inferência permanece `7.45M` parâmetros (`844,808` B/seq).
4. **KDA (`fractal_pcc_mtp_muon_bpe` → `fractal_pcc_full_bpe`):** adicionar KDA (`rank-1` + decaimento por canal) **piorou** `val@256/512/3219 BPB` (`1.0397→1.0483`, `0.9980→1.0047`, `0.9943→1.0011`), reduziu marginalmente `HARD` (`99.61%→99.32%`), aumentou a VRAM em **`2.70×`** (`4.06 → 10.97 GiB`) e desacelerou o treino (`28.5k → 16.9k tok/s`). Única vantagem isolada observada do KDA foi `ΔSwap` ligeiramente maior (`+0.0953` vs `+0.0775`). Decisão para confirmação e Etapa 4: **descartar KDA e consolidar `fractal_pcc_mtp_muon_bpe`**.

---

## 2. Confirmação Multi-Seed (`n=3`: seeds `351, 352, 353`)

### TinyStories BPE (`vocab=4096`) e Decode Inductor (`7936` ctx + `256` novos tokens)

| Arm | Seed | `val@256` BPB | `val@512` BPB | `val@3219` BPB | `val@3219` NLL | `1–256` NLL | `257–512` NLL | `513–2048` NLL | `ΔTrunc` | `ΔSwap` | Decode (`tok/s`) | VRAM (`GiB`) | SHA-256 (`model.safetensors`, prefixo) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `transformer_bpe_med` | `351` | `1.0660` | `1.0373` | `1.0333` | `2.8820` | `2.8853` | `2.7352` | `3.3450` | `+0.0211` | `+0.0722` | `346.0` | `1.72` | `eb17b0b3f551` |
| `transformer_bpe_med` | `352` | `1.0716` | `1.0421` | `1.0374` | `2.8932` | `2.9047` | `2.6753` | `3.1103` | `+0.0204` | `+0.0738` | `340.2` | `1.72` | `92444bd4099a` |
| `transformer_bpe_med` | `353` | `1.0687` | `1.0410` | `1.0375` | `2.8936` | `2.8978` | `2.7214` | `3.4155` | `+0.0237` | `+0.0798` | `337.2` | `2.01` | `659478436854` |
| **`transformer_bpe_med` (Média ± DP)** | `3` | `1.0688 ± 0.0028` | `1.0402 ± 0.0025` | `1.0361 ± 0.0024` | `2.8896 ± 0.0066` | `2.8959 ± 0.0099` | `2.7106 ± 0.0314` | **`3.2903 ± 0.1598`** | `+0.0217 ± 0.0017` | `+0.0753 ± 0.0040` | `341.2 ± 4.5` | `1.82 ± 0.17` | — |
| `fractal_pcc_mtp_muon_bpe` | `351` | `1.0397` | `0.9980` | `0.9943` | `2.7732` | `2.7681` | `2.6292` | `3.8291` | `-0.0053` | `+0.0775` | `1,388.6` | `4.06` | `267358a3735a` |
| `fractal_pcc_mtp_muon_bpe` | `352` | `1.0470` | `1.0014` | `0.9972` | `2.7813` | `2.7807` | `2.5475` | `3.9378` | `+0.0147` | `+0.0869` | `1,401.8` | `3.96` | `3718fbd9f24b` |
| `fractal_pcc_mtp_muon_bpe` | `353` | `1.0487` | `1.0060` | `1.0018` | `2.7941` | `2.7864` | `2.7009` | `3.7909` | `+0.0075` | `+0.0939` | `1,395.8` | `3.94` | `0f6b15bfd585` |
| **`fractal_pcc_mtp_muon_bpe` (Média ± DP)** | `3` | **`1.0451 ± 0.0048`** | **`1.0018 ± 0.0040`** | **`0.9978 ± 0.0038`** | **`2.7828 ± 0.0105`** | **`2.7784 ± 0.0094`** | **`2.6259 ± 0.0768`** | `3.8526 ± 0.0762` | `+0.0056 ± 0.0101` | **`+0.0861 ± 0.0082`** | **`1,395.4 ± 6.6`** | `3.99 ± 0.06` | — |
| **Delta (`Fractal − Transformer`)** | `3` | **`-0.0237`** | **`-0.0384`** | **`-0.0383`** | **`-0.1068`** | **`-0.1175`** | **`-0.0847`** | `+0.5623` | `-0.0161` | **`+0.0108`** | **`4.09×` mais rápido** | `+2.17` | `7.45×` menos estado (`844 KB` vs `6.29 MB`) |

### Byte-Binding-v2 em Contexto até `8.192` (`accuracy / paired_accuracy`)

| Arm | Seed | `@256 (w=32)` | `@1024 (w=32)` | `@4096 (w=32)` | `@8192 (w=32)` | `HARD @8192 (w=64)` |
|---|---|---:|---:|---:|---:|---:|
| `transformer_bpe_med` | `351` | `1.66% / 0.00%` | `1.76% / 0.00%` | `2.15% / 0.00%` | `1.56% / 0.00%` | `2.05% / 0.00%` |
| `transformer_bpe_med` | `352` | `1.46% / 0.00%` | `1.46% / 0.00%` | `1.46% / 0.00%` | `1.17% / 0.20%` | `1.76% / 0.00%` |
| `transformer_bpe_med` | `353` | `1.76% / 0.00%` | `1.76% / 0.00%` | `1.17% / 0.00%` | `1.76% / 0.00%` | `1.56% / 0.00%` |
| **`transformer_bpe_med` (Média ± DP)** | `3` | `1.63 ± 0.15% / 0.00%` | `1.66 ± 0.17% / 0.00%` | `1.60 ± 0.50% / 0.00%` | `1.50 ± 0.30% / 0.07 ± 0.11%` | `1.79 ± 0.25% / 0.00%` |
| `fractal_pcc_mtp_muon_bpe` | `351` | `100.00% / 99.90%` | `100.00% / 100.00%` | `100.00% / 100.00%` | `100.00% / 100.00%` | `99.61% / 99.02%` |
| `fractal_pcc_mtp_muon_bpe` | `352` | `99.90% / 99.90%` | `100.00% / 100.00%` | `100.00% / 99.80%` | `100.00% / 100.00%` | `99.22% / 98.34%` |
| `fractal_pcc_mtp_muon_bpe` | `353` | `99.90% / 99.90%` | `100.00% / 100.00%` | `100.00% / 100.00%` | `100.00% / 99.90%` | `98.63% / 98.24%` |
| **`fractal_pcc_mtp_muon_bpe` (Média ± DP)** | `3` | **`99.93 ± 0.06% / 99.90%`** | **`100.00% / 100.00%`** | **`100.00% / 99.93 ± 0.11%`** | **`100.00% / 99.97 ± 0.06%`** | **`99.15 ± 0.49% / 98.54 ± 0.43%`** |

Erro máximo de paridade eager vs `torch.compile` Inductor (`decode.inductor.parity_max_abs_err`): `8.11e-6`–`1.29e-5` no Transformer e `1.43e-5`–`1.72e-5` no `fractal_pcc_mtp_muon_bpe`, todos com `cudagraph=True` nos candidatos fractais.

---

## 3. Observações Qualitativas (`samples`) e Limitações Honestas

1. **Coerência gerada vs colapso do baseline recorrente antigo:**
   - `fractal_classic_bpe:s351` (`3.60M`) repetiu em loop no prompt 1 (`"\nThe little \nThe little..."`).
   - `fractal_pcc_mtp_muon_bpe:s353` produziu prosa narrativa coerente (`"\nOne day, a little girl named Lily went to the park with her mom. She saw a big tree and wanted to climb it..."`), comparável ao `transformer_bpe_med` de `2.11×` mais parâmetros.
   - *Detalhe cosmético do harness:* os dois prompts fixos de `sample_story_text` terminam com espaço literal (`"Once upon a time, a little "`), quebrando o limite natural de subword BPE (onde espaços antecedem o token seguinte), gerando primeiro token anômalo em vários runs dos dois modelos (`"15"`, `"Vo"`, `"Very"`, `"x"`, `"Okay"`). Corrigir prompts sem espaço final fica para a Etapa 4.
2. **Banda `513–2048` em linguagem natural sem currículo:**
   - Embora `fractal_pcc_mtp_muon_bpe` supere largamente o Transformer em `val@3219 BPB` total (`0.9978` vs `1.0361`) e na banda `257–512` (`2.6259` vs `2.7106`), na banda `513–2048` (`1.679` bytes da partição `val` acima de `512` tokens), treinando apenas em `seq=512`, sua NLL foi maior que a do Transformer (`3.8526 ± 0.0762` vs `3.2903 ± 0.1598`), diferentemente da Campanha 009 onde `--unified-curriculum` (`512→1024→2048`) resolveu extrapolação longa.
   - **Implicação direta para Etapa 4 (`100M`+):** manter exatamente a receita **`fractal_pcc_mtp_muon_bpe` (sem KDA)** e reintroduzir extensão de contexto/currículo (`512 → 1024 → 2048+`) em corpus mais longo (FineWeb-Edu / cosmopedia / livros) antes de avaliar benchmarks externos.
3. **Escopo científico:** os resultados acima comparam arquiteturas sob orçamento controlado (`3.500` passos em TinyStories BPE `16.8 MB`), **não** representam treinamento de modelo fundacional completo nem avaliação em benchmarks abertos de raciocínio/código/chat de LLMs `100M–1B`.
