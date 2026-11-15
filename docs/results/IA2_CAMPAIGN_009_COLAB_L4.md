# IA2 Campanha 009 (Google Colab NVIDIA L4 — Fechamento do Passo 2: `assoc_val_dim=48` + Currículo `512→1024→2048`)

Data: 2026-11-15  
Branch: `llm-ideas` (commit executado: `bd7bcb9a24b7df41706a1f3009bed4a4cf021d65`)  
Hardware/Ambiente: `NVIDIA L4` (`23.66 GB`), PyTorch `2.14.1+cu130`, Python `3.13.16`, `BF16`, `torch.compile(..., dynamic=False)`, modo `unified`, `--unified-curriculum`, `--story-seq-len 512`, `--binding-hard-writes 64`, `--lr 0.003`, seeds `351, 352, 353`.

---

## 1. Objetivo e Mudanças Validadas

1. **Capacidade associativa ampliada (`fractal_golden_classic_v2_big`):** `fractal_assoc_val_dim=48` (`877,157` parâmetros, `432,136` B/seq de estado) mantendo todas as propriedades do `fractal_golden_classic_big` (`864,869` parâmetros, `assoc_val_dim=24`, `422,920` B/seq).
2. **Currículo progressivo de comprimento (`--unified-curriculum`):** `512×16` (passos `1..3000`) → `1024×8` (passos `3001..4500`) → `2048×4` com `writes=48` no binding (passos `4501..6000`), mantendo constante `8.192` tokens/passo.
3. **Correção de compilação (`src/llm2/trainer.py`):** durante a primeira tentativa da seed `351` na transição `3001 (512→1024)`, o `torch.compile` padrão (`dynamic=None`) acionou recompilação simbólica (`s70`) e atingiu `CUDA_ERROR_ILLEGAL_ADDRESS` no autotune do kernel Triton `triton_poi_fused_linalg_vector_norm_25` no PyTorch `2.14.1+cu130`. Especializar o `Trainer` com `torch.compile(..., dynamic=False)` (consistente com `decode_optimization.py` e `state_platforms.py`) eliminou a falha em todas as 3 transições de estágio nas 3 seeds.

---

## 2. Resultados por Seed (`fractal_golden_classic_v2_big`, L4 BF16)

Duração total da campanha (3 seeds): `2354.5s` (`39.2 min`), `status: complete`, `deferred_arms: []`.

| Seed | `val@256` | `val@512` | `val@3219` | Banda `1-256` | Banda `257-512` | Banda `513-2048` | Banda `2049+` | `ΔTrunc` | `ΔSwap` | Decode Inductor (`B/s`) | Wall (`s`) | Peak VRAM (`B`) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `351` | `0.9483` | `0.9159` | `0.8864` | `0.8278` | `0.9295` | `0.9034` | `0.7066` | `+0.0113` | `+0.0123` | `2,186.6` | `929.6` | `1,722,043,904` |
| `352` | `0.9456` | `0.9130` | `0.8828` | `0.8211` | `0.9256` | `0.9022` | `0.7304` | `+0.0110` | `+0.0122` | `2,240.8` | `499.3` | `1,733,020,672` |
| `353` | `0.9403` | `0.9077` | `0.8778` | `0.8183` | `0.9185` | `0.8969` | `0.7293` | `+0.0107` | `+0.0117` | `2,218.4` | `500.7` | `1,743,585,792` |
| **Média ± DP** | **`0.9447 ± 0.0041`** | **`0.9122 ± 0.0041`** | **`0.8823 ± 0.0043`** | **`0.8224 ± 0.0049`** | **`0.9245 ± 0.0056`** | **`0.9008 ± 0.0034`** | **`0.7221 ± 0.0134`** | **`+0.0110 ± 0.0003`** | **`+0.0120 ± 0.0003`** | **`2,215.2 ± 27.3`** | `643.2 ± 248.0` | `1.73 GB` (`1.61 GiB`) |

*(Nota de tempo: a seed `351` incluiu a compilação Inductor para os 3 estágios discretos `512/1024/2048` em `929.6s`; `352` e `353` reutilizaram o cache em disco em `499.3s` e `500.7s`.)*

### Byte-Binding-v2 (`accuracy / paired_accuracy`)

| Seed | `@256 (w=32)` | `@1024 (w=32)` | `@4096 (w=32)` | `@8192 (w=32)` | `HARD @8192 (w=64)` |
|---|---:|---:|---:|---:|---:|
| `351` | `99.1% (99.0%)` | `99.8% (99.6%)` | `100.0% (100.0%)` | `99.9% (99.8%)` | `96.3% (92.5%)` |
| `352` | `99.6% (99.5%)` | `99.7% (99.3%)` | `99.8% (99.6%)` | `99.8% (99.6%)` | `95.1% (90.4%)` |
| `353` | `99.2% (98.7%)` | `99.9% (99.8%)` | `99.6% (99.3%)` | `99.3% (99.0%)` | `93.9% (89.7%)` |
| **Média ± DP** | **`99.3% ± 0.3%` (`99.1% ± 0.4%`)** | **`99.8% ± 0.1%` (`99.6% ± 0.2%`)** | **`99.8% ± 0.2%` (`99.6% ± 0.3%`)** | **`99.7% ± 0.3%` (`99.5% ± 0.4%`)** | **`95.1% ± 1.2%` (`90.9% ± 1.4%`)** |

Checkpoints SHA-256 (`model.safetensors`):
- `s351`: `f4192af7be9c84c7fdfabb0a4cf2316c340645335cd92ccea715b2e512d9512a`
- `s352`: `030d31fb86ef52f1c89b247c6b38ecc413c37fcacf18365a62286cadfc6fef30`
- `s353`: `dfe654d1d38b757e94d075230d2cc5c79a84bed9e738c2c705d55f2a6081876e`

---

## 3. Comparação Direta com a Campanha 008 (L4 BF16, Modo Unified)

| Braço | Params | State (`B/seq`) | `val@512` | `val@3219` | `2049+` | `ΔTrunc` | `ΔSwap` | Bind `@4096 w=32` | Bind `@8192 w=32` | Bind `HARD @8192 w=64` |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `fractal_golden_big` (C008, `s351`) | `864,869` | `422,920` | `0.7622` | `0.7465` | `1.0276` | `-0.0021` | `+0.0082` | `83.5% (72.4%)` | `60.9% (36.8%)` | `56.0% (34.0%)` |
| `fractal_golden_classic_big` (C008, `s351`) | `864,869` | `422,920` | **`0.7646`** | **`0.7420`** | `0.7261` | `+0.0053` | `+0.0079` | `99.4% (98.2%)` | `95.1% (90.4%)` | `78.4% (67.9%)` |
| **`fractal_golden_classic_v2_big` (C009, média 3 seeds)** | `877,157` | `432,136` | `0.9122` | `0.8823` | **`0.7221`** | **`+0.0110`** | **`+0.0120`** | **`99.8% (99.6%)`** | **`99.7% (99.5%)`** | **`95.1% (90.9%)`** |

*(Observação pareada em `s351` antes da entrada do estágio `2048`: em `step=3000` (`512×16`), a `v2_big` marcava `story_val=0.8303`, `bind_acc=97.3%`; em `step=4500` (`1024×8`), `story_val=0.8531`, `bind_acc=99.8%`; em `step=6000` (`2048×4, writes=48`), `story_val=0.9159`, `bind_acc=99.6%`.)*

---

## 4. Conclusões do Passo 2 e Próxima Etapa

1. **Memória associativa de longo prazo validada:** elevar `assoc_val_dim` para `48` + currículo `512→1024→2048` com `writes=48` resolveu o gargalo de capacidade de escrita em `8192`:
   - `Binding @8192 w=32` subiu de `95.1% (90.4%)` → **`99.7% (99.5%)`**.
   - `Binding HARD @8192 w=64` saltou de `78.4% (67.9%)` → **`95.1% (90.9%)`** (**+16.7 pp** em acurácia e **+23.0 pp** em pareada).
   - A banda `2049+` caiu para **`0.7221`** (`0.7066` na `s351`) e a sensibilidade causal de prefixo dobrou (`ΔTrunc = +0.0110` vs `+0.0053`, `ΔSwap = +0.0120` vs `+0.0079`).
2. **Tradeoff observado em modelagem de linguagem pura (`val@512`):** deslocar 50% do orçamento de passos (`3001..6000`) para sequências de `1024/2048` e elevar a mistura de escritas (`writes=48`) aumentou `val@512` (`0.7646` → `0.9122`), mesmo reduzindo `2049+`. Como o foco do **Passo 3** muda para **tokenização BPE (`vocab=4096/8192`) + Prelude-Core-Coda + KDA/Muon** em escala de texto real, esse ajuste de proporção de mistura será recalibrado diretamente no novo pipeline BPE.
3. **Verificações CUDA complementares na L4 (`bd7bcb9`):** `pytest -q -m cuda` aprovou **`55 passed, 765 deselected`** em `695.64s`.
