# Relatório Técnico — Campanha 008 (Colab L4, BF16 + Estado FP32, `fractal_golden_classic_big` vs `fractal_golden_big`)

- **Data:** `2026-11-15`
- **Hardware / Ambiente:** Google Colab Pro via MCP (`https://colaboratory.googleapis.com/mcp`), `1× NVIDIA L4` (`22,03 GiB` / `23.659.151.360` B, UUID `28b2aec4-b9c9-0cb2-57c7-f66ad2465caf`), `Linux-6.6.159+-x86_64`, Python `3.13.16`, PyTorch `2.14.1+cu130`, CUDA `13.0`, cuDNN `92800`.
- **Commit base testado:** `3b178e0c10fe4f1c5508f532114942be2c22c620` (branch `llm-ideas`).
- **Suíte de Invariantes Pré-Campanha na L4:**
  - `python -m pytest -q tests/test_fractal_delta.py`: **`29 passed in 56.59s`** (`0` pulados).
  - `python -m pytest -q -m cuda`: **`55 passed in 131.83s`** (`0` pulados).
- **Protocolo Unificado (`--mode unified --precision bf16 --story-seq-len 512 --binding-hard-writes 64 --lr 0.003 --seeds 351`):**
  - Treino de checkpoint único (`6.000` steps, `batch_size = 16`: `12` sequências TinyStories + `4` sequências `byte-binding-v2` com rampa `4 -> 8 -> 16 -> 32` escritas).
  - Projeções densas e FFNs em `BF16` via Tensor Cores; recorrências DPLR, memória associativa, convoluções causais, `RMSNorm`, fase áurea (`int64` + `float64`) e loss em `FP32` estrito.

---

## 1. Mecanismos Clássicos Incorporados (`fractal_golden_classic_big`)

1. **`fractal_assoc_delta=True` (Regra Delta de Kohonen 1984 / Widrow-Hoff + Retenção Elástica de Hinton & Plaut 1987):**
   - Substitui a acumulação Hebbiana aditiva pura (`M_t = M_{t-1} + g_t k_t v_t^T`) do canal indutivo lag-1 por atualização projetiva DPLR em blocos (`dplr_chunked` / `dplr_sequential` com `heads=1`):
     $$M_t = w_t (I - g_t k_t k_t^\top) M_{t-1} + g_t k_t v_t^\top, \qquad \log w_t = \log(1 - 0{,}01\, g_t)$$
   - Quando um token é neutro (`g_t -> 0`), $w_t \to 1$ preserva os vínculos por milhares de passos sem decaimento passivo; quando um vínculo é escrito/sobrescrito (`g_t -> 1`), a projeção $(I - g_t k_t k_t^\top)$ apaga o valor anterior na direção $k_t$ e ortogonaliza o subespaço contra crosstalk.
2. **`fractal_leaky_budget=True` (Acoplamento Multiescala de Mozer 1991 / El Hihi & Bengio 1995):**
   - Nos níveis fractais lentos (`lvl >= 3`, onde `span > 0` alonga o horizonte $\tau_l$), inicializa `beta_bias = -max(0, span)` para escalar a taxa de escrita inversamente ao horizonte de retenção, mantendo os níveis rápidos (`lvl = 0, 1, 2`) inalterados (`beta_bias = 0.0`).
3. **Custo Estrutural:** **`0` parâmetros extras** (`864.869` vs `864.869`) e **`0` bytes extras de estado** (`422.920` B/seq vs `422.920` B/seq).

---

## 2. Comparação Pareada Direta (`seed=351`, Modo `unified`, `BF16` na L4 vs `FP32` na T4)

### 2.1 TinyStories, Bandas Posicionais, Intervenção de Prefixo e Custo de Treino/Decode

| Braço (Precisão / GPU) | Params | Estado / seq | Treino (s) | VRAM Treino | `val@256` | `val@512` | `val@3219` | `1–256` | `257–512` | `513–2048` | **`2049+`** | **`Δ Trunc`** | `Δ Swap` | Decode Inductor (`B/s`) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `fractal_golden_big:s351` (`FP32` / T4, Camp. 007) | `864.869` | `422.920` B | `1.095,0s` | `2,47 GB` | `0.7894` | `0.7621` | `0.7463` | `0.6812` | `0.7626` | `0.7793` | `1.0281` | `-0.0020` | `+0.0083` | `773,3` |
| `fractal_golden_big:s351` (`BF16` / L4, Camp. 008) | `864.869` | `422.920` B | **`635,6s`** | **`1,72 GB`** | `0.7895` | `0.7622` | `0.7465` | **`0.6813`** | **`0.7628`** | `0.7796` | `1.0276` | `-0.0021` | **`+0.0082`** | **`2.286,2`** |
| **`fractal_golden_classic_big:s351`** (`BF16` / L4, Camp. 008) | **`864.869`** | **`422.920` B** | `642,6s` | `1,73 GB` | `0.7913` | `0.7646` | **`0.7420`** | `0.6868` | `0.7640` | **`0.7694`** | **`0.7261`** | **`+0.0053`** | `+0.0079` | `2.179,5` |

### 2.2 Extrapolação de Memória Simbólica (`byte-binding-v2`, avaliada no mesmo checkpoint unificado)

| Braço (`seed=351`, Modo `unified`) | Bind `@256` (`w=32`, acc / paired) | Bind `@1024` (`w=32`, acc / paired) | **Bind `@4096` (`w=32`, acc / paired)** | **Bind `@8192` (`w=32`, acc / paired)** | **Bind HARD `@8192` (`w=64`, acc / paired)** | `query_nll` `@8192 (w=32)` | `query_nll` `@8192 (w=64)` |
|---|---:|---:|---:|---:|---:|---:|---:|
| `fractal_golden_big:s351` (`FP32` / T4, Camp. 007) | `99.9% / 99.9%` | `100.0% / 100.0%` | `78.9% / 67.3%` | `51.4% / 27.4%` | — | `2.658` | — |
| `fractal_golden_big:s351` (`BF16` / L4, Camp. 008) | `99.8% / 99.8%` | `99.8% / 99.8%` | `83.5% / 72.4%` | `60.9% / 36.8%` | `56.0% / 34.0%` | `2.135` | `2.356` |
| **`fractal_golden_classic_big:s351`** (`BF16` / L4, Camp. 008) | **`99.8% / 99.7%`** | **`100.0% / 100.0%`** | **`99.4% / 98.2%`** | **`95.1% / 90.4%`** | **`78.4% / 67.9%`** | **`0.242`** | **`0.938`** |

---

## 3. Conclusões Empíricas

1. **Validação de `BF16` + Estado `FP32` na GPU NVIDIA L4:**
   - O `fractal_golden_big:s351` em `BF16` reproduziu o `FP32` dentro de `0,0001–0,0002` NLL em todas as métricas (`val@256 = 0.7895` vs `0.7894`, `val@512 = 0.7622` vs `0.7621`, `val@3219 = 0.7465` vs `0.7463`), reduzindo a VRAM de treino em **`-30,5%`** (`2,47 GB -> 1,72 GB`) e atingindo **`82.817` tokens/s** pós-aquecimento e **`2.286,2 B/s`** em decode compilado com CUDA Graphs (`parity_max_abs_err = 1,24e-5`).
2. **Eliminação do Crosstalk Associativo em `@4096` e `@8192` (`16x` além do horizonte de treino `512`):**
   - Com a Regra Delta de Kohonen + retenção elástica (`fractal_assoc_delta`), a acurácia em `@4096` subiu de `83,5%` (`72,4%` paired) para **`99,4%` (`98,2%` paired)** (`+25,8` pp paired), em `@8192 (w=32)` saltou de `60,9%` (`36,8%` paired) para **`95,1%` (`90,4%` paired)** (`+53,6` pp paired, reduzindo `query_nll` de `2.135` para **`0.242`**), e no regime extremo `@8192 (w=64)` dobrou o `paired_accuracy` de `34,0%` para **`67,9%`** (`78,4%` acc).
3. **Eliminação da Deriva de Longo Horizonte (`2049+` e `Δ Trunc`):**
   - Com o acoplamento multiescala de Mozer/Bengio + `fractal_assoc_delta`, o NLL na banda `513–2048` caiu de `0.7796` para **`0.7694`**, na banda extrema `2049+` caiu de `1.0276` para **`0.7261`** (`-0,3015` NLL), o `val@3219` caiu de `0.7465` para **`0.7420`**, e a intervenção causal de prefixo distante (`Δ Trunc`) passou de negativa (`-0.0021`) para **positiva (`+0.0053`)**.
   - Trade-off observado: leve aumento de `+0.0018` em `val@256` (`0.7895 -> 0.7913`) e `+0.0055` na banda curta `1–256` (`0.6813 -> 0.6868`), amplamente compensado pelos ganhos em `513–2048`, `2049+`, `val@3219`, `Δ Trunc` e `@4096–@8192`.
4. **Rastreabilidade de Checkpoints (`SHA256`):**
   - `fractal_golden_big:s351`: `c6173f19bef10d591d6315b0939cc17af67667924a2308dc909d8dd1240f1af6`
   - `fractal_golden_classic_big:s351`: `c1a6f15fceec812fff29049c9af3f5881db81e1fc4097304027e802e39052a39`
