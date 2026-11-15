# Marco de Fechamento da Fase Comparativa (IA2 001–006) e Seleção do `fractal_golden` — 15/11/2026

Este documento registra o encerramento da fase de triagem e comparação entre arquiteturas candidatas na bancada dupla (**TinyStories** + **Byte-Binding v2** + **Compiled Decode Inductor/CUDAGraph**) e formaliza a decisão de avançar exclusivamente com o aperfeiçoamento e escalonamento do **`fractal_golden`** (`FractalDeltaMemory` com `fractal_value_residual=True` e `fractal_golden_phase=True`).

---

## 1. Resumo Executivo: Por que paramos de comparar e escolhemos o `fractal_golden`

Ao longo de seis campanhas consecutivas nas duas GPUs Tesla T4 do Kaggle (`llm2-ia2-campaign-001` a `006`), evoluímos a família `FractalDeltaMemory` desde o primeiro protótipo recorrente DPLR até uma arquitetura pura de 6 níveis compartilhados com atalho residual de valor ($v_0$) e modulação posicional quase-periódica baseada na proporção áurea ($\phi$).

O candidato vencedor — **`fractal_golden`** — cumpriu simultaneamente os quatro critérios que nenhuma variante anterior (LRM, FLRM clássico, mLSTM, RWKV, `ContextMemory` ou Transformer isolado) havia atingido ao mesmo tempo:

1. **Qualidade de linguagem superior em igualdade de escala (Campanhas 004 e 005):**
   - Na escala de `~154k` parâmetros (`seq_len=256`), `fractal_golden_med` atingiu **`0,9179` NLL** em validação curta (`@256`) e **`0,9126` NLL** em histórias longas (`@3219`), superando todos os demais fractais (`fractal_pure_med` `0,9256`, `fractal_free_energy_med` `0,9372`, `fractal_quat_med` `0,9404`, `fractal_ternary_med` `1,1788`) e batendo com folga o **`transformer_2x`** (`190.080` parâmetros, `0,9392` NLL `@256` e `2,7319` NLL `@3219`).
2. **Confirmação em escala ~5,6× maior e contexto dobrado (Campanha 006):**
   - Escalado para **`864.869` parâmetros** (`d_model=256`, 6 níveis compartilhados, `chunk_size=64`) e treinado em `seq_len=512`, o `fractal_golden_big` venceu novamente as alternativas fractais em todos os comprimentos (`0,7707` `@256`, `0,7437` `@512`, `0,7282` `@3219`).
   - Frente ao **`transformer_big`** de 6 camadas independentes e **`1.091.328` parâmetros** (**26,2% maior**), o `fractal_golden_big` ficou a apenas `0,0027`–`0,0037` NLL dentro da janela de treino (`0,7707` vs `0,7680` `@256`; `0,7437` vs `0,7400` `@512`) e **venceu por metade da perda na extrapolação de longo alcance `@3219`** (**`0,7282` vs `1,4256`**, onde o Transformer colapsa fora da janela de 512 bytes).
3. **Memória associativa exata de longo alcance (`Byte-Binding v2` até 8.192 bytes):**
   - Enquanto os Transformers de referência falham completamente na retenção associativa em 8.192 bytes (`0,0%` a `2,5%` de acurácia pareada), o `fractal_golden_big` manteve **`100,0%` de acurácia original e `100,0%` contrafactual pareada** em `256`, `1.024`, `4.096` e `8.192` bytes com `32` escritas, e **`99,51%` original / `99,12%` pareada** no teste extremo de `64` escritas em `8.192` bytes.
4. **Decodificação incremental $O(1)$ em memória e ~8× mais rápida:**
   - Após um prefixo de `7.936` bytes, o passo compilado via `torch.compile(backend="inductor")` + `CUDAGraph` do `fractal_golden_big` gerou a **`1.861,6` bytes/s** com estado fixo de **`422.916` bytes** (`~413 KiB`), contra **`231,7` bytes/s** e **`1.572.872` bytes** (`~1,50 MiB`) do `transformer_big` com janela local de 256 posições (paridade numérica máxima de `1,48e-5`).

---

## 2. Histórico Completo das Campanhas IA2 (001 a 006)

Todas as campanhas foram executadas em hardware real (2× NVIDIA Tesla T4 no Kaggle, PyTorch `2.11.0+cu130`, Python `3.13.16`, precisão `FP32`) e os artefatos completos estão preservados nos pacotes `llm2-ia2-campaign-001-results.zip` a `006` na raiz do workspace.

### 2.1 Evolução de `001` a `005` (Escala `~86k–190k` params, `story_seq_len=256`)

| Campanha / Braço | Params | Estado / seq | TinyStories `@256` ↓ | TinyStories `@3219` ↓ | Binding `@8192 (32w)` pareado ↑ | Decode `@7936` (`Inductor`) ↑ | Marco técnico da rodada |
|---|---:|---:|---:|---:|---:|---:|---|
| **001** `transformer_1x` | 86.400 | KV local | 1,0758 | 2,4349 | 0,0% | 607,3 B/s | Baseline 1× |
| **001** `transformer_2x` | 190.080 | KV local | 0,9392 | 2,7331 | 0,0% | 439,9 B/s | Baseline 2× |
| **001** `context_shared2` | 85.848 | Recorrente | 1,0431 | 1,0308 | 0,0% | 3.999,3 B/s | Referência recorrente 0033 |
| **001** `fractal_delta_med` | 150.320 | 48.256 B | 0,9455 | 0,9256 | 0,0% | 2.431,5 B/s | 1º protótipo `FractalDeltaMemory` + `CUDAGraph` |
| **002** `fractal_delta_1x` | 91.881 | 50.560 B | 1,0543 | 1,0598 | **100,0%** | 2.497,6 B/s | Canal de indução associativa lag-1 resolve Binding 8K |
| **002** `fractal_delta_med` | 154.377 | 75.712 B | 0,9535 | 0,9497 | **100,0%** | 2.793,6 B/s | Retenção 100% em 8K + linguagem competitiva |
| **003** `fractal_delta_1x` | 94.101 | 61.824 B | 1,0254 | 1,0275 | **100,0%** | 1.841,3 B/s | Estado rank-40 (2 cabeças) + `dplr_chunked` 5D escalar |
| **003** `fractal_delta_med` | 154.773 | 98.496 B | 0,9442 | 0,9386 | **100,0%** | 1.823,8 B/s | Redução de gap para `transformer_2x` |
| **004** `fractal_hybrid_med` | 154.173 | 111.516 B | 0,9625 | 0,9583 | **100,0%** | 1.779,0 B/s | Híbrido 4L + janela causal curta (perdeu para 6L puro) |
| **004** `fractal_pure_med` | 153.751 | 144.256 B | **0,9362** | **0,9322** | **100,0%** | 1.508,8 B/s | 6 níveis + `value_residual`: **supera `transformer_2x` (190k)** |
| **005** `fractal_pure_med` | 153.751 | 144.256 B | 0,9256 | 0,9214 | **100,0%** | 1.197,2 B/s | Controle 6L puro na bateria multi-GPU v7 |
| **005** `fractal_golden_med` | 153.763 | 144.260 B | **0,9179** | **0,9126** | **100,0%** | 1.182,5 B/s | **Fase Áurea ($\phi$): melhor NLL curta e longa da série** |
| **005** `fractal_free_energy_med` | 153.969 | 144.256 B | 0,9372 | 0,9315 | **100,0%** | 1.366,2 B/s | Gating preditivo por erro/surpresa (sem ganho sobre puro) |
| **005** `fractal_quat_med` | 153.357 | 172.544 B | 0,9404 | 0,9391 | **100,0%** | 1.203,9 B/s | Projeções em Álgebra de Quatérnios ($\mathbb{H}$) |
| **005** `fractal_ternary_med` | 153.751 | 144.256 B | 1,1788 | 1,1878 | **100,0%** | 1.206,9 B/s | Pesos ternários $\{-1,0,+1\}$ STE (degradação severa em linguagem) |

---

### 2.2 Confirmação Final Escalada — Campanha 006 (`dd6797d`, `story_seq_len=512`, `binding_hard_writes=64`)

Na **Campanha 006** (`61,8 min` nas duas T4), escalamos os melhores fractais em ~5,6× (`~865k` parâmetros, `d_model=256`, `hidden_dim=544`, `memory_dim=96`, `chunk_size=64`), substituímos os `einsum` de `dplr_chunked` por multiplicações matriciais `@`, dobramos o comprimento de sequência de treino no TinyStories para `512` bytes e adicionamos uma avaliação de saturação com `64` escritas em `8.192` bytes contra um Transformer de 6 camadas e `1.091.328` parâmetros:

| Braço (Campanha 006) | Params | Estado / seq | Story `@256` ↓ | Story `@512` ↓ | Story `@3219` ↓ | Bind `8k (32w)` orig / par ↑ | Bind `8k (64w)` orig / par ↑ | Decode `@7936` (`Inductor`) ↑ | Treino Story (wall / VRAM) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **`transformer_big`** (6L) | 1.091.328 | 1.572.872 B *(KV 256)* | **0,7680** | **0,7400** | 1,4256 | 2,5% / 0,0% | 1,1% / 0,0% | 231,7 B/s *(cg=False)* | 306,5s / 555,7 MB |
| **`fractal_golden_big`** (6L) | 864.869 | 422.916 B | 0,7707 | 0,7437 | **0,7282** | **100,0% / 100,0%** | 99,51% / 99,12% | 1.861,6 B/s *(cg=True)* | 1.095,3s / 2,47 GB |
| **`fractal_pure_big`** (6L) | 864.845 | 422.912 B | 0,7791 | 0,7517 | 0,7359 | **100,0% / 100,0%** | 99,80% / 99,61% | 1.926,2 B/s *(cg=True)* | 1.156,2s / 2,52 GB |
| **`fractal_delta_big`** (4L) | 861.225 | 285.696 B | 0,7822 | 0,7543 | 0,7375 | **100,0% / 100,0%** | **100,0% / 99,90%** | **2.474,8 B/s** *(cg=True)* | 751,2s / 1,65 GB |

#### Amostras reais geradas na Campanha 006 (greedy, 96 bytes após 6.000 passos byte-level)

- **`transformer_big`:**
  - `"Once upon a time, a little "` → `girl named Lily. She loved to play with her toys and play with her toys. One day, she went to th`
  - `"One day, Lily found a shiny "` → `shiny shirt in the park. She was so excited to see it and she started to cry. \nLily was so happy`
- **`fractal_golden_big`:**
  - `"Once upon a time, a little "` → `girl named Lily was so excited to see her favorite toy. She was so happy to see her favorite toy`
  - `"One day, Lily found a shiny "` → `thing that was not a toy. She wanted to see what was inside. She said, "I want to play with you `
- **`fractal_pure_big`:**
  - `"Once upon a time, a little "` → `girl came to the park. She saw the bird and the birds and the birds. She was so happy that she s`
  - `"One day, Lily found a shiny "` → `box of things that was not a bit scary. She wanted to see the bird in the sky. She wanted to see`
- **`fractal_delta_big`:**
  - `"Once upon a time, a little "` → `girl named Lily. She loved to play with her toys and they were so happy to see her friends. One `
  - `"One day, Lily found a shiny "` → `star on the ground. She was so excited to see what was inside. She was so excited to see what wa`

---

## 3. Triagem e Filtro das Tecnologias Condicionais para a Nova Fase (Foco no `fractal_golden`)

Com o encerramento da disputa entre famílias arquiteturais, filtramos o catálogo de tecnologias condicionais (Seção 5 do `plano.md` e `dossi_de_pesquisa_e_arquitetura_p_s_transformer_llm_2.md`) em **quatro categorias práticas** para guiar exclusivamente a evolução do `fractal_golden`:

### 3.1 Tecnologias já incorporadas no núcleo do `fractal_golden` (Ativas)
- **Recorrência matricial DPLR / Delta Rule em chunks (`state_kernels.py`):** substitui o KV cache por estado recorrente $O(1)$ com atualização paralela por blocos (`chunk_size=64`) e multiplicações matriciais `@`.
- **Auto-similaridade Fractal com Adaptadores 1D por escala + Resíduo de Valor ($v_0$):** uma célula recorrente compartilhada entre 6 níveis com modulação afim leve (`FractalLevelAdapter`), passo de Euler aprendido e interpolação com o valor $v_0$ da primeira escala (`fractal_value_residual=True`).
- **Modulação de Fase Quase-Periódica Áurea ($\phi$, `fractal_golden_phase=True`):** frequências angulares espaçadas por potências da proporção áurea $\phi = \frac{1+\sqrt{5}}{2}$ com escala aprendida por cabeça/nível, eliminando colisões periódicas e sustentando extrapolação em $6,3\times$ o comprimento de treino (`0,7282` NLL em `3.219` bytes).
- **Canal de Indução Associativa Lag-1 (`_run_assoc`):** memória associativa dedicada de endereçamento normalizado para captura imediata de pares chave-valor ao longo de `8.192` posições.
- **Compilação `Inductor` + `CUDAGraph` no passo incremental:** decodificação com zero overhead de lançamento de kernels Python.

### 3.2 Tecnologias que serão abordadas daqui para frente no `fractal_golden` (Fila Filtrada)

| Prioridade | Tecnologia / Frente | Como e quando entra na melhoria do `fractal_golden` |
|---|---|---|
| **1 — Imediata (Engenharia & Escala no Colab L4/A100)** | **AMP (`BF16`), `torch.compile`, Tokenização BPE e Descompartilhamento Seletivo (`share_scales`)** | Ativar `BF16` nativo em GPUs L4/A100 (mantendo estado/acumuladores DPLR em FP32) para cortar os ~2,47 GB de VRAM e ~18 min de treino; sair de byte-level (`vocab=257`) para **BPE** (`vocab=8192+`); e testar compartilhamento em grupos de níveis (ex.: 2 ou 3 células compartilhadas em 6–12 níveis) para destravar capacidade em dezenas de milhões de parâmetros. |
| **2 — Curto prazo (Arquitetura & Contexto)** | **Currículo de Contexto, Multi-Token Prediction (MTP) e Calibração do Canal Associativo para BPE** | Usar a infraestrutura já existente em `src/llm2/curriculum/` e `src/llm2/objectives.py` para treinar com janelas progressivas (ex.: $512 \to 2048$) e previsão multi-token, além de dimensionar `memory_dim` / `assoc_val_dim` para vocabulário BPE. |
| **3 — Médio prazo (Computação Adaptativa)** | **Profundidade Variável (RINS) $\to$ ACT / PonderNet** | Como o `fractal_golden` aplica a mesma regra através de níveis compatíveis, ele permite treinar com número variável de níveis/refinamentos ($K$ dinâmico) e, uma vez demonstrado ganho em exemplos difíceis, adicionar parada adaptativa (*halting* aprendido via ACT/PonderNet). |
| **4 — Condicional a Gargalo de Escala** | **Controle Espectral / nGPT, Kronecker/Monarch e Quantização Pós-Treino / Destilação** | **Controle espectral/nGPT:** acionar apenas se surgirem instabilidades de norma em profundidades grandes ($L \ge 12$) ou em `BF16`. **Kronecker/Monarch, Quantização (INT8/INT4/ternário pós-treino) e Destilação:** aplicar nas camadas densas `SwiGLU` / `lm_head` apenas quando o modelo atingir escala maior ($\ge 30\text{M}–100\text{M}$ parâmetros), separando ganhos de compressão dos ganhos arquiteturais. |

### 3.3 Tecnologias já testadas na bancada IA2 e descartadas/pausadas neste regime
- **Pesos Ternários STE (`BitNet b1.58`, `fractal_ternary_med`):** quando aplicados diretamente ao bloco compartilhado pequeno, degradaram a perda de linguagem de `0,9256` para `1,1788` NLL. Ficam suspensos no treinamento de modelos compactos compartilhados (só retornam como quantização/QAT em modelos grandes não-compartilhados).
- **Álgebra de Quatérnios (`fractal_quat_med`):** funcionou e preservou `100%` no binding, mas perdeu em qualidade de linguagem (`0,9404` vs `0,9179`) e velocidade para o `fractal_golden` denso.
- **Gating por Energia Livre / Erro Preditivo (`fractal_free_energy_med`):** adicionou parâmetros e sub-rede preditora entre níveis, mas obteve `0,9372` NLL (inferior ao `fractal_pure_med` `0,9256` e ao `fractal_golden_med` `0,9179`).
- **Híbrido raso 4L com janela local (`fractal_hybrid_med`):** trocar 2 níveis fractais por uma janela de atenção local em `154k` parâmetros piorou a NLL (`0,9625` vs `0,9362`). Qualquer reavaliação futura de janela local só ocorrerá como complemento em modelos maiores sem reduzir a profundidade fractal.

### 3.4 Tecnologias excluídas do escopo do `fractal_golden` (Linhas separadas ou baixa prioridade)
- **DEQ (Deep Equilibrium Models):** solvers de ponto fixo e diferenciação implícita trazem alto custo de iteração e instabilidade numérica desnecessários frente aos 6 níveis explícitos com passo de Euler do `fractal_golden`.
- **Disco de Poincaré (Geometria Hiperbólica) e MERA / AdS-CFT:** a combinação de recorrência DPLR multiescala + fase áurea $\phi$ já entrega hierarquia e extrapolação sem o custo numérico e a fragilidade de gradientes de projeções hiperbólicas (`arcosh`).
- **KAN / Splines, Liquid / CfC / LFM2, SNNs, NCA (Regra 110) e L-systems:** baixa prioridade para modelagem de linguagem causal em GPU; não entram no roadmap do `fractal_golden`.
- **FiLM, RMS de contexto, Ciclo-V, NCM de protótipos fixos e JEPA:** permanecem preservados e desligados no repositório histórico (já avaliados nas campanhas 0015–0023 sem vantagem que justifique ativação).
- **Difusão discreta, Flow Matching, Geração em Árvore e Inferência Ativa:** constituem outros paradigmas de geração/agentes e permanecem fora do pipeline autoregressivo do `fractal_golden`.
