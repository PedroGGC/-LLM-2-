# Campanha 015 — Fase B (`Lumina V1 Multimodal`: `SigLIP 2` + `VisionSpatialProjector` + `GRM` `<img_end>` + `FineVision` + `curated-danbooru-2026`) na NVIDIA L4

- **Data:** 22/11/2026
- **Ambiente:** Colab `1× NVIDIA L4` (`PyTorch 2.14.1+cu130`), runtime encerrado imediatamente após coleta (`list_runtimes == {}`), duração total `144.4s`.
- **Encoder visual real:** `google/siglip2-base-patch16-224` congelado em `BF16` (`185.04 MiB` VRAM, `vision_dim=768`, `14×14 = 196` patches → `2×2` *pixel-unshuffle* = `49` tokens visuais + `2` marcadores `<img_start>/<img_end>` = `51` tokens, compressão espacial **`4.0×`**).
- **Streams reais de treino/validação (sem vazamento de val no treino, sem `HF_TOKEN`):**
  - Visão Geral (`70%`): `HuggingFaceM4/FineVision` (`allava_laion`, `120` amostras reais em `15.9s`).
  - Especialização Arte/Ilustração (`30%`): `aipracticecafe/curated-danbooru-2026` (`60` ilustrações AVIF + `prompt` natural em `16.7s`, com `pillow-avif-plugin`).
  - Validação reservada (*held-out*): `20` pares reais (`12` `FineVision` + `8` `curated-danbooru-2026`).

## 1. Gate de Testes
- **CPU local (`tests/test_multimodal_bridge.py` + `tests/test_fractal_v2_grm.py`):** **`11 passed, 2 skipped`** (`26.12s`); pós-fix de schema em `vlm_data.py`: `tests/test_multimodal_bridge.py` **`4 passed, 1 skipped`** (`15.23s`).
- **GPU real (`NVIDIA L4`):** `pytest -q -m cuda tests/test_multimodal_bridge.py` → **`1 passed, 4 deselected`** (`11.73s`, `exit 0`).

## 2. Resultados Reais dos Estágios 1, 2 e 3 (`FractalDelta V2` + `SigLIP 2`)

| Etapa / Métrica | Medição Real na L4 | Invariante Verificada |
| :--- | :---: | :--- |
| **Loss inicial (val reservada `n=20`)** | `5.6215` | Baseline antes de treinar o projetor |
| **Stage 1 (`Projector-Only`, `120` passos, `33.4s`)** | Val `5.6215 → 5.4445` (`2.56M` treináveis / `99.45M` congelados) | `backbone_mutated = False`, `vision_mutated = False` |
| **Stage 2 (`Projector + Prelude + Adapters`, `80` passos, `14.9s`)** | Val **`5.4445 → 3.2423`** (`4.59M` treináveis / `97.43M` congelados) | `core_mutated = False`, `coda_mutated = False` |
| **`QuaRot Triton W4A16` no VLM (`3.554×` compressão)** | Val **`3.2440`** vs `3.2423` BF16 (**`+0.0017` NLL**) | Quantização INT4 preservada em entradas multimodais |
| **Estado $O(1)$ (`1` imagem vs `8` frames GIF vs `+512` tok)** | **`1.038.848 B` == `1.038.848 B` == `1.038.848 B`** (`~1.0 MB`) | `o1_constant_verified = True` |
| **VRAM pico total (`SigLIP 2` + Projetor + `FractalDelta V2`)** | `461.22 MiB` (`1` img) / **`471.51 MiB` (`8` frames GIF)** | Crescimento de apenas `+10.3 MiB` para 8 âncoras `GRM` |
| **Sobrevivência de âncoras `<img_end>` no `GRM` (`+512` tok)** | `img_0 → Top-1 "img_0"`, `img_5 → Top-1 "img_5"` | `8` âncoras intactas e finitas após `512` tokens gerados |

## 3. Limitações Honestas
- Este gate validou a ponte multimodal real em `FractalDelta V2` (`d_model=384`, `vocab=257` byte-level, sem pré-treino longo de texto), provando fluxo de gradiente, congelamento estrito por estágio, compressão `4×`, estado $O(1)$ em 8 frames GIF, quantização `QuaRot INT4` (`+0.0017` NLL) e persistência de âncoras `<img_end>` no `GRM`.
- Capacidade real de descrição visual detalhada e VQA/OCR só emerge quando esta mesma ponte for acoplada ao backbone escalado (`~1.5B–3B` em loop, **Fase C**) e treinada em milhares/dezenas de milhares de passos.
