# Campanha 0005 — FLRM Ciclo-V Assimétrico, Fusão JIT e NCM, 27/10/2026

## Plano

- Implementar recursão assimétrica Ciclo-V (`v_cycle=True`) em `FractalRecursiveBlock`:
  na iteração `k=1` (`step_index == 0`), `SharedFractalCore` processa as três escalas
  (`T=256`, `T/4=64`, `T/16=16`); nas iterações `k>1` (`step_index > 0`), o núcleo
  denso processa apenas as escalas macro comprimidas (`T/4` e `T/16` agrupadas),
  enquanto a escala fina `T=256` preserva seu estado de alta resolução e recebe a
  modulação causal refinada via upsampling e `CrossScaleInteraction`.
- Implementar fusão TorchScript (`@torch.jit.script`, controlada por `fused_film=True`)
  para as operações pointwise críticas de condicionamento FiLM, normalização RMSNorm
  em FP32, rotação RoPE em FP32, ativação SwiGLU e mistura entre escalas, reduzindo a
  fragmentação de lançamentos de kernels CUDA sem alterar a precisão matemática.
- Introduzir o módulo `NeuralContextMemory` (`src/llm2/architectures/ncm.py`) com
  `ncm_slots=16` protótipos semânticos aprendíveis em `R^D` (`6.144` parâmetros em
  `D=384`), consultados de forma estritamente causal pelas escalas macro comprimidas
  (`T/4` e `T/16`).
- Atualizar ` fractal_parameter_formula` e `FractalLatentModel.analytical_cost` para
  refletir a contagem exata de parâmetros (`2.174.220` com NCM de 16 slots) e o custo
  analítico reduzido do Ciclo-V e do NCM.
- Adicionar os aliases de campanha `flrm_vcycle_k2` e `flrm_vcycle_ncm_k2` em
  `src/llm2/research/campaign.py` e `src/llm2/research/presentation.py`.
- Executar inicialmente a **triagem rápida de perfil** (10 passos de warmup + 30
  passos medidos + 1 passo com trace) na Tesla T4 com lote `32x1` para verificar a
  redução de kernels CUDA e tempo por passo antes do treino de 2.000 passos.

## Riscos e limites

- O Ciclo-V reduz a capacidade computacional bruta da segunda iteração sobre a escala
  fina `T=256`; a qualidade final em loss precisa ser confirmada em treino real após a
  triagem de velocidade.
- Os 16 slots do NCM acrescentam `+0,28%` de parâmetros ao backbone (`6.144` pesos).
  A consulta opera posição a posição contra vetores globais aprendidos, preservando
  isolamento causal estrito entre posições temporais e entre exemplos do lote.
- O compilador `@torch.jit.script` funde cadeias element-wise em CUDA, mas o número
  exato de kernels lançados depende do fuser do PyTorch CUDA na Tesla T4; a verificação
  empírica definitiva ocorre no `trace.json` da Campanha 0005 no Kaggle.

## Aceitação

- `ResearchConfig` valida `v_cycle`, `ncm_slots` e `fused_film` mantendo os valores
  antigos (`False`, `0`, `False`) como padrão para não alterar modelos existentes.
- Enumeração de parâmetros coincide exatamente com `fractal_parameter_formula` para
  `flrm_vcycle_k2` (`2.168.076`) e `flrm_vcycle_ncm_k2` (`2.174.220`).
- Testes locais aprovados em CPU: causalidade adversarial em todas as fronteiras de
  bloco, finitude de forward/backward, memorização e equivalência estrita de retomada
  de checkpoint (`resume-equivalence`).
- Revisão publicada na branch `experiment/fractal-lrm` e célula Kaggle entregue com
  `run_number: 5` no modo de triagem rápida (`perfilar_lotes` em `32x1`).
