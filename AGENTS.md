# Instruções para manter o LLM2-Lab

## Escopo científico

Esta entrega é a etapa 1 de infraestrutura. Manter o Transformer causal como baseline.
Não adicionar JEPA, TRM/Coconut, NCM, MoE, servidores, aplicações web ou interfaces gráficas
sem uma etapa explicitamente solicitada. Interfaces para extensões não autorizam implementá-las.

## Trabalho e testes

- Escrever o plano, riscos e critérios de aceitação antes de mudanças substanciais.
- Implementar em etapas e executar testes relevantes em cada etapa; atualizar VALIDATION.md
  com resultados reais e limitações. Não relatar testes pulados como aprovados.
- Usar o Python local e instalação direta solicitada pelo usuário, sem criar venv por padrão.
  Executar comandos pelo mesmo interpretador que recebeu as dependências.
- Comando principal: `python -m pytest -q`. GPU: `python -m pytest -q -m cuda`.
- Não delegar a outros agentes sem solicitação explícita do usuário.
- Preservar separação entre backbone, AttentionKernel, Objective, otimização, Trainer e avaliação.
  Novos objetivos devem controlar a chamada ao backbone e declarar contagem de tokens.
- Usar factories do Trainer para inicializar extensões depois da semente. Colocar código
  experimental em `src/llm2/` para incluí-lo em snapshots e hashes de proveniência.

## Invariantes de correção

- Não contar embeddings compartilhados duas vezes. Conferir contagem por enumeração e fórmula.
- Não remover causalidade, RoPE em FP32, acumulação FP32 no RMSNorm ou loss em FP32.
- SDPA deve receber dropout zero no modo de avaliação.
- Alterações no treino exigem causalidade, forward/backward, finitude, memorização,
  estado de checkpoint e comparação da próxima atualização retomada com a ininterrupta.
- Testes de equivalência estrita: CPU FP32, SDPA math, mesmo ambiente, configuração e código,
  dropout ativo para exercitar RNG e acumulação de gradientes ativa.
- Checkpoints só em limites completos de optimizer-step, após scheduler e zero_grad.
  Preservar objetivo, otimizador, scheduler, RNGs, sampler e contadores.
- Não enfraquecer verificação de dataset, código, configuração ou versões silenciosamente
  para fazer um checkpoint incompatível carregar. Transferência de pesos é outro experimento.
- Não sobrescrever datasets preparados, checkpoints ou execuções existentes.
- Não treinar tokenizadores com validação. Split por documento e deduplicação antecedem BPE.

## Métricas e reprodutibilidade

- Identificar FLOPs como estimativas e documentar fórmula, convenção e operações excluídas.
  Backbones/objetivos desconhecidos recebem estimativa nula até fornecerem estimador próprio.
- CPU deve usar null para VRAM e tempo de GPU, e não zero que simule medição.
- Tokens processados são alvos de treino com repetição; tokens de validação são separados.
- Registrar configurações resolvidas, versões, hashes, código e decisão de backend.
- Não fazer alegações de eficiência, qualidade ou ganhos sem benchmarks comparáveis.
- Validar CUDA/BF16/Inductor na GPU real antes de alegar suporte empiricamente verificado.
- Estados de treino PyTorch usam pickle: carregar apenas checkpoints confiáveis; pesos são
  safetensors com `save_model`/`load_model` para preservar compartilhamento de parâmetros.

