> **Registro histórico — não é o plano vigente.** Arquivado em 28/10/2026.
> O [estado atual](../PROJECT_STATUS.md), os [resultados corrigidos da 0015](../results/CAMPAIGN_0015.md)
> e a [validação vigente](../../VALIDATION.md) prevalecem sobre descrições e conclusões antigas.

# Etapa 1 — FLRM comprimido e Transformer (27/10/2026)

## Plano e escopo

1. Preservar `compressed` como controle. Adicionar `compressed_native`: aplicar
   pooling antes das projeções das duas saídas grossas da interação. A projeção
   linear comuta com a média; manter o atraso causal e os mesmos parâmetros.
2. Adicionar `compressed_grouped` como ablação opcional: mesma interação nativa,
   mas agrupar apenas as duas escalas curtas, com padding à direita. A escala
   fina continua separada. Não prometer aceleração antes das T4.
3. Corrigir o estimador das três projeções de interação do controle e selecionar
   todos os modelos solicitados para perfis isolados, uma seed por variante.
   Expor aquecimento e amostragem no notebook. Padrão: Transformer e FLRM.
4. Cobrir equivalência de saídas/gradientes, limites causais e caudas parciais,
   memorização, retomada exata com dropout/EMA/acumulação e teste CUDA específico.
5. Publicar código/testes/documentação na branch experimental e fornecer célula
   Kaggle fixada no commit. Novas execuções, sem alterar checkpoints antigos.

## Riscos

- Trocar a ordem de médias e produtos introduz arredondamento FP32; equivalência
  entre variantes usa tolerância, não identidade bit a bit.
- Agrupamento muda consumo de RNG com dropout e acrescenta padding: trajetórias
  de treino podem divergir; retomada dentro de cada variante deve ser exata.
- Kernel launches/pooling podem dominar em modelos pequenos. FLOPs são estimativas
  de multiplicações densas/atenção, excluem operações elementares e não predizem tempo.
- O perfil de treino começa de inicialização nova, não altera os checkpoints
  medidos na avaliação. Geração permanece sem cache nesta etapa.
- As T4 só estão disponíveis no Kaggle; verificações locais são CPU.

## Aceitação

- Mesmo conjunto de pesos e contagem de parâmetros nas três variantes.
- Saídas e gradientes FP32/math SDPA próximos em K=1/2, comprimentos com e sem
  blocos completos; nenhuma influência de tokens futuros nas saídas passadas.
- Aprender minibatch e retomar a próxima atualização exatamente, incluindo
  objetivo/otimizador/scheduler/sampler/RNG/contadores, com dropout e acumulação.
- Testes locais relevantes e suíte completa; CUDA pulado localmente explicitado.
- Kaggle deve medir loss completa (menor = melhor), alvos/s e geração tok/s
  (maior = melhor), tempo (menor = melhor) e VRAM (menor consumo na mesma tarefa).
- Não exigir vitória de desempenho como resultado de testes CPU; escolher o
  melhor modo somente depois das medições reais.
