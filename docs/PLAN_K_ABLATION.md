# Próxima campanha — K=1/K=2 sem JEPA, 27/10/2026

## Plano

- Acrescentar aliases explícitos `flrm_compressed_grouped_no_jepa_k1` e `_k2`.
  Ambos selecionam compressed_grouped, somente loss LM, sem teacher/preditor.
  K é fixado pelo alias e prevalece sobre o --iterations global. Aliases antigos
  mantêm seu comportamento; nenhum algoritmo do backbone/Trainer muda.
- Selecionar K1, K2, Transformer pequeno. Com seed 42, a fila da GPU 0 recebe K1
  e depois Transformer; a GPU 1 recebe K2. Avaliação/geração/perfis sequenciais
  na mesma GPU. Fixar no manifesto o K resolvido de cada configuração, não o valor
  global. Transformer declara uma passagem; sua profundidade continua em n_layers.
- Manter dataset, seed 42, FP32, arquitetura e hiperparâmetros iguais, 2.000
  passos e 16.384.000 alvos. Perfis com 10 warmups/30 amostras/trace separado.
- Validar configs pareadas exceto K/output_dir, pesos iniciais e sampler iguais,
  mesma contagem única de parâmetros, estimativa de operações maior com K=2.
  Cobrir memorização/retomada exata e teste CUDA do K1 agrupado sem JEPA.
- Publicar revisão e célula com run_number=2, produzindo campanha-0002-results.zip.
  Não sobrescrever campanha 0001. Documentar testes e limites reais.

## Riscos e limites

- K=1 reduz a recursão e pode piorar qualidade; não prometer velocidade ou loss
  antes da medição T4. Mesmo número de alvos não é igual orçamento computacional.
- Pesos iniciais e sampler são pareados, mas K muda consumo de dropout/RNG e a
  trajetória de treino. Uma seed é triagem. Não interpretar mesmo backbone em
  quantidade como mesmo custo de execução.
- O script antigo com iterations=1 não comparava simultaneamente K1/K2; os aliases
  novos evitam ambiguidade e precisam aparecer separadamente em relatórios/perfis.
- Localmente há CPU apenas. Os testes CUDA da combinação K1 serão coletados,
  mas a validação efetiva depende do Kaggle. Sem mudanças de cache/Fibonacci/DEQ.

## Aceitação

- Dois FLRMs sem auxiliares; configs diferem só em iterations e output_dir.
- K correto em configs, manifesto e labels; ambos os perfis selecionados uma vez.
- Parâmetros iguais por enumeração/fórmula e mesma inicialização/sampler.
- Testes relevantes e suíte completa CPU; skips CUDA explícitos. Integração CPU
  dos três modelos com avaliação completa, geração/perfis/ZIP.
- Commit publicado, conteúdo remoto conferido e célula pronta para T4x2.
