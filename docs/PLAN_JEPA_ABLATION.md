# Comparação JEPA ligado/desligado — 27/10/2026

## Plano

- Adicionar o identificador explícito `flrm_compressed_grouped_no_jepa`. Ele usa
  exatamente `compressed_grouped`, com jepa_weight=0; sem teacher/preditor/EMA.
  O identificador existente mantém JEPA .1. Não usar `flrm_no_jepa` histórico,
  pois este seleciona a arquitetura não comprimida.
- Célula/defaults: sem JEPA, com JEPA, Transformer pequeno; K=2 nos FLRMs, seed 42,
  2.000 passos, mesmos dados e hiperparâmetros. Sem currículo, MTP ou novos objetivos.
- Preservar filas paralelas existentes. Nessa ordem a GPU 0 treina sem JEPA e
  depois Transformer; GPU 1 treina com JEPA. É uma distribuição baseada na
  expectativa de menor custo sem teacher, não garantia de tempo igual nas placas.
- Manter avaliação completa, geração e perfis sequenciais na mesma GPU. Perfis
  com 10 warmups/30 passos medidos; incluir os três modelos. Saídas novas e imutáveis.
- Verificar mapeamento/configurações, ausência real de auxiliares, pesos iniciais
  do backbone e sampler iguais para a seed; ampliar retomada CPU e continuação
  CUDA para JEPA desligado. Validar integração minúscula CPU e publicar célula fixada.

## Riscos e limites

- Sem JEPA muda o objetivo e o custo, não só o valor exibido no console. O ganho de
  velocidade/VRAM/qualidade precisa ser medido no Kaggle; não é garantia.
- Construir o preditor consome RNG. Mesma seed garante pesos iniciais do backbone
  e sampler pareados; não promete os mesmos sorteios de dropout ao longo do treino.
- Uma seed é triagem. A referência Transformer tem K=1 e outro backbone; o par
  FLRM é a ablação do JEPA. Igual número de alvos não é igual custo computacional.
- Teacher não é usado na geração; desativá-lo não promete acelerar inferência.
- Não alterar Trainer, algoritmos de backbone, verificação de checkpoints ou
  dados preparados. Retomar arquivos antigos continua exigindo fonte original.

## Aceitação

- Configs dos FLRMs diferem apenas no peso JEPA e diretório de saída; K efetivo 2.
- Sem JEPA constrói CausalLanguageModeling e tem total de parâmetros igual ao
  backbone. Com JEPA constrói PredictiveObjective com target/predictor existentes.
- Retomada exata CPU FP32/math SDPA com dropout/acumulação e demais estados;
  testes CUDA coletados, indisponibilidade local explicitada.
- Três modelos concluídos em integração CPU, com avaliação completa, perfis e ZIP.
- Testes locais reais registrados em VALIDATION; suporte CUDA da nova combinação
  depende da próxima execução T4. Commit publicado e bootstrap remoto conferido.
