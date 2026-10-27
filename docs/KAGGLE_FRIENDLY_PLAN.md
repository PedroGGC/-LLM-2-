# Notebook Kaggle em português — plano, riscos e aceitação

Solicitação de 26/10/2026: explicar quando maior/menor é melhor e tornar o script tão legível quanto o exemplo enviado, mantendo as garantias da campanha atual.

## Entrega

1. Apresentação de terminal: etapas numeradas, nome do modelo/GPU/seed, progresso percentual, métricas com unidades e direção, indicação explícita de validação parcial/completa, geração visível e tabela final em português. Logs técnicos integrais continuam em arquivos.
2. Modo `avaliar_salvos`: comparar checkpoints intermediários da campanha existente, sem disparar treino. Usar o código original verificado por hashes para avaliar os pesos antigos; permitir caminho restaurado dos dados com o mesmo fingerprint. Saídas novas e ZIP próprio, preservando o experimento original.
3. Bootstrap com duas escolhas explícitas (`avaliar_salvos` e `treinar`), mensagens de hardware/etapas/erros e célula simples. Publicar código comum e integrá-lo na branch fractal; nenhuma alteração de modelo/objetivo/otimização.

## Riscos e limites

- Interface não pode anunciar sucesso após subprocesso falhar, confundir loss parcial com completa ou dar ETA garantido. Não exibir utilização GPU inventada.
- Menos parâmetros/VRAM significa menor custo de recurso, não necessariamente maior qualidade. Loss de treino com JEPA não é a mesma métrica de qualidade que loss LM de validação.
- O ZIP antigo não contém os pesos. Se faltarem, interromper com instrução clara; nunca iniciar novo treino como fallback.
- Avaliar checkpoints com código alterado mudaria o experimento. Executar a CLI do checkout original, depois de conferir hashes fonte, dados e pesos; não relaxar contratos de retomada nem carregar pickle na análise.
- Seleção por validação não é um teste independente. Não comparar velocidades de checkpoints que só foram reavaliados quanto à qualidade.
- Sem GPU local. Testes da apresentação e uma integração pequena de avaliação em CPU verificam o fluxo; o novo modo em T4 só terá validação empírica após rodar no Kaggle.

## Aceitação

- Progresso e tabelas identificam `menor = melhor` para loss e VRAM; `maior = melhor` para velocidade. Ausência de medição aparece como `n/d`, nunca zero.
- Falha preserva logs/ZIP parcial, não imprime conclusão de sucesso e não seleciona checkpoint antigo para substituir treino falho.
- Modo de revisão executa somente avaliação de checkpoints pedidos e reutiliza avaliação completa existente apenas após conferir sua identidade e escopo.
- Regressões cobrem avaliação real CPU sem novo treino, avaliação reutilizada, ausência/corrupção de pesos, incompatibilidade de código/dados e diretórios imutáveis. Suítes relevantes passam e skips CUDA são registrados.
- Célula entregue com commit fixado; scripts têm sintaxe válida e o bootstrap publicado corresponde ao objeto Git. ZIP do usuário e auditoria bruta ficam fora do push.
