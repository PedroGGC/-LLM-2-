# Etapa B: quatro controles, ciência congelada e cache por GPU

31/10/2026. Escopo autorizado: preparar próxima execução após a0025 e entregar
célula Kaggle curta. Não adicionar mecanismos ao backbone.

## Plano de implementação

1. Manter o worker científico na revisão qualificada
   `1c8ce2abb8f47d3167454b3503598854b7756dad`. Outra revisão conterá apenas a nova
   orquestração operacional; registrar ambos os hashes e snapshots. Nenhum hash
   de qualificação será reescrito. Validar fonte e ambiente pelo validador original.
2. Preservar o `platforms.json` original da0025 em documentação versionada para
   permitir nova sessão após limpar working, sem baixar pesos ou repetir seleção.
3. Reutilizar configuração, treino, avaliação, dados, checkpoints e relatórios do
   worker qualificado. A extensão operacional será explícita: subclass de campanha
   e ordenação de tarefas; não substituir matemática, otimizador ou sampler.
4. Uma família de cache por GPU. Conservar entre comandos consecutivos da mesma
   variante; descartar antes de trocar variante, somente quando aquela GPU estiver
   inativa. Outra GPU pode continuar treinando. Locks impedem limpar cache em uso.
5. Executar pilotos isolados nas duas GPUs por variante e, logo após, suas duas
   taxas de triagem no cache quente. Tuning continua750passos, seed901, apenas
   documentos do treino. Piloto concorrente mantém controle de contenção.
6. Primeira sessão: quatro braços da seed241, cada um com6000passos. Duas lanes
   independentes. Após cada par encerrar, medir perfis isolados: primeiro onde o
   cache está quente, depois a GPU cruzada. Registrar recompilação no custo.
7. Margem35%, previsão incluindo compilação fria nos perfis, reserva5min para
   exportação e2GiB livres. Meta2h/teto3h; não reduzir horizonte para caber. Se
   nenhum bloco couber, exportar pilotos/tuning e pendências, sem treinos parciais
   disfarçados de concluídos. Continuar seeds242/243 em sessões posteriores.
8. Manter um snapshot intermediário aos50% e apenas o par definitivo após validar
   cobertura. Botão de download embutido no final, inclusive em falha parcial.

## Riscos e critérios de aceitação

- Não alterar nenhum arquivo do worker qualificado nem importar código científico
  da revisão de operação por engano. Provar escolha exata da fonte por teste.
- Limpar cache de GPU0 nunca deve tocar GPU1 ou checkpoints/dados. Links externos
  não serão seguidos. Cache de família anterior só sai após término do grupo.
- Exportações e metadata devem incluir a política operacional e o snapshot do
  operador, além da proveniência científica original.
- Compilação continua podendo ser cara: medir cold/warm; previsões conservadoras,
  sem promessa de executar toda a matriz em duas horas.
- Testar concorrência, isolamento, falha/deadline, espaço, algoritmo de admissão,
  preservação dos contratos e integração real CPU com dropout/acumulação.
- GPU nova pendente: não declarar B validada na T4 com base no smoke CPU.
