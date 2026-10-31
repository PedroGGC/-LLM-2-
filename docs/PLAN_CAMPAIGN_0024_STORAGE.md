# Campanha 0024: auditoria, downloads e armazenamento

31/10/2026. Escopo autorizado: analisar o ZIP entregue, tornar o download embutido
o padrão do notebook, entregar limpeza do working e reduzir temporários futuros.

## Evidência e decisão

O ZIP (87.636 bytes) é íntegro, contém seis pilotos T4 e nenhum treino/checkpoint.
mLSTM teve os menores tempos, mas todos os logs registram o alarme de 600 segundos
absorvido pelo compilador. O limite por candidato não foi cumprido: não promover
`selected_rule` da 0024 a uma qualificação limpa. Preservar os dados brutos.

## Implementação

1. Registrar auditoria reproduzível, hashes, tempos, limites e conclusão.
2. Gerar botões `data:...;base64` no próprio notebook, com limite de tamanho;
   arquivos maiores terão partes limitadas e instruções de reconstrução verificável.
3. Limpar somente caches conhecidos e checkouts limpos do projeto, verificando raiz,
   processos, links e arquivos científicos. Preservar ZIPs, campanhas, pesos e dados.
4. Buscar apenas a revisão Git necessária, sem histórico completo. Remover o checkout
   descartável ao terminar; preservar metadados/hashes no diretório da campanha.
5. Limpar cache após cada candidato e ao final; limitar cache/espaço em execução.
   Exportar diagnósticos antes de remover temporários. Não reduzir horizontes.
6. Aplicar timeout no supervisor do subprocesso, fora do compilador. Usar formas
   estáticas na qualificação para evitar a generalização dinâmica observada.
7. Requalificar as três regras antes da etapa B. Não migrar silenciosamente a
   proveniência da 0024 para código novo e não alegar ganhos de linguagem.

## Riscos e aceitação

- Nunca remover dados/checkpoints para liberar espaço automaticamente.
- Limpeza bloqueada com workers ativos, checkout alterado, raiz/links externos.
- Remoção de caches implica recompilação futura; contabilizar seu tempo.
- Download embutido consome memória/saída do notebook: limitar cada parte.
- Timeout ou limite de armazenamento não podem produzir candidato aprovado.
- Validar downloads por bytes/hash, falhas/timeout, proteção de dados e links,
  fonte Git exata e exportação parcial. Rodar testes focados e regressão CPU.
- CUDA/Inductor novo precisa de validação na T4; CPU não a substitui.
