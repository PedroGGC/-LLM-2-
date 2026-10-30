# Correção da descoberta de ZIPs — 30/10/2026

## Incidente e objetivo

Na revisão `42d9179`, a descoberta automática tentou abrir um candidato a histórico
que não era um ZIP legível e propagou `BadZipFile` antes de iniciar o lote. O traceback
não identifica qual arquivo. Não atribuir a corrupção a um ZIP específico sem evidência.
O número 0020 pode ter sido reservado antes da falha; preservar tudo e usar 0021.

## Plano e critérios anteriores às alterações

1. Reproduzir ZIP vazio/truncado/texto e relatório inválido entre arquivos válidos.
2. Descoberta relata caminho e motivo, com registro em metadados. Não modifica nem
   apaga arquivos. Continua com histórico válido quando o candidato ilegível pertence
   às tentativas até 0019, já cobertas pela auditoria/história dos protocolos anteriores.
3. Não tratar um ZIP novo ilegível como zero resultados. Para tentativas >=0020,
   exigir outro snapshot válido da mesma tentativa cobrindo todo o lote planejado;
   caso contrário, parar com mensagem que identifica o arquivo e como recuperar seu
   histórico. Isso impede repetição silenciosa de trabalho desconhecido.
4. Fazer descoberta antes de instalar dependências ou reservar novo número. Não
   afrouxar comparação científica de código/configs/dataset/runtime/pesos no operador.
5. Testar mistura de históricos, duplicatas, relatórios excessivos/inválidos,
   snapshots cobrindo lote e histórico recente irrecuperável. Executar regressões
   relevantes e atualizar VALIDATION.md. Código científico continua congelado.
6. Publicar novo SHA e gerar célula curta com número 21, teto de uma hora e dois jobs.

## Riscos e limites

Ignorar sem registro qualquer arquivo ilegível poderia repetir treinos recentes.
A recuperação é explicitamente limitada: evidência auditada para os antigos e
snapshot completo verificável para os novos. Sem esse histórico, não inventar
resultados nem assumir que nada terminou. Nenhum teste de GPU é necessário para
validar leitura de ZIP; isso não demonstra execução empírica do operador na T4.
