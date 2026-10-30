# Ciclo 3 em lotes de até uma hora — 29/10/2026

## Evidência e mudança solicitada

O usuário escolheu teto de 3.600 s. A campanha 0019 tem 12 treinos completos e 48
avaliações (seeds 101/102). O ZIP registra encerramento `partial_budget` aos 9.645 s,
após dois blocos completos; não registra erro de rede. Treino consumiu 137,37 min,
testes de código 11,05 min e avaliação/geração/diagnóstico 6,48 min. O gargalo
principal foi executar muitos horizontes seriais na mesma sessão.

## Plano anterior à implementação

- Preservar horizontes, dataset, quatro avaliações completas, métricas por janela,
  diagnóstico, geração e perfis. Não encurtar schedulers/treinos já definidos.
- No máximo dois treinos novos por sessão, teto total de uma hora e reserva de
  cinco minutos. Admissão pelo piloto por execução, sem exigir seis braços juntos.
- Priorizar o par primário da seed 103. Sessões seguintes consomem a fila restante
  com resultados anteriores anexados; nunca repetir os 12 treinos completos.
- Usar a revisão científica imutável `8798f6b84714d2c6671d970685984270078a982c`
  em checkout separado para TODOS os workers. O operador novo roda fora desse checkout,
  tem SHA/snapshot próprios e não altera fonte/configs de modelo/treino/dados.
  Assim, as identidades científicas originais continuam iguais e a agregação mantém
  suas verificações estritas de código, dataset, versões e pesos.
- Suite completa fica como regressão local e evidência da 0019 (280 CPU / 25 CUDA
  FP32 aprovados). Na sessão curta rodar seleção explícita de correção/retomada/
  dados em CPU e compilação FP32 das famílias presentes em GPU; registrar seleção
  como focada, nunca como execução da suíte completa. Opção completa permanece.
- Perfilar somente modelos/slots presentes no lote, ainda isolados, com warmup e
  duração de validação medidos. Estimativa anterior, cap de tempo e heartbeat legível.
- Exportar análise e checkpoint após cada treino completo, além dos finais.
  Usar nomes novos; não sobrescrever snapshots, dados ou resultados.
- Curar evidência auditada da 0019 no Git. Agregador de evidências deduplica cópias
  idênticas e rejeita repetições conflitantes; reporta pares disponíveis sem converter
  sessões parciais em confirmação. Mantém a exigência original de cinco blocos completos.

## Aceitação e riscos

Auditar hashes/fontes/configs/contadores e reconstruir 48 NLLs por janelas. Simular
CPU o lote, fila incremental, deadline, backups e preservação da identidade científica.
Testar conflito de pesos/dataset/código/runtime e cópias repetidas. Rodar regressões
relevantes e suíte completa local; CUDA novo permanece pendente da sessão real.

Uma hora por sessão não reduz automaticamente o total necessário para cinco seeds.
O foco muda o empacotamento e evita repetição/overhead de testes alheios; nenhum ganho
de velocidade de kernel é alegado. Pesos não estão no ZIP recebido: auditar seus
hashes nos registros não é rehash dos arquivos ausentes. UUIDs/máscaras/host podem
variar apenas na agregação, conforme contrato existente; retomada continua integral.
