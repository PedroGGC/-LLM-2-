# Recuperação 0021 e espaço no Kaggle — 30/10/2026

## Evidência e limites

O snapshot after-01 prova que FLRM seed 103 concluiu 8.000 passos, quatro avaliações
completas e NLL 1,320334273. Os ZIPs são íntegros; pesos e estado opaco de treinamento
têm SHA conferido. Há 2.165.772 elementos de parâmetros únicos (embedding compartilhado
contado uma vez), além do buffer `training_step=8000`. O histórico soma 13 treinos.
O comparador estava pendente nesse instante; seu estado atual não é conhecido.
O usuário relata que a nova tentativa parou por número ocupado e quer liberar espaço.

## Plano anterior às mudanças

1. Preservar integralmente os ZIPs e a pasta 0021. Número ocupado é uma proteção,
   não autoriza apagar reserva ou sobrescrever a execução.
2. Preparar manutenção padrão Python para o usuário executar no Kaggle: inventário
   por pasta, consumo de disco, processos de treino ativos e estado vivo da 0021.
   Não há acesso direto ao disco remoto nesta conversa.
3. Limpar somente checkouts de código reproduzíveis sob prefixes conhecidos, com
   Git limpo, remoto esperado, common-dir dentro do cache e sem arquivos ignorados
   além de caches Python conhecidos. Recusar links/caminhos externos e payloads de
   dados/pesos. Preservar ZIPs, datasets, campanhas, numeração e metadados de bootstrap.
   Havendo processo ativo, não remover nada. Relatar o que ficou protegido e o tamanho.
4. Avisar sobre checkpoint parcial ou conclusão ainda sem relatório exportado antes
   de indicar nova rodada; não iniciar outro treino nessa situação.
5. Calcular próximo número livre observado em pastas, ZIPs e reservas (mínimo 0022).
   Continuar com histórico after-01 preservado ou anexado como Input, no máximo um
   treino na recuperação e mesmo teto de uma hora. Não repetir FLRM 103.
6. Acrescentar mensagens imediatas à célula de entrega, antes/depois do download do
   bootstrap. Não alterar matemática, horizontes, worker congelado ou retomada.
7. Testar proteção de processos, arquivos, caminhos e payloads; caches Git realmente
   reproduzíveis e cálculo de número. Registrar auditoria e checks reais em VALIDATION.

## Riscos

Apagar campanhas antigas pode perder pesos não exportados. A limpeza automática não
faz isso e pode liberar menos espaço que o desejado; o inventário orienta uma próxima
decisão com backups. O tamanho real das pastas do Kaggle é desconhecido até a célula
ser executada. Nenhum processo será encerrado automaticamente nem pickle carregado.
