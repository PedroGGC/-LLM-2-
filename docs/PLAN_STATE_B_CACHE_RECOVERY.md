# Recuperação do cache bruto antes da etapa B

31/10/2026. Log0026: CPU51passed, CUDA2passed; falha antes dos pilotos em
`pinned shard checksum mismatch: train-00000-of-00002.parquet`.

O importador qualificado verifica um arquivo existente e recusa seu hash; não o
baixa novamente, mesmo com `--download`. O log não informa o hash observado nem
permite concluir a origem da corrupção. Nenhum treino/checkpoint iniciado.

## Plano e riscos antes da alteração

1. Manter ciência1c8ce2a e todos os hashes de corpus/dataset originais.
2. Preflight de cache bruto antes de instalar/executar testes CUDA. Examinar arquivos
   pelo SHA-256, sem ler pickle ou avaliar modelos. Informar esperado/observado/bytes.
3. Conservar cache legado integralmente. Publicar em cache verificado separado:
   reutilizar cópia de cada arquivo válido; baixar somente ausentes/inválidos.
4. Arquivo só entra no cache após checksum correto e publicação exclusiva. Tentativa
   incompleta, HTML ou hash divergente não poderá ser consumido pelo worker.
5. Registrar diagnóstico no bootstrap/campanha. Limitar transferência por shard,
   tentativas, timeout e orçamento global; progresso periódico no notebook.
6. Não modificar datasets preparados, resultados0026, modelos, seleção ou objetivos.
   Próxima campanha começa em27 e avança se houver reserva existente.

## Aceitação

Testar cache legado inválido preservado, reutilização de válidos, downloads com
hash incorreto, falha de rede, publicação concorrente, links e ausência de
sobrescrita. Verificar despacho do cache recuperado ao worker congelado e que a
falha de preflight antecede testes CUDA. Conferir os shards históricos disponíveis
no backup local e, se houver dependências, preparar o corpus real e comparar todos
os hashes do dataset. Nunca trocar o hash esperado pelo observado para liberar execução.

Sem evidência do arquivo atual no Kaggle, origem da divergência permanece desconhecida.
Validação local/fixture não substitui a próxima execução GPU.
