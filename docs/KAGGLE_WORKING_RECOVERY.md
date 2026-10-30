# Recuperar a 0021 e liberar espaço no Kaggle

O número ocupado significa que outra pasta/reserva já usa o ID. Os ZIPs after-01
confirmam um FLRM 103 concluído e preservado. O script de manutenção não remove
campanhas, datasets, ZIPs, checkpoints nem a numeração.

## Célula de manutenção

Cole `limpar_working_kaggle.py` em uma célula e execute. Ele usa apenas a biblioteca
padrão, sem treinamento ou novas dependências. Mostra o tamanho por pasta e o estado
atual dos jobs da 0021. Workers do projeto são procurados em `/proc`; se houver algum
ativo, a limpeza é desativada. Nenhum processo é encerrado automaticamente.

A remoção automática cobre dois formatos criados pelos launchers:

- `source`/`scientific-worker` dentro de `llm2-cycle2-git-*` ou `llm2-cycle2-code-*`.
  Os metadados/ZIPs do bootstrap ficam no parent.
- Checkouts antigos `llm2-<12 hex>`, somente se os 12 caracteres coincidirem com o
  HEAD do Git. Esse checkout inteiro pode ser removido.

Ambos exigem Git limpo, remoto esperado, common-dir dentro do cache, paths internos
sem links e nenhum payload de dados/pesos ou arquivo ignorado além dos caches Python
conhecidos. Código pode ser baixado novamente pelo commit original. Todas as pastas
`llm2-campaign-*` são preservadas, inclusive nomes antigos em hexadecimal. A manutenção
também reconhece workers executados nos checkouts antigos.

O espaço liberado depende do inventário real. Se os maiores consumidores forem
campanhas antigas, elas continuarão protegidas: verificar backups antes de decidir
remover seus pesos. Os valores `working_before_gib`/`working_after_gib` medem os
arquivos dessa pasta; `filesystem_free_gib` mede o volume e pode diferir da quota do
notebook exibida pelo Kaggle.

## Célula de continuação

Execute o `rodar_ciclo3_kaggle_1h.py` desta entrega **na célula seguinte, em sequência**.
Ele exige `WORKING_INFO["safe_to_start_new_batch"] == True`, verifica novamente os
workers e utiliza o próximo número livre observado (mínimo 0022). O relógio de uma
hora começa na manutenção, incluindo eventual intervalo entre as células; se ficar
muito tempo parado, execute novamente a manutenção antes de continuar.

Se houver aviso de checkpoint parcial ou conclusão ainda não exportada, o treino
não começa. Preserve o estado e forneça o `campaign.json` atual para preparar a
recuperação correspondente. O snapshot after-01 não informa o que aconteceu depois.

A retomada usa até **um treino novo**, seed 103. Com o ZIP after-01 em working ou
Input, o FLRM concluído é pulado e o próximo pendente é selecionado. Se o comparador
também já tiver sido concluído/exportado, ele será pulado. O worker científico e os
contratos permanecem iguais; não há transferência ou flexibilização de checkpoint.

Em sessão nova, anexe o ZIP after-01 de resultados como Input. Preserve também seu
ZIP de checkpoints localmente; ele não precisa ser recarregado para treinar o controle
independente. Novos resultados/checkpoints continuam separados.

[Plano e riscos](PLAN_KAGGLE_STORAGE_RECOVERY.md) ·
[Auditoria do snapshot](results/CAMPAIGN_0021_PARTIAL.md)
