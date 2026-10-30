# Correção do backup Kaggle — 30/10/2026

Pedido: substituir o ZIP sem compressão, de aproximadamente 8 GB, por um backup
com compressão mais forte. Não executar limpeza completa nem alterar o diagnóstico.

1. Usar ZIP DEFLATE nível 9 por padrão, preservando CRC, hashes e manifesto.
2. Autorizar substituição somente de um nome explícito de backup do próprio
   utilitário, filho direto de working, regular e sem links. Antes de remover esse
   ZIP, verificar seu conteúdo, os originais contra o inventário, workers e espaço
   estimado depois da remoção. Não remover campanhas, dados ou pesos originais.
3. Repetir uma célula concluída deve reutilizar o novo backup compatível. Não
   embutir automaticamente o ZIP antigo em outro backup nem reutilizá-lo como se
   tivesse sido comprimido. Falha de compressão mantém os originais e elimina
   somente a saída temporária incompleta.
4. Exibir progresso durante operações demoradas e link relativo de download;
   oferecer também a localização pelo painel de arquivos do Kaggle.
5. Entregar somente o primeiro script atualizado, com o ZIP indicado pelo usuário
   preenchido. Preservar byte a byte o segundo script de diagnóstico, fixado na
   revisão 6a6af1a; nenhuma alteração em modelo, treino, parâmetros ou protocolo.

Aceitação: compressão real de dados repetitivos, integridade e limpeza pós-hash
preservadas, recusa de substituição com originais alterados, falta de espaço,
worker ativo, alvo externo/link ou ZIP estranho; repetição sem backup aninhado;
falha simulada sem perda dos originais. Verificar fonte publicada e hashes da
entrega, sem executar ações na sessão remota do usuário.

Limites: pesos e ZIPs existentes podem comprimir pouco. Não prometer tamanho nem
prazo antes de medir. Nível 9 custa CPU; maior compressão não diagnostica sozinha
a causa do download sem resposta. Nenhum dado local do usuário será apagado.
