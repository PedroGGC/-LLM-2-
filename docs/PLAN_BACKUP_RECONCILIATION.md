# Reconciliação de inventário na recompressão — 30/10/2026

Pedido: resolver o bloqueio de substituição do ZIP 7,56 GB. A mensagem recebida
não lista diferenças; não há evidência remota para afirmar quais arquivos mudaram.
O comparador anterior exigia igualdade de nomes e mtime, bloqueando adições e
mudanças apenas de timestamps mesmo quando todos os bytes antigos sobreviviam.

1. Comparar identidade por kind/tamanho/SHA ou destino do link. Distinguir adições,
   ausências, bytes/tipos alterados e timestamps; imprimir contagens e amostras.
2. Adições serão incluídas no backup novo; timestamps isolados não indicam perda
   de conteúdo. Essa regra é da substituição de backup, não de retomada científica.
3. Se um payload selecionado do ZIP antigo estiver ausente/diferente, recuperar
   sua versão antiga em diretório exclusivo no working, sem sobrescrever originais,
   e verificar hashes antes de remover o ZIP. Preservar o manifesto antigo e mapa
   de versões. Essa cópia integra o backup comprimido seguinte.
4. Exigir espaço para recuperação antes da exclusão e para nova saída depois;
   abortar com diagnóstico se insuficiente. ZIP antigo íntegro, ausência de workers,
   identidade do alvo e confinamento continuam obrigatórios. Conferir novamente o
   inventário corrente antes da exclusão; falhas preservam o ZIP e os originais.
5. A limpeza completa exige SHA local do novo ZIP, mesmos caminhos/tipos/bytes e
   workers ausentes; timestamps isolados não alteram conteúdo. Adições, ausências
   e alterações reais continuam bloqueando. Nada muda em treino/modelo/protocolo.
6. CRC e SHA em uma leitura completa de cada membro, evitando descompressão
   duplicada. Conferir também hashes dos destinos recuperados antes da exclusão.

Aceitação: adições e timestamp aceitos; mudanças/ausências preservadas em versões
separadas; retomada após recompressão falhar; recusa por espaço, corrupção, worker,
links e mudanças concorrentes. Expor o diff real em erro. Testar CRC/SHA e limpeza
somente em fixtures. Entrega: primeira célula atualizada, segundo script inalterado.

Limites: causa remota só será conhecida ao executar o comparador. Falta de espaço
para preservar conteúdo único continua bloqueando; não descartar versões para
passar. ZIPs/dados continuam fora do Git e nenhuma ação remota será executada aqui.
