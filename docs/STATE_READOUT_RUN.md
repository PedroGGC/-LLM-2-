# Campanha0029 — treino controlado de profundidade e leitura

03/11/2026. [Plano prévio](PLAN_STATE_READOUT.md), [diagnóstico0028](results/CAMPAIGN_0028.md).
Executada na0029; [resultados auditados](results/CAMPAIGN_0029.md) em04/11/2026.
Esta célula fica preservada para reprodução; não é a próxima recomendação de treino.
Use [07_testar_profundidade_e_leitura_kaggle.py](../scripts/07_testar_profundidade_e_leitura_kaggle.py) com Internet e duas T4.
Começa em0029 e avança se o número estiver reservado. Limite7200s desde o início
da célula, incluindo setup, com300s de reserva para exportação. Pode terminar antes.

Revisão fixada:`77b2058e2129f7fcea6f3bf96b4fca4420dddbf9`.
SHA-256 do bootstrap:`3990b8e501c5acf2a88cb25ac13bdada78a61ab53054a6bdcc9cfc470cb8e1b0`.

## Matriz e protocolo

| Variante | Parâmetros | Estado máximo calculado por sequência | Pergunta |
|---|---:|---:|---|
| hierarchy_next | 2.174.232 | 205.872bytes | Hierarquia independente atual |
| hierarchy_immediate | 2.174.232 | 205.872bytes | Ler resumo no fechamento do grupo |
| recurrent | 2.174.232 | 199.728bytes | Duas células em cada byte |
| fine_only | 1.136.652 | 99.864bytes | Uma célula treinada do zero |
| transformer_two | 2.164.992 | não medido para geração | Referência externa |

As hierarquias diferem somente no índice de leitura: mesmas dimensões, média,
ritmo e núcleos. Checksum dos parâmetros iniciais registrado e exigido igual nos
dois braços. Fine_only é treinado do zero; não é o corte pós-treino da0028.
Compartilhamento entre níveis permanece para a próxima comparação com a leitura
selecionada. Nenhum GRM/TTT/ACT/FiLM ou compressão aprendida foi acrescentado.

Seed251,6000passos, WikiText preparado fixado, byte-level, T1024/B8, FP32/Inductor,
dropout0,05, LR0,0004 e scheduler anteriores. Taxa é a receita inicial da0027,
sem ajuste ótimo demonstrado para os controles novos. Validação completa3000/6000,
subconjunto fixo1500/4500; todas registram perda por posição módulo4 relativa ao
reset da janela. Teste reservado intacto. Todos os treinos novos começam do zero.

`research.state_readout=next|immediate`, padrão next. Immediate exige hierarquia.
State_recurrent aceita1/2células; hierarquia exige2níveis. Os dois novos controles
usam architecture_version2; defaults históricos mantêm versão1 e forward conferido
contra1c8ce2a. Esta é uma fonte/configuração/protocolo novos. Qualificação0025
continua evidência histórica da regra, sem reescrever relatório nem aceitar a
fonte atual como se fosse a anterior. Preflights dos novos controles e pilotos
reais precedem treino. Nenhuma retomada científica antiga é migrada silenciosamente.

## Operação e orçamento

1. Reutilizar dados preparados completos, verificando hashes. Se faltarem, recuperar
   shards fixados e preparar dataset novo. Não copiar bins existentes.
2. Testes focados CPU e dois preflights CUDA novos naGPU0. GPU1 executa pilotos e
   treinos; não afirmar que pytest CUDA foi repetido nela.
3. Pilotos isolados frio/aquecido por variante na GPU atribuída:10warmups e30amostras,
   duração do comando, validação e recursos. Frio é primeiro uso dessa família de
   cache, não condição física da GPU. Piloto concorrente estima contenção do host.
4. Prever imports/arranque, treinos, validações e perfis com35% de margem. Admitir
   pares inteiros: leitura primeiro, profundidade depois, Transformer por último.
   Se não couber tudo, dividir os jobs, mantendo6000passos/scheduler e pendências.
5. Treinar um job por T4. Parar ambos antes dos perfis isolados do par, na própria
   GPU de treino com cache aquecido. São cinco perfis nesta triagem; avaliação
   cruzada nas duas GPUs fica para o candidato selecionado.

Uma família de cache por GPU, troca apenas com lane inativa. Limite global
amostrado2GiB e reserva2GiB; estimativa prévia inclui checkpoints transitórios e
ZIP final. Encerramento recolhe workers, remove cache remanescente e descarta o
checkout limpo do bootstrap. Campanhas históricas, dados e pesos ficam preservados.

Tempos de treino são concorrentes; perfis são isolados, cada variante em uma GPU.
Não demonstram economia confirmada nas duas placas. Perfis usam pesos finais em
clone com otimizador novo e updates de warmup/medição; não são retomada do otimizador
final. Fonte, versões, parâmetros, estados, NLL/bits por byte, curvas, fase módulo4,
VRAM, RAM e energia disponível ficam registrados. Ausentes são null; FLOPs dos
estados continuam null. Transformer usa seu estimador existente. Geração não medida.

## Arquivos, falhas e continuação

Um par final: `llm2-campaign-0029-results.zip` e `llm2-campaign-0029-checkpoints.zip`
(ou próximo número), sem after-NN/snapshots cumulativos. Uma fonte canônica no ZIP
de resultados. Backup contém apenas último checkpoint completo de cada job novo.
Após verificar6000, retirar os pesos/estado redundantes3000 da execução nova,
preservando seus metadados. Falha anterior mantém3000. Binários dos dados não entram
nos ZIPs. Pasta de treino e ZIP final de pesos coexistem para permitir download,
sem multiplicar cópias a cada job.

Ambos oferecem download embutido. Arquivos grandes usam partes24MiB sob demanda
viaLLM2_BAIXAR, com SHA e comando de reconstrução. Baixe os dois arquivos de cada lote.
Erro de preflight de dados preserva recibo pequeno antes de descartar checkout.

Nova execução reconhece jobs completos nas pastas nativas deste protocolo no
working. Exige fonte/dados/configuração/versões compatíveis, bloqueia duplicatas
conflitantes e registra a origem; mesma seed não vira nova réplica. ZIPs antigos
não são varridos. Perfis/tempos históricos ficam associados à campanha original.
Se as pastas foram removidas, preserve os ZIPs e selecione as variantes pendentes
emLLM2_SETTINGS['variants']; a auditoria combinará os lotes compatíveis.
Estados incompletos ficam preservados para retomada explícita sob contrato exato;
a célula não desserializa automaticamente estados de jobs interrompidos.

## Decisão

Contrastes: immediate−next e fine_only−recurrent. Uma seed é triagem, sem IC ou
confirmação. Ganho≥0,005NLL é sinal a confirmar; eficiência exige perda≤0,002 e
economia≥10% em medição comparável, pendente de avaliação cruzada neste desenho.
Se leitura imediata não trouxer sinal suficiente, avançar à pergunta de compressão
aprendida/adaptativa. Tecnologias continuam condicionadas ao diagnóstico; não
repetir a mesma matriz indefinidamente. Validação real e limitações emVALIDATION.md.
