# Próxima sessão — diagnóstico congelado da 0027

01/11/2026. [Escopo prévio](PLAN_STATE_DIAGNOSTIC.md),
[auditoria que motivou a rodada](results/CAMPAIGN_0027.md).

Use [06_diagnosticar_pesos_0027_kaggle.py](../scripts/06_diagnosticar_pesos_0027_kaggle.py), em uma célula com Internet e
duas T4. Começa em0028 e avança se o número estiver reservado. O relógio começa
na célula; limite padrão7200s (duas horas), máximo configurável10800s. Não é duração
obrigatória nem há trabalho adicional para preencher tempo. Duração T4 desta
operação ainda não foi medida; um piloto de dois batches estima a cobertura
restante com35% de margem. Se não couber, falha e exporta os resultados completos
anteriores, sem encurtar a avaliação e chamá-la de completa.

Revisão operacional fixada:`55dec8e7b4d10c89722c7a0a9982b621daf74837`.
SHA-256 do bootstrap:`ef5405deed514cd11614595a3bd2da21784baa177e498200cf009bf483fdfe5e`.

## Antes de executar

Preserve `/kaggle/working/llm2-campaign-0027`. O script encontra os pesos finais
nessa pasta. Se ela já foi removida, anexe `llm2-campaign-0027-checkpoints.zip`
como Input do Kaggle; o bootstrap procura esse nome exato no working/input e
aceita somente o SHA da cópia auditada. `results.zip` sozinho não é suficiente.
O ZIP de checkpoints não precisa ser extraído por você.

SHA do backup recebido: `2b4724f97c012876334d0776fb89ee50d564a17566328563603c630f9bebc076`.
Pesos/metadados originais são preservados. Uma restauração copia apenas quatro
safetensors finais e seus metadados para uma pasta da nova operação. Essas cópias
são removidas após validar o ZIP final; nunca extrai/carrega otimizador ou pickle.
Se os dados preparados originais faltarem, reconstrói com shards/importador fixados
e exige o fingerprint original; versões/dados/fontes divergentes bloqueiam análise.

## O que será avaliado

**Nenhum treino novo.** Quatro checkpoints da seed241, fonte científica original
`1c8ce2abb8f47d3167454b3503598854b7756dad`, configurações e pesos congelados.
O código do diagnóstico é uma operação separada, executada como arquivo com
`PYTHONPATH` da fonte original; não altera o backbone nem migra checkpoints.

| Checkpoint | Leituras avaliadas |
|---|---|
| Transformer2blocos | Original |
| Recorrente simples | Original; saída da primeira célula, sem a segunda transformação |
| Hierarquia independente | Original; ramo amplo silenciado; ramo amplo com ganho0,25 e0,5 |
| Hierarquia compartilhada | Mesmas quatro condições |

São11condições em quatro jobs, em dois pares, um job por T4. Uma passagem do
backbone por batch fornece representações para as leituras alternativas. O
caminho amplo continua sendo calculado: essa intervenção não mede economia de
execução. No recorrente simples, a segunda componente medida é a correção da
segunda célula de mesma frequência; não é chamada de memória ampla.

FP32/eager, dropout desligado. Não executa Inductor, nova qualificação, triagem
de taxa, optimizer-step ou teste reservado. A leitura original precisa repetir
a NLL da0027 dentro de5e-5 absolutos; caso contrário, o resultado é marcado como
não reproduzido e a operação falha, preservando o diagnóstico. A tolerância serve
para avaliação numérica; não modifica os contratos de retomada de treinamento.

## Medições e limites de interpretação

- NLL, bits por byte e mesmo conjunto de janelas completas da0027.
- Perda por posição módulo4 e por quatro intervalos da janela1024. As posições
  são relativas ao reset da janela, não ao deslocamento global do documento.
- RMS fino/amplo, razão global de RMS e cosseno global, ponderados pelas posições
  supervisionadas. Para Transformer, essas medidas ficam nulas.
- Reconstrução da saída original a partir dos componentes verificada em cada
  batch hierárquico. Nenhum peso é atualizado.
- Duração operacional concorrente e pico de VRAM do diagnóstico, sem alegar
  aceleração de geração ou comparar energia/custo com os benchmarks da0027.

Intervenções pós-treino medem sensibilidade de pesos já adaptados ao modelo
completo. Não são substitutas de treino somente fino, gate aprendido ou compressão
aprendida. Selecionar um ganho por essa validação é exploração, não confirmação.
Uma futura campanha treinada terá protocolo próprio e seeds novas.

## Arquivos e progresso

Um único `llm2-campaign-0028-results.zip` (ou próximo número disponível), com
`conditions.csv`, `report.json`, quatro JSONs de avaliação, logs, referência e
fontes uma vez cada. Sem ZIP novo de checkpoints, `after-NN` ou snapshots de50%.
Os JSONs por job são publicados atomicamente; falhas/deadline exportam evidência
parcial. Dados e pesos ficam fora do ZIP. Restaurações temporárias próprias são
removidas após validá-lo; originais e backups anteriores não são apagados.

Mensagens no início de cada job e heartbeat a cada30s durante avaliação. Batches,
piloto e previsão detalhada ficam nos logs. Ao final, a tabela de perdas/deltas
é impressa e aparece o botão de download embutido. `LLM2_BAIXAR` fica disponível
para solicitar novamente o arquivo, sem depender de URL do proxy Kaggle.

As verificações CPU locais e integração com fonte original estão em
`VALIDATION.md`. Nenhuma nova medição CUDA/eager local foi feita; a execução no
Kaggle validará esta operação no ambiente real.
