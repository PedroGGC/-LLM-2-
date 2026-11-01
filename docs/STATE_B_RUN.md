# Etapa B — quatro controles com mLSTM

31/10/2026. [Plano prévio](PLAN_STATE_B_DELIVERY.md),
[qualificação0025](results/CAMPAIGN_0025.md),
[plano científico](PLAN_RECURRENT_FRACTAL.md).

## Célula Kaggle

Use [05_testar_hierarquia_mlstm_kaggle.py](../scripts/05_testar_hierarquia_mlstm_kaggle.py)
em uma célula com Internet e duas T4. O download é um botão embutido no notebook.
Não execute a qualificação03 novamente. A limpeza04 é opcional se restarem
temporários antigos; ela preserva dados, pesos e ZIPs.

A primeira sessão seleciona apenas seed241: quatro treinos principais de6000
passos. Seeds242/243 continuam no plano para outros lotes. Meta2h/teto3h desde
o início da célula, incluindo instalação, dados, testes, compilação, triagem,
treinos, avaliação e exportação. Pode terminar antes e não promete concluir se
o piloto mostrar duração incompatível.

## Ciência preservada

Worker científico fixado na revisão qualificada
`1c8ce2abb8f47d3167454b3503598854b7756dad`. A revisão de operação é independente.
O bootstrap busca dois checkouts shallow, instala somente o worker qualificado e
executa o operador como arquivo, com `PYTHONPATH` apontando ao worker. O operador
recusa importar ciência de outro diretório; o validador original exige exatamente
as fontes e versões da0025. Não migra relatório nem retoma pesos incompatíveis.

O relatório original da0025 acompanha a revisão de operação em
[JSON](results/campaign_0025_platforms.json), bytes preservados do ZIP, SHA-256
`cf281d13a522724eabfe84e1e1a59b4244b5093971e9067867030db5b1894da3`.
Não depende da antiga pasta do working. São incluídos `operator.json`, revisão
científica, revisão operacional, hashes e `report/operator-source.zip`.
Snapshots de treino continuam contendo a ciência original. O relatório analítico
também identifica o operador. Não misture custos de operadores diferentes sem
conferir essa identificação; o agregador antigo não impõe esse campo adicional.

| Braço | Organização | Parâmetros |
|---|---|---:|
| Recorrente simples | Duas células mLSTM, sem hierarquia | 2.174.232 |
| Hierarquia independente | Fino por byte; amplo por4; núcleos distintos | 2.174.232 |
| Hierarquia compartilhada | Mesmas operações; um núcleo; estados separados | 1.136.652 |
| Transformer | Dois blocos, referência externa | 2.164.992 |

WikiText preparado do ciclo3, byte-level, T1024/B8, FP32/Inductor, dropout0,05,
6000passos. Completas em3000/6000; subconjunto fixo em1500/4500. Sem consulta ao
teste reservado. Sem GRM/TTT/ACT/FiLM/JEPA/NCM/MTP/currículo novos.

## Ordem operacional e orçamento

1. Testes CPU originais e testes CUDA originais da regra selecionada, incluindo
   hierarquia. Não exigir kernels das alternativas não selecionadas.
2. Para cada variante: piloto isolado emGPU0 eGPU1; depois triagemLR0,0002/0,0004
   no cache da mesma variante emGPU0. São8tentativas de750passos, seed901,
   divisão interna por documento apenas do treino; reinício do zero no treino
   principal. O cache conserva código compilado, nunca pesos/otimizador do piloto.
3. Piloto de dois workers para estimar contenção do host. Todas as medições usadas
   para comparar velocidade/VRAM continuam isoladas.
4. Admitir somente blocos completos de quatro braços. Previsão com35% de margem,
   tempo de compilação medido, avaliações e reserva para perfis finais frios nas
   duas GPUs. Preservar5min para exportação e2GiB livres. Se não couber, exportar
   pilotos/tuning; não encurtar passos nem reportar treino incompleto como concluído.
5. Treinar dois jobs independentes por vez, um em cada T4. Parar ambos antes dos
   perfis isolados do par; medir primeiro na GPU de origem e depois na cruzada.
   Não chamar um perfil concorrente de isolado. Não repetir perfis por seed nova.

Cada GPU mantém somente a família de cache corrente. Ao trocar variante, remove
apenas a pasta daquela GPU, com lock e sem worker ativo nela. Taxas consecutivas
podem conservar cache. Conservação não garante cache hit: warmup e tempo total
são medidos. Cache da outra GPU, dados e checkpoints ficam protegidos.
Limite global monitorado2GiB, amostragem5s; ultrapassar causa parada/exportação.
O limite não garante pico instantâneo abaixo de2GiB entre amostras. Ao encerrar,
o worker original remove o cache restante, e o bootstrap descarta checkouts limpos.

## Arquivos e continuação

Um par de ZIPs temporários ao completar2dos4treinos principais; isso é50% dos
jobs, não50% do tempo de relógio. Após validar cobertura/CRC/SHA dos definitivos,
o temporário é removido. Não gera um `after-NN` por job. Resultados contêm métricas,
curvas, fontes e proveniência; checkpoints contêm o último estado completo por job.
Em falha/deadline permanecem completos anteriores e relatório parcial.

Arquivos grandes oferecem partes de24MiB sob demanda, sem cópias das partes no
working, com comando PowerShell de reconstrução e SHA-256. A função
`LLM2_BAIXAR('/caminho/arquivo.zip', part=1)` fica disponível na célula.

Para próximos lotes, preservar o `campaign.json` com taxas congeladas e a revisão
operacional desta entrega. O bootstrap aceita `seeds=[242]` ou `[243]` e
`tuning_report='/caminho/campaign.json'`. A fonte/dataset/regra/protocolo da triagem
serão novamente conferidos. Não misturar seeds repetidas como réplicas novas.

Uma seed é exploratória: não promove ganho fractal nem equivalência. A decisão
principal continuará separando hierarquia, compartilhamento e confronto externo
com Transformer. GPU/Inductor desta orquestração ainda dependem do teste Kaggle;
integração CPU não substitui essa validação.
