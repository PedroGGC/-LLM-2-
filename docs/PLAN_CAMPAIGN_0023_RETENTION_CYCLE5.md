# Auditoria 0023, retenção de ZIPs e próximo controle — 31/10/2026

Plano registrado antes das mudanças. Não executar treinos científicos nem limpeza
remota. Preservar arquivos recebidos, código congelado 6a6af1a e branch comum.

## Auditoria e decisão

Verificar CRC/membros e SHA dos dois ZIPs locais; manifesto, fonte/configs contra
Git, snapshots, configurações resolvidas e parâmetros/FLOPs; 18 treinos/6000 passos,
72 validações de 1.144.064 alvos reconstruídas das janelas; pesos/estados finais
por SHA sem pickle. Recalcular contrastes, critérios e probes K1/2/4/8. Distinguir
treinos concorrentes, perfis isolados, energia da GPU e custos de compilação.
Curar relatório/dados/gráficos e atualizar README/VALIDATION/roadmap.

Leitura preliminar: fino é candidato de simplificação; escalas não mostram o
benefício central; Transformer de dois blocos lidera. Não concluir superioridade
fractal, equivalência formal ou generalização para outros regimes. Decisão final
depende das checagens acima.

## Arquivos: combinar as duas opções do usuário

Criar um par temporário de ZIPs quando completar `ceil(n_jobs/2)` jobs, contando
treino e avaliação completos; isso não é metade do tempo. O par cobre todos os
jobs completos até aquele instante. Ao publicar/verificar ambos os definitivos,
remover somente o par temporário, registrando recibo. Em falha manter recuperação
e exportação parcial. Não criar par por job nem sobrescrever arquivos publicados.
Pasta de treino e seus checkpoints seguem disponíveis no working.

Limpeza da 0023: validar os definitivos, fonte/identidade e cobertura dos 18 estados
finais; exigir hashes das cópias locais verificadas. Excluir só os ZIPs redundantes
`llm2-campaign-0023-after-NN-(results|checkpoints).zip`, após conferir que não há
workers e que nenhum conteúdo único de checkpoint será perdido. Preservar pasta,
definitivos, datasets e outros arquivos. Sem novo backup gigante do working.

## Próximo controle (ciclo 5, antes de tecnologia nova)

Quatro braços × seeds novas 221/222/223: FLRM somente fino treinado do zero com
K=1/2/4 e Transformer de dois blocos. Mesmos parâmetros (diferença de dois pesos),
dados/precisão/contexto/batch/otimizador da 0023; nenhum FiLM/JEPA/NCM/ACT/DEQ ou
novo mecanismo. Orçamento de FLOPs estimados igual ao fino K2 em 6000 passos;
horizontes próprios fixados antes do piloto, arredondados para múltiplos de quatro,
com scheduler e avaliações em 25/50/75/100%. Tokens vistos diferem e serão reportados.
Não chamar isso de custo de tempo/energia igualado.

Comparação primária K2−K1, K4 secundária e Transformer como referência. Três seeds
são triagem exploratória; eventual sinal precisa cinco novas seeds para confirmação.
O objetivo é distinguir profundidade aprendida de extrapolar K de um modelo K2.
Profundidade variável/RINS permanece depois deste controle; demais frentes continuam
condicionais no roadmap. Não repetir a matriz de escalas sem hipótese nova.

Duas T4 em jobs independentes; meta 2h/teto 3h. Pilotos isolados nas duas GPUs e
concorrente, margem35%; reservar 20 min pós-treino (a 0023 gastou ~22 min para seis
variantes) e 5 min exportação; guardar 2 GiB. Pós-treino enxuto: perfis nas duas
GPUs e diagnósticos/probes da primeira seed, sem replicar perfil por seed.

## Aceitação

Auditoria sem discrepâncias não explicadas; testes focados por etapa e suíte
apropriada. Testar metade/final/erro/deadline, concorrência, ZIP parcial/corrompido,
hash local errado, conteúdo único, confinamento/links e preservação dos definitivos.
Para K1/2/4: causalidade, parâmetros ligados, backward finito, memorização e retomada
CPU/math equivalente com dropout e acumulação. CUDA novo no preflight real, sem
alegar medição local. Exportação/snapshot/estado isolados e proveniência completa.
