# Harness curricular externo

O mesmo `MixedCurriculumStream` serve ao Transformer, LRM e FLRM. Ele substitui somente a
amostragem do Trainer, sem acoplar o controlador ao backbone. Registra proporções efetivas
de exemplos e alvos de texto/procedurais, além da distribuição por habilidade/dificuldade.

## Habilidades, geradores e verificação

`SkillRegistry` valida IDs únicos, dependências existentes e ausência de ciclos. O DAG manual
inclui aritmética, comparação, condicional, composição, sequências, manipulação simbólica,
lógica, múltiplas etapas e ordenação estruturada. Ele é uma hipótese de precedência pedagógica,
não uma descrição validada da organização interna da rede.

Cada problema contém prompt, resposta, habilidade, dificuldade, seed, split, família,
metadados e verificador versionado. `verify_v1` recalcula a resposta a partir dos metadados,
sem `eval`, API comercial ou comparação ingênua com o campo answer. IDs vêm do conteúdo do
prompt e habilidade, sem seed/split, de modo que duplicatas não ganhem IDs artificiais.

Namespaces separados: train/probe/val/test. Sementes são combinadas ao namespace por SHA-256;
faixas numéricas também são disjuntas entre split/dificuldade. Conjuntos são deduplicados por
ID. Composição, lógica, simbólica e ordenação reservam uma família B para teste; não se presume
que isso esgote a generalização. O modelo ainda compartilha vocabulário e regras entre splits.
No treino inicial os números têm cinco dígitos; dificuldade muda operações/quantidade de passos,
não transforma automaticamente a tarefa em um currículo universal de aritmética.

As respostas completas são supervisionadas; prompt/padding usam -100. Problemas que não cabem
no contexto causam erro, nunca são truncados silenciosamente. O harness exige contexto >=128;
configurações com dificuldades maiores devem verificar que cada exemplo cabe. O texto geral
continua usando a loss usual. A acumulação calcula a média de losses dos microbatches; a mistura
opera por microbatch, não promete uma proporção exata de tokens supervisionados. Ambos os
contadores (inputs densos e alvos supervisionados) são registrados para não confundi-los.

## Políticas

- Uniforme: pesos iguais por habilidade/dificuldade.
- Fixa: favorece níveis progressivos conforme número de draws, mantendo revisitação.
- Adaptativa: usa sucesso EMA, mudança recente, incerteza por visitas e prontidão de dependências.
  Uma mistura mínima `exploration/N` mantém probabilidade positiva em todas as células.

Não há descarte permanente de habilidades dominadas. A política é interpretável e determinística
no mesmo ambiente. O sampler tem RNG privado; o controlador tem outro RNG. Checkpoints guardam
ambos, sampler textual, counts, records, histórico, draws e exemplos recentes realmente vistos.

## Sondagem e avaliação científica

Somente `probe` pode atualizar o controlador; fornecer val/test ao método update causa erro.
A sondagem barata usa acurácia de tokens de resposta com teacher forcing, **não exact match
de geração livre nem prova de domínio da habilidade**. Seu modelo usa eval; RNG e modos são
restaurados. Seus forwards e tempo total são cobrados separadamente, e o limite de FLOPs de
aprendizagem inclui essas sondagens.

`reasoning` faz geração greedy e verificação determinística no val/test independente, casos
de mais passos/mais comprimento e, opcionalmente, exemplos recentes vistos no treino
(`procedural_seen.json`). Casos acima do contexto são reportados e contam como falha.
Predições e taxas por habilidade ficam no JSON. O conjunto científico nunca controla o sampler.
Preservação de números/IDs é examinada nas tarefas numéricas/simbólicas; não é uma avaliação
geral de linguagem. Compare também a validação no corpus textual e investigue regressão.

Limitações: geradores pequenos, regras públicas, sementes finitas e controles de memorização
limitados a instâncias/famílias. Não houve campanha multissemente, tuning no teste ou evidência
de que a política adaptativa melhora eficiência. LearnHarness é inspiração pedagógica externa;
o algoritmo não reproduz o Skill-It nem seu protocolo de descoberta automática de skills.
