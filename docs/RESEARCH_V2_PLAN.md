# V2 — auditoria e plano de integração (25/10/2026)

## Estado inicial confirmado

Leitura integral de README, PLAN, VALIDATION, AGENTS, pyproject, src/llm2, testes,
configs, scripts e células do notebook. `python -m pytest -q`: **29 passed, 1 skipped**
(CUDA ausente), 22,80 s. Python local 3.12.2 e Torch CPU existentes serão preservados.
Git estava em master sem commits, com toda a etapa 1 não rastreada e sem remotes.
Preservar essa árvore em commit inicial, criar feature/llm2-research-v2 e uma branch
experiment/fractal-lrm derivada do núcleo comum. Não alterar data/ ou runs/ existentes.

Confirmados: Transformer 33.990.144, 9x512, 8 heads, RoPE/RMSNorm/SwiGLU/SDPA,
pesos compartilhados; AdamW/scheduler; byte/BPE; CLI; RNG/sampler/checkpoint completo.
Pontos de extensão: factories de modelo/objetivo/estimador, state opaco, AttentionKernel.
Limitações reais: métricas auxiliares são descartadas pelo Trainer; ausência de hook EMA,
currículo persistente, seleção de arquitetura na CLI, limites de orçamento e relatórios;
proveniência só percorre YAMLs no primeiro nível; inferência CLI aceita apenas baseline.
Não há validação real de CUDA/A100/Inductor nem evidência de eficiência científica.

## Marcos, dependências e aceitação

0. Auditoria/baseline acima; manter todos os testes existentes sem removê-los.
1. Adicionar seção research opcional, omitida em configs legadas; factories centrais,
   estimativas por componente, hook de atualização do objetivo e métricas auxiliares.
   Preservar layout de checkpoints e campos de schema 1; novos estados ficam em campos
   opcionais. Código/ambiente incompatíveis continuam falhando explicitamente.
2. Núcleo recorrente causal com normalização, gate sigmoid, contexto por posição,
   estado zerado por forward, BPTT completo por padrão, truncamento explícito opcional,
   checkpointing non-reentrant e histórico opcional. Comparar com núcleo não compartilhado.
   Aceitar só após causalidade, gradientes, overfit, geração e retomada exata em CPU.
3. JEPA auxiliar: teacher causal congelado (EMA opcional), predição de hidden futuro com
   stop-gradient e penalidade de variância online; manter LM obrigatória. Registrar variância,
   loss por componente, parâmetros frozen/auxiliares e FLOPs do teacher/predictor.
4. LRM = recursão + JEPA; passos latentes adicionais usam estado contínuo, sem tokens
   fictícios. Permitir supervisão intermediária/progressão de passos; não chamar reprodução
   integral de Coconut/TRM/HRM. Testar combinações antes de prosseguir.
5. Fractal apenas em worktree/branch experimental. Agregações causais local, janelas e
   prefixo global em resoluções alinhadas ao token; troca de escalas por posição, núcleo
   compartilhado opcional e condicionamento FiLM. F1–F4 como ablações independentes.
   A versão inicial não promete economia por compressão de sequência nem parada adaptativa.
6. Registro/DAG validado, nove geradores e verificadores CPU, splits por namespace e famílias,
   probe próprio, currículo uniforme/fixo/adaptativo com exploração/revisitação. Misturar texto
   e tarefas mantendo batch/seq fixos, preservar RNG/controlador/histórico no sampler.
   Avaliação por habilidade usa respostas geradas e exact match; nunca usar test para adaptar.
7. MTP independente da cabeça principal (offset k usa targets[k:], entrada até t apenas).
   Matriz E/F/C executável individualmente, runner sem campanha automática, relatórios JSON/
   CSV/Markdown/SVG, custos separados, profiling opt-in e avaliação de generalização.
8. Notebook preservado e ampliado com seleção e execução explicitamente opt-in, orçamento
   de tempo/FLOPs/tokens, Drive, exportação e smoke opcional. Perfis CPU/CUDA e portes
   maiores apenas configuração. Nenhuma sessão paga ou treinamento longo será iniciado.

## Riscos e escolhas científicas

- Cada iteração/escala tem custo: estimativas contam todas as chamadas, recomputação de
  gradient checkpointing, teacher e auxiliares. FLOPs são analíticos, nunca medição exata.
- JEPA pode colapsar apesar da LM/variância/EMA; monitorar, não prometer prevenção ou ganhos.
- Alvos futuros só no loss; logits causais e estados usados na geração nunca os recebem.
- Pooling de sequência inteira é proibido para posições anteriores; só prefixos/janelas.
- Retomada estrita entre versões de código não é garantida; pesos safetensors legados
  continuam legíveis, e o snapshot V1 permite retomar V1 com o código original.
- Config legada não deve mudar números ou caminhos. Novos campos são opt-in.
- Corpus procedural finito tem risco de memorizar padrões: registrar famílias/IDs e
  separar treino/probe/val/test e casos OOD (composição/comprimento/números/identificadores).
- 200 créditos a 5,30/h são entrada do usuário; reserva de 20% implica teto planejado
  de ~30,19 h. Watchdog só verifica em fronteiras seguras, não limita cobrança exatamente.
- Sem hardware CUDA, testes GPU são pulados e profiling/memória GPU ficam não validados.

## Referências e independência

Implementação própria; não copiar código de terceiros. Consultar artigos e repositórios
oficiais, registrar diferenças e licenças em docs/REFERENCES.md. LearnHarness é inspiração
de feedback pedagógico, não evidência de eficiência de treinamento neural. A hipótese de
100.000x permanece especulativa; esta entrega só prepara experimentos mensuráveis.
