# LRM: hipótese e implementação V2

## Caminho causal

O Transformer original não foi reescrito. `LatentRecursiveModel` usa embeddings compartilhados
com a cabeça textual, estado inicial zero [B,T,D], RMSNorm e um `SharedRecursiveCore`.
Cada chamada do core aplica `norm(z)+W_context*x`, blocos DecoderBlock causais existentes e
`z + sigmoid(gate)*update`. Gate vetorial inicia em -2 (~0,119), controlando a amplitude inicial.
O contexto de posição t não contém dados >t; a atenção permanece SDPA causal com RoPE.

`iterations` controla a profundidade básica. `latent_steps` adiciona refinamentos contínuos
sem produzir ou inserir tokens fictícios. `latent_warmup_steps` libera um refinamento a cada
intervalo de optimizer-steps; avaliação usa a profundidade completa. A etapa do currículo de
profundidade é um buffer persistente. O estimador de FLOPs usa a profundidade realmente ativa
em cada atualização e a profundidade completa do teacher/avaliação quando aplicável.

Não há carry entre segmentos: cada forward começa em zero; passar `state` externo gera erro.
O estado retornado e `intermediate_states` servem à inspeção/supervisão. Guardar históricos
aumenta memória. `record_states` é opt-in; supervisão intermediária também os ativa.
`forward(..., iterations=N)` e a avaliação de raciocínio permitem comparar profundidades.

Full BPTT é o padrão. `detach_every=N` corta **apenas** a conexão entre estados nos limites
explicitados, mantendo o contexto ligado ao embedding. Não é gradiente completo nem diferenciação
implícita. `gradient_checkpointing` usa checkpoint non-reentrant com preservação do RNG e
recomputação contabilizada. O controle `untied` mantém uma cópia independente de cada core,
permitindo comparar unrolling equivalente sem compartilhamento. Compartilhamento não elimina FLOPs.

## JEPA auxiliar

`PredictiveObjective` contém predictor D→D→D, teacher copiado do backbone e congelado,
stop-gradient e EMA opcional após **cada optimizer-step**, não após cada microbatch. Com EMA
desativada o teacher permanece na inicialização: essa é outra ablação, não autoatualização implícita.
O teacher é sempre eval; seus parâmetros e buffers ficam no checkpoint, inclusive `ema_updates`.

Online hidden[t] prevê teacher hidden[t+offset], com normalização e MSE. O teacher pode ver
o prefixo que inclui esse alvo futuro; isso acontece só no objetivo de treino. Nenhum dado do
teacher é usado na saída textual. Em tarefas com máscara, só pares supervisionados válidos
participam do MSE/variância, evitando aprender predominantemente padding.

```text
loss = lm_weight * cross_entropy
     + jepa_weight * (prediction_mse + variance_weight * variance_penalty)
     + mtp_weight * mean(offset_losses)
     + intermediate_weight * mean(intermediate_language_losses)
```

`lm_weight` deve ser positivo. A penalidade de variância usa desvio por dimensão ao longo das
posições/amostras válidas. Monitoramos `representation_std`, `target_std`, `collapse_fraction`,
MSE e penalidade. Isso **não prova** ausência de colapso ou superioridade; não implementa toda a
regularização VICReg, nem o protocolo multivista original LLM-JEPA. Aqui o alvo é hidden futuro,
uma hipótese local própria que deve ser comparada a E0 e testada quanto a atalhos.

O teacher custa um forward completo (incluindo a cabeça calculada pelo backbone atual).
Contamos seus parâmetros frozen, predictor, cabeças extras e respectivas operações. A EMA,
normas, variância e operações elementwise são excluídas da estimativa densa e declaradas como tal.
Snapshots, `objective.safetensors` e `training_state.pt` preservam os auxiliares; a CLI de geração
usa somente o backbone. Validação reporta loss LM/perplexidade separadas da loss total do objetivo.

## MTP

Cabeças lineares independentes predizem offsets adicionais, sem substituir `lm_head`.
Como `targets[t]=token[t+1]`, a primeira cabeça extra usa hidden[t] com `targets[t+1]`.
A loss ignora máscaras -100, corta as bordas alinhadamente e faz média das cabeças.
Não há teacher forcing com tokens futuros no caminho de hidden[t]. É uma ablação simples de
predição de múltiplos offsets, **não** a arquitetura completa dos módulos MTP do DeepSeek-V3,
nem speculative decoding. O teste confere cada offset com um cálculo explícito.

## Custos e limites

O estimador conta cada core, contexto projetado, cabeça textual, teacher, predictor, MTP e
supervisão intermediária. Forward e backward aparecem separados; backward é estimado em
2x forward diferenciável e é conservador sob truncamento. Checkpointing acrescenta forward
do core. A fórmula não inclui otimizador/normas/softmax/elementwise/compilação. `profile --execute`
gera um trace real; o contador de FLOPs do profiler cobre apenas operações suportadas.

Testes CPU cobrem shapes variáveis, causalidade, finitude, overfit, compartilhamento/controle
untied, geração, EMA e retomada com dropout/acumulação/progressão/checkpointing. Esses testes
não demonstram raciocínio, interpretabilidade dos estados ou eficiência científica.
