# Entrega Kaggle P01–P06 — plano, riscos e aceitação

Data: 26/10/2026. Autorização: implementar a próxima entrega definida na conversa, publicar o necessário e fornecer uma célula para Kaggle Notebooks T4x2. O usuário executa GPU no site Kaggle; não depende de GPU local.

## Etapas
1. Núcleo comum: avaliação por janela com saída identificada; teacher hidden-only; máscara JEPA; normalização explícita CE na acumulação; teste CUDA FP32; perfil aquecido.
2. Executor Kaggle: processos isolados por GPU, falha explícita, logs integrais, diretórios novos, avaliação completa, benchmarks/diagnósticos opcionais e exportação.
3. Branch fractal: modo sequencial preservado e modo agrupado com pesos idênticos; diagnóstico de escalas/profundidade fora do treino; comparação controlada.
4. Integração: configurações de piloto, notebook/célula de inicialização com commit fixado, documentação e regressões. Publicar branches sem modificar master ou experimentos antigos.

## Riscos
- Somente CPU disponível localmente. CUDA/T4, ganho de throughput, VRAM e Inductor serão verificados pelo script no Kaggle, sem alegar resultados antecipados.
- Agrupar escalas muda ordem de RNG de dropout e arredondamentos; paridade algébrica será verificada com dropout controlado, e retomada exata dentro de cada modo.
- A ponderação por tokens se aplica à componente LM; auxiliares permanecem médias por microbatch, incluindo a penalidade não aditiva de variância. Modo legado preservado como opção explícita.
- Corrigir código altera proveniência: checkpoints antigos continuam preservados e não recebem relaxamento de contrato de retomada.
- Downloads automáticos limitados ao pequeno corpus público de demonstração real; dados existentes/upload podem ser selecionados. Sem treinos extensos locais.

## Aceitação
- Suítes CPU relevantes e integração aprovadas; skips GPU registrados como skips.
- Causalidade, forward/backward, finitude, memorização e retomada exata CPU FP32/SDPA math com dropout/acumulação.
- Médias de validação reconstruíveis pelas janelas; verificação de identidade de dataset/checkpoint e saídas sem sobrescrita.
- Executor não reutiliza checkpoint antigo após falha; cada comando salva saída integral e recebe dispositivo explícito.
- Teacher hidden-only preserva hidden e evita a cabeça; FLOPs refletem o caminho realmente executado.
- Modo fractal agrupado mantém parâmetros/aliases; não vira default supostamente mais rápido sem medição.
- Célula Kaggle requer apenas colar/executar com Internet e T4x2; instala com o Python do notebook e exporta resultados.
- Código experimental permanece em src/llm2 e FLRM na branch experiment/fractal-lrm. Dados/runs/auditoria local bruta não entram no push.
