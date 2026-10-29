# Ciclo 2 — plano de implementação e protocolo, 29/10/2026

## Escopo aprovado

Preparar o próximo teste após a campanha 0016. Matriz comunicação × FiLM com referências Transformer/LRM; contexto sem RMS, K=2, T=256, batch efetivo 32, FP32/Inductor, LR/scheduler de 2.000 passos. Seeds 81/82/83, novas e iguais para todos os braços. Avaliação completa em 500/1.000/1.500/2.000.

## Implementação em etapas

### Ajuste solicitado: célula curta e teto de duas horas

Restaurar o fluxo de célula curta que baixa código do GitHub por commit completo,
sem exigir Input ZIP. Reduzir o padrão ativo para 7.200 s, incluindo download,
checkout, dependências, testes, piloto, treino, avaliação e exportação; manter
reserva de 300 s. O teto não é duração mínima. Preservar a matriz, 2.000 passos,
scheduler e seeds; blocos que não couberem não começam e a campanha fica parcial.

Implementar caminho Git no bootstrap compartilhando verificações e orçamento com
o caminho ZIP. Validar checkout exato, hashes, célula desativada inerte e limite
propagado ao executor. Publicar os arquivos revisados em `experiment/fractal-lrm`
sem force-push, após os testes, para entregar uma célula com SHA realmente disponível.
Preservar os pacotes imutáveis já entregues; novas células usam novos arquivos.

Aceitação: célula curta executável com commit publicado, padrão de 7.200 s nos
caminhos ativos, setup contado no mesmo relógio, preservação dos controles
científicos e testes CPU relevantes. CUDA/Inductor continua pendente de execução
no hardware real. Risco: a janela menor pode permitir menos seeds completas;
não reduzir passos silenciosamente nem completar o relógio com treinos adicionais.

### Preparação inicial, preservada como histórico da entrega ZIP

1. Congelar configurações e validar o controle sem comunicação, inclusive causalidade, gradientes, parâmetros ativos, memorização e retomada com dropout/acumulação. Preservar os caminhos antigos por padrão.
2. Criar executor específico em `src/llm2/research/`, reutilizando Trainer, profiling, checkpoints e avaliação. Execução serial, ordem balanceada por seed, atribuição alternada às GPUs, piloto prévio por variante/GPU, origem de dados verificada.
3. Controlar sessão global de 14.400 s desde o bootstrap (incluindo setup), com reserva de 300 s para exportação. Não iniciar um novo trabalho quando o piloto indicar falta de tempo. Watchdog interrompe subprocessos travados e preserva checkpoints completos anteriores. Resultados parciais nunca recebem status de campanha completa.
4. Gerar relatório fatorial com resultados individuais, parâmetros registrados/ativos, custos estimados e curvas de avaliações completas; sem chamar budgets desligados de custo igualado.
5. Entregar pacote de fontes com hash e bootstrap Kaggle que use esse pacote local, evitando depender de commit não publicado. Checar dados/código antes de executar; não sobrescrever outputs.
6. Executar testes locais, simulação ponta a ponta CPU e suíte completa. Atualizar VALIDATION e guia com resultados reais e limitações.

## Riscos e critérios de aceitação

- Sem comunicação, os ramos grossos continuam calculados, mas deixam de contribuir para a loss fina; distinguir parâmetros registrados de parâmetros que recebem gradientes. O estimador de backward é conservador e não mede operações elementwise.
- FiLM desativado remove seus pesos e pode alterar a sequência de inicialização exclusiva. Parear explicitamente os demais pesos exclusivos entre as quatro variantes, além do núcleo comum, e registrar hashes. Não confundir ablação com mudança aleatória de inicialização.
- Piloto não escolhe hiperparâmetros nem usa teste reservado; apenas prevê duração. Tempo de compilação, treino, avaliação e benchmark registrados separadamente.
- Testar ausência de influência cruzada, preservação de projeção/gate próprios e os quatro braços em causalidade/backward/memorização/retomada. Testes CPU FP32/math com dropout e acumulação.
- Orçamento global deve incluir subprocessos de teste/setup, não somente Trainer. Encerramento de processos não garante checkpoint do passo interrompido nem controle de cobrança do provedor.
- Dataset TinyShakespeare deve reproduzir hashes de tokens e splits da 0016; hash do manifesto pode mudar com proveniência do código. Não relaxar checks de retomada.
- Testes CUDA/Inductor serão executados pelo script na T4 real. Testes locais em CPU não comprovam GPU; nenhuma campanha científica será executada localmente.
- Nenhum commit/push, nova tecnologia adiada ou nova dependência de serviço é necessário para entregar o pacote.
