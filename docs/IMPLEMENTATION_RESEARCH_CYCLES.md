# Implementação dos ciclos de validação fractal

Solicitação aprovada em 28/10/2026. Implementação científica em `experiment/fractal-lrm`, preservando a base comum e as alterações anteriores de organização.

## Plano antes das mudanças

1. Adicionar normalização opcional de contexto e controle de comunicação entre escalas, preservando defaults, causalidade, parâmetros compartilhados e estimativas declaradas.
2. Testar caminhos com/sem cache, modos de execução, FiLM, ausência de troca, memorização e próxima atualização após retomada com dropout/acumulação.
3. Acrescentar preparo documental com splits explícitos e teste reservado, leitura estrita dos schemas antigo/novo e avaliação `--split val|test`.
4. Preparar importação WikiText-103 com revisão fixada, artigos íntegros, seleção determinística de ~32 MiB e detecção de duplicação; não substituir o dataset 0015.
5. Implementar matrizes dos ciclos, piloto, limite global de sessão de 4h, execução isolada/alternância de GPU, seleção congelada para ciclos dependentes e comparação com custo igualado.
6. Registrar resultados/intervalos, decisão inconclusiva quando apropriado, roadmap condicional e validação real. Exportar fontes para execução externa sem inventar resultados CUDA.

## Riscos

- Normalizar contexto pode amplificar ruído; RMS não é critério de seleção.
- Remover comunicação deve preservar transformações próprias dos ramos, inclusive atraso causal da compressão.
- Novos campos/código mudam proveniência: checkpoints anteriores não terão retomada relaxada.
- O schema 1 deve continuar estrito; schema 2 adiciona somente splits explícitos e metadados documentais identificados.
- Quatro horas são teto de sessão incluindo subprocessos e reserva de exportação; watchdog não é controlador de cobrança nem garantia contra encerramento externo.
- CUDA indisponível localmente: os ciclos na T4 e a seleção de vencedores exigem execução posterior no hardware real. Ciclos 2/3 não escolhem resultados inexistentes.
- Tecnologias adiadas e A100 são condicionais: não implementar automaticamente nem contratar GPU.

## Aceitação

- Defaults reproduzem a operação anterior; normalização sem pesos, FP32, dtype preservado, cache equivalente.
- Sem comunicação, nenhuma saída de ramo depende de outro ramo; projeção/gates próprios permanecem ativos. FiLM desligado é identidade.
- Causalidade, backward finito, memorização e retomada estrita com RNG/acumulação passam em CPU math.
- Documentos/splits não vazam; tokenizer usa treino apenas; hashes, contagens, cauda e split de avaliação são explícitos.
- Configs: ciclo 1 seeds 71/72/73, T256/B32/K2/2000 passos; confirmação 101–105/8000 passos. JEPA/NCM/ciclos assimétricos desligados.
- Planejamento é sem execução por padrão; budget global e seleção são verificáveis com testes e simulações locais.
- `python -m pytest -q`, checks pertinentes e limitações registrados em `VALIDATION.md`; nunca relatar skips CUDA como aprovação.

## Entrega científica

Cada ciclo registra fontes/configs, alvos, tempo, FLOPs estimados, parâmetros únicos, avaliação completa por checkpoint e seed. Limiares pré-registrados: 0,005 NLL para ganho de qualidade, 10% para eficiência, margem 0,002 NLL; confirmação com cinco seeds e intervalo de confiança. Novas tecnologias continuam condicionais conforme o roadmap.
