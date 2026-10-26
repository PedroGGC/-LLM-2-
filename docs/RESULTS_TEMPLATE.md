# Relatório de experimento — preencher com observações reais

- Hipótese e ablação:
- Commit/branch, YAML resolvido, versão de arquitetura e seeds:
- Dataset/tokenizer/fingerprint; splits e famílias; exemplos vistos versus inéditos:
- Hardware, PyTorch/CUDA, dtype, SDPA, compile e warmup:
- Variáveis mantidas constantes e diferenças conhecidas:
- Parâmetros do backbone, auxiliares, frozen e ativos (calculados):
- Fórmula/exclusões de FLOPs; treino, sondagem, validação e geração separados:
- Orçamentos declarados, computação efetiva, stop_reason, tempo total e créditos estimados:
- Loss LM/perplexidade; loss auxiliar/variância/colapso; exact match por habilidade/OOD:
- Primeira validação que atingiu o limiar escolhido; frequência de avaliação:
- Latência/TTFT/tokens por segundo e contexto; bytes de pesos versus VRAM medida:
- Igualdade de retomada, comandos e artefatos:
- Robustez entre sementes; incerteza; limitações e falhas:
- Conclusão restrita ao que os dados sustentam:

Gerar gráficos e CSV/JSON/Markdown com `python -m llm2 report --runs RUN_A RUN_B --output NOVO`.
Dados ausentes não são zero nem uma curva interpolada inventada. Não transformar aprovação
de testes unitários em conclusão sobre qualidade ou eficiência científica.
