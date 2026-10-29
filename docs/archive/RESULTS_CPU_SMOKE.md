> **Registro histórico — não é o plano vigente.** Arquivado em 28/10/2026.
> O [estado atual](../PROJECT_STATUS.md), os [resultados corrigidos da 0015](../results/CAMPAIGN_0015.md)
> e a [validação vigente](../../VALIDATION.md) prevalecem sobre descrições e conclusões antigas.

# Sanidade integrada em CPU — 26/10/2026

Resultados de engenharia, sem campanha científica, CUDA ou créditos de GPU.
Python local 3.12.2, Torch 2.14.1+cpu, FP32, SDPA math, seed 42, um thread.
Instalação direta, sem venv. Matplotlib foi incluído como dependência opcional de pesquisa.

## Execuções observadas

Corpus sintético preparado da etapa 1, tokenizador byte (vocab 257), contexto 128,
microbatch 1, acumulação 1 e dropout 0,1. Não são os modelos de referência de 34M.
Parâmetros totais incluem auxiliares e teacher congelado, sem duplicar pesos compartilhados.
Tokens são alvos supervisionados / entradas densas processadas; padding e prompts ignorados
na loss explicam diferenças nos currículos. FLOPs são estimativas analíticas, com exclusões
em ARCHITECTURE_LRM.md; aprendizado inclui probes, validação é contabilizada separadamente.

| Execução | Passos | Parâmetros backbone / totais | Tokens alvos / entradas | Validation LM loss | FLOPs aprendizado estimados | Parada |
|---|---:|---:|---:|---:|---:|---|
| e0 | 4 | 98,688 / 98,688 | 512 / 512 | 5.06318641 | 402,849,792 | max_steps |
| e3 | 4 | 61,824 / 131,840 | 512 / 512 | 5.13053250 | 595,656,704 | max_steps |
| combined | 4 | 61,824 / 148,288 | 386 / 512 | 5.19848847 | 1,974,172,160 | max_steps |
| f3 | 4 | 66,316 / 140,824 | 512 / 512 | 5.12172031 | 897,646,592 | max_steps |
| c6 | 2 | 66,316 / 140,824 | 130 / 256 | 5.37231207 | 1,795,342,336 | learning_flop_budget |

C6 atingiu o limite de 2 bilhões antes do próximo passo, incluindo o custo das sondagens.
Todos produziram loss finita, checkpoint, métricas e avaliação por geração livre. Exact match
no teste foi **0** nos cinco casos, com um exemplo por habilidade (nove no grupo de teste).
As saídas e falhas individuais estão nos JSONs; os dados não demonstram competência ou ganho.
A mistura texto/procedural é por microbatch, portanto proporções observadas em poucos passos
podem diferir bastante da fração configurada.

Relatórios locais gerados: `runs/v2-20261026/report-main/` e, no worktree experimental,
`runs/v2-20261026/report-fractal/`. Cada um contém report.json, series.csv, report.md e cinco
SVGs: loss/FLOPs, qualidade/FLOPs, parâmetros, bytes de pesos e throughput de geração.
Benchmarks são uma observação por modelo, batch 1, 16 tokens gerados, warmup 2, carga de pesos
excluída, geração eager sem KV cache. Concorrência/carga do computador não foram controladas;
esses números exercitam instrumentação e não permitem ranking. CPU usa null para VRAM e tempo
GPU. Os JSONs preservam config, versões, limitações e hash dos pesos avaliados.

## Testes e retomada

Branch comum: **66 passed, 3 skipped** (68,16 s). Branch fractal: **74 passed, 4 skipped**
(73,48 s). Skips exclusivamente CUDA/BF16. Foram executados causalidade, pooling causal,
forward/backward, finitude, memorização, custos/compartilhamento, EMA, MTP, currículo, budgets,
proveniência, checkpoint e comparação da próxima atualização com dropout e acumulação 2.
Torch compile só foi exercitado com backend eager; Inductor e CUDA continuam pendentes.

E3 em processos independentes: quatro passos ininterruptos versus pausa após dois e retomada
até quatro. Todos os tensores safetensors, objetivo/teacher, otimizador, scheduler, sampler,
RNG e contadores sem tempo foram exatamente iguais. O artefato é
`runs/v2-20261026/resume-equivalence.json`. A ordem das chaves JSON de metadados safetensors
variou; os tensores e aliases foram preservados. Checkpoints V1 permanecem inalterados;
a avaliação V2 de seus pesos foi confirmada. Retomar V1 exige seu snapshot de código original.

## Reprodução

Use o mesmo Python que recebeu as dependências. Neste computador:
`C:/Users/Lux/.pyenv/pyenv-win/versions/3.12.2/python.exe`; os comandos abaixo usam `python`
por legibilidade. Cada nome de saída deve ser novo. Não repetir sobre os resultados existentes.

```powershell
python -m pip install -e ".[dev,research]"
python -m pytest -q
python -m llm2 train --config configs/experiments/e0_baseline.yaml --output-dir runs/reproduce-e0
python -m llm2 train --config configs/experiments/e3_lrm.yaml --output-dir runs/reproduce-e3 --until-step 2
python -m llm2 train --config configs/experiments/e3_lrm.yaml --output-dir runs/reproduce-e3 --resume runs/reproduce-e3/checkpoints/step-00000002
python -m llm2 train --config configs/experiments/combined.yaml --output-dir runs/reproduce-combined
python -m llm2 benchmark --checkpoint runs/reproduce-e3/checkpoints/step-00000004 --max-new-tokens 16 --warmup-tokens 2 --output runs/reproduce-e3/benchmark.json
python -m llm2 reasoning --checkpoint runs/reproduce-e3/checkpoints/step-00000004 --examples 1 --max-new-tokens 32 --output runs/reproduce-e3/reasoning.json
python -m llm2 report --runs runs/reproduce-e0 runs/reproduce-e3 runs/reproduce-combined --output runs/reproduce-report
python scripts/export_results.py --run runs/reproduce-e3 --output dist/reproduce-e3-results.zip
python scripts/package_colab.py --output dist/reproduce-v2-colab.zip
```

Para F3/C6, siga os comandos com PYTHONPATH em ARCHITECTURE_FLRM.md; execute cada YAML
individualmente. Nenhum script dispara a matriz inteira. `plan-budget` é cálculo estimado;
`experiment` e `profile` exigem `--execute`. O notebook também exige opt-in para treino,
avaliação, GPU smoke e exportação.

## Estado versionado e limites

- `master`: b60bd9c preserva integralmente a etapa 1 antes das mudanças V2.
- `feature/llm2-research-v2`: núcleo comum, objetivos, currículo, ferramentas, configs e Colab.
- `experiment/fractal-lrm`: worktree separado; código fractal nunca mesclado na branch comum.
- Marcos comuns: 1614e77 e 761a46a; implementação fractal: 8dde55e; integração comum no
  worktree: 7147ce6. Commits de fechamento são identificáveis por `git log -3 --oneline`.
- Nenhum remote configurado: commits são locais, sem publicação ou push.

A100/BF16/SDPA fused/Inductor, múltiplas sementes, corpora reais, controles com FLOPs/parâmetros
iguais e treino prolongado continuam pendentes. Recursos futuros e riscos abertos estão em
RESEARCH_ROADMAP.md e RESEARCH_V2_PLAN.md. Nada nesta entrega demonstra ganho de eficiência.
