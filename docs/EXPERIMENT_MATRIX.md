# Matriz de experimentos independentes

Configs em `configs/experiments/`; lista legível por máquina em `matrix.yaml`. Nunca são
iniciadas em lote implicitamente. Sementes recomendadas: 17, 42, 123, passadas por `--seed`
com novos diretórios. Todas são smoke configs CPU; para campanha A100 dimensione modelos e
orçamentos conscientemente a partir dos perfis, preservando os mesmos dados/val/tokenizer.

| ID | Config | Variação |
|---|---|---|
| E0 | e0_baseline.yaml | Transformer original |
| E1 | e1_jepa.yaml | Transformer + loss preditiva |
| E2 | e2_recursive.yaml | Core compartilhado, duas iterações |
| controle E2 | e2_untied_control.yaml | Unrolling sem compartilhar cores |
| E3 | e3_lrm.yaml | Recursão + JEPA |
| E4 | e4_continuous.yaml | Passos latentes progressivos + supervisão intermediária |
| E5 | e5_mtp.yaml | Transformer + duas previsões adicionais |
| combinado | combined.yaml | LRM + MTP + refinamento + harness adaptativo |
| C0 | c0_transformer_uniform.yaml | Transformer, tarefas uniformes |
| C1 | c1_transformer_fixed.yaml | Transformer, currículo fixo |
| C2 | c2_transformer_adaptive.yaml | Transformer, currículo adaptativo |
| C3 | c3_lrm_uniform.yaml | LRM com JEPA, uniforme |
| C4 | c4_lrm_adaptive.yaml | LRM com JEPA, adaptativo |

Somente na branch fractal:

| ID | Config | Variação |
|---|---|---|
| F0 | f0_lrm.yaml | LRM sem multiescala |
| F1 | f1_multiscale_unshared.yaml | Três escalas, cores distintos, sem condicionamento |
| F2 | f2_multiscale_shared.yaml | Cores compartilhados entre escalas |
| F3 | f3_conditioned.yaml | F2 + FiLM por escala |
| F4 | f4_deeper.yaml | F3 + recursões adicionais |
| C5 | c5_flrm_uniform.yaml | FLRM uniforme |
| C6 | c6_flrm_adaptive.yaml | FLRM adaptativo |

## Protocolo para uma comparação científica

1. Fixar corpus/tokenizer/fingerprint, split, contexto, avaliação e sementes. C0–C6 devem
   manter a mistura textual e os geradores iguais; só a política/arquitetura deve variar.
2. Relatar separadamente parâmetros do backbone, frozen teacher, auxiliares e total.
   Para comparação por parâmetros, ajustar dimensões e confirmar contagem por código.
   Igual D não equivale a igual número de parâmetros; teacher também não é gratuito.
3. Fixar um teto de FLOPs de aprendizagem comparável (`max_train_flops` inclui sondagens)
   e relatar FLOPs realmente usados. A última etapa pode não caber; não esticar o teto.
   Os smoke YAMLs compartilham teto, mas não prometem terminar com FLOPs idênticos.
4. Registrar loss/qualidade versus FLOPs e primeira validação observada que atinge um limiar.
   Esse cruzamento amostrado é limite superior, não o instante exato de aprendizagem.
5. Executar benchmark de inferência com contexto, dtype, warmup, passos latentes e hardware
   fixos; comparar VRAM do allocator separadamente dos bytes de pesos calculados.
6. Repetir sementes e examinar generalização/linguagem/memorização. Um smoke test não resolve
   a hipótese científica. Só então formular uma conclusão limitada ao protocolo observado.

`report` exporta controles reais (mesmo dataset/contexto/parâmetros/sementes). Se forem
diferentes, não interpretar o gráfico como evidência causal. Relatórios sem benchmark/qualidade
mostram ausência de observações em vez de curvas fictícias.
