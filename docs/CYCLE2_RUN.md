# Próximo teste — ciclo 2 (29/10/2026)

## Entrega e uso no Kaggle

O executor está em `experiment/fractal-lrm`. A entrega vigente é a célula curta
`rodar_ciclo2_kaggle_2h.py`: baixa o bootstrap e faz checkout do mesmo commit completo
publicado no GitHub. Não exige anexar ZIP de fontes. Não reutilize o commit da 0016:
ele não contém este executor.

1. Ative Internet e GPU T4 (uma ou duas; com duas, o executor alterna as placas, sem treinos concorrentes).
2. Cole a célula curta entregue e execute com `RUN_CAMPAIGN = True`. Ela usa `LLM2_REVISION` fixado, instala o projeto com o Python atual, confere Torch e baixa TinyShakespeare com hash fixado. O checkout é novo e seus arquivos recebem hashes de proveniência.
3. `LLM2_SETTINGS["session_seconds"] = 7200` limita toda a sessão a duas horas. É teto, não duração obrigatória; se terminar antes, encerra imediatamente.
4. Ao terminar, baixe o ZIP `llm2-campaign-0017-results.zip` indicado na saída e preserve os outputs completos do notebook. O ZIP de análise não contém pesos/AdamW e sozinho não permite retomar treino.

O número 0017 é o próximo após a campanha auditada. Se já estiver ocupado/reservado
nos Inputs ou working, escolha outro `run_number` em `LLM2_SETTINGS`; nada existente
é sobrescrito. O antigo ZIP/launcher completo de quatro horas foi preservado como
entrega anterior; ele não é a célula vigente de duas horas.

Para gerar uma nova célula/notebook com um commit publicado, no checkout experimental:

```powershell
python scripts/build_cycle2_release.py --revision SHA_COMPLETO_PUBLICADO --output-dir dist/cycle2-git-2h-20261029
```

O notebook gerado inicia com execução desativada. O script `.py` inicia ativado,
como a célula solicitada. Não alterar modelos, seeds ou passos no notebook; o
protocolo fica congelado no código fixado. O caminho ZIP continua disponível para
entregas offline específicas, com verificação de hash e sem substituir arquivos.

## Protocolo congelado

- Seeds **81/82/83**, T=256, vocab=257 bytes, batch=32, acumulação=1, **2.000 passos** e scheduler=2.000; LR=0,0004/warmup=50/dropout=0,05, FP32/Inductor.
- Transformer pequeno (1 bloco/FFN 1280), LRM K=2 (1 core/FFN 1024) e quatro FLRM N1 K=2: comunicação × FiLM ligados/desligados. Contexto **sem RMS**, JEPA/MTP/currículo/NCM/ciclos assimétricos desligados.
- FLRM usa projeção de contexto em cache. LRM mantém o controle da 0016 sem esse cache; diferenças de parâmetros/arquitetura/custo permanecem explícitas.
- Avaliações completas nos passos 500/1.000/1.500/2.000, janelas com NLL e hashes. Diagnóstico inicial/final e geração/benchmark eager no final; todas as avaliações usam a primeira GPU para comparação.
- Inicialização pareada de pesos comuns; igualdade dos exclusivos compartilhados entre braços verificada ignorando somente FiLM ausente. FiLM desligado remove 2.304 parâmetros; mistura sem comunicação mantém seus pesos registrados, mas desconectados da loss.
- Sem comunicação, projeção/gate próprios permanecem e os ramos grossos são retornados como observadores destacados em `state.scale_values`, para preservar o forward na compilação sem introduzir gradiente artificial.

## Piloto e orçamento

O relógio global começa na célula, antes do download do bootstrap, e inclui checkout,
verificação, dependências, dados, testes, piloto, compilação, treino, avaliação e
exportação. Limite vigente **7.200 s**, com **300 s reservados para finalizar/exportar**.
Os 2.000 passos e scheduler permanecem fixos; não são reduzidos para caber no relógio.

O piloto roda primeiro para cada variante em cada GPU disponível: 10 passos de aquecimento e 30 amostras de tempo, seguidas de um passo instrumentado separado. Estima conservadoramente duração por bloco de seed; não escolhe hiperparâmetros. Trabalhos científicos são seriais, com ordem rotacionada e GPUs alternadas por seed.

Se um bloco não couber na previsão, ele não começa. Se algum subprocesso travar ou ultrapassar o prazo, o watchdog o interrompe e a campanha recebe `partial_budget`. Somente checkpoints completos anteriores são preservados; o passo interrompido não é publicado. O bootstrap tem também limite externo para encerrar o grupo de processos que iniciou. Reserva é uma proteção operacional, não garantia de cobrança do provedor nem contra desligamento externo do notebook.

Campanhas parciais não contam como completas. Para repetir um bloco ausente, use um output novo e o subconjunto correspondente de seeds no CLI; não amplie scheduler nem sobrescreva checkpoints. Não há retomada automática de toda a campanha neste executor.

## CLI no checkout experimental

Planejamento sem executar, inclusive em máquina sem CUDA:

```powershell
python scripts/run_cycle2.py --input caminho/tinyshakespeare.txt
```

Execução equivalente no ambiente T4, com dataset preparado idêntico à 0016:

```bash
python scripts/run_cycle2.py --execute --data /caminho/dataset --output /kaggle/working/llm2-campaign-0017 --gpus 0 1
```

O preparo novo pode mudar hash do manifesto pela proveniência. Os três hashes de conteúdo de `train.bin`, `val.bin` e `tokenizer.json` devem coincidir com a 0016; não há bypass científico silencioso. `--engineering-smoke` é outro protocolo: CPU, modelo pequeno, quatro passos e acumulação; não produz evidência científica sobre FLRM.

## O que enviar e interpretar

O pacote de resultados contém `campaign.json`, configurações resolvidas, fontes/proveniência, testes, piloto, avaliações completas, diagnósticos, benchmarks e `report/report.md`/JSON, `runs.csv` e `curves.csv`.

O relatório calcula efeitos de comunicação, FiLM e sua interação **por seed completa**, com intervalos t exploratórios quando houver ao menos duas seeds. Não mistura células ausentes. Menor NLL é melhor; não há custo igualado nem confirmação em outro corpus nesta rodada.

Parâmetros registrados são distintos de `parameters_with_gradients` (soma dos tamanhos dos tensores com gradiente, não contagem de elementos não nulos). Normas/FiLM/mistura/pooling ficam fora das estimativas densas de FLOPs; backward é conservador quando há ramos desconectados. Consulte [plano e riscos](PLAN_CYCLE2.md) e [validação](../VALIDATION.md).
