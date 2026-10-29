# Campanhas Kaggle T4 — operação atual

Use a branch `experiment/fractal-lrm` para N1/G0, pareamento e telemetria da campanha
0015. Consulte [resultados](results/CAMPAIGN_0015.md) e [roadmap](RESEARCH_ROADMAP.md)
antes de escolher uma nova matriz. Os defaults históricos do runner **não selecionam
N1 automaticamente**; declare modelos e flags explicitamente.

## Planejar sem executar

No checkout experimental, para inspecionar a seleção mesmo sem CUDA:

```powershell
python scripts/run_kaggle.py --device cpu --input data/raw/tinyshakespeare.txt --models baseline_small flrm_compressed_native_cached_no_jepa_k2 --seeds 42 43 44 --steps 2000 --batch-size 32 --grad-accum-steps 1 --compile --paired-common-init --evaluate-all-checkpoints --diagnose-init --diagnose-steps 500 2000 --diagnose-batch-size 4
```

Sem `--execute`, esse comando apenas apresenta a seleção. `--device cpu` permite
a inspeção local; não descreve o hardware do experimento. O runner exige `--input`
ou `--data` mesmo no planejamento, mas só abre o dataset na execução. Para uma
execução T4 futura, seria necessário escolher `--device cuda`, fornecer dados
reais e revisar toda a configuração antes de adicionar `--execute`. Não é uma autorização para
rodar novas campanhas. Não fornece uma variante de normalização ainda não implementada.
O runner usa o Python e o src deste checkout, sem depender da instalação editável anterior.

## Notebook e revisão fixada

`notebooks/kaggle_t4.ipynb` é a entrada atual. Ele apresenta os settings com execução
desligada por padrão. A revisão de referência da 0015 é
`aa51b7d9dc541b00b84f587075f757078ecca262`; um novo experimento deve fixar seu próprio
commit publicado e registrar as configurações resolvidas. Não usar branch mutável como
identidade de reprodução nem misturar módulos copiados de revisões diferentes.

Ao habilitar a execução no Kaggle, o bootstrap chama `scripts/kaggle_notebook.py` da
revisão escolhida. Esse helper verifica GPUs/Torch, cria checkout fixado e diretórios
novos, instala dependências, executa testes e inicia somente o modo solicitado.
A inspeção local do notebook não valida seu funcionamento na GPU.

| Modo | Finalidade |
|---|---|
| `treinar` | Nova campanha; requer duas GPUs no helper |
| `avaliar_salvos` | Avaliar checkpoints completos de uma campanha anterior, sem novo treino |
| `perfilar_lotes` | Perfis e telemetria; usa uma GPU no helper |

T4x2 são placas independentes; não oferecem 30 GiB unificados a um modelo. O protocolo
usa FP32. Não presumir BF16/A100/FP16 validados por esses resultados.

## Controles e nomes

| Alias experimental | Interpretação |
|---|---|
| `baseline_small` | Transformer de um bloco; FFN 1280 no perfil D=384 |
| `lrm_no_jepa_k1`, `lrm_no_jepa_k2` | Núcleo Transformer recorrente, LM pura |
| `flrm_compressed_grouped_no_jepa_k1` | FLRM agrupado K=1 |
| `flrm_compressed_grouped_no_jepa_k2` | G0 |
| `flrm_compressed_native_cached_no_jepa_k2` | N1 |

Aliases K fixo prevalecem sobre `iterations` global. `lrm` e vários aliases FLRM sem
`no_jepa` ativam objetivo auxiliar; não usá-los inadvertidamente como controles LM pura.
O pareamento de tensores comuns cobre somente modelos recorrentes.

Nomes numéricos de campanha/revisão são reservados sem sobrescrita. Uma sessão sem outputs
anteriores pode reiniciar a numeração; escolha `run_number` livre quando necessário.
Não sugerir um número antigo fixo como se fosse a próxima campanha.

## Artefatos e métricas

Preservar os outputs completos e checkpoints da sessão. O `*-results.zip` exporta
relatórios, configs, hashes, logs e snapshots, mas exclui pesos/AdamW/binários de dados.
Ele permite análise, não retomada. No computador local, guardar em `artifacts/campaigns/`.

`summary.json` contém validação periódica, possivelmente parcial. A avaliação completa
fica em `evaluations/step-*/language.json` e pode ser reconstruída de `windows.jsonl`.
Comparar LM com LM, sem misturar loss auxiliar. Tempos/VRAM são medições; FLOPs são estimativas.

Perfis precisam separar warmup/compilação, medição sem profiler e trace instrumentado.
Registrar kernel SDPA observado, não apenas política solicitada. Atribuição fixa de
modelos às placas e jobs concorrentes podem afetar tempo: alternar/isolá-los quando
medir eficiência. Geração atual é eager, batch 1, sem KV cache.
