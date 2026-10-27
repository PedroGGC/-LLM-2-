# Próxima campanha no Kaggle Notebooks — T4x2

Esta entrega implementa P01–P06. GPU é executada **no notebook do Kaggle pelo navegador**. Não é preciso instalar nada no computador pessoal. Localmente foram/serão feitos testes CPU; os testes CUDA deste pacote só confirmam suporte quando executados no Kaggle.

## Executar

Ative Internet e o acelerador GPU T4 x2. Cole a célula abaixo, usando o **commit de 40 caracteres informado na entrega** em REVISION. A célula baixa o bootstrap daquele commit e executa o fluxo no Kaggle.

    import os
    from urllib.request import urlopen

    REVISION = "COLE_O_COMMIT_DE_40_CARACTERES_DA_ENTREGA"
    os.environ["LLM2_REVISION"] = REVISION
    LLM2_SETTINGS = {
        "steps": 2000,
        "seeds": [42],
        "models": ["baseline", "lrm", "flrm_reference", "flrm_batched"],
        "profile": True,
    }
    url = f"https://raw.githubusercontent.com/PedroGGC/-LLM-2-/{REVISION}/scripts/kaggle_notebook.py"
    exec(compile(urlopen(url, timeout=60).read().decode("utf-8-sig"), url, "exec"), globals())

O bootstrap cria um checkout separado pelo commit, usa o Python do notebook para instalar dependências e não instala um wheel CPU de Torch por cima da instalação CUDA. Não faz pull sobre os experimentos existentes. O corpus padrão é tinyshakespeare (~1,1 MB) de uma revisão fixada de char-rnn. Documentos são parágrafos, com deduplicação exata antes do split; o tokenizer byte não usa validação para aprender vocabulário. Isso continua sendo um teste pequeno de modelagem de bytes, não linguagem geral.

Para usar os dados preparados anteriores, acrescente a LLM2_SETTINGS:

    "data": "/kaggle/input/SEU_DATASET/processed-real",

A pasta deve conter manifest.json, tokenizer.json, train.bin e val.bin originais. Para arquivos brutos próprios, use em vez de data:

    "input": ["/kaggle/input/SEU_DATASET/corpus.txt"],

Nunca sobrescreva um dataset preparado. Dados novos ficam dentro do novo diretório da campanha. A preparação padrão não é uma reprodução garantida da campanha histórica: compare fingerprints. O nome do arquivo/origem também participa do manifest.

Se os checkpoints antigos estiverem disponíveis e você selecionar o dataset preparado original, pode acrescentar:

    "evaluate_checkpoints": [
        "/kaggle/input/SEUS_CHECKPOINTS/baseline/step-00002000"
    ],

A avaliação exige que o manifest corresponda ao checkpoint; nenhuma verificação é relaxada. O processo de avaliação não carrega o pickle do optimizer, mas os arquivos esperados e seus hashes precisam estar presentes.

## O que a execução faz

1. Confere duas GPUs e o checkout fixado.
2. Executa a suíte CPU e seleciona os testes CUDA FP32 para T4, que exercitam Transformer, LRM e, na branch fractal, os dois modos do FLRM. Testes BF16 não fazem parte dessa seleção; não se confunde emulação com validação de BF16 nativo.
3. Executa quatro modelos, com no máximo um processo de treino por GPU. As duas T4 não formam VRAM unificada.
4. Usa os resumos da execução recém-concluída para localizar/verificar checkpoints. Qualquer erro de subprocesso interrompe a campanha, preserva logs e gera um ZIP parcial.
5. Após os treinos, avalia **toda a validação** e salva loss por janela para os quatro modelos.
6. Mede geração sem KV cache e gera uma amostra; executa diagnósticos opcionais fora do treino.
7. Mede os dois modos FLRM sequencialmente na mesma GPU, com aquecimento, amostras sem profiler e um passo separado instrumentado.
8. Exporta report.json, comparisons.csv, séries, gráficos, logs, configurações, manifests e traces.

Cada execução tem identificador e diretórios novos. As execuções de perfilamento são descartáveis e independentes do treino científico. O ganho de performance do modo agrupado ainda precisa ser medido na T4.

Se a campanha parar em `tests-cpu.log`, antes de `training`, nenhum modelo foi treinado. O ZIP parcial contém os logs da falha. Na primeira entrega, um teste herdava a máscara unitária de GPU do subprocesso e pedia indevidamente o índice lógico 1; a correção isola essa premissa e valida índices sem escapar da máscara do processo pai. Use o novo commit informado na entrega e mantenha os testes habilitados. O bootstrap cria outro checkout e outra campanha; os diretórios anteriores ficam preservados. Os dados preparados podem ser reutilizados via `data` se os quatro arquivos originais ainda existirem.

## Modelos e controles

- baseline: Transformer 10.720.512 parâmetros no perfil padrão, sem JEPA.
- lrm: núcleo compartilhado com duas iterações e JEPA.
- flrm_reference: três escalas, uma iteração, execução sequencial e JEPA.
- flrm_batched: mesmos parâmetros/operações matemáticas, escalas agrupadas e JEPA.
- Opcionais: baseline_small (~2,164M no perfil padrão), baseline_jepa, lrm_no_jepa e flrm_no_jepa.

O default reproduz uma comparação de pacotes próximos aos históricos. Não é uma comparação iso-parâmetros ou iso-FLOPs; auxiliares, profundidades e tamanho diferem. O controle reference/batched isola organização de execução, mas dropout e arredondamentos podem mudar a trajetória de treino. Não esperar a mesma loss final bit a bit entre esses dois treinos; testes numéricos controlados verificam a função/gradientes, e retomada exata é exigida dentro do mesmo modo.

Depois do piloto, uma seleção com os braços sem/com JEPA e múltiplas seeds pode usar:

    "models": ["baseline", "baseline_small", "lrm_no_jepa", "lrm", "flrm_no_jepa", "flrm_batched"],
    "seeds": [17, 42, 123],

Isso é uma campanha maior; não é executada implicitamente. Ajustar modelos ou seeds altera o número de treinos. Configurações resolvidas são exportadas para cada execução.

## Resultados que devem voltar para a análise

Baixe o arquivo indicado no fim, semelhante a:

    /kaggle/working/llm2-campaign-IDENTIFICADOR-results.zip

Envie esse ZIP para a próxima análise. Ele inclui as evidências pequenas e traces, mas **não** pesos safetensors, optimizer ou dados binários; portanto não é um backup retomável. Os checkpoints completos continuam no diretório da campanha. Salve uma versão do notebook Kaggle com outputs para preservá-los.

Em caso de erro, o ZIP parcial é gerado e os logs integrais mostram a falha. Se a interrupção vier de desligamento do serviço/kernel, a exportação final pode não acontecer; os logs e checkpoints já escritos permanecem enquanto o armazenamento da sessão existir.

## Semântica das métricas

- validation_loss no summary de treino é a avaliação periódica limitada, marcada first_batches.
- evaluations/step-*/language.json contém a avaliação final CE-only de todas as janelas completas, marcada all_full_windows.
- windows.jsonl traz índice, offset do primeiro alvo, soma de NLL e número de alvos da janela. Reconstitua a média ponderada, admitindo arredondamento FP32.
- tokens_per_second é a média acumulada de treino; step_tokens_per_second é só o passo.
- optimization_loss é o escalar efetivamente acumulado para backward. training_loss mantém a agregação histórica ponderada pelos tokens; pode divergir da loss composta otimizada quando há máscaras.
- accumulation_weighting=tokens pondera a componente LM por alvos válidos no batch efetivo. Auxiliares continuam médias por microbatch; a penalidade de variância não é aditiva. O modo microbatch histórico continua disponível.
- jepa_hidden_only evita a projeção de vocabulário do teacher. Os pesos continuam compartilhados com o embedding; os FLOPs estimados foram ajustados, sem contar essa cabeça descartada.
- GPU time por eventos é intervalo de execução, não soma dos tempos dos kernels. Pico allocated/reserved é do allocator PyTorch, não consumo total da placa.
- A política SDPA solicitada não certifica backend usado: veja observed_attention_operators no perfil.
- Diagnósticos CKA/variância/posto/gates não provam especialização ou qualidade. São calculados sobre amostra limitada de estados e ficam fora dos tempos de benchmark.

## Preservação e próxima etapa

As mudanças alteram o hash do código. Retomar exatamente uma execução antiga exige seu código/configuração/ambiente originais; não afrouxamos esse contrato. Avaliar pesos antigos compatíveis ou transferi-los é diferente de retomar seu treinamento.

Fibonacci, multirrate/Haar, memória persistente, ACT/DEQ, Kronecker/Monarch e NCM continuam como experimentos posteriores, condicionados às medições desta entrega. Nenhuma redução de VRAM/tempo ou estabilidade BF16 foi presumida.
