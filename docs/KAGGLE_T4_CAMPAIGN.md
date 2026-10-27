# Próxima campanha no Kaggle Notebooks — T4x2

Esta entrega implementa P01–P06. GPU é executada **no notebook do Kaggle pelo navegador**. Não é preciso instalar nada no computador pessoal. Localmente foram/serão feitos testes CPU; os testes CUDA deste pacote só confirmam suporte quando executados no Kaggle.

## Etapa atual: FLRM agrupado com/sem JEPA × Transformer (27/10/2026)

A campanha 8b8ac432c0ea selecionou o agrupamento como candidato de velocidade,
com ganho moderado e aumento de VRAM. O próximo ensaio isola o JEPA dentro dessa
mesma arquitetura. A célula fixada no novo commit deve usar:

```python
LLM2_SETTINGS = {
    "mode": "treinar",
    "steps": 2000,
    "seeds": [42],
    "models": ["flrm_compressed_grouped_no_jepa",
               "flrm_compressed_grouped", "baseline_small"],
    "iterations": 2,
    "progress_interval": 500,
    "profile": True,
    "profile_warmup_steps": 10,
    "profile_measure_steps": 30,
}
```

- **Sem JEPA:** FLRM `compressed_grouped`, K=2, somente loss de próximo token.
  Não constrói teacher/preditor. No perfil D=384/H=1024/vocab=257, tem 2.168.076
  parâmetros backbone/total de treino.
- **Com JEPA:** mesma arquitetura, K=2, JEPA .1/teacher EMA como no teste anterior;
  2.168.076 backbone/4.631.064 total de treino. Mantém escalas, FiLM e compressão.
- **Transformer pequeno:** referência K=1/sem JEPA, 2.164.224 parâmetros backbone/total.

Nessa ordem e com seed única, a GPU 0 recebe o FLRM sem JEPA e depois o Transformer;
a GPU 1 recebe o FLRM com JEPA. As duas placas treinam em paralelo. A distribuição
busca aproveitar a expectativa de treino mais curto sem teacher; medir tempo real.
Avaliação completa, geração e perfis continuam sequenciais na mesma GPU 0.

As configs dos FLRMs diferem somente em jepa_weight e diretório de saída. Testes
conferem pesos iniciais do backbone e o primeiro batch iguais com a mesma seed.
O consumo de RNG pelo preditor muda sorteios posteriores de dropout; não se promete
trajetória estocástica idêntica. Igual quantidade de tokens não significa igual
computação. A ablação do JEPA é o par FLRM; a comparação com Transformer é de pacotes.

Loss completa **menor = melhor**, velocidades **maior = melhor**, tempo **menor =
melhor**, VRAM menor = menor consumo para a mesma tarefa. A perda total de treino
com JEPA inclui termo auxiliar e não deve ser ordenada contra a loss LM pura.
Teacher não participa da geração; não presumir aceleração de inferência sem JEPA.
Seed 42 é triagem; se o efeito for relevante, repetir depois com 42/43/44.

O identificador histórico `flrm_no_jepa` continua selecionando o FLRM não
comprimido. Use **flrm_compressed_grouped_no_jepa** nesta comparação. Cada campanha
cria novos diretórios; mantenha checkpoints anteriores e salve versão com outputs.
O ZIP de análise não contém os pesos. GPU real continua sendo executada no Kaggle.

## Etapa anterior: desempenho do FLRM comprimido × Transformer (27/10/2026)

O foco atual usa somente as famílias Transformer e FLRM. Para medir esta etapa,
use a célula fornecida na entrega com estes ajustes:

```python
LLM2_SETTINGS = {
    "mode": "treinar",
    "steps": 2000,
    "seeds": [42],
    "models": ["baseline_small", "flrm_compressed",
               "flrm_compressed_native", "flrm_compressed_grouped"],
    "iterations": 2,
    "progress_interval": 500,
    "profile": True,
    "profile_warmup_steps": 10,
    "profile_measure_steps": 30,
}
```

São quatro execuções: Transformer pequeno, FLRM comprimido anterior como controle,
FLRM com projeção em resolução nativa e a mesma projeção com escalas curtas agrupadas.
Todas as variantes FLRM têm os mesmos pesos em quantidade, K=2 e objetivos.
O Transformer tem K=1/sem JEPA; a comparação entre famílias é de pacotes completos.
Não é uma ablação isolada do benefício do JEPA ou da recursão.
O LRM permanece no histórico e pode ser selecionado explicitamente, mas sai do padrão.

O treino é distribuído entre as duas T4. Depois, avaliação/geração e perfis são
sequenciais na GPU 0. O perfil agora inclui **todos** os modelos solicitados,
uma seed por variante, com 10 passos de aquecimento, 30 amostras sem profiler e
um passo separado instrumentado. Os perfis usam treinos descartáveis novos,
não retomam nem alteram os checkpoints finais. O console informa alvos/s isolados;
o ZIP inclui `profiles/<modelo-seed>/profile.json` e `trace.json`.

Loss completa **menor = melhor**; treino em alvos/s e geração em tokens/s
**maior = melhor**; tempo **menor = melhor**; VRAM menor = menor consumo para
a mesma tarefa. FLOPs são estimativas v3 corrigidas, não medidas de tempo.
Uma seed é triagem; não concluir superioridade consistente com esse ensaio.
Para a ablação posterior de K, repetir em campanha nova mudando apenas
`iterations` para 1, mantendo as mesmas variantes, dados e demais ajustes.

Os novos testes CUDA FP32 rodam antes do treino; uma falha interrompe a campanha.
Os testes locais são CPU; o suporte real das otimizações nas T4 ainda precisa
desta execução. Os diretórios e checkpoints anteriores são preservados.

## Fluxo disponível: comparar checkpoints já salvos

Esse modo não inicia treino. Compara a qualidade dos checkpoints de 500, 1.000, 1.500 e 2.000 passos, mantendo os pesos e dados originais. Loss de validação **menor = melhor** previsão dos próximos tokens. Não mede novamente velocidade.

Use esta célula com o commit informado na entrega e a pasta da campanha anterior:

```python
import os
from urllib.request import urlopen

REVISION = "COLE_O_COMMIT_DE_40_CARACTERES_DA_ENTREGA"
os.environ["LLM2_REVISION"] = REVISION
LLM2_SETTINGS = {
    "mode": "avaliar_salvos",
    "previous_campaign": "/kaggle/working/llm2-campaign-8f86ba9cc48b",
    "review_steps": [500, 1000, 1500, 2000],
}
url = f"https://raw.githubusercontent.com/PedroGGC/-LLM-2-/{REVISION}/scripts/kaggle_notebook.py"
exec(compile(urlopen(url, timeout=60).read().decode("utf-8-sig"), url, "exec"), globals())
```

Se a sessão anterior acabou, anexe os outputs da versão salva ao notebook e ajuste `previous_campaign` para a pasta restaurada, por exemplo em `/kaggle/input/.../llm2-campaign-8f86ba9cc48b`. Essa pasta deve conter `campaign.json`, `data` e `runs` com os pesos/estado dos checkpoints. O ZIP de análise contém apenas metadados e não basta. Se o dataset original foi salvo em outra pasta, informe também `"data": "/caminho/para/dados-originais"`; seu fingerprint precisa continuar igual.

O bootstrap preserva ou recupera o checkout do commit original para avaliar os pesos antigos. O novo executor verifica hashes do código/dados/pesos e não carrega o pickle do optimizer. A avaliação completa de um checkpoint já presente pode ser reutilizada somente após conferir identidade, escopo e médias por janela. Novas avaliações e seu ZIP ficam em outra pasta `llm2-review-...`. A comparação seleciona o menor erro observado na validação; não substitui teste independente.

## Executar uma nova campanha de treino

Ative Internet e o acelerador GPU T4 x2. Cole a célula abaixo, usando o **commit de 40 caracteres informado na entrega** em REVISION. A célula baixa o bootstrap daquele commit e executa o fluxo no Kaggle.

    import os
    from urllib.request import urlopen

    REVISION = "COLE_O_COMMIT_DE_40_CARACTERES_DA_ENTREGA"
    os.environ["LLM2_REVISION"] = REVISION
    LLM2_SETTINGS = {
        "mode": "treinar",
        "steps": 2000,
        "seeds": [42],
        "models": ["baseline", "lrm", "flrm_reference", "flrm_batched"],
        "profile": True,
    }
    url = f"https://raw.githubusercontent.com/PedroGGC/-LLM-2-/{REVISION}/scripts/kaggle_notebook.py"
    exec(compile(urlopen(url, timeout=60).read().decode("utf-8-sig"), url, "exec"), globals())

O bootstrap cria um checkout separado pelo commit, usa o Python do notebook para instalar dependências e não instala um wheel CPU de Torch por cima da instalação CUDA. Não faz pull sobre os experimentos existentes. O corpus padrão é tinyshakespeare (~1,1 MB) de uma revisão fixada de char-rnn. Documentos são parágrafos, com deduplicação exata antes do split; o tokenizer byte não usa validação para aprender vocabulário. Isso continua sendo um teste pequeno de modelagem de bytes, não linguagem geral.

### Como acompanhar a saída

- Preparação: hardware, código/dependências e início da campanha.
- Treino: etapas numeradas, modelo/seed/GPU física e percentual a cada 100 passos. O tempo restante é uma estimativa, não uma promessa. Cada GPU recebe um processo por vez; as duas não somam VRAM para um único modelo.
- Loss LM de validação: **menor = melhor** previsão. A saída durante o treino é marcada como **PARCIAL**; a tabela final usa todas as janelas completas.
- Velocidade de treino/geração: **maior = melhor**, com unidades explícitas. Treino e geração são tarefas diferentes e suas velocidades não devem ser misturadas.
- Memória VRAM: **menor = menor consumo de recurso** para a mesma tarefa. Parâmetros: menos significa modelo menor; não garante melhor qualidade.
- Geração: mostra o texto produzido. Diagnósticos e medições técnicas ficam nos logs; a tabela em português também é salva em `report/resumo-pt.md`.
- `n/d`: medição indisponível, não zero. Qualquer falha interrompe o processo; o indicador de conclusão só aparece após sucesso real.

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
