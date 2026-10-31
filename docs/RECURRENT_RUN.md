# Executar seleção recorrente e experimento fractal

31/10/2026. [Plano aprovado](PLAN_RECURRENT_FRACTAL.md), [fichas](RECURRENT_CANDIDATES.md).
Etapas A/B implementadas. A [auditoria da0024](results/CAMPAIGN_0024.md) encontrou
timeouts absorvidos pelo compilador: mLSTM é candidato, mas a qualificação limpa
continua pendente. A nova célula requalifica antes de qualquer treino de corpus.
Nenhum treino de produção é iniciado na instalação ou sem `--execute`.

## A: qualificação de plataforma

Cole o [script curto](../scripts/03_selecionar_nucleo_recorrente_kaggle.py) numa
célula Kaggle com Internet e duas T4. A revisão e o SHA do bootstrap estão fixados;
numeração começa em25 e avança se o número já existir. Não inicia automaticamente
a etapa B. O relógio começa na própria célula, antes de baixar/clonar/instalar.

No checkout experimental, Python local já instalado; no Kaggle habilitar duas T4.

```bash
python -m llm2.research.state_study --execute --phase qualify --output /kaggle/working/llm2-state-platforms
```

Não precisa dataset. Testes CPU primeiro; cada candidato compara CUDA/Inductor
contra sua referência sequencial antes do piloto. Uma falha de kernel rejeita
aquele candidato e não impede avaliar os demais. Seis pilotos isolados: três regras
nas duas placas. O processo supervisor impõe600s por caso, incluindo compilação,
e encerra o grupo do worker no Linux; uma exceção absorvida pelo compilador não
prolonga esse limite. Há também watchdog global. Usa formas estáticas na
qualificação. Publica `platforms.json` v2 com candidato
selecionado, erros, ambiente e hashes. Nenhuma seleção por CPU ou dado de teste.
O bootstrap disponibilizado com a entrega fixa revisão Git; preserva Torch e
inclui setup no relógio. Meta2h/teto3h, reserva5min para exportação.

Para liberar temporários existentes, execute antes o
[script de limpeza](../scripts/04_limpar_temporarios_kaggle.py). Ele não exige
outro backup: preserva ZIPs finais, campanhas, dados preparados e checkpoints;
remove somente caches de compilação e checkouts Git limpos. Exibe o inventário
antes/depois. Workers ativos, arquivos desconhecidos e links externos bloqueiam
a exclusão correspondente. Não esvazia indiscriminadamente o working.

Novas sessões buscam apenas a revisão Git fixada, sem histórico completo, e
descartam o checkout limpo ao terminar. Cache é removido após cada candidato e
no encerramento. Limite monitorado2GiB, amostragem5s, reserva livre2GiB; ao atingir
limite a sessão para e exporta evidência, sem apagar pesos. Compilação futura
precisará ser medida novamente após a limpeza; essa economia de disco não é
apresentada como aceleração.

No final, o bootstrap gera botões de download embutidos (`data:`/base64),
sem construir endereços do proxy. Limite24MiB por botão. ZIPs maiores oferecem
`LLM2_BAIXAR('/caminho/arquivo.zip', part=1)` e instruções para reunir e verificar
as partes localmente; não duplica partes no working. Baixe uma parte de cada vez.
O uso do atributo HTML está documentado no
[MDN](https://developer.mozilla.org/en-US/docs/Web/HTML/Reference/Elements/a#download).

## B: hierarquia e compartilhamento

Nova sessão, mesmo código/ambiente da seleção, com o `platforms.json` preservado:

```bash
python -m llm2.research.state_study --execute --phase experiment --platform-report /CAMINHO/platforms.json --cache /kaggle/working/llm2-wikitext-pinned-cache --download --output /kaggle/working/llm2-state-experiment
```

Rejeita seleção com fonte/versões diferentes, anterior à guarda v2 ou sem dois
resultados T4 qualificados. O resultado0024 não é migrado silenciosamente.
Na etapa B, os testes CUDA de hierarquia abrangem a regra selecionada, não exigem
que todas as alternativas rejeitadas tenham kernels compatíveis.
Não chama desempenho da referência CPU de desempenho da arquitetura publicada.
Backbone usa as regras da ficha com RMS/SwiGLU comuns, sem SDPA/DecoderBlock.

Quatro braços x seeds241/242/243: recorrente simples, hierarquia independente,
hierarquia compartilhada e Transformer2blocos; T1024/B8, FP32/Inductor, dropout0,05,
6000passos. Faixas de parâmetros calculadas por enumeração/fórmula:

| Regra | Simples / hierarquia independente | Hierarquia compartilhada | Transformer |
|---|---:|---:|---:|
| Gated Delta | 2.174.232 | 1.136.652 | 2.164.992 |
| RWKV-7 | 2.166.528 | 1.132.800 | 2.164.992 |
| mLSTM | 2.174.232 | 1.136.652 | 2.164.992 |

Os espaços são iguais (D384), projeções entre escalas são identidades. Simples
aplica duas células a todos os bytes; hierarquia aplica uma aos bytes e outra aos
grupos de quatro estados finos contextualizados. Assim há menos atualizações no
segundo nível. Sharing reutiliza a célula, não a memória. Ambas hierarquias executam
as mesmas operações; não alarga a versão compartilhada para igualar parâmetros.

### Dados e avaliação

Dados originais WikiText do ciclo3 são verificados integralmente e não sobrescritos.
Nova política explícita: janelas de treino e avaliação nunca cruzam EOS/documentos;
estado reinicia por janela. Validação exclui EOS da NLL, permitindo bits/byte=NLL/ln2.
Caudas incompletas de cada documento ficam fora igualmente para todos os braços.
Consequentemente NLL não é comparação direta com a avaliação packed da0023;
esse contrato faz parte do novo protocolo e não reinterpreta resultados antigos.

Taxas0,0002/0,0004, seed901, 750passos por tentativa (mesma quantidade por braço).
Divisão80/10/10 por documentos SOMENTE do treino original, com hashes conferidos;
terceira parte interna não é consultada. Nenhum tokenizer aprende com validação.
Depois reinicia cada job principal do zero sobre o treino original. Avaliações
completas em3000/6000; subconjunto fixo em1500/4500. Não consulta teste oficial.

### Orçamento e continuidade

Pilotos isolados em ambas GPUs e um piloto concorrente, margem35%, reserva20min
pós-treino mais5min de exportação. Dois workers independentes. Se12jobs não couberem,
admite somente blocos completos de quatro variantes por seed e registra pendentes;
não diminui passos ou scheduler. Exige2GiB livres segundo estimativa.

Para continuar outros blocos, reutilize decisões de taxa e selecione seeds pendentes:

```bash
python -m llm2.research.state_study --execute --phase experiment --platform-report /CAMINHO/platforms.json --tuning-report /CAMINHO/campaign.json --seeds 242 243 --data /CAMINHO/DADOS_PREPARADOS --output /NOVO/DIRETORIO
```

O tuning congelado exige os mesmos fonte/dataset/regra/protocolo. Não retoma checkpoints
incompatíveis. Resultados de sessões distintas precisam identidade compatível e seeds
sem duplicação; não contar réplicas como seeds novas.

```bash
python -m llm2.research.state_merge RESULTADOS_1.zip RESULTADOS_2.zip --output resumo_novo.json
```

O agregador valida identidade e recusa duplicatas. Não reúne amostras temporais de
sessões diferentes como repetições independentes; usa uma série isolada completa
explicitamente identificada. Curvas guardam alvos e tempo operacional medidos;
gráfico usa apenas validações completas, sem conectar NLL do subconjunto à completa.

Um par ZIP temporário aos50% dos jobs admitidos; finais têm resultados e último
checkpoint completo de cada job. Remove o temporário somente após CRC/SHA e cobertura
comprovada pelos definitivos. Falha/deadline preserva evidência e checkpoints completos.
Não gera `after-NN` a cada job. Pesos/ZIPs/cache permanecem fora do Git.

## Limitações e próximos mecanismos

CPU smoke exercita mesma orquestração com4passos/T17 e é rotulado engenharia. Regras
novas ainda sem estimador analítico: FLOPs null; Transformer conserva estimador antigo.
Perfis pós-treino são clones com pesos finais e otimizador novo, não retomada científica.
Energia pós-treino usa janela de até30s quando NVML permite; não representa host
nem energia total da campanha. Benchmark de geração/KV comparável fica null/pendente.
RAM/VRAM/energia têm escopo explícito; CPU VRAM/energia GPU null. Sem garantia de
aceleração por compartilhar pesos ou compilar. Uma seed não promove hipótese.

Etapas C/D são condicionais: não implementados GRM, TTT, ACT, DEQ, quantização,
compressão adaptativa, TinyStories ou escala7–10M nesta entrega. Catálogo e critérios
ficam no plano; não ativar mecanismos apenas pelo nome/contexto/K.
