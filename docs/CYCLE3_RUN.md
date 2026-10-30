# Ciclo 3 — confirmação independente no Kaggle

**Operação vigente:** [lotes de até uma hora](CYCLE3_SHORT_RUN.md), escolhidos após a
campanha 0019. Este documento preserva o protocolo científico e o executor anterior
de quatro horas para reprodução histórica; não é a próxima célula a executar.

Protocolo de 29/10/2026, em `experiment/fractal-lrm`. Nenhum mecanismo novo:
Transformer, LRM K=2 e FLRM com comunicação/cache e sem FiLM. Contexto sem RMS;
JEPA/MTP/NCM/currículo/Ciclo-V desligados. Este guia substitui o ciclo 2 como próxima
execução; [a campanha 0017](results/CAMPAIGN_0017.md) permanece a evidência mais recente.

## Dados e comparações fixados

WikiText-103 raw em inglês, revisão `b08601e04326c79dfdd32d625aee71d232d685c3`,
quatro shards com SHA-256 verificado. Artigos inteiros, byte-level, EOS entre documentos,
concatenação causal sem isolamento por máscara. Seleção do treino por ordem de hash,
incluindo o último artigo intacto: **1.758 artigos / 33.568.955 bytes**, aproximadamente
32 MiB. Validação: 60 artigos / 1.144.308 tokens; teste: 62 / 1.287.718 tokens.
Validação usa todas as janelas completas de 256 e declara a cauda descartada.

A auditoria encontrou 442 cópias exatas internas ao treino e uma entre treino e
teste: “The Hustler (film)”, hash
`c10b5100b0181cc148fa30970a9acaac0780277653f3288ee4883ed1c85bb78f`.
Essa cópia é removida **somente do treino, antes da seleção**; validação e teste
oficiais são preservados. A regra está explícita no importador/manifesto. Qualquer
outra duplicação exata entre divisões interrompe o preparo. Não é deduplicação
semântica ou por trechos. Hashes tokenizados estão congelados no protocolo.

Seeds **101–105**, T256, batch efetivo 32, FP32/Inductor, LR 0,0004, warmup 50,
dropout 0,05 e demais hiperparâmetros da 0017. Pesos inicializados do zero.

| Ensaio / braço | Passos fixos | Parâmetros registrados |
|---|---:|---:|
| Alvos iguais: Transformer | 8.000 | 2.164.224 |
| Alvos iguais: LRM K=2 | 8.000 | 2.017.536 |
| Âncora FLRM, compartilhada pelos dois ensaios | 8.000 | 2.165.772 |
| FLOPs estimados igualados: Transformer | 19.472 | 2.164.224 |
| FLOPs estimados igualados: LRM K=2 | 10.620 | 2.017.536 |
| FLOPs estimados igualados: Transformer de dois blocos / FFN1024 | 11.396 | 3.639.552 |

São seis treinos por seed, 30 no protocolo inteiro. A âncora reutilizada não é
replicação adicional. Cada treino salva e avalia completamente 25/50/75/100% do seu
horizonte; o endpoint primário é o último, sem escolha do melhor checkpoint.
Scheduler e horizonte são fixados antes dos ensaios, sem extensão para ocupar sobra.
O pareamento de FLOPs usa a fórmula existente, arredondando para baixo a múltiplos
de quatro; erro relativo inferior a 0,03%. Operações elementares, pooling e mistura
ficam excluídas. Isso **não** iguala tempo real nem parâmetros.

## Célula curta e limite de tempo

Ative **Internet** e **GPU T4 x2**. Use a célula curta entregue com SHA completo de
commit publicado. O bootstrap clona essa revisão, instala com o Python do notebook,
verifica que Torch foi preservado e baixa/prepara os shards. Não precisa anexar ZIP
de código ou dataset. Cache válido pode ser reaproveitado; outputs não são sobrescritos.

### Correção de portabilidade após a primeira tentativa

A revisão `855c8d2` falhou no Kaggle antes de iniciar testes/treino: o hash fixado de
`tokenizer.json` correspondia aos finais de linha CRLF do Windows, enquanto Linux
gravava LF. Novos JSONs agora têm UTF-8/LF explícitos. O hash do tokenizer foi fixado
no formato canônico; os três hashes binários, corpus, splits e protocolo de treino
permanecem iguais. Verificações continuam estritas e agora identificam o campo,
hash esperado e observado numa falha. Datasets antigos são lidos sem regravação.

A entrega corrigida usa **`run_number=19`** para preservar a tentativa 0018. Rode a
célula atualizada; o cache de shards baixados será reaproveitado e o dataset preparado
será publicado em uma pasta nova. Não é necessário apagar a tentativa anterior.
Nessa entrega, os ZIPs serão `llm2-campaign-0019-results.zip` e
`llm2-campaign-0019-checkpoints.zip`; os nomes 0018 abaixo exemplificam a primeira
entrega. Plano e riscos: [correção de portabilidade](PLAN_CYCLE3_PORTABILITY_FIX.md).

O teto é **14.400 segundos desde o início da célula**, incluindo download do bootstrap,
instalação, dados, testes, pilotos, treino, avaliação e exportação. Os 300 s finais
são reservados para exportação. Pode terminar antes. Testes CPU e CUDA FP32 precedem
o piloto. Perfis são isolados em cada T4: dez passos de warmup, trinta medidos.
Um piloto separado mede a duração da validação completa sem gravar NLL.
Warmup agrega compilação e aquecimento; não há contador de recompilações nem tempo
exclusivo de compilação verificado nesta instrumentação.

Execuções são seriais, com ordem rotacionada e GPU física alternada por seed.
Avaliações usam a primeira GPU. Só inicia um bloco de seis treinos se a previsão
conservadora do piloto couber. O watchdog preserva checkpoints de passos completos
se o tempo exceder a previsão. Uma sessão parcial é `partial_budget`, nunca
confirmação. Cinco seeds podem exigir várias sessões de até quatro horas.

Baixe **os dois arquivos** ao terminar:

- `llm2-campaign-0018-results.zip`: evidência, configs, hashes, logs, métricas por
  janela, relatório, curvas e snapshots de código; exclui pesos e dados binários.
- `llm2-campaign-0018-checkpoints.zip`: último checkpoint completo de cada treino
  iniciado, incluindo pesos/otimizador e metadados. Preserva os finais para o teste
  reservado. O estado de treino contém pickle; só carregar se confiável. Retomada
  exige dados, código e ambiente originais.

`report/report.json` lista `complete_seeds` e `pending_seeds`. Na próxima sessão use
as pendentes em `LLM2_SETTINGS["seeds"]` e outro `run_number`; não repita completas.
Não há retomada automática de treino interrompido: preserve seu checkpoint e
registre eventual retomada estrita separadamente, sem flexibilizar os contratos.

## Agregação e decisão

Combine somente relatórios do mesmo protocolo, conteúdo do dataset, código,
configs versionadas e versões de runtime/tipo de GPU. UUIDs físicos, máscaras de
slot e host CPU podem variar nessa agregação; o contrato de retomada de treino
continua integral e inalterado. Duplicação de braço/seed é rejeitada.

```powershell
python -m llm2.research.cycle3_report --inputs campanha18-results.zip campanha19-results.zip --output artifacts/cycle3-combinado
```

Contraste primário: FLRM − Transformer pequeno com FLOPs estimados igualados.
Intervalo t pareado de 95%, por seed; menos de cinco blocos completos resulta
inconclusivo. Ganho operacional exige média ≤ −0,005 NLL, limite superior abaixo
de zero e ao menos quatro seeds favoráveis. Equivalência exige intervalo inteiro
em ±0,002. Demais contrastes são exploratórios. O intervalo assume diferenças
independentes entre seeds; cinco observações ainda limitam a precisão. Tempo para
atingir um alvo de qualidade pré-fixado não é medido.

O launcher **não avalia o teste reservado**. Depois da análise de validação,
registre a configuração escolhida; só então use `evaluate --split test` nos finais
correspondentes, com os dados preparados originais. O teste não escolhe variantes
ou checkpoints. A100 e novas tecnologias continuam condicionais a essa conclusão.

## Verificação local

```powershell
python -m pip install -e ".[dev,research,wikitext]"
python -m pytest -q
python scripts/run_cycle3.py --cache artifacts/wikitext-cache
```

Sem `--execute`, o último comando só descreve a matriz, sem arquivos/rede/treino.
`--engineering-smoke --device cpu` usa dimensões/horizontes pequenos e nunca conta
como confirmação científica. Verificações reais: [VALIDATION.md](../VALIDATION.md).
