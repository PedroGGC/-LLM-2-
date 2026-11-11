# Geração com estado e KV cache — campanha a partir da 0034

## Reconstrução dos pesos após a 0034

O ZIP de resultados da 0033 permaneceu disponível. Os pesos finais se
perderam, e a tentativa 0034 encerrou na descoberta dos checkpoints.
A operação 0035 reproduz apenas as duas variantes e duas seeds necessárias.
Usa a configuração de cada job auditado, o treino original da revisão
`db507b5`, os mesmos dados e a taxa 0,001 selecionada na 0033. As quatro
execuções começam do zero, mantêm o horizonte de 6.000 passos e registram
a NLL final e o SHA-256 dos novos pesos. A reprodução é uma campanha
própria. Diferenças de runtime podem produzir pesos distintos.

Na mesma sessão, os checkpoints novos alimentam a comparação incremental
em processos separados. Verificar os arquivos e a fonte antes de medir.
A paridade de logits, a avaliação de desenvolvimento e os comprimentos
do protocolo abaixo se aplicam. O teste compara a NLL reavaliada com a
NLL do novo checkpoint. A diferença em relação à 0033 será registrada.

Usar as duas T4, alternando cada variante de placa entre seeds, com teto de
uma hora.
Aplicar o limite de cache e a reserva de exportação. Exportar um ZIP de
resultados e outro com o último checkpoint íntegro de cada job, inclusive
se houver interrupção. Conservar a referência de pesos novos junto dos
checkpoints completos. O bootstrap usa o nome `bootstrap.json` aceito
pela guarda de limpeza dos checkouts.

## Plano prévio

A 0033 encontrou menor NLL no recorrente de duas passagens independentes e
forward de 8K mais rápido em bytes aleatórios. A próxima pergunta é quanto
dessa vantagem chega à geração incremental. Reutilizar os quatro checkpoints
finais do recorrente de 151.056 parâmetros e do Transformer de 148.960
parâmetros, seeds 341 e 342. A execução usa o código original da 0033 para
construir os modelos e adaptadores de inferência separados. Cada peso precisa
corresponder ao SHA-256 da auditoria. O estado de treino fica fora da leitura.

Usar FP32 eager e batch 1. Medir prefill e emissão do primeiro byte, mais
255 bytes por greedy, com prefixos sintéticos idênticos de 256, 1.024 e
7.936 posições. A maior condição cabe no limite total de 8.192. Cada
Transformer pré-aloca K/V até o final do ensaio e aplica RoPE na posição
correta. O recorrente usa o forward em blocos no prefill e sua regra
sequencial de um passo na decodificação. Ambos projetam somente os logits
da última posição no prefill. O aquecimento precede as medições.

Na decodificação do Transformer há uma consulta na última posição. O SDPA
recebe somente as chaves válidas do prefixo e do byte atual. Assim cada
chave disponível está no passado ou na própria posição da consulta.
[Documentação do SDPA](https://docs.pytorch.org/docs/2.15/generated/torch.nn.functional.scaled_dot_product_attention.html).

Antes de medir cada condição, confrontar os logits de todos os passos
incrementais com o forward completo sobre uma continuação fixa. Fixar
tolerâncias absolutas e relativas em 5e-4 e registrar a diferença máxima.
A saída de cada modelo é avaliada contra a própria referência. Um erro
acima da tolerância bloqueia a comparação.

Registrar tempo até o primeiro byte, bytes/s seguintes, tempo total, energia
da placa, estado/cache, VRAM e ambiente. Usar as duas T4 com um job por placa
e inverter a distribuição na segunda seed. Possível concorrência de CPU é
parte do escopo de medição. A energia fica nula se o contador NVML estiver
indisponível. A meta de 2× exige tempo total pelo menos duas vezes menor
nas duas seeds na condição de 7.936 posições, com a paridade aprovada.

Medir NLL dos mesmos bytes de desenvolvimento em janelas de 256 e em
histórias completas. A maior história tem 3.219 bytes. Qualidade em 8K
exigirá dados com documentos maiores. O teste reservado fica fechado.

Teto de 30 minutos desde a célula, com reserva de exportação. Um ZIP de
resultados com botão no notebook. Priorizar o diretório da 0033 em working,
com opção de ler o ZIP de checkpoints em working ou em um input. Se os pesos
faltarem ou divergirem, interromper antes de executar os modelos.

Cole a [célula da 0034](../scripts/12_testar_geracao_com_cache_kaggle.py)
com Internet e GPU T4 x2 no Kaggle. Com `checkpoints=None`, a célula procura
a pasta da 0033, o ZIP de checkpoints no working e o ZIP anexado em
`/kaggle/input`. O campo aceita um caminho explícito para a pasta ou ZIP.
Baixe o ZIP de resultados gerado pelo diagnóstico.
