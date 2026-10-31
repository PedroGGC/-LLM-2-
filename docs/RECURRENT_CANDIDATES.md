# Fichas de plataformas recorrentes — 31/10/2026

Objetivo: selecionar uma regra implementável, não declarar superioridade de uma
família sem treino. A implementação local é uma **comparação de regras num shell
comum**, não reprodução dos três modelos completos. Tokenização/objetivo/forma do
FFN são controlados; sua largura é ajustada para aproximar parâmetros (ver tabela
operacional). Isso limita a transferência das conclusões aos modelos originais.

| Candidato | Atualização e leitura | Estado FP32 por camada/seq | Paralelismo |
|---|---|---|---|
| RWKV-7 | S←S(diag(w)+abᵀ)+vkᵀ, y=S q; a=−k normalizado, b=k normalizado⊙taxa vetorial | H×V×K×4 bytes | Sistema triangular DPLR dentro de chunks; dependência entre chunks |
| mLSTM | Covariância com input gate exponencial e forget gate sigmoid, normalizador n e escala log m; leitura normalizada | (H×V×K+H×K+H)×4 | Somatórios causais estabilizados por chunk |
| Gated Delta | S̄=αS; S←S̄+β(v−S̄k)kᵀ; leitura S q | H×V×K×4 | DPLR com decaimento escalar e solve triangular por chunk |

Recorrente simples tem duas memórias; hierarquia também tem duas, mesmo quando
pesos são compartilhados. Pendências do pooling e última representação ampla
também ocupam memória. Sharing de pesos não compartilha estados entre escalas.

## Fontes oficiais e versões consultadas

- [RWKV-7 v2](https://arxiv.org/abs/2503.14456v2),
  [código oficial](https://github.com/BlinkDL/RWKV-LM/tree/2efa61d16846206f0eac904710a520b602392077), Apache-2.0 conforme anúncio do artigo; conferir licença dos arquivos incorporados.
- [xLSTM v2](https://arxiv.org/abs/2405.04517v2),
  [código oficial](https://github.com/NX-AI/xlstm/tree/ab22eadbd245f293dd8dec38ed29963d73758a12), Apache-2.0.
- [Gated Delta v1](https://arxiv.org/abs/2412.06464v1),
  [código oficial](https://github.com/NVlabs/GatedDeltaNet/tree/b53d6d3a161267432a79c1c04af69fa52bddc921),
  [Nvidia Source Code License-NC](https://github.com/NVlabs/GatedDeltaNet/blob/b53d6d3a161267432a79c1c04af69fa52bddc921/LICENSE),
  com limitação a pesquisa/avaliação não comercial.

HEADs fixados por `git ls-remote` nesta data. A seleção não baixa nem executa esses
repositórios: registra fontes e usa derivação própria das equações. Consultar os
arquivos LICENSE desses commits antes de incorporar código externo. Não foram
incorporados kernels externos. As licenças dos repositórios são registradas para
avaliar uma eventual incorporação futura, não como licença do código local derivado
das equações. PyTorch2.5–2.x, Python>=3.10; nenhuma biblioteca nova
de kernel é dependência obrigatória. Suporte/performance dos kernels oficiais na T4
não foi presumido. A seleção mede nossa execução FP32/Inductor no hardware real.

## Adaptações explícitas

Shell comum: embedding/cabeça ligados, duas células, RMSNorm FP32, QKV, projeção e
SwiGLU. Nenhum DecoderBlock, RoPE ou SDPA no backbone novo. As invariantes do
Transformer histórico não foram modificadas.

RWKV: apenas regra DPLR com decaimento softplus e taxa vetorial; não implementa
token shift, LoRA de gates, residual de valor entre camadas, GroupNorm e channel
mixing do modelo publicado. Não é um checkpoint RWKV compatível.

mLSTM: estabilização mantém escala log >=0, matematicamente equivalente por
reescalonamento da covariância e normalizador; usa shell comum em vez do bloco
xLSTM publicado (convolução/expansão/output gates originais não incorporados).

Gated Delta: normalização Q/K, α sigmoid, β sigmoid, sem convolução local nem
parametrização original do decaimento. Não é reprodução de GatedDeltaNet.

As três regras têm caminho sequencial independente para verificar outputs,
estados e gradientes do chunked. Compile é uma possibilidade medida, não uma
garantia de kernel único. Os solves/somatórios locais são O(chunk²); chunk é fixo
e não constitui atenção densa sobre o contexto inteiro. Estado de execução é
FP32 e não é nn.Parameter. Referência sequencial serve à correção, não à seleção
de velocidade. FLOPs analíticos ficam null até existir estimador específico.

## Decisão operacional

`python -m llm2.research.state_platforms --execute --output NOVO_DIRETORIO`
mede ambos slots0/1 isoladamente, testa gradientes/causalidade/streaming, registra
ambiente e hashes, mede warmup separado e falhas por candidato. Exige duas T4 para
selecionar; CPU `--engineering-smoke` nunca escolhe vencedor. Empate <=5% por
menor estado, depois nome para desempate determinístico. Um candidato precisa
passar nas duas placas. CPU VRAM/energia GPU são null.

O JSON identifica `implementation_scope=common_shell_memory_rules`, impedindo
interpretar a seleção como benchmark das arquiteturas completas dos artigos.
