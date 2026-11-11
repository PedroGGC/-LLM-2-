# Campanha 0036 — reduzir o custo de cada byte gerado

## Protocolo

Usar os quatro safetensors verificados da 0035, seeds 341 e 342, com o
código de modelo fixado em `db507b5`. A campanha é de inferência, com
zero passos de treino. Os pesos, metadados e corpus são conferidos por
hash. O ZIP de entrada é `llm2-campaign-0035-checkpoints.zip`. A pasta da
0035 em working também serve quando contém os quatro checkpoints.

Medir cinco caminhos nas duas T4, em FP32 e batch 1.

| Modelo | Execuções |
|---|---|
| Recorrente de duas passagens | Passo eager e passo Inductor |
| Transformer global | KV cache eager da 0035 |
| Transformer local por camada | KV circular de 256 posições, eager e Inductor |

As variantes compiladas usam grafo inteiro com formas fixas no passo de
um byte. O cache local mantém as últimas 256 chaves por camada e aplica
RoPE à posição absoluta na gravação. O prefill local divide a atenção em
blocos e inclui as chaves anteriores necessárias. A janela inclui a
posição atual. [Documentação de torch.compile](https://docs.pytorch.org/docs/2.15/generated/torch.compile.html).

O perfil registra uma janela separada de 16 passos incrementais, com
operações CPU e eventos CUDA quando disponíveis. As medições de latência
usam sete repetições após aquecimento, a mesma geração greedy de 256
bytes e prefixos sintéticos de 256, 1.024 e 7.936 posições da 0035.
Registrar tempo da primeira chamada compilada, tempo até o primeiro
byte, taxa de decodificação, tempo total, energia, VRAM e estado.
[PyTorch Profiler](https://docs.pytorch.org/docs/2.15/profiler.html).

Comparar os logits de 257 posições contra o forward completo de cada
caminho, com tolerâncias absoluta e relativa de 5e-4. Conferir o cache
local antes e depois de preencher a janela. Reproduzir a NLL T256 do
checkpoint e, no controle local, a avaliação em histórias completas da
auditoria CPU. Erro de compilação ou paridade exclui a variante
compilada e aparece no relatório.

O primeiro alvo é cerca de 2,5 ms por byte, em relação ao tempo total
da referência da 0035. A decisão da 0036 usa o Transformer mais rápido
qualificado em cada seed e comprimento. Atingir 2× em 7.936 posições
exige as duas seeds, os cinco caminhos medidos e paridade aprovada.
Qualidade linguística em 8K permanece uma pergunta própria.

Teto de 30 minutos, com cinco reservados para publicação. Um job por
placa e alternância entre seeds. Guardar os pesos da 0035 e exportar
apenas um ZIP de resultados da 0036, com cache temporário limitado.
