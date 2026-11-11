# Campanha 0035 — geração incremental e contexto longo

A campanha reconstruiu os quatro checkpoints e concluiu os 12 perfis de
geração nas duas T4. Foram 24.000 passos no total, com 8 min 03 s registrados
antes da exportação. A revisão de treino foi a mesma da 0033 e cada novo
checkpoint recebeu seu próprio hash.

O recorrente manteve uma vantagem no tempo até o primeiro byte com prefixo
longo e usou muito menos estado. A geração de 256 bytes levou mais tempo
que no Transformer com KV cache. Isso indica qual parte da execução medir
e otimizar a seguir.

## Inferência nas duas T4

Médias descritivas das seeds 341 e 342. FP32 eager, batch 1, 256 bytes gerados
em greedy com prefixos sintéticos pareados. A energia é da placa na janela
de medição. O pico de VRAM inclui o restante das alocações da execução.

| Prefixo | Modelo | Primeiro byte, ms | Decodificação, bytes/s | Total, ms | Estado/cache | J/byte |
|---:|---|---:|---:|---:|---:|---:|
| 256 | Recorrente | 7,04 | 228,83 | 1.121,95 | 12.480 B | 0,1317 |
| 256 | Transformer | 2,77 | 396,71 | 645,59 | 654.080 B | 0,0792 |
| 1.024 | Recorrente | 7,39 | 230,13 | 1.115,97 | 12.480 B | 0,1313 |
| 1.024 | Transformer | 2,91 | 387,34 | 661,67 | 1.637.120 B | 0,0801 |
| 7.936 | Recorrente | 8,06 | 235,46 | 1.091,13 | 12.480 B | 0,1280 |
| 7.936 | Transformer | 14,79 | 402,24 | 648,92 | 10.484.480 B | 0,1278 |

No prefixo de 7.936, o recorrente chegou ao primeiro byte 1,85× e 1,83×
mais rápido. O tempo total foi 1,66× e 1,71× maior. O estado ocupou 840×
menos bytes que o cache de K/V pré-alocado para a geração completa.
O pico de VRAM foi 76,18 MiB frente a 80,68 MiB, uma redução de 5,6%.
A energia total por byte foi próxima nesse prefixo. Em prefixos menores,
o Transformer consumiu menos energia por byte e também chegou antes
ao primeiro byte. A meta de 2× no tempo total permanece pendente.

Os 12 perfis passaram na paridade com o forward completo do próprio modelo.
Houve concordância top-1 em todas as 257 posições de cada verificação.
A diferença máxima absoluta entre logits foi 8,20e-5.

## Qualidade em desenvolvimento

A reavaliação reproduziu exatamente a NLL T256 dos novos checkpoints.
Os dois recorrentes também ficaram à frente da referência de tamanho
próximo nessa reconstrução. Em relação à 0033, suas NLLs variaram de
0,993253/0,981204 para 1,000544/0,983356. Os Transformers ficaram dentro
de 7e-7 dos valores anteriores. As seeds foram as mesmas.

| Modelo | Janelas reiniciadas em 256 | Histórias completas |
|---|---:|---:|
| Recorrente de duas passagens | **0,991950** | **0,978279** |
| Transformer de tamanho próximo | 1,021963 | 2,024229 |

A coluna de histórias completas usa os mesmos 524.508 bytes de 630
documentos, até 3.219 bytes por história. O Transformer com atenção global
foi treinado em janelas de 256. Essa diferença motiva um controle com atenção
local e a medição de desempenho fora do comprimento de treino. A medição
de geração em 8K usa entradas sintéticas.

## Controle local com os mesmos pesos

Reavaliamos os quatro safetensors em CPU, FP32 e SDPA math. Cada história
foi processada no seu comprimento real. Para o controle local, a máscara
permite as 256 posições mais recentes, incluindo a atual, em cada camada
do Transformer. Pesos, posições de RoPE e alvos foram iguais. Os logits
das primeiras 256 posições passaram na checagem de equivalência em todos
os documentos. O forward completo em CPU reproduziu as NLLs da T4 com
diferença absoluta inferior a 3e-7.

| Caminho | NLL seed 341 | NLL seed 342 | Média |
|---|---:|---:|---:|
| Transformer, atenção global | 2,227444 | 1,821015 | 2,024229 |
| Transformer, atenção local por camada | 1,005812 | 1,006581 | 1,006197 |
| Recorrente, história completa | 0,987542 | 0,969015 | **0,978279** |

A janela local removeu quase toda a diferença observada com a atenção
global. O recorrente ficou 0,018270 e 0,037566 NLL abaixo desse controle
nas duas seeds. O contraste global versus local usa os mesmos pesos
e demonstra o efeito da máscara neste regime. A NLL global cresce
com a posição, chegando a 4,11 na média após o byte 2.048. Esse grupo
tem 4.638 alvos. As contagens e as perdas por faixa estão no registro
estruturado. O controle em CPU mede qualidade de desenvolvimento.
O desempenho do KV cache local permanece para a próxima medição.

## Verificação dos arquivos

O ZIP de resultados tem 4.166.930 bytes e 59 membros. SHA-256
`d78a7024d260daf76e140c93b20da8278e6991a56ca018a75ac667ea5aad4f87`.
O ZIP de checkpoints tem 7.690.870 bytes e 22 membros. SHA-256
`1b8c8143176e46b5bac61f8cc303f33080a9573c23375c0b65d0379a34a7f84c`.
Ambos passaram em CRC e caminhos seguros. Seus manifestos e referências
coincidem. Foram conferidos os hashes dos quatro safetensors e dos quatro
estados de treino, 80 registros de atualização, medianas das amostras,
configurações e fontes fixadas em Git. Os estados PyTorch foram tratados
como bytes para o cálculo de hash. Cache de compilação de 187,83 MiB
removido ao fim. Os checkpoints permanecem no arquivo local.
[Auditoria estruturada](campaign_0035_audit.json).

## Próximo teste

Reutilizar os pesos da 0035 e perfilar a decodificação de um byte. O passo
recorrente levou cerca de 4,25 ms/byte em 8K. Igualar o tempo total atual
do Transformer pede aproximadamente 2,51 ms/byte. Atingir 2× pede cerca
de 1,24 ms/byte, supondo o mesmo prefill e o tempo medido da referência.
Esses são alvos, com nova comparação pareada após a otimização.

A primeira hipótese é reduzir os lançamentos por byte mantendo a
recorrência e os logits. Medir o perfil antes de tentar Inductor ou captura
CUDA com estado de tamanho fixo. Repetir as checagens de logits e medir
também o Transformer com cache limitado às últimas 256 posições por camada.
Conferir esse cache contra o forward com a mesma máscara, incluindo o ponto
de descarte da primeira chave. A comparação local separa a economia de
estado frente a uma referência de contexto limitado do contraste com
cache global.
[Orientação do PyTorch sobre profiling](https://docs.pytorch.org/docs/stable/user_guide/torch_compiler/torch.compiler_profiling_torch_compile.html).
