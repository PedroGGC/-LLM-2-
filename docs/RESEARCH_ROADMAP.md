# Estado da V2 e próximos experimentos

Implementados: recursão causal, controle untied, estados contínuos por posição, progressão
de refinamentos, supervisão intermediária, target encoder congelado/EMA, diagnóstico de
colapso, MTP opcional, harness uniforme/fixo/adaptativo, verificadores e avaliações procedurais,
estimativas por componente, profiling, relatórios, limites e notebook Colab opt-in.
FLRM tem implementação própria isolada, compartilhando essa infraestrutura.

Sanidade CPU é condição de engenharia, não validação das hipóteses de aprendizagem. A ordem
de campanha deve seguir E0–E3, controle untied, F0–F4 e C0–C6, primeiro em 5M–10M, depois
20M–35M se justificado. Não lançar todas automaticamente. Examinar linguagem geral e OOD,
controlar FLOPs/parametrização e repetir sementes antes de ampliar complexidade/escala.

## Não implementado ou não demonstrado

- Reprodução integral LLM-JEPA multivista, HRM/TRM com seus protocolos ou Coconut com conversão
  progressiva de cadeias de pensamento em tokens contínuos. As hipóteses locais são distintas.
- Parada adaptativa, profundidade aprendida, DEQ e gradiente implícito; NCM completo,
  DeltaNet/Mamba, memória persistente entre documentos/escalas e kernels CUDA/Triton próprios.
- DDP, MoE, BitNet, Rust, RL grande, datasets gigantes e treino de 300M–500M.
- Garantia de anti-colapso, interpretabilidade de z, generalização, ganho computacional,
  redução de 100.000x ou competência linguística ampla.
- Campanha multissemente e comparação de qualidade com parâmetros/FLOPs cuidadosamente
  ajustados; os YAMLs são pontos de partida, não um experimento científico já concluído.
- Validação real A100/BF16/fused SDPA/Inductor e medições de VRAM nesse hardware.
- Medição de pico RSS CPU/NVML total: bytes de parâmetros são calculados e profiler mostra
  operações suportadas; isso não é uma medição universal de memória de processo/GPU.

## Riscos abertos

JEPA pode colapsar ou aprender atalhos locais; sondagem com teacher forcing pode não refletir
geração livre; adaptatividade pode explorar vieses do gerador; famílias pequenas não substituem
benchmarks externos. Retomar requer código/versões iguais, e Drive/FUSE pode falhar durante
I/O. Snapshots e hashes não tornam pickle seguro contra fontes maliciosas. Cada relatório deve
listar essas limitações relevantes, distinguir resultados medidos de estimativas e conter
os dados necessários à reprodução.
