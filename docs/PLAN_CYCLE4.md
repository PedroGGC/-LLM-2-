# Ciclo 4 — diagnóstico e entrega Kaggle, 30/10/2026

Plano registrado antes da implementação. Entrega: dois scripts; backup/limpeza
em duas etapas e diagnóstico com meta 7.200 s, limite absoluto 10.800 s.

## Implementação e aceitação

1. Backup stdlib do working, retendo último checkpoint íntegro por treino e todos
   os demais arquivos científicos. Metadados/hashes conferidos sem carregar pickle.
   Pesos antigos não classificados preservados. ZIP exclusivo, CRC/SHA e manifesto.
2. Limpeza somente após SHA da cópia local, revalidação do inventário e ausência
   de workers. Remover filhos de /kaggle/working sem seguir links; preservar raiz
   e /kaggle/input. Recibo em memória com próximo número >=23.
3. compressed_factors=None mantém caminho antigo; [], [4], [16], [4,16] no modo
   compressed_native removem ramos físicos. FiLM/NCM/Ciclos-V desligados no diagnóstico.
   Adaptar fórmula, FLOPs, pareamento e telemetria sem mudar o núcleo recorrente.
4. Seis variantes: quatro FLRM (fino, +4, +16, completo), LRM K2 cached e Transformer
   2 blocos/FFN384. D384; FLRM/LRM FFN1024. Seeds211/212/213, 6.000 passos, checkpoints
   e validação completa em 1.500/3.000/4.500/6.000. WikiText byte-level já fixado;
   T256/B32, FP32/Inductor e hiperparâmetros C3, extras desligados.
5. Perfis isolados em ambas T4, piloto concorrente, estimativa conservadora 35%.
   Dois workers independentes, distribuição rotacionada. Reservar 900 s para
   pós-treino e 300 s para exportação. Preservar 2 GiB livres. Resultados exclusivos
   após cada job, backup incremental somente do job e bundle final dos últimos.
6. Reportar qualidade, FLOPs estimados, tempos concorrentes e isolados separados,
   RAM/VRAM, bytes de pesos/caches, energia se disponível (null caso contrário).
   Avaliar K1/2/4/8 no FLRM completo e LRM; diagnósticos fora do benchmark.
7. Testar causalidade, exclusão de ramos, fórmula/enumeração/alias, backward finito,
   memorização e retomada estrita CPU/math com dropout+acumulação; CUDA no preflight
   real. Testar backup, hash externo, inventário, symlinks, espaço, falhas e deadline.

## Riscos e interpretação

Sem testes CUDA locais neste ambiente CPU. Estimativa de duração não é promessa;
não encurtar/estender schedulers durante treinamento. Energia GPU não mede o host.
Comparação principal completo-fino a alvos iguais; componentes candidatos >=0,005
NLL e três seeds favoráveis; simplificação candidata <=0,002 por seed e >=10% mais
rápida em ambas GPUs. Não afirmar equivalência/superioridade geral por este diagnóstico.
Fontes e contratos C3 permanecem históricos; novos resultados têm protocolo próprio.
Backup requer cópia externa: Git não armazena pesos/ZIPs ignorados. Limpeza destrutiva
somente depois da verificação desse backup; alterações no working bloqueiam a exclusão.
