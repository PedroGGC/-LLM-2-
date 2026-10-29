# Ciclo 3 — confirmação independente e orçamento, 29/10/2026

## Escopo solicitado e congelamento antes da implementação

Implementar a próxima campanha, sem mecanismos novos da tabela. Célula curta por
commit publicado em `experiment/fractal-lrm`; teto de 14.400 s por sessão desde o
download do bootstrap, com 300 s de reserva. O teto não é duração mínima e não
garante completar todas as cinco seeds numa única sessão.

- Dados: WikiText-103 raw, revisão/shards fixados no importador, artigos completos,
  aproximadamente 32 MiB de treino selecionados por SHA-256, validação/teste oficiais,
  deduplicação explícita e byte-level. Congelar hashes de conteúdo após preparo real.
- Base: T256/B32, FP32/Inductor, LR 0,0004, warmup 50, dropout 0,05 e demais ajustes
  da 0017. Recorrentes K=2; FLRM com comunicação/cache, FiLM desligado; contexto sem
  RMS. JEPA/MTP/NCM/currículo/Ciclo-V desligados. Sem transferência de pesos antiga.
- Seeds 101–105. Ensaio de alvos iguais: Transformer pequeno, LRM e FLRM com 8.000
  passos cada. Quatro avaliações completas por treino; endpoint é o horizonte final.
- Ensaio separado de FLOPs estimados igualados: referência FLRM de 8.000 passos;
  Transformer pequeno, LRM e Transformer mais profundo (dois blocos/FFN1024), cada
  qual com horizonte calculado antes do treino e arredondado para baixo a múltiplo
  de quatro. Declarar todas as exclusões e o erro discreto do pareamento. Não chamar
  esse ensaio de tempo igualado. O FLRM âncora é compartilhado entre os dois ensaios,
  sem contar sua reutilização como replicação independente.
- Seis treinos únicos por seed, 30 no total. Rodar serialmente, com ordem rotacionada
  e GPU alternada. Só iniciar bloco completo de seed que o piloto estime caber.
  Exportar seeds pendentes e instruções para uma nova sessão, sem alongar scheduler.
- Contraste primário: FLRM menos Transformer pequeno no ensaio de FLOPs estimados
  igualados. Demais contrastes são secundários/exploratórios. Com menos de cinco
  seeds completas, não declarar confirmação. Teste reservado não é avaliado neste
  launcher: seleção posterior usa validação e pesos finais preservados.

## Etapas e aceitação

1. Corrigir publicação atômica do dataset: manifesto completo antes do rename;
   validar com falha injetada e preparo real dos quatro shards. Sem sobrescrita.
2. Congelar configurações, contagens por enumeração/fórmula e horizontes por custo.
   Testar causalidade/backward/finitude/memorização e retomada CPU math com dropout
   e acumulação, incluindo Transformer profundo.
3. Consolidar avaliações por execução num subprocesso, mantendo os mesmos resultados
   do CLI e sem consultar teste. Medir fases e warmup; não apresentar warmup agregado
   como medição isolada de compilação ou inventar contador de recompilações.
4. Executor com testes CPU/CUDA FP32 prévios, piloto em cada GPU, controle global,
   relatório por ensaio/seed, curvas e intervalos. Exportar análise e backup separado
   dos checkpoints finais, sem sobrescrever os outputs ou pacotes anteriores.
5. Simulação CPU pequena, suíte completa, validação documental e publicação do commit.
   Verificar a URL do bootstrap e gerar a célula curta com SHA realmente disponível.

## Riscos e limites

O ganho pequeno pode não generalizar. Comparação de FLOPs exclui operações não
estimadas; tempo real e parâmetros continuam métricas independentes. O Transformer
profundo terá mais pesos e isso será explícito. Dados oficiais podem conter
duplicação: falhar com diagnóstico em vez de mudar splits silenciosamente. A auditoria
real identificou uma cópia exata treino/teste de “The Hustler (film)”; a exclusão
explícita desse hash somente do treino foi registrada antes de qualquer ensaio.
Validação/teste permanecem oficiais e outras sobreposições são rejeitadas. Sessão
parcial não conta como confirmação. Checkpoints só são retomados com verificações
estritas e código/ambiente compatíveis; combinar sessões exige mesmo protocolo,
dataset, código e versões, sem duplicar seeds. CUDA novo será testado na T4 real;
testes locais CPU não demonstram suporte empírico adicional na GPU.
