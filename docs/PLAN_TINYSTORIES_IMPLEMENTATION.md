# Linguagem após0031 — plano de implementação,10/11/2026

Objetivo autorizado: medir se a memória que resolveu associação também aprende
linguagem, sem repetir a tarefa saturada. Transformer permanece referência externa;
pirâmide compartilhada permanece controle explícito da hipótese fractal.

## Ciência e dados

Quatro braços: memória única86.785pesos, pirâmide compartilhada86.979,
Transformer próximo em parâmetros86.400 e Transformer histórico2.164.992.
Seeds331/332,6000passos, contexto256/B32, FP32/Inductor, dropout0,05.
As comparações serão por alvos e também curvas de qualidade por tempo; nenhum
ganho é pressuposto. Mesmo número de pesos não garante capacidade equivalente.

TinyStories em revisão e prefixos de arquivo fixados porSHA. Baixar somente
prefixos limitados, preservar histórias completas, seleção determinística de
aproximadamente16MiB de treino. Dados oficiais de validação originam desenvolvimento
e teste reservado distintos. Uma divisão adicional do treino serve para calibrar
taxa; não usar desenvolvimento/teste para selecionar taxa.

Tokenização byte-level257, BOS256, loss apenas nos bytes reais. Sampler/evaluador
não atravessam documentos; padding mascarado e estado resetado por janela.
Avaliação percorre todos os bytes de cada documento selecionado, incluindo caudas.
Relatar cobertura e definições exatas; não confundir contexto com alvo ou excluir
as caudas silenciosamente. Deduplicação exata global entre divisões.

Calibrar duas taxas0,0003/0,001 por modelo, seed de calibração própria e400passos
fixos, com scheduler próprio. Congelar a escolha pela perda na divisão de
calibração antes dos dois treinos principais. Não continuar pesos de calibração.
Retê-los no mesmo processo/cache para reduzir compilação, com inicializações
novas e metadados separados. Dataset de teste não será avaliado nesta rodada.

Desenvolvimento completo em1500/3000/4500/6000. Prompts/temperatura/seeds de geração
pré-fixados, exportando todos os exemplos. Geração é qualitativa, sem seleção
manual nem nota automática de raciocínio. Uma baseline de contagens n-gram serve
para diagnosticar se o modelo vai além de estatística local, com orçamento de
estado próprio declarado.

## Otimização da memória

Adicionar execução bloqueada, preservando a execução histórica como padrão.
Superblocos32 calculam atenção causal local e prefixos matriciais porsuperbloco;
preservar partição lógica dequatro posições da pirâmide. Não materializar todos
os prefixos matriciais porgrupo4. Mesmos pesos/operação, com outra ordem de soma.
Exigir equivalência forward/backward/streaming e causalidade. Perfilar memória e
tempo naT4 antes de alegar economia. Versão/config explícitas; não retomar pesos
históricos como se a implementação não tivesse mudado.

## Operação e critérios

T4x2, teto2h desde a célula, reserva5min para exportação, pilotos com35%de margem.
Workers persistentes porjob; primeiro job por variante calibra a taxa, segunda
seed reutiliza seleção verificada. Perfis finais isolados porbarreira. Horizontes
não mudam pelo tempo restante. Matrizes que não couberem ficam parciais explícitas.
Um parZIPfinal, último checkpoint completo porjob; caches temporários próprios
limitados e removidos ao final. Não apagar campanhas/ZIPs/dados antigos.

Aceitação científica inicial: mostrar perda finita decrescente, comparar NLL/bits
porbyte e custo com as duas referências. Qualidade ≥0,005NLL melhor, ou perda até
0,002 com ≥10%de economia no recurso pré-definido, são sinais exploratórios.
Compactação é relatada separadamente. Dois seeds/corpus reduzido não confirmam
superioridade geral; manter confirmação independente e teste reservado no plano.

Validação: integridade/portabilidade de dados, deduplicação, isolamento de docs,
causalidade, equivalência das execuções e gradientes, finitude/memorização,
retomada com dropout/acumulação, seleção train-only e restauração dos pilotos,
falhas/deadline/disco/exportação. CUDA só será aprovado quando testado na GPU.
Entregar célula curta por revisãoGit/SHA, download embutido e documentação real.
