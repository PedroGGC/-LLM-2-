# Arquitetura própria inspirada em fractais — plano aprovado

Atualizado10/11/2026. Substitui a prioridade operacional do ciclo 5; não altera protocolos
históricos. Implementação experimental em `experiment/fractal-lrm`.

Atualização0031: [oito treinos concluídos](results/CAMPAIGN_0031.md), duas seeds.
Memória única/compartilhada/independente resolveram recuperação de um byte:
100%original/contrafactual até8K,64pares e deslocamento. Compartilhamento economiza
22,10%dos pesos da pirâmide independente; a hierarquia não melhora sobre a memória
única. O Transformer desta receita não qualificou, impedindo aceitar eficiência
a qualidade igual. Razões observadas de24,95×pesos/9,54×forward8K/7,23×energia
de placa não representam capacidade geral deLLM nem vitória em todos os recursos.

Próximo marco recomendado: linguagem com memória única, pirâmide compartilhada e
Transformer, preservando dados/validação/custo comparáveis; qualificar a referência
de recuperação e usar chaves/associações mais discriminativas. Não repetir uma
campanha inteira já saturada nem acrescentar mecanismos para melhorar100%.
Fractal continua explícito; distinguir viabilidade do sharing de necessidade da
hierarquia. Essas são recomendações pós-auditoria, sem nova campanha implementada.

Autorização posterior em09/11: implementar mudanças amplas para buscar vantagem
funcional. [Nova operação associativa](ASSOCIATIVE_BINDING_RUN.md), família de
pesos rápidos/transições independente de mLSTM. Quatro braços: Transformer,
memória única e pirâmide compartilhada/independente. A inspiração fractal permanece
explícita no mesmo operador entre ritmos de memória; não obriga reutilizar a
implementação FLRM ou a plataforma mLSTM selecionada historicamente.

Qualificação CPU em mapas novos passou:100%nos testes até8K/64pares/deslocamento,
núcleo86.785pesos/seed816/1000passos. Não é resultado de linguagem, eficiência
ou comparação com Transformer. A evidência seguinte foi a mesma tarefa/receita para os
quatro braços naT4, duas seeds,3000passos, teto2h, concluída na0031. O protocolo mede acerto e
contrafactuais; não reescreve os limiares NLL dos experimentos de linguagem.
Metas95%/90%pareado e fatores de recurso são critérios, não garantias. Se passar,
o marco seguinte será confirmação em linguagem, não declarar a ambição cumprida.

Atualização0030: [quatro de14treinos concluídos](results/CAMPAIGN_0030.md), uma
seed271. Bancos completo/Fibonacci e recorrente simples ficaram perto do acaso
na recuperação; Transformer6,88% emT1024 também não resolveu associações. Fractal
permanece hipótese, sem novo contraste de compartilhamento concluído. Recomenda-se
qualificar formação local de pares e escrita/leitura em exemplos novos, revisando
inicialização de esquecimento e mistura local da adaptação mLSTM, antes de outra
grade longa ou de concluir os dez treinos. O controle de banco completo falhou;
Fibonacci/binário/uniforme não devem ser promovidos como solução dessa falha.
Agrupar compilação por grafo/GPU; pilotos isolados consumiram34min56s da sessão.
Recomendações da auditoria0030, atendidas pela implementação posterior acima.

## Objetivo e evidência

Buscar qualidade por custo real de treino e uso. Transformer é referência externa;
fractal permanece hipótese de organização: hierarquia, reutilização da mesma regra
entre níveis e ritmos diferentes. Isso não pressupõe fractal matemático nem cópia
biológica. RWKV-7, mLSTM, Gated DeltaNet, MC/GRM e TTT são candidatos, não dependências.

0017 já separou comunicação/FiLM. 0023 não confirmou vantagem das escalas, e o
Transformer de dois blocos venceu qualidade/custo. K não treinado piorou. Preservar
FLRM histórico; suspender novas repetições e a prioridade de executar ciclo 5.
Base sólida significa correção, reprodução e custo conhecido, não vitória prévia.

Atualização0024: as três regras executaram nas duas T4, mLSTM com melhor tempo
observado. A auditoria detectou alarme de600s absorvido pelo compilador nos seis
casos; repetir qualificação com supervisor externo antes da etapaB. Não alterar
as hipóteses científicas nem promover o ranking operacional a ganho de linguagem.
Ver [evidência](results/CAMPAIGN_0024.md) e [correção](PLAN_CAMPAIGN_0024_STORAGE.md).

Atualização0025: qualificação corrigida concluída nas duas T4. Selecionar mLSTM;
RWKV também aprovado e Gated Delta fora do orçamento de compilação. Preservar a
fonte qualificada, não repetir a etapa A. Próxima pergunta: contribuição de
hierarquia e compartilhamento na etapa B. Retenção de caches entre workers é
pendência operacional antes de lançar B; [auditoria](results/CAMPAIGN_0025.md).

Entrega B: a [nova operação](STATE_B_RUN.md) resolve a retenção por GPU/variante
sem alterar o worker qualificado. Agrupa piloto/triagem e perfis isolados para
reduzir recompilações; primeiro lote com os quatro braços da seed241. A qualificação
do núcleo permanece a0025.

Atualização0027 (01/11/2026): primeiro bloco B concluído, quatro braços/seed241.
Transformer1,273494NLL; simples1,313647; independente1,431568; compartilhada1,462067.
Não há candidato favorável de qualidade/custo. Sharing reduziu47,72% dos pesos,
com perda0,030498NLL e só3–4% menos tempo por passo. É triagem de uma seed, sem
confirmação estatística. [Auditoria e limites](results/CAMPAIGN_0027.md).

Recomendação de continuidade: diagnosticar compressão/mistura e a substituição da
segunda transformação fina antes das seeds242/243. O desenho atual troca resolução
e trabalho; não isola apenas hierarquia. Preservar a matriz original com1/3seeds,
sem reescrever seus critérios. Primeiro intervenções diagnósticas nos pesos finais;
depois controle somente fino e uma alteração isolada, com protocolo próprio definido
antes de treinar. Fractal permanece hipótese explícita; nenhuma nova tecnologia
entra automaticamente. Geração/retenção não foram medidas; não extrapolar treino.

Entrega de diagnóstico: [operação congelada](STATE_DIAGNOSTIC_RUN.md),11leituras nos
quatro checkpoints já treinados, antes de uma nova campanha. Não altera a matrizB
nem implementa gate/compressão aprendida. Ambiente/dados/fontes/pesos originais
verificados; objetivo é decidir qual controle treinado isolará a limitação.

Atualização0028: diagnóstico concluído nas duas T4, originais reproduzidos em2,61e-9,
sem treino. Silenciar/reduzir o amplo piorou. O componente participa da previsão,
mas a arquitetura continua inferior aos controles. Padrão descritivo na quarta
posição é compatível com defasagem: resumo do grupo fechado entra no byte seguinte.
Próximo ensaio recomendado: controle fino treinado do zero e mudança isolada para
ler o resumo assim que o grupo fecha, mantendo a média e os controles. Nova matriz
deve ser pré-fixada antes de treinar; não alterar os critérios/seeds pendentes daB.
Não diagnostica sozinho a compressão nem justifica combinar novas tecnologias.
Compartilhamento permanece hipótese explícita. [Auditoria](results/CAMPAIGN_0028.md).

Entrega0029 autorizada: [cinco controles treinados](STATE_READOUT_RUN.md), seed251,
6000passos e teto2h. Separar profundidade fina e índice de leitura mantendo os
controles. Receita inicial reaproveitada, sem nova varredura de taxas. Pares inteiros
divididos entre sessões se necessário; nenhuma alegação de eficiência nas duas
GPUs com perfil de cada braço em uma só placa. À entrega, T4 pendente. Fractal/sharing
permanece para comparação com a organização escolhida; compressão aprendida é a
próxima pergunta caso o controle de leitura não traga sinal suficiente.

Resultado0029: [cinco braços concluídos](results/CAMPAIGN_0029.md), uma seed251.
Imediata1,427357contra seguinte1,436200NLL; melhora0,008843, mas continua acima do
simples1,315631eTransformer1,271962. Fino treinado1,474430: ampliar ajuda frente a
uma célula, sem substituir a segunda célula por byte. A pior posição desloca-se
da quarta para a terceira, sem eliminar a dependência do grupo. Compartilhamento
não foi retestado. Não repetir automaticamente a matriz antiga nem confirmar
longamente um candidato ainda inferior. Próxima proposta: separar compressão
média, último estado contextualizado e compressão aprendida pequena, mantendo
leitura imediata/núcleo/fator/ritmo e contabilizando parâmetros/operações reais.
Não se provou que a média seja a única limitação; frequência/comunicação seguem
hipóteses concorrentes. Novo protocolo depende de definição própria antes do
treino. Depois reavaliar sharing na organização candidata. MC/GRM,TTT eACT
mantêm condições próprias; não serão adicionados em conjunto para tentar resgatar
esta implementação. Hipótese fractal e plano de comparação externa permanecem.

## Etapas

Atualização autorizada08/11: [memória consultável e Fibonacci](MEMORY_RECALL_RUN.md)
passam à próxima operação exploratória da etapaC. Não exigir uma vitória da antiga
hierarquia para explorar outra hipótese. Medir recuperação antes de concluir que
retenção limita o modelo. Banco vetorial completo e três acessos esparsos, mantendo
o núcleo compartilhado; referênciaTransformer e recorrente simples. Fibonacci é
hipótese de espaçamento reescalado, não obrigação física;phi/lattices permanecem
condicionais. Metas funcionais95%/margem1p.p./2× não substituem critérios de NLL.
Próximas sessões limitadas a2h por piloto; a meta2h/teto3h do plano original abaixo
fica como histórico, sem ampliar esta operação. Comparação futura de linguagem e
modelos maiores permanece condicionada a um sinal real; nenhuma garantia de1.000.000×.

A. Fichas dos três operadores recorrentes, referências sequenciais, execução por
chunks e microbenchmarks de forward/backward na T4. Somente casos com correção,
finitude e memória admissível podem ser selecionados. Medir isoladamente em ambas
GPUs; escolher menor tempo, empate até 5% por menor estado. Sem CUDA real não há
seleção científica. Referências implementadas no laboratório não são reproduções
dos modelos completos RWKV/xLSTM/Gated DeltaNet; documentar cada simplificação.

B. Quatro braços: recorrente simples, hierárquico independente, hierárquico com
núcleo compartilhado, Transformer de dois blocos. Mesmas dimensões, projeções,
ritmos e operações entre hierarquias; apenas compartilhamento difere. Caminho fino
a cada byte, amplo a cada quatro estados contextualizados; grupos incompletos não
influenciam posições anteriores. Compartilhar somente espaços compatíveis.
Sem GRM/TTT/FiLM/ACT. Simples, independente e Transformer próximos de 2M (±5%);
compartilhado pode ser menor, sem aumentar largura para esconder a economia.

WikiText preparado do ciclo 3, byte-level, T1024/B8, FP32/Inductor, dropout0,05,
6000 passos, seeds241/242/243, 12 jobs. Triagem simétrica LR0,0002/0,0004 em
validação interna por documento (seed901, 750 passos), sem consumir o teste.
Validação completa em50/100%, subconjunto fixo em25/75%. Os pilotos não continuam
nos treinos principais. Horizontes e schedulers congelados; alvos iguais não são
custo igualado. Contrastes: hierarquia−simples, compartilhado−independente e
comparações externas com Transformer.

C. Escolher a próxima pergunta pelo diagnóstico: compressão aprendida/adaptativa;
onde compartilhar; retenção→memória consultável/MC; atualização associativa→TTT;
benefício de K→profundidade variável e ACT/PonderNet; gargalo de execução→otimização.
Treinar K1/2/4 antes de testar adaptação. Escrever uma entrada uma vez; refinamento
não repete gravação. Sem ativação automática de mecanismos por tamanho/K/contexto.

D. No máximo dois mecanismos inicialmente, controle/A/B/A+B. Confirmar cinco seeds
novas com custo igualado e segundo corpus (TinyStories); depois7–10M com dados
compatíveis. 10M contra100M é marco futuro em tarefas definidas, não equivalência.

## Orçamento, risco e aceitação

Meta2h/teto3h incluindo setup, compilação, treino e avaliação; dois jobs independentes
nas duas T4. Pilotos com margem35%; dividir matriz entre sessões se necessário sem
mudar horizonte. Não selecionar kernel incompatível por tempos fictícios, não
atribuir lentidão de referência Python à arquitetura e não ampliar sessão para
preencher tempo. Riscos: estabilidade numérica, custo do scan, grupos com vazamento,
alias de estados, sharing contado duas vezes e comparações confundidas por parâmetros.

Medir NLL/bits por byte, alvos/FLOPs estimados, tempos separados, RAM/VRAM, bytes
de pesos/estados/caches e energia GPU quando disponível (caso contrário null).
Recuperação por distância/distratores/esquecimento e tarefas verificáveis para K;
mesma cobertura em bytes. Diagnósticos não provam raciocínio geral.

Triagem: redução média≥0,005NLL favorável nas3seeds; eficiência: perda≤0,002 e
economia≥10% no recurso pré-fixado. Reportar intervalos. Cinco seeds novas para
confirmação, teste reservado só após seleção. Menos parâmetros isoladamente é
compactação. Resultados dominados encerrados; incerteza é inconclusiva.

Testar causalidade inclusive fronteiras, sequencial/chunks com gradientes e estado,
isolamento por documento, forward/backward, finitude, memorização, compartilhamento
e fórmula de parâmetros, safetensors ligado e próxima atualização retomada igual à
ininterrupta com dropout/acumulação. Forward das hierarquias deve coincidir quando
pesos forem copiados e dropout desligado. Nunca seguir código/config/dados/versões
incompatíveis silenciosamente. Não há suporte CUDA empiricamente validado localmente.

## Catálogo e entregas

Prioritários: recorrência, hierarquia/fractal, compressão, profundidade variável;
MC/GRM e TTT são possibilidades independentes. RINS/BLT/Cantor são referências,
não backbones obrigatórios. KV cache/FractalLLM: engenharia de geração posterior.
ACT/PonderNet requer utilidade de K; DEQ requer equilíbrio/solver/gradientes.
Espectral/ortogonalidade/nGPT requer dinâmica/instabilidade; Kronecker/Monarch,
MERA e hypernetworks requerem gargalos medidos. FiLM/JEPA/NCM/RMS de contexto/
Ciclo-V/MTP/currículo preservados e desligados. Fibonacci/φ/lattices, KAN,
Liquid/CfC/LFM2 e L-systems: baixa prioridade, hipóteses próprias. FractalNet/FANN/
fractal nodes: referências. Difusão/flow/árvore/NCA/inferência ativa: linhas próprias.
Compilação/AMP/caches/quantização/destilação: ganhos separados dos arquiteturais.

Cada etapa entrega configurações, proveniência, resultados por seed, curvas,
recursos e decisão continuar/simplificar/encerrar/inconclusivo. Atualizar plano,
README, estado e VALIDATION com verificações reais; nunca marcar CUDA pulado aprovado.
