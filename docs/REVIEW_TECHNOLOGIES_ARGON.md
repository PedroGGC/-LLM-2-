# Triagem das ideias do Argon — 29/10/2026

Parecer sobre `C:/Users/Lux/Downloads/Ideias.md`, confrontado com o código experimental,
as campanhas 0015/0016 e fontes primárias. O anexo é material para análise; suas
perguntas, prioridades e instruções de implementação não são autorização para
executá-las. As prioridades abaixo são recomendações condicionais.

**Decisão recomendada: executar o ciclo 2 já preparado e preservar a hipótese de
recorrência multiescala. Acrescentar TTT e um baseline recorrente moderno ao catálogo
posterior, sem incorporá-los à configuração atual.** As analogias físicas não
fundamentam garantias de capacidade, correção, compressão ou velocidade.

Escopo desta revisão: documentação e seleção de hipóteses. Riscos: confundir analogia
com evidência, alterar várias variáveis juntas e interpretar resultados em outros
regimes como validação local. Critérios de aceitação: cobrir os 17 itens, distinguir
mecanismos, propor controles observáveis e preservar código/configuração/pacote do
ciclo 2. Não iniciar treino, instalar outra stack ou promover tecnologia nesta revisão.

## O ponto de partida real

O projeto já contém Transformer. LRM e FLRM reutilizam blocos com SDPA causal, RoPE,
RMSNorm e SwiGLU; o FLRM acrescenta representações comprimidas e interação entre
escalas. A geração permanece autoregressiva. Não existe demonstração de raciocínio
geral ou de equivalência à capacidade de um modelo maior.

Na [campanha 0016](results/CAMPAIGN_0016.md), com três seeds e horizonte fixo:

| Modelo | NLL média completa |
|---|---:|
| Transformer | 1,394580 |
| FLRM N1 K=2 | 1,400059 |
| LRM K=2 | 1,403886 |

O Transformer venceu as três seeds. O FLRM melhorou frente ao LRM, com mais
parâmetros/operações e cache diferente. Ainda falta isolar a contribuição multiescala
e comparar com custo igualado. Normalizar contexto equilibrou amplitudes, mas piorou
NLL em 0,040738/0,042958 no LRM/FLRM. Uma justificativa matemática plausível pode
produzir uma intervenção desfavorável neste regime.

## Quatro coisas diferentes que o blueprint reúne

| Eixo | O que muda | Exemplos | Relação com o projeto |
|---|---|---|---|
| Sequência, t | Estado ao receber o próximo token | RWKV, mLSTM, CfC, TTT layers | Alternativa ao processamento temporal atual |
| Profundidade, k | Refinamentos do estado para o mesmo prefixo | LRM, ACT, DEQ | O loop atual é neste eixo |
| Resolução, s | Representações finas e comprimidas do prefixo | FLRM, compressão adaptativa | Hipótese central a ablar |
| Geração/refinamento, j | Construção ou revisão dos tokens de saída | Inserção, difusão, flow matching | Outra formulação de geração e objetivo |

Combinar eixos é possível, mas exige semântica explícita: quando atualizar ou reiniciar
memória, como evitar ler o futuro, o que compartilhar entre escalas e qual custo
atribuir a cada atualização. Repetir um loop latente não produz automaticamente
memória sequencial persistente nem geração paralela.

## Catálogo completo e decisões

| Item | Decisão para o LLM2-Lab | O que é aproveitável e o que corrigir |
|---|---|---|
| **1. Energia livre / inferência ativa** | Fora do plano de linguagem atual; possível linha própria para agentes | Inferência variacional e seleção de ações têm formulações concretas. Exigem modelo generativo, observações e preferências/políticas definidos. Não fornecem, por si, uma loss melhor para texto ou um critério de parada correto. [Formulação dos autores](https://pmc.ncbi.nlm.nih.gov/articles/PMC6848054/). |
| **2. Feynman / fases complexas** | Excluir a promessa de cancelamento automático de raciocínios errados | Amplitudes complexas são matematicamente válidas, mas faltam o mapeamento entre significado e fase e um objetivo que o aprenda. Contradições não chegam com fases opostas por definição. Simulação clássica também paga pelo cálculo dessas amplitudes. [Artigo original](https://authors.library.caltech.edu/records/9h858-5hv71). |
| **3. Reação-difusão** | Baixa prioridade para linguagem; experimento próprio se houver tarefa espacial | Atualizações locais e sinais concorrentes são hipóteses implementáveis; isso não estabelece vantagem em texto. Nas equações apresentadas, v tem o termo autocatalítico +uv²; u é consumido por esse termo. A descrição dos sinais no anexo está invertida. [Sistema Gray–Scott estudado por Pearson](https://arxiv.org/abs/patt-sol/9304003). |
| **4. Holografia / MERA** | Excluir holografia como justificativa de VRAM; manter fatoração tensorial apenas como hipótese técnica posterior | MERA representa eficientemente certas classes de estados quânticos. Isso não oferece compressão sem perdas arbitrária de pesos, documentos ou ativações do modelo. Qualquer adaptação precisa explicitar estrutura, dimensões de ligação, contrações e erro. [MERA original](https://arxiv.org/abs/quant-ph/0610099). |
| **5. Fractais / IFS** | Manter compartilhamento e multiescala como hipótese central | Compartilhar pesos economiza parâmetros frente a blocos distintos. Não garante capacidade equivalente, menor tempo ou dinâmica de IFS contrativa. Geração em árvore é outro problema. FractalNet é uma arquitetura de visão com topologia própria, distinta do nosso FLRM. [FractalNet](https://arxiv.org/abs/1605.07648). |
| **6. Fronteira fractal / ortogonalidade** | Separar o resultado científico da proposta de engenharia | O estudo de Sohl-Dickstein encontra fronteira fractal entre hiperparâmetros de treino estável/divergente; não prescreve pesos ortogonais nem prova aprendizado máximo na borda. Ortogonalidade/controle espectral só entram se um diagnóstico de profundidade justificar. [Estudo original](https://arxiv.org/abs/2402.06184). |
| **7. L-systems / hypernetworks** | Adiar; preferir primeiro o condicionamento pequeno já presente | Uma gramática de crescimento e uma rede geradora de pesos são mecanismos distintos. Hypernetworks são uma direção real, mas devem contabilizar gerador, pesos materializados, ativações e custo de geração. Regras compactas impõem estrutura; não reproduzem gratuitamente qualquer matriz densa. [HyperNetworks](https://arxiv.org/abs/1609.09106). |
| **8. φ / Fibonacci / KAM** | Separar três hipóteses; Fibonacci esparso fica para contexto longo, demais itens com baixa prioridade | Não há garantia de evitar repetição ou obter contexto infinito com frequências áureas. Saltos esparsos são um padrão de conectividade a comparar com local/global e distâncias em potências de dois. Lattices fornecem boas distribuições em problemas específicos, não vetores perfeitamente equidistantes em qualquer dimensão. [RoPE](https://arxiv.org/abs/2104.09864), [amostragem esférica com Fibonacci](https://isas.iar.kit.edu/pdf/SDFMFI23_Frisch.pdf). |
| **9. Regra 110 / NCA** | Baixa prioridade; família experimental separada | Universalidade computacional não demonstra eficiência, aprendizado ou auto-reparo. NCAs aprendidos podem regenerar padrões em tarefas treinadas para isso; nem toda regra celular tem essa propriedade. Atualização local também limita a velocidade de propagação de informação. [Growing NCA](https://distill.pub/2020/growing-ca/). |
| **10. TTT / fast weights** | Candidato posterior de prioridade média para memória/contexto longo | TTT layers usam um modelo como estado e aprendizado auto-supervisionado para atualizá-lo. O artigo avalia 125M–1,3B e registra limitações de I/O; não valida nosso regime de 2M/T256. Complexidade linear em comprimento, com dimensões fixas, não garante menor latência. Não é implementação direta da inferência ativa. [TTT layers](https://arxiv.org/abs/2407.04620). |
| **11. Liquid / CfC / LFM** | Separar CfC de LFM; baixa prioridade para CfC em texto | CfC parte de uma solução aproximada em forma fechada e tem motivação forte em séries temporais/controle. Não há garantia de ser imbatível em modelos pequenos nem de seu tempo dinâmico representar dificuldade lógica. LFM2, por exemplo, usa convoluções curtas com gates e atenção GQA. [CfC](https://www.nature.com/articles/s42256-022-00556-7), [relatório LFM2](https://arxiv.org/abs/2511.23404). |
| **12. RWKV-7 / xLSTM** | Candidatos úteis a baseline independente; escolher inicialmente um | São alternativas recorrentes concretas para testar qualidade/estado/latência. O(1) refere-se ao comprimento do contexto na inferência, com modelo fixo; não significa estado pequeno ou memória constante em todo treinamento. mLSTM tem memória matricial e processamento paralelo; isso não descreve todas as variantes de xLSTM. Nenhum é a melhor base por definição. [RWKV-7](https://arxiv.org/abs/2503.14456), [xLSTM](https://arxiv.org/abs/2405.04517). |
| **13. KAN / ChebyKAN / Wav-KAN** | Baixa prioridade para o SwiGLU de linguagem | O trabalho original destaca ajuste de funções, PDEs e interpretação científica. Esses resultados não garantem que um FFN KAN de 2M substitua um MLP de 20M em linguagem. Mudar a base para polinômios/Fourier exige medir custo, estabilidade e qualidade; escolher outra base não resolve automaticamente esses pontos. [KAN original](https://arxiv.org/abs/2404.19756). |
| **14. Flow matching / difusão discreta** | Possível linha futura de geração em blocos, com objetivo e avaliação próprios | Refinamento paralelo de tokens é uma proposta concreta. Há múltiplos passos do modelo e decisões de trajetória/amostragem; transporte ótimo e trajetória reta não são propriedades universais dessas famílias. Não basta ligar esse mecanismo no decoder causal atual. [Discrete Flow Matching](https://arxiv.org/abs/2407.15595). |
| **15. ACT / DEQ** | Preservar separados e condicionais | ACT aprende quanto computar entre entrada e saída; DEQ resolve equilíbrio e usa diferenciação implícita. Nenhum é requisito automático do fractal. Memória constante em profundidade não significa VRAM quase zero, poucas avaliações da função ou aceleração. [ACT](https://arxiv.org/abs/1603.08983), [DEQ](https://arxiv.org/abs/1909.01377). |
| **16. NCM / memória externa** | Corrigir a nomenclatura e especificar outra memória se a tarefa exigir | Nosso NCM significa Neural Context Memory: protótipos globais aprendidos, sem escrita do documento; implementação experimental em `src/llm2/architectures/ncm.py`. NCA, Neural Turing Machines e memória externa de leitura/escrita são outros mecanismos. Nenhuma dessas siglas garante solução do esquecimento catastrófico. [Neural Turing Machines](https://arxiv.org/abs/1410.5401). |
| **17. JEPA** | Preservado e desligado; sem descarte universal | A evidência local foi custo-benefício fraco. Não isolamos uma causa como perda inevitável de precisão de tokens nem demonstramos que todo JEPA falha em modelos compactos. Reconsiderar somente com tarefa representacional e controles adequados. [Auditoria histórica](results/CAMPAIGN_0015.md). |

## Correções que afetam diretamente o desenho

**Profundidade com poucos pesos não é capacidade equivalente.** K aplicações de um
bloco compartilhado custam aproximadamente K avaliações desse bloco, além da
interação/compressão. Backprop explícito pode reter ativações das iterações. O ganho
possível é usar os parâmetros e a computação de outra forma; precisa aparecer em
qualidade por custo e nos controles contra recorrência simples.

**Paralelismo em árvore não reduz todo o trabalho para log N.** Uma árvore binária
com 1.024 folhas tem 2.047 nós, incluindo a raiz. Dez rodadas de expansão não são dez
operações totais nem garantem coerência entre folhas. Modelos de inserção demonstram
que geração com poucas rodadas merece estudo, inclusive usando Transformer; isso
requer treinamento e fatoração apropriados. [Insertion Transformer](https://arxiv.org/abs/1902.03249).

**Ortogonalidade é uma restrição local.** Para uma matriz unitária U, ||Uh||=||h||.
Isso não vale automaticamente para o bloco completo com resíduos, gates e
não linearidades. Por exemplo, h+Uh dobra o estado se U=I; uma mistura entre h e
−h pode zerá-lo. O gradiente depende do produto dos Jacobianos completos, não apenas
dos autovalores de uma matriz. Preservar norma também não assegura um ponto fixo
único/atrativo. [RNNs unitárias](https://arxiv.org/abs/1511.06464),
[análise de gradientes recorrentes](https://arxiv.org/abs/1211.5063).

**Baixa entropia não significa resposta correta.** Um modelo pode estar confiante e
errado. Na inferência, não temos o próximo token verdadeiro para medir sua NLL.
Energia livre variacional pressupõe distribuições e um modelo generativo definidos;
não é um contador direto de FLOPs ou joules. Antes de uma parada adaptativa, medir
quanto cada exemplo melhora com mais iterações e calibrar um critério observável.
O custo do próprio critério e a execução GPU com profundidades diferentes entram
na conta. Inferência ativa ainda envolve seleção de ações/preferências, além de
atualização de estados. [Formulação dos autores](https://pmc.ncbi.nlm.nih.gov/articles/PMC6848054/).

**Geometria não garante recuperação de informação.** A representação de um inteiro
como soma de Fibonacci não garante que estados intermediários preservem os detalhes
dos tokens. Conectividade, compressão e recuperação são propriedades distintas. Uma
máscara esparsa aplicada por backend denso pode continuar pagando o custo denso.
Escolher φ não estabelece proteção geral contra ressonância: por exemplo,
φ²−φ−1=0 já é uma relação inteira entre suas potências. Qualquer proposta de novas
frequências precisa primeiro definir exatamente sua fórmula e faixa de períodos.

**TTT layers não eliminam o treinamento anterior.** O aprendizado externo treina os
parâmetros do modelo; o aprendizado interno atualiza o estado rápido. Isso não
significa retreinar toda a rede a cada token nem adquirir conhecimento permanente
entre documentos. TTT é um mecanismo específico, distinto de adaptar um Transformer
genérico durante a avaliação. [TTT layers](https://arxiv.org/abs/2407.04620).

**Memória tem diferentes significados.** Retenção no estado de uma sequência,
recordação de documentos e preservação de habilidades após novos treinos são
problemas diferentes. Para memória atualizável, definir escrita, leitura, capacidade,
esquecimento, isolamento entre documentos e estado de checkpoint. Fases ou uma
inicialização geométrica não substituem essas decisões.

## Onde uma implementação futura poderia entrar

Este mapa descreve fronteiras técnicas, sem implementar ou autorizar extensões:

| Hipótese | Local conceitual | Controle indispensável |
|---|---|---|
| FiLM / comunicação | Arquitetura FLRM | Fatorial do ciclo 2 e referências Transformer/LRM |
| Profundidade variável / ACT | Loop de refinamento e, se necessário, objetivo com custo de computação | Mesma arquitetura, distribuições de K explícitas, qualidade e latência por K |
| Compressão adaptativa | Construção causal das representações/patches | Compressão fixa, custo de fronteiras e teste de vazamento do futuro |
| TTT / RWKV / mLSTM / CfC | Backbone ou operador temporal com estado próprio | Baseline isolado antes de compartilhamento entre escalas |
| Padrão Fibonacci | Atenção e backend que realmente execute o padrão | SDPA denso, local/global, potências de dois, cobertura e recuperação |
| KAN | FFN em experimento específico | SwiGLU com ajuste e custo comparáveis |
| DEQ | Solver do estado e caminho de diferenciação | Resíduo do equilíbrio, convergência, gradiente verificado e custo completo |
| Difusão / flow / inserção | Objective, backbone e geração adequados | Métrica de probabilidade comparável quando disponível, qualidade/diversidade e latência total |

TTT/RWKV/mLSTM não são necessariamente substituições da interface atual
`AttentionKernel(q,k,v)`: precisam declarar atualização de estado, reset e execução
por prefixo. O contrato de [Backbone/Objective](../src/llm2/interfaces.py) é a
fronteira mais geral. Uma combinação com FLRM deve evitar atualizar a mesma memória
repetidamente com o mesmo token sem intenção explícita.

Para objetivos de difusão/flow, não comparar diretamente uma bound variacional ou
uma loss de denoising com a NLL autoregressiva e chamá-las de equivalentes. Estabelecer
avaliação compatível antes de concluir qualidade superior.

## Stack e escala

Manter Python + PyTorch/Inductor. Não existe corte técnico em 15M, 500M ou 3B que
obrigue migrar para Rust/JAX. A migração integral agora acrescentaria outra fonte
de diferenças antes de resolver a pergunta científica atual.

Rust/PyO3 só se um perfil mostrar gargalo de CPU que justifique a integração. Uma
árvore de controle rápida no host não acelera sozinha kernels pequenos, transferências
ou ramificações irregulares na GPU. Triton/fusão manual só depois de localizar um
operador caro e comparar equivalência e tempo no hardware real. Candle/Burn dependem
da cobertura das operações e da necessidade de deployment; não são destino obrigatório.

Precisão reduzida, checkpointing de ativações e distribuição de treino são decisões
de memória, hardware e orçamento. BF16/FP16 precisam validação específica e FP16
normalmente requer gradient scaling. Checkpointing troca armazenamento por
recomputação e preserva a formulação explícita do modelo; é um controle de engenharia
mais direto que adotar DEQ só para economizar ativações.
[Documentação AMP](https://docs.pytorch.org/docs/2.15/notes/amp_examples.html),
[checkpointing](https://docs.pytorch.org/docs/2.15/checkpoint.html).

O salto de 2M para 3B também exige orçamento de dados, otimização e avaliação. O plano
vigente continua sendo validação T4, confirmação independente e A100 condicional,
começando aproximadamente em 7M; 34M e escalas maiores dependem dos resultados.

## Programa concreto recomendado

1. **Agora: ciclo 2.** Comunicação × FiLM, referências Transformer/LRM, K=2,
   contexto sem RMS, seeds 81/82/83 e protocolo já congelado. Não acrescentar peças
   deste catálogo. [Pacote e execução](CYCLE2_RUN.md).
2. **Seleção e confirmação.** Se aparecer um componente útil, congelá-lo e confirmar
   em WikiText com documentos/splits explícitos, seeds novas e custo igualado. Se a
   comunicação não ajudar, simplificar e investigar recorrência sem depender do
   rótulo fractal. Se FiLM não ajudar, removê-lo da configuração preferida.
3. **Depois da base sólida: profundidade variável, em experimento próprio.** Treinar
   uma distribuição explícita de K e medir K=1/2/4/8, sem presumir extrapolação de um
   modelo treinado apenas em K=2. Usar tarefas verificáveis se a alegação for
   raciocínio, além de linguagem. Só investigar ACT após demonstrar benefício
   heterogêneo entre exemplos; DEQ segue outra condição, a de equilíbrio estável.
   [Precedente em profundidade recorrente](https://arxiv.org/abs/2502.05171).
4. **Compressão adaptativa, separadamente.** Comparar os fatores fixos 4/16 com
   patches causais determinados pelo conteúdo, medindo o custo de decidir fronteiras.
   Primeiro evitar confundir a mudança de compressão com a mudança de profundidade.
   [Precedente byte-level BLT](https://arxiv.org/abs/2412.09871).
5. **Contexto longo / estado: selecionar um concorrente antes de combinar.** Escolher
   RWKV-7 ou mLSTM; Gated DeltaNet, já no roadmap aprovado, é outra opção da mesma fila
   de alternativas. Selecionar pela implementação disponível, compatibilidade com
   o hardware, custo de integração e tarefa. TTT fica como candidato a memória
   adaptativa se retenção de contexto for um gargalo demonstrado. Comparar
   inicialmente sem FLRM; só depois testar compartilhamento/multiescala no vencedor.

Para escrita/recuperação, um teste inicial útil usa associações chave–valor novas em
cada documento, distâncias variadas, interferência e reset explícito. Isso separa
memória do documento de fatos memorizados nos pesos. Exigir causalidade durante
adaptação e impedir acesso a tokens futuros ou às respostas de avaliação.

Os limiares de continuidade já aprovados permanecem: redução média de pelo menos
0,005 NLL com custo igualado, ou pelo menos 10% menos tempo para qualidade fixada;
margem de equivalência de 0,002 NLL. Confirmação com cinco seeds novas, consistência
e intervalos reportados. Métricas de tarefas novas precisam critérios próprios
pré-fixados. Incerteza elevada ou conflito de métricas produz conclusão inconclusiva.

## Arquitetura concreta a investigar

```text
bytes → embedding → representações causais finas/comprimidas
      → núcleo Transformer compartilhado, com K controlado
      → comunicação e FiLM somente conforme a evidência
      → representação fina → cabeça de linguagem → próximo byte
```

Essa configuração já tem partes implementadas e controles preparados. As melhores
extensões próximas continuam profundidade variável e compressão adaptativa, uma por
vez. Uma família temporal alternativa e geração por refinamento são linhas
comparativas posteriores, cada uma com seu próprio protocolo. O blueprint com as
17 ideias simultâneas não fornece uma arquitetura especificada ou uma hipótese
atribuível: reúne atualizações de estado, objetivos e promessas incompatíveis com
uma comparação controlada no orçamento atual.

## Registro desta revisão

Nenhum código, configuração científica, dataset ou pacote de execução foi alterado.
Não houve novos testes de modelo, benchmark ou treino. Foram consultadas fontes
primárias e contratos locais; os checks desta revisão são documentais. O ZIP e o
launcher já entregues do ciclo 2 conservam seus hashes; esta documentação nova não
faz parte daquele snapshot imutável.
