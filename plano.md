\# Plano consolidado — arquitetura própria inspirada em fractais, com seleção de tecnologias por evidência



\## 1. Objetivo e papel do fractal



Desenvolver uma arquitetura de linguagem com \*\*melhor qualidade por custo de treinamento e uso\*\*, investigando recorrência, hierarquia, compartilhamento, compressão, memória e esforço adaptativo.



\*\*O Transformer será a referência externa a superar. A organização inspirada em fractais continuará como uma das principais hipóteses arquiteturais do projeto.\*\*



Essa inspiração terá significado concreto:



\- Representações em diferentes escalas.

\- Reutilização da mesma regra de processamento entre níveis compatíveis.

\- Processamento frequente de detalhes e menos frequente de representações amplas.

\- Investigação de refinamento adicional conforme a necessidade.



Fractal não será apenas um nome para qualquer arquitetura multiescala. Precisaremos distinguir o benefício da \*\*hierarquia\*\* do benefício da \*\*auto-similaridade e do compartilhamento entre níveis\*\*.



A inspiração no cérebro orientará funções verificáveis: memória de trabalho, retenção seletiva, diferentes ritmos de atualização e esforço variável. A arquitetura poderá mudar conforme os resultados, sem obrigação de reproduzir literalmente um fractal ou um mecanismo biológico.



\*\*Nenhuma tecnologia será obrigatória.\*\* RWKV, mLSTM, Gated DeltaNet, Memory Caching e TTT serão candidatos a componentes ou referências independentes.



O estado atual determina:



\- Comunicação multiescala × FiLM já foi investigada.

\- A campanha 0023 não confirmou benefício das escalas atuais.

\- O Transformer de dois blocos venceu o FLRM naquele regime.

\- Aumentar K após treinamento fixo em K=2 piorou os resultados.

\- O FLRM histórico permanecerá reproduzível; novas repetições dessa configuração ficam suspensas.

\- O Ciclo 5 preparado, ainda não executado, deixa de ser a próxima prioridade.



\*\*Preservamos a hipótese fractal e revisamos sua implementação.\*\* Base sólida significará correção, estabilidade, reprodução e custo conhecido; não dependerá de uma vitória do FLRM antigo.



\## 2. Frentes de pesquisa e tecnologias candidatas



| Frente | Pergunta principal | Candidatos e condição |

|---|---|---|

| \*\*Processamento recorrente\*\* | Como atualizar estados sem autoatenção densa sobre toda a sequência? | RWKV‑7, mLSTM e Gated DeltaNet; selecionar inicialmente uma implementação viável. |

| \*\*Organização fractal\*\* | Reutilizar uma regra entre escalas melhora qualidade por custo ou capacidade por parâmetro? | Hierarquia com compartilhamento explícito, comparada à mesma hierarquia com operações independentes. |

| \*\*Hierarquia multiescala\*\* | Processamento frequente e barato pode cooperar com processamento menos frequente? | Estados rápidos e lentos, com comunicação causal. |

| \*\*Compressão aprendida/adaptativa\*\* | Quais detalhes preservar e quando atualizar uma representação ampla? | Compressão fixa, aprendida e posteriormente fronteiras adaptativas; referências em BLT e Cantor. |

| \*\*Profundidade variável\*\* | Quais exemplos melhoram com computação interna adicional? | Treinamento com diferentes profundidades; RINS como referência de organização; ACT/PonderNet para parada aprendida. |

| \*\*Memória consultável\*\* | Como recuperar informação perdida pelo estado atual? | Memory Caching/GRM e outras memórias associativas, com orçamento explícito. |

| \*\*Memória adaptativa\*\* | Atualizar um modelo interno melhora retenção e adaptação? | TTT/fast weights em experimento separado. |

| \*\*Geração eficiente\*\* | Como reduzir trabalho repetido durante geração? | KV cache na referência Transformer, estados recorrentes na nova arquitetura e investigação posterior de FractalLLM. |



As famílias recorrentes justificam investigação, mas seus resultados não estabelecem ganhos na nossa escala. \[RWKV‑7](https://arxiv.org/abs/2503.14456), \[xLSTM/mLSTM](https://arxiv.org/abs/2405.04517), \[Gated DeltaNet](https://arxiv.org/abs/2412.06464).



Hierarquias recorrentes, BLT e Cantor fornecerão referências de organização e compressão. Aproveitar seus princípios não obriga adotar suas arquiteturas completas. \[Hierarchical Multiscale RNNs](https://arxiv.org/abs/1609.01704), \[BLT](https://arxiv.org/abs/2412.09871), \[Cantor](https://doi.org/10.1109/MC.2025.3624444).



\## 3. Sequência de desenvolvimento e experimentos



\### Etapa A — selecionar uma plataforma recorrente



Produzir fichas comparativas para RWKV‑7, mLSTM e Gated DeltaNet, contendo:



\- Atualização, leitura e esquecimento do estado.

\- Tamanho real dos estados.

\- Execução sequencial e treinamento por chunks.

\- Implementação oficial, versões, licença e requisitos de hardware.

\- Limitações conhecidas e evidência disponível.



Executar testes de correção e microbenchmarks das implementações viáveis na T4.



Selecionar como primeira plataforma a de menor tempo de treinamento medido entre as que passarem em correção e couberem na memória disponível. Em empate de até 5%, preferir menor estado.



Essa escolha será operacional. As outras famílias continuarão disponíveis; incompatibilidade de kernel não será interpretada como inferioridade arquitetural.



\### Etapa B — primeiro experimento de organização hierárquica e fractal



Comparar quatro variantes:



| Variante | Finalidade |

|---|---|

| Núcleo recorrente sem hierarquia | Controle da plataforma selecionada |

| Hierarquia com operações independentes | Medir o efeito da organização multiescala |

| Hierarquia com núcleo compartilhado entre níveis | Medir a contribuição específica da inspiração fractal |

| Transformer de dois blocos | Referência externa |



As duas hierarquias terão as mesmas dimensões, projeções, ritmos de atualização e operações. O compartilhamento será a diferença principal entre elas.



O primeiro desenho terá:



\- Um caminho fino atualizado a cada byte.

\- Um caminho amplo atualizado a cada quatro representações finas.

\- Compressão de estados já contextualizados.

\- Comunicação estritamente causal.

\- Um espaço latente compatível para o núcleo reutilizado.

\- Projeções necessárias presentes nos dois controles e contabilizadas.

\- Ausência de GRM, TTT, FiLM e parada adaptativa.



A economia será buscada pela distribuição do processamento entre níveis. Não manteremos automaticamente todo o processamento fino anterior e acrescentaremos ramos.



Os controles recorrente simples, hierárquico independente e Transformer terão orçamento próximo de 2M parâmetros, com diferença máxima de 5%. A variante compartilhada poderá ter menos parâmetros: \*\*não aumentaremos sua largura para esconder a economia do compartilhamento\*\*, pois isso alteraria também suas operações.



As comparações distinguirão:



\- Hierarquia versus recorrência simples.

\- Compartilhamento versus operações independentes na mesma hierarquia.

\- Qualidade e custo das alternativas frente ao Transformer.



Protocolo inicial:



\- WikiText preparado atual, byte-level.

\- Contexto 1.024 e batch efetivo de oito sequências.

\- 6.000 passos, seeds 241/242/243: \*\*12 treinos\*\*.

\- Triagem simétrica das taxas 0,0002 e 0,0004.

\- Avaliações completas em 50% e 100%; subconjunto fixo em 25% e 75%.

\- FP32 como referência inicial; precisão reduzida terá validação própria.



Esse ensaio testará uma arquitetura \*\*inspirada em fractais\*\*. Não será apresentado como prova de propriedades de um fractal matemático.



\### Etapa C — investigar mecanismos conforme o diagnóstico



A próxima campanha será escolhida pela limitação observada:



\- \*\*Hierarquia útil, mas compressão limitante:\*\* compressão aprendida e depois fronteiras adaptativas.

\- \*\*Compartilhamento prejudicial:\*\* investigar onde compartilhar e onde preservar especialização.

\- \*\*Retenção insuficiente:\*\* memória consultável, incluindo MC/GRM.

\- \*\*Atualização associativa insuficiente:\*\* TTT/fast weights.

\- \*\*Benefício de computação adicional:\*\* profundidade variável e depois parada aprendida.

\- \*\*Gargalo de execução:\*\* otimizar o operador identificado antes de acrescentar mecanismos.



Treinar explicitamente K=1/2/4 antes de avaliar adaptação. Separar escrita de memória de refinamento, evitando gravar repetidamente a mesma entrada ao aumentar K. RINS e PonderNet serão referências para essas perguntas. \[RINS](https://arxiv.org/abs/2502.07503), \[PonderNet](https://arxiv.org/abs/2107.05407).



MC/GRM e TTT continuarão como alternativas com custos e hipóteses próprios. Consultar todos os segmentos pode continuar custoso; atualizar um modelo interno também exige computação. \[Memory Caching](https://arxiv.org/html/2602.24281v1), \[TTT](https://arxiv.org/abs/2407.04620).



\### Etapa D — combinar, confirmar e escalar



Combinar inicialmente no máximo dois mecanismos. Comparar controle, componente A, componente B e A+B.



Confirmar candidatos com:



1\. Cinco seeds novas.

2\. Comparação com custo igualado.

3\. Segundo corpus, começando por TinyStories.

4\. Escala aproximada de 7–10M parâmetros, com dados suficientes.



TinyStories permitirá investigar linguagem em modelos pequenos, mantendo as conclusões restritas à tarefa. \[TinyStories](https://arxiv.org/abs/2305.07759).



A comparação futura de aproximadamente 10M contra 100M permanecerá como marco em tarefas definidas, sem pressupor capacidade equivalente.



\## 4. Medições, validação e continuidade



Cada sessão terá \*\*meta de duas horas e teto de três\*\*, incluindo preparação, compilação, treinamento e avaliação. Dois jobs independentes usarão as duas T4.



Pilotos terão margem de 35%. Matrizes que não couberem serão divididas entre sessões, preservando horizontes e schedulers.



Medir:



\- NLL, bits por byte e curvas de qualidade por custo.

\- Alvos processados e FLOPs estimados, com exclusões declaradas.

\- Tempo de treinamento e geração, separando compilação, prefill e decodificação.

\- RAM, VRAM, pesos, ativações e estados/caches.

\- Energia da GPU quando disponível.

\- Recuperação com distratores, distância, capacidade e esquecimento.

\- Problemas verificáveis de dificuldade controlada para estudar esforço adicional.



As comparações manterão cobertura equivalente em bytes. Perda de linguagem não será apresentada como demonstração de raciocínio geral.



Critérios de triagem:



\- \*\*Qualidade:\*\* redução média de pelo menos 0,005 NLL, favorável nas três seeds.

\- \*\*Eficiência:\*\* perda dentro de 0,002 NLL e economia de pelo menos 10% no recurso principal previamente escolhido.

\- \*\*Confirmação:\*\* cinco seeds novas, intervalos reportados e custo igualado.

\- Menos parâmetros, isoladamente, será descrito como compactação.

\- Configurações piores em qualidade e custo serão encerradas.

\- Incerteza elevada produzirá conclusão inconclusiva.



A validação já utilizada será tratada como desenvolvimento. O teste reservado só será consultado após congelar a seleção.



Testar causalidade, execução sequencial versus chunks, isolamento entre documentos, contagem e compartilhamento de parâmetros, forward/backward, finitude, memorização e retomada equivalente com dropout e acumulação.



Nas hierarquias, verificar também:



\- Bloqueio de influência futura nas fronteiras dos grupos.

\- Participação dos caminhos fino e amplo.

\- Igualdade inicial do forward compartilhado e independente quando os pesos correspondentes forem copiados e o dropout estiver desligado.

\- Ausência de duplicação dos pesos compartilhados na contagem.



\## 5. Tecnologias condicionais e entregas



Continuarão registradas, sem incorporação automática:



\- \*\*ACT/PonderNet:\*\* esforço adaptativo após demonstrar utilidade da profundidade variável.

\- \*\*DEQ:\*\* equilíbrio, solver e gradientes verificados.

\- \*\*Controle espectral, ortogonalidade e nGPT:\*\* dinâmica ou instabilidade que justifique investigação.

\- \*\*Kronecker/Monarch, MERA e hypernetworks:\*\* gargalos de pesos ou contrações demonstrados.

\- \*\*FiLM, JEPA, NCM, RMS de contexto, Ciclo‑V, MTP e currículo:\*\* preservados e desligados; retorno por pergunta específica.

\- \*\*Fibonacci/φ, lattices, KAN e variantes, Liquid/CfC/LFM2 e L-systems:\*\* baixa prioridade, com hipóteses próprias.

\- \*\*FractalNet, FANN e fractal nodes:\*\* referências de desenho.

\- \*\*Difusão, flow matching, geração em árvore, NCA e inferência ativa:\*\* linhas futuras separadas.

\- \*\*Compilação, AMP, caches, quantização e destilação:\*\* engenharia complementar, com ganhos separados dos ganhos arquiteturais.



Cada etapa entregará configurações, proveniência, resultados por seed, curvas, medições e decisão explícita: continuar, simplificar, interromper ou considerar inconclusivo.



O plano mestre, README, estado do projeto e registro de validação distinguirão evidência histórica, hipóteses ativas e possibilidades futuras. O código experimental continuará separado da branch comum.



\*\*A investigação fractal permanecerá explícita: descobrir se hierarquia e reutilização entre escalas oferecem uma vantagem real, e quais tecnologias ajudam essa organização a funcionar melhor.\*\*



