# Referências consultadas e diferenças da implementação

Consulta em 25–26/10/2026. Código V2 escrito para este repositório, reutilizando os seus
blocos da etapa 1; **nenhum código de terceiros foi copiado/vendorizado**. Licenças abaixo
são das fontes consultadas, não uma atribuição automática de licença ao LLM2-Lab.

| Fonte primária | Uso e diferença |
|---|---|
| [LLM-JEPA](https://arxiv.org/abs/2509.14252), [repositório](https://github.com/galilai-group/llm-jepa) | Inspira previsão em espaço de representações; aqui prevemos hidden futuro causal, sem reproduzir o protocolo multivista |
| [TRM](https://github.com/SamsungSAILMontreal/TinyRecursiveModels) (MIT) | Inspira compartilhamento/recursão; sem copiar treinamento, ACT ou resultados |
| [HRM](https://arxiv.org/abs/2506.21734) | Inspiração de profundidade recorrente; não implementamos os dois níveis do artigo |
| [Coconut](https://github.com/facebookresearch/coconut) (MIT) | Inspira computação contínua; refinamos estados por posição, sem replicar a troca de CoT por embeddings |
| [DeepSeek-V3](https://github.com/deepseek-ai/DeepSeek-V3) | Inspira offsets auxiliares MTP; nossas cabeças lineares não são o módulo original nem speculative decoding |
| [FractalNet](https://arxiv.org/abs/1605.07648) | Referência de hierarquia; FLRM usa operações multiescala causais próprias e residuals |
| [VICReg](https://arxiv.org/abs/2105.04906) | Inspira monitoramento/penalidade de variância; não incluímos toda a regularização covariante |
| [Skill-It](https://github.com/HazyResearch/skill-it) (Apache-2.0) | Seleção de dados por habilidades; política própria simples, sem descoberta automática de grafo |
| [Curriculum Learning](https://icml.cc/Conferences/2009/papers/119.pdf) | Fundamentação de ordenação de dificuldades |
| [Self-Paced Learning](https://papers.nips.cc/paper_files/paper/2010/hash/e57c6b956a6521b28495f2886ca0977a-Abstract.html) | Fundamentação de seleção conforme estado de aprendizagem, sem reproduzir seu otimizador |
| [LearnHarness](https://github.com/amosblomqvist/learnharness) | Inspiração pedagógica de feedback; não evidência de aceleração de treino neural |
| [Adaptive Computation Time](https://arxiv.org/abs/1603.08983) | Referência futura; nenhuma parada aprendida implementada |
| [Deep Equilibrium Models](https://arxiv.org/abs/1909.01377) | Referência futura; não usamos solução de ponto fixo nem gradiente implícito |
| [NCM](https://github.com/HazyResearch/ncm), [BLT](https://github.com/facebookresearch/blt), [Engram](https://github.com/deepseek-ai/Engram), [FLA](https://github.com/fla-org/flash-linear-attention) | Arquiteturas externas não reproduzidas; o banco local de protótipos não equivale a um NCM completo ou memória persistente |

Consultar as licenças completas de código/pesos/datasets novamente antes de qualquer reutilização
futura. Os resultados publicados por esses projetos pertencem aos seus protocolos; não são
resultados deste laboratório e não sustentam uma promessa de redução de 100.000x aqui.
