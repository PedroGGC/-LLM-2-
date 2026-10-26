# Manifesto da "IA 2": Além do Próximo Token

> **"A IA 1 aprendeu a imitar a linguagem superficial. A IA 2 precisa aprender a pensar em espaço latente antes de falar."**

---

## 1. O Diagnóstico: O Teto Estrutural da "IA 1"

A revolução dos Grandes Modelos de Linguagem (LLMs atuais) baseia-se em um princípio elegante, mas com limitações fundamentais: o **Transformer Causal Autoregressivo padrão alimentado por Predição do Próximo Token ($P(x_t \mid x_{<t})$)**.

Embora esse paradigma tenha gerado resultados impressionantes através de força bruta de escala (parâmetros, dados e computação), ele sofre de gargalos intrínsecos intransponíveis por mera escala:

1. **Computação Estática por Token:**
   Em um Transformer feed-forward tradicional, o modelo gasta rigorosamente a mesma quantidade de operações de ponto flutuante (FLOPs) para predizer uma vírgula trivial ou para deduzir um teorema matemático complexo. Não há mecanismo interno de deliberação antes da emissão.
2. **A Ilusão do Raciocínio por Cadeia Textual Discreta:**
   O "Chain-of-Thought" (CoT) atual força o modelo a verbalizar todo o seu processo em tokens de texto discretos. Isso gera três problemas graves:
   - Poluição exponencial do contexto de atenção ($O(N^2)$).
   - Acúmulo e propagação multiplicativa de erros de amostragem (*compounding errors*).
   - Ineficiência cognitiva: cérebros biológicos não pensam palavra por palavra antes de ter uma ideia; o raciocínio ocorre em estados contínuos abstratos e apenas o resultado final é vertido em linguagem.
3. **Miopia Superficial de Alvo:**
   O objetivo de *Cross-Entropy* em tokens discretos pune o modelo igualmente se ele errar um sinônimo perfeito ou cometer um erro catastrófico de lógica. Falta uma compreensão estrutural de *modelo de mundo* que capture dinâmicas semânticas de alto nível.

---

## 2. A Tese da "IA 2": Pilares da Nova Arquitetura

O projeto **LLM2-Lab** foi concebido e implementado do zero para transformar essa tese em ciência experimental reproduzível. A **IA 2** não é uma reformulação cosmética ou um wrapper em APIs existentes: é uma reformulação arquitetural e de objetivos de aprendizagem, fundamentada em cinco pilares sinérgicos:

```
                         ┌──────────────────────────────────────────────┐
                         │              A ARQUITETURA IA 2              │
                         └──────────────────────────────────────────────┘
                                                │
         ┌──────────────────────┬───────────────┴──────────────┬──────────────────────┐
         ▼                      ▼                              ▼                      ▼
  [ 1. LRM ]             [ 2. JEPA ]                     [ 3. FLRM ]            [ 4. MTP ]
  Raciocínio             Modelo de Mundo                 Processamento          Planejamento
  Recorrente             Latente Preditivo               Multi-Escala Fractal   Multi-Passo
  Contínuo               (Anti-Colapso)                  (FiLM + Janelas)       (Horizontes)
         │                      │                              │                      │
         └──────────────────────┴───────────────┬──────────────┴──────────────────────┘
                                                │
                                                ▼
                                    [ 5. CURRÍCULO PROCEDURAL ]
                                    Geração & Verificação Simbólica OOD
```

### Pilar 1: Raciocínio Recorrente em Espaço Latente (LRM — Latent Recursive Model)
- **O que faz:** Introduz um estado de pensamento contínuo $z_t \in \mathbb{R}^D$ associado a cada posição temporal, atualizado iterativamente por um núcleo causal compartilhado (*Shared Recursive Core*).
- **Mecanismo:** Através de portas sigmoides aprendidas e refinamento latente progressivo (`latent_steps`), o modelo pode "pensar" por $N$ iterações no espaço contínuo antes de projetar para os logits da linguagem.
- **Vantagem:** Permite alocar profundidade computacional dinâmica para resolver problemas sem emitir tokens fictícios ou inchar o contexto autoregressivo.

### Pilar 2: Modelo de Mundo Preditivo e Representacional (JEPA — Joint Embedding Predictive Architecture)
- **O que faz:** Em vez de supervisionar o modelo apenas com o token textual imediato, introduz um objetivo auxiliar de predição latente futura inspirado na visão de Yann LeCun.
- **Mecanismo:** Um codificador professor causal (*Target Teacher*, atualizado via EMA) produz a representação abstrata de estados futuros $z_{t+\text{offset}}$. A rede online deve prever essa representação sem gradiente no professor, combinada com uma **penalidade de variância dimensional** que impede matematicamente o colapso de representação (*representation collapse*).
- **Vantagem:** O modelo aprende a estrutura conceitual subjacente do raciocínio e das relações causa-efeito, e não apenas correlações superficiais de n-gramas.

### Pilar 3: Recorrência Fractal Multi-Escala (FLRM — Fractal Latent Recursive Model)
- **O que faz:** Organiza a percepção e o raciocínio em múltiplas escalas temporais hierárquicas simultâneas.
- **Mecanismo:** Três resoluções causais operam em paralelo:
  1. *Escala Local:* Granularidade fina token a token (janela 1).
  2. *Escala Intermediária:* Agrupamentos locais de conceitos e submetas (janela 4).
  3. *Escala Global:* Prefixo causal cumulativo de longo alcance.
  As escalas interagem por meio de modulação FiLM (*Feature-wise Linear Modulation*) e portas de fusão compartilhadas, gerando um processamento recursivo de invariância de escala (fractal).
- **Vantagem:** Captura tanto detalhes sintáticos microscópicos quanto a coerência semântica macroscópica sem explodir a complexidade.

### Pilar 4: Predição Multi-Token Paralela (MTP — Multi-Token Prediction)
- **O que faz:** Cabeças preditivas secundárias aprendem a antecipar tokens em horizontes futuros ($t+1, t+2, \dots, t+k$) simultaneamente a partir da representação latente.
- **Vantagem:** Obriga a espinha dorsal (*backbone*) a desenvolver representações mais ricas e voltadas a planejamento de longo prazo, acelerando a convergência e a capacidade de antecipação.

### Pilar 5: Currículo Procedural Fechado e Verificação Formal
- **O que faz:** Substitui o pré-treino passivo em textos desestruturados por um motor determinístico de geração sintética e verificação procedural.
- **Mecanismo:** Nove famílias de geradores formais (aritmética, manipulação simbólica, parênteses e escopos, lógica booleana, sequências temporais) com separação rígida entre treino, validação e domínios fora de distribuição (*OOD - Out Of Distribution*).
- **Vantagem:** Impossibilita o modelo de "trapacear" por memorização estocástica e fornece métricas exatas de acurácia de raciocínio lógico.

---

## 3. Filosofia Científica: Engenharia Sem Autoengano

Muitos projetos proclamam "a próxima geração da IA" apoiando-se em marketing e benchmarks inflados. O **LLM2-Lab** foi construído sob um pacto inegociável de rigor empírico:

- **Transparência de Custo Real:** Todo FLOP de recorrência, passo latente, teacher JEPA e cabeça MTP é explicitamente contabilizado na fórmula analítica de orçamentos. Não há "computação mágica gratuita".
- **Reprodutibilidade Estrita:** Checkpoints determinísticos preservam sementes RNG de Python, NumPy, PyTorch CPU e CUDA, contadores de amostragem e estados de otimizador em safetensors e tensores isolados.
- **Ablações Científicas Limpas:** Cada hipótese possui sua respectiva linha de controle (E0–E3 para LRM, F0–F4 para Fractal, C0–C6 para Currículo). O código fractal reside em branch própria (`experiment/fractal-lrm`) para não contaminar a baseline de comparação causal da Etapa 1.
- **Verificação Contínua:** Cobertura integral de testes automatizados com garantias formais de causalidade temporal estrita (máscara triangular sem vazamento para o passado), parada de gradiente no JEPA e invariância de prefixo.

---

## 4. Onde Estamos e Próximos Passos

1. **Etapa 1 (Concluída):** Transformer causal puro 34M, tokenizadores offline, pipeline de dados, validação bit-a-bit e infraestrutura de treino CPU/A100.
2. **Etapa 2 (Concluída no código):** LRM, JEPA auxiliar com regularização de variância, FLRM fractal multi-escala, MTP, currículo procedural com geradores e verificadores, e suite completa de testes unitários aprovada.
3. **Etapa 3 (Em Andamento):** Execução da matriz de experimentos em hardware acelerado (GPU A100), mapeando empiricamente se a recorrência latente contínua e o objetivo JEPA superam a perda de perplexidade e a generalização OOD frente ao Transformer puro com orçamentos de FLOPs equivalentes.

---

*O LLM2-Lab é a fundação em código aberto para quem não quer apenas consumir o que a IA 1 oferece, mas construir os alicerces teóricos e práticos do que a IA 2 será.*
