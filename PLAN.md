# LLM2-Lab — plano técnico da etapa 1

## Escopo e sequência

1. **Fundação e backbone:** pacote Python, YAML estrito, contratos de backbone/atenção/objetivo,
   Transformer causal pre-norm, RoPE, RMSNorm, SwiGLU, embeddings compartilhados.
   Verificar shapes, causalidade, contagem de parâmetros, forward/backward e finitude.
2. **Dados:** corpus UTF-8 local, split por documento antes do treino do tokenizador,
   BPE byte-level treinado só no split de treino; tokenizador byte para smoke tests.
   Artefatos com hashes, manifesto e sequências sem padding; sampler com estado independente.
   Verificar isolamento do split, reprodução dos artefatos e retomada do sampler.
3. **Treinamento:** objetivo causal separado, AdamW e scheduler separados, acumulação,
   clipping, FP32 CPU/BF16 CUDA, SDPA, compile opcional, avaliação e geração.
   Checkpoints atômicos em limites de optimizer-step, pesos safetensors e estado completo.
   Verificar memorização, estabilidade e igualdade de execução retomada/ininterrupta.
4. **Integração:** CLI, config CPU e ~34M/A100, notebook Colab, TensorBoard/JSONL,
   documentação, smoke end-to-end, testes opcionais CUDA/compile e CI CPU.

## Dependências

Python >=3.10; PyTorch >=2.5,<3; numpy; PyYAML; tokenizers; safetensors;
TensorBoard; pytest (desenvolvimento). Instalação direta no Python local, sem venv,
conforme solicitação. No Colab, preservar o PyTorch CUDA do runtime e instalar o pacote.
Registrar versões efetivas, Git commit/diff, config resolvida e hashes em cada execução.

## Decisões iniciais

- Referência: vocab=8192, d_model=512, n_layers=9, n_heads=8, hidden=1472,
  contexto=512: 33.990.144 parâmetros únicos; conferir por enumeração e fórmula.
- Inicialização aleatória; sem pesos pré-treinados. Loss autoregressiva como primeiro objetivo.
- Dados locais versionados por SHA-256; nenhuma fonte remota implícita. Exemplo sintético
  apenas para validar infraestrutura, sem alegar qualidade linguística.
- Atenção causal entre documentos concatenados separados por EOS; documentar essa política.
- Sampler com reposição e generator próprio, sem workers/prefetch; evita cursor ambíguo.
- Checkpoint só após optimizer-step e validação periódica; preserva modelo, objetivo,
  otimizador, scheduler, RNG Python/NumPy/Torch/ CUDA, sampler e contadores.
- Config e datasets incompatíveis devem falhar ao retomar; max_steps pode aumentar,
  mas a duração do schedule é explícita e imutável.
- Interfaces permitem estados opacos e objetivos adicionais no futuro. Não implementar
  JEPA, recursão, NCM, MoE ou combinações experimentais nesta etapa.

## Métricas e critérios de aceitação

- Parâmetros únicos **calculados**; FLOPs **estimados** a partir das operações densas,
  atenção e projeção de vocabulário, com fórmula e exclusões documentadas.
- Registrar tokens alvos treinados, tokens/s de treino, training/validation loss,
  pico CUDA allocated/reserved, tempo de treino sincronizado e tempo CUDA por eventos;
  CPU deve reportar VRAM/tempo de GPU como não aplicáveis.
- BF16 com RMSNorm/RoPE/loss em FP32 nas operações sensíveis; erro se hardware não suporta.
- Mesmo batch/semente/config deve reproduzir os artefatos de dados; manifesto e arquivos
  adulterados devem ser rejeitados.
- Shapes corretos; mudanças em tokens futuros não alteram logits passados; backward finito;
  entradas extremas no RMSNorm e teste opcional BF16 não devem produzir NaN/Inf.
- Modelo pequeno deve memorizar um minibatch com loss reduzida substancialmente.
- Checkpoint deve preservar estado relevante e produzir a mesma próxima atualização de
  uma execução ininterrupta (CPU FP32, kernels determinísticos, mesmo software/hardware).
- CLI deve preparar dados, treinar, validar, retomar e gerar texto em um smoke test.

## Riscos e limites

- Sem A100 local, não é possível afirmar throughput, uso de 40 GB ou validar seus kernels.
  Testes CUDA serão condicionais e notebook oferecerá verificação no hardware real.
- Determinismo entre versões, dispositivos e backends de SDPA/compile não é garantido.
  Comparação científica estrita deve fixar ambiente e usar SDPA math/eager determinístico.
- Compile tem custo de warmup e disponibilidade distinta em Windows/Colab; é opt-in.
- Checkpoints de estado usam pickle PyTorch e só devem ser abertos de fontes confiáveis.
- FLOPs analíticos não são profiler/benchmark; métricas de tempo abrangem os intervalos
  documentados, sem alegações de superioridade.
- Dados mínimos e batch CPU servem à engenharia, não demonstram capacidade de linguagem.

## Registro de execução

Este plano é escrito antes da implementação. Resultados efetivamente executados serão
registrados em VALIDATION.md ao concluir as etapas.
