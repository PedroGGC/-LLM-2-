# Diagnóstico pós-treino da 0027 — escopo prévio, riscos e aceitação

01/11/2026. Próxima operação autorizada após a auditoria0027. Não treinar nem
alterar a matriz original; seeds242/243 permanecem pendentes.

Avaliar os quatro pesos finais da241, com fonte científica original1c8ce2a,
configurações/dados/versões verificadas. Usar validação oficial de desenvolvimento,
mesmas janelas T1024/B8, FP32, dropout desligado, execução eager nesta operação.
O teste reservado não será avaliado. Não carregar `training_state.pt` ou pickle.

Condições, com todos os pesos congelados:

- Transformer: original.
- Recorrente simples: original e somente saída da primeira célula.
- Cada hierarquia: original, ramo amplo silenciado, ganho amplo0,25 e0,5.

São11condições sobre quatro checkpoints. Uma passagem do backbone por batch
fornece os componentes e permite calcular leituras alternativas. Não apresentar
intervenção pós-treino como ablação treinada, nem ganhos como aceleração.

Medir NLL/bits por byte, perda por posição módulo4 e posição na janela, RMS dos
caminhos fino/amplo, proporção e cosseno. Confirmar reconstrução da saída original
com os componentes; repetir NLL original com tolerância absoluta pré-fixada5e-5
frente à0027. Divergência bloqueia a interpretação e exporta diagnóstico parcial.

Dois workers independentes, um por T4, quatro jobs em pares. Sem Inductor nesta
operação: não criar cache de compilação nem executar nova qualificação/triagem.
Limite padrão7200s incluindo setup; máximo permitido10800s, reserva300s. Piloto
de dois batches por worker estima duração com margem35% antes da avaliação inteira.
Recusar job que não caiba; não reduzir cobertura para chamá-lo de completo.

Descobrir pasta0027 ou arquivos de nomes exatos em working/input. Conferir pesos,
metadados e referência fixada. Se precisar restaurar, copiar somente metadados e
safetensors finais para pasta própria; nunca extrair o ZIP inteiro. Se não houver
pesos, falhar antes de gastar GPU. Reutilizar dados originais com fingerprint exato;
se ausentes, reconstruir com importador e shards originais, sem aceitar outro hash.
Preservar originais e não sobrescrever nenhuma execução/dataset.

Um ZIP final de resultados, sem cópias de pesos/otimizadores. JSON por job publicado
atomicamente; em falha/deadline, exportar os completos e o erro. Remover apenas
restaurações temporárias criadas por esta operação, após validação do resultado.
Botão de download embutido; não gerar `after-NN` ou backups cumulativos.

Validação local: intervenção identidade, causalidade, componentes equivalentes nas
duas hierarquias, ausência de mudanças nos pesos/RNG/modos, contagens e perdas
ponderadas, hooks removidos mesmo em falha, segurança de caminhos/ZIPs, fontes,
hash/config/versões divergentes bloqueados, ausência de pickle/otimizador,
falha/deadline/parcial, restauração sem duplicar payloads e bootstrap fixado.
Integração CPU com fonte original em subprocesso; CUDA/eager no ambiente real
continua dependente da rodada Kaggle. Registrar testes reais emVALIDATION.md.

Decisão: verificar se o ramo amplo ajuda ou atrapalha os pesos atuais, localizar
desbalanceamento e identificar o controle treinado necessário a seguir. Não
inferir sozinho que compressão aprendida, gate ou um núcleo fino treinado resolverá.
