# Memória contextual após 0032 — plano prévio, 10/11/2026

Autorização: alterar o núcleo e entregar uma rodada Kaggle para buscar vantagem
real sobre Transformer. Metas 2x/4x/8x são marcos condicionados a qualidade e
recurso declarado; não há fundamento para prometer 1.000.000x.

Diagnóstico: o núcleo anterior consulta com o byte atual e acumula transições
locais. Esta entrega testa outra família, preservando os protocolos anteriores.

## Núcleo proposto e precedentes

Memória matricial por cabeça: S_t = a_t S_(t-1) + b_t k_t v_t^T; leitura q_t S_t.
Consultas/chaves/valores/portas vêm de representação causal com convolução local.
Esquecimento dependente do conteúdo torna as atualizações sensíveis à ordem.
Passagens sucessivas recebem a representação contextual produzida pela anterior.
Cada passagem mantém estado próprio, mesmo quando compartilha pesos.

O princípio de retenção/portas tem precedentes em
[RetNet](https://arxiv.org/abs/2307.08621) e
[GLA](https://arxiv.org/abs/2312.06635). Nossa regra é uma adaptação escalar por
cabeça com scan em PyTorch; não reproduz os modelos/kernels completos dos artigos,
não implementa delta rule e não será anunciada como descoberta inédita.
[Gated DeltaNet](https://arxiv.org/abs/2412.06464) mantém atualização corretiva
como hipótese posterior; [TTT](https://arxiv.org/abs/2407.04620) permanece separado.

Quatro cabeças com ritmos iniciais 8/32/128/512, aprendíveis por portas. Q/K
normalizados em FP32, leitura normalizada por cabeça, residual, gate de saída e
SwiGLU. Sem DecoderBlock ou autoatenção densa global. Blocos locais32 + prefix scan
afim entre blocos, com referência sequencial e streaming. Scan por duplicação
tem trabalho O(N log N) no número de blocos nesta implementação; não prometer
treinamento estritamente linear nem velocidade dos kernels oficiais.

## Oito braços

- Transformers externos: ~86k, ~149k e 2,16M.
- Núcleo contextual simples, D80/FFN144, ~86k.
- Duas passagens, pesos compartilhados, estados separados, mesma largura.
- Duas passagens independentes, inicialização correspondente copiada, ~151k.
- Hierarquia fino/amplo compartilhada: fino a cada byte; média causal de quatro
  estados finos contextualizados; amplo a cada grupo completo, mesmo núcleo.
- Mesma hierarquia com operações independentes.

Comparar compartilhado/independente mantendo operações/dimensões; comparar
passagem fina adicional com processamento comprimido. Comunicação usa apenas
grupos completos; o grupo concluído pode influenciar o byte seguinte previsto
na posição de seu último input. Testar fronteiras e chunks irregulares.
Hierarquia é explícita, sem chamar toda recorrência de fractal matemático.

## Campanha e aceitação

TinyStories preparado da0032, byte-level/T256/B32, FP32/Inductor/dropout0,05.
Seeds341/342,6000passos,16treinos. Calibração train-only400passos nas taxas
0,0003/0,001, seed941; treino principal novo. Desenvolvimento completo em
1500/3000/4500/6000; teste reservado intacto. Mesmos prompts, 256 bytes novos
por exemplo, seis exemplos por job, sem seleção de textos.
Métrica e cobertura do corpus preservadas; fontes e protocolo novos explícitos.

Dois workers/T4, teto2h desde a célula, piloto com35%de margem e exportação5min.
Manter operação compartilhada com receitas explícitas, sem monkeypatch de globais
nem copiar todo o runner. Estados de calibração descartáveis; apenas um parZIPfinal,
último checkpoint completo porjob, cache próprio limitado/removido. Reuso somente
de jobs com protocolo/dados/config/código/ambiente iguais; nenhuma migração da0032.

Relatório porseed: NLL/bpb/textos, parâmetros, treino/forward/VRAM/energia com seus
escopos. Limiar exploratório0,005NLL e equivalência0,002. Ganho2x exige referência
nomeada, recurso medido, qualidade equivalente e ambas seeds; confirmação depois.
Relações de parâmetros serão compactação, não ganhos de qualidade equivalentes.

## Validação antes de entregar

Kernel sequencial versus blocos: forward, gradientes, estados, decaimento extremo.
Causalidade e invariância a chunking; isolamento de documentos; memória porpassagem.
Shared/untied com pesos copiados e dropout0 devem iniciar com mesmo forward;
contagem sem duplicatas e embedding ligado. Hierarquia deve participar da saída.
Probe de ordem da0032 precisa deixar de ser colisão estrutural; não equivale a
compreensão. Memorização/finitude e smoke de linguagem em cal_val, sem escolher
resultado principal. Retomada bit a bit com dropout/acumulação e piloto restaurado.
Runner: testes legados, novos jobs/configs/identidades, falha/deadline/disco,
ZIPs/downloads e integração CPU. CUDA/Inductor devem passar nas T4 antes de treino.
README/estado/plano/VALIDATION registrarão resultados reais e limitações.
