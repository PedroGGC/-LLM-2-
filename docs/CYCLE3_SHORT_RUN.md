# Ciclo 3 — lotes de até uma hora

Vigente após a [campanha 0019](results/CAMPAIGN_0019.md). O usuário escolheu teto de
**3.600 s por sessão**, incluindo bootstrap, dependências, dados, validação, piloto,
treinos, avaliações e exportação. Os últimos 300 s são reserva de exportação.
Uma hora é teto; o lote pode terminar antes. Execuções continuam seriais e isoladas.

**Atualização 0022:** [16 treinos preservados, quatro pares primários](results/CAMPAIGN_0022.md).
O par 104 terminou em cerca de 32 minutos, com as quatro avaliações de cada modelo.
Recomenda-se revisar a hipótese antes de iniciar outro lote: a regra positiva de
quatro vitórias FLRM em cinco seeds já não pode ser cumprida após duas vitórias em
quatro pares. As instruções abaixo descrevem a infraestrutura e as entregas anteriores;
o relatório atual orienta a decisão sobre os 14 jobs restantes.

## O que muda

- Até **dois treinos novos por execução**, em vez de tentar seis modelos de uma seed
  juntos. Um piloto decide se cada treino inteiro ainda cabe. Horizonte nunca é encurtado.
- Os 12 treinos da 0019 já estão na evidência curada versionada. Não serão repetidos.
  Na próxima sessão: FLRM âncora e Transformer com FLOPs estimados iguais, seed 103.
  Pelos tempos de 0019, esperar aproximadamente 35–45 min; confirmar pelo piloto,
  sem prometer essa duração em outro ambiente.
- Validação de código **focada por padrão**: JSON/dataset, protocolo, avaliação,
  causalidade, backward finito, memorização e retomada com dropout/acumulação em CPU;
  compilação/treino/checkpoint FP32 em CUDA para as famílias presentes no lote.
  A seleção e os resultados ficam nos logs. Não é uma execução da suíte completa.
  `validation_mode="full"` continua disponível, dentro do mesmo teto.
- Pilotos só dos modelos/slots realmente usados, com os mesmos dez warmups e trinta
  passos medidos. Validação completa tem piloto de duração separado, sem gravar NLL.
- Progresso da sessão a cada 30 s, além dos marcos do treinamento.
- ZIPs exclusivos de resultados e checkpoints após cada treino completo e ao final.

## O que permanece medido

Mesmo dataset/splits, T256/B32, FP32/Inductor, configurações, K e horizontes. Quatro
checkpoints e **quatro avaliações completas** de validação por treino; perda por
janela, contagem de alvos, curvas, perfis, VRAM, parâmetros, FLOPs estimados, tempo,
geração e diagnóstico final. Teste reservado continua fora do launcher.

Isso reduz a duração de cada sessão e elimina trabalho repetido; não torna gratuita
a confirmação de cinco seeds. Após a próxima seed, reavaliar o valor de completar
toda a matriz. Se a investigação parar cedo, a conclusão permanece exploratória.

## Código científico congelado

O operador publicado tem seu próprio SHA e snapshot em `operator/source.zip`.
Os workers usam outro checkout, **exatamente na revisão científica da 0019**:
`8798f6b84714d2c6671d970685984270078a982c`.
Os hashes de fontes/configs são conferidos antes da execução e nos checkpoints.
O código novo de organização não é injetado no backbone, Trainer, dados ou avaliação.
Runtime e fingerprint do dataset precisam coincidir com os resultados anteriores.

Não houve relaxamento de retomada ou agregação para aceitar código incompatível.
Resultados repetidos idênticos são deduplicados; duplicatas conflitantes de pesos,
loss, configurações ou curvas interrompem a combinação. Máscaras de slot/UUID e
host podem variar somente conforme o contrato de agregação já existente.

## Como executar e continuar

Ative Internet e GPU T4 x2; cole a célula curta entregue. Ela começa em
`run_number=21` na entrega corrigida (`20` na inicial), `session_seconds=3600`,
`max_runs=2`, seeds disponíveis 103/104/105.
O limite de dois jobs significa que somente o primeiro par pendente será executado.

Ao terminar, baixe:

- `llm2-campaign-0020-results.zip`: resultados novos e histórico agregado.
- `llm2-campaign-0020-checkpoints.zip`: pesos/estado dos treinos desta sessão.

Se houver interrupção posterior a um treino concluído, os arquivos
`llm2-campaign-0020-after-01-results.zip` e `...-checkpoints.zip` já estarão disponíveis.
Snapshots não substituem os anteriores. Estado de treino é pickle confiável apenas
se produzido pelo seu próprio fluxo; pesos continuam safetensors.

Para a próxima sessão, **anexe o último ZIP de resultados como Input** e use outro
`run_number`. Arquivos compatíveis em `/kaggle/working` ou `/kaggle/input` são
detectados automaticamente; assim a fila pula os jobs já concluídos. Os 12 da 0019
dispensam anexo porque sua evidência auditada acompanha o operador. Não há retomada
automática de um treino interrompido: ele conserva seu último optimizer-step para
retomada estrita separada.

O ZIP contém `history/queue.json`, com IDs restantes, e
`history/available_pairs.json`, com contrastes exploratórios para os pares já
disponíveis. O relatório principal mantém o requisito original de cinco blocos
completos para confirmação. Uma sessão completa não significa protocolo completo.

## Descoberta de ZIPs ilegíveis

A revisão `42d9179` podia parar com `BadZipFile` ao encontrar um arquivo vazio,
truncado ou que só tinha a extensão `.zip`. A correção registra caminho e motivo
em avisos e em `bootstrap.json/history_discovery`. Não modifica os candidatos.
Se a descoberta parar cedo, o diagnóstico fica também em `history-discovery.json`
na pasta temporária de código, antes de reservar número de execução.

Arquivos ilegíveis de tentativas até 0019 são contornados usando a evidência auditada
que acompanha o operador. Para tentativas >=0020, um histórico ilegível exige outro
snapshot válido da mesma tentativa que comprove a conclusão de todo seu lote.
Sem isso, a execução informa o arquivo e pede histórico válido, preservando a garantia
de não assumir que treinos desconhecidos não terminaram. Conflitos entre resultados,
código, dados e configurações continuam sendo rejeitados pelo operador.

A entrega usa 0021 porque a tentativa anterior pode ter reservado 0020 antes da falha.
Não é necessário apagar a reserva ou os arquivos anteriores. O teto de uma hora,
os dois jobs e o worker científico congelado continuam iguais.

[Plano/riscos](PLAN_CYCLE3_SHORT_SESSIONS.md) · [Validação real](../VALIDATION.md) ·
[Protocolo científico original](CYCLE3_RUN.md)
