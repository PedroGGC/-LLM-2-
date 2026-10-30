# Campanha 0021 — snapshot após o primeiro treino, 30/10/2026

Recebidos os ZIPs `after-01`: resultados 3.391.995 bytes, SHA-256
`9043115ab066614e5a567f334a32d9be5ed75e5d90793536d9a52112f2bee223`;
checkpoints 26.791.183 bytes, SHA-256
`4bd407d901ea4af8a11f3c12b346a5d308fc69b13c6e3c243540e1ffd6ca49b5`.
CRC íntegro, sem nomes repetidos. Operador `c518ee7`, worker científico `8798f6b`.

## Trabalho preservado

FLRM com comunicação, sem FiLM, seed 103: **8.000 passos completos**, 65.536.000
alvos, quatro avaliações completas de validação. NLL final **1,320334273**;
806,41 s de tempo acumulado de treino, 825,89 s na fase externa. O snapshot foi
gerado aos 1.114,44 s (18min34s) de sessão. Histórico agregado: **13 treinos completos**.

No momento do snapshot, `flops_transformer-seed-103` estava pendente. O estado atual
dessa execução não foi recebido; ela pode ter avançado após gerar o arquivo. O
`FileExistsError` da nova tentativa comprova somente que o número já estava ocupado,
e protege a pasta original. Não comprova falha do treino anterior nem autoriza apagar
seus dados. Os arquivos confirmam atividade real apesar de a saída não estar visível.

## Auditoria

- Quatro NLLs reconstruídas das janelas completas: 1.144.064 alvos cada, erro máximo
  4,05051504e−9. Identidade de código/configs/dataset igual à 0019.
- Hash dos pesos `54cbce20834855ad6b2de0c4bb3ebb13ffbaf8766ac7b56d2ae0b9269835da7c`;
  `model.safetensors` e `training_state.pt` conferidos contra metadados e avaliações.
  Pickle não foi carregado.
- Header safetensors: 2.165.772 elementos de parâmetros únicos, mais buffer inteiro
  `training_step=8000`. `lm_head.weight` compartilha `embedding.weight`; não foi
  contado duas vezes.
- Snapshot completo do operador comparado byte a byte ao Git, com checkout de
  referência LF explícito para evitar conversão de finais de linha na auditoria Windows.
- Logs da validação focada e perfil/treino T4 foram preservados. Isso verifica a
  execução desse primeiro job; não representa conclusão do segundo ou de todo o lote.

Os arquivos originais foram preservados. Auditoria reproduzível no checkout
experimental: `artifacts/audits/campaign-0021-2026-10-30/audit.py` e `audit.json`.

## Continuação

Antes de iniciar outro lote, conferir processos e o `campaign.json` vivo. Se o
comparador ainda não tiver terminado e não houver checkpoint parcial a recuperar,
usar novo número (pelo menos 0022), com até um job para completar o par 103.
Preservar/anexar o ZIP de resultados after-01 para a fila pular o FLRM já concluído.
Não comparar a média FLRM de três seeds à média de um controle com somente duas.

Limpeza preparada: [recuperação do working](../KAGGLE_WORKING_RECOVERY.md).
