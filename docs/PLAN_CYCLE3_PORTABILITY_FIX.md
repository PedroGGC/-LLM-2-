# Correção de portabilidade do ciclo 3 — 29/10/2026

## Problema e escopo

A execução Kaggle de `855c8d2` terminou antes dos testes/treinos, em
`verify_dataset`. O hash fixado de `tokenizer.json` foi gerado em Windows com CRLF
(`a9be2688...`); a mesma serialização JSON em Linux usa LF (`f88f84a3...`).
Reprodução local: o arquivo de 43 bytes contém quatro CRLF; convertido a LF são
39 bytes e o segundo hash. Os tokens/vocabulário são os mesmos, mas bytes e hash
não são. Não aceitar hashes alternativos nem remover o tokenizer da verificação.

## Plano, riscos e aceitação antes das mudanças

1. Reproduzir o defeito com teste que simula defaults LF/CRLF em qualquer host.
2. Fixar `write_json` em UTF-8/LF explícito para novas gravações; preservar o formato
   JSON e a leitura de datasets antigos. Não migrar nem sobrescrever arquivos existentes.
3. Atualizar somente o hash de tokenizer do protocolo para o LF canônico. Manter
   hashes dos três binários, revisão/shards, seleção e exclusão de duplicação.
   Acrescentar diagnóstico discriminado com campo/esperado/observado.
4. Preparar novamente o corpus real em diretório novo e verificar todos os hashes.
   Testar rejeição de adulteração e leitura estrita de dataset legado CRLF.
5. Rodar regressões e suíte completa; registrar resultados reais e publicar novo SHA.
   Entregar célula curta com novo `run_number`, teto de quatro horas e cache preservado.

A função de JSON é compartilhada por manifestos e metadados: novas gravações no
Windows terão bytes diferentes. Datasets/checkpoints existentes não são reescritos;
retomada continua exigindo código, configuração, arquivos e ambiente originais.
Sem alteração de treinamento, modelos ou critérios científicos. CUDA não está
disponível localmente; os testes do hardware real continuam pré-requisito no launcher.
