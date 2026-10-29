# Auditoria de organização — 28/10/2026

## Escopo e plano antes das mudanças

Solicitação: organizar o projeto, revisar documentação e README, auditar arquivos versionados e `.gitignore`, e limpar os dois checkouts preservando a separação da pesquisa fractal.

1. Inventariar arquivos rastreados, saídas locais, notebooks, referências e dependências entre documentação, scripts e testes.
2. Separar documentação vigente de planos/relatórios históricos; criar índice e estado científico atual, incluindo o Transformer dentro de LRM/FLRM e a errata da campanha 0015.
3. Consolidar os artefatos locais em uma área ignorada, com manifesto de movimentação e hashes. Não apagar resultados, datasets ou checkpoints.
4. Reescrever o README para o uso atual de cada checkout, manter um guia operacional e ajustar links/referências afetados.
5. Corrigir regras de ignore sem esconder código/configurações úteis. Remover do índice apenas artefatos gerados efetivamente encontrados como rastreados; não reescrever o histórico Git.
6. Validar links, notebooks/pacote de distribuição, configuração e suíte de testes relevante em cada checkout. Registrar resultados reais em VALIDATION.md.

## Riscos e limites

- Preservar resultados, checkpoints e contratos de reprodução. Planos concluídos não são instruções atuais.
- Corrigir links e referências afetados; código/configuração de treino não será reformatado só por estética porque entra nos hashes de proveniência.
- Selecionar explicitamente este src ao executar CLI/testes; instalação editável pode apontar ao checkout comum.
- Manter recursos opcionais e baseline; esta limpeza não decide nova arquitetura nem inicia experimentos.
- Não criar commit/push nem reescrever histórico Git.

## Critérios de aceitação

- README e índice atuais, histórico separado e campanha 0015 documentada com errata.
- Ignore preciso e ausência de artefatos gerados indevidos no índice.
- Links, pacote e suíte verificados, com skips/limitações reais em VALIDATION.md.
- Alterações apenas organizacionais/documentais ou auxiliares necessárias, preservando comportamento científico e separação de branches.

## Resultados

### Organização e documentação

- README reescrito para o estado real de cada checkout; Transformer já compõe LRM/FLRM e continua sendo o controle científico.
- Índice, estado do projeto, guia operacional, política de arquivos, catálogo de configs e roadmap condicional publicados.
- Evidência resumida da campanha 0015 versionável em `docs/results/`: relatório corrigido, 18 linhas de resultados e proveniência. A revisão não criou novas medições de treino.
- Nove documentos históricos arquivados na base comum e 17 no checkout experimental. Receberam aviso de histórico e links corrigidos; os originais permanecem no Git. `AGENTS.md` mudou somente o caminho do plano V2 arquivado.
- Dezenove ZIPs, três diretórios de auditoria e quatro notas avulsas foram centralizados em `artifacts/` na base comum. Datasets, checkpoints e execuções existentes foram preservados.
- O manifesto local registra 52 movimentações e verificou 1.311 arquivos antes das atualizações documentais. Após a organização, os 1.285 arquivos de evidências/notas (294.149.052 bytes) mantêm tamanho e SHA-256 originais. Os 26 documentos históricos receberam apenas identificação/referências de arquivo após essa primeira verificação.

### Git e ignore

- Inventário inicial: 108 arquivos rastreados na base comum e 134 no experimento. Nenhum artefato gerado indevido exigiu remoção do índice.
- Nenhum ZIP, peso, binário de dataset, log ou bytecode foi encontrado no histórico acessível pelas branches/remotes. Os 50 objetos encontrados ao incluir todas as refs pertencem a snapshots internos `refs/codex/turn-diffs/checkpoints/`; não foram apagados nem confundidos com commits publicados.
- `.gitignore` cobre dados, execuções, artefatos, caches, checkpoints, outputs e arquivos `.env`; mantém código, YAML, tabelas curadas e templates de ambiente visíveis.
- Dezoito casos positivos/negativos de ignore passaram em cada checkout; `git ls-files -ci --exclude-standard` ficou vazio.
- A busca por padrões conhecidos de credenciais nos arquivos rastreados não encontrou candidatos. Isso não constitui garantia universal sobre segredos.

### Manutenção e limites da refatoração

- Empacotador passou a selecionar código/documentação por diretório e extensão, excluindo caches, arquivos ocultos e artefatos locais. Continua recusando sobrescrita do ZIP.
- Notebook Kaggle atualizado para revisão fixada, settings explícitos e `RUN_CAMPAIGN=False`; teste executa suas células padrão impedindo rede/processos externos. Notebook Colab recebeu correção do contexto histórico; não houve execução na nuvem.
- Testes de empacotamento verificam integridade do ZIP, seleção de conteúdo e recusa de sobrescrita. Ambos os notebooks são verificados quanto a sintaxe e ausência de outputs salvos.
- `src/`, YAMLs científicos e `pyproject.toml` não foram alterados. Nenhum módulo opcional foi removido nem houve merge entre branches.
- A inspeção estrutural não encontrou corpos de funções idênticos com pelo menos 15 linhas. Os módulos grandes de campanha/diagnóstico continuam candidatos a uma refatoração própria, com escopo e testes específicos; tamanho isolado não justificou mudar hashes científicos nesta limpeza.
- O runner experimental tem uma limitação existente ao imprimir símbolos Unicode em terminal Windows CP1252. O guia documenta `PYTHONUTF8=1`; seu código científico não foi modificado para corrigir apresentação.

### Validação e conclusão

O resultado das suítes por checkout está em [VALIDATION.md](../VALIDATION.md). Links relativos locais, regras de ignore e `git diff --check` foram conferidos. Inspeção de parâmetros em meta e planejamento sem execução funcionaram; o exemplo Kaggle foi corrigido para declarar origem dos dados e permitir inspeção em CPU.

O manifesto, os checks de preservação e os resumos de testes ficam em `artifacts/cleanup-2026-10-28/` da base comum, fora do Git. Não são logs brutos completos do pytest. Links externos, GPU real e execução habilitada dos notebooks não foram revalidados nesta etapa.

A organização está concluída. As alterações permanecem locais e revisáveis, sem commit ou push. Uma worktree organizada ainda mostra no Git os arquivos modificados/novos desta entrega; isso é esperado até um commit. A próxima campanha e a introdução de tecnologias continuam pendentes de definição científica, não de limpeza.
