# Política de arquivos e versionamento

## Versionar

- `src/`, testes, scripts, configurações, notebooks sem outputs e requisitos.
- Documentação vigente e decisões de pesquisa revisadas.
- Evidência pequena e curada em `docs/results/`: tabelas, CSV/JSON resumidos, protocolo,
  hashes dos pacotes recebidos e limitações. Não copiar traces ou milhares de janelas.
- Histórico documental relevante em `docs/archive/`, marcado como histórico.

## Manter localmente, fora do Git

| Pasta | Conteúdo |
|---|---|
| `data/` | Corpus bruto, tokenizador e datasets preparados |
| `runs/` | Treinos, checkpoints, métricas, TensorBoard e snapshots |
| `artifacts/campaigns/` | ZIPs recebidos de campanhas/revisões |
| `artifacts/audits/` | Auditorias completas e artefatos extraídos |
| `artifacts/notes/` | Dossiês e prompts de continuidade recebidos |
| `artifacts/cleanup-2026-10-28/` | Manifesto local da reorganização e logs de validação |
| `dist/` | Pacotes gerados para distribuição |

Essas pastas não são apagadas pela política de ignore. Fazer backup separado dos
checkpoints completos: ZIP de análise não é backup retomável. Não sobrescrever dados
ou execuções existentes. Novos artefatos devem ir diretamente à pasta correspondente.

## Conferir antes de um commit

```powershell
git status --short
git diff --check
git ls-files -ci --exclude-standard
git check-ignore -v artifacts/campaigns/llm2-campaign-0015-results.zip
```

Adicionar uma regra ao `.gitignore` não remove arquivos já rastreados. Se um artefato
gerado for encontrado no índice, usar remoção apenas do índice após conferir que a
cópia local será preservada. Não reescrever histórico para uma limpeza de rotina.

A auditoria de 28/10 não encontrou ZIPs, pesos, binários de datasets ou logs rastreados
nas branches nem em seu histórico local/remoto disponível. Objetos de artefatos em
`refs/codex/turn-diffs/checkpoints/...` são snapshots internos da ferramenta; não são
commits publicados das branches. Não apagar essas referências nem executar limpeza
agressiva do banco de objetos para ocultá-las.

## Histórico e reprodução

O manifesto local de movimentação contém origem, destino, tamanho e SHA-256 dos
arquivos preservados. Relatórios históricos dentro de `artifacts/audits/` podem citar
caminhos da estrutura antiga; esses caminhos são registros de origem, não o índice
vigente. Para localização atual, consultar o manifesto e esta tabela.

Código de `src/`, `pyproject.toml` e configs participa da proveniência dos treinos.
Refatoração cosmética desses arquivos não deve invalidar checkpoints sem necessidade.
Reprodução estrita usa o snapshot/ambiente original, sem relaxar verificações.

Credenciais `.env` são ignoradas; templates `.env.example`/`.env.template` continuam
versionáveis. Configurações YAML, relatórios curados e exemplos de código não são
ocultados por regras genéricas como `*.json`, `*.yaml` ou `*.zip`.
