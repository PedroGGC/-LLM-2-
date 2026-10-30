# Ciclo 4 — diagnóstico das escalas e dois scripts Kaggle

Protocolo novo, aprovado em 30/10/2026, na branch `experiment/fractal-lrm`.
O ciclo 3 e suas regras permanecem históricos. Este ensaio atribui contribuição
às escalas; não confirma superioridade geral nem eficiência com custo igualado.
Plano prévio: [PLAN_CYCLE4.md](PLAN_CYCLE4.md).

## 1. Backup, download e limpeza

Cole o conteúdo de `01_backup_e_limpeza_kaggle.py` numa célula, com `MODO="backup"`.
O script verifica workers, inventaria arquivos, confere hashes declarados nos
checkpoints sem abrir pickle e cria um ZIP exclusivo com manifesto e SHA-256.
Retém o último checkpoint completo por treino; se um grupo não puder ser
verificado, preserva todo o grupo. Pesos antigos não classificados, datasets,
fontes, configurações, relatórios e ZIPs existentes são incluídos conservadoramente.
Links têm suas definições registradas; seus destinos externos não são copiados.
Caches temporários explicitamente conhecidos são excluídos.

Baixe esse ZIP para o computador. Calcule o hash da **cópia baixada**:

```powershell
Get-FileHash -LiteralPath 'C:\caminho\llm2-working-backup-....zip' -Algorithm SHA256
```

Na mesma célula, altere `MODO="limpar"` e preencha `BACKUP_LOCAL_SHA256` com esse
resultado. O script revalida o ZIP, a ausência de workers e o inventário científico.
Somente então remove **todos os filhos de `/kaggle/working`**, inclusive o próprio
backup remoto, datasets, pesos, ZIPs, checkouts e reservas de números. A raiz fica;
`/kaggle/input`, destinos de links e arquivos do computador ficam fora do alcance.
O script não encerra processos automaticamente.

O recibo impresso e a variável `LLM2_CLEANUP_INFO` guardam o próximo número, mínimo
23, e o espaço liberado. Guarde esse número se reiniciar o kernel. Alterações no
inventário bloqueiam a exclusão e exigem novo backup. Falta de espaço para gerar
o ZIP interrompe antes de qualquer exclusão; não há fallback destrutivo.

**Git não é backup dos pesos.** ZIPs, dados binários e checkpoints são ignorados.
O ZIP de resultados 0022 recebido localmente não inclui checkpoints. A existência
de relatórios no Git não comprova a guarda local de todos os pesos antigos.

## 2. Nova célula de diagnóstico

Cole `02_rodar_diagnostico_fractal.py` em outra célula e execute. O relógio começa
nessa célula, separadamente da manutenção. A célula usa revisão completa e hashes
dos bootstraps, preserva o Torch instalado e seleciona código novo sem importar
uma instalação editável antiga no kernel. Requer Internet e duas T4. Se o kernel
foi reiniciado, preencha `run_number` com o número do recibo; colisões avançam para
um número livre, sem sobrescrever execução anterior.

Meta operacional: **7.200 s**. Teto: **10.800 s**, incluindo instalação, testes,
dados, compilação, treino, avaliação, diagnóstico e exportação. Não significa
esperar três horas se concluir antes. O piloto estima a matriz com margem de 35%,
900 s para pós-treino e 300 s de reserva de exportação. Se não couber no teto,
exporta o planejamento parcial e não inicia os treinos científicos. A estimativa
não garante duração: o deadline preserva checkpoints completos e exporta parcial.

| Variante | Escalas | Parâmetros por enumeração e fórmula |
|---|---|---:|
| FLRM fino | 1 | 2.164.994 |
| FLRM +4 | 1/4 | 2.165.382 |
| FLRM +16 | 1/16 | 2.165.382 |
| FLRM completo | 1/4/16 | 2.165.772 |
| LRM K2 com projeção de contexto em cache | 1 | 2.017.536 |
| Transformer, dois blocos e FFN384 | 1 | 2.164.992 |

Seeds **211/212/213**, 6.000 passos e quatro avaliações completas em
1.500/3.000/4.500/6.000: **18 treinos e 72 avaliações principais**. WikiText-103
raw com revisão/hashes do ciclo 3; treino de aproximadamente 32 MiB, byte-level,
T256, batch efetivo 32, FP32/Inductor, dropout 0,05. FLRM/LRM usam K2, D384,
FFN1024 e um bloco no núcleo compartilhado. As escalas retiradas não são
instanciadas nem calculadas. O controle fino mantém projeção/gate próprios.
`compressed_factors=None` mantém o caminho histórico; seleção explícita permite
`[]`, `[4]`, `[16]`, `[4,16]`. FiLM, JEPA, NCM, RMS de contexto, Ciclo-V, MTP e
currículo ficam desligados. O teste reservado não é consultado.

Um job independente por GPU, com distribuição rotacionada entre seeds. Pilotos
medem as seis variantes isoladas em ambas GPUs e o efeito de concorrência. Perfis
finais também são isolados; seus tempos não são misturados aos treinos concorrentes.
Há heartbeat a cada 30 s e logs técnicos completos. Os processos de compilação
criados pelo executor Linux pertencem a grupos próprios, encerrados no deadline;
isso não autoriza encerrar outros treinos existentes.

## 3. Exportação e leitura

Após cada job: ZIP de resultados e ZIP dos checkpoints **daquele job**. O bundle
final reúne o último checkpoint completo de cada treino iniciado. Snapshots são
exclusivos; não recompactam todos os pesos a cada conclusão. A admissão reserva
2 GiB livres além da estimativa de pesos, checkpoints, caches e exportações.
Baixe resultados e checkpoints; não presuma que o Git recebeu esses arquivos.

O relatório contém resultados por seed, curvas por FLOPs estimados, contrastes
pareados e intervalos exploratórios. Guarda as quatro validações e suas janelas.
O perfil final usa pesos finais da seed 211 com otimizador novo num clone de
medição, sem atualizar os checkpoints científicos; 90 amostras formam três grupos
de 30. Mede VRAM de treino/geração, RAM do processo quando disponível, bytes de
pesos/projeções temporárias e energia da placa via NVML quando disponível. Energia
por potência amostrada é explicitamente estimada; sem telemetria os valores são
`null`. Energia do host/startup/compilação fica fora dessa janela. A geração usa
eager, sem KV cache; seu custo é separado do treino compilado.

Nos seis modelos finais FLRM completo/LRM, a sensibilidade K=1/2/4/8 usa validação
completa adicional. Esses probes excedem as 72 avaliações principais e não provam
benefício de treinar com profundidade variável. Gates, RMS e gradientes são
coletados separadamente dos benchmarks. Relatórios CPU usam VRAM/tempo GPU nulos.

Comparação principal: **FLRM completo − somente fino**. Escalas candidatas exigem
redução média de pelo menos 0,005 NLL e benefício nas três seeds. Simplificações
candidatas exigem perda até 0,002 em cada seed e pelo menos 10% menos tempo por
passo nas duas GPUs. São critérios exploratórios; eficiência exigirá confirmação
posterior com custo igualado. NLL melhor não demonstra capacidade geral ou raciocínio.

FLOPs são estimativas, com duas operações por multiplicação-acumulação e backward
aproximado por duas vezes o forward. Para as escalas selecionadas de comprimentos
`s_i=ceil(T/f_i)`, o núcleo por iteração soma
`core_layers × (8 s_i D² + 6 s_i D H + 4 s_i² D)`; a interação acrescenta
`2 × soma(s_i) × D²`. A projeção de contexto em cache custa
`2 × soma(s_i) × D²` uma vez por forward; a cabeça ligada ao embedding custa
`2 T D vocab`. Multiplicar por K onde há iteração e pelo batch/contagem de passos.
As fórmulas não medem tempo nem energia; excluem normas, softmax, RoPE, gates,
pooling/upsampling, loss, otimizador e movimento de memória. Recomputação tem
contabilização separada quando habilitada; fica desligada neste ensaio.

## 4. Depois deste diagnóstico

A ordem e as condições de todas as tecnologias ficam no
[roadmap consolidado](RESEARCH_ROADMAP.md). Primeiro interpretar estas ablações;
nenhuma tecnologia nova dessa lista entra na configuração deste ciclo.

## 5. Reproduzir a entrega

Na revisão publicada, gerar os dois arquivos em diretório novo:

```powershell
python scripts/build_cycle4_release.py --revision REVISAO_COMPLETA --output-dir dist/cycle4-nova-entrega
```

O manifesto `release.json` registra hashes dos dois entrypoints. Não contém pesos
nem é um terceiro script. Não reutilizar um diretório de entrega já existente.
Testes reais e limites de validação constam em [VALIDATION.md](../VALIDATION.md).
