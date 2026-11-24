# Plano Reservado para Colab `L4 / A100` — Ablação `Titans / TTT` + Smoke `HALO` & `OPD`

**Status:** Preparado e validado em CPU (`tests/test_lumina1_upgrades.py`), **aguardando janela de cota do Google Colab (`L4` / `A100`) e definição final do professor de `Lumina 1`**.

---

## 1. Por que testar `Titans / TTT` especificamente em `L4 / A100` (e não travar a `Lumina 1` na T4)?

- **Mecanismo implementado em [`src/llm2/architectures/titans_ttt.py`](../../src/llm2/architectures/titans_ttt.py):** `TitansMemoryCell` mantém uma MLP de 2 camadas (`W1, W2`) + momento (`S1, S2`) como estado recorrente $O(1)$, atualizada causalmente em tempo de teste via gradiente analítico de surpresa $\ell(M; k_t, v_t) = \frac{1}{2}\|M(k_t) - v_t\|_2^2$ com portas aprendidas de esquecimento ($\alpha_t$), momento ($\eta_t$) e passo ($\theta_t$).
- **Motivo do hardware `L4 / A100`:** `Gated DeltaNet-2` é o caso linear de 1 passo de TTT (barato em `sm_75` Turing); já a MLP de 2 camadas do `Titans` exige maior largura de banda SRAM/BF16 para competir em `tok/s`. A `L4` (`BF16` nativo, `48 MB` L2) ou `A100` (`1,5 TB/s`) são o ambiente certo para decidir empiricamente se `Titans` substitui parte das camadas `DeltaNet` em uma **Lumina 2** futura ou serve como módulo de memória longa externa.

## 2. Comandos Prontos (NÃO EXECUTAR automaticamente)

1. **Ablação `Titans / TTT` vs `Gated DeltaNet-2` (Colab `L4 / A100`, `BF16`):**
   ```bash
   python scripts/bench_lumina1_halo_opd_and_titans.py --mode titans_colab --seq-len 2048
   ```
   **Critério de aprovação:** `TitansMemoryCell` só entra em arquitetura principal se superar `Gated DeltaNet-2` em recuperação associativa / perplexidade longa **sem** degradar o throughput de treino em mais de `2×` na `L4/A100`.

2. **Smoke `HALO` (Sonda de Sensibilidade) + `Global + KV INT4` + `OPD` + `QuaRot W4A16` (`act_bits=16`):**
   ```bash
   python scripts/bench_lumina1_halo_opd_and_titans.py --mode halo_opd --teacher <ID_PROFESSOR_DEFINIDO>
   ```
   **Critério de aprovação:** `rope_theta` extraído exato (sem fallback `10000`), `cos >= 0.9999` nas camadas globais selecionadas pelo `HALO`, `OPD` finito em `Top-128 KL`, e `QuaRot W4A16` real (`act_bits=16`) com throughput superior ao `5.1 tok/s` da C018.
