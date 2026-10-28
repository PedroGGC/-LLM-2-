"""Portuguese console views; measured values remain in the original JSON logs."""

import math
import statistics


MODEL_NAMES = {
    "baseline": "Transformer", "baseline_small": "Transformer pequeno",
    "baseline_jepa": "Transformer + JEPA", "lrm": "LRM", "lrm_no_jepa": "LRM sem JEPA",
    "lrm_no_jepa_k1": "LRM sem JEPA (K=1)", "lrm_no_jepa_k2": "LRM sem JEPA (K=2)",
    "flrm_reference": "FLRM sequencial", "flrm_batched": "FLRM agrupado",
    "flrm_compressed": "FLRM comprimido",
    "flrm_compressed_native": "FLRM comprimido — projeção nativa + JEPA",
    "flrm_compressed_native_no_jepa": "FLRM comprimido nativo sem JEPA",
    "flrm_compressed_native_no_jepa_k1": "FLRM nativo sem JEPA (K=1)",
    "flrm_compressed_native_no_jepa_k2": "FLRM nativo sem JEPA (K=2, N0)",
    "flrm_compressed_native_cached_no_jepa_k2": "FLRM nativo + ctx-cache sem JEPA (K=2, N1)",
    "flrm_compressed_grouped": "FLRM comprimido agrupado + JEPA",
    "flrm_compressed_grouped_no_jepa": "FLRM comprimido agrupado sem JEPA",
    "flrm_compressed_grouped_no_jepa_k1": "FLRM agrupado sem JEPA (K=1)",
    "flrm_compressed_grouped_no_jepa_k2": "FLRM agrupado sem JEPA (K=2, G0)",
    "flrm_compressed_grouped_cached_no_jepa_k2": "FLRM agrupado + ctx-cache sem JEPA (K=2, G1)",
    "flrm_fused_k2": "FLRM Simétrico + Fusão sem NCM (K=2)",
    "flrm_ncm_k2": "FLRM Simétrico + NCM-16 sem Fusão (K=2)",
    "flrm_vcycle_k2": "FLRM Ciclo-V + Fusão JIT (K=2)",
    "flrm_vcycle_ncm_k2": "FLRM Ciclo-V + Fusão JIT + NCM-16 (K=2)",
    "flrm_fused_ncm_k2": "FLRM Simétrico + Fusão + NCM-16 (K=2)",
    "flrm_inv_vcycle_ncm_k2": "FLRM Ciclo-V Invertido + Fusão + NCM-16 (K=2)",
    "flrm_no_jepa": "FLRM sem JEPA",
}


def number(value, digits=3):
    if value is None or not math.isfinite(value):
        return "n/d"
    return f"{value:,.{digits}f}".translate(str.maketrans({",": ".", ".": ","}))


def stage(index, total, title):
    print(f"\n{'='*78}\n[{index}/{total}] {title}\n{'='*78}", flush=True)


def legend():
    print("Como ler os resultados:", flush=True)
    print("  Loss LM de validação: menor = melhor previsão do próximo token.")
    print("  Velocidade (tokens ou alvos/s): maior = melhor.")
    print("  VRAM (memória da GPU): menor = menor consumo, para a mesma tarefa.")
    print("  Parâmetros: menos = modelo menor; qualidade precisa ser medida pela loss.")
    print("  Treino com JEPA inclui perdas auxiliares; compare qualidade pela validação LM.")
    print("  n/d = medição não disponível. Um ensaio não demonstra superioridade geral.", flush=True)


def progress(metrics, job, elapsed):
    step, total = metrics["step"], job["max_steps"]
    if not total or step < 0:
        return ""
    percent = min(100., 100*step/total)
    gpu = job.get("physical_gpu")
    device = f"GPU física {gpu}" if gpu is not None else "CPU"
    label = MODEL_NAMES.get(job["model"], job["model"])
    ckpt_interval = job.get("checkpoint_interval", 500)
    saved_tag = " | 💾 Checkpoint salvo" if step and (step % ckpt_interval == 0 or step >= total) else ""
    fields = [f"[{device} | {label} | seed {job['seed']}] {step}/{total} ({percent:.0f}%){saved_tag}",
              f"loss treino={number(metrics.get('training_loss'))} (menor = melhor neste objetivo)",
              f"treino={number(metrics.get('tokens_per_second_mean'), 0)} alvos/s (maior = melhor)"]
    if metrics.get("validation_loss_partial") is not None:
        fields.append(f"validação PARCIAL={number(metrics['validation_loss_partial'], 5)} (menor = melhor)")
    if step and elapsed is not None and step < total:
        fields.append(f"restante aprox. {number(elapsed*(total-step)/step/60, 1)} min")
    return " | ".join(fields)


def comparison_text(rows):
    lines = ["# Comparação da campanha", "",
             "Loss na validação completa: menor = melhor. Velocidade: maior = melhor.",
             "VRAM alocada no treino: menor = menor consumo de memória.", "",
             "| Modelo / seed | Loss completa (menor = melhor) | Treino alvos/s (maior = melhor) | VRAM MiB (menor consumo = melhor) | Geração tokens/s (maior = melhor) |",
             "|---|---:|---:|---:|---:|"]
    for r in rows:
        label = MODEL_NAMES.get(r["model"], r["model"])
        memory = r.get("peak_vram_allocated_bytes")
        memory = memory/2**20 if memory is not None else None
        lines.append(f"| {label} / {r['seed']} | {number(r.get('validation_loss'), 6)} | "
                     f"{number(r.get('tokens_per_second'), 0)} | {number(memory, 1)} | "
                     f"{number(r.get('generation_tokens_per_second'), 1)} |")
    lines += ["", "Treinos podem ocorrer em GPUs diferentes. Geração é medida depois, na mesma GPU, sem KV cache.",
              "Tamanhos e objetivos podem diferir; esta tabela não isola o efeito da arquitetura.",
              "A loss de validação mede previsão de tokens; não é uma prova de raciocínio ou inteligência geral."]
    return "\n".join(lines)+"\n"


def batch_comparison_text(rows):
    has_backends = any(r.get("backend") is not None for r in rows)
    has_repeats = any((r.get("repeat") or 1) > 1 for r in rows)
    lines = ["# Comparação do perfil de lotes (microbatch × acumulação)", "",
             "Throughput (alvos/s): maior = melhor (mais rápido). Tempo de passo: menor = melhor.",
             "VRAM alocada: menor = menor consumo de memória.", ""]
    if has_backends or has_repeats:
        lines += [
            "## Amostras individuais",
            "",
            "| Modelo | Backend | Rep. | Formato | Throughput alvos/s ↑ | Tempo/passo ms ↓ | Warmup 1º passo s | VRAM aloc. MiB ↓ | `cudaLaunchKernel` ↓ | GFLOPs/passo |",
            "|---|:---:|:---:|:---:|---:|---:|---:|---:|---:|---:|",
        ]
        for r in rows:
            label = MODEL_NAMES.get(r["model"], r["model"])
            backend = r.get("backend") or ("inductor" if r.get("compile") else "eager")
            rep = r.get("repeat", 1)
            fmt = f"{r['batch_size']}×{r['grad_accum_steps']}"
            toks = r.get("tokens_per_second")
            step_s = r.get("mean_step_seconds")
            ms = step_s * 1000 if step_s is not None else None
            w1 = r.get("first_warmup_step_seconds")
            vram = r.get("peak_vram_allocated_bytes")
            vram_mib = vram / (1024 * 1024) if vram is not None else None
            launches = r.get("cuda_launch_kernel_calls")
            flops = r.get("analytical_step_flops_estimate")
            gflops = flops / 1e9 if flops is not None else None
            lines.append(
                f"| {label} | `{backend}` | {rep} | {fmt} | {number(toks, 0)} | {number(ms, 2)} ms | "
                f"{number(w1, 2)} s | {number(vram_mib, 1)} MiB | {number(launches, 0)} | {number(gflops, 2)} |"
            )
        if has_repeats:
            lines += [
                "",
                "## Agregado por Modelo e Backend (Média e Desvio-Padrão entre Repetições)",
                "",
                "| Modelo | Backend | Formato | Throughput médio (DP) alvos/s ↑ | Tempo médio (DP) ms/passo ↓ | VRAM aloc. MiB ↓ | `cudaLaunchKernel` ↓ | GFLOPs/passo |",
                "|---|:---:|:---:|---:|---:|---:|---:|---:|",
            ]
            grouped = {}
            for r in rows:
                key = (r["model"], r.get("backend") or ("inductor" if r.get("compile") else "eager"), r["batch_size"], r["grad_accum_steps"])
                grouped.setdefault(key, []).append(r)
            for (model, backend, b, a), items in grouped.items():
                label = MODEL_NAMES.get(model, model)
                toks_list = [x["tokens_per_second"] for x in items if x.get("tokens_per_second") is not None]
                ms_list = [x["mean_step_seconds"] * 1000 for x in items if x.get("mean_step_seconds") is not None]
                tok_mean = statistics.mean(toks_list) if toks_list else None
                tok_sd = statistics.stdev(toks_list) if len(toks_list) >= 2 else 0.0
                ms_mean = statistics.mean(ms_list) if ms_list else None
                ms_sd = statistics.stdev(ms_list) if len(ms_list) >= 2 else 0.0
                vram = items[0].get("peak_vram_allocated_bytes")
                vram_mib = vram / (1024 * 1024) if vram is not None else None
                launches = items[0].get("cuda_launch_kernel_calls")
                flops = items[0].get("analytical_step_flops_estimate")
                gflops = flops / 1e9 if flops is not None else None
                lines.append(
                    f"| {label} | `{backend}` | {b}×{a} | {number(tok_mean, 0)} (±{number(tok_sd, 0)}) | "
                    f"{number(ms_mean, 2)} (±{number(ms_sd, 2)}) ms | {number(vram_mib, 1)} MiB | "
                    f"{number(launches, 0)} | {number(gflops, 2)} |"
                )
    else:
        lines += [
            "| Modelo | Formato (microbatch × acum) | Throughput alvos/s (maior = melhor) | Tempo médio / passo (menor = melhor) | VRAM alocada (menor = melhor) |",
            "|---|:---:|---:|---:|---:|",
        ]
        for r in rows:
            label = MODEL_NAMES.get(r["model"], r["model"])
            fmt = f"{r['batch_size']} × {r['grad_accum_steps']} ({r['batch_size']*r['grad_accum_steps']} seqs)"
            toks = r.get("tokens_per_second")
            step_s = r.get("mean_step_seconds")
            ms = step_s * 1000 if step_s is not None else None
            vram = r.get("peak_vram_allocated_bytes")
            vram_mib = vram / (1024 * 1024) if vram is not None else None
            lines.append(f"| {label} | {fmt} | {number(toks, 0)} | {number(ms, 1)} ms | {number(vram_mib, 1)} MiB |")
    lines += ["", "Medido em execuções de perfil isoladas na GPU 0 (warmup + amostras cronometradas).",
              "Todos os formatos mantêm o mesmo lote efetivo de sequências por atualização.",
              "A taxa de transferência mede a velocidade de execução; não mede a qualidade final da loss."]
    return "\n".join(lines) + "\n"


def review_text(rows):
    lines = ["# Checkpoints salvos — qualidade na validação completa", "",
             "Loss: menor = melhor previsão. Este modo não executa treino nem mede velocidade.", "",
             "| Modelo / seed | Passo salvo | Loss completa (menor = melhor) | Alvos avaliados | Origem |",
             "|---|---:|---:|---:|---|"]
    for r in rows:
        label = MODEL_NAMES.get(r["model"], r["model"])
        origin = "reutilizada e verificada" if r["reused"] else "avaliada agora"
        lines.append(f"| {label} / {r['seed']} | {r['step']} | {number(r['validation_loss'], 6)} | "
                     f"{number(r['validation_tokens'], 0)} | {origin} |")
    lines += ["", "Melhor checkpoint observado por modelo/seed (menor loss nesta validação):"]
    for model, seed in dict.fromkeys((r["model"], r["seed"]) for r in rows):
        best = min((r for r in rows if (r["model"], r["seed"]) == (model, seed)),
                   key=lambda r:r["validation_loss"])
        lines.append(f"- {MODEL_NAMES.get(model, model)} / seed {seed}: passo {best['step']}, "
                     f"loss {number(best['validation_loss'], 6)}.")
    lines += ["", "Escolha feita pela validação; não substitui avaliação em um conjunto de teste independente.",
              "Uma seed e modelos com tamanhos/objetivos diferentes não estabelecem superioridade geral."]
    return "\n".join(lines)+"\n"
