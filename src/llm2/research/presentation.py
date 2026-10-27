"""Portuguese console views; measured values remain in the original JSON logs."""

import math


MODEL_NAMES = {
    "baseline": "Transformer", "baseline_small": "Transformer pequeno",
    "baseline_jepa": "Transformer + JEPA", "lrm": "LRM", "lrm_no_jepa": "LRM sem JEPA",
    "flrm_reference": "FLRM sequencial", "flrm_batched": "FLRM agrupado",
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
    fields = [f"[{device} | {label} | seed {job['seed']}] {step}/{total} ({percent:.0f}%)",
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
