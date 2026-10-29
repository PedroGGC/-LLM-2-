"""Tables/curves from complete checkpoint evaluations; incomplete factorial cells stay explicit."""

import csv
import json
from pathlib import Path
import statistics

from ..data import write_json
from .cycle2_protocol import MODELS


def collect_rows(campaign):
    rows, curves = [], []
    for job in campaign.manifest["runs"]:
        if job["status"] != "complete":
            continue
        folder = Path(job["directory"])
        summary = json.loads((folder / "summary.json").read_text(encoding="utf-8"))
        metadata = json.loads((folder / "run.json").read_text(encoding="utf-8"))
        benchmark = json.loads((Path(job["evaluation"]).parent / "benchmark.json").read_text(encoding="utf-8"))
        rows.append({"model": job["model"], "seed": job["seed"], "validation_loss": job["full_validation_loss"],
            "validation_tokens": job["validation_tokens"], "training_targets": summary["tokens_processed"], "train_wall_seconds": summary["train_wall_seconds"],
            "training_flops_estimated": summary["training_flops_estimate"],
            "registered_parameters": metadata["model_parameters_calculated"],
            "parameters_in_tensors_with_gradients": summary["parameters_with_gradients"],
            "peak_vram_mib": summary["peak_vram_allocated_bytes"] / 1024**2 if summary["peak_vram_allocated_bytes"] is not None else None,
            "generation_tokens_per_second": benchmark["tokens_per_second"],
            "physical_gpu": job["physical_gpu"], "stop_reason": summary["stop_reason"]})
        timing_by_step = {}
        for line in (folder / "metrics.jsonl").read_text(encoding="utf-8").splitlines():
            value = json.loads(line)
            if "train_wall_seconds" in value:
                timing_by_step[value["step"]] = value["train_wall_seconds"]
        for evaluation in job["checkpoint_evaluations"]:
            curves.append({"model": job["model"], "seed": job["seed"], "step": evaluation["step"],
                           "validation_loss": evaluation["validation_loss"], "validation_tokens": evaluation["validation_tokens"],
                           "train_wall_seconds": timing_by_step.get(evaluation["step"]),
                           "scope": "all_full_windows"})
    return rows, curves


def write_report(campaign):
    destination = campaign.output / "report"
    destination.mkdir(exist_ok=True)
    rows, curves = collect_rows(campaign)
    for name, values in (("runs.csv", rows), ("curves.csv", curves)):
        if values:
            with (destination / name).open("x", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(values[0]))
                writer.writeheader(); writer.writerows(values)
    lookup = {(r["model"], r["seed"]): r for r in rows}
    factorial = []
    for seed in campaign.manifest["seeds"]:
        if all((model, seed) in lookup for model in MODELS):
            losses = [lookup[model, seed]["validation_loss"] for model in MODELS[2:]]
            # x1f1, x1f0, x0f1, x0f0. Negative effect means quality improved.
            a, b, c, d = losses
            factorial.append({"seed": seed, "exchange_effect_film_on": a-c, "exchange_effect_film_off": b-d,
                              "film_effect_exchange_on": a-b, "film_effect_exchange_off": c-d,
                              "interaction_difference_in_differences": (a-b)-(c-d)})
    effects = {}
    for key in (k for k in (factorial[0] if factorial else {}) if k != "seed"):
        values = [row[key] for row in factorial]
        mean = statistics.mean(values)
        sd = statistics.stdev(values) if len(values) >= 2 else None
        critical = {2: 12.706204736, 3: 4.302652730}.get(len(values))
        margin = critical*sd/len(values)**.5 if critical is not None else None
        effects[key] = {"mean": mean, "sd": sd, "n": len(values),
                        "ci95_t_exploratory": [mean-margin, mean+margin] if margin is not None else None}
    result = {"rows": rows, "curves": curves, "factorial_by_seed": factorial, "effects": effects,
        "complete_seed_blocks": len(factorial), "planned_seed_blocks": len(campaign.manifest["seeds"]),
        "equal_training_targets": bool(rows) and len({row["training_targets"] for row in rows}) == 1 and all(row["stop_reason"] == "max_steps" for row in rows),
        "cost_equalized": False, "confirmatory": False,
        "limitations": "Three-seed exploratory factorial, same reused corpus. Backward FLOPs are conservative for disconnected branches; elementwise/pooling/FiLM/mixing excluded. Active count covers tensors with gradients, not individual nonzero elements. No five-seed independent confirmation."}
    write_json(destination / "report.json", result)
    lines = ["# Ciclo 2 — comunicação × FiLM", "", f"Status: {campaign.manifest['status']}. Blocos completos: {len(factorial)}/{len(campaign.manifest['seeds'])}.", "",
             "Somente avaliações completas entram nesta tabela. Menor NLL é melhor; custo não foi igualado.", "",
             "| Modelo | Seed | NLL | Treino, s | Registrados / com gradiente |", "|---|---:|---:|---:|---:|"]
    for row in rows:
        lines.append(f"| {row['model']} | {row['seed']} | {row['validation_loss']:.6f} | {row['train_wall_seconds']:.2f} | {row['registered_parameters']} / {row['parameters_in_tensors_with_gradients']} |")
    lines += ["", "## Efeitos fatoriais", "", "Valores negativos indicam menor NLL ao ativar o componente. Apenas blocos completos são usados; intervalos exploratórios, sem correção de multiplicidade.", "", "```json", json.dumps(effects, indent=2), "```", "", result["limitations"]]
    (destination / "report.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    return result
