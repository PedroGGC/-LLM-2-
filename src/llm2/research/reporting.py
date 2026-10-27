"""Export observed series and explicit estimates; never synthesize missing scientific results."""

import csv
import json
from pathlib import Path

from ..data import write_json


def learning_flops(row):
    value = row.get("training_flops_estimate")
    return None if value is None else value + (row.get("curriculum_probe_flops_estimate") or 0)


def generate_report(runs, output, *, target_loss=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    output = Path(output)
    if output.exists():
        raise FileExistsError("report directory already exists")
    records, series = [], []
    for directory in map(Path, runs):
        metadata = json.loads((directory / "run.json").read_text(encoding="utf-8"))
        summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
        rows = [json.loads(line) for line in (directory / "metrics.jsonl").read_text(encoding="utf-8").splitlines()]
        checkpoint_name = Path(summary["checkpoint"]).name
        def artifact(name):
            candidate = directory / "evaluations" / checkpoint_name / name
            if not candidate.exists():
                candidate = directory / name
            value = json.loads(candidate.read_text(encoding="utf-8")) if candidate.exists() else {}
            if value.get("checkpoint_model_sha256"):
                checkpoint_meta = json.loads((directory / "checkpoints" / checkpoint_name / "metadata.json").read_text(encoding="utf-8"))
                if value["checkpoint_model_sha256"] != checkpoint_meta["files"]["model.safetensors"]:
                    raise ValueError(f"stale evaluation artifact: {candidate}")
            return value
        benchmark, quality = artifact("benchmark.json"), artifact("reasoning.json")
        full_validation = artifact("language.json")
        r = metadata["config"].get("research", {})
        first_target = next((learning_flops(row)
                             for row in rows if target_loss is not None and row.get("validation_loss", float("inf")) <= target_loss), None)
        record = {"run": directory.name, "path": str(directory.resolve()), "seed": metadata["config"]["train"]["seed"],
                  "architecture": r.get("architecture", "transformer"), "curriculum": r.get("curriculum", "off"),
                  "model_parameters_calculated": metadata["model_parameters_calculated"],
                  "total_parameters_calculated": metadata["total_parameters_calculated"],
                  "model_weight_bytes_calculated": metadata.get("model_weight_bytes_calculated", 4*metadata["model_parameters_calculated"]),
                  "dataset_sha256": metadata["dataset_sha256"], "config": metadata["config"],
                  "environment": metadata["environment"], "summary": summary, "benchmark": benchmark, "quality": quality,
                  "full_validation": full_validation,
                  "first_observed_target_loss_flops_estimate": first_target}
        records.append(record)
        for row in rows:
            series.append({"run": directory.name, "step": row["step"],
                           "learning_flops_estimate": learning_flops(row),
                           "training_loss": row.get("training_loss"), "validation_loss": row.get("validation_loss")})
    if not records:
        raise ValueError("at least one completed run required")
    output.mkdir(parents=True)
    controls = {"same_dataset": len({r["dataset_sha256"] for r in records}) == 1,
                "same_sequence_length": len({r["config"]["data"]["seq_len"] for r in records}) == 1,
                "equal_model_parameters": len({r["model_parameters_calculated"] for r in records}) == 1,
                "same_training_microbatch": len({r["config"]["data"]["batch_size"] for r in records}) == 1,
                "same_gradient_accumulation": len({r["config"]["train"]["grad_accum_steps"] for r in records}) == 1,
                "same_training_precision": len({r["config"]["train"]["precision"] for r in records}) == 1,
                "same_hardware": len({r["environment"]["device"] for r in records}) == 1,
                "same_software_packages": len({json.dumps(r["environment"]["packages"], sort_keys=True) for r in records}) == 1,
                "same_learning_flop_ceiling": len({r["config"].get("research", {}).get("max_train_flops", 0) for r in records}) == 1,
                "seeds": sorted({r["seed"] for r in records}),
                "target_loss": target_loss,
                "limitations": "Analytical FLOPs, sparse target crossings are upper bounds. Differing controls prevent a causal efficiency claim. Small smoke runs are not a benchmark campaign."}
    write_json(output / "report.json", {"runs": records, "comparison_controls": controls})
    with (output / "comparisons.csv").open("w", newline="", encoding="utf-8") as stream:
        fields = ["run", "seed", "architecture", "model_parameters", "total_parameters",
                  "validation_loss_full", "validation_tokens_full", "train_wall_seconds",
                  "tokens_per_second_mean", "peak_vram_allocated_bytes", "training_flops_estimate"]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for r in records:
            s, v = r["summary"], r["full_validation"]
            writer.writerow(dict(zip(fields, [r["run"], r["seed"], r["architecture"],
                r["model_parameters_calculated"], r["total_parameters_calculated"],
                v.get("validation_loss"), v.get("validation_tokens"), s.get("train_wall_seconds"),
                s.get("tokens_per_second"), s.get("peak_vram_allocated_bytes"), s.get("training_flops_estimate")])))
    with (output / "series.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["run", "step", "learning_flops_estimate", "training_loss", "validation_loss"])
        writer.writeheader()
        writer.writerows(series)
    def save_chart(name, xlabel, ylabel, draw):
        fig, axis = plt.subplots(figsize=(7, 4))
        draw(axis)
        axis.set(xlabel=xlabel, ylabel=ylabel)
        axis.grid(alpha=0.2)
        fig.tight_layout()
        fig.savefig(output / f"{name}.svg")
        plt.close(fig)
    def loss_chart(axis):
        for record in records:
            values = [row for row in series if row["run"] == record["run"] and row["validation_loss"] is not None and row["learning_flops_estimate"] is not None]
            axis.plot([r["learning_flops_estimate"] for r in values], [r["validation_loss"] for r in values], marker="o", label=record["run"])
        axis.legend(fontsize=7)
    def quality_chart(axis):
        found = False
        for record in records:
            q = record["quality"].get("groups", {}).get("unseen_test_and_families", {}).get("exact_match")
            if q is not None and learning_flops(record["summary"]) is not None:
                s = record["summary"]
                axis.scatter([learning_flops(s)], [q], label=record["run"])
                found = True
        if found:
            axis.legend(fontsize=7)
        else:
            axis.text(0.5, 0.5, "No generated-answer evaluation supplied", ha="center", transform=axis.transAxes)
    save_chart("loss_vs_flops", "Learning FLOPs (estimate, includes curriculum probes)", "Validation language loss (observed)", loss_chart)
    save_chart("quality_vs_flops", "Learning FLOPs (estimate)", "Generated-answer exact match (observed)", quality_chart)
    save_chart("parameters", "Run", "Unique parameters (calculated)",
               lambda a: a.bar([r["run"] for r in records], [r["total_parameters_calculated"] for r in records]))
    gpu_memory = [r for r in records if r["benchmark"].get("peak_vram_allocated_bytes") is not None]
    if gpu_memory:
        save_chart("memory", "Run", "Peak allocated VRAM bytes (observed PyTorch allocator)",
                   lambda a: a.bar([r["run"] for r in gpu_memory], [r["benchmark"]["peak_vram_allocated_bytes"] for r in gpu_memory]))
    else:
        save_chart("memory", "Run", "Model weight bytes (calculated; excludes activations)",
                   lambda a: a.bar([r["run"] for r in records], [r["model_weight_bytes_calculated"] for r in records]))
    measured = [r for r in records if r["benchmark"].get("tokens_per_second") is not None]
    save_chart("throughput", "Run", "Generation tokens/s (observed, stated warmup)",
               lambda a: a.bar([r["run"] for r in measured], [r["benchmark"]["tokens_per_second"] for r in measured])
               if measured else a.text(0.5, 0.5, "No inference benchmark supplied", ha="center", transform=a.transAxes))
    lines = ["# Research comparison", "", "Observed engineering results; FLOPs are estimates. No superiority claim.", "",
             "| Run | Seed | Model / total params | Validation loss | Learning FLOPs estimate |", "|---|---:|---:|---:|---:|"]
    for record in records:
        s = record["summary"]
        lines.append(f"| {record['run']} | {record['seed']} | {record['model_parameters_calculated']} / {record['total_parameters_calculated']} | {s.get('validation_loss')} | {learning_flops(s)} |")
    lines += ["", "## Controls and limitations", "", "```json", json.dumps(controls, indent=2), "```", "",
              "Full final LM evaluations, when supplied, are in comparisons.csv/report.json; the loss curves above use periodic evaluations and their declared scope.",
              "Exact configuration, software versions, budgets, benchmarks and predictions are preserved in report.json.",
              "![Loss](loss_vs_flops.svg)", "![Quality](quality_vs_flops.svg)", "![Memory](memory.svg)", "![Throughput](throughput.svg)", "![Parameters](parameters.svg)"]
    (output / "report.md").write_text("\n".join(lines), encoding="utf-8")
    return {"output": str(output.resolve()), "runs": len(records), "comparison_controls": controls}
