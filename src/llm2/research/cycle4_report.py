"""Exploratory component report; never relabels the Cycle 3 confirmation."""
import csv
import json
from pathlib import Path
import statistics

from ..data import write_json
from .cycle3_report import effect
from .cycle4_protocol import VARIANTS, SEEDS, PROTOCOL_ID


def analyze(rows, curves, profiles, *, smoke=False):
    lookup = {(r["variant"],r["seed"]): r for r in rows}
    if len(lookup) != len(rows): raise ValueError("duplicate variant/seed")
    if any(v not in VARIANTS or s not in SEEDS for v,s in lookup): raise ValueError("unknown evidence")
    contrasts = {}
    definitions = {"primary_full_minus_fine": {"flrm_full": 1,"flrm_fine": -1},
        "scale4_without16": {"flrm_4": 1,"flrm_fine": -1},
        "scale16_without4": {"flrm_16": 1,"flrm_fine": -1},
        "scale4_with16": {"flrm_full": 1,"flrm_16": -1},
        "scale16_with4": {"flrm_full": 1,"flrm_4": -1},
        "interaction": {"flrm_full": 1,"flrm_4": -1,"flrm_16": -1,"flrm_fine": 1},
        "fine_minus_lrm": {"flrm_fine": 1,"lrm_cached": -1},
        "full_minus_transformer": {"flrm_full": 1,"transformer_two": -1}}
    for name, coefficients in definitions.items():
        seeds = [s for s in SEEDS if all((v,s) in lookup for v in coefficients)]
        values = [sum(weight*lookup[v,s]["validation_loss"] for v,weight in coefficients.items()) for s in seeds]
        contrasts[name] = {"seeds": seeds, **effect(values)}
    primary = contrasts["primary_full_minus_fine"]
    positive = (not smoke and primary["n"] == 3 and primary["mean"] <= -.005
                and primary["seeds_favoring_flrm"] == 3)
    simplifications = {}
    by_profile = {(r["variant"],r["gpu_slot"]): r for r in profiles}
    for variant in ("flrm_fine","flrm_4","flrm_16"):
        seeds = [s for s in SEEDS if (variant,s) in lookup and ("flrm_full",s) in lookup]
        losses = [lookup[variant,s]["validation_loss"]-lookup["flrm_full",s]["validation_loss"] for s in seeds]
        speeds = {}
        for gpu in ("0","1"):
            a,b = by_profile.get((variant,gpu)),by_profile.get(("flrm_full",gpu))
            speeds[gpu] = 1-a["median_step_seconds"]/b["median_step_seconds"] if a and b else None
        simplifications[variant] = {"seeds": seeds, "nll_differences": losses,
            "isolated_speed_fraction_by_gpu": speeds,
            "candidate": not smoke and len(seeds)==3 and all(x<=.002 for x in losses)
                and all(x is not None and x>=.10 for x in speeds.values())}
    return {"protocol_id": PROTOCOL_ID, "rows": rows, "curves": curves, "isolated_profiles": profiles,
        "contrasts": contrasts, "quality_component_candidate": positive,
        "simplifications": simplifications, "test_evaluated": False, "engineering_smoke": smoke,
        "decision": "exploratory_only", "complete_seeds": [s for s in SEEDS if all((v,s) in lookup for v in VARIANTS)],
        "limitations": "Equal targets, not matched compute; parameter counts differ. t intervals exploratory; secondary contrasts unadjusted. Concurrent train times are operational, not isolated benchmarks. Energy is GPU-only when available. No formal equivalence or general fractal advantage."}


def save(destination, result, *, plot=False):
    destination = Path(destination); destination.mkdir(parents=True, exist_ok=False)
    write_json(destination/"report.json", result)
    for name, records in (("runs.csv",result["rows"]),("curves.csv",result["curves"])):
        if records:
            with (destination/name).open("x",encoding="utf-8",newline="") as stream:
                writer=csv.DictWriter(stream,fieldnames=list(records[0])); writer.writeheader();writer.writerows(records)
    lines = ["# Ciclo 4 — diagnóstico exploratório","",
        "| Variante | Seed | NLL completa | Alvos | Treino concorrente, s |",
        "|---|---:|---:|---:|---:|"]
    for row in result["rows"]:
        lines.append(f"| {row['variant']} | {row['seed']} | {row['validation_loss']:.9f} | {row['training_targets']} | {row['train_wall_seconds']:.2f} |")
    lines += ["",json.dumps(result["contrasts"],ensure_ascii=False,indent=2),"",result["limitations"]]
    (destination/"report.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    if plot and result["curves"]:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axis = plt.subplots(figsize=(9,5))
        for variant in VARIANTS:
            for seed in SEEDS:
                points = [p for p in result["curves"] if p["variant"]==variant and p["seed"]==seed]
                if points: axis.plot([p["training_flops_estimated"] for p in points],
                    [p["validation_loss"] for p in points],label=f"{variant}/{seed}",alpha=.7)
        axis.set(xlabel="FLOPs estimados de treino",ylabel="NLL completa de validação")
        axis.grid(alpha=.2);axis.legend(fontsize=6);fig.tight_layout()
        fig.savefig(destination/"quality_by_cost.png",dpi=150);plt.close(fig)
