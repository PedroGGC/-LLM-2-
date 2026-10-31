"""Attribution contrasts, independent of any candidate family winning."""
import math
from .state_protocol import VARIANTS,SEEDS,PROTOCOL_ID
from .cycle3_report import effect


def plot_curves(curves,path,*,engineering_smoke=False):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    figure,axes=plt.subplots(1,2,figsize=(11,4))
    colors=dict(zip(VARIANTS,("#486cb5","#d4842b","#39916d","#9b549a")))
    for variant in VARIANTS:
        for seed in SEEDS:
            # Partial-subset NLL is a different estimand: never connect it to full NLL.
            points=sorted((p for p in curves if p["variant"]==variant and p["seed"]==seed and p["full"]),key=lambda p:p["step"])
            if not points:continue
            for ax,key in zip(axes,("training_targets","train_wall_seconds")):
                ax.plot([p[key] for p in points],[p["validation_loss"] for p in points],
                        marker="o",alpha=.7,color=colors[variant],label=variant if seed==min(p["seed"] for p in curves if p["variant"]==variant) else None)
    axes[0].set_xlabel("Training targets (repetitions included)")
    axes[1].set_xlabel("Concurrent operational training seconds")
    for ax in axes:ax.set_ylabel("Full document-window byte NLL");ax.grid(alpha=.2)
    axes[0].legend(fontsize=7)
    title="Exploratory curves; equal targets, not equal compute; partial evaluations excluded"
    if engineering_smoke:title="ENGINEERING SMOKE (CPU; not scientific results)\n"+title
    figure.suptitle(title,fontsize=10)
    figure.tight_layout();figure.savefig(path,dpi=150);plt.close(figure)


def analyze(rows,profiles=(),*,smoke=False):
    lookup={(r["variant"],r["seed"]):r for r in rows}
    if len(lookup)!=len(rows) or any(v not in VARIANTS or s not in SEEDS for v,s in lookup):
        raise ValueError("duplicate/foreign evidence")
    if any(not math.isfinite(r["validation_loss"]) for r in rows):raise ValueError("nonfinite evidence")
    definitions={"hierarchy_minus_recurrent":("hierarchy_independent","recurrent"),
        "sharing_minus_independent":("hierarchy_shared","hierarchy_independent"),
        **{v+"_minus_transformer":(v,"transformer_two") for v in VARIANTS[:-1]}}
    contrasts={}
    for name,(a,b) in definitions.items():
        seeds=[s for s in SEEDS if (a,s) in lookup and (b,s) in lookup]
        result=effect([lookup[a,s]["validation_loss"]-lookup[b,s]["validation_loss"] for s in seeds])
        result["seeds_favoring_first"]=result.pop("seeds_favoring_flrm")
        result["quality_candidate"]=not smoke and result["n"]==3 and result["mean"]<=-.005 and result["seeds_favoring_first"]==3
        contrasts[name]={"seeds":seeds,**result}
        savings={}
        for slot in ("0","1"):
            pa=next((p for p in profiles if p.get("variant")==a and p.get("slot")==slot),None)
            pb=next((p for p in profiles if p.get("variant")==b and p.get("slot")==slot),None)
            savings[slot]=1-pa["median_step_seconds"]/pb["median_step_seconds"] if pa and pb else None
        contrasts[name]["isolated_time_saving_by_gpu"]=savings
        contrasts[name]["efficiency_candidate"]=(not smoke and len(seeds)==3
            and all(lookup[a,s]["validation_loss"]-lookup[b,s]["validation_loss"]<=.002 for s in seeds)
            and all(s is not None and s>=.1 for s in savings.values()))
    return {"protocol_id":PROTOCOL_ID,"rows":rows,"isolated_profiles":list(profiles),"contrasts":contrasts,
        "complete_seeds":[s for s in SEEDS if all((v,s) in lookup for v in VARIANTS)],
        "engineering_smoke":smoke,"test_evaluated":False,"decision":"exploratory_only",
        "limitations":"Common-shell rules, not original published models. Equal targets, not matched time/FLOPs. Sharing changes parameters. Three seeds; secondary contrasts unadjusted. No formal equivalence; confirm with five new seeds."}
