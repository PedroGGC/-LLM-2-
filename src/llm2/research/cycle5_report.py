"""Exploratory depth contrasts; no automatic promotion of recurrent mechanisms."""
from .cycle3_report import effect
from .cycle5_protocol import VARIANTS,SEEDS,PROTOCOL_ID


def analyze(rows,curves,profiles,*,smoke=False):
    lookup={(r["variant"],r["seed"]):r for r in rows}
    if len(lookup)!=len(rows):raise ValueError("duplicate variant/seed")
    if any(v not in VARIANTS or s not in SEEDS for v,s in lookup):raise ValueError("foreign evidence")
    definitions={"primary_k2_minus_k1":("fine_k2","fine_k1"),
        "secondary_k4_minus_k1":("fine_k4","fine_k1"),"k4_minus_k2":("fine_k4","fine_k2"),
        **{v+"_minus_transformer":(v,"transformer_two") for v in VARIANTS[:-1]}}
    contrasts={}
    for key,(a,b) in definitions.items():
        seeds=[s for s in SEEDS if (a,s) in lookup and (b,s) in lookup]
        value=effect([lookup[a,s]["validation_loss"]-lookup[b,s]["validation_loss"] for s in seeds])
        value["seeds_favoring_first"]=value.pop("seeds_favoring_flrm")
        contrasts[key]={"seeds":seeds,**value}
    primary=contrasts["primary_k2_minus_k1"]
    candidate=not smoke and primary["n"]==3 and primary["mean"]<=-.005 and primary["seeds_favoring_first"]==3
    return {"cycle":5,"protocol_id":PROTOCOL_ID,"rows":rows,"curves":curves,"isolated_profiles":profiles,
        "contrasts":contrasts,"recurrence_depth_candidate":candidate,"engineering_smoke":smoke,
        "test_evaluated":False,"decision":"exploratory_only",
        "complete_seeds":[s for s in SEEDS if all((v,s) in lookup for v in VARIANTS)],
        "limitations":"Matched estimated FLOPs, not equal tokens/time/energy. Three seeds and secondary unadjusted contrasts. K1/K2/K4 are trained separately; no adaptive-depth training. Positive signals need five new seeds."}
