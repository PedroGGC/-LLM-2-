"""Frozen-weight interventions, executed as a file against the original science.

Input discovery is stdlib-only. No optimizer, torch.load, training or test-set
evaluation. The reference fixes source, configurations, environment and weights.
"""
import argparse
import csv
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import sys
import time
import traceback
import zipfile

VARIANTS = ("recurrent", "hierarchy_independent", "hierarchy_shared", "transformer_two")
REFERENCE = "docs/results/campaign_0027_diagnostic_reference.json"
REPRODUCTION_TOLERANCE = 5e-5


def sha(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def regular(path):
    path = Path(path).absolute()
    if any(p.is_symlink() for p in (path, *path.parents)) or not path.is_file():
        raise ValueError(f"Arquivo regular ausente ou com link: {path}")
    return path


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("x", encoding="utf-8", newline="\n") as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False); f.write("\n")
    os.replace(temporary, path)


def inspect_inputs(reference, campaign, roots):
    """Read only. Verify original metadata/weights, or exact final backup ZIP."""
    campaign = Path(campaign)
    archive = None
    tasks = []
    for item in reference["jobs"]:
        relative = Path(item["checkpoint"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("unsafe checkpoint reference")
        native = campaign / relative
        if (native / "model.safetensors").exists():
            for name, expected in (("metadata.json", item["metadata_sha256"]),
                                   ("model.safetensors", item["model_sha256"])):
                if sha(regular(native / name)) != expected:
                    raise ValueError(f"Checkpoint original divergiu: {native/name}")
            tasks.append({"variant":item["variant"], "checkpoint":str(native), "archive":None})
        else:
            if archive is None:
                candidates = []
                for root in roots:
                    root = Path(root)
                    if root.is_dir():
                        candidates.extend(root.rglob("llm2-campaign-0027-checkpoints.zip"))
                candidates = sorted(set(candidates), key=str)
                for path in candidates:
                    if sha(regular(path)) == reference["checkpoints_zip_sha256"]:
                        archive = path; break
                if archive is None:
                    raise FileNotFoundError("Faltam os pesos da0027. Preserve a pasta original ou anexe "
                        "llm2-campaign-0027-checkpoints.zip como Input; results.zip não contém pesos.")
                with zipfile.ZipFile(archive) as z:
                    if (len(z.namelist()) != len(set(z.namelist())) or len(z.namelist()) > 100
                            or sum(i.file_size for i in z.infolist()) > 256*1024**2 or z.testzip()):
                        raise ValueError("invalid checkpoint backup")
            with zipfile.ZipFile(archive) as z:
                for name, expected in (("metadata.json",item["metadata_sha256"]),
                                       ("model.safetensors",item["model_sha256"])):
                    if hashlib.sha256(z.read((relative/name).as_posix())).hexdigest() != expected:
                        raise ValueError("checkpoint ZIP payload mismatch")
            tasks.append({"variant":item["variant"], "checkpoint":None, "archive":str(archive)})
    return tasks


def restore_weights(task, item, output):
    if task["checkpoint"]:
        return Path(task["checkpoint"]), None
    folder = Path(output) / "restored-weights" / item["variant"]
    folder.mkdir(parents=True, exist_ok=False)
    with zipfile.ZipFile(regular(task["archive"])) as z:
        for name, expected in (("metadata.json",item["metadata_sha256"]),
                               ("model.safetensors",item["model_sha256"])):
            payload = z.read(item["checkpoint"] + "/" + name)
            if hashlib.sha256(payload).hexdigest() != expected:
                raise ValueError("backup changed after preflight")
            with (folder/name).open("xb") as f: f.write(payload)
    return folder, folder


def remove_restored(folder, output):
    """Unlink only the two owned files; never recursively delete originals."""
    folder,output=Path(folder),Path(output).resolve()
    if (folder.is_symlink() or folder.parent.is_symlink()
            or folder.resolve().parent!=output/"restored-weights"
            or set(p.name for p in folder.iterdir())!={"metadata.json","model.safetensors"}):
        raise ValueError("unsafe temporary weights cleanup")
    paths=[regular(folder/name) for name in ("metadata.json","model.safetensors")]
    for path in paths:path.unlink()
    folder.rmdir()


def verify_runtime(reference, *, smoke=False):
    from llm2.runtime import environment, source_provenance
    current = environment()
    actual = source_provenance()
    if actual["source_sha256"] != reference["source_sha256"]:
        raise ValueError("original scientific source differs")
    if actual["versioned_configs_sha256"] != reference["versioned_configs_sha256"]:
        raise ValueError("original scientific configurations differ")
    if any(current.get(k) != reference["environment"].get(k)
           for k in ("python", "packages", "cuda_version", "cudnn_version")):
        raise ValueError("checkpoint runtime versions differ; evaluation blocked")
    if not smoke and (current["device"] != "Tesla T4" or not current["cuda_devices"]):
        raise ValueError("diagnosis requires T4; CPU is engineering smoke only")
    return current


def components(model, ids):
    """Observe cells; original forward still controls all causal mixing and state."""
    import torch
    captured = []
    hooks = [cell.register_forward_hook(lambda module, inputs, result: captured.append(result[0]))
             for cell in model.cells]
    try:
        original = model(ids, return_logits=False).hidden_states
    finally:
        for hook in hooks: hook.remove()
    if len(captured) != 2:
        raise ValueError("expected exactly two temporal calls")
    fine = captured[0]
    if not model.hierarchy:
        return original, fine, captured[1] - fine
    coarse = captured[1]
    candidates = torch.cat((coarse.new_zeros(coarse.shape[0],1,coarse.shape[-1]),coarse),1)
    indices = torch.arange(ids.shape[1],device=ids.device)//model.factor
    broad = candidates[:,indices]
    rebuilt = model.norm(fine+broad)
    torch.testing.assert_close(rebuilt, original, rtol=2e-6, atol=2e-6)
    return original, fine, broad


def representations(model, ids):
    if not hasattr(model, "cells"):
        return {"original":model(ids,return_logits=False).hidden_states}, None
    original, fine, broad = components(model, ids)
    values = {"original":original, "fine_only":model.norm(fine)}
    if model.hierarchy:
        values.update({f"coarse_gain_{gain}":model.norm(fine+gain*broad) for gain in (.25,.5)})
    return values, (fine, broad)


def evaluate_interventions(model, batches, *, factor=4, deadline=math.inf, expected_batches=None,
                           progress=True, pilot_margin=1.35):
    import torch
    from torch.nn import functional as F
    from llm2.runtime import capture_rng, restore_rng, attention_context
    rng = capture_rng(); modes = [(m,m.training) for m in model.modules()]
    model.eval()
    totals = {}; moments = {"fine_square":0.,"broad_square":0.,"dot":0.,"elements":0}
    sequences = count = 0; began = time.perf_counter(); pilot = []
    try:
        with torch.no_grad(), attention_context(False):
            for index, batch in enumerate(batches):
                if time.monotonic() >= deadline: raise TimeoutError("diagnostic work deadline")
                before = time.perf_counter()
                ids, targets = batch["input_ids"], batch["targets"]
                values, pair = representations(model, ids)
                valid = targets != -100
                phases = torch.arange(ids.shape[1],device=ids.device) % factor
                # Four fixed quarter-window bins; no model-dependent selection.
                quarters = (torch.arange(ids.shape[1],device=ids.device)*4//ids.shape[1]).clamp_max(3)
                for name, hidden in values.items():
                    losses = F.cross_entropy(model.lm_head(hidden).float().flatten(0,1),
                        targets.flatten(),ignore_index=-100,reduction="none").reshape_as(targets)
                    if not torch.isfinite(losses).all(): raise FloatingPointError("nonfinite diagnostic loss")
                    item = totals.setdefault(name,{"nll_sum":0.,"targets":0,"phase_nll_sum":[0.]*factor,
                        "phase_targets":[0]*factor,"quarter_nll_sum":[0.]*4,"quarter_targets":[0]*4})
                    item["nll_sum"] += losses.double().sum().item(); item["targets"] += valid.sum().item()
                    for key, bins in (("phase",phases),("quarter",quarters)):
                        for position in range(factor if key=="phase" else 4):
                            mask = valid & (bins==position)[None,:]
                            item[key+"_nll_sum"][position] += losses[mask].double().sum().item()
                            item[key+"_targets"][position] += mask.sum().item()
                if pair is not None:
                    fine,broad=pair; mask=valid.unsqueeze(-1).expand_as(fine)
                    for key,value in (("fine_square",fine.square()),("broad_square",broad.square()),("dot",fine*broad)):
                        moments[key] += value[mask].double().sum().item()
                    moments["elements"] += mask.sum().item()
                sequences += ids.shape[0]; count += 1
                elapsed = time.perf_counter()-before; pilot.append(elapsed)
                if count == 2 and expected_batches:
                    forecast = pilot_margin*max(pilot)*(expected_batches-count)
                    print(f"Piloto: {count} batches; previsão restante com35%: {forecast:.1f}s.",flush=True)
                    if time.monotonic()+forecast >= deadline:
                        raise TimeoutError("full diagnostic coverage does not fit; no shortened complete result")
                if progress and (count==1 or count%16==0):
                    print(f"Avaliação {count}/{expected_batches or '?'} batches; {sequences} janelas; "
                          f"{time.perf_counter()-began:.1f}s.",flush=True)
    finally:
        for module,mode in modes:module.training=mode
        restore_rng(rng)
    if not count or expected_batches and count!=expected_batches:
        raise ValueError("incomplete validation coverage")
    for item in totals.values():
        item["validation_loss"] = item["nll_sum"]/item["targets"]
        item["bits_per_byte"] = item["validation_loss"]/math.log(2)
        for key in ("phase","quarter"):
            item[key+"_loss"] = [s/n if n else None for s,n in zip(item[key+"_nll_sum"],item[key+"_targets"])]
    moments.update(fine_rms=math.sqrt(moments["fine_square"]/moments["elements"]) if moments["elements"] else None,
        broad_rms=math.sqrt(moments["broad_square"]/moments["elements"]) if moments["elements"] else None,
        broad_to_fine_rms=math.sqrt(moments["broad_square"]/moments["fine_square"]) if moments["fine_square"] else None,
        cosine=moments["dot"]/math.sqrt(moments["fine_square"]*moments["broad_square"])
            if moments["fine_square"] and moments["broad_square"] else None)
    return {"conditions":totals,"moments":moments,"validation_batches":count,"validation_sequences":sequences,
            "wall_seconds":time.perf_counter()-began,"timing_scope":"concurrent eager diagnostics; not speed benchmark",
            "optimizer_steps":0,"test_evaluated":False}


def run_worker(args):
    import torch
    from safetensors.torch import load_model
    from llm2.config import ExperimentConfig
    from llm2.data import PreparedData
    from llm2.runtime import setup_runtime
    from llm2.research.factory import build_backbone
    from llm2.research.document_windows import validation
    reference=json.loads(Path(args.reference).read_text(encoding="utf-8"))
    verify_runtime(reference,smoke=args.engineering_smoke)
    item=next(j for j in reference["jobs"] if j["variant"]==args.variant)
    checkpoint=Path(args.checkpoint)
    if sha(regular(checkpoint/"metadata.json"))!=item["metadata_sha256"] or sha(regular(checkpoint/"model.safetensors"))!=item["model_sha256"]:
        raise ValueError("weights or metadata changed after preflight")
    metadata=json.loads((checkpoint/"metadata.json").read_text(encoding="utf-8"))
    if metadata["config"]!=item["config"] or metadata["provenance"]["source_sha256"]!=reference["source_sha256"]:
        raise ValueError("checkpoint config/source mismatch")
    data=PreparedData(args.data)
    if data.fingerprint!=reference["dataset_sha256"]:raise ValueError("original dataset fingerprint mismatch")
    cfg=ExperimentConfig.from_dict(item["config"])
    device=setup_runtime(cfg.train)
    model=build_backbone(cfg).to(device)
    load_model(model,str(checkpoint/"model.safetensors"),device=str(device))
    if sum(p.numel() for p in model.parameters())!=item["parameters"]:
        raise ValueError("unique parameter count differs")
    expected=item["validation_batches"]
    result=evaluate_interventions(model,validation(data,cfg.data.seq_len,cfg.data.eval_batch_size,device=device),
        deadline=args.deadline,expected_batches=expected)
    delta=result["conditions"]["original"]["validation_loss"]-item["validation_loss"]
    result.update(variant=args.variant,seed=cfg.train.seed,reference_validation_loss=item["validation_loss"],
        reproduction_delta=delta,reproduction_tolerance=REPRODUCTION_TOLERANCE,
        reproduced=abs(delta)<=REPRODUCTION_TOLERANCE,checkpoint_model_sha256=item["model_sha256"],
        peak_vram_bytes=torch.cuda.max_memory_allocated(device) if device.type=="cuda" else None,
        second_component_role="lagged coarse output" if getattr(model,"hierarchy",False) else
            "second full-rate cell correction" if hasattr(model,"cells") else None,
        note="post-training sensitivity; not a separately trained ablation or speed benchmark")
    write_json(args.result,result)
    if not result["reproduced"]:raise ValueError("original NLL was not reproduced; stop interpretation")
    if result["conditions"]["original"]["targets"]!=item["validation_targets"]:
        raise ValueError("validation target coverage mismatch")
    print(f"{args.variant}: NLL original {result['conditions']['original']['validation_loss']:.8f}; "
          f"delta0027={delta:+.2e}; pesos congelados.",flush=True)


def export_results(output):
    output=Path(output); final=output.parent/(output.name+"-results.zip")
    temporary=final.with_suffix(".zip.part")
    if final.exists() or temporary.exists():raise FileExistsError("diagnostic export exists")
    files=[p for p in output.rglob("*") if p.is_file() and "restored-weights" not in p.parts and "data" not in p.relative_to(output).parts]
    with zipfile.ZipFile(temporary,"x",compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for path in files:
            regular(path);z.write(path,path.relative_to(output).as_posix())
    with zipfile.ZipFile(temporary) as z:
        if z.testzip() or any(p.endswith((".pt",".safetensors",".bin",".parquet")) for p in z.namelist()):
            raise ValueError("invalid diagnostic export")
    temporary.rename(final)
    return {"path":str(final),"bytes":final.stat().st_size,"sha256":sha(final)}


def report_completed(campaign):
    rows=[]
    for job in sorted(campaign.manifest["runs"],key=lambda j:VARIANTS.index(j["variant"])):
        result=job["result"]; original=result["conditions"]["original"]["validation_loss"]
        for condition,value in result["conditions"].items():
            rows.append({"variant":job["variant"],"condition":condition,
                "validation_loss":value["validation_loss"],"delta_original":value["validation_loss"]-original,
                "bits_per_byte":value["bits_per_byte"],"validation_targets":value["targets"],
                "reproduced_original":result["reproduced"]})
    if rows:
        with (campaign.output/"conditions.csv").open("x",encoding="utf-8",newline="") as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator="\n")
            writer.writeheader();writer.writerows(rows)
    write_json(campaign.output/"report.json",{"rows":rows,"optimizer_steps":0,"test_evaluated":False,
        "interpretation":"Frozen-weight sensitivity, not separately trained ablation; one seed; development validation.",
        "phase_definition":"position modulo4 from each reset window; not document-global offset",
        "quarter_definition":"four equal bins of positions in each reset window",
        "timing_limitation":"Concurrent eager diagnostics; no inference speed claim or energy comparison.",
        "probes_predeclared":["original","fine_only","coarse_gain_0.25","coarse_gain_0.5"],
        "post_selection_warning":"Use new seeds for any subsequent trained model; probe selection is not confirmation."})
    for row in rows:
        print(f"{row['variant']}/{row['condition']}: {row['validation_loss']:.7f} "
              f"({row['delta_original']:+.7f} frente ao original)",flush=True)


def supervise(args):
    import torch
    from llm2.runtime import source_provenance, save_source_snapshot
    from llm2.research.session import SessionCampaign, SessionBudget
    from llm2.research.cycle3_short import heartbeat
    root=Path(args.worker_source).resolve()
    import llm2
    if Path(llm2.__file__).resolve().parents[2]!=root:raise ValueError("wrong scientific package imported")
    reference=json.loads(Path(args.reference).read_text(encoding="utf-8"))
    env=verify_runtime(reference,smoke=args.engineering_smoke)
    if (args.engineering_smoke != (env["device"]=="cpu") or len(set(args.gpus))!=2):
        raise ValueError("two T4 slots or explicit two-lane CPU smoke required")
    tasks=inspect_inputs(reference,args.original,[Path(args.original).parent,Path("/kaggle/input")])
    if shutil.disk_usage(Path(args.output).parent).free<2*1024**3+64*1024**2:
        raise OSError("insufficient disk including2GiB reserve")
    budget=SessionBudget(args.session_seconds,300,
        elapsed=args.bootstrap_elapsed_seconds+time.monotonic()-args.started)
    campaign=SessionCampaign(root,Path(args.output),device="cpu" if args.engineering_smoke else "cuda",gpus=args.gpus,budget=budget)
    campaign.manifest.update(kind="state_post_training_diagnostic",reference=reference,
        engineering_smoke=args.engineering_smoke,test_evaluated=False,optimizer_steps=0,
        operation=json.loads(Path(args.operator_metadata).read_text(encoding="utf-8")),conditions=11)
    restored=[];error=None
    try:
        save_source_snapshot(campaign.output/"scientific-source.zip",source_provenance())
        with zipfile.ZipFile(campaign.output/"diagnostic-source.zip","x",compression=zipfile.ZIP_DEFLATED) as z:
            op=Path(args.operator_root)
            for name in campaign.manifest["operation"]["files_sha256"]:
                z.write(op/name,name)
        data=Path(args.original)/"data"
        if not (data/"manifest.json").exists():
            data=campaign.output/"data"
            campaign.command(["-m","llm2.research.wikitext","--cache",args.cache,"--output",data,"--download"],"prepare-wikitext.log")
        from llm2.data import PreparedData
        if PreparedData(data).fingerprint!=reference["dataset_sha256"]:raise ValueError("prepared dataset differs")
        shutil.copyfile(data/"manifest.json",campaign.output/"dataset.manifest.json")
        for task in tasks:
            item=next(j for j in reference["jobs"] if j["variant"]==task["variant"])
            cp,temporary=restore_weights(task,item,campaign.output)
            task["checkpoint"]=str(cp)
            if temporary:restored.append(temporary)
        def evaluate_task(index):
            task=tasks[index];slot=args.gpus[index%2];variant=task["variant"]
            print(f"GPU{slot}: avaliando {variant}; pesos congelados.",flush=True)
            result=campaign.output/(variant+".json")
            command=[str(Path(__file__).resolve()),"--mode","worker","--reference",args.reference,
                "--variant",variant,"--checkpoint",task["checkpoint"],"--data",str(data),"--result",str(result),
                "--deadline",str(time.monotonic()+budget.work_remaining)]
            if args.engineering_smoke:command.append("--engineering-smoke")
            campaign.command(command,variant+".log",gpu=None if args.engineering_smoke else slot)
            value=json.loads(result.read_text(encoding="utf-8"))
            with campaign.lock:
                campaign.manifest["runs"].append({"variant":variant,"gpu_slot":slot,"status":"complete","result":value})
                campaign.save()
        with heartbeat(campaign,"diagnóstico sem treino"):
            for pair in ((0,3),(1,2)):
                with ThreadPoolExecutor(max_workers=2) as pool:
                    futures=[pool.submit(evaluate_task,i) for i in pair]
                    try:
                        for future in as_completed(futures):future.result()
                    except BaseException:
                        campaign.abort();raise
        campaign.manifest["status"]="complete"
    except Exception as exc:
        error=exc;campaign.manifest.update(status="failed",error=str(exc),traceback=traceback.format_exc())
    finally:
        campaign.abort()
        campaign.manifest["session_budget"]={**budget.snapshot(),
            "scope":"bootstrap/setup/tests/data + frozen-weight evaluation; export uses reserve"}
        campaign.save()
        report_completed(campaign)
        bundle=export_results(campaign.output)
        # Only our own restored copies, after a verified bundle; originals untouched.
        for folder in restored:
            remove_restored(folder,campaign.output)
        print(f"Status: {campaign.manifest['status']}; resultados: {bundle['path']}; {bundle['bytes']/1024**2:.2f}MiB",flush=True)
    if error:raise RuntimeError("diagnosis failed; complete results and errors exported") from error
    return campaign.manifest


def main(argv=None):
    started=time.monotonic()
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode",choices=("supervise","worker"),default="supervise")
    for name in ("reference","worker-source","operator-root","operator-metadata","original","output","cache","variant","checkpoint","data","result"):
        p.add_argument("--"+name)
    p.add_argument("--gpus",nargs=2,default=["0","1"])
    p.add_argument("--deadline",type=float,default=math.inf)
    p.add_argument("--session-seconds",type=float,default=7200.)
    p.add_argument("--bootstrap-elapsed-seconds",type=float,default=0.)
    p.add_argument("--engineering-smoke",action="store_true")
    args=p.parse_args(argv)
    args.started=started
    if not 300<args.session_seconds<=10800:raise ValueError("invalid diagnostic session limit")
    return run_worker(args) if args.mode=="worker" else supervise(args)


if __name__=="__main__":main()
