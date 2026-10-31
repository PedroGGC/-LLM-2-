"""Verified campaign bundle coverage and deletion of redundant archives; stdlib only."""
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import stat
import uuid
import zipfile


def file_sha(path):
    digest=hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda:stream.read(1024**2),b""):digest.update(block)
    return digest.hexdigest()


def member_hashes(archive):
    names=archive.namelist()
    if len(names)!=len(set(names)):raise ValueError("duplicate ZIP members")
    hashes={}
    for info in archive.infolist():
        name=info.filename;parts=PurePosixPath(name)
        if parts.is_absolute() or ".." in parts.parts or "\\" in name or stat.S_ISLNK(info.external_attr>>16):
            raise ValueError("unsafe ZIP member")
        digest=hashlib.sha256()
        with archive.open(info) as stream:
            for block in iter(lambda:stream.read(1024**2),b""):digest.update(block)
        hashes[name]=digest.hexdigest()
    return hashes


def identity(campaign):
    return (campaign["cycle"],campaign["protocol_id"],campaign.get("dataset_sha256"),
            campaign["provenance"]["source_sha256"])


def verify_pair(results,checkpoints,*,require_complete=False):
    with zipfile.ZipFile(results) as z,zipfile.ZipFile(checkpoints) as w:
        result_hashes=member_hashes(z);weight_hashes=member_hashes(w)
        campaign=json.loads(z.read("campaign.json"));other=json.loads(w.read("campaign.json"))
        if campaign!=other:raise ValueError("bundle campaign manifests differ")
        if require_complete and campaign["status"]!="complete":raise ValueError("campaign is not complete")
        jobs={job["id"]:job for job in campaign["runs"]};covered={}
        if require_complete and not jobs:raise ValueError("no completed scientific jobs")
        for name in weight_hashes:
            match=re.fullmatch(r"runs/([^/]+)/checkpoints/step-([0-9]+)/metadata\.json",name)
            if not match:continue
            job_id,step=match[1],int(match[2]);job=jobs[job_id]
            if job_id in covered:raise ValueError("multiple checkpoints per job in bundle")
            meta=json.loads(w.read(name));folder=name.removesuffix("metadata.json")
            if meta["schema_version"]!=1 or meta["config"]!=job["config_resolved"]:
                raise ValueError("checkpoint config/schema differs from campaign")
            if meta["dataset_sha256"]!=campaign["dataset_sha256"] or meta["provenance"]["source_sha256"]!=campaign["provenance"]["source_sha256"]:
                raise ValueError("checkpoint provenance differs from campaign")
            if not {"model.safetensors","training_state.pt"}<=meta["files"].keys():
                raise ValueError("checkpoint lacks complete training state")
            for member,digest in meta["files"].items():
                if member not in ("model.safetensors","training_state.pt","objective.safetensors") or weight_hashes.get(folder+member)!=digest:
                    raise ValueError("checkpoint payload checksum mismatch")
            if step<=0 or step>job["max_steps"]:raise ValueError("invalid checkpoint step")
            if job["status"]=="complete" and (step!=job["max_steps"] or meta["files"]["model.safetensors"]!=job["checkpoint_model_sha256"]):
                raise ValueError("complete run missing final weights")
            covered[job_id]=step
        completed={job_id for job_id,job in jobs.items() if job["status"]=="complete"}
        if not completed<=covered.keys():raise ValueError("bundle missing a completed job")
        if require_complete and (completed!=jobs.keys() or covered.keys()!=jobs.keys()):
            raise ValueError("final bundle does not cover all jobs")
    return {"campaign":campaign,"identity":identity(campaign),"covered":covered,
            "results_members":result_hashes,"checkpoint_members":weight_hashes,
            "results_sha256":file_sha(results),"checkpoints_sha256":file_sha(checkpoints)}


def verify_superseded(path,final,*,kind):
    with zipfile.ZipFile(path) as archive:
        hashes=member_hashes(archive);campaign=json.loads(archive.read("campaign.json"))
        if identity(campaign)!=final["identity"]:raise ValueError("foreign campaign archive")
        finals=final["results_members"] if kind=="results" else final["checkpoint_members"]
        for name,digest in hashes.items():
            if name in ("campaign.json","report/report.json","report/report.md","report/runs.csv",
                        "report/curves.csv","report/quality_by_cost.png"):continue
            if finals.get(name)!=digest:raise ValueError("archive contains unique content: "+name)
    return hashes


def fingerprint(path):
    info=path.lstat()
    if path.is_symlink() or not stat.S_ISREG(info.st_mode):raise ValueError("archive must be a regular file")
    return info.st_size,info.st_mtime_ns,info.st_ino


def prune_archives(base,paths,*,final_results,final_checkpoints,require_complete=False,guard=lambda:[],expected_hashes=None):
    base=Path(base).resolve();paths=[Path(p) for p in paths]
    if len(paths)!=len(set(paths)) or any(p in (final_results,final_checkpoints) for p in paths):
        raise ValueError("duplicate targets or attempted deletion of final archive")
    for path in [final_results,final_checkpoints,*paths]:
        if path.parent.resolve()!=base or path.is_symlink() or path.resolve().parent!=base:
            raise ValueError("archive path outside working")
    if guard():raise RuntimeError("workers active; archive cleanup blocked")
    final_before={p:fingerprint(p) for p in (final_results,final_checkpoints)}
    final=verify_pair(final_results,final_checkpoints,require_complete=require_complete)
    if expected_hashes is not None and expected_hashes!=(final["results_sha256"],final["checkpoints_sha256"]):
        raise ValueError("final archives differ from verified local copies")
    candidates=[]
    for path in paths:
        before=fingerprint(path)
        kind="checkpoints" if path.name.endswith("-checkpoints.zip") else "results"
        verify_superseded(path,final,kind=kind)
        if fingerprint(path)!=before:raise RuntimeError("archive changed during verification")
        candidates.append((path,before))
    if guard() or any(fingerprint(p)!=info for p,info in final_before.items()):
        raise RuntimeError("workers/final archives changed; nothing deleted")
    if any(fingerprint(path)!=info for path,info in candidates):raise RuntimeError("cleanup targets changed")
    deleted=[]
    for path,info in candidates:
        if path in final_before:raise ValueError("cannot delete final archive")
        path.unlink();deleted.append({"name":path.name,"bytes":info[0]})
    return {"deleted":deleted,"freed_bytes":sum(r["bytes"] for r in deleted),
            "results_sha256":final["results_sha256"],"checkpoints_sha256":final["checkpoints_sha256"]}


def clean_incrementals(number,results_sha256,checkpoints_sha256,*,base="/kaggle/working",guard=None,root_check=None):
    if guard is None or root_check is None:
        from .kaggle_reset import active_workers,root_directory
        guard=guard or active_workers;root_check=root_check or root_directory
    base=root_check(base)
    if type(number) is not int or number<1:raise ValueError("invalid campaign number")
    for digest in (results_sha256,checkpoints_sha256):
        if not re.fullmatch(r"[0-9a-fA-F]{64}",digest):raise ValueError("local SHA-256 required")
    stem=f"llm2-campaign-{number:04d}"
    results=base/(stem+"-results.zip");weights=base/(stem+"-checkpoints.zip")
    if guard():raise RuntimeError("workers active; cleanup blocked")
    if results.is_symlink() or weights.is_symlink():raise ValueError("final archive cannot be a link")
    if file_sha(results)!=results_sha256.lower() or file_sha(weights)!=checkpoints_sha256.lower():
        raise ValueError("remote finals differ from verified local copies")
    pattern=re.compile(re.escape(stem)+r"-after-[0-9]+-(?:results|checkpoints)\.zip")
    targets=sorted(p for p in base.iterdir() if pattern.fullmatch(p.name))
    print(f"Verificando {len(targets)} ZIPs redundantes contra os definitivos.",flush=True)
    receipt=prune_archives(base,targets,final_results=results,final_checkpoints=weights,
                           require_complete=True,guard=guard,expected_hashes=(results_sha256.lower(),checkpoints_sha256.lower()))
    receipt.update(campaign=number,next_run_number=number+1,
                   scope="only verified after-NN ZIPs; campaign directory, final bundles, data and code preserved")
    path=base/(stem+"-cleanup-"+uuid.uuid4().hex[:8]+".json")
    with path.open("x",encoding="utf-8") as stream:json.dump(receipt,stream,indent=2)
    print(json.dumps({"deleted_archives":len(targets),"freed_gib":receipt["freed_bytes"]/1024**3,
                      "receipt":path.name,"next_run_number":number+1},indent=2),flush=True)
    return receipt
