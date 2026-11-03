import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace
import zipfile

import pytest
import torch

from llm2.data import write_json
from llm2.research import state_readout as run
from llm2.research import state_readout_protocol as protocol
from llm2.research.state_readout_worker import retain_final,initial_digest
from llm2.runtime import environment


def fake_checkpoint(folder):
    folder.mkdir(parents=True)
    files={}
    for name in ("model.safetensors","training_state.pt"):
        payload=(folder.name+name).encode();(folder/name).write_bytes(payload)
        files[name]=hashlib.sha256(payload).hexdigest()
    write_json(folder/"metadata.json",{"schema_version":1,"config":{},"dataset_sha256":"data",
        "provenance":{},"architecture_version":2,"files":files})


def test_retention_verifies_final_and_keeps_only_last_complete(tmp_path):
    old=tmp_path/"checkpoints/step-00000002";final=tmp_path/"checkpoints/step-00000004"
    fake_checkpoint(old);fake_checkpoint(final)
    before={p.name:p.read_bytes() for p in final.iterdir()}
    rows=retain_final(tmp_path,final)
    assert len(rows)==1 and not old.exists() and all((final/n).read_bytes()==v for n,v in before.items())
    assert json.loads((tmp_path/"retired-checkpoints.json").read_text())[0]["checkpoint"]==old.name


def test_retention_bad_final_unknown_files_and_external_are_preserved(tmp_path):
    old=tmp_path/"checkpoints/step-00000002";final=tmp_path/"checkpoints/step-00000004"
    fake_checkpoint(old);fake_checkpoint(final)
    original=(final/"model.safetensors").read_bytes();(final/"model.safetensors").write_bytes(b"bad")
    with pytest.raises(ValueError,match="checksum"):retain_final(tmp_path,final)
    assert old.exists()
    (final/"model.safetensors").write_bytes(original);(old/"extra.txt").write_text("keep")
    with pytest.raises(ValueError,match="unknown"):retain_final(tmp_path,final)
    assert (old/"extra.txt").read_text()=="keep"
    with pytest.raises(ValueError,match="outside"):retain_final(tmp_path/"other",final)


def test_protocol_parameters_and_only_one_hierarchy_setting_differs():
    a=protocol.configuration("hierarchy_next","data","run")
    b=protocol.configuration("hierarchy_immediate","data","run")
    left,right=a.to_dict(),b.to_dict();left["research"].pop("state_readout");right["research"].pop("state_readout")
    assert left==right
    stats={v:protocol.statistics(protocol.configuration(v,"data","run")) for v in protocol.VARIANTS}
    assert stats["fine_only"]["parameters"]==1136652
    assert stats["fine_only"]["state_bytes_per_sequence"]==99864
    assert stats["hierarchy_next"]["parameters"]==stats["hierarchy_immediate"]["parameters"]==2174232
    assert stats["transformer_two"]["parameters"]==2164992
    with pytest.raises(ValueError):protocol.jobs("data","out",["0","1"],variants=["bad"])


def test_budget_admits_only_whole_pairs_and_preserves_horizon():
    jobs=protocol.jobs("data","out",["0","1"])
    profiles={j["variant"]:({"command_wall_seconds":200.,"median_step_seconds":.1,"full_validation_seconds":10.},
                            {"command_wall_seconds":20.,"median_step_seconds":.1,"full_validation_seconds":10.}) for j in jobs}
    by={j["variant"]:j for j in jobs}
    full,total=run.admission(protocol.PAIRS,by,profiles,{},10000.,1.2)
    assert sum(map(len,full))==5 and total>0
    first_pair=full[0];pair_cost=max(j["estimated_job_seconds"] for j in first_pair)+sum(j["estimated_post_seconds"] for j in first_pair)
    limited,_=run.admission(protocol.PAIRS,by,profiles,{},pair_cost+.01,1.2)
    assert [[j["variant"] for j in p] for p in limited]==[list(protocol.PAIRS[0])]
    assert all(j["max_steps"]==j["config_resolved"]["optimizer"]["schedule_steps"]==6000 for j in jobs)
    none,_=run.admission(protocol.PAIRS,by,profiles,{},1.,1.2);assert not none


def test_history_skips_unrelated_corruption_and_rejects_changed_config(tmp_path):
    output=tmp_path/"llm2-campaign-0030";output.mkdir()
    source={"file":"hash"};campaign=SimpleNamespace(output=output,manifest={"provenance":{"source_sha256":source}})
    data=SimpleNamespace(root=tmp_path/"data",fingerprint="data")
    cfg=protocol.configuration("fine_only",data.root,"old",smoke=True,device="cpu")
    job={"variant":"fine_only","status":"complete","config_resolved":cfg.to_dict(),"counters":{"step":4},
         "max_steps":4,"evaluations":[{"full":True}],"checkpoint_model_sha256":"hash"}
    value={"cycle":"state_readout_v1","protocol_id":protocol.PROTOCOL_ID,"engineering_smoke":True,
        "dataset_sha256":"data","provenance":{"source_sha256":source},"environment":environment(),"runs":[job]}
    prior=tmp_path/"llm2-campaign-0029";prior.mkdir();write_json(prior/"campaign.json",value)
    bad=tmp_path/"llm2-campaign-0013";bad.mkdir();(bad/"campaign.json").write_text("interrupted")
    assert len(run.native_history(campaign,data,smoke=True))==1
    value["runs"][0]["config_resolved"]["optimizer"]["lr"]*=2;write_json(prior/"campaign.json",value)
    with pytest.raises(ValueError,match="configuration differs"):run.native_history(campaign,data,smoke=True)


def test_initialization_digest_does_not_change_rng_or_weights():
    from llm2.research.factory import build_backbone
    cfg=protocol.configuration("hierarchy_immediate","data","out",smoke=True,device="cpu")
    model=build_backbone(cfg);rng=torch.get_rng_state().clone();before=copy.deepcopy(model.state_dict())
    first=initial_digest(model);assert first==initial_digest(model) and torch.equal(rng,torch.get_rng_state())
    assert all(torch.equal(v,before[k]) for k,v in model.state_dict().items())


def test_integration_five_trainings_and_native_continuation(tmp_path):
    from llm2.document_data import prepare_documents
    raw=tmp_path/"docs.jsonl"
    raw.write_text(''.join(json.dumps({"id":s,"text":s+" controlled abcdefghijklmnopqrstuvwxyz "*12,"split":s})+'\n' for s in ("train","val","test")),encoding="utf-8")
    data=tmp_path/"prepared";prepare_documents([raw],data,kind="byte")
    import os
    env={**os.environ,"PYTHONPATH":str(protocol.ROOT/"src"),"PYTHONIOENCODING":"utf-8","CUDA_VISIBLE_DEVICES":""}
    output=tmp_path/"llm2-campaign-0029"
    call=[sys.executable,"-m","llm2.research.state_readout","--output",str(output),"--data",str(data),"--device","cpu","--engineering-smoke"]
    result=subprocess.run(call,env=env,cwd=protocol.ROOT,capture_output=True,text=True,encoding="utf-8",timeout=240)
    assert result.returncode==0,result.stdout+result.stderr
    campaign=json.loads((output/"campaign.json").read_text(encoding="utf-8"))
    assert campaign["status"]=="complete" and len(campaign["runs"])==5 and not campaign["test_evaluated"]
    hierarchy=[j for j in campaign["runs"] if j["variant"].startswith("hierarchy_")]
    assert len({j["initial_parameters_sha256"] for j in hierarchy})==1
    for job in campaign["runs"]:
        assert job["counters"]["step"]==4 and len(job["evaluations"])==4
        assert all(sum(e["phase_targets"])==e["validation_tokens"] for e in job["evaluations"])
        folder=Path(job["directory"])/"checkpoints"
        assert [p.name for p in folder.glob("step-*")]==["step-00000004"]
        assert (Path(job["directory"])/"retired-checkpoints.json").is_file()
    verified=run.verify_pair(tmp_path/"llm2-campaign-0029-results.zip",tmp_path/"llm2-campaign-0029-checkpoints.zip",require_complete=True)
    assert len(verified["covered"])==5 and not list(tmp_path.glob("*-after-*.zip"))
    with zipfile.ZipFile(tmp_path/"llm2-campaign-0029-results.zip") as z:
        assert z.namelist().count("source.zip")==1
        assert not any(p.endswith((".safetensors",".pt",".bin")) for p in z.namelist())
    # Same protocol, same seed: reusing completed native records never trains duplicates.
    call[call.index("--output")+1]=str(tmp_path/"llm2-campaign-0030")
    again=subprocess.run(call,env=env,cwd=protocol.ROOT,capture_output=True,text=True,encoding="utf-8",timeout=60)
    assert again.returncode==0,again.stdout+again.stderr
    continuation=json.loads((tmp_path/"llm2-campaign-0030/campaign.json").read_text(encoding="utf-8"))
    assert continuation["status"]=="complete" and not continuation["runs"] and len(continuation["historical_runs"])==5


@pytest.mark.parametrize("failure",["worker","space"])
def test_failure_exports_partial_pair_preserving_source_data(tmp_path,monkeypatch,failure):
    from llm2.document_data import prepare_documents
    raw=tmp_path/"raw.jsonl"
    raw.write_text(''.join(json.dumps({"id":s,"text":s+' abcdefghijklmnopqrstuvwxyz '*8,"split":s})+'\n' for s in ('train','val','test')),encoding="utf-8")
    data=tmp_path/"prepared";prepare_documents([raw],data,kind="byte")
    before={p.name:p.read_bytes() for p in data.iterdir()}
    args=SimpleNamespace(output=str(tmp_path/"llm2-campaign-0029"),data=str(data),cache=None,download=False,
        session_seconds=7200,bootstrap_elapsed_seconds=0.,started=time.monotonic(),device="cpu",gpus=["0","1"],
        engineering_smoke=True,bootstrap_metadata=None,variants=list(protocol.VARIANTS))
    def fake_worker(campaign,args,cfg,name,mode,slot,**kwargs):
        if mode=="train":raise RuntimeError("injected training failure")
        return {"median_step_seconds":.01,"warmup_seconds":.02,"full_validation_seconds":.01}
    monkeypatch.setattr(run,"worker",fake_worker)
    if failure=="space":monkeypatch.setattr(run.shutil,"disk_usage",lambda *a:SimpleNamespace(free=0))
    with pytest.raises(RuntimeError,match="campaign failed"):run.execute(args)
    output=Path(args.output)
    result=run.verify_pair(output.with_name(output.name+'-results.zip'),output.with_name(output.name+'-checkpoints.zip'))
    assert result['campaign']['status']=='failed' and len(result['campaign']['pending_variants'])==5
    assert all(j['max_steps']==4 for j in result['campaign']['runs'])
    assert all((data/name).read_bytes()==raw for name,raw in before.items())
