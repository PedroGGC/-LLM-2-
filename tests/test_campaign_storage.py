import copy
import hashlib
import json
from pathlib import Path
import zipfile
import pytest
from llm2.research import campaign_storage as storage


def write_pair(base,stem,*,completed=2,total=2,status="complete",step=4):
    jobs=[];weights={};results={"data/manifest.json":b"dataset","data/tokenizer.json":b"tokenizer"}
    for i in range(total):
        model=f"weights-{i}".encode();state=f"optimizer-{i}".encode()
        cfg={"seed":i};digest=hashlib.sha256(model).hexdigest()
        jobs.append({"id":f"j{i}","config_resolved":cfg,"max_steps":4,
                     "status":"complete" if i<completed else "pending","checkpoint_model_sha256":digest})
        results[f"runs/j{i}/run.json"]=json.dumps(cfg).encode()
        if i>=completed:continue
        folder=f"runs/j{i}/checkpoints/step-{step:08d}/"
        meta={"schema_version":1,"config":cfg,"dataset_sha256":"data", "provenance":{"source_sha256":{"src.py":"hash"}},
              "files":{"model.safetensors":digest,"training_state.pt":hashlib.sha256(state).hexdigest()}}
        weights.update({folder+"metadata.json":json.dumps(meta).encode(),folder+"model.safetensors":model,
                        folder+"training_state.pt":state})
    campaign={"cycle":5,"protocol_id":"protocol","dataset_sha256":"data",
              "provenance":{"source_sha256":{"src.py":"hash"}},"runs":jobs,"status":status}
    results["campaign.json"]=weights["campaign.json"]=json.dumps(campaign).encode()
    results["report/report.json"]=json.dumps({"progress":completed}).encode()
    paths=[]
    for kind,data in (("results",results),("checkpoints",weights)):
        path=base/(stem+"-"+kind+".zip")
        with zipfile.ZipFile(path,"w") as z:
            for name,value in data.items():z.writestr(name,value)
        paths.append(path)
    return tuple(paths)


def append(path,name,value):
    with zipfile.ZipFile(path,"a") as archive:archive.writestr(name,value)


def test_verified_redundants_removed_final_and_campaign_preserved(tmp_path):
    final=write_pair(tmp_path,"llm2-campaign-0023")
    old=write_pair(tmp_path,"llm2-campaign-0023-after-01",completed=1,status="training")
    campaign=tmp_path/"llm2-campaign-0023";campaign.mkdir();(campaign/"keep.pt").write_bytes(b"keep")
    expected=[storage.file_sha(p) for p in final]
    result=storage.clean_incrementals(23,*expected,base=tmp_path,guard=lambda:[],root_check=lambda p:Path(p).resolve())
    assert len(result["deleted"])==2 and all(not p.exists() for p in old)
    assert [storage.file_sha(p) for p in final]==expected
    assert (campaign/"keep.pt").read_bytes()==b"keep"
    assert storage.clean_incrementals(23,*expected,base=tmp_path,guard=lambda:[],root_check=lambda p:Path(p).resolve())["deleted"]==[]


@pytest.mark.parametrize("fault",["local_hash","corrupt","unique","earlier_checkpoint","worker","foreign","link"])
def test_unsafe_cleanup_keeps_everything(tmp_path,fault):
    final=write_pair(tmp_path,"llm2-campaign-0023")
    old=write_pair(tmp_path,"llm2-campaign-0023-after-01",completed=1,status="training",
                   step=2 if fault=="earlier_checkpoint" else 4)
    expected=[storage.file_sha(p) for p in final];guard=lambda:[]
    if fault=="local_hash":expected[0]="0"*64
    if fault=="corrupt":old[0].write_bytes(b"broken ZIP")
    if fault=="unique":append(old[0],"report/unique-weights.pt",b"unique scientific data")
    if fault=="worker":guard=lambda:[123]
    if fault=="foreign":
        with zipfile.ZipFile(old[0]) as archive:data={n:archive.read(n) for n in archive.namelist()}
        campaign=json.loads(data["campaign.json"]);campaign["protocol_id"]="foreign"
        data["campaign.json"]=json.dumps(campaign).encode()
        with zipfile.ZipFile(old[0],"w") as archive:
            for name,value in data.items():archive.writestr(name,value)
    if fault=="link":
        target=tmp_path/"external";target.write_bytes(b"keep");old[0].unlink()
        try:old[0].symlink_to(target)
        except OSError:pytest.skip("symlink unavailable")
    with pytest.raises((ValueError,RuntimeError,zipfile.BadZipFile,KeyError)):
        storage.clean_incrementals(23,*expected,base=tmp_path,guard=guard,root_check=lambda p:Path(p).resolve())
    assert all(p.exists() for p in [*final,*old])


def test_missing_final_job_blocks_cleanup(tmp_path):
    final=write_pair(tmp_path,"final",completed=1,status="complete")
    with pytest.raises(ValueError):storage.verify_pair(*final,require_complete=True)


def test_pending_worker_or_changed_final_hash_blocks_before_delete(tmp_path):
    final=write_pair(tmp_path,"final");old=write_pair(tmp_path,"old",completed=1,status="training")
    calls=[]
    def guard():calls.append(1);return [] if len(calls)==1 else [123]
    with pytest.raises(RuntimeError):storage.prune_archives(tmp_path,old,final_results=final[0],final_checkpoints=final[1],guard=guard)
    assert all(p.exists() for p in old)
    with pytest.raises(ValueError):storage.prune_archives(tmp_path,old,final_results=final[0],final_checkpoints=final[1],expected_hashes=("0"*64,"0"*64))
    assert all(p.exists() for p in old)


def test_final_cannot_be_a_cleanup_target(tmp_path):
    final=write_pair(tmp_path,"final")
    with pytest.raises(ValueError):storage.prune_archives(tmp_path,[final[0]],final_results=final[0],final_checkpoints=final[1])
    assert all(p.exists() for p in final)
