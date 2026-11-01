import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
from types import SimpleNamespace
import zipfile

import pytest
from test_training import training_config

from llm2.research import state_b_bootstrap as bootstrap
from llm2.research import state_b_operator as operator
from llm2.research.kaggle_light import remove_cache,tree_bytes

ROOT=Path(__file__).resolve().parents[1]
ENTRY=ROOT/"src/llm2/research/state_b_operator.py"
REPORT=ROOT/"docs/results/campaign_0025_platforms.json"


def fake_campaign(tmp_path):
    path=tmp_path/"llm2-campaign-9999"
    path.mkdir()
    return SimpleNamespace(output=path,gpus=["0","1"],lock=threading.RLock(),manifest={},save=lambda:None)


def cache_file(c,slot):
    path=c.output/"compiler-cache"/f"gpu-{slot}"/"generated.so"
    path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(b"cache")
    return path


def test_original_qualification_is_preserved():
    report=json.loads(REPORT.read_text(encoding="utf-8"))
    assert operator.digest(REPORT)==bootstrap.PLATFORM_SHA256==operator.PLATFORM_SHA256
    assert report["selected_rule"]=="mlstm" and report["qualification_version"]==2
    assert operator.SCIENTIFIC_REVISION==bootstrap.SCIENTIFIC_REVISION
    for name,digest in report["source_sha256"].items():
        payload=subprocess.check_output(["git","show",f"{bootstrap.SCIENTIFIC_REVISION}:{name}"],cwd=ROOT)
        assert hashlib.sha256(payload).hexdigest()==digest


def test_cache_retains_family_and_never_cleans_other_gpu(tmp_path):
    c=fake_campaign(tmp_path)
    manager=operator.SlotCaches(c,remove_cache,tree_bytes)
    manager.run("0","recurrent",lambda:cache_file(c,"0"))
    original=cache_file(c,"1")
    same=c.output/"compiler-cache/gpu-0/generated.so"
    assert manager.run("0","recurrent",lambda:same.read_bytes())==b"cache"
    manager.run("0","shared",lambda:None)
    assert not same.exists() and original.read_bytes()==b"cache"
    assert len(c.manifest["operator_cache_evictions"])==1
    assert c.manifest["operator_commands"][1]["reused_family"]


def test_cache_cannot_be_removed_while_own_slot_is_active(tmp_path):
    c=fake_campaign(tmp_path);manager=operator.SlotCaches(c,remove_cache,tree_bytes)
    entered,release,finished=threading.Event(),threading.Event(),threading.Event()
    def active():
        cache_file(c,"0");entered.set();assert release.wait(3)
    a=threading.Thread(target=lambda:manager.run("0","first",active))
    b=threading.Thread(target=lambda:(manager.run("0","second",lambda:None),finished.set()))
    a.start();assert entered.wait(2);b.start()
    assert not finished.wait(.1)
    assert (c.output/"compiler-cache/gpu-0/generated.so").exists()
    # Other lane is independently available.
    manager.run("1","other",lambda:cache_file(c,"1"))
    release.set();a.join(3);b.join(3)
    assert finished.is_set() and not manager.active
    assert (c.output/"compiler-cache/gpu-1/generated.so").exists()


def test_cache_exception_releases_slot_without_deleting_science(tmp_path):
    c=fake_campaign(tmp_path);manager=operator.SlotCaches(c,remove_cache,tree_bytes)
    science=c.output/"weights.safetensors";science.write_bytes(b"keep")
    def failure():
        cache_file(c,"0");raise TimeoutError("worker ended")
    with pytest.raises(TimeoutError):manager.run("0","first",failure)
    assert not manager.active
    manager.run("0","second",lambda:None)
    assert science.read_bytes()==b"keep"


def test_post_reserve_includes_cold_compile_on_all_eight_profiles():
    profiles={i:{"warmup_seconds":400.,"median_step_seconds":.1,"full_validation_seconds":2.} for i in range(8)}
    assert operator.profile_reserve(profiles)>8*400
    assert operator.profile_reserve(profiles,smoke=True)==0


def test_bootstrap_inactive_or_invalid_never_uses_network(monkeypatch):
    monkeypatch.setattr(bootstrap,"urlopen",lambda *a,**k:pytest.fail("unexpected network"))
    assert bootstrap.bootstrap_git("anything")=={"status":"not_started"}
    with pytest.raises(ValueError):bootstrap.bootstrap_git("bad",run=True)


def test_bootstrap_rejects_changed_helper(monkeypatch):
    monkeypatch.setattr(bootstrap,"urlopen",lambda *a,**k:io.BytesIO(b"bad helper"))
    with pytest.raises(ValueError,match="checkout helper"):
        bootstrap.bootstrap_git("a"*40,run=True)


@pytest.mark.parametrize("preflight_fails",[False,True])
def test_bootstrap_dispatches_file_operator_with_qualified_pythonpath(tmp_path,monkeypatch,preflight_fails):
    import importlib.metadata
    science=tmp_path/"scientific-source";science.mkdir()
    op_parent=tmp_path/"op-parent";op_parent.mkdir()
    worker_parent=tmp_path/"worker-parent";worker_parent.mkdir()
    qualified=json.loads(REPORT.read_text(encoding="utf-8"))
    op_files={name:operator.digest(ROOT/name) for name in (
        "src/llm2/research/state_b_operator.py","src/llm2/research/state_b_bootstrap.py","src/llm2/research/wikitext_cache.py","docs/results/campaign_0025_platforms.json")}
    op_meta={"source_git_revision":"a"*40,"source_files_sha256":op_files}
    sci_meta={"source_git_revision":bootstrap.SCIENTIFIC_REVISION,"source_files_sha256":qualified["source_sha256"]}
    output=tmp_path/"llm2-campaign-0026"
    monkeypatch.setattr(bootstrap.runpy,"run_path",lambda *a:{"numbered_output":lambda *a,**k:output})
    calls=[]
    def command(args,**kwargs):
        calls.append((args,kwargs))
        if "--report" in args:
            if preflight_fails:raise OSError("raw preflight rejected")
            Path(args[args.index("--report")+1]).write_text(json.dumps({"status":"verified","selected_cache":str(tmp_path/"llm2-wikitext-verified-cache")}))
    session=SimpleNamespace(remaining=lambda **k:1000.,elapsed=2.)
    arguments=(ROOT,science,op_parent,worker_parent,op_meta,sci_meta,
        {"bounded_command":command},session,(tmp_path,["0","1"],importlib.metadata.version("torch")),26,(241,),None,7200,10800)
    if preflight_fails:
        with pytest.raises(OSError,match="preflight rejected"):bootstrap._run(*arguments)
        assert len(calls)==1 and calls[0][0][1].endswith("wikitext_cache.py")
        return
    result=bootstrap._run(*arguments)
    args,kwargs=calls[-1]
    assert args[1]==str(ENTRY)
    assert kwargs["env"]["PYTHONPATH"]==str(science/"src")
    assert args[args.index("--phase")+1]=="experiment"
    assert args[args.index("--seeds")+1]=="241" and "--engineering-smoke" not in args
    assert result["scientific_revision"]==bootstrap.SCIENTIFIC_REVISION
    assert calls[0][0][1].endswith("wikitext_cache.py")
    assert "--no-cache-dir" in calls[1][0]
    assert args[args.index("--cache")+1]==str(tmp_path/"llm2-wikitext-verified-cache")


def test_frozen_tuning_mismatch_is_not_accepted():
    fake=SimpleNamespace(PROTOCOL_ID="protocol",VARIANTS=("a","b"))
    c=SimpleNamespace(manifest={"provenance":{"source_sha256":{"file":"hash"}}})
    data=SimpleNamespace(fingerprint="data")
    value={"protocol_id":"protocol","selected_rule":"mlstm","dataset_sha256":"data",
           "provenance":{"source_sha256":{"file":"hash"}},"learning_rates":{"a":.0002,"b":.0004}}
    assert operator.valid_frozen_tuning(fake,value,c,data,"mlstm")==value["learning_rates"]
    value["provenance"]["source_sha256"]={"other":"code"}
    with pytest.raises(ValueError):operator.valid_frozen_tuning(fake,value,c,data,"mlstm")


def test_wrong_scientific_package_is_rejected(tmp_path):
    with pytest.raises(ValueError,match="wrong scientific package"):
        operator.run(tmp_path,{},ROOT,[])


def test_real_stage_b_uses_pinned_scientific_worker(training_config,tmp_path):
    """Real subprocesses, fixed old science, new operation; not CUDA evidence."""
    science=tmp_path/"qualified-source";science.mkdir()
    payload=subprocess.check_output(["git","archive","--format=zip",bootstrap.SCIENTIFIC_REVISION],cwd=ROOT)
    with zipfile.ZipFile(io.BytesIO(payload)) as z:z.extractall(science)
    names=("src/llm2/research/state_b_operator.py","src/llm2/research/state_b_bootstrap.py","src/llm2/research/wikitext_cache.py","docs/results/campaign_0025_platforms.json")
    meta={"revision":"engineering-uncommitted","scientific_revision":bootstrap.SCIENTIFIC_REVISION,
          "files_sha256":{n:operator.digest(ROOT/n) for n in names},"scope":"CPU engineering fixture"}
    meta_path=tmp_path/"operator.json";meta_path.write_text(json.dumps(meta))
    output=tmp_path/"llm2-campaign-9999"
    env=os.environ.copy();env.update(PYTHONPATH=str(science/"src"),PYTHONUNBUFFERED="1",PYTHONIOENCODING="utf-8")
    args=[sys.executable,str(ENTRY),"--worker-source",str(science),"--operator-root",str(ROOT),
          "--operator-metadata",str(meta_path),"--execute","--phase","experiment","--device","cpu",
          "--engineering-smoke","--data",training_config.data.path,"--output",str(output),"--seeds","241"]
    result=subprocess.run(args,cwd=science,env=env,capture_output=True,text=True,encoding="utf-8",errors="replace",timeout=600)
    assert result.returncode==0,result.stdout[-12000:]+result.stderr[-12000:]
    c=json.loads((output/"campaign.json").read_text())
    assert c["status"]=="complete" and c["selected_rule"]=="mlstm"
    assert len(c["runs"])==4 and all(j["status"]=="complete" for j in c["runs"])
    assert len(c["tuning"])==8 and len(c["isolated_profiles"])==8
    assert not c["test_evaluated"] and c["engineering_smoke"]
    assert c["provenance"]["source_sha256"]==json.loads(REPORT.read_text())["source_sha256"]
    for job in c["runs"]:
        meta=json.loads((Path(job["checkpoint"])/"metadata.json").read_text())
        assert meta["provenance"]["source_sha256"]==c["provenance"]["source_sha256"]
        assert job["config_resolved"]["research"]["state_rule"]=="mlstm"
    assert (output/"report/operator-source.zip").exists()
    assert (output/"report/quality_by_cost.png").exists()
    assert (tmp_path/"llm2-campaign-9999-results.zip").exists()
    assert (tmp_path/"llm2-campaign-9999-checkpoints.zip").exists()
    assert not list(tmp_path.glob("*-midway-*.zip"))
    report=json.loads((output/"report/report.json").read_text())
    assert report["operator"]["scientific_revision"]==bootstrap.SCIENTIFIC_REVISION
