from concurrent.futures import ThreadPoolExecutor
import hashlib
import io
import json
import os
from pathlib import Path
import runpy
import threading
import time
from types import SimpleNamespace
import urllib.request
import zipfile

import pytest
from llm2.research import cycle4, cycle4_bootstrap
from llm2.research.cycle4_protocol import VARIANTS, SEEDS
from llm2.research.cycle4_report import analyze
from llm2.research.resource_telemetry import energy_result, EnergyMeter
from llm2.research.session import SessionBudget, SessionBudgetExceeded
from llm2.research.profiling import profile_step
from llm2.research.cycle4_protocol import configuration
from llm2.research.cycle4_evaluate import depth_probe
from llm2.trainer import Trainer
from test_cycle3 import explicit_data

ROOT=Path(__file__).resolve().parents[1]


def test_two_lanes_overlap_without_sharing_job_state():
    barrier=threading.Barrier(2);seen=[];lock=threading.Lock()
    def worker(slot):
        barrier.wait(timeout=5)
        with lock:seen.append((slot,threading.get_ident()))
    cycle4.run_lanes(["0","1"],worker,lambda:pytest.fail("unexpected cancellation"))
    assert {x[0] for x in seen}=={"0","1"} and len({x[1] for x in seen})==2


def test_concurrent_phase_publication_holds_manifest_lock():
    lock=threading.RLock();c=SimpleNamespace(lock=lock,manifest={})
    def save():
        assert lock._is_owned()
        json.dumps(c.manifest)
    c.save=save
    def worker(slot):
        for i in range(20):
            with cycle4.phase(c,"test",gpu_slot=slot,iteration=i):pass
    cycle4.run_lanes(["0","1"],worker,lambda:pytest.fail("unexpected cancellation"))
    assert len(c.manifest["phases"])==40
    assert all(r["wall_seconds"]>=0 for r in c.manifest["phases"])


def test_lane_error_cancels_other_lane():
    started=threading.Event();cancelled=threading.Event()
    def worker(slot):
        if slot=="0":
            assert started.wait(5);raise RuntimeError("injected")
        started.set();assert cancelled.wait(5)
    with pytest.raises(RuntimeError,match="injected"):
        cycle4.run_lanes(["0","1"],worker,cancelled.set)
    assert cancelled.is_set()


def test_admission_predicts_two_queues_and_refuses_deadline_before_training(tmp_path):
    jobs=[{"gpu_slot":"0","estimated_job_seconds":5000,"registered_parameters":2,"max_steps":6000},
          {"gpu_slot":"1","estimated_job_seconds":5000,"registered_parameters":2,"max_steps":6000}]
    assert cycle4.forecast_seconds(jobs,["0","1"])==5000
    args=SimpleNamespace(gpus=["0","1"],engineering_smoke=False,target_seconds=7200,session_seconds=10800)
    c=SimpleNamespace(manifest={"runs":jobs},budget=SessionBudget(10800,300,elapsed=5000),
        output=tmp_path,save=lambda:None)
    with pytest.raises(SessionBudgetExceeded):cycle4.admit(c,args)
    assert all(j["max_steps"]==6000 for j in jobs)


def test_isolated_profiles_cannot_run_with_active_worker():
    c=SimpleNamespace(lock=threading.RLock(),processes={object()})
    with pytest.raises(RuntimeError,match="isolated"):cycle4.require_idle(c)


def test_energy_is_null_or_explicitly_measured_estimated():
    result=energy_result(None,None,[],0,0)
    assert result["gpu_joules"] is None and result["method"]=="unavailable"
    assert EnergyMeter("cpu").energy_mj() is None
    result=energy_result(1000,2500,[],15,30)
    assert result["gpu_joules"]==1.5 and result["gpu_joules_per_training_target"]==.1
    assert energy_result(None,None,[(0,10),(1,20)],10,1)["method"].startswith("estimated")


def test_exploratory_report_primary_simplification_and_duplicate_checks():
    losses={"flrm_full":.98,"flrm_fine":1.,"flrm_4":.981,"flrm_16":.999,
            "lrm_cached":.97,"transformer_two":.96}
    rows=[{"variant":v,"seed":s,"validation_loss":losses[v]} for v in VARIANTS for s in SEEDS]
    profiles=[{"variant":v,"gpu_slot":g,"median_step_seconds":.1 if v=="flrm_full" else .08}
              for v in VARIANTS for g in ("0","1")]
    result=analyze(rows,[],profiles)
    assert result["quality_component_candidate"] and result["decision"]=="exploratory_only"
    assert result["simplifications"]["flrm_4"]["candidate"]
    assert not result["simplifications"]["flrm_fine"]["candidate"]
    assert not analyze(rows[:-1],[],profiles,smoke=True)["quality_component_candidate"]
    with pytest.raises(ValueError,match="duplicate"):analyze(rows+rows[:1],[],profiles)


def test_delivery_cells_inert_and_missing_receipt_blocks_before_network(monkeypatch,tmp_path):
    builder=runpy.run_path(str(ROOT/"scripts/build_cycle4_release.py"))
    def blocked(*a,**k):raise AssertionError("unexpected network")
    monkeypatch.setattr(urllib.request,"urlopen",blocked)
    for name,code in builder["cells"]("a"*40,active=False).items():exec(compile(code,name,"exec"),{})
    active=builder["cells"]("a"*40)["02_rodar_diagnostico_fractal.py"]
    with pytest.raises(ValueError,match="recibo"):exec(compile(active,"active","exec"),{})
    manifest=builder["build"](tmp_path,"a"*40)
    assert len(manifest["files"])==2 and manifest["session_seconds"]==10800
    with pytest.raises(FileExistsError):builder["build"](tmp_path,"a"*40)


def test_active_delivery_passes_cleanup_number_and_fresh_clock(monkeypatch):
    builder=runpy.run_path(str(ROOT/"scripts/build_cycle4_release.py"))
    payload=(ROOT/"src/llm2/research/cycle4_bootstrap.py").read_bytes()
    monkeypatch.setattr(urllib.request,"urlopen",lambda *a,**k:io.BytesIO(payload))
    captured={}
    def inject(code,namespace):
        namespace["bootstrap_git"]=lambda revision,**kwargs:captured.update(revision=revision,**kwargs)
    namespace={"LLM2_CLEANUP_INFO":{"next_run_number":25},"exec":inject,"maintenance_started_at":0}
    before=time.monotonic()
    exec(compile(builder["cells"]("a"*40)["02_rodar_diagnostico_fractal.py"],"active","exec"),namespace)
    assert captured["run_number"]==25 and captured["target_seconds"]==7200 and captured["session_seconds"]==10800
    assert before<=captured["started_at"]<=time.monotonic()


@pytest.mark.parametrize("kwargs",[{"run_number":None},{"run_number":22},{"session_seconds":14400},
                                  {"target_seconds":3600},{"started_at":float("nan")}])
def test_bad_bootstrap_settings_fail_before_network(monkeypatch,kwargs):
    monkeypatch.setattr(cycle4_bootstrap,"urlopen",lambda *a,**k:pytest.fail("unexpected network"))
    settings={"run_number":23};settings.update(kwargs)
    with pytest.raises(ValueError):cycle4_bootstrap.bootstrap_git("a"*40,run=True,**settings)


def test_real_two_process_training_validation_incremental_export(explicit_data,tmp_path,monkeypatch):
    original=cycle4.ordered_jobs
    monkeypatch.setattr(cycle4,"ordered_jobs",lambda *a,**k:original(*a,**k)[:2])
    def fast_pilot(c,args):
        for j in c.manifest["runs"]:j["estimated_job_seconds"]=.01
        c.manifest["pilot_completed_before_training"]=True
    monkeypatch.setattr(cycle4,"pilot",fast_pilot)
    monkeypatch.setattr(cycle4,"postprocess",lambda *a:None)
    args=cycle4.parser().parse_args(["--data",str(explicit_data),"--output",str(tmp_path/"campaign"),
        "--execute","--device","cpu","--engineering-smoke","--skip-tests"])
    result=cycle4.execute(args)
    assert result["status"]=="complete"
    with zipfile.ZipFile(result["results_zip"]) as archive:
        report=json.loads(archive.read("report/report.json"))
        assert len(report["rows"])==2 and len(report["curves"])==8 and report["engineering_smoke"]
        assert not any(n.endswith(".pt") for n in archive.namelist())
    for path in tmp_path.glob("campaign-after-*-checkpoints.zip"):
        with zipfile.ZipFile(path) as archive:
            assert sum(n.endswith("training_state.pt") for n in archive.namelist())==1
    with zipfile.ZipFile(result["checkpoints_zip"]) as archive:
        assert sum(n.endswith("training_state.pt") for n in archive.namelist())==2


def test_weights_profile_and_depth_probe_leave_scientific_checkpoint_unchanged(explicit_data,tmp_path):
    cfg=configuration("flrm_full",explicit_data,tmp_path/"run",211,device="cpu",smoke=True)
    with Trainer(cfg) as trainer:
        trainer.fit();checkpoint=trainer.last_checkpoint
    before={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in checkpoint.iterdir()}
    result=profile_step(cfg,tmp_path/"profile",warmup_steps=1,measure_steps=2,weights_checkpoint=checkpoint)
    assert result["weights_source"]["model_sha256"]==before["model.safetensors"]
    assert result["energy"]["gpu_joules"] is None
    assert result["step_metrics"]["gpu_time_seconds"] is None
    probe=depth_probe(checkpoint,tmp_path/"depth",device="cpu")
    assert [r["iterations_override"] for r in probe["rows"]]==[1,2,4,8]
    assert len({r["validation_tokens"] for r in probe["rows"]})==1
    assert not probe["test_evaluated"]
    assert before=={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in checkpoint.iterdir()}


@pytest.mark.parametrize("reason",["deadline","deadline_race","failure","disk"])
def test_early_stop_exports_partial_evidence_without_starting_training(explicit_data,tmp_path,monkeypatch,reason):
    monkeypatch.setattr(cycle4,"pilot",lambda *a:None)
    def stop(*args):
        if reason=="deadline":raise SessionBudgetExceeded("injected deadline")
        if reason=="deadline_race":
            budget=args[0].budget
            budget.clock=lambda:budget.started+budget.seconds
            raise RuntimeError("other lane observed cancellation first")
        if reason=="disk":raise OSError("injected insufficient disk")
        raise RuntimeError("injected failure")
    monkeypatch.setattr(cycle4,"admit",stop)
    monkeypatch.setattr(cycle4,"train_parallel",lambda *a:pytest.fail("must not train"))
    output=tmp_path/"partial"
    args=cycle4.parser().parse_args(["--data",str(explicit_data),"--output",str(output),
        "--execute","--device","cpu","--engineering-smoke","--skip-tests"])
    if reason.startswith("deadline"):assert cycle4.execute(args)["status"]=="partial_budget"
    else:
        with pytest.raises(RuntimeError,match="evidence preserved"):cycle4.execute(args)
    with zipfile.ZipFile(output.parent/(output.name+"-results.zip")) as archive:
        result=json.loads(archive.read("report/report.json"))
        assert result["rows"]==[] and result["decision"]=="exploratory_only"


def test_matrix_space_guard_preserves_reserve(tmp_path,monkeypatch):
    jobs=[{"gpu_slot":s,"estimated_job_seconds":1,"registered_parameters":2165772} for s in ("0","1")]
    args=SimpleNamespace(gpus=["0","1"],engineering_smoke=False,target_seconds=7200,session_seconds=10800)
    c=SimpleNamespace(manifest={"runs":jobs},budget=SessionBudget(10800,300),output=tmp_path,save=lambda:None)
    monkeypatch.setattr(cycle4.shutil,"disk_usage",lambda *a:SimpleNamespace(free=2*1024**3))
    with pytest.raises(OSError,match="2 GiB"):cycle4.admit(c,args)


def test_each_job_preserves_postprocessing_reserve():
    values=[]
    campaign=SimpleNamespace(budget=SimpleNamespace(require=values.append))
    cycle4.require_job_budget(campaign,SimpleNamespace(engineering_smoke=False),{"estimated_job_seconds":100})
    assert values==[1000]


@pytest.mark.skipif(os.name!="posix",reason="owned process groups require POSIX; checked in Kaggle preflight")
def test_deadline_stops_owned_process_group(tmp_path,monkeypatch):
    from llm2.research.session import SessionCampaign
    monkeypatch.setenv("LLM2_CYCLE4","1")
    c=SessionCampaign(ROOT,tmp_path/"session",device="cpu",gpus=[],budget=SessionBudget(3,.5))
    with pytest.raises(SessionBudgetExceeded):
        c.command(["-c","import subprocess,sys,time; subprocess.Popen([sys.executable,'-c','import time;time.sleep(30)']);time.sleep(30)"],"deadline.log")
    assert not c.processes and not c._group_killers
