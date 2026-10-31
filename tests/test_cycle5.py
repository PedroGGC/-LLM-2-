import copy
import json
from pathlib import Path
from types import SimpleNamespace
import zipfile
import pytest
import torch
from llm2.research import cycle4,cycle5
from llm2.research.cycle5_protocol import configuration,ordered_jobs,VARIANTS,SEEDS
from llm2.research.factory import build_backbone
from llm2.runtime import attention_context,capture_rng
from llm2.objectives import CausalLanguageModeling
from llm2.trainer import Trainer
from test_training import training_config,assert_nested_equal
from test_cycle3 import explicit_data


def test_matrix_fixed_compute_parameters_and_no_new_mechanism(tmp_path):
    jobs=ordered_jobs(tmp_path/"data",tmp_path/"out",["0","1"])
    assert len(jobs)==12 and {j["seed"] for j in jobs}==set(SEEDS)
    expected={"fine_k1":11372,"fine_k2":6000,"fine_k4":3084,"transformer_two":10496}
    for j in jobs:
        c=j["config_resolved"]
        assert j["max_steps"]==c["optimizer"]["schedule_steps"]==expected[j["variant"]]
        assert j["registered_parameters"]==j["parameters_formula"]
        assert j["registered_parameters"] in (2164994,2164992)
        assert 0<=j["relative_flop_budget_shortfall"]<.001
        assert c["data"]["batch_size"]*c["train"]["grad_accum_steps"]==32
        assert not c["research"]["jepa_weight"] and not c["research"]["scale_conditioning"]
        assert not c["research"]["mtp_weight"] and not c["research"]["ncm_slots"]
    for v in VARIANTS:assert len({j["gpu_slot"] for j in jobs if j["variant"]==v})==2


@pytest.mark.parametrize("variant",VARIANTS)
@attention_context(True)
def test_depth_causal_finite_memorization(tmp_path,variant):
    cfg=configuration(variant,tmp_path/"unused",tmp_path/"run",221,steps=8,device="cpu",smoke=True)
    cfg.model.dropout=0;model=build_backbone(cfg);ids=torch.arange(18).repeat(2,1)
    assert model.lm_head.weight is model.embedding.weight
    model.eval();output=model(ids[:,:-1]).logits
    changed=ids[:,:-1].clone();changed[:,9:]=(changed[:,9:]+1)%cfg.model.vocab_size
    torch.testing.assert_close(model(changed).logits[:,:9],output[:,:9],atol=0,rtol=0)
    batch={"input_ids":ids[:,:-1],"targets":ids[:,1:]};objective=CausalLanguageModeling()
    first=objective(model,batch).loss.item();optimizer=torch.optim.AdamW(model.parameters(),lr=.01)
    for _ in range(80):
        optimizer.zero_grad(set_to_none=True);loss=objective(model,batch).loss
        assert torch.isfinite(loss);loss.backward()
        assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
        optimizer.step()
    assert objective(model,batch).loss.item()<first/15


@pytest.mark.parametrize("variant",VARIANTS)
def test_depth_strict_resume_dropout_accumulation(training_config,tmp_path,variant):
    cfg=configuration(variant,training_config.data.path,tmp_path/"run",221,steps=8,device="cpu",smoke=True)
    cfg.model.dropout=.2
    def state(t):return copy.deepcopy((t.model.state_dict(),t.objective.state_dict(),t.optimizer.state_dict(),
        t.scheduler.state_dict(),t.sampler.state_dict(),capture_rng()))
    with Trainer(cfg) as t:
        t.fit(until_step=2);checkpoint=t.last_checkpoint;saved_counters=copy.deepcopy(t.counters)
        expected_loss=t.train_step()["training_loss"];expected=state(t)
    cfg.train.output_dir=str(tmp_path/"resume")
    with Trainer(cfg,resume=checkpoint) as t:
        assert t.counters==saved_counters
        assert t.train_step()["training_loss"]==expected_loss
        assert_nested_equal(expected,state(t))


@pytest.mark.parametrize("partial",[False,True])
def test_halfway_once_final_removes_only_temp(explicit_data,tmp_path,monkeypatch,partial):
    original=cycle5.STUDY.ordered_jobs
    study=copy.copy(cycle5.STUDY);study.ordered_jobs=lambda *a,**k:original(*a,**k)[:2]
    def pilot(c,args):
        for j in c.manifest["runs"]:j["estimated_job_seconds"]=.01
    def post(*a):
        if partial:raise cycle4.SessionBudgetExceeded("simulated postprocessing deadline")
    monkeypatch.setattr(cycle4,"pilot",pilot);monkeypatch.setattr(cycle4,"postprocess",post)
    args=cycle4.parser().parse_args(["--data",str(explicit_data),"--output",str(tmp_path/"campaign"),
        "--execute","--device","cpu","--engineering-smoke","--skip-tests","--export-policy","halfway-temporary"])
    result=cycle4.execute(args,study=study)
    assert result["status"]==("partial_budget" if partial else "complete")
    names=sorted(p.name for p in tmp_path.glob("*.zip"))
    expected=["campaign-checkpoints.zip","campaign-results.zip"]
    if partial:expected+=["campaign-midway-checkpoints.zip","campaign-midway-results.zip"]
    assert names==sorted(expected)
    with zipfile.ZipFile(result["results_zip"]) as archive:
        report=json.loads(archive.read("report/report.json"));assert report["cycle"]==5 and len(report["rows"])==2
    receipt=tmp_path/"campaign/export-retention.json"
    if partial:assert not receipt.exists()
    else:assert len(json.loads(receipt.read_text())["deleted"])==2


def test_halfway_means_six_completed_jobs_of_twelve(monkeypatch,tmp_path):
    jobs=[{"status":"pending"} for _ in range(12)]
    import threading
    campaign=SimpleNamespace(manifest={"runs":jobs},lock=threading.RLock())
    calls=[]
    def publish(c,suffix,**kwargs):calls.append((suffix,len(kwargs["checkpoint_jobs"])));return (None,None)
    monkeypatch.setattr(cycle4,"publish",publish);monkeypatch.setattr(cycle4,"verify_pair",lambda *a:None)
    for i in range(1,13):
        jobs[i-1]["status"]="complete"
        cycle4.maybe_publish_progress(campaign,SimpleNamespace(export_policy="halfway-temporary"),i)
    assert calls==[("midway",6)]
    assert campaign.manifest["export_progress"]["fraction"]==.5


def test_depth_report_is_exploratory_and_requires_three_fresh_pairs():
    from llm2.research.cycle5_report import analyze
    rows=[{"variant":v,"seed":s,"validation_loss":.98 if v=="fine_k2" else 1.} for v in VARIANTS for s in SEEDS]
    result=analyze(rows,[],[])
    assert result["recurrence_depth_candidate"] and result["decision"]=="exploratory_only"
    assert not analyze(rows,[],[],smoke=True)["recurrence_depth_candidate"]
    assert not analyze([r for r in rows if r["seed"]!=SEEDS[-1]],[],[])["recurrence_depth_candidate"]
    with pytest.raises(ValueError):analyze(rows+rows[:1],[],[])


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(),reason="CUDA unavailable")
@pytest.mark.parametrize("variant",VARIANTS)
def test_depth_cuda_fp32_inductor_checkpoint(training_config,tmp_path,variant):
    cfg=configuration(variant,training_config.data.path,tmp_path/"cuda",221,steps=8,device="cpu",smoke=True)
    cfg.train.device="cuda";cfg.train.compile=True;cfg.train.deterministic=False
    with Trainer(cfg) as t:
        assert torch.isfinite(torch.tensor(t.train_step()["training_loss"]))
        checkpoint=t.save()
    cfg.train.output_dir=str(tmp_path/"resume")
    with Trainer(cfg,resume=checkpoint) as t:assert torch.isfinite(torch.tensor(t.train_step()["training_loss"]))
