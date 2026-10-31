import copy
import json
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from llm2.data import PreparedData
from llm2.research.document_windows import DocumentBatchStream,sampler_factory,validation,tuning_dataset
from llm2.research.state_protocol import RULES,VARIANTS,configuration,ordered_jobs,validate_parameter_budgets
from llm2.research.state_platforms import select,qualify
from llm2.research.state_report import analyze
from llm2.research.state_study import schedule_batches
from llm2.research.state_worker import DocumentTrainer
from test_training import training_config,assert_nested_equal


def test_document_sampler_boundaries_resume_and_validation():
    tokens=np.array(([1]*19+[256])*3+([2]*24+[256])*3,dtype=np.uint32)
    stream=DocumentBatchStream(tokens,256,8,4,17)
    for _ in range(10):assert not (stream.next()["input_ids"]==256).any()
    state=stream.state_dict();expected=stream.next();stream.load_state_dict(state)
    assert_nested_equal(expected,stream.next())
    data=SimpleNamespace(arrays={"val":tokens},tokenizer=SimpleNamespace(eos_id=256))
    for batch in validation(data,8,2):
        assert not (batch["input_ids"]==256).any()
        assert not (batch["targets"]==256).any()
    with pytest.raises(ValueError):DocumentBatchStream(tokens,256,1000,2,1)


def test_tuning_uses_training_documents_only(training_config,tmp_path):
    original=PreparedData(training_config.data.path)
    path=tuning_dataset(original.root,tmp_path/"tuning")
    derived=PreparedData(path)
    assigned=set().union(*(set(s["document_hashes"]) for s in derived.manifest["splits"].values()))
    assert assigned==set(original.manifest["splits"]["train"]["document_hashes"])
    assert not assigned.intersection(original.manifest["splits"]["val"]["document_hashes"])
    with pytest.raises(FileExistsError):tuning_dataset(original.root,tmp_path/"tuning")


def test_platform_selection_requires_two_real_t4s_and_correctness():
    rows=[]
    for rule,speed,state in (("gated_delta",1.,200),("rwkv7",1.04,100),("mlstm",2.,50)):
        for slot in ("0","1"):
            rows.append({"rule":rule,"slot":slot,"status":"passed","correctness":{"passed":True},
                "gpu_name":"Tesla T4","median_step_seconds":speed,"peak_vram_bytes":1024,
                "state_bytes_per_sequence":state})
    assert select(rows)=="rwkv7"  # within 5%, smaller state
    assert select(rows,engineering_smoke=True) is None
    assert select(rows[:1]) is None
    broken=copy.deepcopy(rows)
    for row in broken:row["correctness"]["passed"]=False
    assert select(broken) is None


@pytest.mark.parametrize("rule",RULES)
def test_protocol_parameter_budgets_and_complete_matrix(rule):
    stats=validate_parameter_budgets(rule)
    assert stats["hierarchy_shared"]["parameters"]<stats["hierarchy_independent"]["parameters"]
    jobs=ordered_jobs(rule,"data","output",["0","1"])
    assert len(jobs)==12
    assert {j["max_steps"] for j in jobs}=={6000}
    assert all(j["config_resolved"]["data"]["seq_len"]==1024 for j in jobs)
    admitted,pending=schedule_batches([dict(j,estimated_job_seconds=100.) for j in jobs],["0","1"],
        remaining_seconds=1500.,post_seconds=1200.)
    assert len(admitted)==4 and len(pending)==8
    assert {j["seed"] for j in admitted}=={241}


def test_partial_evidence_never_promotes_fractal():
    rows=[{"variant":v,"seed":241,"validation_loss":1. if v=="hierarchy_shared" else 2.} for v in VARIANTS]
    result=analyze(rows)
    assert not any(c["quality_candidate"] for c in result["contrasts"].values())
    with pytest.raises(ValueError):analyze(rows+[rows[0]])


def test_document_worker_exact_resume(training_config,tmp_path):
    cfg=training_config
    from llm2.config import ResearchConfig
    cfg.research=ResearchConfig(architecture="state_hierarchy",scale_conditioning=False,state_chunk_size=4)
    with DocumentTrainer(cfg) as trainer:
        trainer.fit(until_step=2);checkpoint=trainer.last_checkpoint
        metrics=trainer.train_step()
        expected=copy.deepcopy((trainer.model.state_dict(),trainer.optimizer.state_dict(),trainer.sampler.state_dict()))
    cfg.train.output_dir=str(tmp_path/"document-resume")
    with DocumentTrainer(cfg,resume=checkpoint) as trainer:
        actual=trainer.train_step()
        assert_nested_equal(expected,(trainer.model.state_dict(),trainer.optimizer.state_dict(),trainer.sampler.state_dict()))
        assert metrics["training_loss"]==actual["training_loss"]


@pytest.mark.parametrize("rule",RULES)
def test_qualification_cpu_is_correction_only(rule):
    assert qualify(rule)["passed"]


def test_bootstrap_inactive_never_accesses_network(monkeypatch):
    from llm2.research import state_bootstrap
    monkeypatch.setattr(state_bootstrap,"urlopen",lambda *a,**k:pytest.fail("unexpected network"))
    assert state_bootstrap.bootstrap_git("anything",run=False)=={"status":"not_started"}
    with pytest.raises(ValueError):state_bootstrap.bootstrap_git("not-a-revision",run=True)


def test_merge_identity_and_duplicate_seeds(tmp_path):
    from llm2.research.state_merge import merge
    identity={"selected_rule":"gated_delta","source_sha256":{"a":"b"},"dataset_sha256":"data",
        "learning_rates":{v:.0004 for v in VARIANTS},"environment":{"python":"test","packages":{},"device":"cpu","cuda_version":None}}
    files=[]
    for seed in (241,242,243):
        rows=[{"variant":v,"seed":seed,"validation_loss":1.0} for v in VARIANTS]
        report=analyze(rows,smoke=True);report["identity"]=identity
        path=tmp_path/f"{seed}.json";path.write_text(json.dumps(report));files.append(path)
    result=merge(files,tmp_path/"merged.json")
    assert result["complete_seeds"]==[241,242,243]
    assert not result["contrasts"]["sharing_minus_independent"]["quality_candidate"]
    with pytest.raises(ValueError):merge([files[0],files[0]],tmp_path/"duplicate.json")


def test_forbidden_legacy_options_and_state_shape(tiny_config):
    from llm2.config import ResearchConfig
    from llm2.architectures.state_recurrent import StateRecurrentModel
    with pytest.raises(ValueError):ResearchConfig(architecture="state_hierarchy",scale_conditioning=True)
    with pytest.raises(ValueError):ResearchConfig(architecture="state_hierarchy",scale_conditioning=False,mtp_weight=.1,mtp_tokens=1)
    model=StateRecurrentModel(tiny_config,ResearchConfig(architecture="state_hierarchy",scale_conditioning=False))
    state=model(torch.ones(2,8,dtype=torch.long)).state
    with pytest.raises(ValueError,match="state batch"):model(torch.ones(1,8,dtype=torch.long),state=state)


def test_mocked_candidate_failure_does_not_abort_other_platforms(monkeypatch,tmp_path):
    from llm2.research import state_platforms as platforms
    monkeypatch.setattr(torch.cuda,"is_available",lambda:True)
    monkeypatch.setattr(torch.cuda,"device_count",lambda:2)
    monkeypatch.setattr(torch.cuda,"empty_cache",lambda:None)
    monkeypatch.setattr(platforms,"environment",lambda:{"packages":{}})
    monkeypatch.setattr(platforms,"source_provenance",lambda:{"source_sha256":{}})
    def mock_benchmark(rule,slot,**kwargs):
        if rule=="gated_delta":raise RuntimeError("unsupported mocked backend")
        return {"rule":rule,"slot":str(slot),"status":"passed","correctness":{"passed":True},
            "gpu_name":"Tesla T4","median_step_seconds":1. if rule=="rwkv7" else 2.,
            "peak_vram_bytes":1024,"state_bytes_per_sequence":100}
    monkeypatch.setattr(platforms,"benchmark",mock_benchmark)
    report=platforms.main(["--execute","--output",str(tmp_path/"mock-only-platforms")])
    assert report["selected_rule"]=="rwkv7"
    assert sum(r["status"]=="failed" for r in report["rows"])==2
