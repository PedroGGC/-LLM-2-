import copy
import pytest
import torch

from llm2.config import ResearchConfig
from llm2.architectures.fractal_recursive import FractalLatentModel, fractal_parameter_formula
from llm2.model import count_parameters
from llm2.objectives import CausalLanguageModeling
from llm2.research.cycle4_protocol import configuration, VARIANTS, ordered_jobs
from llm2.research.diagnostics import diagnose_model
from llm2.research.factory import build_backbone
from llm2.runtime import attention_context, capture_rng
from llm2.trainer import Trainer
from test_training import training_config, assert_nested_equal


def test_protocol_matrix_and_formula_counts(tmp_path):
    jobs=ordered_jobs(tmp_path/"data",tmp_path/"out",["0","1"])
    assert len(jobs)==18 and len({j["id"] for j in jobs})==18
    assert sorted({j["seed"] for j in jobs})==[211,212,213]
    expected={"flrm_fine":2164994,"flrm_4":2165382,"flrm_16":2165382,
              "flrm_full":2165772,"lrm_cached":2017536,"transformer_two":2164992}
    for job in jobs:
        c=job["config_resolved"]
        assert job["registered_parameters"]==job["parameters_formula"]==expected[job["variant"]]
        assert job["max_steps"]==c["optimizer"]["schedule_steps"]==6000
        assert c["data"]["batch_size"]*c["train"]["grad_accum_steps"]==32
        assert c["train"]["checkpoint_interval"]==1500
        assert not c["research"]["scale_conditioning"] and not c["research"]["ncm_slots"]
        assert c["research"]["jepa_weight"]==c["research"]["mtp_weight"]==0
    for variant in VARIANTS:
        assert len({j["gpu_slot"] for j in jobs if j["variant"]==variant})==2


@pytest.mark.parametrize("factors",[[],[4],[16],[4,16]])
@attention_context(True)
def test_selected_branches_causality_and_formula(tiny_config,factors):
    research=ResearchConfig(architecture="fractal",scale_execution="compressed_native",
        compressed_factors=factors,scale_conditioning=False,cache_context_projection=True)
    model=FractalLatentModel(tiny_config,research)
    assert count_parameters(model)==fractal_parameter_formula(tiny_config,research)
    assert model.lm_head.weight is model.embedding.weight
    assert hasattr(model.block,"comp4")==(4 in factors)
    assert hasattr(model.block,"comp16")==(16 in factors)
    seen=[]
    hook=model.block.cores[0].register_forward_hook(lambda m,a,o:seen.append(o.shape[1]))
    ids=torch.randint(0,tiny_config.vocab_size,(2,31))
    model.eval();output=model(ids).logits;hook.remove()
    assert seen==[31,*((31+f-1)//f for f in factors)]*2
    for cut in (1,3,4,15,16,30):
        changed=ids.clone();changed[:,cut:]=(changed[:,cut:]+1)%tiny_config.vocab_size
        torch.testing.assert_close(model(changed).logits[:,:cut],output[:,:cut],rtol=0,atol=0)
    output.square().mean().backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)


def test_default_and_explicit_full_preserve_rng_forward_backward(tiny_config):
    common=dict(architecture="fractal",scale_execution="compressed_native",scale_conditioning=False)
    models=[];states=[]
    for factors in (None,[4,16]):
        torch.manual_seed(122)
        models.append(FractalLatentModel(tiny_config,ResearchConfig(**common,compressed_factors=factors)))
        states.append(torch.get_rng_state())
    assert torch.equal(*states)
    ids=torch.arange(12).repeat(2,1)
    outputs=[]
    for model in models:
        torch.manual_seed(55);value=model(ids).logits;value.square().mean().backward();outputs.append(value)
    assert torch.equal(*outputs)
    for a,b in zip(models[0].parameters(),models[1].parameters()):
        assert torch.equal(a,b)
        assert (a.grad is None and b.grad is None) or torch.equal(a.grad,b.grad)


@pytest.mark.parametrize("bad",[[16,4],[4,4],[1],[True],True,"4"])
def test_bad_compressed_factor_config(bad):
    with pytest.raises(ValueError):
        ResearchConfig(architecture="fractal",scale_execution="compressed_native",
                       scale_conditioning=False,compressed_factors=bad)


@pytest.mark.parametrize("variant",VARIANTS)
@attention_context(True)
def test_variant_memorization(tmp_path,variant):
    cfg=configuration(variant,tmp_path/"unused",tmp_path/"run",211,device="cpu",smoke=True)
    cfg.model.dropout=0
    model=build_backbone(cfg);objective=CausalLanguageModeling()
    ids=torch.arange(18).repeat(2,1);batch={"input_ids":ids[:,:-1],"targets":ids[:,1:]}
    initial=objective(model,batch).loss.item()
    optimizer=torch.optim.AdamW(model.parameters(),lr=.01)
    for _ in range(80):
        optimizer.zero_grad(set_to_none=True);loss=objective(model,batch).loss
        assert torch.isfinite(loss);loss.backward();optimizer.step()
    assert objective(model,batch).loss.item()<initial/15


@pytest.mark.parametrize("variant",VARIANTS)
def test_strict_resume_dropout_accumulation(training_config,tmp_path,variant):
    cfg=configuration(variant,training_config.data.path,tmp_path/"run",211,device="cpu",smoke=True)
    cfg.model.dropout=.2
    def deterministic_counters(trainer):
        return {k:v for k,v in trainer.counters.items() if "seconds" not in k and "vram" not in k}
    with Trainer(cfg) as trainer:
        trainer.fit(until_step=2);checkpoint=trainer.last_checkpoint
        saved_counters=copy.deepcopy(trainer.counters)
        expected_loss=trainer.train_step()["training_loss"]
        expected=copy.deepcopy((trainer.model.state_dict(),trainer.objective.state_dict(),trainer.optimizer.state_dict(),
            trainer.scheduler.state_dict(),trainer.sampler.state_dict(),deterministic_counters(trainer),capture_rng()))
    cfg.train.output_dir=str(tmp_path/"resume")
    with Trainer(cfg,resume=checkpoint) as trainer:
        assert trainer.counters==saved_counters
        assert trainer.train_step()["training_loss"]==expected_loss
        assert_nested_equal(expected,(trainer.model.state_dict(),trainer.objective.state_dict(),trainer.optimizer.state_dict(),
            trainer.scheduler.state_dict(),trainer.sampler.state_dict(),deterministic_counters(trainer),capture_rng()))


def test_reduced_telemetry_preserves_modes_grads_rng(tiny_config):
    cfg=ResearchConfig(architecture="fractal",scale_execution="compressed_native",compressed_factors=[16],
        scale_conditioning=False,cache_context_projection=True)
    model=FractalLatentModel(tiny_config,cfg);ids=torch.arange(17).view(1,-1)%tiny_config.vocab_size
    rng=capture_rng();before=copy.deepcopy(model.state_dict())
    result=diagnose_model(model,ids,max_samples=16,targets=ids)
    assert result["stage1_telemetry"]["kind"]=="selected_native_scales"
    assert len(result["stage1_telemetry"]["core_calls"])==4
    assert model.training and all(p.grad is None for p in model.parameters())
    assert_nested_equal(before,model.state_dict());assert_nested_equal(rng,capture_rng())


def test_reduced_checkpointing_and_nonshared_rejection(tiny_config):
    with pytest.raises(ValueError,match="shared scale weights"):
        ResearchConfig(architecture="fractal",scale_execution="compressed_native",compressed_factors=[4],
            scale_conditioning=False,share_scales=False)
    cfg=ResearchConfig(architecture="fractal",scale_execution="compressed_native",compressed_factors=[4],
        scale_conditioning=False,gradient_checkpointing=True)
    model=FractalLatentModel(tiny_config,cfg)
    assert count_parameters(model)==fractal_parameter_formula(tiny_config,cfg)
    model(torch.arange(17).view(1,-1)).logits.square().mean().backward()
    assert len(model.block.cores)==1
    assert all(any(p.grad is not None and p.grad.abs().sum()>0 for p in core.parameters())
               for core in model.block.cores)


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(),reason="CUDA unavailable")
@pytest.mark.parametrize("variant",VARIANTS)
def test_cuda_fp32_inductor_checkpoint(training_config,tmp_path,variant):
    cfg=configuration(variant,training_config.data.path,tmp_path/"cuda",211,device="cpu",smoke=True)
    cfg.train.device="cuda";cfg.train.deterministic=False;cfg.train.compile=True
    with Trainer(cfg) as trainer:
        row=trainer.train_step()
        assert torch.isfinite(torch.tensor(row["training_loss"])) and row["peak_vram_allocated_bytes"]>0
        checkpoint=trainer.save()
    cfg.train.output_dir=str(tmp_path/"cuda-resume")
    with Trainer(cfg,resume=checkpoint) as trainer:
        assert torch.isfinite(torch.tensor(trainer.train_step()["training_loss"]))
