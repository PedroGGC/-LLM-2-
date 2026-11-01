"""Diagnostic correctness; load operator as file so science can remain frozen."""
import copy
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import sys
import tarfile
import time
import zipfile

import pytest
import torch
from safetensors.torch import save_model

ROOT=Path(__file__).resolve().parents[1]
ENTRY=ROOT/"src/llm2/research/state_diagnostic.py"
spec=importlib.util.spec_from_file_location("diagnostic_operator",ENTRY)
diagnostic=importlib.util.module_from_spec(spec);spec.loader.exec_module(diagnostic)


def model_for(variant):
    from llm2.research.state_protocol import configuration
    from llm2.research.factory import build_backbone
    cfg=configuration(variant,"mlstm","unused","unused",241,smoke=True,device="cpu")
    model=build_backbone(cfg)
    return model,cfg


@pytest.mark.parametrize("variant",diagnostic.VARIANTS)
def test_identity_finite_losses_coverage_and_no_mutation(variant,monkeypatch):
    torch.set_num_threads(1);torch.manual_seed(9)
    model,cfg=model_for(variant)
    before={k:v.clone() for k,v in model.state_dict().items()}
    state=torch.get_rng_state().clone()
    ids=torch.arange(34).reshape(2,17)%cfg.model.vocab_size
    targets=ids.roll(-1,1);targets[:,-1]=-100
    batch={"input_ids":ids,"targets":targets}
    monkeypatch.setattr(torch,"load",lambda *a,**k:pytest.fail("pickle load"))
    result=diagnostic.evaluate_interventions(model,[batch,batch],expected_batches=2,progress=False)
    assert model.training and torch.equal(state,torch.get_rng_state())
    assert all(torch.equal(v,before[k]) for k,v in model.state_dict().items())
    expected=4 if "hierarchy" in variant else 2 if variant=="recurrent" else 1
    assert len(result["conditions"])==expected
    assert all(r["targets"]==64 and sum(r["phase_targets"])==64 and sum(r["quarter_targets"])==64
               for r in result["conditions"].values())
    from llm2.objectives import CausalLanguageModeling
    from llm2.evaluation import evaluate
    baseline=evaluate(model,CausalLanguageModeling(),[batch,batch],deterministic=False)
    assert result["conditions"]["original"]["validation_loss"]==pytest.approx(baseline["validation_loss"],abs=1e-6)
    assert result["optimizer_steps"]==0 and not result["test_evaluated"]
    assert all(not cell._forward_hooks for cell in getattr(model,"cells",[]))


@pytest.mark.parametrize("variant",("hierarchy_independent","hierarchy_shared"))
def test_components_causality_and_original_identity(variant):
    model,cfg=model_for(variant);model.eval()
    ids=torch.arange(17)[None,:]
    with torch.no_grad():
        original,fine,broad=diagnostic.components(model,ids)
        torch.testing.assert_close(model.norm(fine+broad),original,rtol=0,atol=0)
        changed=ids.clone();changed[:,8:]+=50
        values,_=diagnostic.representations(model,ids)
        altered,_=diagnostic.representations(model,changed)
        for name in values:torch.testing.assert_close(values[name][:,:8],altered[name][:,:8],rtol=0,atol=0)
        assert not torch.count_nonzero(broad[:,:4])


def test_hooks_removed_and_modes_rng_restored_on_failure(monkeypatch):
    model,_=model_for("hierarchy_shared")
    ids=torch.arange(17)[None,:];rng=torch.get_rng_state().clone()
    def fail(*a,**k):raise RuntimeError("injected")
    monkeypatch.setattr(model.norm,"forward",fail)
    with pytest.raises(RuntimeError,match="injected"):
        diagnostic.evaluate_interventions(model,[{"input_ids":ids,"targets":ids}],progress=False)
    assert model.training and torch.equal(rng,torch.get_rng_state())
    assert all(not cell._forward_hooks for cell in model.cells)


def test_deadline_and_pilot_never_publish_shortened_complete():
    model,_=model_for("recurrent");ids=torch.arange(17)[None,:]
    batch={"input_ids":ids,"targets":ids}
    with pytest.raises(TimeoutError,match="deadline"):
        diagnostic.evaluate_interventions(model,[batch],deadline=0,progress=False)
    with pytest.raises(TimeoutError,match="coverage does not fit"):
        diagnostic.evaluate_interventions(model,[batch,batch],deadline=time.monotonic()+10,
            expected_batches=10000000,progress=False)
    with pytest.raises(ValueError,match="incomplete"):
        diagnostic.evaluate_interventions(model,[batch],expected_batches=3,progress=False)


def input_fixture(tmp_path):
    cp=tmp_path/"llm2-campaign-0027/runs/job/checkpoints/step-00006000"
    cp.mkdir(parents=True)
    (cp/"metadata.json").write_bytes(b'{"files":{}}')
    (cp/"model.safetensors").write_bytes(b"weights")
    ref={"jobs":[{"variant":"recurrent","checkpoint":"runs/job/checkpoints/step-00006000",
        "metadata_sha256":diagnostic.sha(cp/"metadata.json"),"model_sha256":diagnostic.sha(cp/"model.safetensors")}],
        "checkpoints_zip_sha256":""}
    return cp,ref


def test_input_native_hash_mismatch_missing_and_link_block(tmp_path):
    cp,ref=input_fixture(tmp_path);original=tmp_path/"llm2-campaign-0027"
    tasks=diagnostic.inspect_inputs(ref,original,[tmp_path]);assert tasks[0]["checkpoint"]==str(cp)
    (cp/"model.safetensors").write_bytes(b"bad")
    with pytest.raises(ValueError,match="divergiu"):diagnostic.inspect_inputs(ref,original,[tmp_path])
    (cp/"model.safetensors").unlink()
    with pytest.raises(FileNotFoundError,match="Faltam os pesos"):diagnostic.inspect_inputs(ref,original,[tmp_path])
    ref["jobs"][0]["checkpoint"]="../escape"
    with pytest.raises(ValueError,match="unsafe"):diagnostic.inspect_inputs(ref,original,[tmp_path])


def test_backup_restores_only_weights_metadata_and_rejects_change(tmp_path):
    cp,ref=input_fixture(tmp_path);backup=tmp_path/"llm2-campaign-0027-checkpoints.zip"
    with zipfile.ZipFile(backup,"x") as z:
        for p in cp.iterdir():z.write(p,p.relative_to(tmp_path/"llm2-campaign-0027").as_posix())
        z.writestr("training_state.pt",b"NEVER LOAD")
    ref["checkpoints_zip_sha256"]=diagnostic.sha(backup)
    tasks=diagnostic.inspect_inputs(ref,tmp_path/"missing",[tmp_path]);assert tasks[0]["archive"]==str(backup)
    output=tmp_path/"new";output.mkdir()
    restored,owned=diagnostic.restore_weights(tasks[0],ref["jobs"][0],output)
    assert set(p.name for p in restored.iterdir())=={"model.safetensors","metadata.json"}
    assert owned==restored and (cp/"model.safetensors").read_bytes()==b"weights"
    ref["jobs"][0]["model_sha256"]="0"*64
    with pytest.raises(ValueError,match="payload mismatch"):
        diagnostic.inspect_inputs(ref,tmp_path/"missing",[tmp_path])
    diagnostic.remove_restored(restored,output)
    assert not restored.exists() and (cp/"model.safetensors").is_file() and backup.is_file()


def test_linked_ancestor_and_external_cleanup_rejected(tmp_path,monkeypatch):
    root=tmp_path/"original";root.mkdir();weights=root/"weights";weights.write_bytes(b"keep")
    method=Path.is_symlink
    monkeypatch.setattr(Path,"is_symlink",lambda self:self==root or method(self))
    with pytest.raises(ValueError,match="link"):diagnostic.regular(weights)
    with pytest.raises(ValueError,match="unsafe"):diagnostic.remove_restored(root,tmp_path/"output")
    assert weights.read_bytes()==b"keep"


def test_export_excludes_data_and_weights_and_preserves_partial(tmp_path):
    output=tmp_path/"llm2-campaign-0028";output.mkdir()
    (output/"data").mkdir();(output/"data/train.bin").write_bytes(b"preserve")
    (output/"restored-weights").mkdir();(output/"restored-weights/model.safetensors").write_bytes(b"preserve")
    diagnostic.write_json(output/"campaign.json",{"status":"failed"})
    diagnostic.write_json(output/"recurrent.json",{"status":"complete"})
    result=diagnostic.export_results(output)
    with zipfile.ZipFile(result["path"]) as z:assert set(z.namelist())=={"campaign.json","recurrent.json"}
    assert (output/"data/train.bin").read_bytes()==b"preserve"
    with pytest.raises(FileExistsError):diagnostic.export_results(output)


def test_runtime_source_versions_and_config_maps_are_strict(monkeypatch):
    from llm2 import runtime
    ref={"source_sha256":{"x":"a"},"versioned_configs_sha256":{"c":"b"},
         "environment":{"python":"x","packages":{},"cuda_version":None,"cudnn_version":None}}
    monkeypatch.setattr(runtime,"source_provenance",lambda:{"source_sha256":{"x":"a"},"versioned_configs_sha256":{"c":"b"}})
    monkeypatch.setattr(runtime,"environment",lambda:{**ref["environment"],"device":"cpu"})
    diagnostic.verify_runtime(ref,smoke=True)
    other=copy.deepcopy(ref);other["source_sha256"]={}
    with pytest.raises(ValueError,match="source differs"):diagnostic.verify_runtime(other,smoke=True)
    other=copy.deepcopy(ref);other["environment"]["python"]="different"
    with pytest.raises(ValueError,match="versions differ"):diagnostic.verify_runtime(other,smoke=True)
    other=copy.deepcopy(ref);other["versioned_configs_sha256"]={}
    with pytest.raises(ValueError,match="configurations differ"):diagnostic.verify_runtime(other,smoke=True)


def test_integration_original_source_cpu_four_weights_no_training(tmp_path):
    revision="1c8ce2abb8f47d3167454b3503598854b7756dad"
    raw=subprocess.check_output(["git","archive",revision],cwd=ROOT)
    science=tmp_path/"science";science.mkdir()
    with tarfile.open(fileobj=io.BytesIO(raw)) as z:z.extractall(science,filter="data")
    create=tmp_path/"create.py"
    create.write_text('''
import json,sys,torch
from pathlib import Path
from safetensors.torch import save_model
from llm2.document_data import prepare_documents
from llm2.data import PreparedData,write_json,sha256_file
from llm2.runtime import environment,source_provenance
from llm2.research.state_protocol import configuration
from llm2.research.factory import build_backbone
from llm2.research.document_windows import validation
from llm2.evaluation import evaluate
from llm2.objectives import CausalLanguageModeling
base=Path(sys.argv[1]);base.mkdir();original=base/'llm2-campaign-0027';original.mkdir()
docs=base/'docs.jsonl'
docs.write_text(''.join(json.dumps({'id':s,'text':s+' abcdefghijklmnopqrstuvwxyz '*6,'split':s})+'\\n' for s in ['train','val','test']))
prepare_documents([docs],original/'data',kind='byte');data=PreparedData(original/'data')
torch.set_num_threads(1);torch.manual_seed(77);jobs=[]
for variant in ['recurrent','hierarchy_independent','hierarchy_shared','transformer_two']:
 cfg=configuration(variant,'mlstm',data.root,base/'unused',241,smoke=True,device='cpu')
 model=build_backbone(cfg);cp=original/'runs'/variant/'checkpoints/step-00006000';cp.mkdir(parents=True)
 save_model(model,str(cp/'model.safetensors'))
 md={'config':cfg.to_dict(),'provenance':source_provenance()};write_json(cp/'metadata.json',md)
 val=evaluate(model,CausalLanguageModeling(),validation(data,17,2),deterministic=False)
 jobs.append({'variant':variant,'checkpoint':cp.relative_to(original).as_posix(),'config':cfg.to_dict(),
 'metadata_sha256':sha256_file(cp/'metadata.json'),'model_sha256':sha256_file(cp/'model.safetensors'),
 'validation_loss':val['validation_loss'],'validation_targets':val['validation_tokens'],
 'validation_batches':val['validation_batches'],'parameters':sum(p.numel() for p in model.parameters())})
ref={'source_sha256':source_provenance()['source_sha256'],'versioned_configs_sha256':source_provenance()['versioned_configs_sha256'],
'environment':environment(),'dataset_sha256':data.fingerprint,'jobs':jobs}
write_json(base/'reference.json',ref);write_json(base/'operator.json',{'files_sha256':{'src/llm2/research/state_diagnostic.py':'test'}})
''',encoding="utf-8")
    import os
    env={**os.environ,"PYTHONPATH":str(science/"src"),"PYTHONIOENCODING":"utf-8","CUDA_VISIBLE_DEVICES":""}
    base=tmp_path/"fixture"
    subprocess.run([sys.executable,str(create),str(base)],env=env,cwd=science,check=True,capture_output=True,text=True)
    output=base/"llm2-campaign-0028"
    completed=subprocess.run([sys.executable,str(ENTRY),"--worker-source",str(science),"--operator-root",str(ROOT),
        "--reference",str(base/"reference.json"),"--operator-metadata",str(base/"operator.json"),
        "--original",str(base/"llm2-campaign-0027"),"--output",str(output),"--engineering-smoke"],
        env=env,cwd=science,capture_output=True,text=True,timeout=120)
    assert completed.returncode==0,completed.stdout+completed.stderr
    campaign=json.loads((output/"campaign.json").read_text(encoding="utf-8"))
    assert campaign["status"]=="complete" and len(campaign["runs"])==4
    assert sum(len(j["result"]["conditions"]) for j in campaign["runs"])==11
    assert all(j["result"]["reproduced"] and j["result"]["optimizer_steps"]==0 for j in campaign["runs"])
    with zipfile.ZipFile(base/"llm2-campaign-0028-results.zip") as z:
        assert z.testzip() is None and not any(p.endswith((".safetensors",".pt")) for p in z.namelist())
        assert "conditions.csv" in z.namelist() and "report.json" in z.namelist()
    # Failed reproduction exports the diagnostic evidence, never a promoted result.
    reference=json.loads((base/"reference.json").read_text(encoding="utf-8"))
    reference["jobs"][0]["validation_loss"]+=.1
    diagnostic.write_json(base/"invalid-reference.json",reference)
    failed_output=base/"llm2-campaign-0029"
    failed=subprocess.run([sys.executable,str(ENTRY),"--worker-source",str(science),"--operator-root",str(ROOT),
        "--reference",str(base/"invalid-reference.json"),"--operator-metadata",str(base/"operator.json"),
        "--original",str(base/"llm2-campaign-0027"),"--output",str(failed_output),"--engineering-smoke"],
        env=env,cwd=science,capture_output=True,text=True,timeout=120)
    assert failed.returncode!=0
    with zipfile.ZipFile(base/"llm2-campaign-0029-results.zip") as z:
        assert json.loads(z.read("campaign.json"))["status"]=="failed"
        assert json.loads(z.read("recurrent.json"))["reproduced"] is False
        assert z.testzip() is None
    assert all(diagnostic.sha(base/"llm2-campaign-0027"/j["checkpoint"]/"model.safetensors")==j["model_sha256"] for j in reference["jobs"])
