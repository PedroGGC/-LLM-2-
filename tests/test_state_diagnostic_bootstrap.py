import importlib.util
import io
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location("diagnostic_bootstrap",ROOT/"src/llm2/research/state_diagnostic_bootstrap.py")
bootstrap=importlib.util.module_from_spec(spec);spec.loader.exec_module(bootstrap)


def test_disabled_invalid_and_changed_helper_do_not_execute(monkeypatch):
    monkeypatch.setattr(bootstrap,"urlopen",lambda *a,**k:pytest.fail("network"))
    assert bootstrap.bootstrap_git("anything")=={"status":"not_started"}
    with pytest.raises(ValueError):bootstrap.bootstrap_git("bad",run=True)
    with pytest.raises(ValueError):bootstrap.bootstrap_git("a"*40,run=True,session_seconds=20000)
    monkeypatch.setattr(bootstrap,"urlopen",lambda *a,**k:io.BytesIO(b"changed helper"))
    with pytest.raises(ValueError,match="helper differs"):bootstrap.bootstrap_git("a"*40,run=True)


@pytest.mark.parametrize("missing_weights",(False,True))
def test_dispatch_checks_weights_before_install_and_keeps_original_pythonpath(tmp_path,monkeypatch,missing_weights):
    reference=json.loads((ROOT/"docs/results/campaign_0027_diagnostic_reference.json").read_text(encoding="utf-8"))
    op_meta={"source_git_revision":"a"*40,"source_files_sha256":{name:bootstrap.sha(ROOT/name) for name in bootstrap.FILES}}
    worker_meta={"source_files_sha256":reference["source_sha256"]}
    original=tmp_path/"llm2-campaign-0027";(original/"data").mkdir(parents=True)
    (original/"data/manifest.json").write_text("{}")
    op_parent=tmp_path/"op";op_parent.mkdir();worker_parent=tmp_path/"sci";worker_parent.mkdir()
    science=tmp_path/"science";science.mkdir();output=tmp_path/"llm2-campaign-0028"
    events=[]
    def inputs(*a):
        events.append("weights")
        if missing_weights:raise FileNotFoundError("missing weights")
        return [{"verified":True}]
    def run_path(path):
        if str(path).endswith("state_diagnostic.py"):return {"inspect_inputs":inputs}
        if str(path).endswith("naming.py"):return {"numbered_output":lambda *a,**k:output}
        pytest.fail(str(path))
    monkeypatch.setattr(bootstrap.runpy,"run_path",run_path)
    def command(args,**kwargs):events.append((args,kwargs))
    import importlib.metadata
    args=(ROOT,science,op_parent,worker_parent,op_meta,worker_meta,{"bounded_command":command},
        SimpleNamespace(remaining=lambda **k:1000.,elapsed=7.),
        (tmp_path,["0","1"],importlib.metadata.version("torch")),28,7200,str(original))
    if missing_weights:
        with pytest.raises(FileNotFoundError,match="missing weights"):bootstrap.dispatch(*args)
        assert events==["weights"]
    else:
        value=bootstrap.dispatch(*args)
        assert events[0]=="weights"
        commands=events[1:]
        assert len(commands)==3 and "pip" in commands[0][0] and "pytest" in commands[1][0]
        call,kwargs=commands[2]
        assert call[1].endswith("state_diagnostic.py")
        assert kwargs["env"]["PYTHONPATH"]==str(science/"src")
        assert "--session-seconds" in call and "--bootstrap-elapsed-seconds" in call
        assert "--checkpoint" not in call and value["optimizer_steps"]==0
        meta=json.loads((op_parent/"bootstrap.json").read_text(encoding="utf-8"))
        assert meta["scientific_revision"]=="1c8ce2abb8f47d3167454b3503598854b7756dad"


def test_altered_operator_or_science_rejected_before_inputs(tmp_path):
    ref=json.loads((ROOT/"docs/results/campaign_0027_diagnostic_reference.json").read_text(encoding="utf-8"))
    op={"source_git_revision":"a"*40,"source_files_sha256":{name:bootstrap.sha(ROOT/name) for name in bootstrap.FILES}}
    args=[ROOT,tmp_path,tmp_path,tmp_path,op,{"source_files_sha256":{}},None,None,None,28,7200,"missing"]
    # Dispatch reads hardware before hashing; no setup is invoked by these tests.
    args[8]=(tmp_path,["0","1"],"unused")
    with pytest.raises(ValueError,match="scientific source mismatch"):bootstrap.dispatch(*args)
    op["source_files_sha256"]={}
    with pytest.raises(ValueError,match="diagnostic source differs"):bootstrap.dispatch(*args)
