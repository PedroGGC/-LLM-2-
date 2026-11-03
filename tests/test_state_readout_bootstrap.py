import importlib.util
import io
import json
import hashlib
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location("readout_bootstrap",ROOT/"src/llm2/research/state_readout_bootstrap.py")
bootstrap=importlib.util.module_from_spec(spec);spec.loader.exec_module(bootstrap)


def test_bootstrap_disabled_invalid_and_changed_payload(monkeypatch):
    monkeypatch.setattr(bootstrap,"urlopen",lambda *a,**k:pytest.fail("network"))
    assert bootstrap.bootstrap_git("bad")=={"status":"not_started"}
    for kwargs in ({"session_seconds":10800},{"run_number":28},{"variants":["bad"]}):
        with pytest.raises(ValueError):bootstrap.bootstrap_git("a"*40,run=True,**kwargs)
    monkeypatch.setattr(bootstrap,"urlopen",lambda *a,**k:io.BytesIO(b"wrong helper"))
    with pytest.raises(ValueError,match="helper differs"):bootstrap.bootstrap_git("a"*40,run=True)


def test_helper_and_guard_match_frozen_git_bytes():
    for name,expected in (("cycle2_bootstrap.py",bootstrap.HELPER_SHA256),("kaggle_light.py",bootstrap.GUARD_SHA256)):
        raw=subprocess.check_output(["git","show",f"{bootstrap.HELPER_REVISION}:src/llm2/research/{name}"],cwd=ROOT)
        assert hashlib.sha256(raw).hexdigest()==expected


def test_failed_preflight_receipt_survives_checkout_removal(tmp_path):
    parent=tmp_path/"checkout";parent.mkdir()
    (parent/"bootstrap.json").write_text(json.dumps({"status":"failed","error":"checksum"}))
    receipt=bootstrap.preserve_preflight_error(tmp_path,parent)
    (parent/"bootstrap.json").unlink();parent.rmdir()
    assert json.loads(receipt.read_text())=={"status":"failed","error":"checksum"}


def test_prepared_data_requires_all_payloads_and_no_overwrite(tmp_path):
    folder=tmp_path/"llm2-campaign-0027/data";folder.mkdir(parents=True)
    (folder/"manifest.json").write_text("{}")
    assert bootstrap.existing_data(tmp_path) is None
    for name in ("tokenizer.json","train.bin","val.bin","test.bin"):(folder/name).write_bytes(b"untouched")
    assert bootstrap.existing_data(tmp_path)==folder.resolve()
    assert all((folder/name).read_bytes()==b"untouched" for name in ("train.bin","val.bin","test.bin"))


@pytest.mark.parametrize("prepared",[False,True])
def test_dispatch_single_checkout_reuses_data_or_recovers_before_install(tmp_path,monkeypatch,prepared):
    parent=tmp_path/"checkout";parent.mkdir();root=parent/"source";root.mkdir()
    if prepared:
        data=tmp_path/"llm2-campaign-0027/data";data.mkdir(parents=True)
        for name in ("manifest.json","tokenizer.json","train.bin","val.bin","test.bin"):(data/name).write_bytes(b"original")
    output=tmp_path/"llm2-campaign-0029"
    monkeypatch.setattr(bootstrap.runpy,"run_path",lambda *a:{"numbered_output":lambda *a,**k:output})
    calls=[]
    def command(args,**kwargs):calls.append((list(map(str,args)),kwargs))
    import importlib.metadata
    hardware=(tmp_path,["0","1"],importlib.metadata.version("torch"))
    meta={"source_files_sha256":{"src/llm2/research/wikitext.py":"hash"},"source_git_revision":"a"*40}
    value=bootstrap.dispatch(root,parent,meta,{"bounded_command":command},
        SimpleNamespace(remaining=lambda **k:1000.,elapsed=4.),hardware,29,7200,bootstrap.VARIANTS)
    if not prepared:
        assert calls[0][0][1].endswith("wikitext_cache.py")
        assert "--source-sha256" in calls[0][0]
    else:
        assert "pip" in calls[0][0] and "--data" in calls[-1][0] and "--cache" not in calls[-1][0]
    args,kwargs=calls[-1]
    assert args[1:3]==["-m","llm2.research.state_readout"]
    assert kwargs["env"]["PYTHONPATH"]==str(root/"src")
    assert args[args.index("--session-seconds")+1]=="7200"
    assert args[args.index("--variants")+1:args.index("--variants")+6]==list(bootstrap.VARIANTS)
    assert value["campaign"]==str(output)
    assert "resume" not in ' '.join(args)
