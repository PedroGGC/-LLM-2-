import ast
import builtins
import hashlib
import io
from pathlib import Path
import subprocess
from urllib import request

import pytest

ROOT=Path(__file__).resolve().parents[1]
CELL=ROOT/"scripts/05_testar_hierarquia_mlstm_kaggle.py"


def pinned_source():
    tree=ast.parse(CELL.read_text(encoding="utf-8"))
    revision=next(ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign)
        and isinstance(n.targets[0],ast.Name) and n.targets[0].id=="LLM2_REVISION")
    return subprocess.check_output(["git","show",revision+":src/llm2/research/state_b_bootstrap.py"],cwd=ROOT)


def test_release_matches_git_bytes_and_keeps_qualified_science():
    payload=pinned_source()
    assert hashlib.sha256(payload).hexdigest() in CELL.read_text(encoding="utf-8")
    assert b'1c8ce2abb8f47d3167454b3503598854b7756dad' in payload


def test_inactive_cell_never_accesses_network(monkeypatch):
    tree=ast.parse(CELL.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node,ast.Assign) and isinstance(node.targets[0],ast.Name) and node.targets[0].id=="RUN_CAMPAIGN":
            node.value=ast.Constant(False)
    monkeypatch.setattr(request,"urlopen",lambda *a,**k:pytest.fail("unexpected network"))
    builtins.exec(compile(ast.fix_missing_locations(tree),str(CELL),"exec"),{})


def test_cell_dispatch_and_download_callback_mocked(monkeypatch):
    monkeypatch.setattr(request,"urlopen",lambda *a,**k:io.BytesIO(pinned_source()))
    calls=[]
    callback=lambda *a,**k:None
    def bootstrap_exec(code,namespace):
        builtins.exec(code,namespace)
        namespace["bootstrap_git"]=lambda revision,**kwargs:calls.append((revision,kwargs)) or {"mock":True}
        namespace["download_file"]=callback
    namespace={"exec":bootstrap_exec}
    builtins.exec(compile(CELL.read_text(encoding="utf-8"),str(CELL),"exec"),namespace)
    assert len(calls)==1
    settings=calls[0][1]
    assert settings["run_number"]==26 and settings["seeds"]==[241]
    assert settings["target_seconds"]==7200 and settings["session_seconds"]==10800
    assert settings["started_at"]>0 and namespace["LLM2_BAIXAR"] is callback


def test_changed_bootstrap_is_not_executed(monkeypatch):
    monkeypatch.setattr(request,"urlopen",lambda *a,**k:io.BytesIO(b"raise RuntimeError('must not execute')"))
    with pytest.raises(ValueError,match="Bootstrap diferente"):
        builtins.exec(compile(CELL.read_text(encoding="utf-8"),str(CELL),"exec"),{})
