import ast
import builtins
import hashlib
import io
from pathlib import Path
import subprocess
from urllib import request

import pytest

ROOT=Path(__file__).resolve().parents[1]
CELL=ROOT/"scripts/06_diagnosticar_pesos_0027_kaggle.py"


def pinned_source():
    tree=ast.parse(CELL.read_text(encoding="utf-8"))
    revision=next(ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign)
        and isinstance(n.targets[0],ast.Name) and n.targets[0].id=="LLM2_REVISION")
    return subprocess.check_output(["git","show",revision+":src/llm2/research/state_diagnostic_bootstrap.py"],cwd=ROOT)


def test_pinned_bytes_and_original_science():
    source=pinned_source()
    constants=[n.value for n in ast.walk(ast.parse(CELL.read_text(encoding="utf-8")))
               if isinstance(n,ast.Constant) and isinstance(n.value,str)]
    assert hashlib.sha256(source).hexdigest() in constants
    assert b'1c8ce2abb8f47d3167454b3503598854b7756dad' in source


def test_inactive_cell_has_no_network(monkeypatch):
    tree=ast.parse(CELL.read_text(encoding="utf-8"))
    for n in tree.body:
        if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name) and n.targets[0].id=="RUN_CAMPAIGN":n.value=ast.Constant(False)
    monkeypatch.setattr(request,"urlopen",lambda *a,**k:pytest.fail("network"))
    builtins.exec(compile(ast.fix_missing_locations(tree),str(CELL),"exec"),{})


def test_cell_dispatch_and_embedded_download_callback(monkeypatch):
    monkeypatch.setattr(request,"urlopen",lambda *a,**k:io.BytesIO(pinned_source()))
    calls=[];callback=lambda *a,**k:None
    def bootstrap_exec(code,namespace):
        builtins.exec(code,namespace)
        namespace["bootstrap_git"]=lambda revision,**kwargs:calls.append((revision,kwargs)) or {"mock":True}
        namespace["download_file"]=callback
    namespace={"exec":bootstrap_exec}
    builtins.exec(compile(CELL.read_text(encoding="utf-8"),str(CELL),"exec"),namespace)
    assert len(calls)==1 and calls[0][1]["run_number"]==28 and calls[0][1]["session_seconds"]==7200
    assert calls[0][1]["started_at"]>0 and namespace["LLM2_BAIXAR"] is callback
    assert "seeds" not in calls[0][1] and "steps" not in calls[0][1]


def test_modified_bootstrap_is_rejected(monkeypatch):
    monkeypatch.setattr(request,"urlopen",lambda *a,**k:io.BytesIO(b"raise RuntimeError('do not execute')"))
    with pytest.raises(ValueError,match="Bootstrap diferente"):
        builtins.exec(compile(CELL.read_text(encoding="utf-8"),str(CELL),"exec"),{})
