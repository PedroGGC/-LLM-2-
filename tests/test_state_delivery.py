import ast
import builtins
import hashlib
import io
from pathlib import Path
import subprocess
from urllib import request
import pytest

ROOT=Path(__file__).resolve().parents[1]
CELL=ROOT/"scripts/03_selecionar_nucleo_recorrente_kaggle.py"


def pinned_source():
    tree=ast.parse(CELL.read_text(encoding="utf-8"))
    revision=next(ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign)
        and isinstance(n.targets[0],ast.Name) and n.targets[0].id=="LLM2_REVISION")
    return subprocess.check_output(["git","show",revision+":src/llm2/research/state_bootstrap.py"],cwd=ROOT)


def test_kaggle_source_hash_matches_pinned_git_object():
    digest=hashlib.sha256(pinned_source()).hexdigest()
    assert digest in CELL.read_text(encoding="utf-8")


def test_kaggle_inactive_no_network(monkeypatch):
    tree=ast.parse(CELL.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node,ast.Assign) and isinstance(node.targets[0],ast.Name) and node.targets[0].id=="RUN_CAMPAIGN":
            node.value=ast.Constant(False)
    monkeypatch.setattr(request,"urlopen",lambda *a,**k:pytest.fail("inactive cell used network"))
    builtins.exec(compile(ast.fix_missing_locations(tree),str(CELL),"exec"),{})


def test_kaggle_cell_dispatch_mock_only(monkeypatch):
    monkeypatch.setattr(request,"urlopen",lambda *a,**k:io.BytesIO(pinned_source()))
    calls=[]
    def bootstrap_exec(code,namespace):
        builtins.exec(code,namespace)
        namespace["bootstrap_git"]=lambda revision,**kwargs:calls.append((revision,kwargs)) or {"mock":True}
    namespace={"exec":bootstrap_exec}
    builtins.exec(compile(CELL.read_text(encoding="utf-8"),str(CELL),"exec"),namespace)
    assert calls[0][1]["phase"]=="qualify"
    assert calls[0][1]["target_seconds"]==7200 and calls[0][1]["session_seconds"]==10800
    assert calls[0][1]["started_at"]>0 and namespace["LLM2_STATE_INFO"]=={"mock":True}


def test_kaggle_unexpected_source_is_rejected(monkeypatch):
    monkeypatch.setattr(request,"urlopen",lambda *a,**k:io.BytesIO(b"raise RuntimeError('must not execute')"))
    with pytest.raises(ValueError,match="Bootstrap diferente"):
        builtins.exec(compile(CELL.read_text(encoding="utf-8"),str(CELL),"exec"),{})
