import ast
import hashlib
import io
from pathlib import Path
import re
import subprocess
import urllib.request
import pytest

ROOT=Path(__file__).resolve().parents[1]
SCRIPT=ROOT/'scripts/08_testar_memoria_e_fibonacci_kaggle.py'


def literals():
    text=SCRIPT.read_text(encoding='utf-8')
    revision=next(ast.literal_eval(n.value) for n in ast.parse(text).body if isinstance(n,ast.Assign)
                  and any(isinstance(t,ast.Name) and t.id=='LLM2_REVISION' for t in n.targets))
    digest=re.search(r'hexdigest\(\) != "([a-f0-9]{64})"',text)[1]
    return text,revision,digest


def test_cell_literal_matches_published_git_bootstrap():
    text,revision,digest=literals()
    raw=subprocess.check_output(['git','show',revision+':src/llm2/research/recall_bootstrap.py'],cwd=ROOT)
    assert hashlib.sha256(raw).hexdigest()==digest
    assert '"session_seconds": 7200' in text


def test_cell_disabled_without_network(monkeypatch):
    monkeypatch.setattr(urllib.request,'urlopen',lambda *a,**k:pytest.fail('unexpected network'))
    source=literals()[0].replace('RUN_CAMPAIGN = True','RUN_CAMPAIGN = False')
    namespace={};exec(compile(source,str(SCRIPT),'exec'),namespace)
    assert 'LLM2_RECALL_INFO' not in namespace


def test_changed_bootstrap_is_blocked_before_exec(monkeypatch):
    monkeypatch.setattr(urllib.request,'urlopen',lambda *a,**k:io.BytesIO(b'raise RuntimeError("should not execute")'))
    with pytest.raises(ValueError,match='Bootstrap diferente'):exec(compile(literals()[0],str(SCRIPT),'exec'),{})


def test_dispatch_includes_whole_session_clock_and_download_callback(monkeypatch):
    source,revision,digest=literals()
    fake=b'def bootstrap_git(revision, **kwargs):\n    globals()["download_file"] = lambda path: path\n    return {"revision": revision, "kwargs": kwargs}\n'
    urls=[]
    def fetch(url,timeout):urls.append((url,timeout));return io.BytesIO(fake)
    monkeypatch.setattr(urllib.request,'urlopen',fetch)
    monkeypatch.setattr(hashlib,'sha256',lambda raw:type('Digest',(),{'hexdigest':lambda self:digest})())
    namespace={};exec(compile(source,str(SCRIPT),'exec'),namespace)
    value=namespace['LLM2_RECALL_INFO']
    assert value['revision']==revision and value['kwargs']['run'] is True
    assert value['kwargs']['session_seconds']==7200 and value['kwargs']['seeds']==[271,272]
    assert value['kwargs']['started_at']==namespace['started_at']
    assert revision in urls[0][0] and urls[0][1]==60
    assert namespace['LLM2_BAIXAR']('file.zip')=='file.zip'
