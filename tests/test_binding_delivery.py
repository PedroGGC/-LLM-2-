import ast
import hashlib
import io
from pathlib import Path
import re
import subprocess
import urllib.request
import pytest

ROOT=Path(__file__).resolve().parents[1]
SCRIPT=ROOT/'scripts/09_testar_associacao_e_piramide_kaggle.py'


def literals():
    text=SCRIPT.read_text(encoding='utf-8')
    revision=next(ast.literal_eval(n.value) for n in ast.parse(text).body if isinstance(n,ast.Assign)
        and any(isinstance(t,ast.Name) and t.id=='LLM2_REVISION' for t in n.targets))
    digest=re.search(r'hexdigest\(\) != "([a-f0-9]{64})"',text)[1]
    return text,revision,digest


def test_cell_hash_matches_exact_git_bootstrap():
    text,revision,digest=literals()
    raw=subprocess.check_output(['git','show',revision+':src/llm2/research/binding_bootstrap.py'],cwd=ROOT)
    assert hashlib.sha256(raw).hexdigest()==digest
    assert '"session_seconds": 7200' in text


def test_disabled_cell_has_no_network(monkeypatch):
    monkeypatch.setattr(urllib.request,'urlopen',lambda *a,**k:pytest.fail('unexpected network'))
    namespace={};exec(compile(literals()[0].replace('RUN_CAMPAIGN = True','RUN_CAMPAIGN = False'),str(SCRIPT),'exec'),namespace)
    assert 'LLM2_BINDING_INFO' not in namespace


def test_changed_payload_never_executes(monkeypatch):
    monkeypatch.setattr(urllib.request,'urlopen',lambda *a,**k:io.BytesIO(b'raise RuntimeError("must not execute")'))
    with pytest.raises(ValueError,match='Bootstrap diferente'):exec(compile(literals()[0],str(SCRIPT),'exec'),{})


def test_cell_dispatch_clock_seeds_and_download_callback(monkeypatch):
    text,revision,digest=literals();urls=[]
    fake=b'def bootstrap_git(revision, **kwargs):\n    globals()["download_file"] = lambda path: path\n    return {"revision": revision, "kwargs": kwargs}\n'
    def fetch(url,timeout):urls.append((url,timeout));return io.BytesIO(fake)
    monkeypatch.setattr(urllib.request,'urlopen',fetch)
    monkeypatch.setattr(hashlib,'sha256',lambda raw:type('Digest',(),{'hexdigest':lambda self:digest})())
    namespace={};exec(compile(text,str(SCRIPT),'exec'),namespace);value=namespace['LLM2_BINDING_INFO']
    assert value['revision']==revision and value['kwargs']['run'] is True
    assert value['kwargs']['session_seconds']==7200 and value['kwargs']['seeds']==[311,312]
    assert value['kwargs']['started_at']==namespace['started_at']
    assert revision in urls[0][0] and urls[0][1]==60
    assert namespace['LLM2_BAIXAR']('file.zip')=='file.zip'
