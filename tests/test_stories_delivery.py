import ast
import hashlib
import io
from pathlib import Path
import re
import subprocess
from types import SimpleNamespace
import urllib.request
import pytest

ROOT=Path(__file__).resolve().parents[1]


@pytest.fixture(params=[
    pytest.param(
        ('10_testar_linguagem_tinystories_kaggle.py', 'LLM2_STORIES_INFO',
         {'run_number': 32, 'session_seconds': 7200, 'seeds': [331, 332]}),
        id='stories',
    ),
    pytest.param(
        ('11_testar_memoria_contextual_kaggle.py', 'LLM2_CONTEXT_INFO',
         {'run_number': 33, 'profile': 'context', 'session_seconds': 7200, 'seeds': [341, 342]}),
        id='context',
    ),
])
def delivery(request):
    name, result_key, settings = request.param
    script = ROOT / 'scripts' / name
    source = script.read_text(encoding='utf-8')
    revision = next(ast.literal_eval(node.value) for node in ast.parse(source).body
                    if isinstance(node, ast.Assign) and any(
                        isinstance(target, ast.Name) and target.id == 'LLM2_REVISION' for target in node.targets))
    digest = re.search(r'hexdigest\(\) != "([a-f0-9]{64})"', source)[1]
    return SimpleNamespace(script=script, source=source, revision=revision, digest=digest,
                           result_key=result_key, settings=settings)


def test_cell_hash_matches_exact_git_bootstrap(delivery):
    raw = subprocess.check_output(
        ['git', 'show', delivery.revision + ':src/llm2/research/stories_bootstrap.py'], cwd=ROOT,
    )
    assert hashlib.sha256(raw).hexdigest() == delivery.digest


def test_disabled_cell_has_no_network(monkeypatch, delivery):
    monkeypatch.setattr(urllib.request, 'urlopen', lambda *a, **k: pytest.fail('unexpected network'))
    namespace = {}
    source = delivery.source.replace('RUN_CAMPAIGN = True', 'RUN_CAMPAIGN = False')
    exec(compile(source, str(delivery.script), 'exec'), namespace)
    assert delivery.result_key not in namespace


def test_changed_payload_never_executes(monkeypatch, delivery):
    monkeypatch.setattr(urllib.request, 'urlopen', lambda *a, **k: io.BytesIO(b'raise RuntimeError("must not execute")'))
    with pytest.raises(ValueError, match='Bootstrap diferente'):
        exec(compile(delivery.source, str(delivery.script), 'exec'), {})


def test_cell_dispatch_clock_seeds_and_download_callback(monkeypatch, delivery):
    urls = []
    fake = b'def bootstrap_git(revision, **kwargs):\n    globals()["download_file"] = lambda path: path\n    return {"revision": revision, "kwargs": kwargs}\n'

    def fetch(url, timeout):
        urls.append((url, timeout))
        return io.BytesIO(fake)

    monkeypatch.setattr(urllib.request, 'urlopen', fetch)
    monkeypatch.setattr(hashlib, 'sha256', lambda raw: SimpleNamespace(hexdigest=lambda: delivery.digest))
    namespace = {}
    exec(compile(delivery.source, str(delivery.script), 'exec'), namespace)
    value = namespace[delivery.result_key]
    assert value['revision'] == delivery.revision
    assert value['kwargs'] == {'run': True, 'started_at': namespace['started_at'], **delivery.settings}
    assert delivery.revision in urls[0][0] and urls[0][1] == 60
    assert namespace['LLM2_BAIXAR']('file.zip') == 'file.zip'
