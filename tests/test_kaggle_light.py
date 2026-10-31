import base64
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import threading
import time
from types import SimpleNamespace

import pytest

from llm2.research import kaggle_light as light
from llm2.research.notebook_download import download_file
from llm2.research.session import SessionBudget, SessionCampaign
from llm2.research.state_platforms import select
from llm2.research.state_study import storage_budget

ROOT = Path(__file__).resolve().parents[1]


def test_embedded_download_bytes_hash_and_no_disk_copies(tmp_path):
    path = tmp_path / 'results & test.zip'
    payload = bytes(range(256)) * 3
    path.write_bytes(payload)
    shown = []
    info = download_file(path, display=shown.append)
    assert info['sha256'] == hashlib.sha256(payload).hexdigest()
    encoded = re.search(r'base64,([^\"]+)', shown[0])[1]
    assert base64.b64decode(encoded) == payload
    assert 'download="results &amp; test.zip"' in shown[0]
    assert 'https://' not in shown[0]
    assert list(tmp_path.iterdir()) == [path]


def test_large_download_parts_reconstruct_without_files(tmp_path, capsys):
    path = tmp_path / 'test.zip'
    payload = b'0123456789abcdefghij'
    path.write_bytes(payload)
    assert download_file(path, part_bytes=7)['parts'] == 3
    assert 'part=1' in capsys.readouterr().out
    shown = []
    for part in range(1, 4):
        download_file(path, part=part, part_bytes=7, display=shown.append)
    reconstructed = b''.join(base64.b64decode(re.search(r'base64,([^\"]+)', h)[1]) for h in shown)
    assert reconstructed == payload and list(tmp_path.iterdir()) == [path]
    with pytest.raises(ValueError): download_file(path, part=0)
    with pytest.raises(ValueError): download_file(path, part_bytes=33*1024**2)


def test_cleanup_preserves_science_and_reports_consumption(tmp_path):
    campaign = tmp_path / 'llm2-campaign-0024'
    cache = campaign / 'compiler-cache' / 'gpu-0'
    cache.mkdir(parents=True)
    (cache / 'generated.so').write_bytes(b'cache'*1024)
    weights = campaign / 'runs' / 'model.safetensors'
    weights.parent.mkdir()
    weights.write_bytes(b'weights')
    final = tmp_path / 'llm2-campaign-0024-results.zip'
    final.write_bytes(b'final')
    report = light.clean_working(tmp_path, guard=lambda: [])
    assert report['freed_file_bytes'] == len(b'cache'*1024)
    assert not (campaign / 'compiler-cache').exists()
    assert weights.read_bytes() == b'weights' and final.read_bytes() == b'final'
    assert report['before'][0]['bytes'] > report['after'][0]['bytes']


def test_cleanup_blocks_workers_and_unexpected_weights(tmp_path):
    cache = tmp_path / 'llm2-campaign-0024' / 'compiler-cache'
    cache.mkdir(parents=True)
    weights = cache / 'unknown.safetensors'
    weights.write_bytes(b'do not remove')
    with pytest.raises(RuntimeError): light.clean_working(tmp_path, guard=lambda: [17])
    report = light.clean_working(tmp_path, guard=lambda: [])
    assert weights.exists() and report['preserved'] and not report['deleted']
    with pytest.raises(ValueError): light.remove_cache(tmp_path, tmp_path, guard=lambda: [])


def test_cleanup_does_not_follow_external_links(tmp_path):
    outside = tmp_path / 'outside'
    outside.mkdir()
    (outside / 'keep.txt').write_text('keep')
    working = tmp_path / 'working'
    cache = working / 'llm2-campaign-0024' / 'compiler-cache'
    cache.mkdir(parents=True)
    try: (cache / 'link').symlink_to(outside, target_is_directory=True)
    except OSError: pytest.skip('OS does not permit symlinks')
    light.remove_cache(working, cache, guard=lambda: [])
    assert (outside / 'keep.txt').read_text() == 'keep'
    (working / 'llm2-campaign-0025').symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError): light.remove_cache(working, working / 'llm2-campaign-0025' / 'compiler-cache', guard=lambda: [])


def git(repo, *args):
    subprocess.run(['git', '-C', str(repo), *args], check=True, capture_output=True)


def make_checkout(tmp_path):
    parent = tmp_path / 'llm2-cycle2-git-test'
    source = parent / 'source'
    source.mkdir(parents=True)
    git(source, 'init')
    git(source, 'config', 'user.name', 'fixture')
    git(source, 'config', 'user.email', 'fixture@example.invalid')
    git(source, 'remote', 'add', 'origin', light.REMOTE)
    (source / 'code.py').write_text('pass\n')
    git(source, 'add', 'code.py')
    git(source, '-c', 'core.hooksPath=', 'commit', '-m', 'fixture')
    git(source, 'update-ref', 'refs/remotes/origin/experiment/fractal-lrm', 'HEAD')
    return parent, source


def test_clean_checkout_removed_but_dirty_or_unknown_data_preserved(tmp_path):
    parent, source = make_checkout(tmp_path)
    (source / 'code.py').write_text('changed\n')
    with pytest.raises(ValueError, match='modified'): light.remove_checkout(tmp_path, parent, guard=lambda: [])
    git(source, 'checkout', '--', 'code.py')
    (parent / 'unknown.txt').write_text('user data')
    with pytest.raises(ValueError, match='unknown'): light.remove_checkout(tmp_path, parent, guard=lambda: [])
    (parent / 'unknown.txt').unlink()
    receipt = light.remove_checkout(tmp_path, parent, guard=lambda: [])
    assert receipt['bytes'] > 0 and not parent.exists()


def test_unpublished_clean_checkout_is_preserved(tmp_path):
    parent, source = make_checkout(tmp_path)
    (source/'code.py').write_text('new local commit\n')
    git(source, 'add', 'code.py')
    git(source, '-c', 'core.hooksPath=', 'commit', '-m', 'unpublished fixture')
    with pytest.raises(ValueError, match='unpublished'):
        light.remove_checkout(tmp_path, parent, guard=lambda: [])
    assert (source/'code.py').exists()


def test_supervisor_timeout_cannot_be_swallowed_by_worker(tmp_path, monkeypatch):
    monkeypatch.setenv('LLM2_CYCLE4', '1')
    campaign = SessionCampaign(ROOT, tmp_path/'session', device='cpu', gpus=[], budget=SessionBudget(20, 2))
    began = time.monotonic()
    with pytest.raises(TimeoutError, match='supervisor'):
        campaign.command(['-c', 'import time\ntry: time.sleep(30)\nexcept BaseException: time.sleep(30)'],
                         'timeout.log', timeout=.3)
    assert time.monotonic()-began < 5
    assert not campaign.processes and not campaign.cancelled.is_set()
    campaign.command(['-c', 'print("next candidate")'], 'next.log', timeout=5)


def test_storage_limit_aborts_without_removing_files(tmp_path):
    cache = tmp_path/'compiler-cache'
    cache.mkdir();payload=cache/'generated.so';payload.write_bytes(b'x'*100)
    stopped = threading.Event()
    c = SimpleNamespace(output=tmp_path, manifest={}, abort=stopped.set)
    with pytest.raises(OSError, match='compiler_cache_limit'):
        with storage_budget(c, smoke=True, cache_limit=50): pass
    assert stopped.is_set() and payload.exists()
    assert c.manifest['storage_budget']['peak_cache_bytes'] == 100


def test_old_and_over_budget_platform_rows_cannot_be_selected():
    rows = [dict(rule='mlstm', slot=str(s), status='passed', correctness={'passed': True},
                 gpu_name='Tesla T4',median_step_seconds=.1,peak_vram_bytes=1000,state_bytes_per_sequence=100)
            for s in (0,1)]
    assert select(rows) is None
    for row in rows:
        row.update(timeout_enforced_by='parent_process',candidate_limit_seconds=600.,
                   supervisor_elapsed_seconds=100.,compile_policy='static_shapes')
    assert select(rows) == 'mlstm'
    rows[0]['supervisor_elapsed_seconds'] = 800.
    assert select(rows) is None


def test_shallow_checkout_fetches_only_exact_revision(tmp_path, monkeypatch):
    from llm2.research import cycle2_bootstrap as bootstrap
    parent, origin = make_checkout(tmp_path)
    revision = light.git(origin, 'rev-parse', 'HEAD')
    work = tmp_path/'working';work.mkdir()
    original = bootstrap.bounded_command
    calls=[]
    def local_command(args, **kwargs):
        args=list(map(str,args));calls.append(args.copy())
        if args[:4] == ['git','remote','add','origin']: args[-1]=str(origin)
        return original(args, **kwargs)
    monkeypatch.setattr(bootstrap,'bounded_command',local_command)
    source, _, metadata = bootstrap.checkout_git_source(revision, bootstrap._BootstrapSession(360), (work,[],None), shallow=True)
    assert light.git(source,'rev-parse','HEAD') == revision
    assert metadata['shallow'] and light.git(source,'rev-parse','--is-shallow-repository') == 'true'
    assert any(a[:3] == ['git','fetch','--depth=1'] for a in calls)
    assert not any('clone' in a for a in calls)


def test_qualification_timeout_is_failed_and_next_candidate_continues(tmp_path, monkeypatch):
    from llm2.research import state_study as study
    monkeypatch.setattr(study, 'RULES', ('gated_delta', 'mlstm'))
    monkeypatch.setattr(study, 'environment', lambda: {})
    c = SimpleNamespace(output=tmp_path/'llm2-campaign-0025', manifest={}, save=lambda:None,
                        processes=set(), cancelled=threading.Event())
    c.output.mkdir()
    limits=[]
    def command(args, name, **kwargs):
        limits.append(kwargs['timeout'])
        if 'gated_delta' in args: raise TimeoutError('supervisor limit')
        target=Path(args[args.index('--output')+1]);target.mkdir(parents=True)
        row=dict(rule='mlstm',slot='0',status='passed',correctness={'passed':True},gpu_name='Tesla T4',
                 median_step_seconds=.1,peak_vram_bytes=1000,state_bytes_per_sequence=100,compile_policy='static_shapes')
        (target/'platforms.json').write_text(json.dumps({'rows':[row]}))
    c.command=command
    study.qualify(c,SimpleNamespace(gpus=['0','1'],engineering_smoke=False))
    assert limits == [600.]*4
    assert c.manifest['selected_rule'] == 'mlstm'
    assert sum(r['status']=='failed' for r in c.manifest['qualification_rows']) == 2
    assert json.loads((c.output/'platforms.json').read_text())['qualification_version'] == 2


def test_auditor_marks_logged_swallowed_timeout(tmp_path):
    import zipfile
    from llm2.research.audit_state_campaign import audit
    c={'status':'complete','provenance':{'git':{'commit':'a'*40}},'protocol_id':'test','runs':[]}
    row={'rule':'mlstm','slot':'0','status':'passed','median_step_seconds':.1,'first_step_seconds':700.}
    p={'rows':[row],'environment':{},'selected_rule':'mlstm'}
    path=tmp_path/'results.zip'
    with zipfile.ZipFile(path,'w') as z:
        z.writestr('campaign.json',json.dumps(c));z.writestr('platforms.json',json.dumps(p))
        z.writestr('logs/qualify-mlstm-0.log','candidate exceeded 600 seconds')
    report=audit(path)
    assert report['qualification_decision']=='requalify_external_timeout'
    assert report['disk_footprint_bytes'] is None


def test_bootstrap_embeds_partial_result_on_worker_error(tmp_path, monkeypatch):
    from llm2.research import state_bootstrap as bootstrap
    source=tmp_path/'source';source.mkdir()
    output=tmp_path/'llm2-campaign-0025'
    calls=[];downloaded=[]
    def command(args, **kwargs):
        calls.append(args)
        if 'pip' in args: return
        (tmp_path/'llm2-campaign-0025-results.zip').write_bytes(b'partial evidence')
        raise RuntimeError('mocked worker error')
    monkeypatch.setattr(bootstrap.runpy,'run_path',lambda *a:{'numbered_output':lambda *a,**k:output})
    monkeypatch.setattr(bootstrap,'download_file',lambda p:downloaded.append(Path(p)),raising=False)
    torch_version=bootstrap.importlib.metadata.version('torch')
    with pytest.raises(RuntimeError,match='mocked worker'):
        bootstrap._run('a'*40,'qualify',25,None,None,(241,),10800.,7200.,
                       SimpleNamespace(remaining=lambda **k:1000.,elapsed=1.),(tmp_path,['0','1'],torch_version),
                       source,tmp_path,{}, {'tree_bytes':lambda p:100},command)
    assert '--no-cache-dir' in calls[0]
    assert downloaded == [tmp_path/'llm2-campaign-0025-results.zip']
