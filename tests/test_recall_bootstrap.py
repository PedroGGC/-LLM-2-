import hashlib
import subprocess
from pathlib import Path
from types import SimpleNamespace
import pytest
from llm2.research import recall_bootstrap as boot


def test_helpers_match_frozen_git_bytes():
    root=Path(__file__).resolve().parents[1]
    for name,digest in (('cycle2_bootstrap',boot.HELPER_SHA256),('kaggle_light',boot.GUARD_SHA256)):
        raw=subprocess.check_output(['git','show',f'{boot.HELPER_REVISION}:src/llm2/research/{name}.py'],cwd=root)
        assert hashlib.sha256(raw).hexdigest()==digest


def test_bootstrap_dispatch_preserves_clock_torch_number_and_downloads(tmp_path,monkeypatch):
    root=tmp_path/'source';root.mkdir();parent=tmp_path/'checkout';parent.mkdir();calls=[];downloads=[]
    session=SimpleNamespace(elapsed=17.,remaining=lambda **k:7000.)
    helper={'_BootstrapSession':lambda seconds,started:(calls.append(('clock',seconds,started)) or session),
        '_hardware':lambda:(tmp_path,['0','1'],'test-torch'),
        'checkout_git_source':lambda *a,**k:(root,parent,{'source_files_sha256':{},'source_git_revision':'a'*40})}
    def command(arguments,**kwargs):
        calls.append(('command',list(map(str,arguments)),kwargs))
        if 'llm2.research.recall_campaign' in arguments:
            (tmp_path/'llm2-campaign-0031-results.zip').write_bytes(b'result')
            (tmp_path/'llm2-campaign-0031-checkpoints.zip').write_bytes(b'weights')
    helper['bounded_command']=command
    def number(work,*,number,search_roots):
        if number==30:raise FileExistsError()
        return work/'llm2-campaign-0031'
    guard={'active_workers':lambda:[],'remove_checkout':lambda *a:({'bytes':1})}
    monkeypatch.setattr(boot,'fetch',lambda name,digest:helper if name=='cycle2_bootstrap' else guard)
    monkeypatch.setattr(boot.importlib.metadata,'version',lambda name:'test-torch')
    monkeypatch.setattr(boot.runpy,'run_path',lambda p:({'download_file':lambda p:downloads.append(p.name)}
        if p.endswith('notebook_download.py') else {'numbered_output':number}))
    result=boot.bootstrap_git('a'*40,run=True,started_at=123.)
    assert calls[0]==('clock',7200,123.)
    assert len(downloads)==2 and result['campaign'].endswith('0031')
    args=next(c[1] for c in calls if c[0]=='command' and 'llm2.research.recall_campaign' in c[1])
    assert args[args.index('--bootstrap-elapsed-seconds')+1]=='17.0'
    assert args[args.index('--seeds')+1:args.index('--session-seconds')]==['271','272']


def test_active_workers_block_before_checkout(monkeypatch):
    helper={'_BootstrapSession':lambda *a:None,'_hardware':lambda:(None,['0','1'],'torch')}
    monkeypatch.setattr(boot,'fetch',lambda name,digest:helper if name=='cycle2_bootstrap' else {'active_workers':lambda:['active']})
    with pytest.raises(RuntimeError,match='workers active'):boot.bootstrap_git('a'*40,run=True)
