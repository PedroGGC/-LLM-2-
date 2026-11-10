from types import SimpleNamespace
import pytest
from llm2.research import stories_bootstrap as boot


def test_dispatch_clock_number_torch_cache_and_downloads(tmp_path,monkeypatch):
    root=tmp_path/'source';root.mkdir();parent=tmp_path/'checkout';parent.mkdir();calls=[];downloads=[]
    session=SimpleNamespace(elapsed=19.,remaining=lambda **kwargs:7000.)
    helper={'_BootstrapSession':lambda seconds,started:(calls.append(('clock',seconds,started)) or session),
        '_hardware':lambda:(tmp_path,['0','1'],'test-torch'),
        'checkout_git_source':lambda *a,**k:(root,parent,{'source_files_sha256':{},'source_git_revision':'a'*40})}
    def command(arguments,**kwargs):
        calls.append(('command',list(map(str,arguments)),kwargs))
        if 'llm2.research.stories_campaign' in arguments:
            (tmp_path/'llm2-campaign-0033-results.zip').write_bytes(b'result')
            (tmp_path/'llm2-campaign-0033-checkpoints.zip').write_bytes(b'weights')
    helper['bounded_command']=command
    def numbering(work,*,number,search_roots):
        if number==32:raise FileExistsError()
        return work/'llm2-campaign-0033'
    guard={'active_workers':lambda:[],'remove_checkout':lambda *a:({'bytes':1})}
    monkeypatch.setattr(boot,'fetch',lambda name,digest:helper if name=='cycle2_bootstrap' else guard)
    monkeypatch.setattr(boot.importlib.metadata,'version',lambda name:'test-torch')
    monkeypatch.setattr(boot.runpy,'run_path',lambda path:({'download_file':lambda p:downloads.append(p.name)}
        if path.endswith('notebook_download.py') else {'numbered_output':numbering}))
    value=boot.bootstrap_git('a'*40,run=True,started_at=123.)
    assert value['campaign'].endswith('0033') and len(downloads)==2 and calls[0]==('clock',7200,123.)
    cmd=next(c[1] for c in calls if c[0]=='command' and 'llm2.research.stories_campaign' in c[1])
    assert cmd[cmd.index('--bootstrap-elapsed-seconds')+1]=='19.0'
    assert cmd[cmd.index('--seeds')+1:cmd.index('--session-seconds')]==['331','332']
    assert cmd[cmd.index('--cache')+1].endswith('llm2-tinystories-prefix-cache')


def test_active_workers_block_before_checkout(monkeypatch):
    helper={'_BootstrapSession':lambda *a:None,'_hardware':lambda:(None,['0','1'],'torch')}
    monkeypatch.setattr(boot,'fetch',lambda name,digest:helper if name=='cycle2_bootstrap' else {'active_workers':lambda:['active']})
    with pytest.raises(RuntimeError,match='workers active'):boot.bootstrap_git('a'*40,run=True)
