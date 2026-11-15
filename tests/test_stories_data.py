import hashlib
import io
from pathlib import Path
import numpy as np
import pytest
import torch
from llm2.research import stories_data as data
from llm2.research.stories_protocol import configuration


def fixture_data(path):
    splits={}
    for split in ('train','cal_val','val','test'):
        documents=[(f'{split}: Once upon a time, a little fox found a red ball. '+str(i)+'\n').encode()*(i+1) for i in range(2)]
        splits[split]={hashlib.sha256(raw).hexdigest():raw for raw in documents}
    return data.write_dataset(path,splits,spec={'engineering_smoke':True})


def test_document_boundaries_tails_and_sampler_resume(tmp_path):
    d=fixture_data(tmp_path/'data');cfg=configuration('transition',d.root,tmp_path/'run',331,smoke=True,device='cpu')
    stream=data.StoryStream(d,cfg);stream.next('cpu');saved=stream.state_dict();expected=stream.next('cpu')
    other=data.StoryStream(d,cfg);other.load_state_dict(saved)
    for k,v in other.next('cpu').items():assert torch.equal(v,expected[k])
    count=0
    for pack in data.evaluation_batches(d,'val',32,2,'cpu'):
        mask=pack['targets']!=-100;count+=int(mask.sum())
        assert ((pack['targets'][mask]>=0)&(pack['targets'][mask]<256)).all()
        assert (pack['input_ids'][~mask]==256).all()
    assert count==d.manifest['splits']['val']['bytes']
    for i,row in enumerate(d.docs['val']):
        x,y=data.window(d,'val',i,0,512)
        assert x[0]==256 and (y[row['bytes']:]==-100).all()
        actual=y[y!=-100].astype(np.uint8).tobytes();assert hashlib.sha256(actual).hexdigest()==row['sha256']
    with pytest.raises(ValueError,match='reserved test'):list(data.evaluation_batches(d,'test',32,2,'cpu'))
    with pytest.raises(ValueError,match='pinned'):data.StoriesData(d.root)


def test_global_dedup_and_complete_record_policy(tmp_path):
    docs,duplicates=data.parse_prefix(b'\nOne story.\n<|endoftext|>\nOne story.\n<|endoftext|>unfinished')
    assert duplicates==1 and list(docs.values())==[b'One story.']
    with pytest.raises(ValueError,match='leakage'):
        data.write_dataset(tmp_path/'bad',{s:docs for s in ('train','cal_val','val','test')},spec={'engineering':True})


def test_manifest_portable_and_tamper_rejected(tmp_path):
    a=fixture_data(tmp_path/'a');b=fixture_data(tmp_path/'b');assert a.fingerprint==b.fingerprint
    assert b'\r\n' not in (a.root/'manifest.json').read_bytes()
    path=a.root/'val.bin';raw=path.read_bytes();a.arrays['val']._mmap.close()
    path.write_bytes(bytes([raw[0]^1])+raw[1:])
    with pytest.raises(ValueError,match='checksum'):data.StoriesData(a.root,require_pinned=False)


def test_bounded_prefix_download_and_wrong_range(tmp_path,monkeypatch):
    raw=b'small complete story';digest=hashlib.sha256(raw).hexdigest()
    monkeypatch.setitem(data.PREFIXES,'test.txt',(len(raw),digest))
    class Response(io.BytesIO):
        status=206;headers={'Content-Range':f'bytes 0-{len(raw)-1}/999999'}
    calls=[]
    def opener(request,timeout):calls.append(request.get_header('Range'));return Response(raw+b'ignored tail')
    path=data.download_prefix(tmp_path,'test.txt',opener=opener)
    assert path.read_bytes()==raw and calls==[f'bytes=0-{len(raw)-1}']
    assert data.download_prefix(tmp_path,'test.txt',opener=lambda *a,**k:pytest.fail('unneeded download'))==path
    class Wrong(Response):headers={'Content-Range':'bytes 10-20/99999'}
    with pytest.raises(ValueError,match='range'):data.download_prefix(tmp_path/'wrong','test.txt',opener=lambda *a,**k:Wrong(raw))
    with pytest.raises(ValueError,match='checksum'):data.download_prefix(tmp_path/'bad','test.txt',opener=lambda *a,**k:Response(b'X'*len(raw)))
    assert not list((tmp_path/'bad').glob('*.part'))


def test_bpe_stories_train_only_tokenizer_and_bpb(tmp_path):
    from llm2.research.stories_worker import score
    from llm2.config import ExperimentConfig, DataConfig, ModelConfig, ResearchConfig
    from llm2.research.factory import build_backbone

    splits = {
        'train': {
            hashlib.sha256(f'Once upon a time {i}, a little cat played happily. '.encode() * 4).hexdigest():
            (f'Once upon a time {i}, a little cat played happily. '.encode() * 4)
            for i in range(6)
        },
        'cal_val': {hashlib.sha256(b'A happy fox ran home.').hexdigest(): b'A happy fox ran home.'},
        'val': {hashlib.sha256(b'xzqwv xzqwv xzqwv xzqwv xzqwv').hexdigest(): b'xzqwv xzqwv xzqwv xzqwv xzqwv'},
        'test': {hashlib.sha256(b'Another story for reserved test split.').hexdigest(): b'Another story for reserved test split.'},
    }
    bpe = data.write_bpe_dataset(tmp_path / 'bpe_data', splits, source_fingerprint='unpinned', vocab_size=320)
    assert bpe.schema == 'stories-doc-bpe-v1'
    assert bpe.tokenizer.vocab.get('xzqwv') is None if hasattr(bpe.tokenizer, 'vocab') else True
    assert bpe.tokenizer.engine.token_to_id('xzqwv') is None
    with pytest.raises(ValueError, match='pinned'):
        data.StoriesData(bpe.root, require_pinned=True)

    cfg = ExperimentConfig(
        model=ModelConfig(vocab_size=bpe.vocab_size, d_model=32, n_layers=2, n_heads=2, hidden_dim=64, max_seq_len=128),
        data=DataConfig(path=str(bpe.root), seq_len=32, batch_size=2, eval_batch_size=2),
        research=ResearchConfig(architecture='transformer'),
    )
    model = build_backbone(cfg).eval()
    res = score(model, bpe, 'val', 32, 2, torch.device('cpu'))
    expected_bytes = bpe.manifest['splits']['val']['bytes']
    assert res['bytes'] == expected_bytes
    assert abs(res['bits_per_byte'] - (res['nll'] * res['tokens']) / (expected_bytes * np.log(2.0))) < 1e-6

