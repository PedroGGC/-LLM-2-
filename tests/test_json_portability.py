import hashlib
import json
from pathlib import Path
import runpy

import pytest

from llm2.data import PreparedData, TextTokenizer, sha256_file, write_json
from llm2.document_data import prepare_documents
from llm2.research import cycle3_protocol
from llm2.research.wikitext import FILES, REVISION, TRAIN_EXCLUSIONS


BYTE_TOKENIZER_LF = b'{\n  "engine": null,\n  "kind": "byte"\n}\n'
BYTE_TOKENIZER_SHA256 = 'f88f84a387d00bb8027db376d50fff07c86c4ee137f1678351521fefe0f8d252'


@pytest.mark.parametrize('default_newline', ['\n', '\r\n'])
def test_tokenizer_json_is_identical_with_windows_and_linux_defaults(tmp_path, monkeypatch, default_newline):
    original = Path.write_text

    def platform_write_text(path, text, **kwargs):
        kwargs.setdefault('newline', default_newline)
        return original(path, text, **kwargs)

    monkeypatch.setattr(Path, 'write_text', platform_write_text)
    output = tmp_path/'tokenizer.json'
    TextTokenizer().save(output)
    assert output.read_bytes() == BYTE_TOKENIZER_LF
    assert hashlib.sha256(output.read_bytes()).hexdigest() == BYTE_TOKENIZER_SHA256


def test_json_keeps_unicode_and_text_newlines_without_platform_translation(tmp_path):
    output = tmp_path/'metadata.json'
    value = {'z': 'Ω 🧪\r\nnext', 'a': 1}
    write_json(output, value)
    assert output.read_bytes() == (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2)+'\n').encode('utf-8')
    assert json.loads(output.read_bytes()) == value


@pytest.fixture
def document_data(tmp_path):
    raw = tmp_path/'documents.jsonl'
    raw.write_text(''.join(json.dumps({'id':split,'split':split,'text':f'{split} article\nΩ\n'})+'\n'
                           for split in ('train','val','test')),encoding='utf-8')
    output = tmp_path/'prepared'
    prepare_documents([raw],output)
    return output


def legacy_crlf_dataset(path):
    tokenizer = path/'tokenizer.json'
    tokenizer.write_bytes(BYTE_TOKENIZER_LF.replace(b'\n',b'\r\n'))
    manifest = json.loads((path/'manifest.json').read_text(encoding='utf-8'))
    manifest['files']['tokenizer.json'] = sha256_file(tokenizer)
    payload = (json.dumps(manifest,ensure_ascii=False,sort_keys=True,indent=2)+'\n').replace('\n','\r\n')
    (path/'manifest.json').write_bytes(payload.encode('utf-8'))


def test_old_crlf_dataset_still_loads_strictly_without_migration(document_data):
    legacy_crlf_dataset(document_data)
    before = {p.name:p.read_bytes() for p in document_data.iterdir()}
    data = PreparedData(document_data)
    assert data.fingerprint == hashlib.sha256(before['manifest.json']).hexdigest()
    assert data.tokenizer.encode('Ω\n') == list('Ω\n'.encode('utf-8'))
    assert before == {p.name:p.read_bytes() for p in document_data.iterdir()}
    # The integrity check remains exact, including the old CRLF tokenizer bytes.
    (document_data/'tokenizer.json').write_bytes(BYTE_TOKENIZER_LF)
    with pytest.raises(ValueError,match='dataset checksum mismatch: tokenizer.json'):
        PreparedData(document_data)


def test_cycle3_names_the_mismatched_hash_without_accepting_legacy_bytes(document_data,monkeypatch):
    manifest = json.loads((document_data/'manifest.json').read_text(encoding='utf-8'))
    manifest['corpus'] = {'revision':REVISION,'shard_sha256':FILES,
                          'requested_train_bytes':cycle3_protocol.TRAIN_BYTES,
                          'training_overlap_exclusions':sorted(TRAIN_EXCLUSIONS)}
    write_json(document_data/'manifest.json',manifest)
    # Substitute only corpus fixture hashes; exercise all strict production checks.
    monkeypatch.setattr(cycle3_protocol,'DATA_HASHES',dict(manifest['files']))
    assert cycle3_protocol.verify_dataset(document_data).manifest['files']['tokenizer.json']==BYTE_TOKENIZER_SHA256
    legacy_crlf_dataset(document_data)
    with pytest.raises(ValueError,match='pinned Cycle 3') as error:
        cycle3_protocol.verify_dataset(document_data)
    differences = json.loads(str(error.value).split('content: ',1)[1])
    assert differences == [{'field':'files.tokenizer.json','expected':BYTE_TOKENIZER_SHA256,
                            'observed':'a9be2688d42639c5915aa9222fcd9a389acbe6483c47ce170f3b0405e7644667'}]


def test_recovery_release_uses_a_new_campaign_number_in_both_cells(tmp_path):
    module = runpy.run_path(str(Path(__file__).resolve().parents[1]/'scripts/build_cycle3_release.py'))
    result = module['build_git'](tmp_path,'a'*40,run_number=19)
    assert result['run_number']==19
    assert '"run_number": 19' in (tmp_path/'rodar_ciclo3_kaggle_4h.py').read_text(encoding='utf-8')
    notebook = json.loads((tmp_path/'rodar_ciclo3_kaggle_4h.ipynb').read_text(encoding='utf-8'))
    assert '"run_number": 19' in ''.join(notebook['cells'][-1]['source'])
