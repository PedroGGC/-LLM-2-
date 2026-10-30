import json
from pathlib import Path
import zipfile
import runpy

import pytest

from llm2.research import cycle3_short_bootstrap as bootstrap


def result(path,*,protocol='same',complete=False):
    with zipfile.ZipFile(path,'w') as archive:
        archive.writestr('report/report.json',json.dumps({'protocol_id':protocol,'rows':[
            {'arm':'anchor_flrm','seed':103,'checkpoint_model_sha256':'a'*64},
            {'arm':'flops_transformer','seed':103,'checkpoint_model_sha256':'b'*64}]}))
        if complete:
            archive.writestr('campaign.json',json.dumps({'operational_mode':'one_hour_batch','max_new_runs':2,
                'runs':[{'arm':'anchor_flrm','seed':103,'status':'complete','checkpoint_model_sha256':'a'*64},
                        {'arm':'flops_transformer','seed':103,'status':'complete','checkpoint_model_sha256':'b'*64}]}))


def test_public_discovery_survives_an_invalid_old_zip(tmp_path):
    (tmp_path/'llm2-campaign-0018-results.zip').write_bytes(b'not a zip')
    valid=tmp_path/'llm2-campaign-0019-results.zip';result(valid)
    assert bootstrap.previous_results(tmp_path,'same')==[valid]


@pytest.mark.parametrize('payload',[b'',b'not a zip',b'PK\x03\x04truncated'])
def test_legacy_bad_zip_is_reported_preserved_and_does_not_hide_valid_history(tmp_path,payload,capsys):
    broken=tmp_path/'llm2-campaign-0018-results.zip';broken.write_bytes(payload)
    valid=tmp_path/'llm2-campaign-0019-results.zip';result(valid)
    audit=[]
    assert bootstrap.previous_results(tmp_path,'same',input_root=tmp_path/'inputs',audit=audit)==[valid]
    assert broken.read_bytes()==payload
    assert str(broken) in capsys.readouterr().out
    assert audit[0]['status']=='ignored_legacy_invalid' and audit[0]['path']==str(broken)


def test_recent_broken_history_is_not_silently_treated_as_empty(tmp_path):
    broken=tmp_path/'llm2-campaign-0020-results.zip';broken.write_bytes(b'broken')
    with pytest.raises(ValueError,match='0020') as error:
        bootstrap.previous_results(tmp_path,'same',input_root=tmp_path/'inputs')
    assert str(broken) in str(error.value)


def test_complete_snapshot_covers_unreadable_final_of_same_attempt(tmp_path):
    (tmp_path/'llm2-campaign-0020-results.zip').write_bytes(b'broken final')
    snapshot=tmp_path/'llm2-campaign-0020-after-02-results.zip';result(snapshot,complete=True)
    audit=[]
    assert bootstrap.previous_results(tmp_path,'same',input_root=tmp_path/'inputs',audit=audit)==[snapshot]
    assert any(r['status']=='recovered_from_complete_snapshot' for r in audit)


def test_incomplete_snapshot_cannot_cover_unknown_recent_results(tmp_path):
    (tmp_path/'llm2-campaign-0020-results.zip').write_bytes(b'broken final')
    snapshot=tmp_path/'llm2-campaign-0020-after-01-results.zip';result(snapshot)
    with pytest.raises(ValueError,match='0020'):
        bootstrap.previous_results(tmp_path,'same',input_root=tmp_path/'inputs')


@pytest.mark.parametrize('payload',['not json','[]'])
def test_invalid_legacy_report_does_not_crash_discovery(tmp_path,payload):
    path=tmp_path/'llm2-campaign-0019-results.zip'
    with zipfile.ZipFile(path,'w') as archive:archive.writestr('report/report.json',payload)
    audit=[]
    assert bootstrap.previous_results(tmp_path,'same',input_root=tmp_path/'inputs',audit=audit)==[]
    assert audit[0]['status']=='ignored_legacy_invalid'


@pytest.mark.parametrize('metadata',['[]','{"runs": null}'])
def test_invalid_legacy_metadata_is_reported_without_unhandled_type_errors(tmp_path,metadata):
    path=tmp_path/'llm2-campaign-0019-results.zip';result(path)
    with zipfile.ZipFile(path,'a') as archive:archive.writestr('campaign.json',metadata)
    audit=[]
    assert bootstrap.previous_results(tmp_path,'same',input_root=tmp_path/'inputs',audit=audit)==[]
    assert audit[0]['status']=='ignored_legacy_invalid'


def test_complete_snapshot_must_include_new_rows_in_cumulative_history(tmp_path):
    (tmp_path/'llm2-campaign-0020-results.zip').write_bytes(b'broken')
    snapshot=tmp_path/'llm2-campaign-0020-after-02-results.zip';result(snapshot,complete=True)
    with zipfile.ZipFile(snapshot,'a') as archive:
        archive.writestr('history/report.json',json.dumps({'protocol_id':'same','rows':[]}))
    with pytest.raises(ValueError,match='0020'):
        bootstrap.previous_results(tmp_path,'same',input_root=tmp_path/'inputs')


def test_unrelated_protocol_does_not_need_session_metadata(tmp_path):
    path=tmp_path/'llm2-campaign-0030-results.zip';result(path,protocol='other')
    audit=[]
    assert bootstrap.previous_results(tmp_path,'same',input_root=tmp_path/'inputs',audit=audit)==[]
    assert audit[0]['status']=='different_protocol'


def test_discovery_preserves_crc_failure_without_aborting_legacy_scan(tmp_path):
    path=tmp_path/'llm2-campaign-0019-results.zip';result(path)
    path.write_bytes(path.read_bytes().replace(b'"same"',b'"sXme"'))
    audit=[]
    assert bootstrap.previous_results(tmp_path,'same',input_root=tmp_path/'inputs',audit=audit)==[]
    assert audit[0]['status']=='ignored_legacy_invalid' and 'CRC' in audit[0]['reason']


def test_recovery_delivery_uses_new_number_in_script_and_notebook(tmp_path):
    module=runpy.run_path(str(Path(__file__).resolve().parents[1]/'scripts/build_cycle3_short_release.py'))
    metadata=module['build'](tmp_path,'a'*40,run_number=21)
    assert metadata['run_number']==21
    assert '"run_number": 21' in (tmp_path/'rodar_ciclo3_kaggle_1h.py').read_text()
    notebook=json.loads((tmp_path/'rodar_ciclo3_kaggle_1h.ipynb').read_text(encoding='utf-8'))
    assert '"run_number": 21' in ''.join(notebook['cells'][-1]['source'])


def test_bad_identifier_in_metadata_does_not_raise_unhandled_type_error(tmp_path):
    path=tmp_path/'llm2-campaign-0019-results.zip';result(path)
    with zipfile.ZipFile(path,'a') as archive:
        archive.writestr('campaign.json',json.dumps({'operational_mode':'one_hour_batch','max_new_runs':2,
            'runs':[{'arm':[],'seed':103,'status':'complete'},{'arm':'other','seed':103,'status':'complete'}]}))
    audit=[]
    assert bootstrap.previous_results(tmp_path,'same',input_root=tmp_path/'inputs',audit=audit)==[]
    assert audit[0]['status']=='ignored_legacy_invalid'
