import hashlib
import io
import json
from pathlib import Path

import pytest

from llm2.research import wikitext_cache as cache

REVISION="a"*40
NAME="train-00000-of-00002.parquet"
OTHER="test-00000-of-00001.parquet"


def digest(data):return hashlib.sha256(data).hexdigest()


def test_corrupt_legacy_is_preserved_only_bad_shard_downloaded(tmp_path):
    old=tmp_path/"llm2-wikitext-pinned-cache";old.mkdir()
    (old/NAME).write_bytes(b"broken")
    (old/OTHER).write_bytes(b"valid test")
    calls=[]
    def opener(url,**kwargs):calls.append(url);return io.BytesIO(b"valid train")
    report=cache.prepare_cache(tmp_path,{NAME:digest(b"valid train"),OTHER:digest(b"valid test")},REVISION,
                               opener=opener,reserve_bytes=0)
    root=Path(report["selected_cache"])
    assert (old/NAME).read_bytes()==b"broken" and (root/NAME).read_bytes()==b"valid train"
    assert len(calls)==1 and NAME in calls[0] and REVISION in calls[0]
    assert report["files"][0]["legacy"]["sha256"]==digest(b"broken")
    assert report["files"][1]["action"]=="copy_valid_legacy"
    (old/OTHER).write_bytes(b"changed later")
    assert (root/OTHER).read_bytes()==b"valid test"  # independent copy, not a legacy hardlink
    assert not list(root.glob("*.part"))


def test_verified_cache_reused_without_network(tmp_path):
    folder=tmp_path/"llm2-wikitext-verified-cache";folder.mkdir()
    (folder/NAME).write_bytes(b"correct")
    report=cache.prepare_cache(tmp_path,{NAME:digest(b"correct")},REVISION,reserve_bytes=0,
                               opener=lambda *a,**k:pytest.fail("network not needed"))
    assert report["files"][0]["action"]=="reuse_verified"


def test_bad_download_never_published_and_no_expected_hash_changed(tmp_path):
    calls=[];expected=digest(b"real parquet")
    def bad(url,**kwargs):calls.append(url);return io.BytesIO(b"<html>gateway</html>")
    report={}
    with pytest.raises(ValueError,match="observed="):
        cache.prepare_cache(tmp_path,{NAME:expected},REVISION,opener=bad,reserve_bytes=0,report=report)
    assert len(calls)==2
    folder=tmp_path/"llm2-wikitext-verified-cache"
    assert not (folder/NAME).exists() and not list(folder.glob("*.part"))
    assert report["files"][0]["expected_sha256"]==expected


def test_network_failure_preserves_legacy(tmp_path):
    old=tmp_path/"llm2-wikitext-pinned-cache";old.mkdir();(old/NAME).write_bytes(b"old")
    def fail(*args,**kwargs):raise OSError("offline")
    with pytest.raises(OSError,match="offline"):
        cache.prepare_cache(tmp_path,{NAME:digest(b"new")},REVISION,opener=fail,reserve_bytes=0)
    assert (old/NAME).read_bytes()==b"old"
    assert not list((tmp_path/"llm2-wikitext-verified-cache").iterdir())


def test_concurrent_publication_never_overwrites(tmp_path):
    temporary=tmp_path/"own.part";temporary.write_bytes(b"valid")
    destination=tmp_path/NAME;destination.write_bytes(b"valid")
    cache.publish(temporary,destination,digest(b"valid"))
    destination.write_bytes(b"other writer")
    with pytest.raises(ValueError,match="never overwritten"):
        cache.publish(temporary,destination,digest(b"valid"))
    assert destination.read_bytes()==b"other writer"


def test_changed_verified_cache_is_preserved(tmp_path):
    folder=tmp_path/"llm2-wikitext-verified-cache";folder.mkdir();(folder/NAME).write_bytes(b"unknown")
    with pytest.raises(ValueError,match="preserved for inspection"):
        cache.prepare_cache(tmp_path,{NAME:digest(b"correct")},REVISION,reserve_bytes=0)
    assert (folder/NAME).read_bytes()==b"unknown"


def test_transfer_limit_and_unsafe_names(tmp_path):
    with pytest.raises(ValueError,match="transfer limit"):
        cache.download("test",io.BytesIO(),digest(b"abc"),opener=lambda *a,**k:io.BytesIO(b"abc"),max_bytes=2)
    with pytest.raises(ValueError,match="unsafe"):
        cache.prepare_cache(tmp_path,{"../outside.parquet":digest(b"x")},REVISION,reserve_bytes=0)


def test_specification_is_parsed_without_executing_source(tmp_path):
    source=tmp_path/"wikitext.py"
    source.write_text(f"REVISION={REVISION!r}\nFILES={ {NAME:digest(b'x')}!r}\nraise RuntimeError('must not execute')\n")
    files,revision=cache.specification(source,cache.sha_file(source))
    assert files=={NAME:digest(b"x")} and revision==REVISION
    with pytest.raises(ValueError,match="importer source"):
        cache.specification(source,"b"*64)


def test_linked_cache_is_not_followed(tmp_path):
    external=tmp_path/"external";external.mkdir();(external/NAME).write_bytes(b"preserve")
    working=tmp_path/"working";working.mkdir()
    try:(working/"llm2-wikitext-pinned-cache").symlink_to(external,target_is_directory=True)
    except OSError:pytest.skip("symlinks unavailable")
    with pytest.raises(ValueError,match="linked"):
        cache.prepare_cache(working,{NAME:digest(b"preserve")},REVISION,reserve_bytes=0)
    assert (external/NAME).read_bytes()==b"preserve"


def test_failed_preflight_receipt_survives_checkout_cleanup(tmp_path,monkeypatch):
    from llm2.research import state_b_bootstrap as bootstrap
    parent=tmp_path/"checkout";parent.mkdir()
    report={"status":"failed","error":"checksum mismatch","files":[{"name":NAME,"observed":"other"}]}
    (parent/"bootstrap.json").write_text(json.dumps(report))
    displayed=[]
    monkeypatch.setattr(bootstrap,"download_file",lambda p:displayed.append(p),raising=False)
    receipt=bootstrap.export_cache_failure(tmp_path,parent)
    (parent/"bootstrap.json").unlink();parent.rmdir()
    assert json.loads(receipt.read_text())==report and displayed==[receipt]
