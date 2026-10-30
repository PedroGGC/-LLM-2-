import hashlib
import json
from pathlib import Path
import zipfile

import pytest
from llm2.research import kaggle_reset as reset


@pytest.fixture
def working(tmp_path,monkeypatch):
    monkeypatch.setattr(reset,"active_workers",lambda:[])
    root=tmp_path/"working";root.mkdir()
    for step in (1,2):
        folder=root/f"llm2-campaign-0022/runs/one/checkpoints/step-{step:08d}"
        folder.mkdir(parents=True)
        (folder/"model.safetensors").write_bytes(f"weights-{step}".encode())
        (folder/"training_state.pt").write_bytes(b"opaque pickle, never load")
        meta={"files":{n:reset.sha256_file(folder/n) for n in ("model.safetensors","training_state.pt")}}
        (folder/"metadata.json").write_text(json.dumps(meta))
    (root/"data").mkdir();(root/"data/train.bin").write_bytes(b"data")
    (root/"unclassified.pth").write_bytes(b"legacy")
    (root/"llm2-campaign-0022-results.zip").write_bytes(b"raw legacy file, preserve unchanged")
    return root


def test_backup_latest_and_verified_clear_preserve_root_external_and_data(working,tmp_path):
    external=tmp_path/"external";external.mkdir();(external/"keep").write_text("outside")
    receipt=reset.backup_working(working)
    path=Path(receipt["path"]);manifest=reset.read_backup(path,expected_root=working.resolve())
    assert manifest["next_run_number"]==23 and len(manifest["verified_latest_checkpoints"])==1
    with zipfile.ZipFile(path) as z:
        assert z.testzip() is None
        assert "working/data/train.bin" in z.namelist()
        assert "working/unclassified.pth" in z.namelist()
        assert not any("step-00000001/" in n for n in z.namelist())
        assert any("step-00000002/training_state.pt" in n for n in z.namelist())
    reused=reset.backup_working(working);assert reused["reused"] and reused["path"]==receipt["path"]
    result=reset.clear_working(receipt["sha256"],working)
    assert working.is_dir() and not list(working.iterdir())
    assert result["status"]=="cleaned" and result["next_run_number"]==23
    assert (external/"keep").read_text()=="outside"


@pytest.mark.parametrize("change",["wrong_hash","changed","new_file","worker"])
def test_purge_refuses_without_matching_external_copy_and_stable_inventory(working,monkeypatch,change):
    receipt=reset.backup_working(working);digest=receipt["sha256"]
    if change=="wrong_hash":digest="0"*64
    if change=="changed":(working/"data/train.bin").write_bytes(b"changed")
    if change=="new_file":(working/"new-training-state.pt").write_bytes(b"new")
    if change=="worker":monkeypatch.setattr(reset,"active_workers",lambda:[123])
    with pytest.raises((ValueError,RuntimeError)):reset.clear_working(digest,working)
    assert (working/"data/train.bin").exists() and Path(receipt["path"]).exists()


def test_unverified_checkpoint_group_is_conservatively_preserved(working):
    latest=working/"llm2-campaign-0022/runs/one/checkpoints/step-00000002/model.safetensors"
    latest.write_bytes(b"changed versus metadata")
    receipt=reset.backup_working(working)
    with zipfile.ZipFile(receipt["path"]) as z:
        assert any("step-00000001/model" in n for n in z.namelist())
        assert any("step-00000002/model" in n for n in z.namelist())
    assert receipt["warnings"]
    manifest=reset.read_backup(receipt["path"])
    assert manifest["verified_latest_checkpoints"][0].endswith("step-00000001")


def test_backup_insufficient_space_does_not_delete(working,monkeypatch):
    from collections import namedtuple
    disk=namedtuple("disk","total used free")
    monkeypatch.setattr(reset.shutil,"disk_usage",lambda *a:disk(100,99,1))
    with pytest.raises(OSError,match="Espaço"):reset.backup_working(working)
    assert (working/"unclassified.pth").exists()
    assert not list(working.glob("llm2-working-backup-*.zip"))


def test_symlink_definitions_never_followed_and_purge_unlinks_only(working,tmp_path):
    external=tmp_path/"outside";external.mkdir();(external/"secret").write_text("keep")
    link=working/"external"
    try:link.symlink_to(external,target_is_directory=True)
    except OSError:pytest.skip("symlink privilege unavailable on Windows")
    receipt=reset.backup_working(working)
    manifest=reset.read_backup(receipt["path"])
    assert manifest["inventory"]["external"]["kind"]=="symlink"
    with zipfile.ZipFile(receipt["path"]) as z:assert not any("secret" in n for n in z.namelist())
    reset.clear_working(receipt["sha256"],working)
    assert (external/"secret").read_text()=="keep"


def test_linked_metadata_is_not_read_as_checkpoint_metadata(working,tmp_path):
    target=tmp_path/"external-metadata";target.write_text("not JSON and never read")
    meta=working/"llm2-campaign-0022/runs/one/checkpoints/step-00000002/metadata.json"
    meta.unlink()
    try:meta.symlink_to(target)
    except OSError:pytest.skip("symlink privilege unavailable on Windows")
    receipt=reset.backup_working(working)
    assert receipt["warnings"]
    assert reset.read_backup(receipt["path"])["inventory"][meta.relative_to(working).as_posix()]["kind"]=="symlink"


def legacy_backup(working):
    """Build the old stored layout, including the missing compression field."""
    receipt=reset.backup_working(working,compression="stored")
    path=Path(receipt["path"])
    with zipfile.ZipFile(path) as archive:contents={n:archive.read(n) for n in archive.namelist()}
    manifest=json.loads(contents["backup_manifest.json"]);manifest.pop("compression")
    contents["backup_manifest.json"]=json.dumps(manifest).encode()
    with zipfile.ZipFile(path,"w",compression=zipfile.ZIP_STORED) as archive:
        for name,value in contents.items():archive.writestr(name,value)
    return path


def test_recompress_legacy_backup_real_deflate_and_retry_without_nesting(working):
    (working/"data/train.bin").write_bytes(b"compressible prepared data\n"*80000)
    old=legacy_backup(working);before=reset.inventory(working,exclude=(old,));old_size=old.stat().st_size
    receipt=reset.backup_working(working,replace_backup=old.name)
    new=Path(receipt["path"])
    assert not old.exists() and new.stat().st_size<old_size/20
    assert reset.inventory(working,exclude=(new,))==before
    assert receipt["compression"]=="deflate9" and receipt["replaced_backup"]["name"]==old.name
    with zipfile.ZipFile(new) as archive:
        assert all(i.compress_type==zipfile.ZIP_DEFLATED for i in archive.infolist())
        assert not any(old.name in n for n in archive.namelist())
    assert reset.backup_working(working,replace_backup=old.name)["reused"]
    assert reset.clear_working(receipt["sha256"],working)["status"]=="cleaned"


@pytest.mark.parametrize("change",["changed","missing","worker","space","corrupt"])
def test_replacement_refuses_before_deleting_old_zip(working,monkeypatch,change):
    from types import SimpleNamespace
    old=legacy_backup(working)
    if change=="changed":(working/"data/train.bin").write_bytes(b"new data")
    if change=="missing":(working/"data/train.bin").unlink()
    if change=="worker":monkeypatch.setattr(reset,"active_workers",lambda:[123])
    if change=="space":monkeypatch.setattr(reset.shutil,"disk_usage",lambda *a:SimpleNamespace(free=1))
    if change=="corrupt":
        with zipfile.ZipFile(old) as archive:
            info=archive.getinfo("working/data/train.bin")
        with old.open("r+b") as stream:
            stream.seek(info.header_offset+26);lengths=stream.read(4)
            offset=info.header_offset+30+int.from_bytes(lengths[:2],"little")+int.from_bytes(lengths[2:],"little")
            stream.seek(offset);original=stream.read(1);stream.seek(offset);stream.write(bytes([original[0]^1]))
    with pytest.raises((ValueError,RuntimeError,OSError)):reset.backup_working(working,replace_backup=old.name)
    assert old.exists() and (working/"unclassified.pth").read_bytes()==b"legacy"


@pytest.mark.parametrize("name",["../outside.zip",r"..\outside.zip","*.zip","/kaggle/input/backup.zip"])
def test_replacement_target_cannot_escape_working(working,name):
    with pytest.raises(ValueError):reset.backup_working(working,replace_backup=name)
    assert (working/"data/train.bin").exists()


def test_replacement_refuses_link_and_directory(working,tmp_path):
    name="llm2-working-backup-20261030-225306-9a2303.zip"
    directory=working/name;directory.mkdir()
    with pytest.raises(ValueError):reset.backup_working(working,replace_backup=name)
    directory.rmdir()
    outside=tmp_path/"outside";outside.write_bytes(b"keep")
    try:(working/name).symlink_to(outside)
    except OSError:pytest.skip("symlink privilege unavailable")
    with pytest.raises(ValueError):reset.backup_working(working,replace_backup=name)
    assert outside.read_bytes()==b"keep"


def test_direct_replacement_helper_rejects_external_target(working,tmp_path):
    outside=tmp_path/"llm2-working-backup-20261030-225306-9a2303.zip"
    outside.write_bytes(b"keep")
    with pytest.raises(ValueError):reset.remove_replaced_backup(working,outside)
    assert outside.read_bytes()==b"keep"


def test_compression_failure_removes_only_temp_and_can_retry(working,monkeypatch):
    old=legacy_backup(working);before=reset.inventory(working,exclude=(old,))
    with monkeypatch.context() as patch:
        patch.setattr(zipfile.ZipFile,"write",lambda *a,**k:(_ for _ in ()).throw(OSError("injected compression failure")))
        with pytest.raises(OSError,match="injected"):reset.backup_working(working,replace_backup=old.name)
    assert reset.inventory(working)==before and not old.exists()
    assert not list(working.glob(".llm2-backup-writing-*"))
    assert reset.backup_working(working,replace_backup=old.name)["compression"]=="deflate9"


@pytest.mark.parametrize("compression",["deflate9","lzma"])
def test_supported_compression_roundtrip(working,compression):
    result=reset.backup_working(working,compression=compression)
    assert reset.read_backup(result["path"])["compression"]==compression
    assert reset.clear_working(result["sha256"],working)["status"]=="cleaned"
