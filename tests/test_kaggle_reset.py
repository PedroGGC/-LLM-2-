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


@pytest.mark.parametrize("change",["worker","space","corrupt"])
def test_replacement_refuses_before_deleting_old_zip(working,monkeypatch,change):
    from types import SimpleNamespace
    old=legacy_backup(working)
    if change=="worker":monkeypatch.setattr(reset,"active_workers",lambda:[123])
    if change=="space":monkeypatch.setattr(reset.shutil,"disk_usage",lambda *a:SimpleNamespace(free=1))
    if change=="corrupt":
        with zipfile.ZipFile(old) as archive:
            info=archive.getinfo("working/data/train.bin")
        with old.open("r+b") as stream:
            stream.seek(info.header_offset+26);lengths=stream.read(4)
            offset=info.header_offset+30+int.from_bytes(lengths[:2],"little")+int.from_bytes(lengths[2:],"little")
            stream.seek(offset);original=stream.read(1);stream.seek(offset);stream.write(bytes([original[0]^1]))
    with pytest.raises((ValueError,RuntimeError,OSError,zipfile.BadZipFile)):reset.backup_working(working,replace_backup=old.name)
    assert old.exists() and (working/"unclassified.pth").read_bytes()==b"legacy"


@pytest.mark.parametrize("change",["added","timestamp"])
def test_safe_inventory_drift_recompresses_and_reports_paths(working,capsys,change):
    import os
    old=legacy_backup(working)
    data=working/"data/train.bin"
    if change=="added":(working/"new.log").write_bytes(b"new notebook artifact")
    else:
        info=data.stat();os.utime(data,ns=(info.st_atime_ns,info.st_mtime_ns+1000000000))
    result=reset.backup_working(working,replace_backup=old.name)
    assert not old.exists() and data.read_bytes()==b"data"
    assert result["replaced_backup"]["recovery_directory"] is None
    diff=result["replaced_backup"]["inventory_difference"]
    assert not diff["missing"] and not diff["changed"]
    if change=="added":
        assert diff["added"]==["new.log"]
        with zipfile.ZipFile(result["path"]) as archive:
            assert archive.read("working/new.log")==b"new notebook artifact"
    else:assert diff["timestamps_only"]==["data/train.bin"]
    assert ("new.log" if change=="added" else "data/train.bin") in capsys.readouterr().out


@pytest.mark.parametrize("change",["changed","missing"])
def test_unique_previous_bytes_recovered_without_overwriting_current(working,change):
    old=legacy_backup(working);data=working/"data/train.bin"
    if change=="changed":data.write_bytes(b"current version")
    else:data.unlink()
    result=reset.backup_working(working,replace_backup=old.name)
    replacement=result["replaced_backup"]
    folder=working/replacement["recovery_directory"]
    assert (folder/"files/data/train.bin").read_bytes()==b"data"
    if change=="changed":assert data.read_bytes()==b"current version"
    else:assert not data.exists()
    assert replacement["recovered_files"]==["data/train.bin"]
    assert replacement["previous_payload_preserved"]
    assert (folder/"original_backup_manifest.json").is_file()
    with zipfile.ZipFile(result["path"]) as archive:
        assert archive.read("working/"+folder.name+"/files/data/train.bin")==b"data"
    assert not old.exists()
    reset.clear_working(result["sha256"],working)
    assert not list(working.iterdir())


def test_file_replaced_with_external_link_recovers_old_bytes_without_following_link(working,tmp_path):
    old=legacy_backup(working);data=working/"data/train.bin"
    outside=tmp_path/"outside";outside.write_bytes(b"external stays private")
    data.unlink()
    try:data.symlink_to(outside)
    except OSError:pytest.skip("symlink privilege unavailable")
    result=reset.backup_working(working,replace_backup=old.name)
    folder=working/result["replaced_backup"]["recovery_directory"]
    assert (folder/"files/data/train.bin").read_bytes()==b"data"
    reset.clear_working(result["sha256"],working)
    assert outside.read_bytes()==b"external stays private"


def test_recovery_needs_space_before_any_deletion(working,monkeypatch,capsys):
    from types import SimpleNamespace
    old=legacy_backup(working);(working/"data/train.bin").unlink()
    monkeypatch.setattr(reset.shutil,"disk_usage",lambda *a:SimpleNamespace(free=32*1024**2))
    with pytest.raises(OSError,match="recuperação"):reset.backup_working(working,replace_backup=old.name)
    assert old.exists() and not list(working.glob("*recovery-*"))
    assert "data/train.bin" in capsys.readouterr().out


def test_recovery_failure_keeps_old_zip_and_current_data(working,monkeypatch):
    old=legacy_backup(working);data=working/"data/train.bin";data.write_bytes(b"current")
    original=Path.open
    def fail_recovery(path,*args,**kwargs):
        if args and args[0]=="xb" and any(p.startswith(".llm2-recovery-writing-") for p in path.parts):
            raise OSError("injected recovery failure")
        return original(path,*args,**kwargs)
    monkeypatch.setattr(Path,"open",fail_recovery)
    with pytest.raises(OSError,match="injected recovery"):reset.backup_working(working,replace_backup=old.name)
    assert old.exists() and data.read_bytes()==b"current"
    assert not list(working.glob("*recovery-*"))


def test_concurrent_change_during_recovery_blocks_deletion(working,monkeypatch):
    old=legacy_backup(working);data=working/"data/train.bin";data.write_bytes(b"current")
    original=reset.preserve_previous_versions
    def change(*a,**k):
        folder=original(*a,**k);data.write_bytes(b"concurrent");return folder
    monkeypatch.setattr(reset,"preserve_previous_versions",change)
    with pytest.raises(RuntimeError,match="durante a verificação"):reset.backup_working(working,replace_backup=old.name)
    assert old.exists() and data.read_bytes()==b"concurrent"
    assert not list(working.glob("*recovery-*"))


def test_compression_failure_keeps_recovered_previous_versions_for_retry(working,monkeypatch):
    old=legacy_backup(working);data=working/"data/train.bin";data.write_bytes(b"current")
    with monkeypatch.context() as patch:
        patch.setattr(zipfile.ZipFile,"write",lambda *a,**k:(_ for _ in ()).throw(OSError("compression failed")))
        with pytest.raises(OSError,match="compression failed"):reset.backup_working(working,replace_backup=old.name)
    assert not old.exists() and data.read_bytes()==b"current"
    folders=list(working.glob("llm2-backup-recovery-*"));assert len(folders)==1
    assert (folders[0]/"files/data/train.bin").read_bytes()==b"data"
    result=reset.backup_working(working,replace_backup=old.name)
    with zipfile.ZipFile(result["path"]) as archive:
        assert archive.read("working/"+folders[0].name+"/files/data/train.bin")==b"data"


def test_recovered_destination_hash_checked_before_old_backup_deleted(working,monkeypatch):
    old=legacy_backup(working);(working/"data/train.bin").write_bytes(b"current")
    original=reset.sha256_file
    monkeypatch.setattr(reset,"sha256_file",lambda path:"0"*64
        if any(p.startswith(".llm2-recovery-writing-") for p in Path(path).parts) else original(path))
    with pytest.raises(ValueError,match="recovered version hash"):reset.backup_working(working,replace_backup=old.name)
    assert old.exists() and not list(working.glob("*recovery-*"))


def test_cleanup_ignores_only_timestamps_and_backup_can_be_reused(working):
    import os
    result=reset.backup_working(working)
    data=working/"data/train.bin";info=data.stat()
    os.utime(data,ns=(info.st_atime_ns,info.st_mtime_ns+1000000000))
    assert reset.backup_working(working)["reused"]
    assert reset.clear_working(result["sha256"],working)["status"]=="cleaned"


def test_cleanup_still_blocks_new_payload_and_reports_it(working,capsys):
    result=reset.backup_working(working)
    (working/"new-checkpoint.pt").write_bytes(b"must back up first")
    with pytest.raises(RuntimeError,match="Inventário mudou"):reset.clear_working(result["sha256"],working)
    assert "new-checkpoint.pt" in capsys.readouterr().out
    assert Path(result["path"]).exists() and (working/"new-checkpoint.pt").exists()


def test_stream_verification_detects_bad_crc_without_separate_testzip_pass(working,monkeypatch):
    old=legacy_backup(working)
    with zipfile.ZipFile(old) as archive:info=archive.getinfo("working/data/train.bin")
    with old.open("r+b") as stream:
        stream.seek(info.header_offset+26);lengths=stream.read(4)
        stream.seek(info.header_offset+30+int.from_bytes(lengths[:2],"little")+int.from_bytes(lengths[2:],"little"))
        stream.write(b"X")
    monkeypatch.setattr(zipfile.ZipFile,"testzip",lambda *a:pytest.fail("unnecessary full CRC pass"))
    with pytest.raises(zipfile.BadZipFile,match="CRC"):reset.read_backup(old)


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
