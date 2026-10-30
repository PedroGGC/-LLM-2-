import io
from pathlib import Path
import runpy
import urllib.request


ROOT=Path(__file__).resolve().parents[1]


def test_recompression_cell_passes_only_explicit_zip_and_deflate9(monkeypatch):
    builder=runpy.run_path(str(ROOT/"scripts/build_cycle4_release.py"))
    name="llm2-working-backup-20261030-225306-9a2303.zip"
    source=(ROOT/"src/llm2/research/kaggle_reset.py").read_bytes()
    monkeypatch.setattr(urllib.request,"urlopen",lambda *a,**k:io.BytesIO(source))
    called={}
    def inject(code,namespace):
        def backup(**kwargs):
            called.update(kwargs)
            return {"path":"/kaggle/working/backup.zip"}
        namespace["backup_working"]=backup
    namespace={"exec":inject}
    code=builder["cells"]("a"*40,replace_backup=name)["01_backup_e_limpeza_kaggle.py"]
    exec(compile(code,"backup-cell","exec"),namespace)
    assert called=={"compression":"deflate9","replace_backup":name}
    assert 'FileLink(Path(LLM2_BACKUP_INFO["path"]).name)' in code


def test_cleanup_cell_does_not_attempt_recompression(monkeypatch):
    builder=runpy.run_path(str(ROOT/"scripts/build_cycle4_release.py"))
    source=(ROOT/"src/llm2/research/kaggle_reset.py").read_bytes()
    monkeypatch.setattr(urllib.request,"urlopen",lambda *a,**k:io.BytesIO(source))
    called=[]
    def inject(code,namespace):namespace["clear_working"]=lambda digest:called.append(digest)
    code=builder["cells"]("a"*40)["01_backup_e_limpeza_kaggle.py"]
    code=code.replace("MODO = 'backup'","MODO = 'limpar'").replace('BACKUP_LOCAL_SHA256 = ""','BACKUP_LOCAL_SHA256 = "'+"c"*64+'"')
    exec(compile(code,"cleanup-cell","exec"),{"exec":inject})
    assert called==["c"*64]
