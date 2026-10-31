import io
from pathlib import Path
import runpy
import time
import urllib.request
import pytest
from llm2.research import cycle5_bootstrap

ROOT=Path(__file__).resolve().parents[1]


def test_two_cells_build_inert_and_exclusive(tmp_path,monkeypatch):
    builder=runpy.run_path(str(ROOT/"scripts/build_cycle5_release.py"))
    monkeypatch.setattr(urllib.request,"urlopen",lambda *a,**k:pytest.fail("unexpected network"))
    for name,code in builder["cells"]("a"*40,active=False).items():exec(compile(code,name,"exec"),{})
    output=tmp_path/"release";manifest=builder["build"](output,"a"*40)
    assert len(manifest["files"])==2 and len(list(output.glob("*.py")))==2
    assert manifest["primary_full_evaluations"]==48 and manifest["jobs"]==12
    with pytest.raises(FileExistsError):builder["build"](output,"a"*40)


def test_cleanup_cell_uses_verified_local_hashes_and_narrow_cleanup(monkeypatch):
    builder=runpy.run_path(str(ROOT/"scripts/build_cycle5_release.py"));called={}
    def fetch(url,**kwargs):return io.BytesIO((ROOT/"src/llm2/research"/url.rsplit("/",1)[-1]).read_bytes())
    monkeypatch.setattr(urllib.request,"urlopen",fetch)
    def inject(code,namespace):
        if code.co_filename.endswith("kaggle_reset.py"):
            namespace.update(active_workers=lambda:[],root_directory=lambda p:p)
        else:
            def clean(number,results,checkpoints,**kwargs):
                called.update(number=number,results=results,checkpoints=checkpoints,kwargs=kwargs)
            namespace["clean_incrementals"]=clean
    code=builder["cells"]("a"*40)["01_limpar_zips_redundantes_0023.py"]
    exec(compile(code,"cleanup-cell","exec"),{"exec":inject})
    assert called["number"]==23 and called["results"]==builder["RESULTS_SHA"]
    assert called["checkpoints"]==builder["CHECKPOINTS_SHA"]
    assert called["kwargs"]["guard"]()==[]


def test_new_train_cell_fresh_clock_and_two_hour_target(monkeypatch):
    builder=runpy.run_path(str(ROOT/"scripts/build_cycle5_release.py"));called={}
    payload=(ROOT/"src/llm2/research/cycle5_bootstrap.py").read_bytes()
    monkeypatch.setattr(urllib.request,"urlopen",lambda *a,**k:io.BytesIO(payload))
    def inject(code,namespace):namespace["bootstrap_git"]=lambda rev,**kwargs:called.update(revision=rev,**kwargs)
    before=time.monotonic()
    exec(compile(builder["cells"]("a"*40)["02_rodar_controle_profundidade.py"],"train-cell","exec"),{"exec":inject})
    assert called["run_number"]==24 and called["target_seconds"]==7200 and called["session_seconds"]==10800
    assert before<=called["started_at"]<=time.monotonic()


@pytest.mark.parametrize("settings",[{"run_number":23},{"session_seconds":14400},{"target_seconds":3600},{"started_at":float("nan")}])
def test_invalid_bootstrap_settings_never_access_network(monkeypatch,settings):
    monkeypatch.setattr(cycle5_bootstrap,"urlopen",lambda *a,**k:pytest.fail("unexpected network"))
    with pytest.raises(ValueError):cycle5_bootstrap.bootstrap_git("a"*40,run=True,**settings)
