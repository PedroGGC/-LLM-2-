import ast
import hashlib
import json
from pathlib import Path
import runpy
import zipfile

import pytest


ROOT = Path(__file__).resolve().parents[1]


def test_release_binds_launcher_to_source_and_notebook_is_inert(tmp_path, monkeypatch):
    build = runpy.run_path(str(ROOT / "scripts/build_cycle2_release.py"))["build"]
    metadata = build(tmp_path / "release")
    archive = tmp_path / "release/llm2-cycle2-source-20261029.zip"
    assert hashlib.sha256(archive.read_bytes()).hexdigest() == metadata["source_archive_sha256"]
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        assert {"scripts/run_cycle2.py", "src/llm2/research/cycle2.py", "src/llm2/research/session.py"} <= set(z.namelist())
        assert not any(n.startswith(("data/", "runs/", "artifacts/", "dist/")) for n in z.namelist())
    launcher = (tmp_path / "release/rodar_ciclo2_kaggle.py").read_text(encoding="utf-8")
    ast.parse(launcher)
    assert metadata["source_archive_sha256"] in launcher
    notebook = json.loads((tmp_path / "release/rodar_ciclo2_kaggle.ipynb").read_text(encoding="utf-8"))
    import subprocess
    import urllib.request
    def unwanted(*args, **kwargs): raise AssertionError("default notebook attempted an external action")
    monkeypatch.setattr(subprocess, "Popen", unwanted)
    monkeypatch.setattr(urllib.request, "urlopen", unwanted)
    namespace = {}
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            assert cell["execution_count"] is None and not cell["outputs"]
            exec(compile("".join(cell["source"]), "cycle2-notebook", "exec"), namespace)
    assert namespace["RUN_CYCLE2"] is False
    original = archive.read_bytes()
    with pytest.raises(FileExistsError): build(tmp_path / "release")
    assert archive.read_bytes() == original
