import ast
import json
from pathlib import Path

import pytest


@pytest.mark.parametrize("name", ["colab_a100.ipynb", "kaggle_t4.ipynb"])
def test_notebook_is_valid_python_without_saved_outputs(name):
    notebook = json.loads((Path(__file__).resolve().parents[1] / "notebooks" / name).read_text(encoding="utf-8"))
    assert notebook["nbformat"] == 4
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            ast.parse("".join(cell["source"]))
            assert cell.get("execution_count") is None
            assert not cell.get("outputs")


def test_kaggle_defaults_do_not_start_download_or_training(monkeypatch):
    import subprocess
    import urllib.request

    def unexpected_action(*args, **kwargs):
        raise AssertionError("Notebook default attempted an external action")

    monkeypatch.setattr(urllib.request, "urlopen", unexpected_action)
    monkeypatch.setattr(subprocess, "run", unexpected_action)
    monkeypatch.setattr(subprocess, "Popen", unexpected_action)
    notebook = json.loads((Path(__file__).resolve().parents[1] / "notebooks/kaggle_t4.ipynb").read_text(encoding="utf-8"))
    namespace = {}
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            exec(compile("".join(cell["source"]), "kaggle_t4.ipynb", "exec"), namespace)
    assert namespace["RUN_CAMPAIGN"] is False

