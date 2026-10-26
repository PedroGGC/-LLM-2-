import ast
import json
from pathlib import Path


def test_colab_notebook_is_valid_python():
    notebook = json.loads((Path(__file__).resolve().parents[1] / "notebooks/colab_a100.ipynb").read_text(encoding="utf-8"))
    assert notebook["nbformat"] == 4
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            ast.parse("".join(cell["source"]))

