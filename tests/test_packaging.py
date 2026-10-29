"""Distribution archives must not collect local caches, credentials or model artifacts."""

from pathlib import Path
import runpy
import subprocess
import sys
import zipfile


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "package_colab.py"


def test_source_selection_excludes_local_artifacts(tmp_path):
    included = {"src/llm2/model.py", "tests/test_model.py", "configs/model.yaml",
                "notebooks/demo.ipynb", "docs/results/summary.csv", "docs/results/metadata.json",
                "requirements/tested.txt", "README.md", ".gitignore"}
    excluded = {"src/llm2/__pycache__/model.pyc", "src/llm2/__pycache__/stale.py",
                "src/llm2/local.safetensors", "scripts/.env", "scripts/debug.log",
                "notebooks/.ipynb_checkpoints/demo.ipynb", "docs/.private/note.md",
                "docs/results/export.zip", "runs/run.json", "artifacts/summary.csv",
                "data/train.bin", "src/llm2_lab.egg-info/metadata.py"}
    for name in included | excluded:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("fixture", encoding="utf-8")
    select = runpy.run_path(str(SCRIPT))["source_files"]
    actual = {p.relative_to(tmp_path).as_posix() for p in select(tmp_path)}
    assert actual == included


def test_package_is_usable_and_does_not_overwrite(tmp_path):
    output = tmp_path / "source.zip"
    result = subprocess.run([sys.executable, str(SCRIPT), "--output", str(output)],
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    original = output.read_bytes()
    with zipfile.ZipFile(output) as archive:
        assert archive.testzip() is None
        names = set(archive.namelist())
        assert {"src/llm2/model.py", "pyproject.toml", "README.md", ".gitignore"} <= names
        assert not any(n.startswith(("artifacts/", "runs/", "data/")) for n in names)
        assert not any("__pycache__" in n or ".ipynb_checkpoints" in n for n in names)
    repeated = subprocess.run([sys.executable, str(SCRIPT), "--output", str(output)],
                              capture_output=True, text=True)
    assert repeated.returncode != 0
    assert output.read_bytes() == original
