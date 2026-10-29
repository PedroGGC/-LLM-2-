"""Build pinned short Kaggle cells or a verified source-ZIP release; never overwrite."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import runpy
import tempfile
import zipfile


ROOT = Path(__file__).resolve().parents[1]


def git_cell(revision, *, run=True, session_seconds=7200, run_number=17):
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("use a complete 40-character commit SHA")
    if not 300 < session_seconds <= 14400 or not isinstance(run_number, int) or run_number <= 0:
        raise ValueError("invalid session seconds/run number")
    return f'''import time
from urllib.request import urlopen

# CICLO 2: COMUNICACAO ENTRE ESCALAS x FiLM (KAGGLE T4 x2)
RUN_CAMPAIGN = {run!r}
LLM2_REVISION = {revision!r}
LLM2_SETTINGS = {{
    "run_number": {run_number},
    "session_seconds": {session_seconds},  # teto total de 2h; pode terminar antes
}}

if RUN_CAMPAIGN:
    started_at = time.monotonic()
    url = (
        "https://raw.githubusercontent.com/PedroGGC/-LLM-2-/"
        f"{{LLM2_REVISION}}/src/llm2/research/cycle2_bootstrap.py"
    )
    with urlopen(url, timeout=60) as response:
        bootstrap = response.read(1024 * 1024 + 1)
    if len(bootstrap) > 1024 * 1024:
        raise ValueError("Bootstrap maior que o limite esperado")
    namespace = {{"__name__": "llm2_cycle2_pinned_bootstrap"}}
    exec(compile(bootstrap.decode("utf-8"), url, "exec"), namespace)
    namespace["bootstrap_git"](
        LLM2_REVISION, run=True, started_at=started_at, **LLM2_SETTINGS
    )
else:
    print("Execucao desativada. Defina RUN_CAMPAIGN = True para iniciar.")
'''


def build_git(output, revision, *, session_seconds=7200):
    active = git_cell(revision, session_seconds=session_seconds)
    inactive = git_cell(revision, run=False, session_seconds=session_seconds)
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    names = ("rodar_ciclo2_kaggle_2h.py", "rodar_ciclo2_kaggle_2h.ipynb", "llm2-cycle2-git-2h.json")
    if any((output / name).exists() for name in names):
        raise FileExistsError("release destination exists; use another --output-dir")
    staging = Path(tempfile.mkdtemp(prefix=".cycle2-git-", dir=output))
    (staging / names[0]).write_text(active, encoding="utf-8")
    notebook = {"nbformat": 4, "nbformat_minor": 5,
        "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                     "language_info": {"name": "python"}},
        "cells": [
            {"id": "cycle2-guide", "cell_type": "markdown", "metadata": {}, "source": [
                "# Ciclo 2 — comunicação × FiLM\n",
                "Ative Internet e GPU T4 x2. Não é necessário anexar ZIP de fontes.\n",
                "Mude RUN_CAMPAIGN para True e execute a célula curta.\n",
                "O teto é de duas horas, incluindo preparação e exportação; pode terminar antes.\n",
                "Preserve os outputs completos além do ZIP de resultados.\n"]},
            {"id": "cycle2-run", "cell_type": "code", "metadata": {}, "execution_count": None,
             "outputs": [], "source": inactive.splitlines(keepends=True)}]}
    (staging / names[1]).write_text(json.dumps(notebook, ensure_ascii=False, indent=1)+"\n", encoding="utf-8")
    metadata = {"release": "cycle2-git-2h", "source_mode": "git", "revision": revision,
                "session_seconds": session_seconds, "export_reserve_seconds": 300,
                "protocol": "six arms x seeds 81/82/83; T256/B32/K2/2000; FP32/Inductor; no context RMS",
                "files": {name: {"sha256": hashlib.sha256((staging / name).read_bytes()).hexdigest(),
                                 "bytes": (staging / name).stat().st_size} for name in names[:2]}}
    (staging / names[2]).write_text(json.dumps(metadata, indent=2)+"\n", encoding="utf-8")
    for name in names:
        os.link(staging / name, output / name)
        (staging / name).unlink()
    staging.rmdir()
    return metadata


def build(output):
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    names = ("llm2-cycle2-source-20261029.zip", "rodar_ciclo2_kaggle.py", "rodar_ciclo2_kaggle.ipynb",
             "llm2-cycle2-release-20261029.json")
    if any((output / name).exists() for name in names):
        raise FileExistsError("release destination exists; use another --output-dir")
    staging = Path(tempfile.mkdtemp(prefix=".cycle2-release-", dir=output))
    select = runpy.run_path(str(ROOT / "scripts/package_colab.py"))["source_files"]
    with zipfile.ZipFile(staging / names[0], "x", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in select(ROOT):
            archive.write(path, arcname=path.relative_to(ROOT).as_posix())
    source = staging / names[0]
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    helper = (ROOT / "src/llm2/research/cycle2_bootstrap.py").read_text(encoding="utf-8")
    header = ("# Ciclo 2 — cole TODO este arquivo em uma célula Python do Kaggle.\n"
              "# Antes: adicione o ZIP de fontes fornecido como Input e ative Internet/T4.\n"
              "# Não use o bootstrap antigo da campanha 0016.\n"
              "RUN_CYCLE2 = True\nRUN_NUMBER = 17\nSOURCE_ZIP = None  # None: procurar automaticamente nos Inputs\n\n")
    body = helper + f"\n\nbootstrap({digest!r}, run=RUN_CYCLE2, run_number=RUN_NUMBER, archive=SOURCE_ZIP)\n"
    (staging / names[1]).write_text(header+body, encoding="utf-8")
    notebook = {"nbformat": 4, "nbformat_minor": 5,
        "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                     "language_info": {"name": "python"}},
        "cells": [
            {"id": "cycle2-guide", "cell_type": "markdown", "metadata": {}, "source": [
                "# Ciclo 2 — comunicação × FiLM\n",
                "Adicione o ZIP de fontes como Input, ative Internet e GPU T4, mude RUN_CYCLE2 para True e execute.\n",
                "Até 18 treinos seriais; piloto prévio; limite global de 2h; resultados em llm2-campaign-0017-results.zip.\n",
                "O script verifica o SHA-256 do pacote; preserve os outputs completos para guardar checkpoints.\n"]},
            {"id": "cycle2-run", "cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
             "source": (header.replace("RUN_CYCLE2 = True", "RUN_CYCLE2 = False")+body).splitlines(keepends=True)}]}
    (staging / names[2]).write_text(json.dumps(notebook, ensure_ascii=False, indent=1)+"\n", encoding="utf-8")
    with zipfile.ZipFile(source) as archive:
        if archive.testzip() is not None:
            raise ValueError("release ZIP CRC failed")
        file_count = len(archive.namelist())
    metadata = {"release": "cycle2-20261029", "source_archive_sha256": digest, "source_files": file_count,
                "files": {name: {"sha256": hashlib.sha256((staging / name).read_bytes()).hexdigest(),
                                 "bytes": (staging / name).stat().st_size} for name in names[:3]},
                "protocol": "six arms x seeds 81/82/83; T256/B32/K2/2000; FP32/Inductor; no context RMS",
                "session_seconds": 7200, "export_reserve_seconds": 300,
                "limitations": "GPU/Inductor validation occurs when the supplied T4 script runs; local CPU tests are not CUDA verification"}
    (staging / names[3]).write_text(json.dumps(metadata, indent=2)+"\n", encoding="utf-8")
    for name in names:
        os.link(staging / name, output / name)
        (staging / name).unlink()
    staging.rmdir()
    return metadata


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default=str(ROOT / "dist"))
    parser.add_argument("--revision", help="published commit SHA for a short Git-based launcher; otherwise build a ZIP release")
    args = parser.parse_args()
    print(json.dumps(build_git(args.output_dir, args.revision) if args.revision else build(args.output_dir), indent=2))
