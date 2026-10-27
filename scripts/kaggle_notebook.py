"""Paste/run through the pinned bootstrap cell in docs/KAGGLE_T4_CAMPAIGN.md.

Runs ONLY in Kaggle's browser notebook. No local GPU or local environment is needed.
"""
import hashlib
import json
import os
from pathlib import Path
import re
import runpy
import subprocess
import sys
from urllib.request import urlopen

PIN = os.environ.get("LLM2_REVISION", "")
SETTINGS = globals().get("LLM2_SETTINGS", {})
MODE = SETTINGS.get("mode", "treinar")
if MODE not in ("treinar", "avaliar_salvos"):
    raise ValueError("Escolha mode='treinar' ou mode='avaliar_salvos'.")
if not re.fullmatch(r"[0-9a-f]{40}", PIN):
    raise ValueError("Informe em LLM2_REVISION o commit de 40 caracteres fornecido na entrega.")
WORK = Path("/kaggle/working")
if not WORK.is_dir():
    raise RuntimeError("Execute esta célula no site Kaggle com Internet e GPU T4x2 ativadas.")

print("="*78)
print("🚀 LLM2-Lab — comparação de modelos no Kaggle")
print("Modo:", "AVALIAR CHECKPOINTS SALVOS — SEM NOVO TREINO" if MODE == "avaliar_salvos" else "NOVA CAMPANHA DE TREINO")
print("="*78)
print("\n[Preparação 1/3] Conferir as GPUs")

import torch
required_gpus = 2 if MODE == "treinar" else 1
if not torch.cuda.is_available() or torch.cuda.device_count() < required_gpus:
    raise RuntimeError(f"Ative GPU T4 x2 nas configurações do Kaggle. Este modo precisa de {required_gpus} GPU(s).")
torch_release = tuple(int(part) for part in str(torch.__version__).split("+")[0].split(".")[:2])
if not (torch_release[0] == 2 and torch_release[1] >= 5):
    raise RuntimeError("Use uma imagem Kaggle com PyTorch CUDA >=2.5,<3; Torch não será substituído automaticamente.")
for index in range(torch.cuda.device_count()):
    print(f"  ✓ GPU {index}: {torch.cuda.get_device_name(index)}")
print("A GPU também usa CPU para coordenar operações; CPU ocupada não significa treino sem GPU.")

URL = "https://github.com/PedroGGC/-LLM-2-.git"


def checkout(revision):
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("A revisão do código precisa ter 40 caracteres hexadecimais.")
    repo = WORK / f"llm2-{revision[:12]}"
    if not repo.exists():
        subprocess.run(["git", "clone", "--no-checkout", "--single-branch",
                        "--branch", "experiment/fractal-lrm", URL, str(repo)], check=True)
        subprocess.run(["git", "-C", str(repo), "checkout", "--detach", revision], check=True)
    head = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    if head != revision:
        raise RuntimeError(f"O checkout {repo} tem outra revisão. Os experimentos antigos serão preservados.")
    subprocess.run(["git", "-C", str(repo), "diff", "--exit-code", "HEAD"], check=True)
    return repo


print("\n[Preparação 2/3] Preparar o código e as dependências do notebook")
REPO = checkout(PIN)
subprocess.run([sys.executable, "-m", "pip", "install", "--quiet",
                "-e", str(REPO)+"[dev,research]"], check=True)

# Load the pinned helper without reusing a package cached by an earlier cell.
numbered_output = runpy.run_path(str(REPO / "src/llm2/research/naming.py"))["numbered_output"]
OUTPUT = numbered_output(WORK, "llm2-review" if MODE == "avaliar_salvos" else "llm2-campaign",
                         number=SETTINGS.get("run_number"), search_roots=[Path("/kaggle/input")])
SESSION = OUTPUT.name.rsplit("-", 1)[1]
if MODE == "avaliar_salvos":
    previous = Path(SETTINGS.get("previous_campaign", ""))
    if not (previous / "campaign.json").is_file():
        raise FileNotFoundError("Não encontrei a campanha anterior. Em previous_campaign, indique a pasta dos "
                                "outputs completos salvos. O ZIP de análise sozinho não contém os pesos.")
    old = json.loads((previous / "campaign.json").read_text(encoding="utf-8"))
    original_repo = checkout(old["provenance"]["git"]["commit"])
    command = [sys.executable, str(REPO / "scripts/run_kaggle.py"), "--execute",
               "--review-campaign", str(previous), "--evaluation-root", str(original_repo),
               "--output", str(OUTPUT), "--device", "cuda", "--gpus", "0",
               "--review-steps", *map(str, SETTINGS.get("review_steps", [500, 1000, 1500, 2000]))]
    if SETTINGS.get("data"):
        command += ["--data", str(SETTINGS["data"])]
else:
    command = [sys.executable, str(REPO / "scripts/run_kaggle.py"), "--execute",
               "--output", str(OUTPUT), "--device", "cuda", "--gpus", "0", "1",
               "--steps", str(SETTINGS.get("steps", 2000)),
               "--max-seconds", str(SETTINGS.get("max_seconds", 1800)),
               "--progress-interval", str(SETTINGS.get("progress_interval", 500)),
               "--iterations", str(SETTINGS.get("iterations", 2)),
               "--models", *SETTINGS.get("models", ["flrm_compressed_grouped_no_jepa", "flrm_compressed_grouped", "baseline_small"]),
               "--seeds", *map(str, SETTINGS.get("seeds", [42]))]

    data = SETTINGS.get("data")
    inputs = SETTINGS.get("input", [])
    if data and inputs:
        raise ValueError("Escolha a pasta de dados preparados OU arquivos brutos, não ambos.")
    if data:
        command += ["--data", str(data)]
    elif inputs:
        command += ["--input", *map(str, inputs)]
    else:
        # ~1.1 MB, immutable public source revision; no large dataset download.
        corpus_revision = "6f9487a6fe5b420b7ca9afb0d7c078e37c1d1b4e"
        corpus_url = f"https://raw.githubusercontent.com/karpathy/char-rnn/{corpus_revision}/data/tinyshakespeare/input.txt"
        folder = WORK / f"llm2-input-{SESSION}"
        folder.mkdir(exist_ok=False)
        with urlopen(corpus_url, timeout=60) as response:
            content = response.read(2*1024*1024+1)
        if not content or len(content) > 2*1024*1024:
            raise ValueError("Tamanho do corpus inesperado; download limitado a 2 MiB.")
        content.decode("utf-8")
        corpus = folder / "tinyshakespeare.txt"
        corpus.write_bytes(content)
        provenance = {"url": corpus_url, "sha256": hashlib.sha256(content).hexdigest(), "bytes": len(content)}
        (folder / "source.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
        command += ["--input", str(corpus)]
        print("Corpus de demonstração:", provenance)

    if SETTINGS.get("profile", True):
        command += ["--profile", "--warmup-steps", str(SETTINGS.get("profile_warmup_steps", 10)),
                    "--measure-steps", str(SETTINGS.get("profile_measure_steps", 30))]
    if SETTINGS.get("cpu_tests", True):
        command += ["--cpu-tests"]
    if SETTINGS.get("cuda_tests", True):
        command += ["--cuda-tests"]
    if not SETTINGS.get("benchmark", True):
        command += ["--no-benchmark"]
    if not SETTINGS.get("diagnostics", True):
        command += ["--no-diagnostics"]
    if SETTINGS.get("evaluate_checkpoints"):
        command += ["--evaluate-checkpoints", *map(str, SETTINGS["evaluate_checkpoints"])]
print("\n[Preparação 3/3] Iniciar as etapas da campanha")
print("Revisão fixada:", PIN)
print("Nova pasta de resultados:", OUTPUT)
print("Qualquer falha interrompe o processo e preserva os logs disponíveis.")
# Child imports THIS source, irrespective of an earlier editable installation in the notebook.
env = os.environ.copy()
env["PYTHONPATH"] = str(REPO / "src")
env["PYTHONUNBUFFERED"] = "1"
env["PYTHONIOENCODING"] = "utf-8"
subprocess.run(command, cwd=REPO, env=env, check=True)
archive = OUTPUT.parent / f"{OUTPUT.name}-results.zip"
print("\n✅ CONCLUÍDO. Baixe este ZIP e envie para análise:")
print(archive)
print("Os pesos permanecem na pasta original. Salve uma versão do Kaggle com os outputs para preservá-los.")
try:
    from IPython.display import display, FileLink
    display(FileLink(str(archive)))
except ImportError:
    pass
