import hashlib
import time
from urllib.request import urlopen

RUN_CAMPAIGN = True
LLM2_REVISION = "ffc6fb832c756771f92bb3212294813e7109e10f"
LLM2_SETTINGS = {
    "phase": "qualify",
    "run_number": 24,
    "target_seconds": 7200,
    "session_seconds": 10800,
}

if RUN_CAMPAIGN:
    started_at = time.monotonic()
    print("Etapa A: selecionar plataforma na T4; os 12 treinos ficam para outra sessão.", flush=True)
    url = (
        "https://raw.githubusercontent.com/PedroGGC/-LLM-2-/"
        f"{LLM2_REVISION}/src/llm2/research/state_bootstrap.py"
    )
    with urlopen(url, timeout=60) as response:
        source = response.read(1024 * 1024 + 1)
    if hashlib.sha256(source).hexdigest() != "045fb1d3823b82c5ee164054ffc4ab8b6f7757ec47d21679e41152b72041016e":
        raise ValueError("Bootstrap diferente da revisão verificada.")
    namespace = {"__name__": "llm2_state_pinned_bootstrap"}
    exec(compile(source.decode("utf-8"), url, "exec"), namespace)
    LLM2_STATE_INFO = namespace["bootstrap_git"](
        LLM2_REVISION, run=True, started_at=started_at, **LLM2_SETTINGS
    )
else:
    print("Execução desativada.")
