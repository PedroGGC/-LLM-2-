import hashlib
import time
from urllib.request import urlopen

RUN_CAMPAIGN = True
LLM2_REVISION = "1c8ce2abb8f47d3167454b3503598854b7756dad"
LLM2_SETTINGS = {
    "phase": "qualify",
    "run_number": 25,
    "target_seconds": 7200,
    "session_seconds": 10800,
}

if RUN_CAMPAIGN:
    started_at = time.monotonic()
    print("Qualificação corrigida: 6 pilotos nas duas T4, limite externo de 10 min por caso.", flush=True)
    url = (
        "https://raw.githubusercontent.com/PedroGGC/-LLM-2-/"
        f"{LLM2_REVISION}/src/llm2/research/state_bootstrap.py"
    )
    with urlopen(url, timeout=60) as response:
        source = response.read(1024 * 1024 + 1)
    if hashlib.sha256(source).hexdigest() != "2c56657c559379137ed23d5ff7614f5c8711580c1fe0a25e695cdbfe1943147b":
        raise ValueError("Bootstrap diferente da revisão verificada.")
    namespace = {"__name__": "llm2_state_pinned_bootstrap"}
    exec(compile(source.decode("utf-8"), url, "exec"), namespace)
    try:
        LLM2_STATE_INFO = namespace["bootstrap_git"](
            LLM2_REVISION, run=True, started_at=started_at, **LLM2_SETTINGS
        )
    finally:
        if "download_file" in namespace:
            LLM2_BAIXAR = namespace["download_file"]
else:
    print("Execução desativada.")
