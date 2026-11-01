import hashlib
import time
from urllib.request import urlopen

RUN_CAMPAIGN = True
LLM2_REVISION = "55dec8e7b4d10c89722c7a0a9982b621daf74837"
LLM2_SETTINGS = {
    "run_number": 28,
    "session_seconds": 7200,
}

if RUN_CAMPAIGN:
    started_at = time.monotonic()
    print("Diagnóstico0027: 11 condições, duas T4, sem treino novo; limite2h.", flush=True)
    url = (
        "https://raw.githubusercontent.com/PedroGGC/-LLM-2-/"
        f"{LLM2_REVISION}/src/llm2/research/state_diagnostic_bootstrap.py"
    )
    with urlopen(url, timeout=60) as response:
        source = response.read(1024 * 1024 + 1)
    if hashlib.sha256(source).hexdigest() != "ef5405deed514cd11614595a3bd2da21784baa177e498200cf009bf483fdfe5e":
        raise ValueError("Bootstrap diferente da revisão verificada.")

    namespace = {"__name__": "llm2_state_diagnostic_bootstrap"}
    exec(compile(source.decode("utf-8"), url, "exec"), namespace)
    try:
        LLM2_DIAG_INFO = namespace["bootstrap_git"](
            LLM2_REVISION, run=True, started_at=started_at, **LLM2_SETTINGS
        )
    finally:
        if "download_file" in namespace:
            LLM2_BAIXAR = namespace["download_file"]
else:
    print("Diagnóstico desativado.")
