import hashlib
import time
from urllib.request import urlopen

RUN_CAMPAIGN = True
LLM2_REVISION = "77b2058e2129f7fcea6f3bf96b4fca4420dddbf9"
LLM2_SETTINGS = {
    "run_number": 29,
    "session_seconds": 7200,
}

if RUN_CAMPAIGN:
    started_at = time.monotonic()
    print("Cinco controles, seed251, 6000passos; duas T4, limite2h.", flush=True)
    url = (
        "https://raw.githubusercontent.com/PedroGGC/-LLM-2-/"
        f"{LLM2_REVISION}/src/llm2/research/state_readout_bootstrap.py"
    )
    with urlopen(url, timeout=60) as response:
        source = response.read(1024 * 1024 + 1)
    if hashlib.sha256(source).hexdigest() != "3990b8e501c5acf2a88cb25ac13bdada78a61ab53054a6bdcc9cfc470cb8e1b0":
        raise ValueError("Bootstrap diferente da revisão verificada.")

    namespace = {"__name__": "llm2_state_readout_bootstrap"}
    exec(compile(source.decode("utf-8"), url, "exec"), namespace)
    try:
        LLM2_READOUT_INFO = namespace["bootstrap_git"](
            LLM2_REVISION, run=True, started_at=started_at, **LLM2_SETTINGS
        )
    finally:
        if "download_file" in namespace:
            LLM2_BAIXAR = namespace["download_file"]
else:
    print("Treino desativado.")
