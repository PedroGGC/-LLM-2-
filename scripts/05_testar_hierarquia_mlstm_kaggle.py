import hashlib
import time
from urllib.request import urlopen

RUN_CAMPAIGN = True
LLM2_REVISION = "639695979ee4d328950ef93502aef6045cccf378"
LLM2_SETTINGS = {
    "run_number": 27,
    "seeds": [241],
    "target_seconds": 7200,
    "session_seconds": 10800,
}

if RUN_CAMPAIGN:
    started_at = time.monotonic()
    print("Etapa B: 4 treinos principais + triagem; duas T4, meta2h/teto3h.", flush=True)
    url = (
        "https://raw.githubusercontent.com/PedroGGC/-LLM-2-/"
        f"{LLM2_REVISION}/src/llm2/research/state_b_bootstrap.py"
    )
    with urlopen(url, timeout=60) as response:
        source = response.read(1024 * 1024 + 1)
    if hashlib.sha256(source).hexdigest() != "c60ee6f646d5c26bb80cc54c029028f414dbb9d0f45bb89ebb8ce90680e4f6e6":
        raise ValueError("Bootstrap diferente da revisão verificada.")
    namespace = {"__name__": "llm2_state_b_pinned_bootstrap"}
    exec(compile(source.decode("utf-8"), url, "exec"), namespace)
    try:
        LLM2_STATE_INFO = namespace["bootstrap_git"](
            LLM2_REVISION, run=True, started_at=started_at, **LLM2_SETTINGS
        )
    finally:
        if "download_file" in namespace:
            LLM2_BAIXAR = namespace["download_file"]
else:
    print("Treino desativado.")
