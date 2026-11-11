import hashlib
import time
from urllib.request import urlopen

RUN_CAMPAIGN = True
LLM2_REVISION = "3f71accb5413a6efe9afdffeb692cb018d250f6c"
LLM2_SETTINGS = {
    "run_number": 34,
    "session_seconds": 1800,
    "checkpoints": None,
}

if RUN_CAMPAIGN:
    started_at = time.monotonic()
    print("Geração com estado recorrente e KV cache. T4 x2, teto de 30 min.", flush=True)
    url = (
        "https://raw.githubusercontent.com/PedroGGC/-LLM-2-/"
        f"{LLM2_REVISION}/src/llm2/research/decode_bootstrap.py"
    )
    with urlopen(url, timeout=60) as response:
        source = response.read(1024 * 1024 + 1)
    if hashlib.sha256(source).hexdigest() != "0d0e0e7bbaf97bb3c86f114e20bdb2abb10cb08c93779d6893a9b478fc71ff9d":
        raise ValueError("Bootstrap diferente da revisão verificada.")

    namespace = {"__name__": "llm2_decode_bootstrap"}
    exec(compile(source.decode("utf-8"), url, "exec"), namespace)
    try:
        LLM2_DECODE_INFO = namespace["bootstrap_git"](
            LLM2_REVISION, run=True, started_at=started_at, **LLM2_SETTINGS
        )
    finally:
        if "download_file" in namespace:
            LLM2_BAIXAR = namespace["download_file"]
else:
    print("Diagnóstico desativado.")
