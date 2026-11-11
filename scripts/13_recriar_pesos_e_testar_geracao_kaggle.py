import hashlib
import time
from urllib.request import urlopen

RUN_CAMPAIGN = True
LLM2_REVISION = "94b5aa83d2aff64dd068e74e7f6fbeedc6baa5f0"
LLM2_SETTINGS = {
    "run_number": 35,
    "session_seconds": 3600,
    "rebuild": True,
}

if RUN_CAMPAIGN:
    started_at = time.monotonic()
    print("Quatro treinos e diagnóstico de geração. T4 x2, teto de 1h.", flush=True)
    url = (
        "https://raw.githubusercontent.com/PedroGGC/-LLM-2-/"
        f"{LLM2_REVISION}/src/llm2/research/decode_bootstrap.py"
    )
    with urlopen(url, timeout=60) as response:
        source = response.read(1024 * 1024 + 1)
    if hashlib.sha256(source).hexdigest() != "a23b26f4c4d9be1b7480c7109be58473d441834a588e981085731a8110e721e0":
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
    print("Campanha desativada.")
