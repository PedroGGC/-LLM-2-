import hashlib
import time
from urllib.request import urlopen

RUN_CAMPAIGN = True
LLM2_REVISION = "0b16bf144b788df8b899cf6368463a2143fd7bff"
LLM2_SETTINGS = {
    "run_number": 36,
    "session_seconds": 1800,
    "optimize": True,
    "checkpoints": None,
}

if RUN_CAMPAIGN:
    started_at = time.monotonic()
    print("Perfil e compilação da geração. T4 x2, teto de 30 min.", flush=True)
    url = (
        "https://raw.githubusercontent.com/PedroGGC/-LLM-2-/"
        f"{LLM2_REVISION}/src/llm2/research/decode_bootstrap.py"
    )
    with urlopen(url, timeout=60) as response:
        source = response.read(1024 * 1024 + 1)
    if hashlib.sha256(source).hexdigest() != "1cfb61dedb20662ff11ababe310bd476e279177e141f836af29b2f9470e4ee9e":
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
