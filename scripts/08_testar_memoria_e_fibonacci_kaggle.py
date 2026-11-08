import hashlib
import time
from urllib.request import urlopen

RUN_CAMPAIGN = True
LLM2_REVISION = "dc2888a876aa0f3718a27114ea1d3c637fadb25f"
LLM2_SETTINGS = {
    "run_number": 30,
    "session_seconds": 7200,
    "seeds": [271, 272],
}

if RUN_CAMPAIGN:
    started_at = time.monotonic()
    print("Memória/fractal/Fibonacci: sete modelos, duas seeds; limite de 2h.", flush=True)
    url = (
        "https://raw.githubusercontent.com/PedroGGC/-LLM-2-/"
        f"{LLM2_REVISION}/src/llm2/research/recall_bootstrap.py"
    )
    with urlopen(url, timeout=60) as response:
        source = response.read(1024 * 1024 + 1)
    if hashlib.sha256(source).hexdigest() != "c50fb24dfff81b00b41dcc6074c5bca102f83cb110f79785349e1902e27de21c":
        raise ValueError("Bootstrap diferente da revisão verificada.")

    namespace = {"__name__": "llm2_recall_bootstrap"}
    exec(compile(source.decode("utf-8"), url, "exec"), namespace)
    try:
        LLM2_RECALL_INFO = namespace["bootstrap_git"](
            LLM2_REVISION, run=True, started_at=started_at, **LLM2_SETTINGS
        )
    finally:
        if "download_file" in namespace:
            LLM2_BAIXAR = namespace["download_file"]
else:
    print("Treino desativado.")
