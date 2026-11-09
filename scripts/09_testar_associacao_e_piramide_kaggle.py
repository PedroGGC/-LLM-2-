import hashlib
import time
from urllib.request import urlopen

RUN_CAMPAIGN = True
LLM2_REVISION = "bcd7780e8d929ce938c69a8eaf8167241518e744"
LLM2_SETTINGS = {
    "run_number": 31,
    "session_seconds": 7200,
    "seeds": [311, 312],
}

if RUN_CAMPAIGN:
    started_at = time.monotonic()
    print("Associação e pirâmide: quatro modelos, duas seeds; T4x2, limite de 2h.", flush=True)
    url = (
        "https://raw.githubusercontent.com/PedroGGC/-LLM-2-/"
        f"{LLM2_REVISION}/src/llm2/research/binding_bootstrap.py"
    )
    with urlopen(url, timeout=60) as response:
        source = response.read(1024 * 1024 + 1)
    if hashlib.sha256(source).hexdigest() != "0bee532c03cb07974a7649be32ddbe6a9715c64f73db289880590de5f736f2f9":
        raise ValueError("Bootstrap diferente da revisão verificada.")

    namespace = {"__name__": "llm2_binding_bootstrap"}
    exec(compile(source.decode("utf-8"), url, "exec"), namespace)
    try:
        LLM2_BINDING_INFO = namespace["bootstrap_git"](
            LLM2_REVISION, run=True, started_at=started_at, **LLM2_SETTINGS
        )
    finally:
        if "download_file" in namespace:
            LLM2_BAIXAR = namespace["download_file"]
else:
    print("Treino desativado.")
