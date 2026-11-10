import hashlib
import time
from urllib.request import urlopen

RUN_CAMPAIGN = True
LLM2_REVISION = "6f469fa6b80f2109b92c8b7be4a052c8c2ffaddd"
LLM2_SETTINGS = {
    "run_number": 32,
    "session_seconds": 7200,
    "seeds": [331, 332],
}

if RUN_CAMPAIGN:
    started_at = time.monotonic()
    print("TinyStories: memória, pirâmide e dois Transformers; T4x2, teto 2h.", flush=True)
    url = (
        "https://raw.githubusercontent.com/PedroGGC/-LLM-2-/"
        f"{LLM2_REVISION}/src/llm2/research/stories_bootstrap.py"
    )
    with urlopen(url, timeout=60) as response:
        source = response.read(1024 * 1024 + 1)
    if hashlib.sha256(source).hexdigest() != "c28f5076fda23b50ff4b32ab987a1e5e11c9c17392c45719af8ee383ed798e8a":
        raise ValueError("Bootstrap diferente da revisão verificada.")

    namespace = {"__name__": "llm2_stories_bootstrap"}
    exec(compile(source.decode("utf-8"), url, "exec"), namespace)
    try:
        LLM2_STORIES_INFO = namespace["bootstrap_git"](
            LLM2_REVISION, run=True, started_at=started_at, **LLM2_SETTINGS
        )
    finally:
        if "download_file" in namespace:
            LLM2_BAIXAR = namespace["download_file"]
else:
    print("Treino desativado.")
