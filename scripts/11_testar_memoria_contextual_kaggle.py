import hashlib
import time
from urllib.request import urlopen

RUN_CAMPAIGN = True
LLM2_REVISION = "db507b5258615e94cab7620dc5816fb31c36d8c7"
LLM2_SETTINGS = {
    "run_number": 33,
    "profile": "context",
    "session_seconds": 7200,
    "seeds": [341, 342],
}

if RUN_CAMPAIGN:
    started_at = time.monotonic()
    print("Memória contextual e fractal no TinyStories. T4 x2, teto de 2h.", flush=True)
    url = (
        "https://raw.githubusercontent.com/PedroGGC/-LLM-2-/"
        f"{LLM2_REVISION}/src/llm2/research/stories_bootstrap.py"
    )
    with urlopen(url, timeout=60) as response:
        source = response.read(1024 * 1024 + 1)
    if hashlib.sha256(source).hexdigest() != "dc53834fb3b7c6f10f81914abc8e33105e34a008274e55232b75ec415326d815":
        raise ValueError("Bootstrap diferente da revisão verificada.")

    namespace = {"__name__": "llm2_context_bootstrap"}
    exec(compile(source.decode("utf-8"), url, "exec"), namespace)
    try:
        LLM2_CONTEXT_INFO = namespace["bootstrap_git"](
            LLM2_REVISION, run=True, started_at=started_at, **LLM2_SETTINGS
        )
    finally:
        if "download_file" in namespace:
            LLM2_BAIXAR = namespace["download_file"]
else:
    print("Treino desativado.")
