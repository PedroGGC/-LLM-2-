import hashlib
from urllib.request import urlopen

# "limpar" remove somente temporários conhecidos; "inventario" apenas lista.
# Campanhas, ZIPs, dados preparados e checkpoints permanecem no working.
MODO = "limpar"
LLM2_REVISION = "1c8ce2abb8f47d3167454b3503598854b7756dad"

if MODO not in ("limpar", "inventario"):
    raise ValueError("Use MODO='limpar' ou 'inventario'.")

print("Conferindo temporários e processos; arquivos científicos serão preservados.", flush=True)
url = (
    "https://raw.githubusercontent.com/PedroGGC/-LLM-2-/"
    f"{LLM2_REVISION}/src/llm2/research/kaggle_light.py"
)
with urlopen(url, timeout=60) as response:
    source = response.read(1024 * 1024 + 1)
if hashlib.sha256(source).hexdigest() != "8c05edcf6d0450e273ac2f1fc118a8167bc9833490f841c9cfb1d57d96262007":
    raise ValueError("Limpeza diferente da revisão verificada.")

namespace = {"__name__": "llm2_scoped_cleanup"}
exec(compile(source.decode("utf-8"), url, "exec"), namespace)
LLM2_CLEANUP_INFO = namespace["clean_working"](
    "/kaggle/working", apply=MODO == "limpar"
)
