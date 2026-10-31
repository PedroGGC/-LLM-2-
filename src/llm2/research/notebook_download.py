"""Notebook-local downloads; no proxy URL or duplicate files on disk. Stdlib only."""
import base64
import hashlib
import html
import math
from pathlib import Path

PART_BYTES = 24 * 1024**2


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def download_file(path, *, part=None, part_bytes=PART_BYTES, display=None):
    """Embed one bounded payload. Large downloads request individual numbered parts."""
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise FileNotFoundError(f"Arquivo regular não encontrado: {path}")
    if type(part_bytes) is not int or not 1 <= part_bytes <= PART_BYTES:
        raise ValueError("invalid download part size")
    before = path.stat()
    count = max(1, math.ceil(before.st_size / part_bytes))
    digest = sha256(path)
    info = {"path": str(path), "bytes": before.st_size, "sha256": digest, "parts": count}
    print(f"{path.name}: {before.st_size / 1024**2:.2f} MiB | SHA-256: {digest}", flush=True)
    if count > 1 and part is None:
        print(f"Download em {count} partes, sem criar cópias no working. Execute uma de cada vez:\n"
              f"LLM2_BAIXAR({str(path)!r}, part=1)\n"
              f"Repita part=2 até part={count}. Depois reúna na mesma pasta local (PowerShell):", flush=True)
        # Filenames come from our generated campaign paths; quote PowerShell literals.
        name = path.name.replace("'", "''")
        print(f"$destino = [IO.File]::Create('{name}')\ntry {{\n"
              f"  1..{count} | ForEach-Object {{\n"
              f"    $parte = [IO.File]::OpenRead(('{name}.part{{0:D3}}' -f $_))\n"
              "    try { $parte.CopyTo($destino) } finally { $parte.Dispose() }\n"
              "  }\n} finally { $destino.Dispose() }\n"
              f"Get-FileHash -LiteralPath '{name}' -Algorithm SHA256", flush=True)
        return info
    part = 1 if part is None else part
    if type(part) is not int or not 1 <= part <= count:
        raise ValueError(f"part must be between 1 and {count}")
    with path.open("rb") as stream:
        stream.seek((part - 1) * part_bytes)
        payload = stream.read(part_bytes)
    after = path.stat()
    if (before.st_size, before.st_mtime_ns, before.st_ino) != (after.st_size, after.st_mtime_ns, after.st_ino):
        raise RuntimeError("Arquivo mudou durante a preparação do download")
    name = path.name if count == 1 else f"{path.name}.part{part:03d}"
    encoded = base64.b64encode(payload).decode("ascii")
    markup = (f'<a href="data:application/octet-stream;base64,{encoded}" '
              f'download="{html.escape(name, quote=True)}" '
              'style="display:inline-block;padding:14px;margin:6px;background:#1769aa;'
              'color:white;border-radius:8px;text-decoration:none">'
              f'⬇ Baixar {html.escape(name)}</a>')
    if display is None:
        try:
            from IPython.display import HTML, display as show
            show(HTML(markup))
        except ImportError:
            print("Botão disponível apenas no notebook; arquivo preservado:", path, flush=True)
    else:
        display(markup)
    return {**info, "part": part, "part_bytes": len(payload), "part_sha256": hashlib.sha256(payload).hexdigest()}
