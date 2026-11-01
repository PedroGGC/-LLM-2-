"""Standalone raw-shard recovery. Immutable hashes come from the qualified importer."""
import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import tempfile
import time
from urllib.request import urlopen

MAX_SHARD_BYTES=512*1024**2


def sha_file(path):
    digest=hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda:stream.read(1024**2),b""):digest.update(block)
    return digest.hexdigest()


def state(path,expected):
    path=Path(path)
    if path.is_symlink():return {"status":"linked","sha256":None,"bytes":None}
    if not path.exists():return {"status":"missing","sha256":None,"bytes":None}
    if not path.is_file():return {"status":"not_file","sha256":None,"bytes":None}
    before=path.stat()
    if before.st_size>MAX_SHARD_BYTES:return {"status":"oversized","sha256":None,"bytes":before.st_size}
    actual=sha_file(path);after=path.stat()
    if path.is_symlink() or (before.st_ino,before.st_size,before.st_mtime_ns)!=(after.st_ino,after.st_size,after.st_mtime_ns):
        raise RuntimeError("raw cache changed during verification")
    return {"status":"valid" if actual==expected else "invalid","sha256":actual,"bytes":after.st_size}


def root_directory(base):
    base=Path(base)
    if base.is_symlink():raise ValueError("linked working root")
    base=base.resolve()
    inputs=Path("/kaggle/input").resolve()
    if not base.is_dir() or base.parent==base or base==inputs or base.is_relative_to(inputs):
        raise ValueError("invalid working root")
    return base


def internal_directory(base,name):
    path=base/name
    if path.is_symlink() or path.resolve().parent!=base or path.exists() and not path.is_dir():
        raise ValueError("raw cache directory is linked or invalid")
    return path


def publish(temporary,path,expected):
    """Exclusive atomic publication. A concurrent valid publication is reusable."""
    if temporary.parent!=path.parent or temporary.is_symlink() or sha_file(temporary)!=expected:
        raise ValueError("invalid staging file")
    try:os.link(temporary,path)
    except FileExistsError:
        if state(path,expected)["status"]!="valid":raise ValueError("existing verified cache differs; never overwritten")


def download(url,stream,expected,*,opener=urlopen,max_bytes=MAX_SHARD_BYTES):
    digest=hashlib.sha256();count=0;last=time.monotonic()
    with opener(url,timeout=60) as response:
        while True:
            block=response.read(1024**2)
            if not block:break
            count+=len(block)
            if count>max_bytes:raise ValueError("raw shard exceeds transfer limit")
            stream.write(block);digest.update(block)
            if time.monotonic()-last>=10:
                print(f"  Download: {count/1024**2:.1f} MiB recebidos...",flush=True);last=time.monotonic()
    actual=digest.hexdigest()
    if actual!=expected:
        raise ValueError(f"download checksum mismatch: expected={expected}, observed={actual}, bytes={count}")
    return count


def prepare_cache(base,files,revision,*,opener=urlopen,attempts=2,reserve_bytes=2*1024**3,report=None):
    base=root_directory(base)
    if not re.fullmatch(r"[0-9a-f]{40}",revision) or not files or type(attempts) is not int or not 1<=attempts<=3:
        raise ValueError("invalid pinned raw-cache specification")
    for name,digest in files.items():
        if not re.fullmatch(r"[a-z]+-[0-9]+-of-[0-9]+\.parquet",name) or not re.fullmatch(r"[0-9a-f]{64}",digest):
            raise ValueError("unsafe raw shard name/checksum")
    legacy=internal_directory(base,"llm2-wikitext-pinned-cache")
    verified=internal_directory(base,"llm2-wikitext-verified-cache")
    report={} if report is None else report
    report.update(revision=revision,legacy_cache=str(legacy),selected_cache=str(verified),files=[],status="checking",
                  policy="legacy preserved; separate checksum-verified raw cache; no prepared datasets changed")
    verified.mkdir(exist_ok=True)
    for name,expected in files.items():
        old=state(legacy/name,expected);current=state(verified/name,expected)
        row={"name":name,"expected_sha256":expected,"legacy":old,"verified_before":current}
        report["files"].append(row)
        print(f"Cache {name}: legado={old['status']}, verificado={current['status']}",flush=True)
        if old["status"]=="invalid":
            print(f"  Legado preservado: bytes={old['bytes']}, esperado={expected}, observado={old['sha256']}",flush=True)
        if current["status"]=="valid":
            row["action"]="reuse_verified";continue
        if current["status"]!="missing":
            raise ValueError(f"verified cache unexpectedly changed: {name}; preserved for inspection")
        if shutil.disk_usage(base).free<reserve_bytes+MAX_SHARD_BYTES:
            raise OSError("insufficient free space for raw transfer plus 2GiB reserve")
        tries=1 if old["status"]=="valid" else attempts
        errors=[]
        for attempt in range(tries):
            temporary=None
            try:
                with tempfile.NamedTemporaryFile(dir=verified,prefix="llm2-raw-",suffix=".part",delete=False) as stream:
                    temporary=Path(stream.name)
                    if old["status"]=="valid":
                        with (legacy/name).open("rb") as source:shutil.copyfileobj(source,stream)
                        row["action"]="copy_valid_legacy"
                    else:
                        print(f"  Baixando revisão fixada, tentativa {attempt+1}/{tries}...",flush=True)
                        url=f"https://huggingface.co/datasets/Salesforce/wikitext/resolve/{revision}/wikitext-103-raw-v1/{name}?download=true"
                        row["download_bytes"]=download(url,stream,expected,opener=opener)
                        row["action"]="download_verified"
                publish(temporary,verified/name,expected)
                row["verified_after"]=state(verified/name,expected)
                row["attempts"]=attempt+1
                print(f"  SHA-256 conferido: {name}",flush=True)
                break
            except (OSError,ValueError) as exc:
                errors.append(str(exc));row["errors"]=errors.copy()
                print(f"  Tentativa rejeitada: {exc}",flush=True)
                if attempt+1==tries:raise
            finally:
                if temporary is not None:temporary.unlink(missing_ok=True)
    for name,expected in files.items():
        if state(verified/name,expected)["status"]!="valid":raise ValueError("cache changed after publication")
    report["status"]="verified"
    print("Cache bruto verificado; fonte e hashes científicos originais preservados.",flush=True)
    return report


def specification(source,expected_sha):
    source=Path(source)
    if source.is_symlink() or sha_file(source)!=expected_sha:raise ValueError("qualified importer source mismatch")
    tree=ast.parse(source.read_text(encoding="utf-8"));values={}
    for node in tree.body:
        if isinstance(node,ast.Assign) and isinstance(node.targets[0],ast.Name) and node.targets[0].id in ("FILES","REVISION"):
            values[node.targets[0].id]=ast.literal_eval(node.value)
    return values["FILES"],values["REVISION"]


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base",required=True);parser.add_argument("--source",required=True)
    parser.add_argument("--source-sha256",required=True);parser.add_argument("--report",required=True)
    args=parser.parse_args(argv);base=root_directory(args.base)
    report_path=Path(args.report)
    if report_path.is_symlink() or not report_path.resolve().is_relative_to(base) or not report_path.parent.is_dir():
        raise ValueError("report path outside working")
    report={}
    try:
        files,revision=specification(args.source,args.source_sha256)
        return prepare_cache(base,files,revision,report=report)
    except Exception as exc:
        report.update(status="failed",error=str(exc));raise
    finally:
        report_path.write_text(json.dumps(report,indent=2,ensure_ascii=False)+"\n",encoding="utf-8",newline="\n")


if __name__=="__main__":main()
