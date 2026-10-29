"""Standalone pinned Kaggle launcher for the independent-corpus confirmation."""
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import re
import runpy
import sys
import time
from urllib.request import urlopen


def bootstrap_git(revision, *, run=False, run_number=18, session_seconds=14400.,
                  seeds=(101, 102, 103, 104, 105), started_at=None):
    if not run:
        print('Execution disabled; no setup or training started.')
        return {'status': 'not_started'}
    if not re.fullmatch(r'[0-9a-f]{40}', revision):
        raise ValueError('use a complete 40-character commit SHA')
    if (not seeds or len(set(seeds)) != len(seeds)
            or any(type(s) is not int or s not in range(101, 106) for s in seeds)):
        raise ValueError('select unique seeds from 101..105')
    now = time.monotonic()
    started_at = now if started_at is None else float(started_at)
    if (not math.isfinite(started_at) or not 0 <= started_at <= now
            or not 300 < float(session_seconds) <= 14400
            or type(run_number) is not int or run_number <= 0):
        raise ValueError('invalid session duration, start or run number')
    url = f'https://raw.githubusercontent.com/PedroGGC/-LLM-2-/{revision}/src/llm2/research/cycle2_bootstrap.py'
    with urlopen(url, timeout=60) as response:
        payload = response.read(1024**2 + 1)
    if len(payload) > 1024**2:
        raise ValueError('bootstrap helper exceeds expected size')
    helper = {'__name__': 'llm2_cycle3_helper'}
    exec(compile(payload.decode('utf-8'), url, 'exec'), helper)
    session = helper['_BootstrapSession'](session_seconds, started_at)
    hardware = helper['_hardware']()
    source, parent, metadata = helper['checkout_git_source'](revision, session, hardware)
    if hashlib.sha256((source/'src/llm2/research/cycle2_bootstrap.py').read_bytes()).digest() != hashlib.sha256(payload).digest():
        raise ValueError('downloaded helper differs from the checked-out revision')
    work, gpus, before_torch = hardware
    command = helper['bounded_command']
    print(f'Cycle 3: WikiText-103, six jobs per seed; maximum {session.limit/3600:g} hours including setup/export.', flush=True)
    command([sys.executable, '-m', 'pip', 'install', '--quiet', '-e', str(source)+'[dev,research,wikitext]'],
            cwd=source, timeout=session.remaining())
    if importlib.metadata.version('torch') != before_torch:
        raise RuntimeError('installation changed Torch; review/restart the environment')
    output = runpy.run_path(str(source/'src/llm2/research/naming.py'))['numbered_output'](
        work, number=run_number, search_roots=[Path('/kaggle/input')])
    metadata.update(source_root=str(source), torch_preserved=before_torch,
                    helper_sha256=hashlib.sha256(payload).hexdigest(),
                    setup_elapsed_seconds=session.elapsed, session_limit_seconds=session.limit)
    metadata_file = parent/'bootstrap.json'
    metadata_file.write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    environment = os.environ.copy()
    environment.update(PYTHONPATH=str(source/'src'), PYTHONUNBUFFERED='1', PYTHONIOENCODING='utf-8')
    arguments = [sys.executable, str(source/'scripts/run_cycle3.py'), '--execute',
                 '--cache', work/'llm2-wikitext-pinned-cache', '--download', '--output', output,
                 '--gpus', *gpus, '--seeds', *seeds, '--session-seconds', session.limit,
                 '--export-reserve-seconds', session.reserve, '--bootstrap-metadata', metadata_file,
                 '--bootstrap-elapsed-seconds', session.elapsed]
    try:
        command(arguments, cwd=source, timeout=session.remaining(export=True), env=environment)
    finally:
        print(f'\nCampaign: {output}\nDownload both {output.name}-results.zip and {output.name}-checkpoints.zip.', flush=True)
    return {'campaign': str(output), 'session_elapsed_seconds': session.elapsed}
