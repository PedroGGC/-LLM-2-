"""Load the operator while every llm2 import resolves to the pinned worker checkout."""
import argparse
from pathlib import Path
import runpy
import sys
import time

started=time.monotonic()
p=argparse.ArgumentParser(add_help=False);p.add_argument('--worker-root',required=True)
known,_=p.parse_known_args()
sys.path.insert(0,str(Path(known.worker_root).resolve()/'src'))
operator=Path(__file__).resolve().parents[1]/'src/llm2/research/cycle3_short.py'
namespace=runpy.run_path(str(operator),run_name='llm2_cycle3_short_operator')
namespace['main'](started=started)
