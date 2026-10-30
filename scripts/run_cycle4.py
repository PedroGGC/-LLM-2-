"""Run the pinned checkout before any editable installation."""
from pathlib import Path
import sys
import time

started=time.monotonic()
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from llm2.research.cycle4 import main

if __name__=="__main__":main(started=started)
