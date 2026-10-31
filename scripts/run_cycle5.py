"""Count all imports/setup in the independent Cycle 5 session budget."""
import time
started=time.monotonic()
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from llm2.research.cycle5 import main
if __name__=="__main__":main(started=started)
