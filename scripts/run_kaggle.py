"""Run the explicitly selected Kaggle campaign using this checkout."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from llm2.research.campaign import main


if __name__ == "__main__":
    main()
