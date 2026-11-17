"""Compatibility entry point; implementation lives in src/ridge_teacher/."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from ridge_teacher.features.selection import FoldCpGSelector
from ridge_teacher.train.search import make_search
from ridge_teacher.data.contracts import load_inputs
from ridge_teacher.cli import main

if __name__ == "__main__":
    main()
