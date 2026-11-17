"""Compatibility entry point; implementation lives in src/ridge_teacher/."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from ridge_teacher.data.cache import main

if __name__ == "__main__":
    main()
