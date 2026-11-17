"""Single-command handoff entry point; no pip installation of this package needed."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from ridge_teacher.runner import main

if __name__ == "__main__":
    main()
