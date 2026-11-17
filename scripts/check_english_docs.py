"""Reject Chinese text or Chinese-localization names in Git-indexed documents.

Reads staged blobs, so ignored local notes are never inspected or published.
Uses only the Python standard library. Run before committing or in CI.
"""
from pathlib import Path
import re
import subprocess
import sys

DOCUMENT_EXTENSIONS = {".md", ".mdx", ".markdown", ".rst", ".txt", ".html", ".htm", ".adoc"}
HAN = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff\U00020000-\U0002fa1f]")
LOCALIZED_NAME = re.compile(r"(?:^|[_.-])zh(?:[_.-]|$)", re.IGNORECASE)


def main():
    root = Path(__file__).resolve().parents[1]
    tracked = subprocess.check_output(["git", "ls-files", "-z"], cwd=root).decode().split("\0")
    documents = [name for name in tracked if Path(name).suffix.lower() in DOCUMENT_EXTENSIONS]
    failures = []
    for name in documents:
        if LOCALIZED_NAME.search(Path(name).name) or HAN.search(name):
            failures.append(f"{name}: Chinese-localization filename must stay untracked")
        content = subprocess.check_output(["git", "show", ":" + name], cwd=root).decode("utf-8")
        lines = [str(i) for i, line in enumerate(content.splitlines(), 1) if HAN.search(line)]
        if lines:
            failures.append(f"{name}: Chinese text on lines {', '.join(lines[:10])}")
    if failures:
        print("Repository documentation must be English. Keep Chinese copies local and ignored.")
        print("\n".join(failures))
        return 1
    print(f"PASS: {len(documents)} tracked documents contain no Chinese text or localization filenames.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
