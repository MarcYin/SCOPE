from __future__ import annotations

import sys
from pathlib import Path


def _ensure_src_on_path() -> None:
    src = Path(__file__).resolve().parents[1] / "src"
    if src.exists():
        src_str = str(src)
        if src_str not in sys.path:
            sys.path.insert(0, src_str)


_ensure_src_on_path()

from scope.variables import render_variable_markdown


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    target = repo_root / "docs" / "variable-glossary.md"
    target.write_text(render_variable_markdown(), encoding="utf-8")
    print(target)


if __name__ == "__main__":
    main()
