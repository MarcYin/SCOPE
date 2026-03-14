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

from scope_torch.cli.fetch_upstream import main


if __name__ == "__main__":
    main()
