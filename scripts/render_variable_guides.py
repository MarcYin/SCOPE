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

from scope.variables import render_workflow_variable_markdown


WORKFLOWS = (
    "reflectance",
    "fluorescence",
    "thermal",
    "energy-balance",
)


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    target_root = repo_root / "docs" / "workflow-variables"
    target_root.mkdir(parents=True, exist_ok=True)
    for workflow in WORKFLOWS:
        target = target_root / f"{workflow}.md"
        target.write_text(render_workflow_variable_markdown(workflow), encoding="utf-8")
        print(target)


if __name__ == "__main__":
    main()
