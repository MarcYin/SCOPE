from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


DEFAULT_SCOPE_REPO = "https://github.com/Christiaanvandertol/SCOPE.git"
DEFAULT_SCOPE_COMMIT = "e4c2e5109a309e6d2636fd6aa33e0e54b6dd88de"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fetch the pinned upstream MATLAB SCOPE repository required by asset-backed examples and parity workflows."
    )
    parser.add_argument(
        "--dest",
        default="upstream/SCOPE",
        help="Destination directory for the upstream SCOPE checkout.",
    )
    parser.add_argument(
        "--repo-url",
        default=DEFAULT_SCOPE_REPO,
        help="Git URL for the upstream SCOPE repository.",
    )
    parser.add_argument(
        "--commit",
        default=DEFAULT_SCOPE_COMMIT,
        help="Pinned upstream SCOPE commit to fetch and check out.",
    )
    return parser.parse_args()


def fetch_upstream_scope(dest: str | Path, *, repo_url: str = DEFAULT_SCOPE_REPO, commit: str = DEFAULT_SCOPE_COMMIT) -> Path:
    destination = Path(dest)
    destination.parent.mkdir(parents=True, exist_ok=True)

    if not (destination / ".git").exists():
        subprocess.run(
            ["git", "clone", repo_url, str(destination)],
            check=True,
        )

    subprocess.run(
        ["git", "-C", str(destination), "fetch", "--depth", "1", "origin", commit],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(destination), "checkout", commit],
        check=True,
    )
    return destination


def main() -> None:
    args = parse_args()
    resolved = fetch_upstream_scope(args.dest, repo_url=args.repo_url, commit=args.commit)
    print(resolved.resolve())


if __name__ == "__main__":
    main()
