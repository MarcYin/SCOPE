from __future__ import annotations

import argparse
import json
from typing import Sequence

from ..variables import iter_variables, search_variables


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Search the SCOPE variable glossary for physical meanings, units, and workflow usage."
    )
    parser.add_argument("query", nargs="?", help="Variable name or free-text search query.")
    parser.add_argument(
        "--kind",
        choices=("dimension", "option", "input", "output"),
        help="Restrict results to one variable kind.",
    )
    parser.add_argument("--category", help="Restrict results to a single category, for example 'reflectance' or 'meteorology'.")
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON instead of a plain-text table.")
    parser.add_argument("--all", action="store_true", help="List all registry entries.")
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    return build_parser().parse_args(argv)


def run(args: argparse.Namespace) -> list[dict[str, object]]:
    if args.all and args.query:
        raise ValueError("--all and query cannot be used together")
    if args.all:
        matches = list(iter_variables())
    else:
        matches = search_variables(args.query, kind=args.kind, category=args.category)
    rows = [match.to_dict() for match in matches]
    if args.json:
        print(json.dumps(rows, indent=2))
    else:
        print(_render_text(rows))
    return rows


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    run(args)


def _render_text(rows: list[dict[str, object]]) -> str:
    if not rows:
        return "No glossary entries matched."
    name_width = max(len(str(row["name"])) for row in rows)
    kind_width = max(len(str(row["kind"])) for row in rows)
    category_width = max(len(str(row["category"])) for row in rows)
    lines = []
    for row in rows:
        aliases = ", ".join(row["aliases"])
        workflows = ", ".join(row["workflows"])
        meta = "; ".join(part for part in (workflows, aliases, row["notes"]) if part)
        header = (
            f"{row['name']:<{name_width}}  "
            f"{row['kind']:<{kind_width}}  "
            f"{row['category']:<{category_width}}  "
            f"{row['units']}"
        )
        lines.append(header)
        lines.append(f"  {row['meaning']}")
        if meta:
            lines.append(f"  {meta}")
        lines.append("")
    return "\n".join(lines).rstrip()


if __name__ == "__main__":
    main()
