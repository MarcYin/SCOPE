from __future__ import annotations

import argparse
from typing import Sequence

from . import fetch_upstream, prepare_scope_input, run, variables


def _run_fetch_upstream(args: argparse.Namespace) -> None:
    print(fetch_upstream.run(args).resolve())


def _run_prepare_scope_input(args: argparse.Namespace) -> None:
    prepare_scope_input.run(args)


def _run_scope_dataset(args: argparse.Namespace) -> None:
    print(run.run(args).resolve())


def _run_variables(args: argparse.Namespace) -> None:
    variables.run(args)


def build_parser() -> argparse.ArgumentParser:
    fetch_parent = fetch_upstream.build_parser()
    prepare_parent = prepare_scope_input.build_parser()
    run_parent = run.build_parser()
    variables_parent = variables.build_parser()
    parser = argparse.ArgumentParser(
        prog="scope",
        description="Top-level command-line interface for SCOPE-RTM.",
    )
    parser.add_argument(
        "--version",
        action="store_true",
        help="Print the installed package version and exit.",
    )
    subparsers = parser.add_subparsers(dest="command")
    fetch_parser = subparsers.add_parser(
        "fetch-upstream",
        help="Fetch the pinned upstream MATLAB SCOPE checkout.",
        description=fetch_parent.description,
        parents=[fetch_parent],
        add_help=False,
    )
    fetch_parser.set_defaults(func=_run_fetch_upstream)
    prepare_parser = subparsers.add_parser(
        "prepare",
        help="Build a runner-ready SCOPE input dataset.",
        description=prepare_parent.description,
        parents=[prepare_parent],
        add_help=False,
    )
    prepare_parser.set_defaults(func=_run_prepare_scope_input)
    run_parser = subparsers.add_parser(
        "run",
        help="Run a prepared SCOPE input dataset and write outputs.",
        description=run_parent.description,
        parents=[run_parent],
        add_help=False,
    )
    run_parser.set_defaults(func=_run_scope_dataset)
    variables_parser = subparsers.add_parser(
        "vars",
        help="Search the variable glossary.",
        description=variables_parent.description,
        parents=[variables_parent],
        add_help=False,
    )
    variables_parser.set_defaults(func=_run_variables)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.version:
        from .. import __version__

        print(__version__)
        return
    if hasattr(args, "func"):
        args.func(args)
        return

    parser.print_help()
