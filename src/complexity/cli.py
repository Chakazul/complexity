"""Command line: `complexity serve` for the interactive UI, `complexity run` for headless runs."""

from __future__ import annotations

import argparse
import contextlib
import csv
import sys

from complexity import registry
from complexity.session import Run, RunSpec


def _list() -> None:
    print("Systems (system:pattern):")
    for key, cls in registry.systems().items():
        for p, name in cls.patterns.items():
            print(f"  {key}:{p:<20} {cls.name} · {name}")
    print("Measures:")
    for key, cls in registry.measures().items():
        print(f"  {key:<20} {cls.name} [{cls.unit}, window {cls.window}]")


def _run(args) -> None:
    measures = args.measures.split(",") if args.measures else list(registry.measures())
    size = tuple(int(v) for v in args.size.split("x")) if args.size else None
    runs = []
    for item in args.runs:
        system, pattern = item.split(":")
        runs.append(Run(RunSpec(system, pattern, size, args.seed), measures))
    with open(args.out, "w", newline="") if args.out else contextlib.nullcontext(sys.stdout) as out:
        writer = csv.writer(out)
        writer.writerow(["run", "system", "pattern", "t", *measures])
        for step in range(args.steps + 1):
            for i, run in enumerate(runs):
                if step % args.every == 0:
                    values, _ = run.measure()
                    writer.writerow(
                        [i, run.spec.system, run.spec.pattern, run.t,
                         *[f"{values[k]:.6g}" if k in values else "" for k in measures]]
                    )
                run.step()
    if args.out:
        print(f"wrote {args.out}", file=sys.stderr)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="complexity")
    sub = parser.add_subparsers(dest="command", required=True)

    serve = sub.add_parser("serve", help="start the interactive UI")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)

    sub.add_parser("list", help="list systems, patterns and measures")

    run = sub.add_parser("run", help="run headless and write measure values as CSV")
    run.add_argument("runs", nargs="+", help="system:pattern, e.g. life:gosper_glider_gun")
    run.add_argument("--measures", help="comma-separated measure keys (default: all)")
    run.add_argument("--steps", type=int, default=500)
    run.add_argument("--every", type=int, default=1, help="measure every N steps")
    run.add_argument("--size", help="grid size HxW, e.g. 128x128")
    run.add_argument("--seed", type=int, default=0)
    run.add_argument("--out", help="CSV path (default: stdout)")

    args = parser.parse_args(argv)
    if args.command == "serve":
        import uvicorn

        print(f"Open http://{args.host}:{args.port}")
        uvicorn.run("complexity.app.server:app", host=args.host, port=args.port, log_level="warning")
    elif args.command == "list":
        _list()
    else:
        _run(args)


if __name__ == "__main__":
    main()
