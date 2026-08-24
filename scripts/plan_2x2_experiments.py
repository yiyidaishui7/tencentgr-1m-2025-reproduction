#!/usr/bin/env python3
"""Write exact commands for the seed-2025 maxlen 101/50 x MM/no-MM study."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiment_plan import build_experiment_plan  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("reproduction_config.json"))
    parser.add_argument("--data-path", type=Path, required=True)
    parser.add_argument("--runs-root", type=Path, default=Path("runs"))
    parser.add_argument("--eval-root", type=Path, default=Path("eval"))
    parser.add_argument("--scratch-root", type=Path, default=Path("scratch"))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--python", default="python", dest="python_executable")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    plan = build_experiment_plan(
        config,
        data_path=args.data_path,
        runs_root=args.runs_root,
        eval_root=args.eval_root,
        scratch_root=args.scratch_root,
        device=args.device,
        python_executable=args.python_executable,
    )
    payload = json.dumps(plan, indent=2) + "\n"
    if args.output is None:
        print(payload, end="")
        return
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(payload, encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
