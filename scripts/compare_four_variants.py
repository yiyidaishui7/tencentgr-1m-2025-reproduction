#!/usr/bin/env python3
"""Generate the row-aligned maxlen 101/50 x MM/no-MM comparison report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from comparison_utils import REQUIRED_VARIANTS, compare_four_variants  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    for variant in REQUIRED_VARIANTS:
        parser.add_argument(f"--{variant}", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--k", type=int, default=10)
    return parser.parse_args()


def load_archive(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {name: archive[name] for name in archive.files}


def main() -> None:
    args = parse_args()
    predictions = {variant: load_archive(getattr(args, variant)) for variant in REQUIRED_VARIANTS}
    report = compare_four_variants(predictions, k=args.k)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
