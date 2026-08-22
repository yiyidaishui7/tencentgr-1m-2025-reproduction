"""Download the reproducible TencentGR-1M subset used by the baseline."""

from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import snapshot_download


DEFAULT_REPO_ID = "TAAC2025/TencentGR-1M"
ALLOW_PATTERNS = [
    "README.md",
    "indexer.pkl",
    "candidate/*",
    "item_feat/*",
    "seq/*",
    "user_feat/*",
    "mm_emb/emb_81_32_parquet/*",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    parser.add_argument("--max-workers", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=args.repo_id,
        repo_type="dataset",
        local_dir=output,
        allow_patterns=ALLOW_PATTERNS,
        max_workers=args.max_workers,
    )
    print(output)


if __name__ == "__main__":
    main()
