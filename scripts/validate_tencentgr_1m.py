"""Validate the TencentGR-1M files needed by the official baseline."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import pyarrow.dataset as ds


INDEXER_SHA256 = "bc2e63823b527e7baee75e14db08595c32fd6e3e45fec23822d8b623c2d5dcaf"
PARQUET_GROUPS = {
    "candidate": {"item_id", "retrieval_id"},
    "item_feat": {"item_id"},
    "seq": {"user_id", "seq"},
    "user_feat": {"user_id"},
    "mm_emb/emb_81_32_parquet": {"anonymous_cid", "emb"},
}
EXPECTED_PARQUET_FILES = {
    "candidate": 1,
    "item_feat": 10,
    "seq": 10,
    "user_feat": 10,
    "mm_emb/emb_81_32_parquet": 5,
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("data_dir", type=Path)
    return parser.parse_args()


def main() -> None:
    data_dir = parse_args().data_dir.expanduser().resolve()
    if not data_dir.is_dir():
        raise FileNotFoundError(f"Dataset directory does not exist: {data_dir}")

    indexer = data_dir / "indexer.pkl"
    if not indexer.is_file():
        raise FileNotFoundError(indexer)
    actual_hash = sha256(indexer)
    if actual_hash != INDEXER_SHA256:
        raise RuntimeError(
            f"indexer.pkl checksum mismatch: expected {INDEXER_SHA256}, got {actual_hash}"
        )

    total_rows = 0
    for relative_path, required_columns in PARQUET_GROUPS.items():
        path = data_dir / relative_path
        if not path.is_dir():
            raise FileNotFoundError(path)
        parquet_files = list(path.glob("*.parquet"))
        expected_files = EXPECTED_PARQUET_FILES[relative_path]
        if len(parquet_files) != expected_files:
            raise RuntimeError(
                f"{relative_path} is incomplete: expected {expected_files} parquet files, "
                f"found {len(parquet_files)}"
            )
        dataset = ds.dataset(path, format="parquet")
        columns = set(dataset.schema.names)
        missing = required_columns - columns
        if missing:
            raise RuntimeError(f"{relative_path} is missing columns: {sorted(missing)}")
        rows = dataset.count_rows()
        if rows <= 0:
            raise RuntimeError(f"{relative_path} contains no rows")
        total_rows += rows
        print(f"{relative_path}: rows={rows:,}, columns={len(columns)}")

    mm_dataset = ds.dataset(data_dir / "mm_emb/emb_81_32_parquet", format="parquet")
    first_batch = next(iter(mm_dataset.scanner(columns=["emb"], batch_size=1).to_batches()))
    embedding = first_batch.column("emb")[0].as_py()
    if len(embedding) != 32:
        raise RuntimeError(f"emb_81 dimension mismatch: expected 32, got {len(embedding)}")

    print(f"indexer.pkl: sha256={actual_hash}")
    print(f"validated_rows={total_rows:,}, emb_81_dim=32")


if __name__ == "__main__":
    main()
