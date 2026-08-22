"""Audit item-ID alignment across TencentGR-1M index, embeddings, and candidates."""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import pyarrow.dataset as ds

from validate_tencentgr_1m import INDEXER_SHA256, sha256


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("data_dir", type=Path)
    return parser.parse_args()


def main() -> None:
    data_dir = parse_args().data_dir.expanduser().resolve()
    indexer_path = data_dir / "indexer.pkl"
    actual_hash = sha256(indexer_path)
    if actual_hash != INDEXER_SHA256:
        raise RuntimeError(
            f"Refusing to unpickle indexer with unexpected SHA-256: {actual_hash}"
        )

    with indexer_path.open("rb") as handle:
        indexer = pickle.load(handle)
    indexed_item_ids = {str(item_id) for item_id in indexer["i"]}

    mm_path = data_dir / "mm_emb" / "emb_81_32_parquet"
    mm_item_ids: set[str] = set()
    mm_rows = 0
    for batch in ds.dataset(mm_path, format="parquet").scanner(
        columns=["anonymous_cid"], batch_size=250_000
    ).to_batches():
        values = batch.column("anonymous_cid").to_pylist()
        mm_item_ids.update(str(value) for value in values if value is not None)
        mm_rows += batch.num_rows

    candidate_rows = 0
    candidate_index_hits = 0
    candidate_mm_hits = 0
    for batch in ds.dataset(data_dir / "candidate", format="parquet").scanner(
        columns=["item_id"], batch_size=250_000
    ).to_batches():
        for value in batch.column("item_id").to_pylist():
            key = str(value)
            candidate_rows += 1
            candidate_index_hits += key in indexed_item_ids
            candidate_mm_hits += key in mm_item_ids

    mm_index_hits = len(mm_item_ids & indexed_item_ids)
    report = {
        "indexed_items": len(indexed_item_ids),
        "mm_rows": mm_rows,
        "unique_mm_items": len(mm_item_ids),
        "mm_items_in_index": mm_index_hits,
        "mm_index_coverage": mm_index_hits / max(len(indexed_item_ids), 1),
        "candidate_rows": candidate_rows,
        "candidate_items_in_index": candidate_index_hits,
        "candidate_index_hit_rate": candidate_index_hits / max(candidate_rows, 1),
        "candidate_items_with_mm": candidate_mm_hits,
        "candidate_mm_hit_rate": candidate_mm_hits / max(candidate_rows, 1),
    }
    print(json.dumps(report, indent=2, sort_keys=True))

    if mm_index_hits == 0:
        raise RuntimeError("No multimodal IDs align with indexer item IDs")


if __name__ == "__main__":
    main()
