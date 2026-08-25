"""Environment-backed configuration shared by the public OnePiece runner."""

from __future__ import annotations

import hashlib
import os
import random
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pyarrow.compute as pc
import torch


def required_path(name: str) -> Path:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"required environment variable is missing: {name}")
    path = Path(value).expanduser().resolve()
    if not path.exists():
        raise RuntimeError(f"configured path does not exist for {name}: {path}")
    return path


ROOT = required_path("ONEPIECE_WORK_ROOT")
SOURCE_DIR = required_path("ONEPIECE_SOURCE_DIR")
CONTRACT = required_path("ONEPIECE_CONTRACT")
INDEXER = required_path("ONEPIECE_INDEXER")
HOME = Path(os.environ.get("HF_HOME", ROOT / "hf_home")).expanduser().resolve()
INDEXER_SHA256 = os.environ.get("ONEPIECE_INDEXER_SHA256", "").lower()
if len(INDEXER_SHA256) != 64:
    raise RuntimeError("ONEPIECE_INDEXER_SHA256 must contain a SHA-256 digest")

ITEM_FEATURES = [
    "100", "117", "118", "101", "102", "119", "120",
    "114", "112", "121", "115", "122", "116",
]
USER_FEATURES = ["103", "104", "105", "109"]
SEED = int(os.environ.get("ONEPIECE_TRAINING_SEED", "20250825"))
MAXLEN = 101


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_dense_features(dataset, id_name: str, feature_names: list[str], size: int):
    table = dataset.data.table
    ids = (
        pc.fill_null(table[id_name], 0)
        .combine_chunks()
        .to_numpy(zero_copy_only=False)
        .astype(np.int64)
    )
    if int(ids.min()) < 1 or int(ids.max()) > size or len(np.unique(ids)) != len(ids):
        raise RuntimeError(f"invalid or duplicate {id_name} values")
    values = {}
    maxima = {}
    for name in feature_names:
        column = (
            pc.fill_null(table[name], 0)
            .combine_chunks()
            .to_numpy(zero_copy_only=False)
            .astype(np.int32)
        )
        dense = np.zeros(size + 1, dtype=np.int32)
        dense[ids] = column
        values[name] = dense
        maxima[name] = int(column.max(initial=0))
    return values, maxima


def make_args() -> SimpleNamespace:
    return SimpleNamespace(
        device="cuda:0", mode="train", maxlen=MAXLEN, debug=False,
        mm_emb_id=[], mm_sid=[], base_user_sparse=USER_FEATURES,
        base_item_sparse=ITEM_FEATURES, base_user_array=[], user_sparse=None,
        item_sparse=None, user_array=None, item_array=None,
        user_continual=None, item_continual=None, context_item_sparse=[],
        feature_dropout_list=[], feature_dropout_rate=0.0, bucket_sizes=[],
        user_cache_path=str(ROOT), sid=False, sid_codebook_layer=2,
        sid_codebook_size=128, norm_first=True, rms_norm=False,
        sparse_embedding=False, rope=False, use_hstu=True, hstu_rope=False,
        mm_emb_gate=False, random_perturbation=False,
        random_perturbation_value=0.0, use_moe=False, moe_num_experts=4,
        moe_top_k=2, moe_intermediate_size=128,
        moe_load_balancing_alpha=0.0, moe_load_balancing_update_freq=1,
        learnable_temp=False, infonce_temp=0.02, hidden_units=128,
        dnn_hidden_units=1, hash_emb_size=8, timestamp_bucket_emb_size=8,
        dropout_rate=0.1, num_heads=4, num_blocks=4,
        feed_forward_hidden_units=2, similarity_function="cosine",
        interest_k=1, reward=False, reward_only=False, log_interval=1_000_000,
        infonce=True,
    )

