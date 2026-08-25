import hashlib
import json
import math
import os
import pickle
import re
import sys
import time
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import torch
import torch.nn.functional as F
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset

import onepiece_common as common


ROOT = common.ROOT
HOME = common.HOME
CONTRACT = common.CONTRACT
INDEXER = common.INDEXER
ARCHITECTURE = os.environ.get(
    "ONEPIECE_ARCHITECTURE", os.environ.get("ONEPIECE_VARIANT", "hstu")
).strip().lower()
if ARCHITECTURE not in {"hstu", "transformer"}:
    raise RuntimeError(f"unsupported ONEPIECE_ARCHITECTURE: {ARCHITECTURE}")
PHYSICAL_GPU = int(os.environ.get("ONEPIECE_PHYSICAL_GPU", "0"))
if PHYSICAL_GPU < 0:
    raise RuntimeError(f"ONEPIECE_PHYSICAL_GPU must be non-negative, got {PHYSICAL_GPU}")
DEFAULT_EXPERIMENT_IDS = {"hstu": "hstu", "transformer": "transformer"}
EXPERIMENT_ID = os.environ.get(
    "ONEPIECE_EXPERIMENT_ID", DEFAULT_EXPERIMENT_IDS[ARCHITECTURE]
).strip().lower()
if not re.fullmatch(r"[a-z0-9_]{2,32}", EXPERIMENT_ID):
    raise RuntimeError("ONEPIECE_EXPERIMENT_ID must match [a-z0-9_]{2,32}")
OUT = ROOT / "o" / EXPERIMENT_ID
STATUS = ROOT / "l" / f"{EXPERIMENT_ID}_status.json"
ITEM_FEATURES = common.ITEM_FEATURES
USER_FEATURES = common.USER_FEATURES
SEED = int(os.environ.get("ONEPIECE_TRAINING_SEED", str(common.SEED)))
SPLIT_SEED = 2025
MAXLEN = int(os.environ.get("ONEPIECE_MAXLEN", "101"))
BATCH_SIZE = int(os.environ.get("ONEPIECE_BATCH_SIZE", "32"))
EPOCHS = int(os.environ.get("ONEPIECE_EPOCHS", "6"))
EVAL_BATCH_SIZE = int(os.environ.get("ONEPIECE_EVAL_BATCH_SIZE", "128"))
CANDIDATE_BATCH_SIZE = int(os.environ.get("ONEPIECE_CANDIDATE_BATCH_SIZE", "8192"))
LOG_EVERY = int(os.environ.get("ONEPIECE_LOG_EVERY", "500"))
WARMUP_STEPS = int(os.environ.get("ONEPIECE_WARMUP_STEPS", "1000"))
HIDDEN_UNITS = int(os.environ.get("ONEPIECE_HIDDEN_UNITS", "128"))
NUM_BLOCKS = int(os.environ.get("ONEPIECE_NUM_BLOCKS", "4"))
NUM_HEADS = int(os.environ.get("ONEPIECE_NUM_HEADS", "4"))
ENABLE_SID = os.environ.get("ONEPIECE_ENABLE_SID", "0").strip().lower() in {"1", "true", "yes"}
SID_CODEBOOK_SIZE = int(os.environ.get("ONEPIECE_SID_CODEBOOK_SIZE", "4096"))
SID_PATH = (
    Path(os.environ.get("ONEPIECE_SID_PATH", "")).expanduser() if ENABLE_SID else None
)
if ENABLE_SID and (SID_PATH is None or not SID_PATH.is_file()):
    raise RuntimeError("ONEPIECE_SID_PATH must point to a frozen SID array when SID is enabled")
SID_SHA256 = common.sha256(SID_PATH) if SID_PATH is not None else None
if min(
    MAXLEN, BATCH_SIZE, EPOCHS, EVAL_BATCH_SIZE, CANDIDATE_BATCH_SIZE,
    LOG_EVERY, WARMUP_STEPS, HIDDEN_UNITS, NUM_BLOCKS, NUM_HEADS,
) <= 0:
    raise RuntimeError("training dimensions and counts must be positive")
if HIDDEN_UNITS % NUM_HEADS:
    raise RuntimeError("ONEPIECE_HIDDEN_UNITS must be divisible by ONEPIECE_NUM_HEADS")
if SID_CODEBOOK_SIZE <= 0 or SID_CODEBOOK_SIZE > np.iinfo(np.uint16).max:
    raise RuntimeError("invalid ONEPIECE_SID_CODEBOOK_SIZE")

RUN_CONFIG = {
    "architecture": ARCHITECTURE,
    "experiment_id": EXPERIMENT_ID,
    "seed": SEED,
    "split_seed": SPLIT_SEED,
    "maxlen": MAXLEN,
    "batch_size": BATCH_SIZE,
    "epochs": EPOCHS,
    "eval_batch_size": EVAL_BATCH_SIZE,
    "candidate_batch_size": CANDIDATE_BATCH_SIZE,
    "warmup_steps": WARMUP_STEPS,
    "hidden_units": HIDDEN_UNITS,
    "num_blocks": NUM_BLOCKS,
    "num_heads": NUM_HEADS,
    "sid": ENABLE_SID,
    "sid_codebook_size": SID_CODEBOOK_SIZE if ENABLE_SID else None,
    "sid_sha256": SID_SHA256,
}
RUN_SIGNATURE = hashlib.sha256(
    json.dumps(RUN_CONFIG, sort_keys=True, separators=(",", ":")).encode("utf-8")
).hexdigest()


def write_status(state, **values):
    payload = {"state": state, "time": time.time(), **values}
    temp = STATUS.with_suffix(".tmp")
    temp.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temp, STATUS)
    os.chmod(STATUS, 0o600)


def fill_transition_arrays(seq_array, pos_array, token_array, next_token_array, next_action_array, row_index, events):
    valid = [event for event in events if event["item_id"] is not None and int(event["item_id"]) > 0]
    item_ids = [0] + [int(event["item_id"]) for event in valid]
    action_types = [0] + [0 if event["action_type"] is None else int(event["action_type"]) for event in valid]
    if len(item_ids) < 2:
        return False, np.empty(0, dtype=np.int64)
    length = MAXLEN + 1
    transitions = len(item_ids) - 1
    take = min(length, transitions)
    source_start = transitions - take
    dest_start = length - take
    seq_array[row_index, dest_start:] = np.asarray(item_ids[source_start : source_start + take], dtype=np.int32)
    pos_array[row_index, dest_start:] = np.asarray(item_ids[source_start + 1 : source_start + take + 1], dtype=np.int32)
    token_array[row_index, dest_start:] = 1
    next_token_array[row_index, dest_start:] = 1
    next_action_array[row_index, dest_start:] = np.asarray(
        action_types[source_start + 1 : source_start + take + 1], dtype=np.int8
    )
    if source_start == 0:
        token_array[row_index, dest_start] = 2
    return True, np.asarray(item_ids[1:], dtype=np.int64)


def fill_query_arrays(seq_array, token_array, row_index, events):
    valid = [event for event in events if event["item_id"] is not None and int(event["item_id"]) > 0]
    item_ids = [0] + [int(event["item_id"]) for event in valid]
    length = MAXLEN + 1
    take = min(length, len(item_ids))
    source_start = len(item_ids) - take
    dest_start = length - take
    seq_array[row_index, dest_start:] = np.asarray(item_ids[source_start:], dtype=np.int32)
    token_array[row_index, dest_start:] = 1
    if source_start == 0:
        token_array[row_index, dest_start] = 2


class FeatureCollator:
    def __init__(self, item_features, user_features):
        self.item_features = item_features
        self.user_features = user_features

    def feature_tensors(self, seq, token_type, user_ids):
        features = {
            name: torch.from_numpy(values[seq])
            for name, values in self.item_features.items()
        }
        user_token_rows, user_token_columns = np.where(token_type == 2)
        for name, values in self.user_features.items():
            sequence_values = np.zeros_like(seq, dtype=np.int32)
            sequence_values[user_token_rows, user_token_columns] = values[user_ids[user_token_rows]]
            features[name] = torch.from_numpy(sequence_values)
        return features


class FullTrainDataset(Dataset, FeatureCollator):
    def __init__(self, count, item_count, item_features, user_features, sid_by_item=None):
        FeatureCollator.__init__(self, item_features, user_features)
        length = MAXLEN + 1
        self.user_ids = np.zeros(count, dtype=np.int32)
        self.seq = np.zeros((count, length), dtype=np.int32)
        self.pos = np.zeros((count, length), dtype=np.int32)
        self.token_type = np.zeros((count, length), dtype=np.int8)
        self.next_token_type = np.zeros((count, length), dtype=np.int8)
        self.next_action_type = np.zeros((count, length), dtype=np.int8)
        self.item_counts = np.zeros(item_count + 1, dtype=np.int64)
        self.filled = 0
        self.item_log_p = None
        self.sid_by_item = sid_by_item

    def append(self, row):
        if self.filled >= len(self.user_ids):
            raise RuntimeError("too many training rows")
        success, all_items = fill_transition_arrays(
            self.seq, self.pos, self.token_type, self.next_token_type,
            self.next_action_type, self.filled, row["seq"],
        )
        if not success:
            raise RuntimeError(f"empty training sequence for user {row['user_id']}")
        self.user_ids[self.filled] = int(row["user_id"])
        np.add.at(self.item_counts, all_items, 1)
        self.filled += 1

    def finalize(self):
        if self.filled != len(self.user_ids):
            raise RuntimeError(f"training rows {self.filled} != {len(self.user_ids)}")
        positive = self.item_counts > 0
        total = int(self.item_counts.sum())
        if total <= 0:
            raise RuntimeError("no training interactions")
        minimum = float(np.log(1.0 / total))
        self.item_log_p = np.full(len(self.item_counts), minimum, dtype=np.float32)
        self.item_log_p[positive] = np.log(self.item_counts[positive] / total).astype(np.float32)

    def __len__(self):
        return len(self.user_ids)

    def __getitem__(self, index):
        return index

    def collate(self, indices):
        indices = np.asarray(indices, dtype=np.int64)
        user_ids = self.user_ids[indices]
        seq = self.seq[indices]
        pos = self.pos[indices]
        token_type = self.token_type[indices]
        next_token_type = self.next_token_type[indices]
        next_action_type = self.next_action_type[indices]
        seq_feat = self.feature_tensors(seq, token_type, user_ids)
        pos_feat = {
            name: torch.from_numpy(values[pos])
            for name, values in self.item_features.items()
        }
        for name in self.user_features:
            pos_feat[name] = torch.zeros_like(seq_feat[name])
        shape = seq.shape
        sid = (
            torch.from_numpy(np.asarray(self.sid_by_item[pos], dtype=np.int32))
            if self.sid_by_item is not None
            else torch.zeros((*shape, 2), dtype=torch.int32)
        )
        return (
            torch.from_numpy(seq), torch.from_numpy(pos),
            torch.from_numpy(token_type.astype(np.int32)),
            torch.from_numpy(next_token_type.astype(np.int32)),
            torch.from_numpy(next_action_type.astype(np.int32)),
            seq_feat, pos_feat, torch.zeros(shape, dtype=torch.int32),
            sid,
            torch.from_numpy(self.item_log_p[pos]),
            torch.ones(shape, dtype=torch.float32),
        )


class FormalEvalDataset(Dataset, FeatureCollator):
    def __init__(self, users, targets, prefixes, item_features, user_features):
        FeatureCollator.__init__(self, item_features, user_features)
        length = MAXLEN + 1
        self.user_ids = users.astype(np.int32, copy=True)
        self.targets = targets.astype(np.uint64, copy=True)
        self.prefixes = prefixes.astype(np.int16, copy=True)
        self.seq = np.zeros((len(users), length), dtype=np.int32)
        self.token_type = np.zeros((len(users), length), dtype=np.int8)
        self.row_by_user = {int(user): index for index, user in enumerate(users)}
        self.filled = np.zeros(len(users), dtype=np.bool_)

    def maybe_fill(self, row):
        row_index = self.row_by_user.get(int(row["user_id"]))
        if row_index is None:
            return
        prefix = int(self.prefixes[row_index])
        fill_query_arrays(self.seq, self.token_type, row_index, row["seq"][:prefix])
        self.filled[row_index] = True

    def finalize(self):
        if not bool(self.filled.all()):
            raise RuntimeError(f"missing {int((~self.filled).sum())} evaluation users")

    def __len__(self):
        return len(self.user_ids)

    def __getitem__(self, index):
        return index

    def collate(self, indices):
        indices = np.asarray(indices, dtype=np.int64)
        user_ids = self.user_ids[indices]
        seq = self.seq[indices]
        token_type = self.token_type[indices]
        return (
            torch.from_numpy(seq),
            torch.from_numpy(token_type.astype(np.int32)),
            self.feature_tensors(seq, token_type, user_ids),
            self.targets[indices], self.prefixes[indices], user_ids,
        )


def load_candidate_arrays(candidate_dataset, item_index, contract_raw, feature_statistics):
    table = candidate_dataset.data.table
    raw_ids = pc.fill_null(table["item_id"], 0).combine_chunks().to_numpy(zero_copy_only=False).astype(np.uint64)
    retrieval_ids = pc.fill_null(table["retrieval_id"], -1).combine_chunks().to_numpy(zero_copy_only=False).astype(np.int64)
    order = np.argsort(retrieval_ids)
    if not np.array_equal(retrieval_ids[order], np.arange(len(retrieval_ids), dtype=np.int64)):
        raise RuntimeError("candidate retrieval IDs are not contiguous")
    raw_ids = raw_ids[order]
    if not np.array_equal(raw_ids, contract_raw):
        raise RuntimeError("candidate raw IDs differ from the frozen contract")

    internal_ids = np.fromiter(
        (int(item_index.get(int(value), item_index.get(str(int(value)), 0))) for value in raw_ids),
        dtype=np.int64,
        count=len(raw_ids),
    ).astype(np.int32)
    features = {}
    for name in ITEM_FEATURES:
        struct_values = table[name].combine_chunks()
        cold = pc.fill_null(pc.struct_field(struct_values, "cold_start"), 1)
        text_values = pc.fill_null(pc.struct_field(struct_values, "feature_value"), "0")
        numeric = pc.cast(text_values, pa.int64(), safe=False)
        decoded = pc.if_else(pc.equal(cold, 0), numeric, 0).to_numpy(zero_copy_only=False).astype(np.int32)
        decoded = decoded[order]
        if int(decoded.max(initial=0)) > int(feature_statistics[name]):
            raise RuntimeError(f"candidate feature {name} exceeds training vocabulary")
        features[name] = decoded
    return internal_ids, np.arange(len(raw_ids), dtype=np.uint64), features


def candidate_embeddings(model, internal_ids, candidate_features):
    outputs = []
    model.eval()
    model.set_mode("infer")
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        for start in range(0, len(internal_ids), CANDIDATE_BATCH_SIZE):
            stop = min(start + CANDIDATE_BATCH_SIZE, len(internal_ids))
            ids = torch.from_numpy(internal_ids[start:stop]).to(model.dev).view(-1, 1)
            features = {
                name: torch.from_numpy(values[start:stop]).view(-1, 1)
                for name, values in candidate_features.items()
            }
            embedding = model.feat2emb(ids, features, include_user=False).squeeze(1)
            if model.similarity_function == "cosine":
                embedding = F.normalize(embedding, p=2, dim=-1)
            outputs.append(embedding.float())
    return torch.cat(outputs, dim=0)


def metrics(topk, targets, prefixes):
    matches = topk == targets[:, None]
    hits = matches.any(axis=1)
    ranks = np.where(hits, matches.argmax(axis=1) + 1, 0)
    ndcg_rows = np.zeros(len(targets), dtype=np.float64)
    ndcg_rows[hits] = 1.0 / np.log2(ranks[hits] + 1)

    def summarize(mask):
        count = int(mask.sum())
        hit_count = int(hits[mask].sum())
        hr = hit_count / count
        ndcg = float(ndcg_rows[mask].mean())
        return {
            "evaluated_users": count, "hits": hit_count,
            "hit_rate_at_10": hr, "ndcg_at_10": ndcg,
            "competition_score": 0.31 * hr + 0.69 * ndcg,
        }

    all_mask = np.ones(len(targets), dtype=np.bool_)
    result = summarize(all_mask)
    slices = {}
    for name, mask in {
        "history_0_20": prefixes <= 20,
        "history_21_50": (prefixes >= 21) & (prefixes <= 50),
        "history_51_80": (prefixes >= 51) & (prefixes <= 80),
        "history_81_plus": prefixes >= 81,
    }.items():
        slices[name] = summarize(mask)
    return result, slices


def evaluate(model, eval_dataset, internal_ids, retrieval_ids, candidate_features):
    started = time.time()
    write_status("evaluation", stage="candidates", progress=0)
    candidates = candidate_embeddings(model, internal_ids, candidate_features)
    loader = DataLoader(
        eval_dataset, batch_size=EVAL_BATCH_SIZE, shuffle=False,
        num_workers=0, collate_fn=eval_dataset.collate,
    )
    topk_rows = []
    target_rows = []
    prefix_rows = []
    user_rows = []
    model.eval()
    model.set_mode("infer")
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        for batch_index, (seq, token_type, seq_feat, targets, prefixes, users) in enumerate(loader, 1):
            query, _, _ = model.predict(seq.to(model.dev), seq_feat, token_type.to(model.dev))
            scores = query.float() @ candidates.T
            indices = torch.topk(scores, k=10, dim=1).indices.cpu().numpy()
            topk_rows.append(retrieval_ids[indices])
            target_rows.append(targets)
            prefix_rows.append(prefixes)
            user_rows.append(users)
            if batch_index == 1 or batch_index % 50 == 0 or batch_index == len(loader):
                write_status(
                    "evaluation", stage="queries", batch=batch_index,
                    batches=len(loader), users=min(batch_index * EVAL_BATCH_SIZE, len(eval_dataset)),
                )
    topk = np.concatenate(topk_rows)
    targets = np.concatenate(target_rows)
    prefixes = np.concatenate(prefix_rows)
    users = np.concatenate(user_rows)
    overall, slices = metrics(topk, targets, prefixes)
    return overall, slices, {
        "topk_retrieval_ids": topk,
        "target_retrieval_ids": targets,
        "prefix_lengths": prefixes,
        "user_reids": users,
    }, time.time() - started


def scheduler_factor(step, total_steps):
    if step < WARMUP_STEPS:
        return (step + 1) / WARMUP_STEPS
    progress = (step - WARMUP_STEPS) / max(total_steps - WARMUP_STEPS, 1)
    return 0.5 * (1.0 + math.cos(math.pi * min(max(progress, 0.0), 1.0)))


def save_resume(path, model, optimizer, scheduler, epoch, global_step, history):
    temporary = path.with_suffix(".tmp")
    torch.save(
        {
            "state_dict": model.state_dict(), "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(), "epoch": epoch,
            "global_step": global_step, "history": history,
            "contract_sha256": common.sha256(CONTRACT),
            "indexer_sha256": common.INDEXER_SHA256,
            "architecture": ARCHITECTURE,
            "run_signature": RUN_SIGNATURE,
        },
        temporary,
    )
    os.chmod(temporary, 0o600)
    os.replace(temporary, path)


def main():
    started = time.time()
    os.umask(0o077)
    OUT.mkdir(parents=True, exist_ok=True)
    os.chmod(OUT, 0o700)
    common.set_seed(SEED)
    if common.sha256(INDEXER) != common.INDEXER_SHA256:
        raise RuntimeError("unexpected indexer hash")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("expected exactly one visible CUDA device")
    torch.set_float32_matmul_precision("high")
    with INDEXER.open("rb") as handle:
        indexer = pickle.load(handle)
    with np.load(CONTRACT, allow_pickle=False) as archive:
        contract = {name: archive[name] for name in archive.files}

    os.environ["HF_HOME"] = str(HOME)
    os.environ["HF_DATASETS_CACHE"] = str(HOME / "datasets")
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    write_status("preparing", stage="arrow")
    sequences = load_dataset("TAAC2025/TencentGR-1M", "seq", split="train", cache_dir=str(HOME / "datasets"))
    item_dataset = load_dataset("TAAC2025/TencentGR-1M", "item_feat", split="train", cache_dir=str(HOME / "datasets"))
    user_dataset = load_dataset("TAAC2025/TencentGR-1M", "user_feat", split="train", cache_dir=str(HOME / "datasets"))
    candidate_dataset = load_dataset("TAAC2025/TencentGR-1M", "candidate", split="train", cache_dir=str(HOME / "datasets"))
    user_count = len(indexer["u"])
    item_count = len(indexer["i"])
    if len(sequences) != user_count or len(item_dataset) != item_count or len(candidate_dataset) != 660000:
        raise RuntimeError("dataset/indexer count mismatch")

    write_status("preparing", stage="features")
    item_features, _ = common.load_dense_features(item_dataset, "item_id", ITEM_FEATURES, item_count)
    user_features, _ = common.load_dense_features(user_dataset, "user_id", USER_FEATURES, user_count)
    feature_statistics = {name: len(indexer["f"][name]) for name in ITEM_FEATURES + USER_FEATURES}
    for name, values in {**item_features, **user_features}.items():
        if int(values.max(initial=0)) > int(feature_statistics[name]):
            raise RuntimeError(f"feature {name} exceeds indexer vocabulary")
    feature_types = {
        "user_sparse": USER_FEATURES, "user_array": [], "user_continual": [],
        "item_sparse": ITEM_FEATURES, "context_item_sparse": [],
        "item_array": [], "item_continual": [], "item_emb": [],
    }

    split_generator = torch.Generator().manual_seed(SPLIT_SEED)
    train_subset, validation_subset = torch.utils.data.random_split(
        range(user_count), [0.9, 0.1], generator=split_generator
    )
    validation_users = {int(value) + 1 for value in validation_subset.indices}
    if not set(map(int, contract["eval_user_reids"])).issubset(validation_users):
        raise RuntimeError("evaluation contract does not align with validation split")
    sid_by_item = None
    if ENABLE_SID:
        sid_by_item = np.load(SID_PATH, mmap_mode="r", allow_pickle=False)
        if sid_by_item.shape != (item_count + 1, 2) or sid_by_item.dtype != np.uint16:
            raise RuntimeError("SID array shape or dtype mismatch")
        if bool(np.any(sid_by_item[0] != 0)):
            raise RuntimeError("SID padding row must be zero")
        if int(sid_by_item[1:].min()) < 1 or int(sid_by_item[1:].max()) > SID_CODEBOOK_SIZE:
            raise RuntimeError("SID values fall outside the configured codebook")
    train_dataset = FullTrainDataset(
        len(train_subset), item_count, item_features, user_features, sid_by_item=sid_by_item
    )
    eval_dataset = FormalEvalDataset(
        contract["eval_user_reids"], contract["eval_target_retrieval_ids"],
        contract["eval_prefix_lengths"], item_features, user_features,
    )
    write_status("preparing", stage="sequences", processed=0, total=len(sequences))
    for row_index, row in enumerate(sequences, 1):
        user_id = int(row["user_id"])
        if user_id not in validation_users:
            train_dataset.append(row)
        eval_dataset.maybe_fill(row)
        if row_index % 100000 == 0:
            write_status("preparing", stage="sequences", processed=row_index, total=len(sequences))
    train_dataset.finalize()
    eval_dataset.finalize()
    internal_ids, retrieval_ids, candidate_features = load_candidate_arrays(
        candidate_dataset, indexer["i"], contract["candidate_raw_creative_ids"], feature_statistics,
    )

    sys.path.insert(0, str(common.SOURCE_DIR))
    from model import BaselineModel

    args = common.make_args()
    args.maxlen = MAXLEN
    args.hidden_units = HIDDEN_UNITS
    args.num_blocks = NUM_BLOCKS
    args.num_heads = NUM_HEADS
    args.dropout_rate = 0.1
    args.log_interval = 1_000_000
    args.use_hstu = ARCHITECTURE == "hstu"
    args.sid = ENABLE_SID
    args.sid_codebook_size = SID_CODEBOOK_SIZE
    write_status("preparing", stage="model")
    model = BaselineModel(user_count, item_count, feature_statistics, feature_types, args).to(args.device)
    model._metric_counter = 100
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    steps_per_epoch = len(train_dataset) // BATCH_SIZE
    total_steps = steps_per_epoch * EPOCHS
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=lambda step: scheduler_factor(step, total_steps)
    )
    parameters = sum(parameter.numel() for parameter in model.parameters())
    resume_path = OUT / "resume.pt"
    start_epoch = 1
    global_step = 0
    history = []
    if resume_path.exists():
        checkpoint = torch.load(resume_path, map_location="cpu", weights_only=False)
        if checkpoint["contract_sha256"] != common.sha256(CONTRACT):
            raise RuntimeError("resume contract mismatch")
        if checkpoint.get("architecture") != ARCHITECTURE:
            raise RuntimeError("resume architecture mismatch")
        if checkpoint.get("run_signature") != RUN_SIGNATURE:
            raise RuntimeError("resume configuration mismatch")
        model.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        start_epoch = int(checkpoint["epoch"]) + 1
        global_step = int(checkpoint["global_step"])
        history = checkpoint["history"]
        del checkpoint

    training_started = time.time()
    for epoch in range(start_epoch, EPOCHS + 1):
        epoch_started = time.time()
        generator = torch.Generator().manual_seed(SEED + epoch)
        loader = DataLoader(
            train_dataset, batch_size=BATCH_SIZE, shuffle=True, generator=generator,
            num_workers=0, collate_fn=train_dataset.collate, drop_last=True,
        )
        model.train()
        model.set_mode("train")
        losses = []
        write_status(
            "training", epoch=epoch, epochs=EPOCHS, step=0,
            steps=len(loader), global_step=global_step, gpu=PHYSICAL_GPU,
        )
        for step, batch in enumerate(loader, 1):
            (
                seq, pos, token_type, next_token_type, next_action_type,
                seq_feat, pos_feat, _action_type, sid, pos_log_p, ranking_loss_mask,
            ) = batch
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                loss, _ = model.forward_train(
                    seq.to(args.device), pos.to(args.device), token_type,
                    next_token_type, next_action_type.to(args.device), seq_feat,
                    pos_feat, sid.to(args.device), pos_log_p.to(args.device),
                    ranking_loss_mask.to(args.device), args=args, dataset=None,
                )
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite loss at epoch {epoch} step {step}")
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if not torch.isfinite(grad_norm):
                raise RuntimeError(f"non-finite gradient norm at epoch {epoch} step {step}")
            optimizer.step()
            scheduler.step()
            global_step += 1
            losses.append(float(loss.detach().cpu()))
            if step == 1 or step % LOG_EVERY == 0 or step == len(loader):
                elapsed = time.time() - epoch_started
                record = {
                    "epoch": epoch, "epochs": EPOCHS, "step": step,
                    "steps": len(loader), "global_step": global_step,
                    "loss": losses[-1], "mean_loss": float(np.mean(losses)),
                    "learning_rate": optimizer.param_groups[0]["lr"],
                    "users_per_second": step * BATCH_SIZE / elapsed,
                    "peak_memory_mib": torch.cuda.max_memory_allocated() / 1024**2,
                }
                print(json.dumps(record, sort_keys=True), flush=True)
                write_status(
                    "training", gpu=PHYSICAL_GPU, architecture=ARCHITECTURE,
                    experiment_id=EXPERIMENT_ID, **record,
                )
        epoch_record = {
            "epoch": epoch, "mean_loss": float(np.mean(losses)),
            "last_loss": losses[-1], "seconds": time.time() - epoch_started,
            "learning_rate": optimizer.param_groups[0]["lr"],
        }
        history.append(epoch_record)
        save_resume(resume_path, model, optimizer, scheduler, epoch, global_step, history)
        print(json.dumps({"checkpoint": epoch_record}, sort_keys=True), flush=True)

    overall, slices, predictions, evaluation_seconds = evaluate(
        model, eval_dataset, internal_ids, retrieval_ids, candidate_features
    )
    result = {
        "status": "pass", "metrics": overall, "slices": slices,
        "experiment_id": EXPERIMENT_ID,
        "run_signature": RUN_SIGNATURE,
        "evaluation_seconds": evaluation_seconds,
        "audit": {
            "training_users": len(train_dataset), "validation_users": len(validation_users),
            "eligible_eval_users": len(eval_dataset), "candidates": len(retrieval_ids),
            "cold_start_candidates": int((internal_ids == 0).sum()),
        },
        "model": {
            "architecture": "HSTU" if args.use_hstu else "Transformer", "num_blocks": args.num_blocks,
            "hidden_units": args.hidden_units, "num_heads": args.num_heads,
            "maxlen": args.maxlen,
            "parameters": parameters,
            "loss": (
                "sample-bias-corrected InfoNCE + SID1/SID2 cross-entropy"
                if ENABLE_SID else "sample-bias-corrected InfoNCE"
            ),
            "sid_auxiliary": ENABLE_SID,
            "sid_codebook_size": SID_CODEBOOK_SIZE if ENABLE_SID else None,
        },
        "training": {
            "epochs": EPOCHS, "batch_size": BATCH_SIZE,
            "global_steps": global_step, "history": history,
            "seconds": time.time() - training_started,
        },
        "resources": {
            "device": torch.cuda.get_device_name(0), "physical_gpu": PHYSICAL_GPU,
            "peak_memory_mib": torch.cuda.max_memory_allocated() / 1024**2,
            "wall_seconds": time.time() - started,
        },
        "protocol": {
            "split_seed": SPLIT_SEED, "training_seed": SEED,
            "contract_sha256": common.sha256(CONTRACT),
            "indexer_sha256": common.INDEXER_SHA256,
            "history_filtering": False,
            "sid_mapping_sha256": SID_SHA256,
        },
    }
    model_path = OUT / "model.pt"
    temporary_model = OUT / "model.tmp"
    torch.save({"state_dict": model.state_dict(), "result": result}, temporary_model)
    os.chmod(temporary_model, 0o600)
    os.replace(temporary_model, model_path)
    metrics_path = OUT / "offline_metrics.json"
    metrics_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    predictions_path = OUT / "offline_predictions.npz"
    np.savez_compressed(predictions_path, **predictions)
    history_path = OUT / "history.json"
    history_path.write_text(json.dumps(history, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    config_path = OUT / "config.json"
    config_path.write_text(
        json.dumps({key: value for key, value in vars(args).items() if isinstance(value, (str, int, float, bool, list, type(None)))}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    names = ["model.pt", "offline_metrics.json", "offline_predictions.npz", "history.json", "config.json"]
    manifest_path = OUT / "SHA256SUMS"
    manifest_path.write_text(
        "".join(f"{common.sha256(OUT / name)}  {name}\n" for name in names),
        encoding="utf-8",
    )
    for path in OUT.iterdir():
        os.chmod(path, 0o600)
    if resume_path.exists():
        resume_path.unlink()
    write_status(
        "complete", gpu=PHYSICAL_GPU, architecture=ARCHITECTURE,
        experiment_id=EXPERIMENT_ID, **result,
    )
    print(json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        write_status(
            "failed", gpu=PHYSICAL_GPU, architecture=ARCHITECTURE,
            experiment_id=EXPERIMENT_ID, error=repr(error),
        )
        raise
