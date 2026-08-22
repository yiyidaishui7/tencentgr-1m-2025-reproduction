from __future__ import annotations

import argparse
import json
import os
import shutil
import time
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pyarrow.dataset as ds
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from candidate_utils import candidate_item_column, canonical_id_key
from dataset import MyTestDataset
from infer import (
    _read_fbin,
    _read_u64bin,
    exact_inner_product_topk,
    get_candidate_emb_parquet,
)
from model import BaselineModel
from offline_eval_utils import competition_metrics, select_last_click_holdout
from runtime_utils import load_model_state_dict, resolve_device, uses_accelerator


@dataclass(frozen=True)
class EvalSample:
    dataset_index: int
    user_id: int
    target_position: int
    target_retrieval_id: int
    prefix_length: int


def _candidate_retrieval_ids(data_path: Path) -> dict[str, int]:
    candidate_dataset = ds.dataset(str(data_path / "candidate"), format="parquet")
    item_column = candidate_item_column(set(candidate_dataset.schema.names))
    scanner = candidate_dataset.scanner(
        columns=[item_column, "retrieval_id"],
        batch_size=100_000,
    )

    retrieval_ids: dict[str, int] = {}
    for batch in scanner.to_batches():
        raw_items = batch.column(item_column).to_pylist()
        raw_retrieval_ids = batch.column("retrieval_id").to_pylist()
        for raw_item, retrieval_id in zip(raw_items, raw_retrieval_ids):
            if raw_item is not None and retrieval_id is not None:
                retrieval_ids[canonical_id_key(raw_item)] = int(retrieval_id)
    return retrieval_ids


def _validation_indices(
    dataset_size: int,
    *,
    valid_ratio: float,
    seed: int,
) -> list[int]:
    if not 0.0 < valid_ratio < 1.0:
        raise ValueError("valid_ratio must be between 0 and 1")
    generator = torch.Generator().manual_seed(seed)
    _, valid_subset = torch.utils.data.random_split(
        range(dataset_size),
        [1.0 - valid_ratio, valid_ratio],
        generator=generator,
    )
    return sorted(int(index) for index in valid_subset.indices)


class LastClickEvalDataset(MyTestDataset):
    """Temporal holdout over users excluded from baseline training."""

    def __init__(
        self,
        data_path: Path,
        args: argparse.Namespace,
        *,
        valid_ratio: float,
        seed: int,
        max_users: int,
    ) -> None:
        super().__init__(data_path, args)
        candidate_ids = _candidate_retrieval_ids(data_path)
        validation_indices = _validation_indices(
            len(self),
            valid_ratio=valid_ratio,
            seed=seed,
        )

        audit = Counter(validation_users=len(validation_indices))
        samples: list[EvalSample] = []
        for dataset_index in validation_indices:
            user_id = dataset_index + 1
            try:
                start, length = self.user_indices[user_id]
            except KeyError:
                audit["missing_user_sequence"] += 1
                continue

            events = self.full_events[start : start + length]
            holdout = select_last_click_holdout(
                events["item_id"],
                events["action_type"],
            )
            if holdout is None:
                audit["users_without_click"] += 1
                continue

            audit["users_with_click"] += 1
            target_raw_id = self.indexer_i_rev.get(holdout.target_item_id)
            if target_raw_id is None:
                audit["target_missing_from_indexer"] += 1
                continue

            target_retrieval_id = candidate_ids.get(canonical_id_key(target_raw_id))
            if target_retrieval_id is None:
                audit["target_missing_from_candidate_pool"] += 1
                continue

            samples.append(
                EvalSample(
                    dataset_index=dataset_index,
                    user_id=user_id,
                    target_position=holdout.target_position,
                    target_retrieval_id=target_retrieval_id,
                    prefix_length=holdout.prefix_length,
                )
            )

        audit["eligible_users"] = len(samples)
        if max_users > 0:
            samples = samples[:max_users]
        audit["evaluated_users"] = len(samples)
        if not samples:
            raise RuntimeError("No eligible last-click holdout users were found")

        self.samples = samples
        self.audit = dict(audit)

    def __len__(self) -> int:
        if hasattr(self, "samples"):
            return len(self.samples)
        return super().__len__()

    def __getitem__(self, sample_index: int):
        sample = self.samples[sample_index]
        start, _ = self.user_indices[sample.user_id]
        prefix_events = self.full_events[start : start + sample.target_position]

        user_features = self._process_cold_start_feat(
            self.user_feat_dict.get(str(sample.user_id), {})
        )
        sequence_records: list[tuple[int, dict, int]] = [
            (sample.user_id, user_features, 2)
        ]
        for event in prefix_events:
            item_id = int(event["item_id"])
            if item_id == 0:
                continue
            item_features = self._process_cold_start_feat(
                self.item_feat_dict.get(str(item_id), {})
            )
            sequence_records.append((item_id, item_features, 1))

        seq = np.zeros(self.maxlen + 1, dtype=np.int32)
        token_type = np.zeros(self.maxlen + 1, dtype=np.int32)
        seq_feat = np.empty(self.maxlen + 1, dtype=object)
        seq_feat[:] = None

        output_index = self.maxlen
        for item_id, features, record_type in reversed(sequence_records):
            seq[output_index] = item_id
            token_type[output_index] = record_type
            seq_feat[output_index] = self.fill_missing_feat(features, item_id)
            output_index -= 1
            if output_index < 0:
                break

        for index in range(self.maxlen + 1):
            if seq_feat[index] is None:
                seq_feat[index] = self.feature_default_value

        return (
            seq,
            token_type,
            seq_feat,
            sample.target_retrieval_id,
            sample.prefix_length,
            sample.user_id,
        )

    @staticmethod
    def collate_fn(batch):
        seq, token_type, seq_feat, targets, prefix_lengths, user_ids = zip(*batch)
        return (
            torch.from_numpy(np.asarray(seq)),
            torch.from_numpy(np.asarray(token_type)),
            list(seq_feat),
            np.asarray(targets, dtype=np.uint64),
            np.asarray(prefix_lengths, dtype=np.int16),
            np.asarray(user_ids, dtype=np.int32),
        )


def _slice_metrics(
    topk_ids: np.ndarray,
    target_ids: np.ndarray,
    prefix_lengths: np.ndarray,
) -> dict[str, dict[str, int | float]]:
    boundaries = {
        "history_0_20": (0, 20),
        "history_21_50": (21, 50),
        "history_51_80": (51, 80),
        "history_81_plus": (81, None),
    }
    slices: dict[str, dict[str, int | float]] = {}
    for name, (lower, upper) in boundaries.items():
        mask = prefix_lengths >= lower
        if upper is not None:
            mask &= prefix_lengths <= upper
        if mask.any():
            slices[name] = competition_metrics(topk_ids[mask], target_ids[mask])
    return slices


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output_dir", required=True, type=Path)
    parser.add_argument("--scratch_dir", required=True, type=Path)
    parser.add_argument("--batch_size", default=1024, type=int)
    parser.add_argument("--num_workers", default=4, type=int)
    parser.add_argument("--retrieval_batch_size", default=2048, type=int)
    parser.add_argument("--valid_ratio", default=0.1, type=float)
    parser.add_argument("--seed", default=2025, type=int)
    parser.add_argument("--max_users", default=0, type=int)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--maxlen", default=101, type=int)
    parser.add_argument("--hidden_units", default=32, type=int)
    parser.add_argument("--num_blocks", default=1, type=int)
    parser.add_argument("--num_heads", default=1, type=int)
    parser.add_argument("--dropout_rate", default=0.2, type=float)
    parser.add_argument("--l2_emb", default=0.0, type=float)
    parser.add_argument("--norm_first", action="store_true")
    parser.add_argument(
        "--mm_emb_id",
        nargs="+",
        default=["81"],
        choices=[str(value) for value in range(81, 87)],
    )
    parser.add_argument("--disable_mm_emb", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = get_args()
    args.device = resolve_device(args.device, torch)
    args.data_path = args.data_path.resolve()
    args.checkpoint = args.checkpoint.resolve()
    args.output_dir = args.output_dir.resolve()
    args.scratch_dir = args.scratch_dir.resolve()
    if not args.data_path.is_dir():
        raise FileNotFoundError(f"Dataset path does not exist: {args.data_path}")
    if not args.checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {args.checkpoint}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.scratch_dir.exists():
        shutil.rmtree(args.scratch_dir)
    args.scratch_dir.mkdir(parents=True)

    started_at = time.time()
    try:
        dataset = LastClickEvalDataset(
            args.data_path,
            args,
            valid_ratio=args.valid_ratio,
            seed=args.seed,
            max_users=args.max_users,
        )
        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=dataset.collate_fn,
            pin_memory=uses_accelerator(args.device),
        )

        model = BaselineModel(
            dataset.usernum,
            dataset.itemnum,
            dataset.feat_statistics,
            dataset.feature_types,
            args,
        ).to(args.device)
        state_dict = load_model_state_dict(args.checkpoint, args.device, torch)
        model.load_state_dict(state_dict)
        model.eval()

        os.environ["EVAL_DATA_PATH"] = str(args.data_path)
        os.environ["EVAL_RESULT_PATH"] = str(args.scratch_dir)
        get_candidate_emb_parquet(
            dataset.indexer["i"],
            dataset.feature_types,
            dataset.feature_default_value,
            dataset.mm_emb_dict,
            model,
        )

        query_batches: list[np.ndarray] = []
        target_batches: list[np.ndarray] = []
        prefix_batches: list[np.ndarray] = []
        user_batches: list[np.ndarray] = []
        with torch.no_grad():
            for seq, token_type, seq_feat, targets, prefix_lengths, user_ids in tqdm(
                loader,
                desc="Encoding holdout queries",
            ):
                query = model.predict(
                    seq.to(args.device),
                    seq_feat,
                    token_type.to(args.device),
                )
                query_batches.append(query.detach().cpu().numpy().astype(np.float32))
                target_batches.append(targets)
                prefix_batches.append(prefix_lengths)
                user_batches.append(user_ids)

        query_vectors = np.concatenate(query_batches)
        target_ids = np.concatenate(target_batches)
        prefix_lengths = np.concatenate(prefix_batches)
        user_ids = np.concatenate(user_batches)
        candidate_vectors = _read_fbin(args.scratch_dir / "embedding.fbin")
        candidate_ids = _read_u64bin(args.scratch_dir / "id.u64bin")
        topk_ids = exact_inner_product_topk(
            query_vectors,
            candidate_vectors,
            candidate_ids,
            device=args.device,
            top_k=10,
            batch_size=args.retrieval_batch_size,
        )

        metrics = competition_metrics(topk_ids, target_ids)
        report = {
            "evaluation_contract": {
                "split": "same seeded 90/10 user split as baseline training",
                "seed": args.seed,
                "valid_ratio": args.valid_ratio,
                "target": "last action_type=1 item for each validation user",
                "history": "events strictly before the held-out click",
                "candidate_pool": "official TencentGR-1M 660k candidates",
                "exclusions": "users without a click or whose target is absent from candidates",
            },
            "model": {
                "checkpoint": args.checkpoint.name,
                "disable_mm_emb": args.disable_mm_emb,
                "mm_emb_id": args.mm_emb_id,
                "device": args.device,
            },
            "audit": dataset.audit,
            "metrics": metrics,
            "slices": _slice_metrics(topk_ids, target_ids, prefix_lengths),
            "runtime_seconds": time.time() - started_at,
        }
        with (args.output_dir / "offline_metrics.json").open("w", encoding="utf-8") as file_obj:
            json.dump(report, file_obj, ensure_ascii=False, indent=2)
        np.savez_compressed(
            args.output_dir / "offline_predictions.npz",
            topk_retrieval_ids=topk_ids,
            target_retrieval_ids=target_ids,
            prefix_lengths=prefix_lengths,
            user_reids=user_ids,
        )
        print(json.dumps(report, ensure_ascii=False, indent=2))
    finally:
        shutil.rmtree(args.scratch_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
