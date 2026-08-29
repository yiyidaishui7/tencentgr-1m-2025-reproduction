# OnePiece HSTU/Transformer reproduction runbook

This runbook describes the public, path-neutral version of the formal runner.
It does not include dataset files, upstream model source, checkpoints, or
machine-specific credentials.

## 1. Freeze upstream and apply the runtime patch

```bash
git clone https://github.com/shuoyang2/OnePiece.git
cd OnePiece
git checkout 73e51021dfafb75382baf9acd6a72ce47e5b705b
git apply /path/to/tencentgr-reproduction/patches/onepiece-runtime-fixes.patch
```

The patch removes import-time dependency installation and supplies the missing
`interest_k` argument in two metric calls. Install dependencies into an
isolated environment before importing the upstream modules.

Before a run, `onepiece_source_contract.py` canonicalizes line endings and
verifies `model.py`, `dataset.py`, `utils.py`, and `deepseek_moe.py` against
the frozen commit. The runner also signs those hashes, the runtime patch, its
own implementation modules, the offline contract, and the indexer into
`run_signature`. A checkpoint created by any different signature is rejected.

## 2. Prepare the immutable inputs

- Download `TAAC2025/TencentGR-1M` and its `indexer.pkl` with
  `huggingface_hub`; record the indexer SHA-256.
- Build an offline contract from at least two already row-aligned prediction
  files and the official retrieval-ID mapping:

```bash
python scripts/build_onepiece_protocol.py \
  --predictions /evidence/run_a/offline_predictions.npz \
                /evidence/run_b/offline_predictions.npz \
  --retrieval-map /evidence/retrive_id2creative_id.json \
  --output /evidence/onepiece_protocol
```

The formal experiment documented in this repository additionally audited each
of its 78,921 held-out targets against the original sequence data. Reusing the
provided private contract is preferable to regenerating it from fewer sources.

## 3. Configure a single run

Create private work directories and export paths. The public runner never
selects a physical GPU by itself; `CUDA_VISIBLE_DEVICES` is authoritative, and
the physical index is recorded separately for provenance.

```bash
mkdir -p /private/work/{l,o,hf_home}
export ONEPIECE_WORK_ROOT=/private/work
export ONEPIECE_SOURCE_DIR=/path/to/OnePiece/code
export ONEPIECE_CONTRACT=/evidence/onepiece_protocol/offline_contract.npz
export ONEPIECE_INDEXER=/evidence/indexer.pkl
export ONEPIECE_INDEXER_SHA256='<sha256>'
export ONEPIECE_TRAINING_SEED=20250825
export HF_HOME=/private/work/hf_home
export HF_DATASETS_CACHE=/private/work/hf_home/datasets
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
```

Run HSTU:

```bash
CUDA_VISIBLE_DEVICES=0 \
ONEPIECE_PHYSICAL_GPU=0 \
ONEPIECE_VARIANT=hstu \
python -u scripts/run_onepiece_formal.py
```

Run the controlled Transformer variant in a separate work root or on another
available GPU:

```bash
CUDA_VISIBLE_DEVICES=1 \
ONEPIECE_PHYSICAL_GPU=1 \
ONEPIECE_VARIANT=transformer \
python -u scripts/run_onepiece_formal.py
```

The two JSON configs under `configs/` make the intended equality of all
non-architecture settings explicit.

For the capacity-scaling runs, keep the frozen data/evaluation protocol and
set only the model dimensions plus a unique private output identifier:

```bash
CUDA_VISIBLE_DEVICES=0 \
ONEPIECE_PHYSICAL_GPU=0 \
ONEPIECE_ARCHITECTURE=hstu \
ONEPIECE_EXPERIMENT_ID=hstu_8x256 \
ONEPIECE_HIDDEN_UNITS=256 \
ONEPIECE_NUM_BLOCKS=8 \
ONEPIECE_NUM_HEADS=8 \
python -u scripts/run_onepiece_formal.py

CUDA_VISIBLE_DEVICES=1 \
ONEPIECE_PHYSICAL_GPU=1 \
ONEPIECE_ARCHITECTURE=hstu \
ONEPIECE_EXPERIMENT_ID=hstu_8x512 \
ONEPIECE_HIDDEN_UNITS=512 \
ONEPIECE_NUM_BLOCKS=8 \
ONEPIECE_NUM_HEADS=8 \
python -u scripts/run_onepiece_formal.py
```

The runner hashes the effective architecture, seed, split, batch, schedule,
and evaluation settings into `run_signature`; a resume checkpoint with a
different signature is rejected.

To run a frozen model-derived Semantic ID auxiliary-loss ablation, provide a
NumPy `uint16` array with shape `[item_vocabulary + 1, 2]`. Row zero is the
padding SID; every real item row must contain values in `[1, codebook_size]`.
The array SHA-256 is included in the run signature and final protocol record.

```bash
ONEPIECE_ENABLE_SID=1 \
ONEPIECE_SID_PATH=/evidence/sid_by_internal_id.npy \
ONEPIECE_SID_CODEBOOK_SIZE=4096 \
ONEPIECE_SID_LOSS_WEIGHT=0.02 \
ONEPIECE_SID_LOSS_WARMUP_STEPS=28176 \
ONEPIECE_SID_LOSS_DELAY_STEPS=0 \
ONEPIECE_ENABLE_BEAM_EVAL=1 \
ONEPIECE_BEAM_SIZE=20 \
ONEPIECE_BEAM_TOP_K=384 \
ONEPIECE_EXPERIMENT_ID=hstu_8x512_sid_002 \
python -u scripts/run_onepiece_formal.py
```

The two alignment configs under `configs/onepiece_hstu_8x512_sid_*.json`
freeze the 0.02/0.05 single-variable comparison. Beam evaluation first decodes
SID pairs, rejects pairs absent from the collision-free item mapping, and then
reranks legal items with the same two-tower score used by full-candidate Top-10.

The no-SID reference and SID run must otherwise use the same architecture,
seed, split, batch, optimizer, schedule, epochs and exact evaluation contract.

The archived collision-free SID experiment is a model-derived, global
residual two-level approximation. It is not a strict implementation of the
README's within-L1 L2 clustering rule: capacity repair moved 766 items across
L1 clusters, so it does not preserve the documented `sid1 fixed` invariant.
Keep that limitation in any downstream claim.

### Post-cutoff exposure-mask ablation

The frozen README excludes ranking transitions strictly after
`2025-05-31 00:00 Asia/Shanghai`, represented here by Unix timestamp
`1748620800`. In the frozen `reward=False` code path, `ranking_loss_mask`
does not reach InfoNCE, so the aligned runner applies the same intent to the
`next_token_type` mask that InfoNCE actually consumes.

```bash
ONEPIECE_ENABLE_POST_CUTOFF_EXPOSURE_MASK=1 \
ONEPIECE_POST_CUTOFF_EXPOSURE_TIMESTAMP=1748620800 \
ONEPIECE_EXPERIMENT_ID=hstu_8x512_post_cutoff_mask \
ONEPIECE_HIDDEN_UNITS=512 \
ONEPIECE_NUM_BLOCKS=8 \
ONEPIECE_NUM_HEADS=8 \
python -u scripts/run_onepiece_formal.py
```

Treat this as a single-variable training comparison only when its control is
evaluated by the same candidate post-processing version.

### Aligned candidate evaluation

The runner first validates all 660,000 official candidate rows against the
frozen contract, then excludes the 148,971 candidates absent from the training
item index. The active pool therefore contains 511,029 warm candidates while
preserving their original retrieval IDs. Every one of the 78,921 targets must
remain in that pool or evaluation fails closed.

For each user, all non-padding items in the truncated query history are masked
before Top-10. SID Beam candidates use the same history-filtered two-tower
score; when fewer than ten legal SID candidates remain, ANN fallback fills the
row in score order without duplicates or sentinel IDs. Outputs must contain
exactly ten warm, unseen retrieval IDs per user.

Scores already published in the result documents used the historical
660,000-candidate, no-history-filter protocol. They remain reproducible
historical evidence but are not relabeled as results from this aligned
post-processing contract.

## 4. Acceptance gates

Each completed output must contain `model.pt`, `offline_metrics.json`,
`offline_predictions.npz`, `history.json`, `config.json`, and `SHA256SUMS`.

```bash
python scripts/verify_onepiece_artifact.py \
  --artifact-dir /evidence/hstu \
  --contract /evidence/onepiece_protocol/offline_contract.npz

python scripts/verify_onepiece_artifact.py \
  --artifact-dir /evidence/transformer \
  --contract /evidence/onepiece_protocol/offline_contract.npz
```

Then compute the row-aligned architecture and baseline comparison:

```bash
python scripts/compare_onepiece_architectures.py \
  --hstu /evidence/hstu \
  --transformer /evidence/transformer \
  --baseline mm101=/evidence/mm101 \
  --baseline nomm101=/evidence/nomm101 \
  --baseline mm50=/evidence/mm50 \
  --baseline nomm50=/evidence/nomm50 \
  --output /evidence/onepiece_architecture_comparison.json
```

Do not interpret the result as an official leaderboard score. The comparison
is valid only for the frozen local last-click protocol and does not estimate
training-seed uncertainty.
