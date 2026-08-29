"""Pure training-contract helpers shared by OnePiece reproduction runners."""

from __future__ import annotations


# Frozen upstream OnePiece uses 2025-05-31 00:00:00 Asia/Shanghai
# (2025-05-30 16:00:00 UTC) as its post-cutoff boundary.
POST_CUTOFF_EXPOSURE_TIMESTAMP = 1_748_620_800


def ranking_loss_weight(
    next_timestamp: int | None,
    *,
    enabled: bool,
    cutoff_timestamp: int = POST_CUTOFF_EXPOSURE_TIMESTAMP,
) -> float:
    """Return the upstream-compatible ranking weight for one transition."""

    if not enabled or next_timestamp is None:
        return 1.0
    return 0.0 if int(next_timestamp) > cutoff_timestamp else 1.0


def effective_next_token_type(next_token_type: int, ranking_weight: float) -> int:
    """Apply a ranking weight to the next-item mask consumed by InfoNCE/SID."""

    return int(next_token_type) if float(ranking_weight) > 0.0 else 0
