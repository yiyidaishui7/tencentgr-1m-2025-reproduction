# Attribution

This reproduction is based on the official **2025 Tencent Ads Algorithm
Competition Baseline**:

- Source: https://github.com/TencentAdvertisingAlgorithmCompetition/baseline_2025
- License: Creative Commons Attribution-NonCommercial 4.0 International

The evaluation uses **TencentGR-1M**:

- Dataset: https://huggingface.co/datasets/TAAC2025/TencentGR-1M
- Publisher: TAAC2025 / Tencent Advertising Algorithm Competition
- License: Creative Commons Attribution 4.0 International

Modifications in this reproduction include runtime portability, deterministic
data splitting, candidate-ID auditing, offline Top-10 evaluation, ANN-related
tests, checkpoint synchronization, and a no-multimodal-feature ablation.

The advanced architecture study references **OnePiece**:

- Source: https://github.com/shuoyang2/OnePiece
- Frozen revision: `73e51021dfafb75382baf9acd6a72ce47e5b705b`
- Technical report: https://arxiv.org/abs/2512.07424

The upstream OnePiece repository is not vendored or relicensed here. The
public `patches/onepiece-runtime-fixes.patch` contains only the minimal runtime
interoperability changes needed by this reproduction; users obtain the
upstream source separately and remain responsible for its terms.
