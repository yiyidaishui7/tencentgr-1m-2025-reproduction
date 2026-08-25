"""Render the verified OnePiece/baseline comparison as a Chinese Markdown report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


LABELS = {
    "mm101": "Baseline MM101",
    "nomm101": "Baseline no-MM101",
    "mm50": "Baseline MM50",
    "nomm50": "Baseline no-MM50",
    "hstu": "OnePiece HSTU 4×128",
    "transformer": "OnePiece Transformer 4×128",
}
ORDER = ("mm101", "nomm101", "mm50", "nomm50", "hstu", "transformer")


def number(value: float, digits: int = 7) -> str:
    return f"{value:.{digits}f}"


def percent(value: float) -> str:
    return f"{value * 100:+.2f}%"


def render(comparison: dict, formal: dict[str, dict]) -> str:
    overall = comparison["overall"]
    missing = [name for name in ORDER if name not in overall]
    if missing:
        raise RuntimeError(f"comparison is missing runs: {missing}")
    if comparison["row_alignment"]["rows"] != 78_921:
        raise RuntimeError("unexpected evaluation row count")

    lines = [
        "# OnePiece 正式复现结果",
        "",
        "> 本页由通过 SHA-256 与冻结行协议验收的产物自动生成。所有分数均为本地最后点击留出，",
        "> 不是官方排行榜成绩；每个架构只有一个训练 seed。",
        "",
        "## 六组同协议结果",
        "",
        "| 变体 | Hits@10 | HR@10 | NDCG@10 | 综合分 |",
        "|---|---:|---:|---:|---:|",
    ]
    for name in ORDER:
        metrics = overall[name]
        lines.append(
            f"| {LABELS[name]} | {metrics['hits']:,} | "
            f"{number(metrics['hit_rate_at_10'])} | {number(metrics['ndcg_at_10'])} | "
            f"{number(metrics['competition_score'])} |"
        )

    hstu_score = overall["hstu"]["competition_score"]
    transformer_score = overall["transformer"]["competition_score"]
    baseline_score = overall["nomm50"]["competition_score"]
    winner = "hstu" if hstu_score >= transformer_score else "transformer"
    arch_effect = comparison["transformer_minus_hstu"]["competition_score"]
    lines.extend(
        [
            "",
            "## 严格架构对照",
            "",
            "| 对比 | 综合分差值 | 相对差值 | 固定评测人口 95% 区间 |",
            "|---|---:|---:|---:|",
            "| Transformer − HSTU | "
            f"{number(arch_effect['mean_delta'])} | "
            f"{percent(arch_effect['mean_delta'] / hstu_score)} | "
            f"[{number(arch_effect['normal_95_low'])}, {number(arch_effect['normal_95_high'])}] |",
        ]
    )
    for candidate in ("hstu", "transformer"):
        effect = comparison["onepiece_vs_baselines"][
            f"{candidate}_minus_nomm50"
        ]["competition_score"]
        lines.append(
            f"| {LABELS[candidate]} − Baseline no-MM50 | "
            f"{number(effect['mean_delta'])} | {percent(effect['mean_delta'] / baseline_score)} | "
            f"[{number(effect['normal_95_low'])}, {number(effect['normal_95_high'])}] |"
        )

    lines.extend(
        [
            "",
            "## 训练与资源",
            "",
            "| 变体 | 参数量 | Epoch | 全局 Step | 训练耗时 | 精确评测耗时 | 峰值显存 |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for name in ("hstu", "transformer"):
        result = formal[name]
        lines.append(
            f"| {LABELS[name]} | {result['model']['parameters']:,} | "
            f"{result['training']['epochs']} | {result['training']['global_steps']:,} | "
            f"{result['training']['seconds'] / 60:.1f} min | "
            f"{result['evaluation_seconds']:.1f} s | "
            f"{result['resources']['peak_memory_mib'] / 1024:.2f} GiB |"
        )

    max_epochs = max(len(formal[name]["training"]["history"]) for name in formal)
    lines.extend(
        [
            "",
            "## Epoch 平均 InfoNCE loss",
            "",
            "| Epoch | HSTU | Transformer |",
            "|---:|---:|---:|",
        ]
    )
    for epoch in range(max_epochs):
        values = []
        for name in ("hstu", "transformer"):
            history = formal[name]["training"]["history"]
            values.append(number(history[epoch]["mean_loss"], 4) if epoch < len(history) else "—")
        lines.append(f"| {epoch + 1} | {values[0]} | {values[1]} |")

    lines.extend(
        [
            "",
            "## 历史长度切片综合分",
            "",
            "| 历史长度 | 用户 | HSTU | Transformer | no-MM50 |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    slice_labels = {
        "history_0_20": "0–20",
        "history_21_50": "21–50",
        "history_51_80": "51–80",
        "history_81_plus": "81+",
    }
    for key, label in slice_labels.items():
        values = comparison["slices"][key]
        users = values["hstu"]["evaluated_users"]
        lines.append(
            f"| {label} | {users:,} | {number(values['hstu']['competition_score'])} | "
            f"{number(values['transformer']['competition_score'])} | "
            f"{number(values['nomm50']['competition_score'])} |"
        )

    winner_delta = overall[winner]["competition_score"] / baseline_score - 1.0
    interval_crosses_zero = (
        arch_effect["normal_95_low"] <= 0.0 <= arch_effect["normal_95_high"]
    )
    lines.extend(
        [
            "",
            "## 可辩护结论",
            "",
            f"- 两个 OnePiece 变体中，固定 seed 点估计较高的是 **{LABELS[winner]}**。",
            f"- 该点估计相对既有最佳 no-MM50 baseline 的综合分变化为 **{percent(winner_delta)}**。",
            "- Transformer − HSTU 的逐用户固定人口区间"
            + ("跨过 0，不能声称稳定架构优势。" if interval_crosses_zero else "未跨过 0，但仍不包含训练 seed 不确定性。"),
            "- 所有逐用户区间均不包含训练 seed 不确定性，多 seed 复核仍是确认模型差异的必要步骤。",
            "- HSTU/Transformer 对照只改变序列编码器；与 SASRec-style baseline 的比较同时改变了",
            "  词表参数化、特征管线、损失函数、Batch 与训练轮数，因此属于系统级结果，不是纯架构消融。",
            "- 4×128 是共享单卡约束下的资源缩放配置，不等于上游报告的 8×512、24×512 或 12×1024。",
            "- 本轮未启用 SID、MoE、多模态和时间上下文；不能把结果外推为对完整 OnePiece 最终方案的判断。",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--comparison", type=Path, required=True)
    parser.add_argument("--hstu-metrics", type=Path, required=True)
    parser.add_argument("--transformer-metrics", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    comparison = json.loads(args.comparison.read_text(encoding="utf-8"))
    formal = {
        "hstu": json.loads(args.hstu_metrics.read_text(encoding="utf-8")),
        "transformer": json.loads(args.transformer_metrics.read_text(encoding="utf-8")),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(render(comparison, formal) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
