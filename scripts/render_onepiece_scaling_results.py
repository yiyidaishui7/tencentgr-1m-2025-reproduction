"""Render the verified OnePiece capacity-scaling comparison as Markdown."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


ORDER = ("hstu_4x128", "hstu_8x256", "hstu_8x512")
LABELS = {
    "hstu_4x128": "HSTU 4×128",
    "hstu_8x256": "HSTU 8×256",
    "hstu_8x512": "HSTU 8×512",
}


def number(value: float) -> str:
    return f"{value:.7f}"


def percent(value: float) -> str:
    return f"{value * 100:+.2f}%"


def render(result: dict) -> str:
    if result.get("status") != "pass" or result.get("reference") != "hstu_4x128":
        raise RuntimeError("scaling comparison did not pass the expected reference gate")
    missing = [name for name in ORDER if name not in result.get("overall", {})]
    if missing:
        raise RuntimeError(f"scaling comparison is missing runs: {missing}")

    lines = [
        "# OnePiece HSTU 容量扩展结果",
        "",
        "> 所有结果均通过产物 SHA-256、冻结行对齐和指标重算；分数来自本地严格最后点击留出，",
        "> 不是官方排行榜成绩。每个规模只有一个训练 seed。",
        "",
        "## 总体指标与资源",
        "",
        "| 模型 | 参数量 | HR@10 | NDCG@10 | 综合分 | 相对 4×128 | 训练时间 | 峰值显存 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    reference_score = result["overall"]["hstu_4x128"]["competition_score"]
    for name in ORDER:
        metrics = result["overall"][name]
        metadata = result["metadata"].get(name, {})
        model = metadata.get("model") or {}
        training = metadata.get("training") or {}
        resources = metadata.get("resources") or {}
        relative = metrics["competition_score"] / reference_score - 1.0
        lines.append(
            f"| {LABELS[name]} | {int(model.get('parameters', 0)):,} | "
            f"{number(metrics['hit_rate_at_10'])} | {number(metrics['ndcg_at_10'])} | "
            f"{number(metrics['competition_score'])} | {percent(relative)} | "
            f"{float(training.get('seconds', 0)) / 60:.1f} min | "
            f"{float(resources.get('peak_memory_mib', 0)) / 1024:.2f} GiB |"
        )

    lines.extend([
        "",
        "## 固定评测人口配对差异",
        "",
        "| 对比 | 综合分差值 | 相对差值 | 固定评测人口 95% 区间 |",
        "|---|---:|---:|---:|",
    ])
    for name in ORDER[1:]:
        comparison = result["comparisons"][f"{name}_minus_hstu_4x128"]
        effect = comparison["competition_score"]
        lines.append(
            f"| {LABELS[name]} − HSTU 4×128 | {effect['mean_delta']:+.7f} | "
            f"{percent(comparison['relative_competition_score'])} | "
            f"[{effect['normal_95_low']:.7f}, {effect['normal_95_high']:.7f}] |"
        )

    lines.extend([
        "",
        "## 历史长度切片",
        "",
        "| 历史长度 | 用户数 | HSTU 4×128 | HSTU 8×256 | HSTU 8×512 |",
        "|---|---:|---:|---:|---:|",
    ])
    labels = {
        "history_0_20": "0–20",
        "history_21_50": "21–50",
        "history_51_80": "51–80",
        "history_81_plus": "81+",
    }
    for slice_name, label in labels.items():
        values = result["slices"][slice_name]
        users = values["hstu_4x128"]["evaluated_users"]
        lines.append(
            f"| {label} | {users:,} | "
            + " | ".join(number(values[name]["competition_score"]) for name in ORDER)
            + " |"
        )

    winner = max(ORDER, key=lambda name: result["overall"][name]["competition_score"])
    lines.extend([
        "",
        "## 可辩护结论",
        "",
        f"- 本地冻结协议下点估计最高的是 **{LABELS[winner]}**。",
        "- 容量实验同时改变层数与宽度，因此只能归因于整体 capacity scaling，不能拆分为纯深度或纯宽度收益。",
        "- 配对区间只描述固定评测人口的逐用户差异，不包含不同训练 seed 的不确定性。",
        "- 上游技术报告的分数使用不同配置与评测条件，不能与本地离线分数作严格数值复现率比较。",
    ])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--comparison", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = json.loads(args.comparison.read_text(encoding="utf-8"))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(render(result) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
