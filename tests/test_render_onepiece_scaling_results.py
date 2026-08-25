from __future__ import annotations

from scripts.render_onepiece_scaling_results import render


def metric(score: float, users: int = 10) -> dict:
    return {
        "evaluated_users": users,
        "hits": int(score * users),
        "hit_rate_at_10": score,
        "ndcg_at_10": score,
        "competition_score": score,
    }


def test_renders_scaling_table_effects_slices_and_caveats():
    names = ("hstu_4x128", "hstu_8x256", "hstu_8x512")
    result = {
        "status": "pass",
        "reference": "hstu_4x128",
        "overall": {name: metric(score) for name, score in zip(names, (0.1, 0.11, 0.12))},
        "metadata": {
            name: {
                "model": {"parameters": 100},
                "training": {"seconds": 60},
                "resources": {"peak_memory_mib": 1024},
            }
            for name in names
        },
        "comparisons": {
            f"{name}_minus_hstu_4x128": {
                "competition_score": {
                    "mean_delta": delta,
                    "normal_95_low": delta - 0.001,
                    "normal_95_high": delta + 0.001,
                },
                "relative_competition_score": delta / 0.1,
            }
            for name, delta in (("hstu_8x256", 0.01), ("hstu_8x512", 0.02))
        },
        "slices": {
            slice_name: {name: metric(score, 2) for name, score in zip(names, (0.1, 0.11, 0.12))}
            for slice_name in (
                "history_0_20", "history_21_50", "history_51_80", "history_81_plus"
            )
        },
    }
    report = render(result)
    assert "HSTU 8×512" in report
    assert "+20.00%" in report
    assert "[0.0190000, 0.0210000]" in report
    assert "capacity scaling" in report
    assert "训练 seed" in report
