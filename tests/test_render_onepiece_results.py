from __future__ import annotations

from scripts.render_onepiece_results import ORDER, render


def aggregate(score):
    return {
        "evaluated_users": 78921,
        "hits": 100,
        "hit_rate_at_10": score + 0.01,
        "ndcg_at_10": score,
        "competition_score": score,
    }


def effect(delta):
    return {
        "mean_delta": delta,
        "normal_95_low": delta - 0.001,
        "normal_95_high": delta + 0.001,
    }


def test_render_includes_metrics_resources_and_limitations():
    scores = {
        "mm101": 0.020,
        "nomm101": 0.021,
        "mm50": 0.0215,
        "nomm50": 0.022,
        "hstu": 0.024,
        "transformer": 0.023,
    }
    comparison = {
        "row_alignment": {"rows": 78921},
        "overall": {name: aggregate(scores[name]) for name in ORDER},
        "transformer_minus_hstu": {"competition_score": effect(-0.001)},
        "onepiece_vs_baselines": {
            "hstu_minus_nomm50": {"competition_score": effect(0.002)},
            "transformer_minus_nomm50": {"competition_score": effect(0.001)},
        },
        "slices": {
            key: {name: aggregate(scores[name]) for name in ORDER}
            for key in (
                "history_0_20", "history_21_50", "history_51_80", "history_81_plus"
            )
        },
    }
    formal = {
        name: {
            "model": {"parameters": 1000},
            "training": {
                "epochs": 2, "global_steps": 20, "seconds": 120,
                "history": [{"mean_loss": 5.0}, {"mean_loss": 4.0}],
            },
            "evaluation_seconds": 30,
            "resources": {"peak_memory_mib": 1024},
        }
        for name in ("hstu", "transformer")
    }

    report = render(comparison, formal)

    assert "OnePiece HSTU 4×128" in report
    assert "+9.09%" in report
    assert "训练 seed 不确定性" in report
    assert "系统级结果，不是纯架构消融" in report

