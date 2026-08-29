from pathlib import Path


ROOT = Path(__file__).parents[1]


def test_sid_results_disclose_the_nonhierarchical_reallocation_gap():
    text = (ROOT / "docs" / "ONEPIECE_SID_RESULTS.md").read_text(
        encoding="utf-8"
    )

    assert "766" in text
    assert "全局 residual" in text
    assert "不是 README" in text
    assert "SID1 固定" in text


def test_alignment_summary_labels_old_scores_as_historical_protocol():
    text = (ROOT / "docs" / "ONEPIECE_ALIGNMENT_RESULTS.md").read_text(
        encoding="utf-8"
    )

    assert "历史协议" in text
    assert "不是严格" in text
    assert "148,971" in text


def test_runbook_freezes_aligned_candidate_and_mask_behavior():
    text = (ROOT / "docs" / "ONEPIECE_RUNBOOK.md").read_text(
        encoding="utf-8"
    )

    assert "511,029" in text
    assert "148,971" in text
    assert "history" in text.lower()
    assert "ANN fallback" in text
    assert "1748620800" in text
