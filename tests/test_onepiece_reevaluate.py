from pathlib import Path


ROOT = Path(__file__).parents[1]
SCRIPT = ROOT / "scripts" / "reevaluate_onepiece_artifact.py"


def test_reevaluator_is_path_neutral_and_reuses_the_aligned_evaluation_contract():
    source = SCRIPT.read_text(encoding="utf-8")

    assert "ONEPIECE_SOURCE_MODEL" in source
    assert "ONEPIECE_REEVAL_OUTPUT" in source
    assert "import run_onepiece_formal as runner" in source
    assert "runner.evaluate" in source
    assert "validate_targets_in_candidate_pool" in source
    assert "/tmp/sunche" not in source
    assert "s84448890" not in source


def test_reevaluator_signs_source_model_and_writes_prediction_manifest():
    source = SCRIPT.read_text(encoding="utf-8")

    assert '"source_model_sha256"' in source
    assert '"reevaluation_signature"' in source
    assert '"history_filtering": True' in source
    assert '"cold_candidate_filtering": True' in source
    assert "offline_predictions.npz" in source
    assert "SHA256SUMS" in source


def test_reevaluator_does_not_retrain_or_create_optimizer_state():
    source = SCRIPT.read_text(encoding="utf-8")

    assert "optimizer" not in source
    assert "loss.backward" not in source
    assert "model.load_state_dict" in source
