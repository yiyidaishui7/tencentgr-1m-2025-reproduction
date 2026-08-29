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


def test_reevaluator_enters_offline_mode_before_importing_huggingface_datasets():
    source = SCRIPT.read_text(encoding="utf-8")

    assert source.index('os.environ["HF_HUB_OFFLINE"] = "1"') < source.index(
        "from datasets import load_dataset"
    )
    assert source.index('os.environ["HF_DATASETS_OFFLINE"] = "1"') < source.index(
        "from datasets import load_dataset"
    )


def test_reevaluator_binds_all_four_dataset_splits_and_user_row_count():
    source = SCRIPT.read_text(encoding="utf-8")

    assert "DATASET_CONTRACT" in source
    for name, fingerprint in {
        "seq": "bcabd99de59b2fdd",
        "item_feat": "defdfd291f3184cb",
        "user_feat": "20308ee19a2f0c82",
        "candidate": "932a40e9007e0e8d",
    }.items():
        assert name in source
        assert fingerprint in source
    assert "len(user_dataset) != user_count" in source
    assert '"dataset_contract": DATASET_CONTRACT' in source
    assert '"dataset_receipt": dataset_receipt' in source


def test_reevaluator_rejects_arrow_files_outside_the_frozen_content_manifest():
    source = SCRIPT.read_text(encoding="utf-8")

    for digest in (
        "14a9addcb894d8a10c97cd364d197fd8fd179fbc94d8ced5aef691fd5c2ffa1b",
        "53031408cec8d88179c28460e2205d4628882e52b35f3da0b85ff514e2e89188",
        "bd4d0a269df96504f5db443ded25cf5186d5294c4419719db17172416df4e25b",
        "4d53e58de753a995adff862976be97b280d77271f94aac5febfec81fa616f291",
        "389f5324ade9deddb34d84e160c4ede9ba8897efe88265c582f5db85af3ae0c2",
        "17ab92da05c8ba2b46ea4bbc5903b16872f356c90949c42a743675aa5d6f9b41",
        "d85a8177e25f4c0d0a1676c712cc9e7bf1fef926a1bbf74bb1ff593accedd121",
    ):
        assert digest in source
    assert 'expected["files"]' in source
    assert 'observed_files != expected["files"]' in source


def test_reevaluator_atomically_publishes_a_complete_output_directory():
    source = SCRIPT.read_text(encoding="utf-8")

    assert "WORK_OUTPUT" in source
    assert "os.replace(WORK_OUTPUT, OUTPUT)" in source
    assert source.index("os.replace(WORK_OUTPUT, OUTPUT)") < source.index(
        'write_status("complete", **result)'
    )


def test_reevaluator_rechecks_every_frozen_input_before_and_after_use():
    source = SCRIPT.read_text(encoding="utf-8")

    assert "def verify_frozen_inputs" in source
    assert source.count("verify_frozen_inputs(") >= 4
    assert "source_contract.verify_source_files" in source
    assert "dataset_receipt_after" in source
    assert "frozen input changed during reevaluation" in source
