import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from experiment_plan import REQUIRED_VARIANTS, build_experiment_plan


class ExperimentPlanTests(unittest.TestCase):
    def setUp(self):
        config_path = Path(__file__).parents[1] / "reproduction_config.json"
        self.config = json.loads(config_path.read_text(encoding="utf-8"))

    def test_builds_canonical_row_aligned_four_variant_plan(self):
        plan = build_experiment_plan(
            self.config,
            data_path=Path("/data/TencentGR-1M"),
            runs_root=Path("runs"),
            eval_root=Path("eval"),
            scratch_root=Path("scratch"),
            device="cuda:0",
        )

        self.assertEqual(plan["schema_version"], 1)
        self.assertEqual(
            [variant["name"] for variant in plan["variants"]],
            list(REQUIRED_VARIANTS),
        )
        for variant in plan["variants"]:
            expected_maxlen, expected_mm = REQUIRED_VARIANTS[variant["name"]]
            train = variant["train"]
            evaluate = variant["evaluate"]
            self.assertEqual(variant["maxlen"], expected_maxlen)
            self.assertEqual(variant["multimodal_enabled"], expected_mm)
            self.assertEqual(train["argv"][train["argv"].index("--maxlen") + 1], str(expected_maxlen))
            self.assertEqual(evaluate["argv"][evaluate["argv"].index("--maxlen") + 1], str(expected_maxlen))
            self.assertEqual("--disable_mm_emb" in train["argv"], not expected_mm)
            self.assertEqual("--disable_mm_emb" in evaluate["argv"], not expected_mm)
            self.assertIn(variant["name"], train["env"]["TRAIN_CKPT_PATH"])
            self.assertTrue(evaluate["checkpoint_glob"].endswith("/model.pt"))

        comparison = plan["comparison"]["argv"]
        for variant_name in REQUIRED_VARIANTS:
            self.assertIn(f"--{variant_name}", comparison)

    def test_rejects_an_incomplete_or_drifted_matrix(self):
        del self.config["controlled_experiment"]["variants"]["nomm50"]

        with self.assertRaisesRegex(ValueError, "exactly"):
            build_experiment_plan(
                self.config,
                data_path=Path("/data/TencentGR-1M"),
                runs_root=Path("runs"),
                eval_root=Path("eval"),
                scratch_root=Path("scratch"),
                device="cuda:0",
            )


class ExperimentPlanCliTests(unittest.TestCase):
    def test_cli_writes_a_machine_readable_plan_without_starting_training(self):
        root = Path(__file__).parents[1]
        script = root / "scripts" / "plan_2x2_experiments.py"
        with tempfile.TemporaryDirectory() as temporary_directory:
            output = Path(temporary_directory) / "plan.json"
            completed = subprocess.run(
                [
                    sys.executable,
                    str(script),
                    "--config",
                    str(root / "reproduction_config.json"),
                    "--data-path",
                    "/data/TencentGR-1M",
                    "--output",
                    str(output),
                    "--device",
                    "cuda:0",
                ],
                text=True,
                capture_output=True,
                check=False,
            )

            self.assertEqual(completed.returncode, 0, completed.stderr)
            plan = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(len(plan["variants"]), 4)
            self.assertEqual(plan["contract"]["seed"], 2025)
            self.assertNotIn("execute", plan)


if __name__ == "__main__":
    unittest.main()
