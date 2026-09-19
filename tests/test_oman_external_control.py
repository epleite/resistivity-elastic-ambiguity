import copy
import hashlib
import json
import unittest
from pathlib import Path

from scripts.run_oman_external_control import (
    DEFAULT_INPUT,
    DEFAULT_REFERENCE,
    Mulberry32,
    load_rows,
    outer_cv,
    paired_metrics,
    verify,
)


class OmanExternalControlTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not DEFAULT_INPUT.is_file():
            raise unittest.SkipTest(
                "Oman source-derived table is not redistributed; run "
                "scripts/prepare_oman_data.py first"
            )
        all_rows = load_rows(DEFAULT_INPUT)
        cls.primary_rows = [row for row in all_rows if row["primary_complete"]]
        cls.formation_rows = [row for row in all_rows if row["f_complete"]]
        cls.reference = json.loads(DEFAULT_REFERENCE.read_text(encoding="utf-8"))

    @staticmethod
    def _metrics(rows, electrical_key):
        elastic, _ = outer_cv(rows, ("vp_wet", "vs_wet", "density"), "E")
        joint, _ = outer_cv(
            rows,
            ("vp_wet", "vs_wet", "density", electrical_key),
            "J",
        )
        return paired_metrics(elastic, joint)

    def test_primary_point_metrics_match_frozen_result(self):
        metrics = self._metrics(self.primary_rows, "log_r35")
        expected = self.reference["primary"]
        self.assertEqual(len(self.primary_rows), expected["n"])
        for field in (
            "rmse_elastic",
            "rmse_joint",
            "rmse_ratio",
            "mae_elastic",
            "mae_joint",
            "mae_ratio",
        ):
            self.assertAlmostEqual(metrics[field], expected[field], places=11)

    def test_resampling_generator_matches_locked_javascript_sequence(self):
        generator = Mulberry32(14052026)
        expected = (
            0.33207294251769781,
            0.62266458035446703,
            0.39959681825712323,
            0.00071164802648127079,
            0.47234084201045334,
        )
        self.assertEqual(tuple(generator.random() for _ in expected), expected)

    def test_public_oman_manifest_hashes_every_included_file(self):
        root = Path(__file__).resolve().parents[1]
        manifest = json.loads(
            (root / "results/oman/phase14e_release_manifest.json").read_text(
                encoding="utf-8"
            )
        )
        for item in manifest["files"]:
            with self.subTest(path=item["path"]):
                payload = (root / item["path"]).read_bytes()
                self.assertEqual(hashlib.sha256(payload).hexdigest(), item["sha256"])

    def test_formation_factor_ratios_match_frozen_result(self):
        metrics = self._metrics(self.formation_rows, "log_f")
        expected = self.reference["formation_factor_sensitivity"]
        self.assertEqual(len(self.formation_rows), expected["n"])
        self.assertAlmostEqual(metrics["rmse_ratio"], expected["rmse_ratio"], places=11)
        self.assertAlmostEqual(metrics["mae_ratio"], expected["mae_ratio"], places=11)

    def test_full_verifier_checks_resampling_statistics(self):
        result = {
            "protocol_id": self.reference["protocol_id"],
            "primary": {
                key: self.reference["primary"][key]
                for key in (
                    "n",
                    "rmse_elastic",
                    "rmse_joint",
                    "rmse_ratio",
                    "mae_elastic",
                    "mae_joint",
                    "mae_ratio",
                    "bootstrap_rmse_ratio_l95",
                    "bootstrap_rmse_ratio_u95",
                    "permutation_p",
                    "permutation_draws",
                    "bootstrap_draws",
                )
            },
            "formation_factor_sensitivity": {
                key: self.reference["formation_factor_sensitivity"][key]
                for key in (
                    "n",
                    "rmse_ratio",
                    "mae_ratio",
                    "bootstrap_rmse_ratio_l95",
                    "bootstrap_rmse_ratio_u95",
                    "permutation_p",
                )
            },
        }
        verify(DEFAULT_REFERENCE, result)
        for variant, field in (
            ("primary", "bootstrap_rmse_ratio_l95"),
            ("primary", "permutation_p"),
            ("formation_factor_sensitivity", "bootstrap_rmse_ratio_u95"),
            ("formation_factor_sensitivity", "permutation_p"),
        ):
            with self.subTest(variant=variant, field=field):
                changed = copy.deepcopy(result)
                changed[variant][field] = float(changed[variant][field]) + 0.1
                with self.assertRaisesRegex(RuntimeError, f"{variant}.{field}"):
                    verify(DEFAULT_REFERENCE, changed)


if __name__ == "__main__":
    unittest.main()
