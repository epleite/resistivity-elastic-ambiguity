import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from jointinv.phase11 import CANDIDATE_FAMILIES, COUPLING_MODES, phase11_design
from jointinv.phase12c import (
    CANDIDATE_KEYS, OFF_LIBRARY_CASES, frozen_press_weights,
    learn_frozen_weights, offlibrary_truth_vectors, overall_phi_decision,
    patch_saturations, prepare_frozen_training_rule, run_phase12c,
    truth_library_distance,
)


ROOT = Path(__file__).resolve().parents[1]
RECOVERY_PATH = ROOT / "outputs/phase11_carbonate/recovery_strategies.csv"
COMPONENT_PATH = ROOT / "outputs/phase11_carbonate/fit_components.csv"


class PhaseOneTwoCOffLibraryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.training_recoveries = pd.read_csv(RECOVERY_PATH)
        cls.training_components = pd.read_csv(COMPONENT_PATH)

    def test_candidate_library_remains_exactly_nine_models(self):
        expected = tuple(
            (family, coupling)
            for family in CANDIDATE_FAMILIES
            for coupling in COUPLING_MODES
        )
        self.assertEqual(CANDIDATE_KEYS, expected)
        self.assertEqual(len(CANDIDATE_KEYS), 9)

    def test_patchy_saturation_preserves_mean_and_bounds(self):
        for saturation in np.linspace(0.02, 0.98, 21):
            low, high = patch_saturations(saturation)
            self.assertGreaterEqual(low, 0.01)
            self.assertLessEqual(high, 0.999)
            self.assertAlmostEqual(0.5 * (low + high), saturation, places=12)

    def test_offlibrary_truths_are_physical_and_structurally_separate(self):
        design = phase11_design(4, 812)
        for case in OFF_LIBRARY_CASES:
            elastic, log_rt, diagnostics = offlibrary_truth_vectors(
                design[0], case, panel_size=6,
            )
            self.assertEqual(elastic.shape, (6, 3))
            self.assertEqual(log_rt.shape, (6, 2))
            self.assertTrue(np.all(np.isfinite(elastic)))
            self.assertTrue(np.all(np.isfinite(log_rt)))
            self.assertTrue(np.all(elastic > 0.0))
            self.assertGreaterEqual(diagnostics["q_clip_fraction"], 0.0)
        distances = truth_library_distance(design, OFF_LIBRARY_CASES, 6)
        minimum = distances.groupby("case").fixed_truth_standardized_rms.min()
        self.assertAlmostEqual(minimum["matched_ema_control"], 0.0, places=11)
        for case in OFF_LIBRARY_CASES:
            if case.is_ood:
                self.assertGreater(minimum[case.name], 1e-5)

    def test_frozen_prior_and_press_update_are_well_formed(self):
        prior = learn_frozen_weights(self.training_components)
        self.assertEqual(prior.shape, (9,))
        self.assertAlmostEqual(float(prior.sum()), 1.0)
        fits = [
            {
                "family": family, "coupling": coupling,
                "success": True, "predictive_score": float(index),
            }
            for index, (family, coupling) in enumerate(CANDIDATE_KEYS)
        ]
        weights = frozen_press_weights(fits, prior)
        self.assertAlmostEqual(float(weights.sum()), 1.0)
        self.assertTrue(np.all(weights >= 0.0))
        fits[0]["success"] = False
        weights = frozen_press_weights(fits, prior)
        self.assertEqual(weights[0], 0.0)

    def test_training_rule_is_deterministic_and_uses_no_ood_rows(self):
        first = prepare_frozen_training_rule(
            self.training_recoveries, self.training_components,
        )
        second = prepare_frozen_training_rule(
            self.training_recoveries.copy(), self.training_components.copy(),
        )
        np.testing.assert_allclose(first[0], second[0])
        pd.testing.assert_frame_equal(first[1], second[1])
        pd.testing.assert_frame_equal(first[2], second[2])
        self.assertEqual(first[3], second[3])
        self.assertTrue((first[1].n_training == 240).all())

    def test_outer_prior_is_invariant_to_held_panel(self):
        baseline = learn_frozen_weights(self.training_components, excluded_panel=0)
        perturbed = self.training_components.copy()
        held = perturbed.panel.eq(0)
        perturbed.loc[held, "predictive_score"] = np.linspace(
            0.0, 1e6, int(held.sum()),
        )
        perturbed.loc[held, "success"] = False
        changed = learn_frozen_weights(perturbed, excluded_panel=0)
        np.testing.assert_allclose(baseline, changed, atol=0.0, rtol=0.0)

    def test_overall_decision_cannot_be_upgraded_by_adapted_strategy(self):
        rows = []
        for case in OFF_LIBRARY_CASES:
            rows.append({
                "case": case.name, "atomic": case.atomic,
                "target": "phi", "strategy": "frozen_press", "verdict": "PASS",
            })
            rows.append({
                "case": case.name, "atomic": case.atomic,
                "target": "phi", "strategy": "adapted_average", "verdict": "PASS",
            })
        gates = pd.DataFrame(rows)
        self.assertEqual(overall_phi_decision(gates).decision.iloc[0], "GO")
        gates.loc[
            (gates["case"] == "hybrid_law")
            & (gates["strategy"] == "frozen_press"), "verdict"
        ] = "FAIL"
        self.assertEqual(overall_phi_decision(gates).decision.iloc[0], "STOP")

    def test_tiny_run_has_complete_unique_outputs(self):
        raw, components, diagnostics, _, gates, _, _, weights, _, distance = run_phase12c(
            self.training_recoveries, self.training_components,
            n_panels=3, replicates=1, panel_size=4, seed=94,
            cases=OFF_LIBRARY_CASES[:1], workers=1, n_bootstrap=20,
        )
        self.assertEqual(len(raw), 3 * 1 * 1 * 3 * 4)
        self.assertEqual(len(components), 3 * 1 * 1 * 9)
        self.assertEqual(len(diagnostics), 3)
        self.assertEqual(len(gates), 1 * 4 * 2)
        self.assertEqual(len(weights), 9)
        self.assertEqual(len(distance), 3 * 1 * 9)
        self.assertFalse(raw.duplicated(
            ["case", "panel", "replicate", "strategy", "target"]
        ).any())
        self.assertFalse(components.duplicated(
            ["case", "panel", "replicate", "family", "coupling"]
        ).any())
        sums = components.groupby(
            ["case", "panel", "replicate"]
        )[["frozen_weight", "adapted_weight"]].sum()
        np.testing.assert_allclose(sums.to_numpy(), 1.0)


if __name__ == "__main__":
    unittest.main()
