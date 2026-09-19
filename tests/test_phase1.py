import unittest

import numpy as np

from jointinv.phase1 import (
    CASES, CARBONATE, carbonate_design, forward_electrical_truth,
    gem_conductivity, run_carbonate_monte_carlo,
)


class PhaseOneCarbonateTests(unittest.TestCase):
    def test_gem_is_bounded_and_monotonic(self):
        low = gem_conductivity(0.10, 10.0, 1e-7, 0.08, 2.0)
        high = gem_conductivity(0.25, 10.0, 1e-7, 0.08, 2.0)
        self.assertGreater(low, 0.0)
        self.assertGreater(high, low)
        self.assertLessEqual(high, 10.0)

    def test_alternative_truths_are_finite(self):
        p = carbonate_design(1, 42)[0]
        for family in ("dual", "ema", "network"):
            rt = forward_electrical_truth(p, family, 0.5)
            self.assertEqual(rt.shape, (2,))
            self.assertTrue(np.all(np.isfinite(rt)))
            self.assertTrue(np.all(rt > 0.0))

    def test_design_respects_secondary_constraint(self):
        samples = carbonate_design(64, 19)
        self.assertTrue(all(p["secondary"] <= 0.72 * p["phi"] for p in samples))
        self.assertTrue(all("connectivity_latent" in p for p in samples))

    def test_tiny_monte_carlo_has_complete_pairing(self):
        raw, summary, gates = run_carbonate_monte_carlo(
            n_panels=1, replicates=1, panel_size=4, seed=7, cases=CASES[:1]
        )
        self.assertEqual(len(raw), 2 * 4)
        self.assertEqual(set(raw["mode"]), {"elastic", "joint"})
        self.assertEqual(len(summary), 2 * 4)
        self.assertEqual(len(gates), 4)
        self.assertTrue(set(gates["verdict"]).issubset({"PASS", "CONDITIONAL", "FAIL"}))


if __name__ == "__main__":
    unittest.main()
