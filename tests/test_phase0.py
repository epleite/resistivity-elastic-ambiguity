import unittest

import numpy as np

from jointinv.analysis import analyze_scenario, finite_difference_jacobian
from jointinv.models import SCENARIOS, forward_elastic, forward_electrical


class PhaseZeroTests(unittest.TestCase):
    def test_forward_models_are_finite_and_physical(self):
        for scenario in SCENARIOS.values():
            elastic = forward_elastic(scenario.truth, scenario)
            electrical = forward_electrical(scenario.truth, scenario)
            self.assertTrue(np.all(np.isfinite(elastic)) and np.all(elastic > 0))
            self.assertTrue(np.all(np.isfinite(electrical)) and np.all(electrical > 0))
            self.assertGreater(elastic[0], elastic[1])

    def test_central_jacobian_shape(self):
        scenario = SCENARIOS["clean_sandstone"]
        names = ["phi", "sw"]
        jac = finite_difference_jacobian(
            lambda p: forward_elastic(p, scenario), scenario.truth, names
        )
        self.assertEqual(jac.shape, (3, 2))
        self.assertTrue(np.all(np.isfinite(jac)))

    def test_marginalized_gain_is_bounded_and_tight_prior_not_worse(self):
        for scenario in SCENARIOS.values():
            result = analyze_scenario(scenario)
            broad = result["cases"]["deep+shallow|broad"]["parameter_variance_reduction"]
            tight = result["cases"]["deep+shallow|tight"]["parameter_variance_reduction"]
            for name in broad:
                self.assertGreaterEqual(broad[name], -1e-8)
                self.assertLessEqual(broad[name], 1.0 + 1e-8)
                self.assertGreaterEqual(tight[name], -1e-8)
                self.assertLessEqual(tight[name], 1.0 + 1e-8)
                self.assertGreaterEqual(tight[name] + 1e-8, broad[name])


if __name__ == "__main__":
    unittest.main()
