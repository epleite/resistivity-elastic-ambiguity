import unittest

import numpy as np

import jointinv.phase05 as p5
from jointinv.families import forward_elastic_family, forward_electrical_family


class PhaseZeroFiveTests(unittest.TestCase):
    def test_lhs_is_reproducible_bounded_and_conditioned(self):
        scenario = p5.SCENARIOS["multimodal_carbonate"]
        a = p5.latin_hypercube(scenario, 32, 17)
        b = p5.latin_hypercube(scenario, 32, 17)
        self.assertEqual(a, b)
        for sample in a:
            for name, (lower, upper) in p5.RANGES[scenario.name].items():
                self.assertGreaterEqual(sample[name], lower - 1e-12)
                self.assertLessEqual(sample[name], upper + 1e-12)
            self.assertLessEqual(sample["secondary"], 0.72 * sample["phi"] + 1e-12)

    def test_correlated_rt_whitening(self):
        w = p5._electrical_whitener("deep_shallow")
        sigma = p5.DATA_SIGMA["rt_log"]
        covariance = sigma**2 * np.array([
            [1.0, p5.RT_CORRELATION], [p5.RT_CORRELATION, 1.0]
        ])
        np.testing.assert_allclose(w @ covariance @ w.T, np.eye(2), atol=1e-12)

    def test_pointwise_free_nuisance_is_negative_control(self):
        for scenario in p5.SCENARIOS.values():
            sample = p5.latin_hypercube(scenario, 1, 23)[0]
            case = p5.cases_for(scenario)[0]
            result = p5._one_state(
                sample, scenario, case, "deep_shallow", "free", panel_size=1
            )
            self.assertLess(np.max(np.abs(result["reductions"]["base"])), 1e-8)

    def test_schur_increment_is_symmetric_psd(self):
        rng = np.random.default_rng(8)
        jt = rng.normal(size=(7, 4))
        jn = rng.normal(size=(7, 3))
        pn = np.diag([0.2, 0.4, 0.6])
        increment = p5._effective_information(jt, jn, pn)
        np.testing.assert_allclose(increment, increment.T, atol=1e-11)
        self.assertGreaterEqual(np.linalg.eigvalsh(increment).min(), -1e-9)

    def test_oracle_is_not_worse_than_free_shared_panel(self):
        for scenario in p5.SCENARIOS.values():
            sample = p5.latin_hypercube(scenario, 1, 29)[0]
            case = p5.cases_for(scenario)[0]
            free = p5._one_state(
                sample, scenario, case, "deep_shallow", "free", panel_size=6
            )["reductions"]["base"]
            oracle = p5._one_state(
                sample, scenario, case, "oracle", "calibrated", panel_size=6
            )["reductions"]["base"]
            self.assertTrue(np.all(oracle + 1e-8 >= free))

    def test_forward_families_are_finite_on_design(self):
        for index, scenario in enumerate(p5.SCENARIOS.values()):
            samples = p5.latin_hypercube(scenario, 24, 101 + index)
            for case in p5.cases_for(scenario):
                for sample in samples:
                    elastic = forward_elastic_family(sample, scenario, case.truth_elastic)
                    electrical = forward_electrical_family(
                        sample, scenario, case.truth_electrical, case.truth_coupling
                    )
                    self.assertTrue(np.all(np.isfinite(elastic)) and np.all(elastic > 0))
                    self.assertTrue(np.all(np.isfinite(electrical)) and np.all(electrical > 0))
                    self.assertGreater(elastic[0], elastic[1])


if __name__ == "__main__":
    unittest.main()
