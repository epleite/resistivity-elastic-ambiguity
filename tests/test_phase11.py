import unittest

import numpy as np

from jointinv.phase1 import carbonate_design
from jointinv.phase11 import (
    CANDIDATE_FAMILIES, COUPLING_MODES, TRUTH_CASES, connectivity_latent,
    converged_weights, forward_candidate_state, mixture_recovery, phase11_design,
    predictive_weights, realized_correlation, run_phase11,
)


class PhaseOneOneTests(unittest.TestCase):
    def test_candidate_forward_models_are_physical(self):
        p = carbonate_design(1, 28)[0]
        p["q_conn"] = -0.4
        for family in CANDIDATE_FAMILIES:
            for coupling in COUPLING_MODES:
                rt = forward_candidate_state(p, family, coupling)
                self.assertEqual(rt.shape, (2,))
                self.assertTrue(np.all(np.isfinite(rt)))
                self.assertTrue(np.all(rt > 0.0))

    def test_rigid_connectivity_ignores_free_q(self):
        p = carbonate_design(1, 31)[0]
        p["q_conn"] = -3.0
        first = connectivity_latent(p, "rigid")
        p["q_conn"] = 3.0
        second = connectivity_latent(p, "rigid")
        self.assertAlmostEqual(first, second)
        self.assertNotEqual(connectivity_latent(p, "independent"), first)

    def test_predictive_weights_sum_and_order(self):
        fits = [
            {"predictive_score": 10.0, "coupling": "rigid"},
            {"predictive_score": 12.0, "coupling": "partial"},
            {"predictive_score": 20.0, "coupling": "independent"},
        ]
        weights = predictive_weights(fits)
        self.assertAlmostEqual(float(weights.sum()), 1.0)
        self.assertGreater(weights[0], weights[1])
        self.assertGreater(weights[1], weights[2])

    def test_nonconverged_candidates_receive_zero_weight(self):
        fits = [{"success": True}, {"success": False}, {"success": True}]
        weights = converged_weights(fits, np.array([0.2, 0.7, 0.1]))
        self.assertAlmostEqual(float(weights.sum()), 1.0)
        self.assertEqual(weights[1], 0.0)
        self.assertAlmostEqual(weights[0], 2.0 / 3.0)

    def test_design_realizes_requested_correlation(self):
        design = phase11_design(12, 17)
        self.assertAlmostEqual(
            float(np.mean([panel["q_elastic_design"] for panel in design])),
            0.0,
            places=12,
        )
        for rho in (0.0, 0.5, 0.9, 1.0):
            self.assertAlmostEqual(realized_correlation(design, rho), rho, places=12)

    def test_mixture_retains_between_model_uncertainty(self):
        fits = [
            {"estimate": {"phi": 0.10}, "sd": {"phi": 0.01}, "reduced_chi2": 1.0, "success": True},
            {"estimate": {"phi": 0.20}, "sd": {"phi": 0.01}, "reduced_chi2": 1.0, "success": True},
        ]
        result = mixture_recovery(fits, np.array([0.5, 0.5]), "phi")
        self.assertAlmostEqual(result["estimate"], 0.15, places=5)
        self.assertGreater(result["sd"], 0.05)
        self.assertLess(result["lower90"], 0.10)
        self.assertGreater(result["upper90"], 0.20)

    def test_tiny_phase11_has_complete_outputs(self):
        raw, components, summary, gates, family, coupling = run_phase11(
            n_panels=3, replicates=1, panel_size=4, seed=9,
            truth_cases=TRUTH_CASES[:1],
        )
        self.assertEqual(len(raw), 3 * 5 * 4)
        self.assertEqual(len(components), 3 * 9)
        self.assertTrue({
            "any_bound_hit", "critical_bound_hit", "nuisance_bound_hit",
            "bound_hit_phi", "bound_hit_sw", "bound_hit_aspect",
            "bound_hit_secondary_fraction", "bound_hit_q_conn",
            "bound_hit_log_rw", "bound_hit_archie_m", "bound_hit_archie_n",
            "bound_hit_surface_cond", "bound_hit_invasion",
            "bound_hit_rmf_ratio", "estimate_phi", "estimate_secondary",
            "estimate_q_conn", "sd_phi", "sd_secondary", "sd_q_conn",
            "total_nfev", "convergence_fallback", "elastic_success",
            "elastic_total_nfev", "elastic_convergence_fallback",
        }.issubset(components.columns))
        self.assertTrue(components["success"].all())
        self.assertTrue(components["elastic_success"].all())
        self.assertEqual(len(summary), 5 * 4)
        self.assertEqual(len(gates), 4 * 4)
        self.assertAlmostEqual(components["full_weight"].sum(), 3.0)
        self.assertEqual(len(family), 3)
        self.assertEqual(len(coupling), 3)
        self.assertAlmostEqual(family["mean_weight"].sum(), 1.0)
        self.assertAlmostEqual(coupling["mean_weight"].sum(), 1.0)


if __name__ == "__main__":
    unittest.main()
