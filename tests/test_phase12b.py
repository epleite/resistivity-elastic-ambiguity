import unittest

import numpy as np
import pandas as pd

from jointinv.phase1 import FIT_NAMES as PHASE1_FIT_NAMES, fit_panel
from jointinv.phase11 import TRUTH_CASES, phase11_design, simulate_phase11_observations
from jointinv.phase12b import (
    SentinelSpec, connected_profile_interval, fit_profile_point,
    fit_unrestricted_candidate, select_sentinels, weighted_profile_statistic,
)


class PhaseOneTwoBSentinelTests(unittest.TestCase):
    def test_sentinel_selection_does_not_use_realized_error(self):
        design = pd.DataFrame(phase11_design(5, 19))
        design.insert(0, "panel", np.arange(len(design)))
        rows = []
        for panel in range(5):
            rows.append({
                "case": "network_rho05", "target": "phi", "strategy": "model_average",
                "panel": panel, "replicate": 0, "truth": design.loc[panel, "phi"],
                "estimate": design.loc[panel, "phi"], "sd": 0.01, "c90": 2.0,
            })
        calibrated = pd.DataFrame(rows)
        spec = (SentinelSpec("test", "network_rho05", "phi", "positive_candidate"),)
        first = select_sentinels(calibrated, design, spec)
        calibrated["estimate"] = np.linspace(-100.0, 100.0, len(calibrated))
        second = select_sentinels(calibrated, design, spec)
        self.assertEqual(int(first.panel.iloc[0]), int(second.panel.iloc[0]))
        self.assertEqual(int(first.replicate.iloc[0]), 0)

    def test_connected_profile_interval_interpolates_threshold(self):
        result = connected_profile_interval(
            [-2.0, -1.0, 0.0, 1.0, 2.0], [4.0, 1.0, 0.0, 1.0, 4.0],
            threshold=2.0,
        )
        self.assertAlmostEqual(result["lower"], -4.0 / 3.0)
        self.assertAlmostEqual(result["upper"], 4.0 / 3.0)
        self.assertFalse(result["lower_bound_limited"])
        self.assertFalse(result["upper_bound_limited"])
        self.assertFalse(result["disconnected"])

    def test_weighted_profile_statistic_is_nonnegative(self):
        value = weighted_profile_statistic([10.0, 12.0], [12.0, 16.0], [0.7, 0.3])
        self.assertGreater(value, 0.0)
        self.assertEqual(weighted_profile_statistic([10.0], [9.999999], [1.0]), 0.0)

    def test_fixed_profile_respects_value_and_not_below_unrestricted(self):
        panel_size = 4
        truth = phase11_design(3, 29)[1]
        truth_case = next(case for case in TRUTH_CASES if case.name == "ema_rho05")
        zeros_e = np.zeros((panel_size, 3)); zeros_r = np.zeros((panel_size, 2))
        observed_e, observed_r = simulate_phase11_observations(
            truth, truth_case, panel_size, zeros_e, zeros_r,
        )
        fixed = dict(truth)
        for name in PHASE1_FIT_NAMES:
            fixed.pop(name, None)
        fixed.update({name: truth[name] for name in ("vcl", "cement", "coord", "phi_e")})
        elastic = fit_panel(
            observed_e, observed_r, fixed, "dual_porosity", False,
            panel_size, np.random.default_rng(4), n_starts=1,
        )
        initial = dict(elastic["estimate"])
        initial.update({
            "secondary_fraction": truth["secondary"] / min(0.12, 0.72 * truth["phi"]),
            "log_rw": truth["log_rw"], "archie_m": truth["archie_m"],
            "archie_n": truth["archie_n"], "surface_cond": truth["surface_cond"],
            "invasion": truth["invasion"], "rmf_ratio": 0.75, "q_conn": 0.0,
        })
        unrestricted = fit_unrestricted_candidate(
            observed_e, observed_r, fixed, "ema", "independent", panel_size,
            initial, objective_mode="data",
        )
        conditioned = fit_profile_point(
            observed_e, observed_r, fixed, "ema", "independent", panel_size,
            "phi", truth["phi"], unrestricted["state"], objective_mode="data",
        )
        self.assertAlmostEqual(conditioned["state"]["phi"], truth["phi"], places=12)
        self.assertGreaterEqual(
            conditioned["objective"] + 1e-4, unrestricted["objective"]
        )


if __name__ == "__main__":
    unittest.main()
