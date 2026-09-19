import unittest

import numpy as np
import pandas as pd

from jointinv.phase12 import conformal_order_statistic, crossfit_intervals


class PhaseOneTwoCalibrationTests(unittest.TestCase):
    def test_conformal_order_statistic_uses_finite_sample_rank(self):
        self.assertEqual(conformal_order_statistic(np.arange(1, 10), 0.90), 9.0)

    def test_panel_is_never_used_for_its_own_factor(self):
        rows = []
        for panel, score in enumerate([1.0, 2.0, 20.0]):
            for replicate in range(2):
                rows.append({
                    "case": "case", "panel": panel, "replicate": replicate,
                    "strategy": "elastic", "target": "phi", "truth": 0.2,
                    "estimate": 0.2 + score * 0.01, "sd": 0.01,
                    "lower90": 0.18, "upper90": 0.22,
                    "interval_width90": 0.04, "covered90": True,
                    "weighted_reduced_chi2": 1.0, "success_weight": 1.0,
                })
        result = crossfit_intervals(pd.DataFrame(rows), "global")
        self.assertTrue(np.allclose(result.loc[result.panel.eq(2), "c90"], 2.0))
        self.assertTrue(np.allclose(result.loc[result.panel.eq(0), "c90"], 20.0))

    def test_intervals_respect_physical_bounds(self):
        rows = []
        for panel in range(3):
            rows.append({
                "case": "case", "panel": panel, "replicate": 0,
                "strategy": "elastic", "target": "sw", "truth": 0.95,
                "estimate": 0.99, "sd": 1.0, "lower90": 0.0,
                "upper90": 1.0, "interval_width90": 1.0, "covered90": True,
                "weighted_reduced_chi2": 1.0, "success_weight": 1.0,
            })
        result = crossfit_intervals(pd.DataFrame(rows), "global")
        self.assertTrue((result.upper90 <= 0.98).all())
        self.assertTrue((result.lower90 >= 0.10).all())


if __name__ == "__main__":
    unittest.main()

