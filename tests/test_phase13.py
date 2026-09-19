import unittest
from dataclasses import FrozenInstanceError

import numpy as np
import pandas as pd

from jointinv.phase12c import CANDIDATE_KEYS, OFF_LIBRARY_CASES
from jointinv.phase13 import (
    DEPLOYABLE_FEATURES,
    FORBIDDEN_POLICY_FIELDS,
    SelectivePolicy,
    _cluster_draw_indices,
    apply_policy,
    build_phi_observation_table,
    build_phase13_gates,
    freeze_phase13_policy,
    overall_phase13_decision,
    random_rejection_comparator,
    reconstruct_observation_bank,
    summarize_selective_cases,
)


def _training_components(n_observations=12):
    rows = []
    for observation in range(n_observations):
        case = "training_a" if observation % 2 == 0 else "training_b"
        panel = observation // 2
        replicate = observation % 2
        for candidate, (family, coupling) in enumerate(CANDIDATE_KEYS):
            rows.append({
                "case": case,
                "panel": panel,
                "replicate": replicate,
                "family": family,
                "coupling": coupling,
                "success": True,
                "reduced_chi2": 0.10 + 0.05 * observation + candidate,
            })
    return pd.DataFrame(rows)


def _policy(chi2_threshold=0.20, phi_bound_mass_limit=0.10):
    return SelectivePolicy(
        policy_id="unit_test_policy",
        endpoint="phi",
        threshold_source="unit_test_training_only",
        threshold_quantile=0.90,
        chi2_threshold=chi2_threshold,
        phi_bound_mass_limit=phi_bound_mass_limit,
        candidate_success_min=0.99,
        surviving_prior_mass_min=0.99,
        accept_when="unit-test frozen conjunction",
        deployable_features=DEPLOYABLE_FEATURES,
        training_observations=12,
    )


def _paired_raw(cases=("matched_ema_control",), n_panels=4):
    chi2 = np.array([0.10, 0.20, 0.20, 0.70], dtype=float)
    rows = []
    for case in cases:
        for panel in range(n_panels):
            truth = 0.20 + 0.002 * panel
            for strategy in ("elastic", "frozen_press"):
                if strategy == "elastic":
                    estimate, sd, half_width = truth + 0.010, 0.020, 0.040
                    target_bound, critical_bound = 0.0, 0.0
                else:
                    estimate, sd, half_width = truth + 0.004, 0.010, 0.020
                    target_bound, critical_bound = 0.02, 0.95
                lower, upper = estimate - half_width, estimate + half_width
                rows.append({
                    "case": case,
                    "panel": panel,
                    "replicate": 0,
                    "strategy": strategy,
                    "target": "phi",
                    "truth": truth,
                    "estimate": estimate,
                    "sd": sd,
                    "c90": 2.0,
                    "lower90": lower,
                    "upper90": upper,
                    "interval_width90": upper - lower,
                    "covered90": lower <= truth <= upper,
                    "best_reduced_chi2": float(chi2[panel % len(chi2)]),
                    "target_bound_mass": target_bound,
                    "critical_bound_mass": critical_bound,
                    "raw_candidate_success_fraction": 1.0,
                    "surviving_prior_mass": 1.0,
                    "weighted_reduced_chi2": float(chi2[panel % len(chi2)]),
                })
    return pd.DataFrame(rows)


def _all_case_observations(n_panels=2):
    cases = tuple(case.name for case in OFF_LIBRARY_CASES)
    return build_phi_observation_table(_paired_raw(cases, n_panels=n_panels))


class PhaseOneThreeSelectiveInversionTests(unittest.TestCase):
    def test_policy_contract_is_immutable_and_deployable(self):
        policy, grid = freeze_phase13_policy(_training_components())
        self.assertEqual(policy.endpoint, "phi")
        self.assertEqual(policy.threshold_source, "phase11_matched_candidate_bank")
        self.assertEqual(tuple(policy.deployable_features), DEPLOYABLE_FEATURES)
        self.assertTrue(
            set(policy.deployable_features).isdisjoint(FORBIDDEN_POLICY_FIELDS)
        )
        self.assertEqual(policy.training_observations, 12)
        self.assertTrue(grid["chi2_threshold"].is_monotonic_increasing)
        with self.assertRaises(FrozenInstanceError):
            policy.chi2_threshold = 999.0

    def test_freeze_ignores_oracles_labels_and_input_row_order(self):
        baseline, baseline_grid = freeze_phase13_policy(_training_components())
        perturbed = _training_components()
        n = len(perturbed)
        perturbed["truth"] = np.linspace(-1e6, 1e6, n)
        perturbed["error"] = np.linspace(1e9, -1e9, n)
        perturbed["covered90"] = np.arange(n) % 2 == 0
        perturbed["false_confidence"] = np.arange(n) % 3 == 0
        perturbed["truth_library_distance"] = np.linspace(0.0, 100.0, n)
        perturbed["case"] = perturbed["case"].map({
            "training_a": "renamed_a", "training_b": "renamed_b",
        })
        perturbed = perturbed.sample(frac=1.0, random_state=913).reset_index(drop=True)
        changed, changed_grid = freeze_phase13_policy(perturbed)
        self.assertEqual(baseline.to_dict(), changed.to_dict())
        self.assertEqual(baseline.canonical_hash(), changed.canonical_hash())
        pd.testing.assert_frame_equal(baseline_grid, changed_grid)

    def test_policy_acceptance_is_invariant_to_truth_and_oracles(self):
        observations = build_phi_observation_table(_paired_raw())
        baseline = apply_policy(observations, _policy())
        perturbed = observations.copy()
        perturbed["truth"] = np.linspace(-1000.0, 1000.0, len(perturbed))
        perturbed["joint_error"] = np.linspace(1e8, -1e8, len(perturbed))
        perturbed["joint_covered90"] = ~perturbed["joint_covered90"]
        perturbed["false_confidence"] = ~perturbed["false_confidence"]
        perturbed["harmful_false_confidence"] = True
        perturbed["critical_bound_mass"] = 1.0
        perturbed["case"] = [f"oracle_label_{i}" for i in range(len(perturbed))]
        perturbed["is_ood"] = np.arange(len(perturbed)) % 2 == 0
        perturbed["adapted_average"] = np.linspace(-1e12, 1e12, len(perturbed))
        changed = apply_policy(perturbed, _policy())
        for column in (
            "accepted", "abstained", "abstention_reason", "policy_hash",
            "chi2_threshold",
        ):
            np.testing.assert_array_equal(
                baseline[column].to_numpy(), changed[column].to_numpy(),
            )

    def test_acceptance_sets_are_nested_with_declared_tie_rule(self):
        observations = build_phi_observation_table(_paired_raw())
        masks = [
            apply_policy(observations, _policy(), cutoff)["accepted"].to_numpy(bool)
            for cutoff in (0.10, 0.20, 1.00)
        ]
        np.testing.assert_array_equal(masks[0], [True, False, False, False])
        np.testing.assert_array_equal(masks[1], [True, True, True, False])
        np.testing.assert_array_equal(masks[2], [True, True, True, True])
        self.assertTrue(np.all(~masks[0] | masks[1]))
        self.assertTrue(np.all(~masks[1] | masks[2]))

        nonfinite = observations.copy()
        nonfinite.loc[0, "best_reduced_chi2"] = np.nan
        rejected = apply_policy(nonfinite, _policy(), 1.0).iloc[0]
        self.assertFalse(rejected.accepted)
        self.assertIn("nonfinite_diagnostic", rejected.abstention_reason)

    def test_phi_specific_bound_controls_acceptance_not_critical_mass(self):
        observations = build_phi_observation_table(_paired_raw())
        observations["best_reduced_chi2"] = 0.10
        high_critical = observations.copy()
        high_critical["critical_bound_mass"] = 1.0
        low_critical = observations.copy()
        low_critical["critical_bound_mass"] = 0.0
        np.testing.assert_array_equal(
            apply_policy(high_critical, _policy())["accepted"],
            apply_policy(low_critical, _policy())["accepted"],
        )
        self.assertTrue(apply_policy(high_critical, _policy())["accepted"].all())

        exact = observations.copy()
        exact["target_bound_mass"] = 0.10
        self.assertTrue(apply_policy(exact, _policy())["accepted"].all())
        above = exact.copy()
        above["target_bound_mass"] = np.nextafter(0.10, np.inf)
        self.assertFalse(apply_policy(above, _policy())["accepted"].any())

    def test_phi_pairing_rejects_duplicates_missing_rows_and_truth_disagreement(self):
        raw = _paired_raw()
        expected = build_phi_observation_table(raw)
        shuffled = build_phi_observation_table(
            raw.sample(frac=1.0, random_state=31).reset_index(drop=True)
        )
        pd.testing.assert_frame_equal(expected, shuffled)

        invalid = {
            "duplicate": pd.concat([raw, raw.iloc[[0]]], ignore_index=True),
            "missing": raw.drop(index=raw.index[0]).reset_index(drop=True),
            "truth": raw.assign(
                truth=np.where(
                    (raw["panel"].eq(0)) & raw["strategy"].eq("frozen_press"),
                    raw["truth"] + 0.01,
                    raw["truth"],
                )
            ),
        }
        for label, frame in invalid.items():
            with self.subTest(label=label), self.assertRaises(ValueError):
                build_phi_observation_table(frame)

    def test_zero_joint_acceptance_fails_closed(self):
        observations = _all_case_observations(n_panels=2)
        observations["best_reduced_chi2"] = 100.0
        selected = apply_policy(observations, _policy())
        self.assertFalse(selected["accepted"].any())
        summary = summarize_selective_cases(selected, n_bootstrap=100, seed=17)
        comparator = random_rejection_comparator(
            selected, n_permutations=100, seed=18,
        )
        gates = build_phase13_gates(summary, comparator)
        matched = gates.loc[gates["case"].eq("matched_ema_control")].iloc[0]
        self.assertEqual(matched.verdict, "INCONCLUSIVE_LOW_SUPPORT")
        self.assertFalse(matched.joint_pass)
        self.assertTrue(np.isnan(matched.accepted_coverage90))
        self.assertFalse(
            gates["verdict"].isin(["SAFE_JOINT", "SELECTIVE_PASS"]).any()
        )
        self.assertEqual(overall_phase13_decision(gates).decision.iloc[0], "STOP")

    def test_cluster_draw_resamples_complete_replicate_blocks(self):
        frame = pd.DataFrame({
            "panel": np.repeat(np.arange(4), 3),
            "replicate": np.tile(np.arange(3), 4),
        })
        indices = _cluster_draw_indices(frame, np.random.default_rng(51))
        self.assertEqual(len(indices), len(frame))
        multiplicity = np.bincount(indices, minlength=len(frame))
        for panel in frame["panel"].unique():
            local = multiplicity[frame["panel"].eq(panel).to_numpy()]
            self.assertTrue(np.all(local == local[0]))

    def test_overall_decision_uses_only_primary_frozen_verdicts(self):
        rows = []
        atomic_seen = 0
        for case in OFF_LIBRARY_CASES:
            if case.name == "matched_ema_control":
                verdict = "SAFE_JOINT"
            elif case.name == "combined_stress":
                verdict = "SAFE_DOMAIN_REJECT"
            elif atomic_seen < 2:
                verdict = "SELECTIVE_PASS"
                atomic_seen += 1
            else:
                verdict = "SAFE_JOINT"
                atomic_seen += 1
            rows.append({
                "case": case.name,
                "atomic": case.atomic,
                "verdict": verdict,
                "adapted_verdict": "SELECTIVE_PASS",
            })
        gates = pd.DataFrame(rows)
        self.assertEqual(
            overall_phase13_decision(gates).decision.iloc[0], "SELECTIVE_GO",
        )
        self.assertEqual(
            overall_phase13_decision(
                gates, negative_control_alert_count=1,
            ).decision.iloc[0],
            "CONDITIONAL",
        )
        unsafe = gates.copy()
        atomic_case = next(case.name for case in OFF_LIBRARY_CASES if case.atomic)
        unsafe.loc[unsafe["case"].eq(atomic_case), "verdict"] = "SILENT_FAILURE"
        unsafe["adapted_verdict"] = "SELECTIVE_PASS"
        self.assertEqual(overall_phase13_decision(unsafe).decision.iloc[0], "STOP")

    def test_observation_bank_persists_common_noise_identifiers(self):
        bank, index = reconstruct_observation_bank(
            n_panels=3, replicates=2, panel_size=4, seed=20261307,
        )
        n = len(OFF_LIBRARY_CASES) * 3 * 2
        self.assertEqual(len(index), n)
        self.assertFalse(index["observation_id"].duplicated().any())
        self.assertTrue(
            index.groupby(["panel", "replicate"])["noise_id"].nunique().eq(1).all()
        )
        self.assertTrue(
            index.groupby("noise_id")["case"].nunique().eq(len(OFF_LIBRARY_CASES)).all()
        )
        self.assertEqual(bank["elastic"].shape, (n, 4, 3))
        self.assertEqual(bank["log_rt"].shape, (n, 4, 2))
        self.assertEqual(bank["truth_targets"].shape, (n, 4))

        repeated_bank, repeated_index = reconstruct_observation_bank(
            n_panels=3, replicates=2, panel_size=4, seed=20261307,
        )
        pd.testing.assert_frame_equal(index, repeated_index)
        for name in ("elastic", "log_rt", "truth_targets"):
            np.testing.assert_array_equal(bank[name], repeated_bank[name])


if __name__ == "__main__":
    unittest.main()
