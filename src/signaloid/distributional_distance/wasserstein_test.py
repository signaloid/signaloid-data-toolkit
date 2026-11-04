#   Copyright (c) 2026, Signaloid.
#
#   Permission is hereby granted, free of charge, to any person obtaining a copy
#   of this software and associated documentation files (the "Software"), to
#   deal in the Software without restriction, including without limitation the
#   rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
#   sell copies of the Software, and to permit persons to whom the Software is
#   furnished to do so, subject to the following conditions:
#
#   The above copyright notice and this permission notice shall be included in
#   all copies or substantial portions of the Software.
#
#   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#   IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#   AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
#   FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
#   DEALINGS IN THE SOFTWARE.

"""Tests for signaloid.distributional_distance.wasserstein."""

import math
import subprocess
import sys
import unittest
from typing import Sequence

import numpy as np
from scipy.stats import wasserstein_distance  # type: ignore[import-untyped]

from signaloid.distributional.dirac_delta import DiracDelta
from signaloid.distributional.distributional import DistributionalValue
from signaloid.distributional_distance.wasserstein import (
    _wp_1d_weighted_pair,
    normalized_wasserstein_1_uxhw_wrapper,
    normalized_wasserstein_2_uxhw_wrapper,
    normalized_wasserstein_p_uxhw_wrapper,
    wasserstein_1_distance,
    wasserstein_1_distance_with_weights,
    wasserstein_1_uxhw_wrapper,
    wasserstein_2_uxhw_wrapper,
    wasserstein_p_uxhw_wrapper,
)

DEFAULT_SAMPLE_SIZE = 10_000


def _dv_from_weighted_samples(
    positions: Sequence[float], masses: Sequence[float]
) -> DistributionalValue:
    """Test-local helper: construct a DistributionalValue from explicit
    (positions, masses) arrays.

    """
    dirac_deltas = [
        DiracDelta(position=float(p), mass=float(m)) for p, m in zip(positions, masses)
    ]
    return DistributionalValue(dirac_deltas=dirac_deltas)


class TestWasserstein1Cores(unittest.TestCase):
    """Tests for wasserstein_1_distance and
    wasserstein_1_distance_with_weights — the two pre-existing pure-numpy
    W1 cores. Parity oracle is the new _wp_1d_weighted_pair (which
    implements the same scipy-style merged-CDF integral, with weights)."""

    def test_wasserstein_1_matches_weighted_pair_on_uniform_masses(self) -> None:
        """wasserstein_1_distance (unweighted) agrees with
        _wp_1d_weighted_pair (uniform weights) on the same samples."""
        rng = np.random.default_rng(seed=12345)
        mc_samples_1 = np.sort(rng.standard_normal(DEFAULT_SAMPLE_SIZE))
        mc_samples_2 = np.sort(rng.standard_normal(DEFAULT_SAMPLE_SIZE))

        distance_unweighted = wasserstein_1_distance(
            u_values=mc_samples_1,
            v_values=mc_samples_2,
        )

        uniform_u = np.ones(DEFAULT_SAMPLE_SIZE) / DEFAULT_SAMPLE_SIZE
        uniform_v = np.ones(DEFAULT_SAMPLE_SIZE) / DEFAULT_SAMPLE_SIZE
        distance_weighted = _wp_1d_weighted_pair(
            u_positions=mc_samples_1,
            u_masses=uniform_u,
            v_positions=mc_samples_2,
            v_masses=uniform_v,
            p=1,
        )

        self.assertAlmostEqual(distance_unweighted, distance_weighted, places=12)

    def test_wasserstein_1_with_weights_matches_weighted_pair(self) -> None:
        """wasserstein_1_distance_with_weights (unweighted vs weighted)
        agrees with _wp_1d_weighted_pair when u has uniform weights."""
        rng = np.random.default_rng(seed=67890)
        mc_samples = np.sort(rng.standard_normal(DEFAULT_SAMPLE_SIZE))

        positions = [-2.0, -0.5, 0.5, 2.0]
        masses = [0.15, 0.35, 0.35, 0.15]
        uxhw_dv = _dv_from_weighted_samples(positions, masses)

        distance_mixed = wasserstein_1_distance_with_weights(
            u_values=mc_samples,
            v_values=uxhw_dv.positions,
            v_cum_weights=np.cumsum(uxhw_dv.masses),
        )

        distance_paired = _wp_1d_weighted_pair(
            u_positions=mc_samples,
            u_masses=np.ones(DEFAULT_SAMPLE_SIZE) / DEFAULT_SAMPLE_SIZE,
            v_positions=uxhw_dv.positions,
            v_masses=uxhw_dv.masses,
            p=1,
        )

        self.assertAlmostEqual(distance_mixed, distance_paired, places=12)

    def test_wasserstein_1_with_weights_handles_v_idx_at_len(self) -> None:
        """Regression: when some u_values lie above max(v_values), the
        original port crashed with IndexError because v_idx hit
        len(v_values). The fixed lookup uses a leading-0 cumweights
        array so the tail CDF saturates at 1.0 cleanly."""
        u_values = np.array([0.0, 1.0, 2.0, 10.0])  # 10.0 is above v's max
        v_values = np.array([0.5, 1.5, 2.5])
        v_masses = np.array([0.2, 0.5, 0.3])

        # Must not raise.
        distance = wasserstein_1_distance_with_weights(
            u_values=u_values,
            v_values=v_values,
            v_cum_weights=np.cumsum(v_masses),
        )
        self.assertGreater(distance, 0.0)
        self.assertTrue(np.isfinite(distance))


class TestWasserstein1CoresWithTies(unittest.TestCase):
    """`wasserstein_1_distance` and `wasserstein_1_distance_with_weights`, which
    call `scipy.stats.wasserstein_distance`, agree with `_wp_1d_weighted_pair`,
    which integrates the quantile functions instead of the CDFs, also when
    samples tie."""

    def test_match_the_quantile_formulation_with_ties_and_shared_values(
        self,
    ) -> None:
        """Samples rounded to tenths have many ties, and v shares some with u.
        The weighted variant also matches scipy given the weights themselves,
        rather than their cumulative sum."""
        rng = np.random.default_rng(seed=162)
        for trial in range(50):
            u_values = np.sort(
                np.round(rng.standard_normal(int(rng.integers(1, 300))), 1)
            )
            v_values = np.sort(
                np.concatenate(
                    (
                        np.round(rng.normal(0.3, 1.2, int(rng.integers(1, 300))), 1),
                        u_values[:3],
                    )
                )
            )
            v_weights = rng.uniform(0.01, 1.0, len(v_values))

            u_masses = np.ones(len(u_values))
            with self.subTest(trial=trial):
                self.assertTrue(
                    math.isclose(
                        wasserstein_1_distance(u_values, v_values),
                        _wp_1d_weighted_pair(
                            u_values, u_masses, v_values, np.ones(len(v_values))
                        ),
                        rel_tol=1e-12,
                        abs_tol=1e-15,
                    )
                )
                weighted = wasserstein_1_distance_with_weights(
                    u_values, v_values, np.cumsum(v_weights)
                )
                self.assertTrue(
                    math.isclose(
                        weighted,
                        _wp_1d_weighted_pair(u_values, u_masses, v_values, v_weights),
                        rel_tol=1e-12,
                        abs_tol=1e-15,
                    )
                )
                self.assertTrue(
                    math.isclose(
                        weighted,
                        wasserstein_distance(u_values, v_values, v_weights=v_weights),
                        rel_tol=1e-12,
                        abs_tol=1e-15,
                    )
                )


class TestWeightedPairAnalytic(unittest.TestCase):
    """Analytic / closed-form tests for _wp_1d_weighted_pair."""

    def test_dirac_to_dirac_w1(self) -> None:
        """W1 between two single Diracs equals |a - b|."""
        for a, b in [(0.0, 1.0), (-3.5, 2.5), (10.0, 10.0), (1.0, 1.5)]:
            distance = _wp_1d_weighted_pair(
                u_positions=np.array([a]),
                u_masses=np.array([1.0]),
                v_positions=np.array([b]),
                v_masses=np.array([1.0]),
                p=1,
            )
            self.assertAlmostEqual(distance, abs(a - b), places=12, msg=f"a={a}, b={b}")

    def test_dirac_to_dirac_w2(self) -> None:
        """W2 between two single Diracs equals |a - b|."""
        for a, b in [(0.0, 1.0), (-3.5, 2.5), (10.0, 10.0), (1.0, 1.5)]:
            distance = _wp_1d_weighted_pair(
                u_positions=np.array([a]),
                u_masses=np.array([1.0]),
                v_positions=np.array([b]),
                v_masses=np.array([1.0]),
                p=2,
            )
            self.assertAlmostEqual(distance, abs(a - b), places=12, msg=f"a={a}, b={b}")

    def test_self_distance_is_zero(self) -> None:
        """W_p(X, X) = 0 for both p=1 and p=2."""
        positions = np.array([-1.0, 0.5, 2.5, 4.0])
        masses = np.array([0.1, 0.4, 0.3, 0.2])
        for p in (1, 2):
            distance = _wp_1d_weighted_pair(
                u_positions=positions,
                u_masses=masses,
                v_positions=positions,
                v_masses=masses,
                p=p,
            )
            self.assertAlmostEqual(distance, 0.0, places=12, msg=f"p={p}")

    def test_translation_invariance_w1(self) -> None:
        """W1 is translation-equivariant: W1(X + d, X) = d."""
        positions = np.array([-1.0, 0.5, 2.5, 4.0])
        masses = np.array([0.1, 0.4, 0.3, 0.2])
        for shift in [0.5, 1.0, -2.5, 3.0]:
            distance = _wp_1d_weighted_pair(
                u_positions=positions + shift,
                u_masses=masses,
                v_positions=positions,
                v_masses=masses,
                p=1,
            )
            self.assertAlmostEqual(
                distance, abs(shift), places=12, msg=f"shift={shift}"
            )

    def test_unnormalised_masses_are_normalised_internally(self) -> None:
        """Doubling all masses must not change the resulting distance."""
        u_pos = np.array([-1.0, 0.5, 2.5, 4.0])
        u_mass = np.array([0.1, 0.4, 0.3, 0.2])
        v_pos = np.array([-2.0, 1.0, 3.0])
        v_mass = np.array([0.2, 0.5, 0.3])

        canonical = _wp_1d_weighted_pair(
            u_positions=u_pos,
            u_masses=u_mass,
            v_positions=v_pos,
            v_masses=v_mass,
            p=1,
        )
        scaled = _wp_1d_weighted_pair(
            u_positions=u_pos,
            u_masses=u_mass * 7.0,
            v_positions=v_pos,
            v_masses=v_mass * 0.3,
            p=1,
        )
        self.assertAlmostEqual(canonical, scaled, places=12)


class TestWasserstein1UxhwWrapper(unittest.TestCase):
    """Tests for wasserstein_1_uxhw_wrapper."""

    def test_uxhw_wrapper_bootleg_zero(self) -> None:
        """W1 ≈ 0 when the under-test distribution and the ground truth
        are bootleg-identical (positions/masses constructed to produce
        identical empirical CDFs)."""
        testcases: list[tuple[list[float], list[float]]] = [
            ([1, 2, 3, 4, 5], [0.1, 0.2, 0.1, 0.4, 0.2]),
            ([10, 20, 30, 40, 50], [0.1, 0.2, 0.1, 0.4, 0.2]),
            ([10, 20, 30, 40, 50], [0.3, 0.1, 0.1, 0.3, 0.2]),
            ([-100, 20, 500, 1000, 0], [0.3, 0.1, 0.1, 0.3, 0.2]),
        ]

        for positions, masses in testcases:
            sample_arrays = []
            for pos, mass in zip(positions, masses):
                sample_arrays.append(
                    np.full(shape=math.floor(mass * 100), fill_value=pos)
                )

            test_dv = _dv_from_weighted_samples(positions, masses)
            samples = np.concatenate(sample_arrays, axis=None)
            ground_truth = DistributionalValue.from_samples(samples)
            result = wasserstein_1_uxhw_wrapper(test_dv, ground_truth)

            self.assertAlmostEqual(result, 0, places=6)

    def test_uxhw_wrapper_uniform_mother_zero(self) -> None:
        """W1 between a uniform(a, a+L) sample and its analytic TTR-N
        approximation matches the closed-form W1(L, N) = L / (4N).

        Source for the closed form:

        - for uniform(0, 1) the TTR positions are
              dd_pos(i) = (i + 0.5) / N for i in [0, N).
        - positions = [a + (i + 0.5) * L / N for i in range(N)].
        - W(L, N) = L / (4 * N).
        """
        rng = np.random.default_rng(seed=20260520)
        for a in [-2.0, 0.0, 2.0]:
            for L in [0.5, 1.0, 2.0]:
                samples = rng.uniform(a, a + L, size=DEFAULT_SAMPLE_SIZE)
                ground_truth = DistributionalValue.from_samples(samples)

                for N in [4, 16, 64, 128]:
                    positions = [a + (i + 0.5) * L / N for i in range(N)]
                    masses = [1 / N] * N
                    expected_distance = L / (4 * N)

                    test_dv = _dv_from_weighted_samples(positions, masses)
                    result = wasserstein_1_uxhw_wrapper(test_dv, ground_truth)

                    self.assertAlmostEqual(
                        result,
                        expected_distance,
                        delta=0.05,
                        msg=f"a={a}, L={L}, N={N}",
                    )


class TestWasserstein2UxhwWrapper(unittest.TestCase):
    """Tests for wasserstein_2_uxhw_wrapper."""

    def test_self_distance_zero(self) -> None:
        """W2(X, X) = 0."""
        positions = [-1.0, 0.5, 2.5, 4.0]
        masses = [0.1, 0.4, 0.3, 0.2]
        dv = _dv_from_weighted_samples(positions, masses)
        self.assertAlmostEqual(wasserstein_2_uxhw_wrapper(dv, dv), 0.0, places=12)

    def test_dirac_to_dirac(self) -> None:
        """W2 between two single Diracs equals |a - b|."""
        for a, b in [(0.0, 1.0), (-3.5, 2.5), (1.0, 1.5)]:
            u = _dv_from_weighted_samples([a], [1.0])
            v = _dv_from_weighted_samples([b], [1.0])
            self.assertAlmostEqual(
                wasserstein_2_uxhw_wrapper(u, v),
                abs(a - b),
                places=12,
                msg=f"a={a}, b={b}",
            )

    def test_w2_geq_w1(self) -> None:
        """W2 ≥ W1 for any pair of distributions (Hölder)."""
        rng = np.random.default_rng(seed=11111)
        u_pos = np.sort(rng.standard_normal(50))
        v_pos = np.sort(rng.standard_normal(50) + 0.5)
        u = _dv_from_weighted_samples(list(u_pos), [1.0 / 50] * 50)
        v = _dv_from_weighted_samples(list(v_pos), [1.0 / 50] * 50)
        w1 = wasserstein_1_uxhw_wrapper(u, v)
        w2 = wasserstein_2_uxhw_wrapper(u, v)
        self.assertGreaterEqual(w2, w1 - 1e-12)


class TestGenericWassersteinP(unittest.TestCase):
    """`wasserstein_p_uxhw_wrapper` and `normalized_wasserstein_p_uxhw_wrapper`
    are the generic surfaces. The W1 / W2 (+ normalised) wrappers are
    thin delegates. Verify delegation parity for p in {1, 2} and that
    the generic accepts non-canonical p without crashing."""

    def _pair(self) -> tuple[DistributionalValue, DistributionalValue]:
        u = _dv_from_weighted_samples([-1.0, 0.5, 2.5, 4.0], [0.1, 0.4, 0.3, 0.2])
        v = _dv_from_weighted_samples([-0.5, 1.0, 3.0], [0.25, 0.5, 0.25])
        return u, v

    def test_p1_matches_w1_wrapper(self) -> None:
        """wasserstein_p_uxhw_wrapper(u, v, p=1) == wasserstein_1_uxhw_wrapper(u, v)."""
        u, v = self._pair()
        self.assertAlmostEqual(
            wasserstein_p_uxhw_wrapper(u, v, p=1),
            wasserstein_1_uxhw_wrapper(u, v),
            places=12,
        )

    def test_p2_matches_w2_wrapper(self) -> None:
        """wasserstein_p_uxhw_wrapper(u, v, p=2) == wasserstein_2_uxhw_wrapper(u, v)."""
        u, v = self._pair()
        self.assertAlmostEqual(
            wasserstein_p_uxhw_wrapper(u, v, p=2),
            wasserstein_2_uxhw_wrapper(u, v),
            places=12,
        )

    def test_non_canonical_p_runs(self) -> None:
        """p=3 produces a finite, non-negative number. Guards against
        the kernel rejecting integers outside {1, 2} or downstream
        overflow on cubed differences."""
        u, v = self._pair()
        distance = wasserstein_p_uxhw_wrapper(u, v, p=3)
        self.assertTrue(math.isfinite(distance))
        self.assertGreaterEqual(distance, 0.0)

    def test_normalized_p1_matches_normalized_w1_wrapper(self) -> None:
        """Delegation parity for the normalised variant at p=1."""
        u, v = self._pair()
        self.assertAlmostEqual(
            normalized_wasserstein_p_uxhw_wrapper(u, v, p=1),
            normalized_wasserstein_1_uxhw_wrapper(u, v),
            places=12,
        )

    def test_normalized_p2_matches_normalized_w2_wrapper(self) -> None:
        """Delegation parity for the normalised variant at p=2."""
        u, v = self._pair()
        self.assertAlmostEqual(
            normalized_wasserstein_p_uxhw_wrapper(u, v, p=2),
            normalized_wasserstein_2_uxhw_wrapper(u, v),
            places=12,
        )


class TestNormalizedWasserstein(unittest.TestCase):
    """Tests for normalized_wasserstein_{1,2}_uxhw_wrapper."""

    def test_normalized_w1_scales_by_gt_range(self) -> None:
        """Normalised W1 = W1 / (gt.max - gt.min) when the range is
        nonzero."""
        u = _dv_from_weighted_samples([0.0, 4.0, 8.0], [0.3, 0.4, 0.3])
        ground_truth = _dv_from_weighted_samples([-1.0, 9.0], [0.5, 0.5])
        raw = wasserstein_1_uxhw_wrapper(u, ground_truth)
        normalised = normalized_wasserstein_1_uxhw_wrapper(u, ground_truth)
        expected = raw / (9.0 - (-1.0))
        self.assertAlmostEqual(normalised, expected, places=12)

    def test_normalized_w2_scales_by_gt_range(self) -> None:
        """Normalised W2 = W2 / (gt.max - gt.min) when the range is
        nonzero."""
        u = _dv_from_weighted_samples([0.0, 4.0, 8.0], [0.3, 0.4, 0.3])
        ground_truth = _dv_from_weighted_samples([-1.0, 9.0], [0.5, 0.5])
        raw = wasserstein_2_uxhw_wrapper(u, ground_truth)
        normalised = normalized_wasserstein_2_uxhw_wrapper(u, ground_truth)
        expected = raw / (9.0 - (-1.0))
        self.assertAlmostEqual(normalised, expected, places=12)

    def test_normalized_w1_zero_range_fallback(self) -> None:
        """When the ground-truth support is a single point, the
        normalising factor falls back to 1.0 — so normalised == raw."""
        u = _dv_from_weighted_samples([2.0, 4.0], [0.5, 0.5])
        ground_truth = _dv_from_weighted_samples([3.0], [1.0])
        raw = wasserstein_1_uxhw_wrapper(u, ground_truth)
        normalised = normalized_wasserstein_1_uxhw_wrapper(u, ground_truth)
        self.assertAlmostEqual(normalised, raw, places=12)


class TestWeightedPairValidation(unittest.TestCase):
    """Regression tests for input validation in _wp_1d_weighted_pair.

    DistributionalValue does NOT enforce these invariants (special
    values like NaN/Inf are first-class via nan_dirac_delta etc.), so
    the helper must reject them itself or it silently returns
    nan/inf/wrong-but-plausible numbers.
    """

    def _good_v(self) -> DistributionalValue:
        return _dv_from_weighted_samples([1.0, 2.0, 3.0], [0.3, 0.4, 0.3])

    def test_non_finite_position_rejected(self) -> None:
        """NaN / +inf / -inf positions are rejected by the finite check."""
        for label, positions in (
            ("NaN", [1.0, 2.0, np.nan]),
            ("+inf", [1.0, 2.0, np.inf]),
            ("-inf", [-np.inf, 2.0, 3.0]),
        ):
            with self.subTest(special=label):
                u = _dv_from_weighted_samples(positions, [0.3, 0.4, 0.3])
                with self.assertRaises(ValueError) as ctx:
                    wasserstein_1_uxhw_wrapper(u, self._good_v())
                self.assertIn("finite", str(ctx.exception))

    def test_bad_masses_rejected(self) -> None:
        """All-zero / NaN masses each hit a distinct branch of
        `_validate_wp_inputs` with a distinct error message.

        Negative-mass rejection lives in
        TestDistributionalValueInitMassValidation — DistributionalValue
        rejects negative masses at construction, so they can't reach
        `_validate_wp_inputs` through a DV-shaped input. The validator's
        non-negative check is still reachable for direct array callers
        of `_wp_1d_weighted_pair`."""
        cases = (
            ("all-zero", [0.0, 0.0, 0.0], "positive total mass"),
            ("nan", [np.nan, 0.5, 0.5], "finite"),
        )
        for label, masses, expected in cases:
            with self.subTest(case=label):
                u = _dv_from_weighted_samples([1.0, 2.0, 3.0], masses)
                with self.assertRaises(ValueError) as ctx:
                    wasserstein_1_uxhw_wrapper(u, self._good_v())
                self.assertIn(expected, str(ctx.exception))

    def test_ground_truth_side_validated_too(self) -> None:
        """Validation triggers symmetrically — v_values also pass through
        `_validate_wp_inputs` and reject NaN."""
        v = _dv_from_weighted_samples([1.0, 2.0, np.nan], [0.3, 0.4, 0.3])
        with self.assertRaises(ValueError):
            wasserstein_1_uxhw_wrapper(self._good_v(), v)

    def test_helper_rejects_empty_arrays(self) -> None:
        """Direct call: empty positions are rejected at the helper."""
        with self.assertRaises(ValueError) as ctx:
            _wp_1d_weighted_pair(
                u_positions=np.array([]),
                u_masses=np.array([]),
                v_positions=np.array([1.0]),
                v_masses=np.array([1.0]),
                p=1,
            )
        self.assertIn("non-empty", str(ctx.exception))

    def test_helper_rejects_mismatched_lengths(self) -> None:
        with self.assertRaises(ValueError) as ctx:
            _wp_1d_weighted_pair(
                u_positions=np.array([1.0, 2.0, 3.0]),
                u_masses=np.array([0.5, 0.5]),
                v_positions=np.array([1.0]),
                v_masses=np.array([1.0]),
                p=1,
            )
        self.assertIn("matching lengths", str(ctx.exception))

    def test_helper_rejects_bad_p(self) -> None:
        for bad_p in (0, -1, 1.5, "1"):
            with self.assertRaises(ValueError):
                _wp_1d_weighted_pair(
                    u_positions=np.array([1.0]),
                    u_masses=np.array([1.0]),
                    v_positions=np.array([2.0]),
                    v_masses=np.array([1.0]),
                    p=bad_p,  # type: ignore[arg-type]
                )

    def test_helper_rejects_bool_p(self) -> None:
        """`bool` is a subclass of `int`, so `p=True` would otherwise
        slip through `isinstance(p, int)`. The validator rejects it
        explicitly so callers don't silently get `p=1`."""
        for bad_p in (True, False):
            with self.assertRaises(ValueError):
                _wp_1d_weighted_pair(
                    u_positions=np.array([1.0]),
                    u_masses=np.array([1.0]),
                    v_positions=np.array([2.0]),
                    v_masses=np.array([1.0]),
                    p=bad_p,
                )

    def test_helper_rejects_2d_arrays(self) -> None:
        with self.assertRaises(ValueError) as ctx:
            _wp_1d_weighted_pair(
                u_positions=np.array([[1.0, 2.0]]),
                u_masses=np.array([[0.5, 0.5]]),
                v_positions=np.array([1.0]),
                v_masses=np.array([1.0]),
                p=1,
            )
        self.assertIn("one-dimensional", str(ctx.exception))

    def test_helper_rejects_float_p(self) -> None:
        """`p=1.5` is not an int and must be rejected with a message
        identifying the offending parameter."""
        with self.assertRaises(ValueError) as ctx:
            _wp_1d_weighted_pair(
                u_positions=np.array([1.0]),
                u_masses=np.array([1.0]),
                v_positions=np.array([2.0]),
                v_masses=np.array([1.0]),
                p=1.5,  # type: ignore[arg-type]
            )
        self.assertIn("p", str(ctx.exception))


class TestWrapperTypeValidation(unittest.TestCase):
    """`_require_distributional_pair` rejects non-DistributionalValue
    arguments before any algorithmic work happens. Exercised here for
    each of the four public uxhw wrappers."""

    def _good_dv(self) -> DistributionalValue:
        return _dv_from_weighted_samples([1.0, 2.0, 3.0], [0.3, 0.4, 0.3])

    def test_all_wrappers_reject_non_dv(self) -> None:
        """All four uxhw wrappers funnel through
        `_require_distributional_pair`. Exercise each one × each side
        with a distinct non-DV stand-in so a future refactor that
        bypassed the helper in any single wrapper would be caught.

        Message-format coverage (which argument name appears, which type
        name, etc.) lives in `_validators_test.py`. Here we just verify
        the wrapper raises `ValueError` mentioning the expected type."""
        good = self._good_dv()
        wrappers = (
            ("wasserstein_1_uxhw_wrapper", wasserstein_1_uxhw_wrapper),
            (
                "normalized_wasserstein_1_uxhw_wrapper",
                normalized_wasserstein_1_uxhw_wrapper,
            ),
            ("wasserstein_2_uxhw_wrapper", wasserstein_2_uxhw_wrapper),
            (
                "normalized_wasserstein_2_uxhw_wrapper",
                normalized_wasserstein_2_uxhw_wrapper,
            ),
        )
        bad_stand_ins = ("not a dv", 42, None, [1.0, 2.0], np.array([1.0]))
        for name, wrapper in wrappers:
            for bad in bad_stand_ins:
                for side, args in (
                    ("first", (bad, good)),
                    ("second", (good, bad)),
                ):
                    with self.subTest(wrapper=name, side=side, bad=type(bad).__name__):
                        with self.assertRaises(ValueError) as ctx:
                            wrapper(*args)
                        self.assertIn("DistributionalValue", str(ctx.exception))

    def test_normalized_wrappers_reject_empty_dv(self) -> None:
        """Regression: normalized wrappers used to call `.min()`/`.max()`
        on `ground_truth.positions` BEFORE `_wp_1d_weighted_pair`'s
        empty check, so an empty DistributionalValue produced an
        opaque numpy reduction error instead of a clear ValueError.
        Both sides routed through `_rescale_to_gt_support`'s explicit
        non-empty check now."""
        empty = DistributionalValue(dirac_deltas=[])
        good = self._good_dv()
        for wrapper in (
            normalized_wasserstein_1_uxhw_wrapper,
            normalized_wasserstein_2_uxhw_wrapper,
        ):
            for side, uxhw, gt in (
                ("uxhw", empty, good),
                ("ground_truth", good, empty),
            ):
                with self.subTest(wrapper=wrapper.__name__, side=side):
                    with self.assertRaises(ValueError) as ctx:
                        wrapper(uxhw, gt)
                    self.assertIn("non-empty", str(ctx.exception))


class TestLegacyHelperValidation(unittest.TestCase):
    """`wasserstein_1_distance_with_weights` is a legacy public helper
    that previously crashed with IndexError on empty `v_cum_weights`
    (line: `v_cum_weights[-1]`). Now raises ValueError up-front."""

    def test_empty_v_cum_weights_rejected(self) -> None:
        with self.assertRaises(ValueError) as ctx:
            wasserstein_1_distance_with_weights(
                u_values=np.array([1.0, 2.0]),
                v_values=np.array([]),
                v_cum_weights=np.array([]),
            )
        self.assertIn("v_cum_weights", str(ctx.exception))

    def test_empty_u_values_rejected(self) -> None:
        with self.assertRaises(ValueError) as ctx:
            wasserstein_1_distance_with_weights(
                u_values=np.array([]),
                v_values=np.array([1.0, 2.0]),
                v_cum_weights=np.cumsum(np.array([0.5, 0.5])),
            )
        self.assertIn("u_values", str(ctx.exception))

    def test_v_values_v_cum_weights_length_mismatch_rejected(self) -> None:
        """Regression: mismatched lengths previously let v_idx exceed
        len(v_cum_weights) and raise IndexError mid-loop."""
        with self.assertRaises(ValueError) as ctx:
            wasserstein_1_distance_with_weights(
                u_values=np.array([1.0]),
                v_values=np.array([1.0, 2.0, 3.0]),
                v_cum_weights=np.cumsum(np.array([0.5, 0.5])),  # length 2 vs 3
            )
        self.assertIn("same length", str(ctx.exception))

    def test_zero_v_total_weight_rejected(self) -> None:
        """v_cum_weights[-1] == 0 would cause a divide-by-zero NaN."""
        with self.assertRaises(ValueError) as ctx:
            wasserstein_1_distance_with_weights(
                u_values=np.array([1.0, 2.0]),
                v_values=np.array([1.0, 2.0]),
                v_cum_weights=np.cumsum(np.array([0.0, 0.0])),
            )
        self.assertIn("strictly positive", str(ctx.exception))

    def test_wasserstein_1_distance_rejects_empty(self) -> None:
        """Regression: `wasserstein_1_distance` divides by
        `len(u_values)` / `len(v_values)` — empty inputs previously
        raised an opaque ZeroDivisionError. Now raises ValueError
        with the offending parameter name, matching the
        `_with_weights` sibling."""
        for label, u, v in (
            ("empty u_values", np.array([]), np.array([1.0, 2.0])),
            ("empty v_values", np.array([1.0, 2.0]), np.array([])),
        ):
            with self.subTest(case=label):
                with self.assertRaises(ValueError) as ctx:
                    wasserstein_1_distance(u_values=u, v_values=v)
                expected = "u_values" if "u_values" in label else "v_values"
                self.assertIn(expected, str(ctx.exception))

    def test_all_values_is_deprecated_and_ignored(self) -> None:
        """Passing `all_values`, which scipy does not need, warns and changes
        nothing."""
        u_values = np.array([0.0, 3.0, 10.0])
        v_values = np.array([0.5, 1.5])
        v_cum_weights = np.cumsum(np.array([0.3, 0.7]))
        all_values = np.sort(np.concatenate((u_values, v_values)))

        with self.assertWarns(DeprecationWarning):
            unweighted = wasserstein_1_distance(u_values, v_values, all_values)
        with self.assertWarns(DeprecationWarning):
            weighted = wasserstein_1_distance_with_weights(
                u_values, v_values, v_cum_weights, all_values
            )

        self.assertEqual(unweighted, wasserstein_1_distance(u_values, v_values))
        self.assertEqual(
            weighted,
            wasserstein_1_distance_with_weights(u_values, v_values, v_cum_weights),
        )

    def test_importing_the_package_does_not_load_scipy(self) -> None:
        """scipy.stats takes about a third of a second to load, so it is only
        loaded once a function that uses it is called. This test module loads it
        itself, so a new interpreter checks."""
        loaded = subprocess.run(
            [
                sys.executable,
                "-c",
                "import sys, signaloid.distributional_distance; "
                "print('scipy.stats' in sys.modules)",
            ],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()

        self.assertEqual(loaded, "False")


if __name__ == "__main__":
    unittest.main()
