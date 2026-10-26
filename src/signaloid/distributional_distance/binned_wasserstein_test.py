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

"""Tests for signaloid.distributional_distance.binned_wasserstein."""

import math
import unittest
from typing import Sequence

import numpy as np

from signaloid.distributional.dirac_delta import DiracDelta
from signaloid.distributional.distributional import DistributionalValue
from signaloid.distributional_distance.binned_wasserstein import (
    binned_wasserstein_1_ux_string_wrapper,
    binned_wasserstein_1_uxhw_wrapper,
    wasserstein_1_between_distribution_and_samples,
)
from signaloid.distributional_distance.wasserstein import (
    wasserstein_1_uxhw_wrapper,
)

DEFAULT_SAMPLE_SIZE = 1_000

# Fixture data used by the three `TestBinnedWasserstein` parametric tests.
# Each (samples, expected) pair is the CDF-based W1 distance between the
# UxHw bin specified by `_FIXTURE_UXHW_*` and the corresponding sample
# set. Kept at module scope to avoid the previous 3x duplication.
_FIXTURE_SAMPLE_SETS: list[list[float]] = [
    [-1.5],
    [0.0],
    [1.5],
    [2.5],
    [-2.0, -1.5],
    [-2.0, 0.0],
    [0.0, 1.75],
    [1.75, 2.5],
    [2.0, 2.5],
]
_FIXTURE_EXPECTED_RESULTS: list[float] = [
    1.875,
    0.75,
    1.1875,
    2.125,
    2.125,
    1.375,
    0.5572916666666667,
    1.75,
    1.875,
]
_FIXTURE_UXHW_BOUNDARIES: list[float] = [-1.0, 1.0, 2.0]
_FIXTURE_UXHW_WIDTHS: list[float] = [2.0, 1.0]
_FIXTURE_UXHW_HEIGHTS: list[float] = [0.375, 0.25]


def _dv_from_weighted_samples(
    positions: Sequence[float], masses: Sequence[float]
) -> DistributionalValue:
    """Test-local helper: construct a DistributionalValue from explicit
    (positions, masses) arrays. See sibling wasserstein_test.py for the
    rationale (we don't add `from_weighted_samples` to DistributionalValue)."""
    dirac_deltas = [
        DiracDelta(position=float(p), mass=float(m)) for p, m in zip(positions, masses)
    ]
    return DistributionalValue(dirac_deltas=dirac_deltas)


class TestBinnedWasserstein(unittest.TestCase):
    """Tests for wasserstein_1_between_distribution_and_samples and
    binned_wasserstein_1_uxhw_wrapper."""

    def test_cdf_based_wasserstein1_distance_between_uxhw_and_samples(self) -> None:
        """Test the CDF-based W1 distance between a UxHw distribution and
        an empirical sample distribution. Single examples before / between
        / after the distribution CDF, plus mixed cases covering CDF
        intersections."""
        for test_id, (samples, expected) in enumerate(
            zip(_FIXTURE_SAMPLE_SETS, _FIXTURE_EXPECTED_RESULTS)
        ):
            distance = wasserstein_1_between_distribution_and_samples(
                _FIXTURE_UXHW_BOUNDARIES,
                _FIXTURE_UXHW_HEIGHTS,
                _FIXTURE_UXHW_WIDTHS,
                samples,
            )
            self.assertAlmostEqual(
                distance,
                expected,
                places=12,
                msg=f"test_id={test_id} samples={samples}",
            )

    def test_cdf_based_wasserstein1_distance_between_uxhw_and_weighted_samples(
        self,
    ) -> None:
        """Exercises the weighted-samples code path. Each sample gets a
        deliberately non-normalised uniform weight of 7.0 — the
        function must normalise internally and produce the same
        distances as the unweighted case."""
        for test_id, (samples, expected) in enumerate(
            zip(_FIXTURE_SAMPLE_SETS, _FIXTURE_EXPECTED_RESULTS)
        ):
            # Non-normalised uniform weights — internal normalisation
            # must give the same answer as the unweighted case.
            non_normalised_weights = [7.0] * len(samples)
            distance = wasserstein_1_between_distribution_and_samples(
                _FIXTURE_UXHW_BOUNDARIES,
                _FIXTURE_UXHW_HEIGHTS,
                _FIXTURE_UXHW_WIDTHS,
                samples,
                sample_weights=non_normalised_weights,
            )
            self.assertAlmostEqual(
                distance,
                expected,
                places=12,
                msg=f"test_id={test_id} samples={samples}",
            )

    def test_unnormalised_uxhw_heights_normalise_internally(
        self,
    ) -> None:
        """The UxHw bin heights are deliberately not normalised — the
        function must handle this and produce the same distances as the
        normalised-heights case. (Previously misnamed
        `..._between_wrong_uxhw_and_weighted_samples` — does NOT pass
        sample_weights.)"""
        # Heights are 2x the normalised values — function must normalise.
        unnormalised_heights = [h * 2.0 for h in _FIXTURE_UXHW_HEIGHTS]

        for test_id, (samples, expected) in enumerate(
            zip(_FIXTURE_SAMPLE_SETS, _FIXTURE_EXPECTED_RESULTS)
        ):
            distance = wasserstein_1_between_distribution_and_samples(
                _FIXTURE_UXHW_BOUNDARIES,
                unnormalised_heights,
                _FIXTURE_UXHW_WIDTHS,
                samples,
            )
            self.assertAlmostEqual(
                distance,
                expected,
                places=12,
                msg=f"test_id={test_id} samples={samples}",
            )

    def test_binned_wasserstein_1_uxhw_wrapper(self) -> None:
        """binned_wasserstein_1_uxhw_wrapper produces a deterministic,
        small-but-strictly-positive distance between a coarse 4-point
        symmetric uxhw and a 1000-sample N(0, 1) MC ground truth. The
        RNG is seeded. The assertion has a tight window around the
        precomputed result so silent numerical regressions surface."""
        rng = np.random.default_rng(seed=20260520)
        mc_sample = rng.standard_normal(DEFAULT_SAMPLE_SIZE)

        # Arbitrary weighted distribution spanning the sample support.
        positions = [-2.0, -0.5, 0.5, 2.0]
        masses = [0.15, 0.35, 0.35, 0.15]
        uxhw = _dv_from_weighted_samples(positions, masses)
        ground_truth = DistributionalValue.from_samples(mc_sample)

        result = binned_wasserstein_1_uxhw_wrapper(
            binned_dist=uxhw, ground_truth_dist=ground_truth
        )

        # Precomputed against the seeded RNG above. W1 is a metric so
        # strictly > 0 for non-identical distributions.
        self.assertAlmostEqual(result, 0.18323214623666018, places=12)

    def test_dense_crossings_and_coincident_points(self) -> None:
        """Exercises the vectorised core at scale: a bimodal 64-bin
        distribution against ~2000 weighted samples from a unimodal
        Gaussian, so the two CDFs cross many times. Every other bin
        boundary also appears as a sample and three samples are
        triplicated, so the merge sort sees tied positions of both kinds.
        Inputs are passed as numpy arrays. The expected value was computed
        with the pre-vectorisation, loop-based `wasserstein_1_core`."""
        rng = np.random.default_rng(seed=20260922)
        boundaries = np.linspace(-3.0, 3.0, 65)
        widths = np.diff(boundaries)
        centres = 0.5 * (boundaries[:-1] + boundaries[1:])
        heights = np.exp(-2.0 * (centres - 1.5) ** 2) + np.exp(
            -2.0 * (centres + 1.5) ** 2
        )
        heights = heights / np.sum(heights * widths)
        samples = np.concatenate(
            [
                boundaries[::2],
                np.repeat([-1.0, 0.0, 1.0], 3),
                rng.normal(0.0, 1.2, 2000),
            ]
        )
        weights = rng.random(len(samples))

        distance = wasserstein_1_between_distribution_and_samples(
            boundaries, heights, widths, samples, weights
        )

        self.assertAlmostEqual(distance, 0.5794905929410978, places=12)


class TestBinnedInputValidation(unittest.TestCase):
    """Regression tests for input validation in
    `wasserstein_1_between_distribution_and_samples`. This function
    accepts raw list[float] from external callers, so structural
    invariants (length, non-empty) must be enforced explicitly."""

    def _good_bins(self) -> tuple[list[float], list[float], list[float]]:
        return ([-1.0, 1.0, 2.0], [0.375, 0.25], [2.0, 1.0])

    def _call(self, **overrides: object) -> None:
        """Invoke `wasserstein_1_between_distribution_and_samples` with a
        baseline of valid kwargs, overriding any the caller supplies."""
        boundaries, heights, widths = self._good_bins()
        kwargs: dict[str, object] = {
            "bin_boundaries": boundaries,
            "bin_heights": heights,
            "bin_widths": widths,
            "sample_positions": [0.5],
        }
        kwargs.update(overrides)
        wasserstein_1_between_distribution_and_samples(**kwargs)  # type: ignore[arg-type]

    def test_structural_rejections(self) -> None:
        """Length / non-empty checks in `_validate_binned_lengths`."""
        cases: tuple[tuple[str, dict[str, object], str], ...] = (
            (
                "empty bin_widths",
                {"bin_boundaries": [0.0], "bin_heights": [], "bin_widths": []},
                "non-empty",
            ),
            (
                "widths/heights mismatch",
                {"bin_heights": [0.375, 0.25, 0.1]},
                "same length",
            ),
            (
                "boundaries length wrong",
                {"bin_boundaries": [-1.0, 1.0]},
                "one more entry",
            ),
            (
                "empty sample_positions",
                {"sample_positions": []},
                "not be empty",
            ),
            (
                "samples/weights mismatch",
                {"sample_positions": [0.5, 1.5], "sample_weights": [1.0]},
                "same length",
            ),
        )
        for label, overrides, expected in cases:
            with self.subTest(case=label):
                with self.assertRaises(ValueError) as ctx:
                    self._call(**overrides)
                self.assertIn(expected, str(ctx.exception))

    def test_semantic_rejections(self) -> None:
        """Non-negativity / positivity / non-finite checks in
        `_validate_binned_semantics` — including the hidden-negative
        regressions where cumsum masks the bad entry."""
        cases: tuple[tuple[str, dict[str, object], str], ...] = (
            ("negative height", {"bin_heights": [-0.1, 0.25]}, "bin_heights"),
            ("zero width", {"bin_widths": [2.0, 0.0]}, "bin_widths"),
            ("negative width", {"bin_widths": [2.0, -1.0]}, "bin_widths"),
            (
                "all-zero bin_heights",
                {"bin_heights": [0.0, 0.0]},
                "bin_heights",
            ),
            (
                "all-zero sample_weights",
                {"sample_positions": [0.5, 1.5], "sample_weights": [0.0, 0.0]},
                "sample_weights",
            ),
            (
                "single sample with zero weight",
                {"sample_positions": [0.5], "sample_weights": [0.0]},
                "sample_weights",
            ),
            (
                "negative sample_weight masked by cumsum",
                {
                    "sample_positions": [-0.5, 0.5, 1.5],
                    "sample_weights": [0.5, -0.1, 0.6],
                },
                "sample_weights",
            ),
            (
                "negative bin_height masked by cumsum",
                {
                    "bin_boundaries": [-1.0, 0.0, 1.0, 2.0],
                    "bin_heights": [0.5, -0.1, 0.6],
                    "bin_widths": [1.0, 1.0, 1.0],
                },
                "bin_heights",
            ),
            ("NaN in bin_heights", {"bin_heights": [float("nan"), 0.25]}, "finite"),
            (
                "Inf in sample_positions",
                {"sample_positions": [0.5, float("inf")]},
                "finite",
            ),
            (
                "NaN in sample_weights",
                {
                    "sample_positions": [0.5, 1.5],
                    "sample_weights": [0.5, float("nan")],
                },
                "finite",
            ),
            (
                "non-monotonic boundaries",
                {"bin_boundaries": [1.0, -1.0, 2.0]},
                "strictly increasing",
            ),
            (
                "widths inconsistent with boundary diffs",
                {"bin_widths": [3.0, 1.0]},
                "match the implied widths",
            ),
        )
        for label, overrides, expected in cases:
            with self.subTest(case=label):
                with self.assertRaises(ValueError) as ctx:
                    self._call(**overrides)
                self.assertIn(expected, str(ctx.exception))


class TestSampleSortingRegression(unittest.TestCase):
    """Regression tests for the unsorted-sample-positions bug.

    Before the fix, np.cumsum(sample_weights) was computed in input
    order without first sorting (positions, weights) by position. The
    empirical CDF was therefore wrong whenever positions were unsorted
    and weights were non-uniform — and `ground_truth.positions` is
    typically unsorted unless `.sort()` is called first."""

    def test_permuting_samples_gives_same_distance(self) -> None:
        """Same multiset of (position, weight), two orderings — must
        give identical distances."""
        boundaries = [-1.0, 1.0, 2.0]
        widths = [2.0, 1.0]
        heights = [0.375, 0.25]

        sorted_pos = [-1.0, 0.5, 1.5]
        sorted_w = [0.1, 0.6, 0.3]

        # Permutation of (position, weight) pairs.
        unsorted_pos = [1.5, -1.0, 0.5]
        unsorted_w = [0.3, 0.1, 0.6]

        d_sorted = wasserstein_1_between_distribution_and_samples(
            boundaries, heights, widths, sorted_pos, sorted_w
        )
        d_unsorted = wasserstein_1_between_distribution_and_samples(
            boundaries, heights, widths, unsorted_pos, unsorted_w
        )
        self.assertAlmostEqual(d_sorted, d_unsorted, places=12)

    def test_translation_invariance(self) -> None:
        """W1 is translation-invariant: shifting both distribution and
        samples by the same constant must leave the distance unchanged
        (modulo floating-point noise). This catches accidental absolute-
        position dependence introduced by future refactors."""
        boundaries = list(_FIXTURE_UXHW_BOUNDARIES)
        widths = list(_FIXTURE_UXHW_WIDTHS)
        heights = list(_FIXTURE_UXHW_HEIGHTS)
        samples = [-2.0, 0.0, 1.75]
        shift = 17.5

        d_base = wasserstein_1_between_distribution_and_samples(
            boundaries, heights, widths, samples
        )
        d_shifted = wasserstein_1_between_distribution_and_samples(
            [b + shift for b in boundaries],
            heights,
            widths,
            [s + shift for s in samples],
        )
        self.assertAlmostEqual(d_base, d_shifted, places=12)

    def test_binned_uxhw_wrapper_handles_unsorted_ground_truth(self) -> None:
        """The wrapper passes ground_truth.positions/masses straight in
        without sorting. Verify that two equivalent-but-permuted
        ground truths produce the same distance."""
        positions_sorted = [-2.0, -0.5, 0.5, 2.0]
        masses_sorted = [0.15, 0.35, 0.35, 0.15]
        positions_perm = [2.0, -0.5, -2.0, 0.5]
        masses_perm = [0.15, 0.35, 0.15, 0.35]

        uxhw = _dv_from_weighted_samples([-1.0, 0.0, 1.0], [0.2, 0.6, 0.2])
        gt_sorted = _dv_from_weighted_samples(positions_sorted, masses_sorted)
        gt_perm = _dv_from_weighted_samples(positions_perm, masses_perm)

        d_sorted = binned_wasserstein_1_uxhw_wrapper(
            binned_dist=uxhw, ground_truth_dist=gt_sorted
        )
        d_perm = binned_wasserstein_1_uxhw_wrapper(
            binned_dist=uxhw, ground_truth_dist=gt_perm
        )
        self.assertAlmostEqual(d_sorted, d_perm, places=12)

    def test_sample_coincident_with_boundary(self) -> None:
        """Regression: when a sample sits exactly at a bin boundary, the
        merge-sort tie-break must put the boundary first so
        `wasserstein_1_core`'s segment-counter increments before the
        step-counter. `np.argsort` is not guaranteed stable across
        numpy versions. `np.lexsort((types, positions))` pins the order.

        Hand-computed: uxhw is uniform on [0, 2] (heights=0.5, widths=1
        across boundaries [0, 1, 2]). Sample is δ at x=1 (the interior
        boundary). F_uxhw(x) = x/2 for x ∈ [0, 2]. F_sample is the unit
        step at x=1. W1 = ∫_0^1 (x/2) dx + ∫_1^2 (1 − x/2) dx
        = 0.25 + 0.25 = 0.5.
        """
        distance = wasserstein_1_between_distribution_and_samples(
            bin_boundaries=[0.0, 1.0, 2.0],
            bin_heights=[0.5, 0.5],
            bin_widths=[1.0, 1.0],
            sample_positions=[1.0],
        )
        self.assertAlmostEqual(distance, 0.5, places=12)


class TestBinnedWrapperGroundTruthValidation(unittest.TestCase):
    """The binned wrapper must validate `ground_truth` symmetrically
    with `uxhw` — previously only uxhw was checked, so a non-DV
    ground_truth would fail later with AttributeError."""

    def test_non_dv_rejected_on_either_side(self) -> None:
        """Message-format coverage lives in `_validators_test.py`.
        Here we only assert the wrapper raises `ValueError` on either
        side, mentioning the expected type."""
        good = _dv_from_weighted_samples([-1.0, 0.0, 1.0], [0.2, 0.6, 0.2])
        for side, binned, gt in (
            ("first", "not a DV", good),
            ("second", good, "not a DV"),
        ):
            with self.subTest(side=side):
                with self.assertRaises(ValueError) as ctx:
                    binned_wasserstein_1_uxhw_wrapper(binned_dist=binned, ground_truth_dist=gt)  # type: ignore[arg-type]
                self.assertIn("DistributionalValue", str(ctx.exception))


class TestBinnedWrapperSingleDiracShortCircuit(unittest.TestCase):
    """`binned_wasserstein_1_uxhw_wrapper` short-circuits to the
    non-binned W1 wrapper when uxhw has a single Dirac delta, since
    binning a single point is ill-defined. Verify the short-circuit
    triggers and matches the non-binned W1 result."""

    def test_single_dirac_uxhw_delegates_to_w1(self) -> None:
        uxhw = _dv_from_weighted_samples([2.5], [1.0])
        ground_truth = _dv_from_weighted_samples(
            [-1.0, 0.0, 1.0, 2.0, 3.0], [0.1, 0.2, 0.4, 0.2, 0.1]
        )
        # Short-circuit must yield the same value as the W1 wrapper
        # for a single-Dirac uxhw.
        d_binned = binned_wasserstein_1_uxhw_wrapper(
            binned_dist=uxhw, ground_truth_dist=ground_truth
        )
        d_w1 = wasserstein_1_uxhw_wrapper(dist_u=uxhw, dist_v=ground_truth)
        self.assertAlmostEqual(d_binned, d_w1, places=12)

    def test_short_circuit_with_special_values_present(self) -> None:
        """Regression: the short-circuit check used to look at
        `len(uxhw.positions) == 1`, but DistributionalValue.positions
        includes NaN/±Inf Dirac deltas (after sort) when they carry
        non-zero mass. A DV with 1 finite Dirac + special-value mass
        would skip the short-circuit and pass to PlotData on a
        single-finite-Dirac binning — ill-defined. Checking
        `finite_dirac_deltas` instead routes correctly to the
        non-binned W1 wrapper.
        """
        # Construct: 1 finite Dirac at x=1.0 plus a NaN-position Dirac
        # carrying non-zero mass. After `sort()` the special-value
        # bucket gets populated. `positions` then includes nan/-inf/inf
        # (length 4), so the old check would not short-circuit.
        uxhw_finite_with_special = DistributionalValue(
            dirac_deltas=[
                DiracDelta(position=1.0, mass=0.5),
                DiracDelta(position=float("nan"), mass=0.5),
            ]
        )
        ground_truth = _dv_from_weighted_samples(
            [-1.0, 0.0, 1.0, 2.0, 3.0], [0.1, 0.2, 0.4, 0.2, 0.1]
        )
        # The short-circuit routes to wasserstein_1_uxhw_wrapper, which
        # then rejects the non-finite NaN position via _validate_wp_inputs.
        # We just verify the routing — that we get a clean ValueError
        # from the validator path rather than a PlotData failure on a
        # length-1 finite Dirac list.
        with self.assertRaises(ValueError) as ctx:
            binned_wasserstein_1_uxhw_wrapper(
                binned_dist=uxhw_finite_with_special, ground_truth_dist=ground_truth
            )
        self.assertIn("finite", str(ctx.exception))


class TestBinnedUxStringWrapper(unittest.TestCase):
    """Parse paths for `binned_wasserstein_1_ux_string_wrapper`.

    Math coverage flows transitively through
    `binned_wasserstein_1_uxhw_wrapper` (which has its own tests). Here
    we just verify that:
      - bogus ux strings surface a ValueError rather than crashing
        later (`DistributionalValue.parse` may either return None or
        raise its own ValueError on malformed input),
      - real ux strings parse cleanly and produce the same distance as
        the DistributionalValue-form wrapper called directly."""

    # SAMPLE_UX_STRING is a ~32-atom Gaussian-shaped TTR also used by
    # `uxdata_toolkit_test.py:37` and `sample_generator_test.py:30`.
    # Inlined rather than imported because test modules aren't part of
    # the package API. Copying matches the convention already used in
    # those two callsites.
    SAMPLE_UX_STRING = "-0.000000Ux040000000000000001BCB03C52D58D3CE400000020C0048D2279B8AFF701B1807E6239F600BFFFF1F7A03E82B602A7EB881EDAA6C0BFFAFF0EC92B7D6E0321FCB58BB7F440BFF763B747227C880386DDE1E09EAEC0BFF47A086FFA91B003BA2C11EF08E580BFF1FDCC06ED8E9703F77BE72797F440BFEF88C5501503BA0429DAF9A7420E80BFEB75E0A582D1610454636C25A09D40BFE7B77D572D585D0456D709533085C0BFE43A6FD58615450472E978D476CD40BFE0E4BE4A8177310489FC4176BD8E80BFDB5AB694DC2F6D049CBBFB39689F80BFD51A3EFE52F0A004AAD48A9FB27AC0BFCDF7B07C22983104B59D8153294540BFC1E9102D4ACC1304BCB0EDEE5C1900BFA7D59C0E0AD59404C0374A760BE0C03FA7D59C0E0AD5A204C0374A760BE0C03FC1E9102D4ACC0504BCB0EDEE5C1C803FCDF7B07C22983104B59D81532945403FD51A3EFE52F0A904AAD48A9FB277003FDB5AB694DC2F6D049CBBFB39689F803FE0E4BE4A8177310489FC4176BD8E803FE43A6FD586153C0472E978D476D0C03FE7B77D572D58680456D709533082403FEB75E0A582D1560454636C25A0A1003FEF88C5501503D10429DAF9A7420AC03FF1FDCC06ED8EA203F77BE72797F7E03FF47A086FFA91AE03BA2C11EF08DE603FF763B747227C810386DDE1E09EAB203FFAFF0EC92B7D710321FCB58BB7F4403FFFF1F7A03E82AE02A7EB881EDAA32040048D2279B8AFD801B1807E6239FD40"  # noqa: E501

    def test_bogus_ux_strings_rejected(self) -> None:
        """Both unparseable and empty ux strings surface a ValueError."""
        for label, dux, gtux in (
            ("bogus", "not a valid ux string", "also invalid"),
            ("empty", "", ""),
        ):
            with self.subTest(case=label):
                with self.assertRaises(ValueError):
                    binned_wasserstein_1_ux_string_wrapper(
                        distribution_ux=dux,
                        ground_truth_ux=gtux,
                    )

    def test_round_trips_against_uxhw_wrapper(self) -> None:
        """Happy path: parse a real Ux string and verify the string-form
        wrapper produces the same distance as feeding the parsed
        DistributionalValue into `binned_wasserstein_1_uxhw_wrapper`
        directly. Plumbing-level test — the math is already covered by
        the DistributionalValue-form tests above. Guards against:
          - parse errors on real Ux strings,
          - the wrapper silently swallowing a parsed None,
          - drift between the two call paths.

        Note: self-distance is NOT zero for the binned wrapper. The
        wrapper bins the first argument but treats the second as raw
        weighted samples, so even with identical inputs on both sides
        the two representations differ — the assertion here is parity
        between the two parse paths, not d == 0."""
        # Parse path 1: through the string-form wrapper.
        d_string = binned_wasserstein_1_ux_string_wrapper(
            self.SAMPLE_UX_STRING, self.SAMPLE_UX_STRING
        )

        # Parse path 2: parse once, call the DV wrapper directly.
        parsed = DistributionalValue.parse(self.SAMPLE_UX_STRING)
        # `assert` rather than `if/self.fail()` so mypy narrows the type
        # and the message also serves as a regression guard that
        # SAMPLE_UX_STRING still parses.
        assert parsed is not None, "SAMPLE_UX_STRING failed to parse"
        d_dv = binned_wasserstein_1_uxhw_wrapper(
            binned_dist=parsed, ground_truth_dist=parsed
        )

        self.assertAlmostEqual(d_string, d_dv, places=12)
        # Sanity check: the distance is a finite non-negative real
        # number, ruling out NaN / inf leaking through the parse path.
        self.assertTrue(math.isfinite(d_string))
        self.assertGreaterEqual(d_string, 0.0)


if __name__ == "__main__":
    unittest.main()
